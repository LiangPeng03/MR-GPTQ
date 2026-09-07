#!/usr/bin/env python3
"""Evaluate an exported vLLM checkpoint with the repository PPL protocol."""

import argparse
import math
from pathlib import Path
from collections.abc import Sequence

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.utils.data_utils import get_c4_eval, get_wikitext2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--capture-dir", default=None)
    parser.add_argument(
        "--capture-layers",
        type=int,
        nargs="+",
        default=[0, 1, 3, 7, 15, 23, 31, 35],
    )
    parser.add_argument(
        "--dataset",
        choices=("wikitext2", "c4", "both"),
        default="both",
    )
    return parser.parse_args()


def token_logprob(value) -> float:
    if hasattr(value, "logprob"):
        return float(value.logprob)
    return float(value)


def evaluate(
    llm: LLM,
    samples: Sequence,
    name: str,
    batch_size: int,
    limit: int | None,
) -> float:
    if limit is not None:
        samples = samples[:limit]
    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=1,
    )
    nll = 0.0
    tokens = 0
    for start in range(0, len(samples), batch_size):
        batch = [sample.reshape(-1).tolist() for sample in samples[start:start + batch_size]]
        outputs = llm.generate(
            prompts=[{"prompt_token_ids": token_ids} for token_ids in batch],
            sampling_params=params,
            use_tqdm=False,
        )
        for token_ids, output in zip(batch, outputs):
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None or len(prompt_logprobs) != len(token_ids):
                raise RuntimeError(
                    f"Unexpected prompt_logprobs length for {name}: "
                    f"{None if prompt_logprobs is None else len(prompt_logprobs)} "
                    f"vs {len(token_ids)}"
                )
            for token_id, candidates in zip(token_ids[1:], prompt_logprobs[1:]):
                if candidates is None or token_id not in candidates:
                    raise RuntimeError(
                        f"Missing chosen token {token_id} in prompt_logprobs for {name}"
                    )
                logprob = token_logprob(candidates[token_id])
                if not math.isfinite(logprob):
                    raise RuntimeError(f"Non-finite logprob for {name}: {logprob}")
                nll -= logprob
                tokens += 1
        done = min(start + batch_size, len(samples))
        print(
            f"PPL_PROGRESS dataset={name} samples={done}/{len(samples)} "
            f"tokens={tokens} running_ppl={math.exp(nll / tokens):.6f}",
            flush=True,
        )
    ppl = math.exp(nll / tokens)
    print(f"VLLM_{name.upper()}_PERPLEXITY: {ppl:.4f}", flush=True)
    print(f"VLLM_{name.upper()}_TOKENS: {tokens}", flush=True)
    return ppl


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        dtype="bfloat16",
        # vLLM must have room for the one generated token used to return
        # prompt_logprobs. The evaluated prompt itself remains sequence_length.
        max_model_len=args.sequence_length + 1,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=(args.sequence_length + 1) * args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
        seed=0,
    )

    if args.capture_dir:
        capture_dir = str(Path(args.capture_dir).resolve())
        capture_layers = tuple(args.capture_layers)

        def install_capture_hooks(model):
            import os
            import torch

            os.makedirs(capture_dir, exist_ok=True)

            def save_once(name, value):
                path = os.path.join(capture_dir, f"{name}.pt")
                if os.path.exists(path):
                    return
                if isinstance(value, (tuple, list)):
                    if (
                        len(value) == 2
                        and torch.is_tensor(value[0])
                        and torch.is_tensor(value[1])
                    ):
                        # vLLM carries the residual separately between blocks.
                        value = value[0] + value[1]
                    else:
                        value = value[0]
                torch.save(value.detach().to("cpu"), path)

            model.model.embed_tokens.register_forward_hook(
                lambda _module, _inputs, output: save_once("embedding", output)
            )
            for layer_idx in capture_layers:
                model.model.layers[layer_idx].register_forward_hook(
                    lambda _module, _inputs, output, layer_idx=layer_idx:
                        save_once(f"layer_{layer_idx:02d}", output)
                )

            def save_first(name, value):
                if isinstance(value, (tuple, list)):
                    value = value[0]
                save_once(name, value)

            layer0 = model.model.layers[0]
            layer0.input_layernorm.register_forward_hook(
                lambda _module, _inputs, output:
                    save_first("l0_input_norm", output)
            )
            layer0.self_attn.qkv_proj.register_forward_hook(
                lambda _module, _inputs, output: save_first("l0_qkv", output)
            )
            layer0.self_attn.q_norm.register_forward_hook(
                lambda _module, _inputs, output: save_first("l0_q_norm", output)
            )
            layer0.self_attn.k_norm.register_forward_hook(
                lambda _module, _inputs, output: save_first("l0_k_norm", output)
            )

            def save_vllm_rope(_module, inputs, output):
                save_first("l0_rope_positions", inputs[0])
                save_first("l0_rope_cache", _module.cos_sin_cache)
                save_first("l0_rope_q", output[0])
                save_first("l0_rope_k", output[1])

            layer0.self_attn.rotary_emb.register_forward_hook(save_vllm_rope)
            layer0.self_attn.o_proj.register_forward_pre_hook(
                lambda _module, inputs: save_first("l0_attn_out", inputs[0])
            )
            layer0.self_attn.o_proj.register_forward_hook(
                lambda _module, _inputs, output: save_first("l0_o_proj", output)
            )
            layer0.post_attention_layernorm.register_forward_hook(
                lambda _module, _inputs, output:
                    save_first("l0_post_attn_norm", output)
            )
            layer0.mlp.gate_up_proj.register_forward_hook(
                lambda _module, _inputs, output:
                    save_first("l0_gate_up", output)
            )
            layer0.mlp.down_proj.register_forward_pre_hook(
                lambda _module, inputs: save_first("l0_mlp_act", inputs[0])
            )
            layer0.mlp.down_proj.register_forward_hook(
                lambda _module, _inputs, output:
                    save_first("l0_down_proj", output)
            )
            return {
                "capture_dir": capture_dir,
                "capture_layers": capture_layers,
            }

        print("VLLM_CAPTURE_HOOKS:", llm.apply_model(install_capture_hooks), flush=True)

    if args.dataset in ("wikitext2", "both"):
        evaluate(
            llm,
            get_wikitext2(tokenizer, args.sequence_length),
            "wikitext2",
            args.batch_size,
            args.limit,
        )
    if args.dataset in ("c4", "both"):
        evaluate(
            llm,
            get_c4_eval(tokenizer, args.sequence_length),
            "c4",
            args.batch_size,
            args.limit,
        )
    print("VLLM_PERPLEXITY_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
