import os
import json
import argparse
import warnings
from functools import partial

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table
except ImportError:
    try:
        import lm_eval
        from lm_eval.models.gpt2 import HFLM
        # Monkeypatch to skip strict tokenizer check for non-GPT2 models
        if hasattr(HFLM, "tokenizer_check"):
            HFLM.tokenizer_check = lambda self: None
        from lm_eval.evaluator import make_table
    except ImportError:
        lm_eval = None
        make_table = None

from src.metrics.perplexity import compute_perplexity
from src.transforms.transforms import TRANSFORMS
from src.quantization.quant_ops import NVFP_GROUPSIZE, MXFP_GROUPSIZE
from src.quantization.qconfig import prepare_quantization_config
from src.quantization import rtn_quantization, gptq_quantization
from src.utils.common_utils import fix_seed
from src.utils.data_utils import get_data, get_wikitext2, get_c4_eval

try:
    import wandb
except ImportError:
    wandb = None

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def auto_or_int(value):
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Must be 'auto' or an integer, got '{value}'")


def export_quantized_model(model, quantized_state_dict, non_quantized_state_dict, args):
    config = model.config
    # Prepare directory to save model
    os.makedirs(args.save_path, exist_ok=True)

    blocks = model.model.layers

    # State dict to save
    model_state_dict = {}

    for block_idx, block in enumerate(blocks):
        prefix = f"model.layers.{block_idx}."
        for k, v in block.state_dict().items():
            layer_name, param_name = k.rsplit(".", 1)
            if f"{prefix}{layer_name}" in quantized_state_dict and param_name == "weight":
                for k_compr, v_compr in quantized_state_dict[f"{prefix}{layer_name}"].items():
                    model_state_dict[f"{prefix}{layer_name}.{k_compr}"] = v_compr.cpu()
            elif f"{prefix}{k}" in non_quantized_state_dict:
                model_state_dict[f"{prefix}{k}"] = non_quantized_state_dict[f"{prefix}{k}"].cpu()
            else:
                model_state_dict[f"{prefix}{k}"] = v.cpu()

    # Add non_quantized_state_dict block parameters (dict is non-empty for blockwise_qat)
    model_state_dict.update(non_quantized_state_dict)

    # Process all remaining blocks
    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", False)

    for k, v in model.state_dict().items():
        if not (k.startswith("model.layers") or (k == "lm_head.weight" and tie_word_embeddings)):
            model_state_dict[k] = v.cpu()

    # Split checkpoint into shards
    current_shard_size = 0
    current_shard = {}
    shards = []

    for k, v in model_state_dict.items():
        tensor_size = v.numel() * v.element_size()
        if current_shard_size + tensor_size > args.max_shard_size:
            shards.append(current_shard)
            current_shard = {}
            current_shard_size = 0

        if tensor_size > args.max_shard_size:
            shards.append({k: v})
            continue
        
        current_shard[k] = v
        current_shard_size += tensor_size

    # Dump last shard if it is not empty
    if len(current_shard) > 0:
        shards.append(current_shard)

    safetensors_index = {}
    num_shards = len(shards)
    max_digits = len(str(max(num_shards, 1)))

    # Save shards
    for shard_idx, shard in enumerate(shards):
        current_shard_path = f"model-{str(shard_idx+1).zfill(max_digits)}-of-{str(num_shards).zfill(max_digits)}.safetensors"
        save_file(shard, os.path.join(args.save_path, current_shard_path))
        for k in shard:
            safetensors_index[k] = current_shard_path

    # Save safetensors index
    with open(os.path.join(args.save_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": safetensors_index}, f)

    # Add quantization metadata
    config.quantization_config = prepare_quantization_config(
        args.hadamard_group_size, 
        args.format,
        pseudoquantization=(args.export_quantized_model == "pseudoquant")
    )
    # Save configs
    config.save_pretrained(args.save_path)
    model.generation_config.save_pretrained(args.save_path)

    
def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to quantized model.",
    )
    # Data params
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="The name or path to the calibration dataset.",
    )
    parser.add_argument(
        "--sequence_length", 
        default=2048, 
        type=int, 
        help="Length of calibration sequences."
    )
    parser.add_argument(
        "--num_sequences", 
        default=1024, 
        type=int, 
        help="Number of calibration sequences."
    )
    # Quantization params
    parser.add_argument(
        "--format",
        type=str,
        default="int",
        choices=["int", "fp", "nvfp", "mxfp"],
        help="Quantization format.",
    )
    parser.add_argument(
        "--scale_precision",
        type=str,
        default="fp16",
        choices=["fp16", "e8m0", "e4m3"],
        help="Scale precision.",
    )
    parser.add_argument(
        "--w_granularity",
        type=str,
        default="group",
        choices=["tensor", "channel", "group"],
        help="Weight quantization granularity.",
    )
    parser.add_argument(
        "--w_bits",
        type=int,
        required=True,
        help="Weight quantization bitwidth.",
    )
    parser.add_argument(
        "--w_group_size",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--w_observer",
        type=str,
        default="minmax",
        choices=["minmax", "mse"],
        help="Weight observer.",
    )
    parser.add_argument(
        "--a_bits",
        type=int,
        default=16,
        help="Activation quantization bitwidth.",
    )
    parser.add_argument(
        "--a_granularity",
        type=str,
        default="group",
        choices=["tensor", "channel", "group"],
        help="Activation quantization granularity.",
    )
    parser.add_argument(
        "--a_group_size",
        type=int,
        default=None,
        help="How many activation columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--a_observer",
        type=str,
        default="minmax",
        choices=["minmax","lss"],
        help="Activation observer.",
    )
    parser.add_argument(
        "--export_quantized_model",
        type=str,
        default="",
        choices=["", "realquant", "pseudoquant"],
        help="Whether export quantized model in realquant or pseudoquant format.",
    )
    # AWQ params
    parser.add_argument(
        "--awq",
        type=int,
        default=0,
        help="Run AWQ search before GPTQ with the specified number of grid search steps (e.g. 20). 0 disables AWQ.",
    )
    parser.add_argument(
        "--gajs",
        action="store_true",
        help="Use Grid-Aligned Joint Scaling (Adam) instead of grid search for AWQ.",
    )
    # GPTQ params
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="Run GPTQ quantization.",
    )
    parser.add_argument(
        "--quantization_order",
        type=str,
        default="default",
        choices=["default", "activation"],
        help="Weigth quantization order in GPTQ.",
    )
    parser.add_argument("--rel_damp", type=float, default=1e-2)
    parser.add_argument(
        "--awq_for_act",
        type=float,
        default=0.0,
        help="Apply AWQ-style channel scaling preprocessing to activations with the given threshold (e.g., 10.0).",
    )
    parser.add_argument(
        "--channel_resort",
        type=str,
        default="none",
        choices=["none", "mean", "P95", "minmax", "stagger", "kmeans_fp4", "kmeans_fp4_w"],
        help="Apply grid-aware channel reordering based on channel 'mean' or 'P95', 'minmax', 'stagger' (Co-occurrence-aware), 'kmeans_fp4' or use 'none' to skip.",
    )
    parser.add_argument(
        "--stagger_lambda",
        type=str,
        default="auto",
        help="Balance coefficient for Joint W/A Staggering. Can be 'auto' (sqrt(N/M)) or a float value (e.g. '0.5', '1.0').",
    )
    parser.add_argument(
        "--kmeans_block_size",
        type=int,
        default=0,
        help="Block size for K-means channel resorting applied to down_proj. If 0, uses head_dim. If -1, uses global K-means.",
    )
    parser.add_argument(
        "--kmeans_alpha",
        type=float,
        default=0.0,
        help="Alpha parameter for weighting K-means FP4 loss by weight norm.",
    )
    parser.add_argument(
        "--kmeans_act_alpha",
        type=float,
        default=0.0,
        help="Alpha parameter for weighting K-means FP4 loss by activation magnitude itself.",
    )
    parser.add_argument(
        "--outlier_ratio",
        type=float,
        default=0.01,
        help="Outlier ratio for Truncated MinMax channel reordering.",
    )
    # Transform params
    parser.add_argument(
        "--transform_class",
        type=str,
        default="identity",
        choices=TRANSFORMS.keys(),
        help="The transform class."
    )
    parser.add_argument(
        "--hadamard_group_size",
        type=int,
        default=128,
        help="Hadamard group size"
    )
    # Logging params
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Whether to log to wandb."
    )
    parser.add_argument(
        "--show_act_mse",
        action="store_true",
        help="Whether to compute and show activation quantization MSE."
    )
    # Misc params
    parser.add_argument(
        "--verbose",
        action="store_true"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed.")
    parser.add_argument("--cpu_offload_modules", action="store_true", help="whether to offload modules to CPU.")
    parser.add_argument("--cpu_offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument("--amp", action="store_true", help="whether to enable fp16 autocasting.")
    parser.add_argument("--compile", action="store_true", help="whether to use torch.compile.")
    parser.add_argument("--fuse_global_scale", action="store_true", help="whether to fuse global scale in qkv and gate_up.")
    parser.add_argument("--lock_global_scale", action="store_true", help="whether to lock the NVFP4 global scale to the original FP16 weights.")
    # Eval params
    parser.add_argument("--eval_perplexity", action="store_true", help="whether to eval perplexity after quantization.")
    parser.add_argument("--eval_openllm", action="store_true", help="whether to eval OpenLLM v1 openllm after quantization.")
    # LM eval params
    parser.add_argument(
        "--lm_eval_batch_size",
        type=auto_or_int,
        default="auto",
        help="LM eval batch size to evaluate after quantization.",
    )
    parser.add_argument(
        "--lm_eval_tasks",
        nargs="+",
        type=str,
        default=["mmlu_cot_llama", "arc_challenge_llama", "gsm8k_llama", "hellaswag", "winogrande", "truthfulqa"],
        help="OpenLLMv1 tasks to evaluate after quantization."
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Whether to disable thinking mode for Qwen3.",
    )
    # Save params
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save quantized model",
    )
    parser.add_argument(
        "--max_shard_size", 
        type=int, 
        default=5 * 1024 * 1024 * 1024, 
        help="Maximum shard size in bytes."
    )
    # Parse arguments
    args = parser.parse_args()
    # Check and fix group_size (if needed)
    if args.format == "nvfp":
        if args.w_group_size != NVFP_GROUPSIZE:
            args.w_group_size = NVFP_GROUPSIZE
            print(f"Changed weight group_size to {NVFP_GROUPSIZE} for nvfp format.")
        if args.a_group_size != NVFP_GROUPSIZE:
            args.a_group_size = NVFP_GROUPSIZE
            print(f"Changed activation group_size to {NVFP_GROUPSIZE} for nvfp format.")
        if args.scale_precision != "e4m3":
            args.scale_precision = "e4m3"
            print(f"Changed scale_precision to e4m3 for nvfp format.")
    elif args.format == "mxfp":
        if args.w_group_size != MXFP_GROUPSIZE:
            args.w_group_size = MXFP_GROUPSIZE
            print(f"Changed weight group_size to {MXFP_GROUPSIZE} for mxfp format.")
        if args.a_group_size != MXFP_GROUPSIZE:
            args.a_group_size = MXFP_GROUPSIZE
            print(f"Changed activation group_size to {MXFP_GROUPSIZE} for mxfp format.")
        if args.scale_precision != "e8m0":
            args.scale_precision = "e8m0"
            print(f"Changed scale precision to e8m0 for mxfp format.")
    # Check logging
    if args.log_wandb:
        assert wandb is not None, "wandb is not installed. Please install wandb `pip install wandb`."
    # Check real_quant config
    if args.export_quantized_model:
        assert args.save_path is not None, "`save_path` must be specified when exporting quantized model."
        assert args.format in ["nvfp", "mxfp"], "`export_quantization` is only supported for nvfp and mxfp formats."
        assert args.w_bits == 4, "`export_quantization` is only supported for 4 bit weights."
        assert args.a_bits == 4, "`export_quantization` is only supported for 4 bit activations."
        
    if args.format == "nvfp" and args.channel_resort in ["minmax", "stagger", "kmeans_fp4", "kmeans_fp4_w"]:
        if args.transform_class != "identity":
            print(f"Forcing transform_class to 'identity' since format is nvfp and channel_resort is '{args.channel_resort}'.")
            args.transform_class = "identity"
            
    return args


def main():
    # Use expandable segments to handle the memory pressure of 25-shot ARC
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    args = parse_args()
    # Fix seed
    fix_seed(args.seed)
    # Set device
    device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    # Get dtype
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Init logger
    if args.log_wandb:
        wandb.init(config=args)
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        dtype=args.dtype, 
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if not args.cpu_offload_modules:
        model = model.to(device)
    model.config.use_cache = False
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Sanity check
    if args.eval_openllm:
        if not (hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None):
            warnings.warn("Tokenizer does not have a chat_template. Tasks requiring chat templates (like gsm8k_llama) will be skipped or may fail.")
        if args.disable_thinking:
            if model.config.model_type == "qwen3":
                tokenizer.apply_chat_template = partial(
                    tokenizer.apply_chat_template, 
                    enable_thinking=False
                )
            else:
                warnings.warn("`disable_thinking` has no effect on non-Qwen3 models.")

    quantize_anything = args.w_bits < 16 or args.a_bits < 16

    # Prepare calibration data
    calibration_data = get_data(
        args.dataset_name_or_path,
        tokenizer,
        args.sequence_length,
        args.num_sequences,
        args.seed
    )

    if quantize_anything:
        if args.gptq:
            quantized_state_dict, non_quantized_state_dict = gptq_quantization(model, calibration_data, args, device)
        else:
            quantized_state_dict, non_quantized_state_dict = rtn_quantization(model, calibration_data, args, device)

        if args.export_quantized_model:
            export_quantized_model(model, quantized_state_dict, non_quantized_state_dict, args) 
            tokenizer.save_pretrained(args.save_path)
        
        # CRITICAL: Reclaim 16-32GB of VRAM
        del quantized_state_dict
        del non_quantized_state_dict
        import gc
        for _ in range(3): gc.collect()
        torch.cuda.empty_cache()

    if args.compile:
        model = torch.compile(model)

    if args.eval_perplexity or args.eval_openllm:
        # Move model to CPU briefly to ensure NO hidden quantization tensors are kept on GPU
        model = model.to('cpu')
        if 'calibration_data' in locals():
            del calibration_data
        import gc
        for _ in range(3): gc.collect()
        torch.cuda.empty_cache()
        
        # Now move model back to GPU - it should be the ONLY major thing on VRAM now
        model = model.to(device)
        # Enable KV cache for faster autoregressive evaluation
        model.config.use_cache = True

    wikitext2_ppl = None
    c4_ppl = None

    if args.eval_perplexity:
        eval_data = get_wikitext2(tokenizer, args.sequence_length)
        wikitext2_ppl = compute_perplexity(model, eval_data)
        print(f"Wikitext-2 perplexity: {wikitext2_ppl:.4f}")
        del eval_data
        
        c4_data = get_c4_eval(tokenizer, args.sequence_length)
        c4_ppl = compute_perplexity(model, c4_data)
        print(f"C4 perplexity: {c4_ppl:.4f}")
        del c4_data
        
        if args.log_wandb:
            wandb.log({"eval/wikitext2_ppl": wikitext2_ppl, "eval/c4_ppl": c4_ppl})

        # Free memory before OpenLLM eval
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # OpenLLM v1 openllm (following https://arxiv.org/abs/2411.02355)
    if args.eval_openllm:

        results = {}
        # Cap max_position_embeddings to prevent lm_eval from using excessively long sequences
        _orig_max_pos = model.config.max_position_embeddings
        model.config.max_position_embeddings = min(model.config.max_position_embeddings, 2048)

        def _run_task(task_name, num_fewshot=0, batch_size=1, **extra_kwargs):
            """Run a single lm_eval task with fresh HFLM to avoid accumulated state."""
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            # Create fresh HFLM per task to prevent KV cache / logits accumulation
            lm = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=2048,
            )
            task_results = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                **extra_kwargs,
            )["results"]
            results.update(task_results)
            print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))
            # Immediately free HFLM and all its internal state
            del lm
            gc.collect()
            torch.cuda.empty_cache()

        # Winogrande (5-shot) - Can handle larger batch
        if "winogrande" in args.lm_eval_tasks:
            _run_task("winogrande", num_fewshot=5, batch_size=64)
        # Hellaswag (10-shot) - Medium batch
        if "hellaswag" in args.lm_eval_tasks:
            _run_task("hellaswag", num_fewshot=10, batch_size=8)
        # PIQA (0-shot) - Can handle very large batch
        if "piqa" in args.lm_eval_tasks:
            _run_task("piqa", num_fewshot=0, batch_size=64)
        # ARC Challenge (25-shot) - Must keep batch_size=1
        if "arc_challenge" in args.lm_eval_tasks:
            _run_task("arc_challenge", num_fewshot=25, batch_size=8)
        # GSM8K (requires chat template)
        if "gsm8k_llama" in args.lm_eval_tasks:
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
                _run_task("gsm8k_llama", apply_chat_template=True, fewshot_as_multiturn=True, batch_size=8)
            else:
                print("Skipping gsm8k_llama: no chat template.")
        # MMLU CoT (requires chat template)
        if "mmlu_cot_llama" in args.lm_eval_tasks:
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
                _run_task("mmlu_cot_llama", apply_chat_template=True, fewshot_as_multiturn=True, batch_size=8)
            else:
                print("Skipping mmlu_cot_llama: no chat template.")

        # Log results
        if args.log_wandb:
            wandb.log({"eval/openllm": results}) 
        # Print formatted table
        print("### Final results ###")
        print(make_table({"results": results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))

        # [NEW] Robust summary for scripts
        def get_best_metric(res_dict):
            # Try to find any key containing 'acc' (e.g., 'acc,none', 'acc_norm,none')
            acc_keys = [k for k in res_dict.keys() if 'acc' in k]
            if not acc_keys: return None
            # Prioritize 'acc_norm' over 'acc'
            norm_keys = [k for k in acc_keys if 'norm' in k]
            target_key = norm_keys[0] if norm_keys else acc_keys[0]
            return res_dict.get(target_key)

        summary = {
            "model": args.model_name_or_path,
            "wikitext2_ppl": wikitext2_ppl,
            "c4_ppl": c4_ppl,
            "results": {task: get_best_metric(res) for task, res in results.items()}
        }
        import json
        print(f"\n[RESULT_SUMMARY] {json.dumps(summary)}")


if __name__ == "__main__":
    main()
