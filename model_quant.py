import os
import json
import argparse
import warnings
from functools import partial
import logging

logging.getLogger("lm_eval").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

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
from src.quantization.quant_ops import NVFP_GROUPSIZE, MXFP_GROUPSIZE, FP8_E4M3_MAX
import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
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


def enable_cpu_offload_ppl(model, device):
    """Run PPL forwards one transformer block at a time on ``device``.

    This is intentionally limited to cache-free perplexity evaluation.  The model
    remains on CPU except for the embedding/output modules and the transformer
    block currently executing.  It makes pseudo-quantized models larger than one
    GPU evaluable, at the cost of substantial CPU-GPU transfer overhead.
    """
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if backbone is None or layers is None:
        raise ValueError(
            "--cpu_offload_eval currently supports causal-LM models with "
            "model.layers (for example Llama and Qwen families)."
        )

    # These modules are used outside the individual transformer blocks, so they
    # must stay with the hidden states on the evaluation GPU.
    resident_modules = [
        getattr(backbone, "embed_tokens", None),
        getattr(backbone, "norm", None),
        getattr(backbone, "rotary_emb", None),
        model.get_output_embeddings(),
    ]
    for module in resident_modules:
        if module is not None:
            module.to(device)

    def move_block_to_gpu(module, inputs):
        module.to(device)
        return inputs

    def move_block_to_cpu(module, inputs, output):
        module.cpu()
        return output

    # Keep hook handles alive for the duration of evaluation.  Registering hooks
    # only on blocks avoids moving the whole model through the root module.
    model._cpu_offload_ppl_hook_handles = []
    for block in layers:
        model._cpu_offload_ppl_hook_handles.append(
            block.register_forward_pre_hook(move_block_to_gpu)
        )
        model._cpu_offload_ppl_hook_handles.append(
            block.register_forward_hook(move_block_to_cpu)
        )

    print(
        f"[CPU offload eval] Enabled cache-free block streaming for {len(layers)} "
        f"transformer layers on {device}."
    )

def export_quantized_model(model, quantized_state_dict, non_quantized_state_dict, args):
    config = model.config
    # Prepare directory to save model
    os.makedirs(args.save_path, exist_ok=True)

    blocks = model.model.layers

    # State dict to save
    model_state_dict = {}
    # Channel-resort and GICS are runtime input transforms: GPTQ quantizes the
    # corresponding weights in transformed coordinates.  Store their compact
    # permutation/scale representation under the linear layer so vLLM can load
    # and apply it before activation quantization.  Do not save the training
    # wrapper's transform-module buffers; native vLLM models do not have those
    # modules.
    quantization_only_state_prefixes = (
        "self_attn.qkv_in_transform.",
        "self_attn.o_in_transform.",
        "mlp.gate_up_in_transform.",
        "mlp.down_in_transform.",
    )
    input_transform_targets = (
        ("self_attn.qkv_in_transform", "self_attn.q_proj"),
        ("self_attn.o_in_transform", "self_attn.o_proj"),
        ("mlp.gate_up_in_transform", "mlp.gate_proj"),
        ("mlp.down_in_transform", "mlp.down_proj"),
    )

    for block_idx, block in enumerate(blocks):
        prefix = f"model.layers.{block_idx}."
        for transform_name, layer_name in input_transform_targets:
            transform = block.get_submodule(transform_name)
            transform_state = transform.state_dict()
            perm_keys = [k for k in transform_state if k.endswith("perm_idx")]
            scale_keys = [k for k in transform_state if k.endswith(".scale")]
            allowed_keys = set(perm_keys + scale_keys)
            allowed_keys.update(k for k in transform_state if k.endswith("inv_scale"))
            unsupported_keys = set(transform_state) - allowed_keys
            if unsupported_keys or len(perm_keys) > 1 or len(scale_keys) > 1:
                raise ValueError(
                    "Realquant export supports at most one permutation and one "
                    f"diagonal input scale per layer, but {prefix}{transform_name} "
                    f"contains {sorted(transform_state)}."
                )
            if perm_keys:
                model_state_dict[f"{prefix}{layer_name}.input_perm"] = (
                    transform_state[perm_keys[0]].cpu()
                )
            if scale_keys:
                model_state_dict[f"{prefix}{layer_name}.input_rescale"] = (
                    transform_state[scale_keys[0]].cpu()
                )
        for k, v in block.state_dict().items():
            if k.startswith(quantization_only_state_prefixes):
                continue
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

    invalid_keys = [
        key
        for key in model_state_dict
        if any(f".{state_prefix}" in key for state_prefix in quantization_only_state_prefixes)
    ]
    if invalid_keys:
        raise RuntimeError(
            "Refusing to export quantization-only transform state: "
            f"{invalid_keys[:3]}"
        )

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
        pseudoquantization=(args.export_quantized_model == "pseudoquant"),
        activation_observer=args.a_observer,
        identity_transform=(args.transform_class == "identity"),
    )
    if getattr(args, "nada_wmse", None):
        config.quantization_config.update({
            "nada_wmse": args.nada_wmse,
            "weight_observer": args.w_observer,
            "weight_global_scale_max": FP8_E4M3_MAX,
            "calibration": {
                "dataset": args.dataset_name_or_path, "samples": args.num_sequences,
                "sequence_length": args.sequence_length, "seed": args.seed,
            },
        })
    if args.a_observer in {"scalesweep_mse", "scalesweep_wmse"}:
        config.quantization_config.update({
            "scalesweep_base_rounding": "floor",
            "scalesweep_lower_bound": -8 if args.a_observer == "scalesweep_wmse" else -3,
            "scalesweep_upper_bound": 7,
            "weight_observer": args.w_observer,
            "calibration": {
                "dataset": args.dataset_name_or_path, "samples": args.num_sequences,
                "sequence_length": args.sequence_length, "seed": args.seed,
            },
        })
    # Save configs
    config.save_pretrained(args.save_path)
    model.generation_config.save_pretrained(args.save_path)

    # Transformers 5.x may migrate RoPE settings into ``rope_parameters`` and
    # omit the legacy top-level ``rope_theta`` field.  The custom deployment
    # backend (and older Transformers/vLLM versions) still reads rope_theta.
    # Preserve both representations so exported checkpoints retain the base
    # model's positional encoding across backend versions.
    config_path = os.path.join(args.save_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        exported_config = json.load(f)
    rope_parameters = exported_config.get("rope_parameters")
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        exported_config["rope_theta"] = rope_parameters["rope_theta"]
    elif "rope_theta" not in exported_config and hasattr(config, "rope_theta"):
        exported_config["rope_theta"] = config.rope_theta
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(exported_config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    
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
        "--transform_order",
        type=str,
        default="kmeans_mr",
        choices=["kmeans_mr", "mr_kmeans"],
        help="Ordering of transformations when combining MR and KMeans channel resort",
    )
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
        choices=["minmax", "mse", "mse_n", "scalesweep_mse", "scalesweep_wmse", "scalesearch"],
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
        choices=["minmax","lss","lss_wmse","four_over_six","lss_3round","scalesweep_mse","scalesweep_wmse","scalesearch"],
        help="Activation observer.",
    )
    parser.add_argument(
        "--scalesweep", choices=["mse", "wmse"], default=None,
        help="Select ScaleSweep-MSE or WMSE for both weight and activation observers.",
    )
    parser.add_argument(
        "--nada_wmse", choices=["weight", "weight_activation"], default=None,
        help="NADA hybrid: ScaleSweep-WMSE weights, optionally weighted activation LSS; preserve global bound 448.",
    )
    parser.add_argument(
        "--nada_global_sweep", action="store_true",
        help="Select shared weight global scales from 256,320,384,448 using calibration output error.",
    )
    parser.add_argument(
        "--bitr_power_search", action="store_true",
        help="Select BITR S, S**0.75 or S**0.5 using signed calibration output error.",
    )
    parser.add_argument(
        "--weight_group_hessian", action="store_true",
        help="Score WMSE weight-scale candidates using full group-16 Hessian blocks.",
    )
    parser.add_argument("--nada_diagnostics", type=str, default=None,
                        help="Output directory for fixed-weight activation and post-GPTQ scale diagnostics.")
    parser.add_argument("--nada_refit_groups", type=int, default=8,
                        help="Maximum weight groups per linear layer updated in one refit sweep.")
    parser.add_argument("--gptq_lazy_group_scales", action="store_true",
                        help="Initialize each fixed weight group's WMSE scale at its first act-order visit.")
    parser.add_argument("--scalesearch", action="store_true",
                        help="ScaleSearch Algorithm 1: nearest E4M3 base, offsets [-2,6], W/A MSE.")
    parser.add_argument("--permuquant", action="store_true",
                        help="Paper-based PermuQuant LLM adaptation with alpha search and acceptance.")
    parser.add_argument("--permuquant_tokens", type=int, default=4096,
                        help="Uniformly sampled calibration tokens used for PermuQuant output-MSE selection.")
    parser.add_argument("--permuquant_threshold", type=float, default=0.0,
                        help="Required relative output-MSE improvement to accept a permutation.")
    parser.add_argument(
        "--export_quantized_model",
        type=str,
        default="",
        choices=["", "realquant", "pseudoquant"],
        help="Whether export quantized model in realquant or pseudoquant format.",
    )
    parser.add_argument(
        "--validate_realquant_export",
        action="store_true",
        help=(
            "Validate each NVFP4 realquant representation by decoding it and "
            "comparing it with the post-fix pseudoquant weight; does not require "
            "writing a checkpoint."
        ),
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
    parser.add_argument(
        "--awq_rounds",
        type=int,
        default=1,
        help="Number of AWQ refinement rounds (1 = standard single-pass AWQ, >1 = multi-round). "
             "Only applies to the baseline AWQ grid search, not GAJS.",
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
        choices=["none", "mean", "P95", "minmax", "stagger", "channel_cluster", "kmeans_fp4", "kmeans_fp4_w", "kmeans_fp4_top3", "permuquant"],
        help="Channel reordering strategy: 'none', 'mean', 'P95', 'minmax', 'stagger', 'channel_cluster' (K-Means FP4 clustering). "
             "Legacy aliases 'kmeans_fp4', 'kmeans_fp4_w', 'kmeans_fp4_top3' are also accepted for backward compatibility.",
    )
    parser.add_argument(
        "--channel_seed_strategy",
        type=str,
        default="max_sum",
        choices=["max_sum", "max_min"],
        help="Seed selection for channel_cluster: cumulative-loss max-sum (default) or farthest-first max-min.",
    )
    parser.add_argument(
        "--channel_rescale",
        type=str,
        default="none",
        choices=["none", "gics"],
        help="Channel rescaling strategy: 'none' or 'gics' (GICS coordinate-descent channel scale search).",
    )
    parser.add_argument(
        "--gics_top_k",
        type=int,
        default=5,
        help="Number of top-sensitive channels to optimize per group in GICS coordinate descent.",
    )
    parser.add_argument(
        "--channel_rescale_rounds",
        type=int,
        default=3,
        help="Number of coordinate-descent rounds in channel rescale search.",
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
        default=-1,
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
        "--dacc_weighted_proxy", action="store_true",
        help="Weight DACC pairwise proxy by the sensitivity of the channel actually incurring the error.",
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
    parser.add_argument(
        "--smoothquant_alpha",
        type=float,
        default=None,
        help="SmoothQuant alpha for activation-weight smoothing. "
             "0.0 = max migration to weights, 0.5 = balanced, 1.0 = max migration to activations. "
             "None disables SmoothQuant. Executes before channel_resort/channel_rescale.",
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
    parser.add_argument("--save_calibration_tokens", default=None, type=str,
                        help="Save the exact sampled calibration token IDs for paired baseline runs.")
    parser.add_argument("--cpu_offload_modules", action="store_true", help="whether to offload modules to CPU.")
    parser.add_argument("--cpu_offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument(
        "--cpu_offload_eval",
        action="store_true",
        help=(
            "Stream one transformer block at a time from CPU to GPU during "
            "cache-free perplexity evaluation. Requires --cpu_offload_modules; "
            "not supported for lm-eval or generation tasks."
        ),
    )
    parser.add_argument(
        "--hf_device_map",
        choices=["none", "auto"],
        default="none",
        help=(
            "Transformers device-map mode. 'auto' shards a full-precision model across "
            "GPU and CPU using Accelerate; currently intended for w16/a16 baseline evaluation."
        ),
    )
    parser.add_argument(
        "--hf_max_gpu_memory",
        type=str,
        default="82GiB",
        help="Maximum GPU memory available to --hf_device_map auto (e.g. 82GiB).",
    )
    parser.add_argument(
        "--hf_max_cpu_memory",
        type=str,
        default="460GiB",
        help="Maximum CPU memory available to --hf_device_map auto (e.g. 460GiB).",
    )
    parser.add_argument(
        "--hf_offload_folder",
        type=str,
        default=None,
        help="Folder for optional disk offload used by Transformers/Accelerate.",
    )
    parser.add_argument(
        "--hf_allow_legacy_accelerate",
        action="store_true",
        help=(
            "Experimental compatibility bypass for Transformers 5 with an older installed "
            "Accelerate. It affects only --hf_device_map auto."
        ),
    )
    parser.add_argument("--amp", action="store_true", help="whether to enable fp16 autocasting.")
    parser.add_argument("--compile", action="store_true", help="whether to use torch.compile.")
    parser.add_argument("--fuse_global_scale", action="store_true", help="whether to fuse global scale in qkv and gate_up.")
    parser.add_argument("--lock_global_scale", action="store_true", help="whether to lock the NVFP4 global scale to the original FP16 weights.")
    # Eval params
    parser.add_argument("--eval_perplexity", action="store_true", help="whether to eval perplexity after quantization.")
    parser.add_argument('--eval_openllm', action='store_true', help='Evaluate using lm-eval on standard tasks')
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task for lm_eval (None = full dataset). "
             "Mirrors four-over-six convention: 400 for hellaswag/arc_challenge.",
    )
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
    if sum((args.scalesweep is not None, args.scalesearch, args.permuquant,
            args.nada_wmse is not None)) > 1:
        parser.error("Select one of --scalesweep, --scalesearch, --permuquant, or --nada_wmse.")
    if args.scalesweep is not None:
        args.w_observer = args.a_observer = f"scalesweep_{args.scalesweep}"
    if args.scalesearch:
        args.w_observer = args.a_observer = "scalesearch"
    if args.nada_wmse:
        if (not args.gptq or args.format != "nvfp" or args.w_bits != 4
                or args.a_bits != 4 or args.transform_class != "identity"
                or args.awq or args.gajs):
            parser.error("--nada_wmse requires GPTQ NVFP4 W4A4, identity transform, and no AWQ/GAJS.")
        args.w_observer = "scalesweep_wmse"
        args.a_observer = "lss_wmse" if args.nada_wmse == "weight_activation" else "lss"
        print(f"[NADA WMSE] mode={args.nada_wmse} W={args.w_observer} "
              f"A={args.a_observer} global_bound=448")
    if args.nada_global_sweep and (
        args.nada_wmse != "weight" or not args.fuse_global_scale
        or args.lock_global_scale or args.export_quantized_model
        or args.validate_realquant_export
    ):
        parser.error("--nada_global_sweep currently requires --nada_wmse weight "
                     "and --fuse_global_scale; calibration/PPL only, without export.")
    if args.dacc_weighted_proxy and (args.channel_resort != "channel_cluster"
                                     or args.kmeans_alpha <= 0):
        parser.error("--dacc_weighted_proxy requires channel_cluster and positive --kmeans_alpha.")
    if args.bitr_power_search and (args.nada_wmse != "weight"
                                   or args.channel_resort != "channel_cluster"
                                   or args.channel_rescale != "gics"):
        parser.error("--bitr_power_search requires --nada_wmse weight, channel_cluster and gics.")
    if args.weight_group_hessian and args.nada_wmse != "weight":
        parser.error("--weight_group_hessian requires --nada_wmse weight.")
    if args.gptq_lazy_group_scales and (
        args.nada_wmse != "weight" or args.quantization_order != "activation"
        or not args.fuse_global_scale or args.lock_global_scale
        or args.nada_global_sweep or args.bitr_power_search or args.weight_group_hessian
        or args.dacc_weighted_proxy or args.nada_diagnostics
        or args.export_quantized_model or args.validate_realquant_export
    ):
        parser.error("Lazy group scales require --nada_wmse weight, activation order and "
                     "fused global scales; standalone PPL experiment without export/other ablations.")
    if args.nada_diagnostics and (
        args.nada_wmse != "weight" or not args.eval_perplexity or args.eval_openllm
        or args.export_quantized_model or args.validate_realquant_export or args.compile
        or args.cpu_offload_eval or args.amp or args.num_sequences < 4
        or args.nada_refit_groups < 1 or args.nada_global_sweep
        or args.bitr_power_search or args.weight_group_hessian or args.dacc_weighted_proxy
    ):
        parser.error("--nada_diagnostics requires clean --nada_wmse weight PPL evaluation, "
                     "at least 4 calibration sequences and positive refit groups; "
                     "no export, AMP, compile, CPU-offload evaluation or other ablations.")
    if args.a_observer == "lss_wmse" and (args.export_quantized_model or args.validate_realquant_export):
        parser.error("Weighted activation LSS is HF PPL-only for now; its deployment kernel is not implemented.")
    if args.permuquant:
        args.channel_resort = "permuquant"
    if args.channel_resort == "permuquant":
        if (args.format != "nvfp" or args.transform_class != "identity"
                or args.channel_rescale != "none" or args.awq or args.gajs
                or args.smoothquant_alpha is not None
                or args.w_observer != "minmax" or args.a_observer != "minmax"):
            parser.error("PermuQuant-LLM requires NVFP4, identity transform, and no other preprocessing.")
        if args.permuquant_tokens < args.num_sequences or args.permuquant_threshold < 0:
            parser.error("PermuQuant token budget must cover all calibration sequences and threshold must be nonnegative.")
    if any(observer.startswith("scalesweep_") or observer == "scalesearch"
           for observer in (args.w_observer, args.a_observer)):
        if args.format != "nvfp":
            parser.error("ScaleSweep requires --format nvfp.")
        if args.lock_global_scale:
            parser.error("ScaleSweep computes its own global scale; omit --lock_global_scale.")
        if args.export_quantized_model or args.validate_realquant_export:
            hybrid_weight_export = args.nada_wmse == "weight" and args.a_observer == "lss"
            if ((args.a_observer not in {"scalesweep_mse", "scalesweep_wmse"}
                 and not hybrid_weight_export)
                    or args.w_observer not in {"scalesweep_mse", "scalesweep_wmse"}
                    or args.export_quantized_model != "realquant" or not args.gptq
                    or args.transform_class != "identity" or not args.fuse_global_scale
                    or args.hadamard_group_size != 16):
                parser.error(
                    "ScaleSweep vLLM export requires W/A ScaleSweep observers "
                    "(or --nada_wmse weight with activation LSS), "
                    "--gptq --export_quantized_model realquant "
                    "--transform_class identity --hadamard_group_size 16 --fuse_global_scale."
                )
        print(f"FP8 scale-search observers: weights={args.w_observer}, activations={args.a_observer}")
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
        
    # Removed check that forces transform_class to identity. 
    # We now fully support combining MR and KMeans!

    # --- Backward compatibility: map legacy channel_resort values to new split params ---
    if args.channel_resort == "kmeans_fp4_top3":
        args.channel_resort = "channel_cluster"
        if args.channel_rescale == "none":
            args.channel_rescale = "gics"
        print("[Compat] Mapped legacy 'kmeans_fp4_top3' -> channel_resort='channel_cluster' + channel_rescale='gics'")
    elif args.channel_resort in ("kmeans_fp4", "kmeans_fp4_w"):
        old_val = args.channel_resort
        args.channel_resort = "channel_cluster"
        print(f"[Compat] Mapped legacy '{old_val}' -> channel_resort='channel_cluster'")

    return args


def main():
    # Use expandable segments to handle the memory pressure of 25-shot ARC
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    args = parse_args()
    if args.cpu_offload_eval:
        if not args.cpu_offload_modules:
            raise ValueError("--cpu_offload_eval requires --cpu_offload_modules.")
        if not args.eval_perplexity or args.eval_openllm:
            raise ValueError(
                "--cpu_offload_eval supports only --eval_perplexity. "
                "Do not combine it with --eval_openllm."
            )
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
    if args.hf_device_map == "auto" and (args.w_bits < 16 or args.a_bits < 16):
        raise ValueError(
            "--hf_device_map auto currently supports only w16/a16 baseline evaluation. "
            "Use the existing --cpu_offload_modules path for layer-wise quantization."
        )
    if args.hf_device_map == "auto" and args.cpu_offload_modules:
        raise ValueError("Choose either --hf_device_map auto or --cpu_offload_modules, not both.")
    if args.hf_device_map == "auto" and args.hf_allow_legacy_accelerate:
        # Transformers 5 performs a version-gate before it calls Accelerate's device-map
        # implementation.  Accelerate 0.34 already exposes the basic dispatch APIs used
        # here, so allow an explicit, opt-in compatibility probe without changing the
        # installed environment.  Any missing legacy API will still fail with its original
        # traceback during loading; this flag never affects ordinary quantization runs.
        try:
            import importlib
            import accelerate
            from accelerate import dispatch_model, infer_auto_device_map  # noqa: F401
            import transformers.integrations.accelerate as hf_accelerate
        except ImportError as exc:
            raise RuntimeError(
                "--hf_allow_legacy_accelerate needs an installed Accelerate with "
                "dispatch_model and infer_auto_device_map."
            ) from exc

        if not hf_accelerate.is_accelerate_available():
            # `transformers.integrations.accelerate` conditionally imports helpers such
            # as `accelerate_max_memory` at module-import time.  Patching only its final
            # version check leaves those globals undefined, so patch the two exported
            # availability predicates and reload the integration module once.
            import transformers.utils as hf_utils
            import transformers.utils.import_utils as hf_import_utils

            _legacy_accelerate_available = lambda *unused_args, **unused_kwargs: True
            hf_utils.is_accelerate_available = _legacy_accelerate_available
            hf_import_utils.is_accelerate_available = _legacy_accelerate_available
            hf_accelerate = importlib.reload(hf_accelerate)
            print(
                "[HF device map] Experimental legacy-Accelerate compatibility bypass "
                f"enabled (installed accelerate={accelerate.__version__})."
            )

    model_load_kwargs = dict(
        dtype=args.dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if args.hf_device_map == "auto":
        offload_folder = args.hf_offload_folder or os.path.join(os.getcwd(), "hf_offload")
        os.makedirs(offload_folder, exist_ok=True)
        model_load_kwargs.update(
            device_map="auto",
            max_memory={0: args.hf_max_gpu_memory, "cpu": args.hf_max_cpu_memory},
            offload_folder=offload_folder,
            offload_state_dict=True,
        )
        print(
            "[HF device map] Loading with GPU+CPU dispatch "
            f"(gpu={args.hf_max_gpu_memory}, cpu={args.hf_max_cpu_memory}, "
            f"offload_folder={offload_folder})."
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_load_kwargs)
    is_hf_dispatched = args.hf_device_map == "auto"
    if not args.cpu_offload_modules and not is_hf_dispatched:
        model = model.to(device)
    elif is_hf_dispatched:
        print(f"[HF device map] {getattr(model, 'hf_device_map', {})}")
    # Disable cache during calibration; generation tasks enable it after
    # quantization once the repaired attention cache path is in place.
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
    if args.save_calibration_tokens:
        import tempfile
        from pathlib import Path

        destination = Path(args.save_calibration_tokens)
        tokens = torch.cat(calibration_data, dim=0).cpu()
        if destination.exists():
            previous = torch.load(destination, map_location="cpu", weights_only=True)
            if not torch.equal(previous, tokens):
                raise ValueError(f"Calibration tokens differ from {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                temp_path = Path(temporary.name)
                try:
                    torch.save(tokens, temporary)
                    temporary.close()
                    temp_path.replace(destination)
                finally:
                    temp_path.unlink(missing_ok=True)
        print(f"Calibration tokens: {destination} {tuple(tokens.shape)}")

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

    if args.compile and is_hf_dispatched:
        raise ValueError("--compile is not supported with --hf_device_map auto.")
    if args.compile:
        model = torch.compile(model)

    if args.eval_perplexity or args.eval_openllm:
        # A dispatched model must stay in place: Accelerate owns its GPU/CPU hooks.
        if not is_hf_dispatched:
            # Move model to CPU briefly to ensure NO hidden quantization tensors are kept on GPU
            model = model.to('cpu')
        if 'calibration_data' in locals():
            del calibration_data
        import gc
        for _ in range(3): gc.collect()
        torch.cuda.empty_cache()
        
        # DIAGNOSTIC: Check GPU after cleanup
        location = "after dispatched-model cleanup" if is_hf_dispatched else "after model.to('cpu') + empty_cache"
        print(f"\n[MEM DIAG] {location}:")
        print(f"[MEM DIAG]   PyTorch allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[MEM DIAG]   PyTorch reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        # Strip weight_quantizer from QLinear modules - not needed when _train_mode=False
        from src.quantization.qlinear import QLinear as _QLinear
        for name, module in model.named_modules():
            if isinstance(module, _QLinear):
                module.weight_quantizer = None
        
        # Do not collapse a dispatched 70B model back onto one GPU.  The PPL-only
        # CPU offload mode instead installs per-block transfer hooks below.
        if args.cpu_offload_eval:
            enable_cpu_offload_ppl(model, device)
        elif not is_hf_dispatched:
            model = model.to(device)
        # Keep cache disabled for full-sequence/perplexity work by default.
        # _run_task enables it only for autoregressive generation tasks.
        model.config.use_cache = False

        # === Memory Diagnostics ===
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        print(f"\n[MEM DIAG] Model parameters: {param_bytes / 1e9:.2f} GB")
        print(f"[MEM DIAG] Model buffers: {buffer_bytes / 1e9:.2f} GB")
        print(f"[MEM DIAG] Total model tensors: {(param_bytes + buffer_bytes) / 1e9:.2f} GB")
        # Check for FP32 params
        fp32_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.dtype == torch.float32)
        if fp32_bytes > 0:
            print(f"[MEM DIAG] WARNING: {fp32_bytes / 1e9:.2f} GB of FP32 parameters found!")
            for name, p in model.named_parameters():
                if p.dtype == torch.float32:
                    print(f"  FP32 param: {name} shape={list(p.shape)} ({p.numel() * 4 / 1e6:.1f} MB)")
        # Check non-parameter module attributes that are tensors on GPU
        non_param_gpu_bytes = 0
        for name, module in model.named_modules():
            for attr_name in dir(module):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(module, attr_name)
                    if isinstance(attr, torch.Tensor) and attr.is_cuda and not any(attr.data_ptr() == p.data_ptr() for p in model.parameters()):
                        non_param_gpu_bytes += attr.numel() * attr.element_size()
                except:
                    pass
        if non_param_gpu_bytes > 0:
            print(f"[MEM DIAG] Non-parameter GPU tensors: {non_param_gpu_bytes / 1e9:.2f} GB")
        print(f"[MEM DIAG] PyTorch allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"[MEM DIAG] PyTorch reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"=== End Memory Diagnostics ===\n")

    wikitext2_ppl = None
    c4_ppl = None

    if args.nada_diagnostics:
        from src.quantization.nada_diagnostics import run_diagnostics
        run_diagnostics(model, tokenizer, args)
        return

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

        # Ensure all model weights are contiguous to avoid on-the-fly allocations during F.linear
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
                
        # NOTE: WE MUST CLEAR CACHE AFTER QUANTIZATION BEFORE LM EVAL
        if 'calibration_data' in locals():
            del calibration_data
        gc.collect()
        torch.cuda.empty_cache()

    # OpenLLM v1 openllm (following https://arxiv.org/abs/2411.02355)
    if args.eval_openllm:

        results = {}
        # Cap max_position_embeddings to prevent lm_eval from using excessively long sequences
        _orig_max_pos = model.config.max_position_embeddings
        model.config.max_position_embeddings = min(model.config.max_position_embeddings, 2048)

        def _run_task(task_name, num_fewshot=0, batch_size=1, limit=None, **extra_kwargs):
            """Run a single lm_eval task with fresh HFLM to avoid accumulated state."""
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            # Create fresh HFLM per task to prevent KV cache / logits accumulation.
            # Generation tasks opt into the repaired KV-cache path below.
            use_kv_cache = task_name in {"gsm8k", "gsm8k_llama", "mbpp"}
            model.config.use_cache = use_kv_cache
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.use_cache = use_kv_cache
            effective_batch_size = batch_size
            if isinstance(args.lm_eval_batch_size, int):
                effective_batch_size = min(batch_size, args.lm_eval_batch_size)
            lm = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=effective_batch_size,
                max_length=2048,
            )
            eval_kwargs = {"model": lm, "tasks": [task_name], "num_fewshot": num_fewshot, "confirm_run_unsafe_code": True}
            if limit is not None:
                eval_kwargs["limit"] = limit
            eval_kwargs.update(extra_kwargs)
            task_results = lm_eval.simple_evaluate(**eval_kwargs)["results"]
            results.update(task_results)
            print(make_table({"results": task_results, "versions": {}, "n-shot": {}, "higher_is_better": {}}))
            # Immediately free HFLM and all its internal state
            del lm
            gc.collect()
            torch.cuda.empty_cache()

        # Winogrande (5-shot) - Can handle larger batch
        if "winogrande" in args.lm_eval_tasks:
            _run_task("winogrande", num_fewshot=5, batch_size=64, limit=args.limit)
        # SingleEq (Single-step arithmetic operations)
        if "singleeq" in args.lm_eval_tasks:
            _run_task("singleeq", num_fewshot=5, batch_size=16, limit=args.limit)
        # Hellaswag (10-shot) - Medium batch, limit=400 (four-over-six convention)
        if "hellaswag" in args.lm_eval_tasks:
            _run_task("hellaswag", num_fewshot=10, batch_size=8, limit=4000)
        # PIQA (0-shot) - Can handle very large batch
        if "piqa" in args.lm_eval_tasks:
            _run_task("piqa", num_fewshot=0, batch_size=64, limit=args.limit)
        # ARC Challenge (25-shot) - Must keep batch_size=1, limit=400 (four-over-six convention)
        if "arc_challenge" in args.lm_eval_tasks:
            _run_task("arc_challenge", num_fewshot=25, batch_size=8, limit=args.limit)
        # BoolQ (0-shot) - Passages are long, use smaller batch
        if "boolq" in args.lm_eval_tasks:
            _run_task("boolq", num_fewshot=0, batch_size=8, limit=args.limit)
        # GSM8K (requires chat template for llama specific version)
        if "gsm8k_llama" in args.lm_eval_tasks:
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
                _run_task("gsm8k_llama", apply_chat_template=True, fewshot_as_multiturn=True, batch_size=8, limit=args.limit)
            else:
                print("Skipping gsm8k_llama: no chat template.")
        # Standard GSM8K (5-shot)
        if "gsm8k" in args.lm_eval_tasks and "gsm8k_llama" not in args.lm_eval_tasks:
            _run_task("gsm8k", num_fewshot=5, batch_size=16, limit=args.limit)
        # MMLU (5-shot)
        if "mmlu" in args.lm_eval_tasks:
            _run_task("mmlu", num_fewshot=5, batch_size=4, limit=200)
        # SQuAD V2 (1-shot is common)
        if "squadv2" in args.lm_eval_tasks:
            _run_task("squadv2", num_fewshot=1, batch_size=4, limit=args.limit)
        # DROP (3-shot is common)
        if "drop" in args.lm_eval_tasks:
            _run_task("drop", num_fewshot=3, batch_size=4, limit=args.limit)
        # MathQA (5-shot)
        if "mathqa" in args.lm_eval_tasks:
            _run_task("mathqa", num_fewshot=5, batch_size=4, limit=args.limit)        # ARC Easy (25-shot)
        if "arc_easy" in args.lm_eval_tasks:
            _run_task("arc_easy", num_fewshot=25, batch_size=8, limit=4000)
        # MBPP (3-shot)
        if "mbpp" in args.lm_eval_tasks:
            _run_task("mbpp", num_fewshot=3, batch_size=16, limit=args.limit)
        # Lambada (0-shot)
        if "lambada_openai" in args.lm_eval_tasks or "lambada" in args.lm_eval_tasks:
            task_name = "lambada_openai" if "lambada_openai" in args.lm_eval_tasks else "lambada"
            _run_task(task_name, num_fewshot=0, batch_size=16, limit=args.limit)
            
        # MMLU CoT (requires chat template)
        if "mmlu_cot_llama" in args.lm_eval_tasks:
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
                _run_task("mmlu_cot_llama", apply_chat_template=True, fewshot_as_multiturn=True, batch_size=8, limit=args.limit)
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
            keys = list(res_dict.keys())
            for target in ["pass@1", "exact_match", "acc_norm", "acc", "f1", "word_perplexity", "byte_perplexity"]:
                matches = [k for k in keys if target in k]
                if matches:
                    return res_dict.get(matches[0])
            return list(res_dict.values())[0] if res_dict else None

        filtered_results = {}
        for task, res in results.items():
            if task.startswith("mmlu_") and task not in args.lm_eval_tasks:
                continue
            filtered_results[task] = get_best_metric(res)

        summary = {
            "model": args.model_name_or_path,
            "wikitext2_ppl": wikitext2_ppl,
            "c4_ppl": c4_ppl,
            "results": filtered_results
        }
        import json
        print(f"\n[RESULT_SUMMARY] {json.dumps(summary)}")

if __name__ == "__main__":
    main()
