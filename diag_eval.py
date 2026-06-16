import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import argparse
import copy
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.data_utils import get_data
from src.quantization.qconfig import prepare_quantization_config
from src.quantization.gptq import gptq_quantization

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--dataset_name_or_path", type=str, default="fineweb-edu")
    parser.add_argument("--num_sequences", type=int, default=128)
    parser.add_argument("--sequence_length", type=int, default=2048)
    parser.add_argument("--kmeans_block_size", type=int, default=128)
    
    # Quantization params
    parser.add_argument("--format", type=str, default="nvfp")
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--w_group_size", type=int, default=16)
    parser.add_argument("--a_group_size", type=int, default=16)
    parser.add_argument("--transform_class", type=str, default="identity")
    parser.add_argument("--channel_resort", type=str, default="kmeans_fp4")
    parser.add_argument("--w_observer", type=str, default="mse")
    parser.add_argument("--quantization_order", type=str, default="activation")
    parser.add_argument("--rel_damp", type=float, default=0.01)
    parser.add_argument("--fuse_global_scale", action="store_true", default=True)
    parser.add_argument("--gptq", action="store_true", default=True)
    
    # Defaults needed by GPTQ logic
    parser.add_argument("--stagger_lambda", type=float, default=2.0)
    parser.add_argument("--scale_precision", type=str, default="e4m3")
    parser.add_argument("--gajs", action="store_true")
    parser.add_argument("--awq", type=int, default=0)
    parser.add_argument("--hadamard_group_size", type=int, default=-1)
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--export_quantized_model", type=str, default=None)
    parser.add_argument("--a_observer", type=str, default="minmax")
    parser.add_argument("--cpu_offload_activations", action="store_true")
    parser.add_argument("--lock_global_scale", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--quantize_mlp_separately", action="store_true")
    
    # Missing args from model_quant.py
    parser.add_argument("--w_granularity", type=str, default="group")
    parser.add_argument("--a_granularity", type=str, default="group")
    parser.add_argument("--awq_for_act", type=float, default=0.0)
    parser.add_argument("--outlier_ratio", type=float, default=0.01)
    parser.add_argument("--cpu_offload_modules", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--show_act_mse", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_perplexity", action="store_true")
    parser.add_argument("--eval_openllm", action="store_true")
    parser.add_argument("--disable_thinking", action="store_true")
    
    return parser.parse_args()

class CacheHook:
    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.handle = None
        
    def hook(self, module, inp, out):
        self.inputs.append(inp[0].detach().cpu())
        self.outputs.append(out.detach().cpu())
        
    def clear(self):
        self.inputs = []
        self.outputs = []

def main():
    args = get_args()
    if args.dtype != "auto" and isinstance(args.dtype, str):
        args.dtype = getattr(torch, args.dtype)
        
    # Force offload activations to CPU to save ~4GB of VRAM during GPTQ (prevents OOM on 8B models)
    args.cpu_offload_activations = True
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # To suppress quantization logs
    import logging
    import sys
    import os
    
    class HiddenPrints:
        def __enter__(self):
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

    print(f"Loading {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    
    print("Preparing calibration data...")
    with HiddenPrints():
        calibration_data = get_data(args.dataset_name_or_path, tokenizer, args.num_sequences, args.sequence_length)
    
    target_layers = [0, 15, 31]
    
    # 1. Baseline FP16 Model
    print("Running Baseline FP16 Model to collect activation targets (this will be fast)...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        dtype=args.dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    ).to(device)
    model_fp16.requires_grad_(False)
    model_fp16.config.use_cache = False
    
    # Only use 8 sequences for diagnostic evaluation to prevent massive memory swapping (RAM)
    eval_data = calibration_data[:8]
    
    orig_weights = {}
    orig_hooks = {}
    for layer_idx in target_layers:
        if layer_idx < len(model_fp16.model.layers):
            orig_weights[layer_idx] = model_fp16.model.layers[layer_idx].mlp.down_proj.weight.data.clone().to(device)
            orig_hooks[layer_idx] = CacheHook()
            orig_hooks[layer_idx].handle = model_fp16.model.layers[layer_idx].mlp.down_proj.register_forward_hook(orig_hooks[layer_idx].hook)
            
    with torch.no_grad():
        for batch in eval_data:
            model_fp16(batch.to(device))
            
    for layer_idx in orig_hooks:
        orig_hooks[layer_idx].handle.remove()
    del model_fp16
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    def quantize_and_eval(block_size):
        args.kmeans_block_size = block_size
        model_q = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            dtype=args.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
        ).to(device)
        model_q.requires_grad_(False)
        model_q.config.use_cache = False
        
        # We REMOVED HiddenPrints here so you can see the quantization progress!
        prepare_quantization_config(model_q, args.format, args)
        gptq_quantization(model_q, calibration_data, args, device)
            
        q_hooks = {}
        for layer_idx in target_layers:
            if layer_idx < len(model_q.model.layers):
                q_hooks[layer_idx] = CacheHook()
                q_hooks[layer_idx].handle = model_q.model.layers[layer_idx].mlp.down_proj.register_forward_hook(q_hooks[layer_idx].hook)
                
        with torch.no_grad():
            for batch in eval_data:
                model_q(batch.to(device))
                
        for layer_idx in q_hooks:
            q_hooks[layer_idx].handle.remove()
            
        results = {}
        for layer_idx in target_layers:
            if layer_idx >= len(model_q.model.layers): continue
            
            q_weight = model_q.model.layers[layer_idx].mlp.down_proj.weight.data.to(device)
            w_mse = F.mse_loss(q_weight.float(), orig_weights[layer_idx].float()).item()
            
            o_inputs = torch.cat(orig_hooks[layer_idx].inputs, dim=0).to(device)
            q_inputs = torch.cat(q_hooks[layer_idx].inputs, dim=0).to(device)
            a_mse = F.mse_loss(q_inputs.float(), o_inputs.float()).item()
            
            o_outputs = torch.cat(orig_hooks[layer_idx].outputs, dim=0).to(device)
            q_outputs = torch.cat(q_hooks[layer_idx].outputs, dim=0).to(device)
            out_mse = F.mse_loss(q_outputs.float(), o_outputs.float()).item()
            
            results[layer_idx] = {"W_MSE": w_mse, "A_MSE": a_mse, "Out_MSE": out_mse}
            
        del model_q
        torch.cuda.empty_cache()
        return results

    # Save the target block size before it gets mutated by the first call
    target_block_size = args.kmeans_block_size

    print("Running Global K-means (block_size = -1)...")
    res_global = quantize_and_eval(-1)
    
    print(f"Running Block K-means (block_size = {target_block_size})...")
    res_block = quantize_and_eval(target_block_size)

    print("\n" + "="*85)
    print(f"{'Layer':<6} | {'Metric':<10} | {'Baseline (FP16)':<15} | {'Global K-Means (-1)':<20} | {f'Block K-Means ({target_block_size})':<20} | {'Diff (Block - Global)'}")
    print("-" * 85)
    for layer_idx in target_layers:
        if layer_idx not in res_global: continue
        for metric in ["W_MSE", "A_MSE", "Out_MSE"]:
            val_g = res_global[layer_idx][metric]
            val_b = res_block[layer_idx][metric]
            diff = val_b - val_g
            diff_str = f"{diff:+.4e}"
            print(f"{layer_idx:<6} | {metric:<10} | {0.0:<15.4e} | {val_g:<20.4e} | {val_b:<20.4e} | {diff_str}")
        print("-" * 85)
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
