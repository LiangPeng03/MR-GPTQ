import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import argparse
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.metrics.perplexity import compute_perplexity
from src.utils.data_utils import get_wikitext2, get_c4_eval
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table
except ImportError:
    try:
        import lm_eval
        from lm_eval.models.gpt2 import HFLM
        if hasattr(HFLM, "tokenizer_check"):
            HFLM.tokenizer_check = lambda self: None
        from lm_eval.evaluator import make_table
    except ImportError:
        lm_eval = None

import sys
import os

class ActPerturber:
    def __init__(self, eps, perturb_type):
        self.eps = eps
        self.perturb_type = perturb_type
        self.handles = []
        
    def hook_fn(self, module, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        if not isinstance(x, torch.Tensor):
            return out
            
        abs_x = x.abs()
        flat_x = abs_x.view(-1)
        if flat_x.numel() == 0:
            return out
            
        # Subsample to avoid "quantile() input tensor is too large" error
        if flat_x.numel() > 1000000:
            indices = torch.randint(0, flat_x.numel(), (1000000,), device=flat_x.device)
            sample = flat_x[indices]
        else:
            sample = flat_x
            
        # Using float32 for quantile to avoid bfloat16 issues
        threshold = torch.quantile(sample.float(), 0.9).to(x.device).to(x.dtype)
        
        if self.perturb_type in [1, 3, 4]:
            mask = abs_x <= threshold
        elif self.perturb_type in [2, 5, 6]:
            mask = abs_x > threshold
        else:
            mask = torch.zeros_like(x, dtype=torch.bool)
            
        if self.perturb_type in [1, 2]:
            noise = (torch.rand_like(x) * 2 - 1) * self.eps
            x_new = x.clone()
            x_new[mask] += noise[mask]
        elif self.perturb_type in [3, 5]:
            x_new = x.clone()
            x_new[mask] += self.eps
        elif self.perturb_type in [4, 6]:
            x_new = x.clone()
            x_new[mask] -= self.eps
        else:
            x_new = x
            
        if isinstance(out, tuple):
            return (x_new,) + out[1:]
        return x_new

    def register(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                handle = module.register_forward_hook(self.hook_fn)
                self.handles.append(handle)
                
    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

def calibrate_epsilon(model):
    stds = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            stds.append(module.weight.data.std().item())
    return 0.01 * np.median(stds)

def apply_weight_perturbation(model, eps, perturb_type):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data
            abs_w = w.abs()
            flat_w = abs_w.view(-1)
            if flat_w.numel() == 0: continue
            
            # Subsample to avoid "quantile() input tensor is too large" error
            if flat_w.numel() > 1000000:
                indices = torch.randint(0, flat_w.numel(), (1000000,), device=flat_w.device)
                sample = flat_w[indices]
            else:
                sample = flat_w
                
            threshold = torch.quantile(sample.float(), 0.9).to(w.device).to(w.dtype)
            
            if perturb_type in [1, 3, 4]:
                mask = abs_w <= threshold
            elif perturb_type in [2, 5, 6]:
                mask = abs_w > threshold
            else:
                mask = torch.zeros_like(w, dtype=torch.bool)
                

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    return parser.parse_args()

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def main():
    args = get_args()
    
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        lm_eval = None
        
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.metrics.perplexity import compute_perplexity
    from src.utils.data_utils import get_wikitext2, get_c4_eval
    
    import transformers
    transformers.logging.set_verbosity_error()
    import warnings
    warnings.filterwarnings("ignore")

    print(f"Loading {args.model_name_or_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True
    ).to(device)
    model.requires_grad_(False)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    from src.utils.common_utils import fix_seed
    
    # Scale up epsilon by 10x as requested
    def calibrate_epsilon_10x(m):
        stds = []
        for name, module in m.named_modules():
            if isinstance(module, torch.nn.Linear):
                stds.append(module.weight.data.std().item())
        return 0.1 * np.median(stds)
        
    eps = calibrate_epsilon_10x(model)
    print(f"Calibrated 0.1x Epsilon: {eps:.6f}")
    
    orig_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            orig_weights[name] = module.weight.data.clone().cpu()

    def restore_weights():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data.copy_(orig_weights[name].to(module.weight.device))
                
    def get_mask_and_perturb(tensor, ptype):
        w = tensor.clone()
        abs_w = w.abs()
        
        # Use median to avoid outlier skewing (activations have massive outliers)
        median_val = torch.median(abs_w)
        eps = median_val * 0.5 # Coefficient can be tuned, using 1.0x median for now
        
        flat_t = tensor.abs().view(-1)
        if flat_t.numel() == 0: return tensor
        
        if flat_t.numel() > 1000000:
            indices = torch.randint(0, flat_t.numel(), (1000000,), device=flat_t.device)
            sample = flat_t[indices]
        else:
            sample = flat_t
        threshold_50 = torch.quantile(sample.float(), 0.5).to(tensor.dtype)
        threshold_90 = torch.quantile(sample.float(), 0.9).to(tensor.dtype)
        
        # Determine the target bin
        if ptype == 1:
            bin_mask = tensor.abs() <= threshold_50
        elif ptype == 2:
            bin_mask = (tensor.abs() > threshold_50) & (tensor.abs() <= threshold_90)
        elif ptype == 3:
            bin_mask = tensor.abs() > threshold_90
        else:
            bin_mask = torch.zeros_like(tensor, dtype=torch.bool)
            
        # The number of elements in the Top 10% bin
        target_k = (tensor.abs() > threshold_90).sum().item()
        
        # Randomly sample target_k elements from the chosen bin
        flat_bin_mask = bin_mask.flatten()
        valid_indices = torch.nonzero(flat_bin_mask).squeeze()
        
        final_mask = torch.zeros_like(flat_bin_mask, dtype=torch.bool)
        if valid_indices.numel() > 0:
            if valid_indices.numel() <= target_k:
                # If for some reason the bin has fewer elements, select all
                final_mask[valid_indices] = True
            else:
                # Randomly sample target_k elements
                perm = torch.randperm(valid_indices.size(0), device=valid_indices.device)
                selected_indices = valid_indices[perm[:target_k]]
                final_mask[selected_indices] = True
                
        final_mask = final_mask.view(tensor.shape)
        
        tensor_new = tensor.clone()
        # Apply Signed noise {+eps, -eps}
        noise = (torch.randint_like(tensor, 0, 2) * 2 - 1) * eps
        tensor_new[final_mask] += noise[final_mask]
            
        return tensor_new

    def apply_weight_perturbation(model, ptype):
        for name, module in model.named_modules():
            # Exclude lm_head to prevent massive OOM from vocab-sized matrices
            if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
                module.weight.data = get_mask_and_perturb(module.weight.data, ptype)

    class ActPerturber:
        def __init__(self, ptype):
            self.ptype = ptype
            self.handles = []
            
        def hook_fn(self, module, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            if not isinstance(x, torch.Tensor):
                return out
            x_new = get_mask_and_perturb(x, self.ptype)
            if isinstance(out, tuple):
                return (x_new,) + out[1:]
            return x_new

        def register(self, model):
            for name, module in model.named_modules():
                # Exclude lm_head to prevent massive OOM from vocab-sized tensors
                if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
                    handle = module.register_forward_hook(self.hook_fn)
                    self.handles.append(handle)
                    
        def remove(self):
            for h in self.handles:
                h.remove()
            self.handles = []

    # Print descriptions of each type
    print("\n[Perfect Control Variable Experiment]")
    print("Rule: We perturb exactly the SAME NUMBER of elements (10% of total) in all cases.")
    print("Rule: We apply exactly the SAME NOISE MAGNITUDE (+/- eps) in all cases.")
    print("Type 1: Randomly sample 10% elements from the Bottom 50% values")
    print("Type 2: Randomly sample 10% elements from the Middle 40% values")
    print("Type 3: Select all 10% elements from the Top 10% values")
    print("* Total noise energy is mathematically EQUAL across all types!\n")

    # Table header
    print("-" * 105)
    print(f"{'Target':<6} | {'Type':<4} | {'Description':<30} | {'Wikitext2':<10} | {'C4':<10} | {'PIQA':<10} | {'Wino':<10}")
    print("-" * 105)
    
    type_descriptions = {
        0: "Baseline",
        1: "Bottom 50% (Sampled) +/- e",
        2: "Middle 40% (Sampled) +/- e",
        3: "Top 10% (All) +/- e"
    }

    combinations = []
    for target in ["A", "W", "WA"]:
        for ptype in [1, 2, 3]:
            combinations.append((target, ptype))

    for target, ptype in combinations:
        restore_weights()
        perturber = None
        fix_seed(0)
        
        if ptype != 0 and target != "Base":
            if "W" in target:
                apply_weight_perturbation(model, ptype)
            if "A" in target:
                perturber = ActPerturber(ptype)
                perturber.register(model)
        
        with HiddenPrints():
            model.config.max_position_embeddings = min(model.config.max_position_embeddings, 2048)
            
            testloader = get_wikitext2(tokenizer, sequence_length=2048)
            wiki_ppl = compute_perplexity(model, testloader)
            
            testloader = get_c4_eval(tokenizer, sequence_length=2048)
            c4_ppl = compute_perplexity(model, testloader)
            
            piqa_val = "N/A"
            wino_val = "N/A"
            if lm_eval is not None:
                import gc
                
                # Function to run task exactly like model_quant.py to guarantee same results
                def _run_task_local(task_name, num_fewshot, bs):
                    gc.collect()
                    torch.cuda.empty_cache()
                    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=bs, max_length=2048)
                    import logging
                    logging.getLogger("lm-eval").setLevel(logging.ERROR)
                    res = lm_eval.simple_evaluate(model=lm, tasks=[task_name], num_fewshot=num_fewshot, log_samples=False)["results"]
                    del lm
                    gc.collect()
                    torch.cuda.empty_cache()
                    return res
                
                res_piqa = _run_task_local("piqa", 0, 64)
                if "piqa" in res_piqa:
                    # 优先取 acc_norm,none -> acc_norm -> acc,none -> acc
                    piqa_val = res_piqa["piqa"].get("acc_norm,none", res_piqa["piqa"].get("acc_norm", res_piqa["piqa"].get("acc,none", res_piqa["piqa"].get("acc", "N/A"))))
                    
                res_wino = _run_task_local("winogrande", 5, 64)
                if "winogrande" in res_wino:
                    # 优先取 acc_norm,none -> acc_norm -> acc,none -> acc
                    wino_val = res_wino["winogrande"].get("acc_norm,none", res_wino["winogrande"].get("acc_norm", res_wino["winogrande"].get("acc,none", res_wino["winogrande"].get("acc", "N/A"))))
                    
            if perturber:
                perturber.remove()
            
        piqa_str = f"{piqa_val:.5f}" if isinstance(piqa_val, float) else str(piqa_val)
        wino_str = f"{wino_val:.5f}" if isinstance(wino_val, float) else str(wino_val)
        desc = type_descriptions[ptype]
        
        print(f"{target:<6} | {ptype:<4} | {desc:<30} | {wiki_ppl:<10.3f} | {c4_ppl:<10.3f} | {piqa_str:<10} | {wino_str:<10}")
        sys.stdout.flush()

    print("-" * 105)

if __name__ == "__main__":
    main()
