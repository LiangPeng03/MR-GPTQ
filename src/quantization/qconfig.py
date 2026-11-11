from typing import Any

def prepare_quantization_config(
    hadamard_group_size: int, 
    format: str,
    pseudoquantization: bool = False,
    activation_observer: str = "minmax",
    identity_transform: bool = False,
) -> dict[str, Any]:
    if activation_observer == "lss_wmse":
        raise ValueError("Weighted LSS has no deployment kernel yet; do not export as abs_max/LSS.")
    if format in ["mxfp", "nvfp"]:
        forward_method = "abs_max"
        if format == "nvfp" and activation_observer in {"lss", "scalesweep_mse", "scalesweep_wmse"}:
            forward_method = activation_observer
        return {
            "forward_dtype": f"{format}4",
            "backward_dtype": "bf16",
            "forward_method": forward_method,
            # The current vLLM LSS kernel deliberately supports Ours only:
            # NVFP4 group-16 with no Hadamard transform.
            "lss_identity_transform": forward_method == "lss" and identity_transform,
            "scalesweep_identity_transform": forward_method.startswith("scalesweep_") and identity_transform,
            "hadamard_group_size":hadamard_group_size,
            "modules_to_not_convert": ["lm_head"],
            "quant_method": "fp_quant",
            "store_master_weights": False,
            "pseudoquantization": pseudoquantization
        }
    else:
        raise ValueError(f"Invalid format: {format}")
