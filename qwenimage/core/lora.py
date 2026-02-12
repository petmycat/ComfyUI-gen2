"""
Gen2 QwenImage Core - LoRA Merge/Unmerge Functions

Adapted from VideoX for ComfyUI model structure.
Uses ComfyUI-style key mapping for broad LoRA format compatibility.
"""

from collections import defaultdict

import torch
from safetensors.torch import load_file as load_safetensors


def _build_lora_key_map(transformer):
    """
    Build a mapping from various LoRA key formats to actual model weight keys.
    
    This follows ComfyUI's approach: iterate the model's state_dict and generate
    all possible LoRA key formats that could map to each weight.
    
    Args:
        transformer: The transformer model
        
    Returns:
        dict: {lora_key: model_weight_key}
    """
    key_map = {}
    sd = transformer.state_dict()
    
    for k in sd.keys():
        if k.endswith(".weight"):
            # Remove .weight suffix for the base key
            base_key = k[:-len(".weight")]
            
            # Format 1: lora_unet_ + key with dots replaced by underscores
            key_lora = base_key.replace(".", "_")
            key_map[f"lora_unet_{key_lora}"] = k
            key_map[f"lora_unet__{key_lora}"] = k  # Double underscore variant
            
            # Format 2: Generic format (just the key path)
            key_map[base_key] = k
            
            # Format 3: With transformer prefix variations
            if not base_key.startswith("transformer"):
                key_map[f"transformer.{base_key}"] = k
                key_map[f"transformer_{base_key.replace('.', '_')}"] = k
                key_map[f"lora_unet_transformer_{base_key.replace('.', '_')}"] = k
                key_map[f"lora_unet__transformer_{base_key.replace('.', '_')}"] = k
            
            # Format 4: diffusion_model prefix
            key_map[f"diffusion_model.{base_key}"] = k
    
    return key_map


def _parse_lora_weights(lora_state_dict, key_map):
    """
    Parse LoRA state dict and match keys to model weights using the key map.
    
    Args:
        lora_state_dict: The LoRA safetensors state dict
        key_map: Mapping from LoRA keys to model weight keys
        
    Returns:
        dict: {model_weight_key: {'up': tensor, 'down': tensor, 'alpha': float}}
    """
    # First, group LoRA weights by their base key
    lora_groups = defaultdict(dict)
    
    for lora_key, value in lora_state_dict.items():
        # Determine the base key and weight type
        base_key = None
        weight_type = None
        
        # Handle various LoRA weight naming conventions
        if ".lora_up.weight" in lora_key:
            base_key = lora_key.replace(".lora_up.weight", "")
            weight_type = "up"
        elif ".lora_down.weight" in lora_key:
            base_key = lora_key.replace(".lora_down.weight", "")
            weight_type = "down"
        elif ".lora_A.weight" in lora_key:
            base_key = lora_key.replace(".lora_A.weight", "")
            weight_type = "down"  # lora_A = down
        elif ".lora_B.weight" in lora_key:
            base_key = lora_key.replace(".lora_B.weight", "")
            weight_type = "up"  # lora_B = up
        elif ".alpha" in lora_key:
            base_key = lora_key.replace(".alpha", "")
            weight_type = "alpha"
        elif lora_key.endswith("_lora_up_weight") or lora_key.endswith(".lora_up_weight"):
            base_key = lora_key.rsplit("_lora_up_weight", 1)[0].rsplit(".lora_up_weight", 1)[0]
            weight_type = "up"
        elif lora_key.endswith("_lora_down_weight") or lora_key.endswith(".lora_down_weight"):
            base_key = lora_key.rsplit("_lora_down_weight", 1)[0].rsplit(".lora_down_weight", 1)[0]
            weight_type = "down"
        elif lora_key.endswith("_alpha"):
            base_key = lora_key[:-6]  # Remove _alpha
            weight_type = "alpha"
        else:
            continue
        
        if base_key and weight_type:
            lora_groups[base_key][weight_type] = value
    
    # Now match base keys to model weights
    matched_weights = {}
    unmatched_keys = []
    
    for base_key, weights in lora_groups.items():
        if "up" not in weights or "down" not in weights:
            continue  # Need both up and down
        
        # Try to find matching model weight key
        model_key = None
        
        # Try the base key directly
        if base_key in key_map:
            model_key = key_map[base_key]
        else:
            # Try various normalizations
            normalized_keys = []
            
            # Remove common prefixes
            clean_key = base_key
            for prefix in ["lora_unet__", "lora_unet_", "diffusion_model.", "transformer.", "unet."]:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break
            
            # Try with and without transformer prefix
            normalized_keys.append(clean_key)
            normalized_keys.append(f"transformer_{clean_key}")
            normalized_keys.append(f"lora_unet_{clean_key}")
            normalized_keys.append(f"lora_unet__{clean_key}")
            normalized_keys.append(f"lora_unet_transformer_{clean_key}")
            normalized_keys.append(f"lora_unet__transformer_{clean_key}")
            
            # Also try with dots replaced by underscores and vice versa
            normalized_keys.append(clean_key.replace("_", "."))
            normalized_keys.append(clean_key.replace(".", "_"))
            
            for nk in normalized_keys:
                if nk in key_map:
                    model_key = key_map[nk]
                    break
        
        if model_key:
            alpha = weights.get("alpha", None)
            if alpha is not None:
                alpha = alpha.item() if hasattr(alpha, 'item') else float(alpha)
            matched_weights[model_key] = {
                "up": weights["up"],
                "down": weights["down"],
                "alpha": alpha
            }
        else:
            unmatched_keys.append(base_key)
    
    return matched_weights, unmatched_keys


def gen2_merge_lora(transformer, lora_path, multiplier, device='cpu', dtype=torch.float32):
    """
    Merge LoRA weights into the transformer model using ComfyUI-style key mapping.
    
    Args:
        transformer: The transformer model (ComfyUI's diffusion_model)
        lora_path: Path to the LoRA safetensors file
        multiplier: LoRA strength multiplier
        device: Device to perform operations on
        dtype: Data type for computations
    
    Returns:
        transformer: The modified transformer (same object, modified in-place)
    """
    if lora_path is None:
        return transformer
    
    # Build key map from model's state dict
    key_map = _build_lora_key_map(transformer)
    
    # Load and parse LoRA weights
    lora_state_dict = load_safetensors(lora_path)
    matched_weights, unmatched_keys = _parse_lora_weights(lora_state_dict, key_map)
    
    # Debug: print sample keys
    sample_lora_keys = list(lora_state_dict.keys())[:3]
    print(f"[Gen2 LoRA] Sample LoRA keys: {sample_lora_keys}")
    print(f"[Gen2 LoRA] Matched {len(matched_weights)} layers, {len(unmatched_keys)} unmatched")
    
    merged_count = 0
    failed_layers = []
    
    # Get model's state dict for direct modification
    model_sd = transformer.state_dict()
    
    for model_key, lora_data in matched_weights.items():
        try:
            # Get the weight tensor
            if model_key not in model_sd:
                failed_layers.append(f"{model_key} (not in state_dict)")
                continue
            
            weight = model_sd[model_key]
            origin_dtype = weight.dtype
            origin_device = weight.device
            
            # Prepare LoRA weights
            weight_up = lora_data["up"].to(device, dtype)
            weight_down = lora_data["down"].to(device, dtype)
            
            # Calculate alpha scaling
            alpha = lora_data["alpha"]
            if alpha is not None:
                alpha = alpha / weight_up.shape[1]
            else:
                alpha = 1.0
            
            # Calculate LoRA diff
            weight = weight.to(device, dtype)
            if len(weight_up.shape) == 4:
                lora_diff = torch.mm(
                    weight_up.squeeze(3).squeeze(2), 
                    weight_down.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3)
            else:
                lora_diff = torch.mm(weight_up, weight_down)
            
            # Apply LoRA
            weight = weight + multiplier * alpha * lora_diff
            model_sd[model_key] = weight.to(origin_device, origin_dtype)
            
            merged_count += 1
            if merged_count <= 3:
                print(f"[Gen2 LoRA] Merged: {model_key}")
                
        except Exception as e:
            failed_layers.append(f"{model_key} ({e})")
            continue
    
    # Load modified state dict back
    transformer.load_state_dict(model_sd)
    
    if failed_layers:
        print(f"[Gen2 LoRA] Warning: {len(failed_layers)} layers failed to merge")
        if len(failed_layers) <= 10:
            for fl in failed_layers:
                print(f"  Failed: {fl}")
    
    if unmatched_keys and len(unmatched_keys) <= 10:
        print(f"[Gen2 LoRA] Unmatched LoRA keys (first 10):")
        for uk in unmatched_keys[:10]:
            print(f"  {uk}")
    
    print(f"[Gen2 LoRA] Merged {merged_count} LoRA layers with strength {multiplier}")
    return transformer


def gen2_unmerge_lora(transformer, lora_path, multiplier, device='cpu', dtype=torch.float32):
    """
    Unmerge LoRA weights from the transformer model.
    Reverses the merge operation by subtracting the LoRA weights.
    
    Args:
        transformer: The transformer model (ComfyUI's diffusion_model)
        lora_path: Path to the LoRA safetensors file
        multiplier: LoRA strength multiplier (same as used in merge)
        device: Device to perform operations on
        dtype: Data type for computations
    
    Returns:
        transformer: The modified transformer (same object, modified in-place)
    """
    if lora_path is None:
        return transformer
    
    # Build key map from model's state dict
    key_map = _build_lora_key_map(transformer)
    
    # Load and parse LoRA weights
    lora_state_dict = load_safetensors(lora_path)
    matched_weights, _ = _parse_lora_weights(lora_state_dict, key_map)
    
    unmerged_count = 0
    model_sd = transformer.state_dict()
    
    for model_key, lora_data in matched_weights.items():
        try:
            if model_key not in model_sd:
                continue
            
            weight = model_sd[model_key]
            origin_dtype = weight.dtype
            origin_device = weight.device
            
            weight_up = lora_data["up"].to(device, dtype)
            weight_down = lora_data["down"].to(device, dtype)
            
            alpha = lora_data["alpha"]
            if alpha is not None:
                alpha = alpha / weight_up.shape[1]
            else:
                alpha = 1.0
            
            weight = weight.to(device, dtype)
            if len(weight_up.shape) == 4:
                lora_diff = torch.mm(
                    weight_up.squeeze(3).squeeze(2), 
                    weight_down.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3)
            else:
                lora_diff = torch.mm(weight_up, weight_down)
            
            # SUBTRACT to unmerge
            weight = weight - multiplier * alpha * lora_diff
            model_sd[model_key] = weight.to(origin_device, origin_dtype)
            
            unmerged_count += 1
        except Exception as e:
            print(f"[Gen2 LoRA] Failed to unmerge {model_key}: {e}")
            continue
    
    transformer.load_state_dict(model_sd)
    print(f"[Gen2 LoRA] Unmerged {unmerged_count} LoRA layers")
    return transformer

