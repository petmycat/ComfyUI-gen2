# QwenImage ControlNet Fun - VideoX 2026 Architecture
# Full implementation matching VideoX exactly using diffusers components

"""
Gen2 QwenImage ControlNet - 100% VideoX Compatible

This implementation creates standalone control blocks that:
1. Use diffusers' Attention and FeedForward classes (identical to VideoX)
2. Match VideoX's exact forward logic including stacking/unstacking
3. Integrate with ComfyUI's native model loading (fast, low RAM)
4. Use custom sampler for precise control over hint injection

Architecture matches:
- videox_fun/models/qwenimage_transformer2d.py (QwenImageTransformerBlock)
- videox_fun/models/qwenimage_transformer2d_control.py (QwenImageControlTransformerBlock)
"""

import os
import sys
import gc
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils
import comfy.latent_formats
import folder_paths

# =============================================================================
# VideoX Import Helper - Add videox-fun to path for imports
# =============================================================================

def _setup_videox_imports():
    """
    Add videox-fun custom node to sys.path so we can import videox_fun modules.
    VideoX is installed as a ComfyUI custom node, not a pip package.
    """
    # Find the custom_nodes directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up to custom_nodes: qwenimage -> ComfyUI-gen2 -> custom_nodes
    custom_nodes_dir = os.path.dirname(os.path.dirname(current_dir))
    
    # Path to videox-fun
    videox_path = os.path.join(custom_nodes_dir, "videox-fun")
    
    if os.path.exists(videox_path) and videox_path not in sys.path:
        sys.path.insert(0, videox_path)
        return True
    
    # Try alternative names
    for name in ["videox_fun", "VideoX-Fun", "ComfyUI-VideoX-Fun"]:
        alt_path = os.path.join(custom_nodes_dir, name)
        if os.path.exists(alt_path) and alt_path not in sys.path:
            sys.path.insert(0, alt_path)
            return True
    
    return False

# Setup videox imports on module load
_setup_videox_imports()

# Import diffusers components (same as VideoX)
try:
    from diffusers.models.attention import Attention, FeedForward
    from diffusers.models.normalization import RMSNorm
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.image_processor import VaeImageProcessor
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("[Gen2] Warning: diffusers not available. ControlNet nodes will not work.")

# =============================================================================
# Flash Attention Support (matching VideoX's attention_utils.py)
# =============================================================================

# Try to import Flash Attention 3
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

# Try to import Flash Attention 2
try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# Try to import Sage Attention (based on GPU capability)
try:
    major, minor = torch.cuda.get_device_capability(0)
    sm_version = f"{major}.{minor}"
    if sm_version == "8.0":
        from sageattention_sm80 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif sm_version == "8.6":
        from sageattention_sm86 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif sm_version == "8.9":
        from sageattention_sm89 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif sm_version == "9.0":
        from sageattention_sm90 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    elif major > 9:
        from sageattention_sm120 import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    else:
        SAGE_ATTENTION_AVAILABLE = False
        sageattn = None
except:
    try:
        from sageattention import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    except:
        sageattn = None
        SAGE_ATTENTION_AVAILABLE = False

# Log available attention backends
_attn_backends = []
if FLASH_ATTN_3_AVAILABLE:
    _attn_backends.append("FlashAttn3")
if FLASH_ATTN_2_AVAILABLE:
    _attn_backends.append("FlashAttn2")
if SAGE_ATTENTION_AVAILABLE:
    _attn_backends.append("SageAttn")
_attn_backends.append("SDPA")  # Always available in PyTorch 2.0+

print(f"[Gen2] Available attention backends: {', '.join(_attn_backends)}")

import numpy as np
import inspect
import warnings
from collections import defaultdict
from safetensors.torch import load_file as load_safetensors


# =============================================================================
# Dtype Detection (VideoX Style)
# =============================================================================

# Quantized dtypes that should NOT be used for compute operations
QUANTIZED_DTYPES = {
    torch.float8_e4m3fn,
    torch.float8_e5m2,
}

# Try to add additional quantized types if available
try:
    QUANTIZED_DTYPES.add(torch.int8)
    QUANTIZED_DTYPES.add(torch.uint8)
except AttributeError:
    pass


def get_compute_dtype(model_or_dtype, fallback_dtype=None):
    """
    Get appropriate compute dtype for a model or tensor dtype.
    
    VideoX-style logic:
    - If model dtype is quantized (fp8) or fp32, return bf16/fp16
    - Otherwise return the model's native dtype
    
    This ensures that even when using quantized models (fp8, GGUF),
    the control context, LoRA, and other compute operations use a 
    standard floating point dtype that works with all operations.
    
    Args:
        model_or_dtype: A model, tensor, or dtype to check
        fallback_dtype: Optional fallback if detection fails
        
    Returns:
        torch.dtype suitable for compute operations
    """
    # Extract dtype from various input types
    if isinstance(model_or_dtype, torch.dtype):
        dtype = model_or_dtype
    elif isinstance(model_or_dtype, torch.Tensor):
        dtype = model_or_dtype.dtype
    elif hasattr(model_or_dtype, 'parameters'):
        # It's a model - get dtype from first parameter
        try:
            dtype = next(model_or_dtype.parameters()).dtype
        except StopIteration:
            dtype = fallback_dtype or torch.bfloat16
    elif hasattr(model_or_dtype, 'dtype'):
        dtype = model_or_dtype.dtype
    else:
        dtype = fallback_dtype or torch.bfloat16
    
    # VideoX logic: exclude quantized types and float32
    # Reference: videox-fun/comfyui/qwenimage/nodes.py line 551
    # weight_dtype = transformer.dtype if transformer.dtype not in [
    #     torch.float32, torch.float8_e4m3fn, torch.float8_e5m2
    # ] else get_autocast_dtype()
    
    if dtype in QUANTIZED_DTYPES or dtype == torch.float32:
        # Use autocast dtype (bf16 if supported, else fp16)
        if torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except:
                pass
        return torch.float16
    
    return dtype


def get_autocast_dtype():
    """
    Get the best dtype for autocast (matches VideoX's get_autocast_dtype).
    
    Returns bf16 if supported, otherwise fp16.
    """
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except:
            pass
    return torch.float16


# =============================================================================
# Attention Function (matching VideoX's attention_utils.py exactly)
# =============================================================================

def gen2_flash_attention(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p=0., softmax_scale=None, q_scale=None,
    causal=False, window_size=(-1, -1), deterministic=False,
    dtype=torch.bfloat16, version=None,
):
    """
    Flash Attention implementation matching VideoX.
    
    q: [B, Lq, Nq, C1]
    k: [B, Lk, Nk, C1]
    v: [B, Lk, Nk, C2]
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # Preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn('Flash attention 3 is not available, using flash attention 2 instead.')

    # Apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None, seqused_k=None,
            max_seqlen_q=lq, max_seqlen_k=lk,
            softmax_scale=softmax_scale, causal=causal, deterministic=deterministic
        )[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE, "Flash Attention 2 or 3 required"
        x = flash_attn.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq, max_seqlen_k=lk,
            dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal,
            window_size=window_size, deterministic=deterministic
        ).unflatten(0, (b, lq))

    return x.type(out_dtype)


def gen2_attention(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p=0., softmax_scale=None, q_scale=None,
    causal=False, window_size=(-1, -1), deterministic=False,
    dtype=torch.bfloat16, fa_version=None, attention_type=None,
    attn_mask=None,
):
    """
    Unified attention function matching VideoX's attention_utils.attention exactly.
    
    Automatically selects the best available backend:
    1. SAGE_ATTENTION (if available and not training)
    2. FLASH_ATTENTION (Flash Attn 2/3)
    3. SDPA fallback
    
    Args:
        q: Query tensor [B, Lq, H, D] (BLHD format)
        k: Key tensor [B, Lk, H, D]
        v: Value tensor [B, Lk, H, D]
        attention_type: Override attention type (SAGE_ATTENTION, FLASH_ATTENTION, or SDPA)
    
    Returns:
        Attention output [B, Lq, H, D]
    """
    import os
    attention_type = os.environ.get("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION") if attention_type is None else attention_type
    
    # Sage attention doesn't work with gradients enabled
    if torch.is_grad_enabled() and attention_type == "SAGE_ATTENTION":
        attention_type = "FLASH_ATTENTION"
    
    # Auto-fallback chain: FLASH_ATTENTION -> SDPA
    if attention_type == "FLASH_ATTENTION" and not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        attention_type = "SDPA"

    if attention_type == "SAGE_ATTENTION" and SAGE_ATTENTION_AVAILABLE:
        if q_lens is not None or k_lens is not None:
            warnings.warn('Padding mask is disabled when using sage attention.')
        
        # Convert QKV dtypes if needed
        dtypes = {q.dtype, k.dtype, v.dtype}
        if torch.float16 in dtypes or torch.bfloat16 in dtypes:
            target_dtype = torch.bfloat16 if torch.bfloat16 in dtypes else torch.float16
        elif dtypes == {torch.float32}:
            target_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
        else:
            target_dtype = q.dtype
        
        q, k, v = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)
        out = sageattn(q, k, v, attn_mask=attn_mask, tensor_layout="NHD", is_causal=causal, dropout_p=dropout_p)
        return out

    if attention_type == "FLASH_ATTENTION" and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
        return gen2_flash_attention(
            q=q, k=k, v=v,
            q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p, softmax_scale=softmax_scale, q_scale=q_scale,
            causal=causal, window_size=window_size, deterministic=deterministic,
            dtype=dtype, version=fa_version,
        )
    
    # SDPA fallback (default)
    if q_lens is not None or k_lens is not None:
        warnings.warn('Padding mask is disabled when using scaled_dot_product_attention.')
    
    # SDPA expects [B, H, L, D] format, input is [B, L, H, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p
    )

    out = out.transpose(1, 2).contiguous()
    
    return out


# =============================================================================
# LoRA Merge/Unmerge Functions (Adapted from VideoX for ComfyUI model structure)
# =============================================================================

def gen2_merge_lora(transformer, lora_path, multiplier, device='cpu', dtype=torch.float32):
    """
    Merge LoRA weights into the transformer model (VideoX style).
    
    Adapted from videox_fun.utils.lora_utils.merge_lora to work with 
    ComfyUI's model structure instead of diffusers pipeline.
    
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
    
    LORA_PREFIX_TRANSFORMER = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    
    state_dict = load_safetensors(lora_path)
    updates = defaultdict(dict)
    
    for key, value in state_dict.items():
        # Handle diffusion_model prefix
        if "diffusion_model." in key:
            key = key.replace("diffusion_model.", "")
        if "lora_unet__" not in key:
            key = "lora_unet__" + key
        key = key.replace(".", "_")
        
        # Normalize LoRA key endings
        if key.endswith("_lora_up_weight"):
            key = key[:-15] + ".lora_up.weight"
        if key.endswith("_lora_down_weight"):
            key = key[:-17] + ".lora_down.weight"
        if key.endswith("_lora_A_default_weight"):
            key = key[:-22] + ".lora_A.weight"
        if key.endswith("_lora_B_default_weight"):
            key = key[:-22] + ".lora_B.weight"
        if key.endswith("_lora_A_weight"):
            key = key[:-14] + ".lora_A.weight"
        if key.endswith("_lora_B_weight"):
            key = key[:-14] + ".lora_B.weight"
        if key.endswith("_alpha"):
            key = key[:-6] + ".alpha"
        
        key = key.replace(".lora_A.default.", ".lora_down.")
        key = key.replace(".lora_B.default.", ".lora_up.")
        key = key.replace(".lora_A.", ".lora_down.")
        key = key.replace(".lora_B.", ".lora_up.")
        
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value
    
    merged_count = 0
    failed_layers = []
    
    # Debug: print first few LoRA layer keys to verify mapping
    sample_keys = list(updates.keys())[:3]
    print(f"[Gen2 LoRA] Sample layer keys: {sample_keys}")
    
    for layer, elems in updates.items():
        # Skip text encoder layers - we only handle transformer
        if "lora_te" in layer:
            continue
        
        layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
        curr_layer = transformer
        
        # Navigate to the target layer
        try:
            curr_layer = curr_layer.__getattr__("_".join(layer_infos[1:]))
        except Exception:
            temp_name = layer_infos.pop(0)
            try:
                while len(layer_infos) > -1:
                    try:
                        curr_layer = curr_layer.__getattr__(temp_name + "_" + "_".join(layer_infos))
                        break
                    except Exception:
                        try:
                            curr_layer = curr_layer.__getattr__(temp_name)
                            if len(layer_infos) > 0:
                                temp_name = layer_infos.pop(0)
                            elif len(layer_infos) == 0:
                                break
                        except Exception:
                            if len(layer_infos) == 0:
                                pass  # Will try back search
                            if len(temp_name) > 0:
                                temp_name += "_" + layer_infos.pop(0)
                            else:
                                temp_name = layer_infos.pop(0)
            except Exception:
                # Back search
                layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
                curr_layer = transformer
                
                len_layer_infos = len(layer_infos)
                start_index = 0 if len_layer_infos >= 1 and len(layer_infos[0]) > 0 else 1
                end_indx = len_layer_infos
                
                error_flag = False if len_layer_infos >= 1 else True
                while start_index < len_layer_infos:
                    try:
                        if start_index >= end_indx:
                            error_flag = True
                            break
                        curr_layer = curr_layer.__getattr__("_".join(layer_infos[start_index:end_indx]))
                        start_index = end_indx
                        end_indx = len_layer_infos
                    except Exception:
                        end_indx -= 1
                if error_flag:
                    continue
        
        # Apply LoRA to the layer
        try:
            origin_dtype = curr_layer.weight.data.dtype
            origin_device = curr_layer.weight.data.device
            
            curr_layer = curr_layer.to(device, dtype)
            weight_up = elems['lora_up.weight'].to(device, dtype)
            weight_down = elems['lora_down.weight'].to(device, dtype)
            
            if 'alpha' in elems.keys():
                alpha = elems['alpha'].item() / weight_up.shape[1]
            else:
                alpha = 1.0
            
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += multiplier * alpha * torch.mm(
                    weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3)
            else:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)
            
            curr_layer = curr_layer.to(origin_device, origin_dtype)
            merged_count += 1
            
            # Debug: print first few successful merges
            if merged_count <= 3:
                print(f"[Gen2 LoRA] Merged: {layer} -> {type(curr_layer).__name__}")
        except Exception as e:
            failed_layers.append(layer)
            continue
    
    if failed_layers:
        print(f"[Gen2 LoRA] Warning: {len(failed_layers)} layers failed to merge")
        if len(failed_layers) <= 5:
            for fl in failed_layers:
                print(f"  Failed: {fl}")
    
    print(f"[Gen2 LoRA] Merged {merged_count} LoRA layers with strength {multiplier}")
    return transformer


def gen2_unmerge_lora(transformer, lora_path, multiplier, device='cpu', dtype=torch.float32):
    """
    Unmerge LoRA weights from the transformer model (VideoX style).
    
    This reverses the merge operation by subtracting the LoRA weights.
    
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
    
    LORA_PREFIX_TRANSFORMER = "lora_unet"
    
    state_dict = load_safetensors(lora_path)
    updates = defaultdict(dict)
    
    for key, value in state_dict.items():
        if "diffusion_model." in key:
            key = key.replace("diffusion_model.", "")
        if "lora_unet__" not in key:
            key = "lora_unet__" + key
        key = key.replace(".", "_")
        
        if key.endswith("_lora_up_weight"):
            key = key[:-15] + ".lora_up.weight"
        if key.endswith("_lora_down_weight"):
            key = key[:-17] + ".lora_down.weight"
        if key.endswith("_lora_A_default_weight"):
            key = key[:-22] + ".lora_A.weight"
        if key.endswith("_lora_B_default_weight"):
            key = key[:-22] + ".lora_B.weight"
        if key.endswith("_lora_A_weight"):
            key = key[:-14] + ".lora_A.weight"
        if key.endswith("_lora_B_weight"):
            key = key[:-14] + ".lora_B.weight"
        if key.endswith("_alpha"):
            key = key[:-6] + ".alpha"
        
        key = key.replace(".lora_A.default.", ".lora_down.")
        key = key.replace(".lora_B.default.", ".lora_up.")
        key = key.replace(".lora_A.", ".lora_down.")
        key = key.replace(".lora_B.", ".lora_up.")
        
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value
    
    unmerged_count = 0
    for layer, elems in updates.items():
        if "lora_te" in layer:
            continue
        
        layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
        curr_layer = transformer
        
        try:
            curr_layer = curr_layer.__getattr__("_".join(layer_infos[1:]))
        except Exception:
            temp_name = layer_infos.pop(0)
            try:
                while len(layer_infos) > -1:
                    try:
                        curr_layer = curr_layer.__getattr__(temp_name + "_" + "_".join(layer_infos))
                        break
                    except Exception:
                        try:
                            curr_layer = curr_layer.__getattr__(temp_name)
                            if len(layer_infos) > 0:
                                temp_name = layer_infos.pop(0)
                            elif len(layer_infos) == 0:
                                break
                        except Exception:
                            if len(layer_infos) == 0:
                                pass
                            if len(temp_name) > 0:
                                temp_name += "_" + layer_infos.pop(0)
                            else:
                                temp_name = layer_infos.pop(0)
            except Exception:
                layer_infos = layer.split(LORA_PREFIX_TRANSFORMER + "_")[-1].split("_")
                curr_layer = transformer
                
                len_layer_infos = len(layer_infos)
                start_index = 0 if len_layer_infos >= 1 and len(layer_infos[0]) > 0 else 1
                end_indx = len_layer_infos
                
                error_flag = False if len_layer_infos >= 1 else True
                while start_index < len_layer_infos:
                    try:
                        if start_index >= end_indx:
                            error_flag = True
                            break
                        curr_layer = curr_layer.__getattr__("_".join(layer_infos[start_index:end_indx]))
                        start_index = end_indx
                        end_indx = len_layer_infos
                    except Exception:
                        end_indx -= 1
                if error_flag:
                    continue
        
        try:
            origin_dtype = curr_layer.weight.data.dtype
            origin_device = curr_layer.weight.data.device
            
            curr_layer = curr_layer.to(device, dtype)
            weight_up = elems['lora_up.weight'].to(device, dtype)
            weight_down = elems['lora_down.weight'].to(device, dtype)
            
            if 'alpha' in elems.keys():
                alpha = elems['alpha'].item() / weight_up.shape[1]
            else:
                alpha = 1.0
            
            # SUBTRACT instead of add to unmerge
            if len(weight_up.shape) == 4:
                curr_layer.weight.data -= multiplier * alpha * torch.mm(
                    weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)
                ).unsqueeze(2).unsqueeze(3)
            else:
                curr_layer.weight.data -= multiplier * alpha * torch.mm(weight_up, weight_down)
            
            curr_layer = curr_layer.to(origin_device, origin_dtype)
            unmerged_count += 1
        except Exception as e:
            print(f"[Gen2 LoRA] Failed to unmerge layer {layer}: {e}")
            continue
    
    print(f"[Gen2 LoRA] Unmerged {unmerged_count} LoRA layers")
    return transformer


# =============================================================================
# Config Class for VideoX Pipeline Compatibility
# =============================================================================

class Gen2TransformerConfig:
    """
    Simple config object that mimics VideoX's transformer.config structure.
    Used for compatibility with VideoX pipeline which accesses config attributes.
    """
    def __init__(self, in_channels=64, out_channels=16, guidance_embeds=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.guidance_embeds = guidance_embeds
    
    def get(self, key, default=None):
        return getattr(self, key, default)


# =============================================================================
# Helper Functions for ComfyUI CONDITIONING
# =============================================================================

def extract_from_conditioning(conditioning):
    """
    Extract prompt_embeds and attention_mask from ComfyUI CONDITIONING.
    
    ComfyUI CONDITIONING format:
    [
        (
            encoder_hidden_states,  # [batch, seq_len, hidden_dim]
            {
                "pooled_output": ...,     # optional
                "attention_mask": ...,    # [batch, seq_len] - optional
                ...
            }
        )
    ]
    
    Returns:
        prompt_embeds: Tensor [batch, seq_len, hidden_dim]
        prompt_embeds_mask: Tensor [batch, seq_len] or None
    """
    if conditioning is None or len(conditioning) == 0:
        return None, None
    
    # Get the first conditioning entry
    cond_entry = conditioning[0]
    
    # Extract encoder_hidden_states (prompt_embeds)
    prompt_embeds = cond_entry[0]
    
    # Extract attention_mask from the dict (if available)
    cond_dict = cond_entry[1] if len(cond_entry) > 1 else {}
    prompt_embeds_mask = cond_dict.get("attention_mask", None)
    
    # If no mask provided, create one with all ones (all tokens valid)
    if prompt_embeds_mask is None:
        batch_size, seq_len = prompt_embeds.shape[:2]
        prompt_embeds_mask = torch.ones(batch_size, seq_len, device=prompt_embeds.device, dtype=torch.long)
    
    return prompt_embeds, prompt_embeds_mask


def get_txt_seq_len_from_mask(attention_mask):
    """
    Get actual text sequence length from attention mask.
    
    The attention mask has 1s for valid tokens and 0s for padding.
    The actual sequence length is the sum of 1s.
    
    Args:
        attention_mask: Tensor [batch, seq_len] with 1s and 0s
    
    Returns:
        List of sequence lengths for each batch item
    """
    if attention_mask is None:
        return None
    
    # Sum along sequence dimension to get valid token count per batch
    seq_lens = attention_mask.sum(dim=-1).tolist()
    
    # Handle single batch case
    if isinstance(seq_lens, (int, float)):
        seq_lens = [int(seq_lens)]
    else:
        seq_lens = [int(x) for x in seq_lens]
    
    return seq_lens


# =============================================================================
# RoPE Implementation (from VideoX)
# =============================================================================

def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    use_real: bool = False,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors.
    Matches VideoX's apply_rotary_emb_qwen function.
    
    Args:
        x: Query or key tensor [B, S, H, D]
        freqs_cis: Precomputed frequency tensor (complex exponentials)
        use_real: Whether freqs are in real format (cos, sin) or complex
    
    Returns:
        Tensor with rotary embeddings applied
    """
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)
        
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        # Complex multiplication approach (what VideoX uses by default)
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


# =============================================================================
# RoPE Embedder (from VideoX - QwenEmbedRope)
# =============================================================================

class QwenEmbedRope(nn.Module):
    """
    VideoX's QwenEmbedRope implementation for generating proper RoPE frequencies.
    Generates separate frequencies for image and text sequences.
    
    Note: DO NOT use register_buffer for complex tensors - it loses the imaginary part!
    """
    
    def __init__(self, theta: int = 10000, axes_dim: List[int] = [16, 56, 56], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope
        
        # Pre-compute frequency bases
        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        
        self.pos_freqs = torch.cat([
            self._rope_params(pos_index, axes_dim[0], theta),
            self._rope_params(pos_index, axes_dim[1], theta),
            self._rope_params(pos_index, axes_dim[2], theta),
        ], dim=1)
        
        self.neg_freqs = torch.cat([
            self._rope_params(neg_index, axes_dim[0], theta),
            self._rope_params(neg_index, axes_dim[1], theta),
            self._rope_params(neg_index, axes_dim[2], theta),
        ], dim=1)
    
    def _rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        """Compute rope parameters for given indices and dimension."""
        assert dim % 2 == 0
        freqs = torch.outer(index.float(), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float() / dim))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs
    
    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        """Compute video/image frequencies for given dimensions."""
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        
        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2):], freqs_pos[1][:height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2):], freqs_pos[2][:width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
        
        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()
    
    def forward(self, video_fhw: List, txt_seq_lens: List[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate RoPE frequencies for image and text.
        MATCHES VideoX's QwenEmbedRope.forward() EXACTLY.
        
        Args:
            video_fhw: Video shape - can be:
                - Single tuple: (frame, height, width)
                - List of tuples: [(f, h, w), ...]  for multi-clip
                - Nested list: [[(f, h, w)], [(f, h, w)]] from pipeline (first batch is extracted)
            txt_seq_lens: List of actual text sequence lengths per batch item (from mask.sum())
            device: Target device
        
        Returns:
            (vid_freqs, txt_freqs) tuple for VideoX-style attention
        """
        # Ensure buffers are on correct device
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)
        
        # === VideoX exact logic ===
        # Extract first batch item if nested list (all batches have same image shape)
        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        # Ensure it's a list of tuples
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]
        
        vid_freqs = []
        max_vid_index = 0
        
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)
            
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)
        
        # Compute text frequencies using MAX of txt_seq_lens
        # This is critical for CFG batching where neg/pos have different lengths
        max_len = max(txt_seq_lens) if txt_seq_lens else 0
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)
        
        return vid_freqs, txt_freqs


# =============================================================================
# Attention Processor (from VideoX)
# =============================================================================

class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture.
    Matches VideoX's QwenDoubleStreamAttnProcessor2_0 exactly.
    """
    
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states")
        
        seq_txt = encoder_hidden_states.shape[1]
        
        # Compute QKV for image stream
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        
        # Compute QKV for text stream
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)
        
        # Reshape for multi-head attention [B, S, H, D]
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))
        
        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)
        
        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
        
        # Concatenate for joint attention [txt, img]
        # Shape: [B, S, H, D] (BLHD format, matches VideoX's attention function)
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        
        # Build attention mask (mask text padding only, image tokens always valid)
        attn_mask = None
        if encoder_hidden_states_mask is not None:
            # encoder_hidden_states_mask: [B, seq_txt] with 1 for valid, 0 for pad
            pad_mask = encoder_hidden_states_mask == 0
            if pad_mask.any():
                batch_size = pad_mask.shape[0]
                seq_total = joint_key.shape[1]
                seq_txt = encoder_hidden_states_mask.shape[1]
                attn_mask = torch.zeros(
                    (batch_size, 1, 1, seq_total),
                    device=joint_key.device,
                    dtype=joint_key.dtype,
                )
                attn_mask[:, :, :, :seq_txt] = torch.where(
                    pad_mask.unsqueeze(1).unsqueeze(1),
                    -torch.finfo(joint_key.dtype).max,
                    0.0,
                )
        
        # Use VideoX-compatible attention (Flash Attention / Sage Attention / SDPA fallback)
        joint_hidden_states = gen2_attention(
            joint_query, joint_key, joint_value,
            attn_mask=attn_mask, dropout_p=0.0, causal=False
        )
        
        # Flatten heads: [B, S, H, D] -> [B, S, H*D]
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)
        
        # Split attention outputs
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]
        
        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout
        
        txt_attn_output = attn.to_add_out(txt_attn_output)
        
        return img_attn_output, txt_attn_output


# =============================================================================
# Transformer Block (matches VideoX QwenImageTransformerBlock)
# =============================================================================

class QwenImageTransformerBlock(nn.Module):
    """
    Standalone implementation matching VideoX's QwenImageTransformerBlock exactly.
    Uses diffusers' Attention and FeedForward classes.
    """
    
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.zero_cond_t = zero_cond_t
        
        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        
        # Attention using diffusers' Attention class
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        
        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
    
    def _modulate(self, x, mod_params, index=None):
        """Apply modulation to input tensor."""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        
        if index is not None:
            # Handle zero_cond_t case with conditional modulation
            actual_batch = shift.size(0) // 2
            shift_0, shift_1 = shift[:actual_batch], shift[actual_batch:]
            scale_0, scale_1 = scale[:actual_batch], scale[actual_batch:]
            gate_0, gate_1 = gate[:actual_batch], gate[actual_batch:]
            
            index_expanded = index.unsqueeze(-1)
            
            shift_0_exp = shift_0.unsqueeze(1)
            shift_1_exp = shift_1.unsqueeze(1)
            scale_0_exp = scale_0.unsqueeze(1)
            scale_1_exp = scale_1.unsqueeze(1)
            gate_0_exp = gate_0.unsqueeze(1)
            gate_1_exp = gate_1.unsqueeze(1)
            
            shift_result = torch.where(index_expanded == 0, shift_0_exp, shift_1_exp)
            scale_result = torch.where(index_expanded == 0, scale_0_exp, scale_1_exp)
            gate_result = torch.where(index_expanded == 0, gate_0_exp, gate_1_exp)
        else:
            shift_result = shift.unsqueeze(1)
            scale_result = scale.unsqueeze(1)
            gate_result = gate.unsqueeze(1)
        
        return x * (1 + scale_result) + shift_result, gate_result
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        modulate_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass matching VideoX exactly.
        
        Returns:
            (encoder_hidden_states, hidden_states) - text and image outputs
        """
        # Get modulation parameters
        img_mod_params = self.img_mod(temb)
        
        if self.zero_cond_t:
            temb = torch.chunk(temb, 2, dim=0)[0]
        txt_mod_params = self.txt_mod(temb)
        
        # Split modulation parameters
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
        
        # Image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, modulate_index)
        
        # Text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)
        
        # Joint attention
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        
        img_attn_output, txt_attn_output = attn_output
        
        # Apply gates and residual
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
        
        # Image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, modulate_index)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output
        
        # Text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output
        
        # Clip for numerical stability (VideoX does this ONLY for fp16, NOT bfloat16)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clamp(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clamp(-65504, 65504)
        
        return encoder_hidden_states, hidden_states


# =============================================================================
# Control Transformer Block (matches VideoX QwenImageControlTransformerBlock)
# =============================================================================

class QwenImageControlTransformerBlock(QwenImageTransformerBlock):
    """
    Control block matching VideoX's QwenImageControlTransformerBlock exactly.
    
    Key differences from base block:
    - block_id 0 has before_proj (zero-initialized)
    - all blocks have after_proj (zero-initialized)
    - forward uses stacking logic for hint accumulation
    """
    
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 1e-6,
        zero_cond_t: bool = False,
        block_id: int = 0,
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, qk_norm, eps, zero_cond_t)
        
        self.block_id = block_id
        
        # Block 0 has before_proj to combine control context with hidden_states
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        
        # All blocks have after_proj for hint output
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)
    
    def forward(self, c: torch.Tensor, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with VideoX's exact stacking logic.
        
        Args:
            c: Control tensor. For block 0: projected control context [B, S, D]
               For block 1+: stacked tensor from previous blocks [N, B, S, D]
            x: Hidden states from main model [B, S, D]
            **kwargs: Arguments for parent forward (encoder_hidden_states, temb, etc.)
        
        Returns:
            (encoder_hidden_states, c) where c is stacked [hints..., current_state]
        """
        if self.block_id == 0:
            # First block: combine control context with hidden states
            c = self.before_proj(c) + x
            all_c = []
        else:
            # Subsequent blocks: unpack stacked tensor, take last as input
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        
        # Run full transformer block
        encoder_hidden_states, c = super().forward(c, **kwargs)
        
        # Project for hint output
        c_skip = self.after_proj(c)
        
        # Stack hints and current state
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        
        return encoder_hidden_states, c


# =============================================================================
# Control Model (container for control blocks)
# =============================================================================

class QwenImageControlModel(nn.Module):
    """
    Complete control model matching VideoX's control architecture.
    
    Contains:
    - control_img_in: Projects 132-dim control context to inner_dim
    - control_blocks: List of QwenImageControlTransformerBlock
    - pos_embed: RoPE embedder for generating proper positional frequencies
    """
    
    def __init__(
        self,
        control_layers: List[int] = [0, 12, 24, 36, 48],
        control_in_dim: int = 132,
        inner_dim: int = 3072,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        axes_dims_rope: List[int] = [16, 56, 56],
    ):
        super().__init__()
        
        self.control_layers = control_layers
        self.control_in_dim = control_in_dim
        self.inner_dim = inner_dim
        
        # Project control context to inner dimension
        self.control_img_in = nn.Linear(control_in_dim, inner_dim)
        
        # RoPE embedder for generating VideoX-compatible positional frequencies
        # CRITICAL: Must use scale_rope=True to match VideoX's training!
        # scale_rope=True creates symmetric position encoding centered at image center
        # scale_rope=False creates asymmetric encoding from corner - causes grid artifacts
        self.pos_embed = QwenEmbedRope(
            theta=10000,
            axes_dim=axes_dims_rope,
            scale_rope=True,
        )
        
        # Control blocks (one per control layer)
        self.control_blocks = nn.ModuleList([
            QwenImageControlTransformerBlock(
                dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                block_id=i,
            )
            for i in range(len(control_layers))
        ])
        
        # Mapping from layer index to hint index
        self.layer_to_hint_idx = {layer: idx for idx, layer in enumerate(control_layers)}
    
    def forward_control(
        self,
        x: torch.Tensor,
        control_context: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        img_shape: Tuple[int, int, int],  # (frame, height, width) - packed dimensions
        txt_seq_lens: List[int],  # List of actual text sequence lengths per batch item
    ) -> List[torch.Tensor]:
        """
        Generate control hints matching VideoX's forward_control.
        
        Args:
            x: Hidden states from main model [B, S, D]
            control_context: Packed control context [B, S, 132]
            encoder_hidden_states: Text embeddings [B, txt_len, D]
            encoder_hidden_states_mask: Text attention mask
            temb: Timestep embedding [B, D]
            img_shape: (frame, height, width) tuple for RoPE generation
            txt_seq_lens: List of actual text sequence lengths per batch item (from mask.sum())
        
        Returns:
            List of hint tensors, one per control layer
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Generate our own VideoX-compatible RoPE frequencies
        # Format: [[(f, h, w)], [(f, h, w)]] for batch - QwenEmbedRope extracts first
        img_shapes = [[img_shape]] * batch_size
        # txt_seq_lens is already a list of actual token counts
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device)
        
        # Project control context
        c = self.control_img_in(control_context)
        
        # Build kwargs for block forward
        kwargs = dict(
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )
        
        # Run control blocks sequentially
        for block in self.control_blocks:
            encoder_hidden_states, c = block(c, x, **kwargs)
            # Update encoder_hidden_states for next block (chain update)
            kwargs["encoder_hidden_states"] = encoder_hidden_states
        
        # Extract hints (all but last element, which is final state)
        hints = list(torch.unbind(c))[:-1]
        
        return hints


# =============================================================================
# Latent Utilities
# =============================================================================


# =============================================================================
# ComfyUI Nodes
# =============================================================================

class Gen2_LoadQwenControlNetFun:
    """
    Loads VideoX Fun's QwenImage ControlNet weights into our standalone model.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Support both controlnet and model_patches folders
        controlnet_files = []
        
        # Get files from controlnet folder
        try:
            controlnet_files.extend(folder_paths.get_filename_list("controlnet"))
        except:
            pass
        
        # Get files from model_patches folder (where VideoX Fun ControlNet is often placed)
        try:
            model_patches = folder_paths.get_filename_list("model_patches")
            # Filter for qwen/controlnet files
            for f in model_patches:
                if "controlnet" in f.lower() or "qwen" in f.lower():
                    if f not in controlnet_files:
                        controlnet_files.append(f)
        except:
            pass
        
        if not controlnet_files:
            controlnet_files = ["No ControlNet files found"]
        
        return {
            "required": {
                "controlnet_name": (controlnet_files, ),
            }
        }
    
    RETURN_TYPES = ("GEN2_CONTROLNET",)
    RETURN_NAMES = ("controlnet",)
    FUNCTION = "load"
    CATEGORY = "Gen2/QwenImage/ControlNet"
    
    def load(self, controlnet_name):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers is required for QwenImage ControlNet")
        
        # Try multiple folders to find the file
        controlnet_path = None
        for folder_type in ["controlnet", "model_patches"]:
            try:
                path = folder_paths.get_full_path(folder_type, controlnet_name)
                if path and os.path.exists(path):
                    controlnet_path = path
                    break
            except:
                pass
        
        if controlnet_path is None:
            raise FileNotFoundError(f"ControlNet file not found: {controlnet_name}")
        
        print(f"[Gen2] Loading QwenImage ControlNet Fun: {controlnet_name}")
        
        # Load state dict
        state_dict = comfy.utils.load_torch_file(controlnet_path)
        
        # VideoX control config
        control_layers = [0, 12, 24, 36, 48]
        control_in_dim = 132
        inner_dim = 3072
        num_attention_heads = 24
        attention_head_dim = 128
        
        # Create control model
        control_model = QwenImageControlModel(
            control_layers=control_layers,
            control_in_dim=control_in_dim,
            inner_dim=inner_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
        )
        
        # Load weights - need to map from VideoX's key format to ours
        # VideoX uses: control_blocks.0.img_mod.1.weight, control_img_in.weight, etc.
        # Our model uses the same structure, so direct loading should work
        
        # Filter control-specific weights
        control_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('control_'):
                control_state_dict[key] = value
        
        # Load into model
        missing, unexpected = control_model.load_state_dict(control_state_dict, strict=False)
        
        print(f"[Gen2] ControlNet loaded: {len(control_layers)} blocks at layers {control_layers}")
        if missing:
            print(f"[Gen2] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[Gen2] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        
        return ({
            'model': control_model,
            'control_layers': control_layers,
            'control_in_dim': control_in_dim,
        },)


class Gen2_ApplyQwenControlNetFun:
    """
    Applies QwenImage ControlNet to the model.
    
    Uses GEN2_VAE for proper VideoX-compatible encoding.
    Outputs GEN2_WRAPPED_MODEL for use with Gen2_QwenImageControlSampler.
    
    The wrapped model includes:
    - transformer wrapper (Gen2QwenImageModelWrapper)
    - control model
    - control context (pre-encoded)
    - VAE (for decoding in sampler)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", ),
                "controlnet": ("GEN2_CONTROLNET", ),
                "vae": ("GEN2_VAE", ),
                "control_image": ("IMAGE", ),
                "control_context_scale": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "inpaint_image": ("IMAGE", ),
                "mask": ("MASK", ),
            }
        }
    
    RETURN_TYPES = ("GEN2_WRAPPED_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Gen2/QwenImage"
    
    def prepare_control_context(self, vae, control_image, inpaint_image, mask, device, dtype, height, width):
        """
        Prepare the 132-feature control context matching VideoX's QwenImageControlPipeline exactly.
        
        Now uses GEN2_VAE for proper VideoX-compatible encoding.
        
        VideoX's control_context = [control_latents(16), mask(1), inpaint_latent(16)] = 33 channels
        After 2x2 packing: 33 * 4 = 132 features per sequence position
        """
        # Extract VAE model and config
        vae_model = vae['model']
        vae_config = vae['config']
        vae_dtype = vae['dtype']
        vae_offload = vae['device']
        
        # Move VAE to compute device
        vae_model = vae_model.to(device)
        
        # VAE scale factor for QwenImage (8x spatial compression)
        vae_scale_factor = 8
        
        # Get image dimensions
        batch_size = control_image.shape[0]
        _, img_h, img_w, _ = control_image.shape
        
        # Compute latent dimensions (must be divisible by vae_scale_factor * 2 for packing)
        latent_height = 2 * (int(height) // (vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (vae_scale_factor * 2))
        num_channels_latents = 16  # QwenImage uses 16 latent channels
        
        # Create latent normalization tensors from VAE config (VideoX style)
        latents_mean = torch.tensor(vae_config['latents_mean']).view(1, num_channels_latents, 1, 1, 1).to(device)
        latents_std = 1.0 / torch.tensor(vae_config['latents_std']).view(1, num_channels_latents, 1, 1, 1).to(device)
        
        # VideoX-style processors (match resize/normalize/binarize behavior)
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
        mask_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        
        # --- Process mask (VideoX: default is ones, meaning "generate everywhere") ---
        if mask is not None:
            mask_condition = mask_processor.preprocess(mask, height=height, width=width)
            mask_condition = torch.where(mask_condition >= 0.5, torch.ones_like(mask_condition), torch.zeros_like(mask_condition))
            mask_condition = torch.tile(mask_condition, [1, 3, 1, 1]).to(dtype=dtype, device=device)
        else:
            mask_condition = torch.ones(batch_size, 3, height, width, dtype=dtype, device=device)
        
        def _to_bchw(image_tensor):
            if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim == 4:
                # ComfyUI IMAGE is BHWC; diffusers expects BCHW or BHWC
                if image_tensor.shape[-1] in (1, 3, 4):
                    return image_tensor.permute(0, 3, 1, 2)
            return image_tensor
        
        # --- Process inpaint image (VideoX: default is zeros) ---
        if inpaint_image is not None:
            inpaint_image_bchw = _to_bchw(inpaint_image)
            init_image = image_processor.preprocess(inpaint_image_bchw, height=height, width=width)
            init_image = init_image.to(dtype=dtype, device=device) * (mask_condition < 0.5)
            init_image = init_image.unsqueeze(2)
            
            with torch.no_grad():
                inpaint_latent = vae_model.encode(init_image)[0].mode()
            inpaint_latent = ((inpaint_latent - latents_mean) * latents_std).to(dtype=dtype)
        else:
            inpaint_latent = torch.zeros(
                batch_size, num_channels_latents, 1, latent_height, latent_width,
                dtype=dtype, device=device
            )
        
        # --- Process control image ---
        if control_image is not None:
            control_image_bchw = _to_bchw(control_image)
            control_image = image_processor.preprocess(control_image_bchw, height=height, width=width)
            control_image = control_image.to(dtype=dtype, device=device)
            control_image = control_image.unsqueeze(2)
            
            with torch.no_grad():
                control_latents = vae_model.encode(control_image)[0].mode()
            control_latents = ((control_latents - latents_mean) * latents_std).to(dtype=dtype)
        else:
            control_latents = torch.zeros_like(inpaint_latent)
        
        # --- Prepare mask for latent space ---
        mask_latent = F.interpolate(
            1 - mask_condition[:, :1],
            size=inpaint_latent.size()[-2:],
            mode='nearest'
        ).to(dtype=dtype, device=device)
        mask_latent = mask_latent.unsqueeze(2)
        
        # --- Concatenate control context ---
        # VideoX order: [control_latents(16), mask(1), inpaint_latent(16)] = 33 channels
        control_context = torch.cat([control_latents, mask_latent, inpaint_latent], dim=1)
        
        # Get dimensions for packing
        ctrl_batch, ctrl_channels, ctrl_frames, ctrl_h, ctrl_w = control_context.shape
        
        # Pack to sequence format using VideoX's _pack_latents with num_frame
        control_context = self._pack_latents_5d(
            control_context, ctrl_batch, ctrl_channels, ctrl_h, ctrl_w, num_frame=ctrl_frames
        )
        
        # Move VAE back to offload device
        vae_model = vae_model.to(vae_offload)
        
        print(f"[Gen2] Control context (VideoX style): image={height}x{width}, "
              f"latent={latent_height}x{latent_width}, packed_seq={control_context.shape[1]}, "
              f"features={control_context.shape[2]}")
        
        return control_context, latent_height, latent_width
    
    @staticmethod
    def _pack_latents_5d(latents, batch_size, num_channels_latents, height, width, num_frame=None):
        """
        Pack latents from (B, C, T, H, W) to (B, seq_len, C*4) format.
        Matches VideoX's _pack_latents exactly.
        """
        if num_frame is None:
            latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        else:
            latents = latents.view(batch_size, num_channels_latents, num_frame, height // 2, 2, width // 2, 2)
            latents = latents.permute(0, 2, 3, 5, 1, 4, 6)
            latents = latents.reshape(batch_size, num_frame * (height // 2) * (width // 2), num_channels_latents * 4)
        return latents
    
    def apply(self, model, controlnet, vae, control_image, control_context_scale,
              inpaint_image=None, mask=None):
        
        device = mm.get_torch_device()
        
        # Get the underlying diffusion model from ComfyUI's ModelPatcher
        comfyui_diffusion_model = model.model.diffusion_model
        
        # Get model storage dtype (may be quantized: fp8, int8, etc.)
        model_storage_dtype = next(comfyui_diffusion_model.parameters()).dtype
        
        # Get compute dtype using VideoX-style detection
        # This ensures we use bf16/fp16 even when model is quantized (fp8, GGUF)
        # Reference: videox-fun/comfyui/qwenimage/nodes.py
        vae_storage_dtype = vae.get('dtype', model_storage_dtype)
        compute_dtype = get_compute_dtype(vae_storage_dtype, fallback_dtype=torch.bfloat16)
        
        # For control context encoding, use compute dtype
        dtype = compute_dtype
        
        print(f"[Gen2] Model storage dtype: {model_storage_dtype}")
        print(f"[Gen2] VAE storage dtype: {vae_storage_dtype}")
        print(f"[Gen2] Compute dtype: {compute_dtype}")
        
        # Get image dimensions from control_image
        _, img_h, img_w, _ = control_image.shape
        
        # Round to divisible by 16 (vae_scale_factor * 2) for proper packing
        height = (img_h // 16) * 16
        width = (img_w // 16) * 16
        
        # Prepare control context (VideoX style)
        print(f"[Gen2] Preparing control context (VideoX style)...")
        control_context, lh, lw = self.prepare_control_context(
            vae, control_image, inpaint_image, mask, device, dtype, height, width
        )
        
        # Create VideoX-compatible wrapper
        wrapped_transformer = Gen2QwenImageModelWrapper(
            comfyui_model=comfyui_diffusion_model,
            control_model=controlnet['model'],
            latent_height=lh,
            latent_width=lw,
            control_layers=controlnet['control_layers'],
        )
        
        # Package wrapped model with all necessary components
        wrapped_model = {
            'wrapped_model': wrapped_transformer,
            'control_model': controlnet['model'],
            'control_context': control_context,
            'control_context_scale': control_context_scale,
            'control_layers': controlnet['control_layers'],
            'latent_height': lh,
            'latent_width': lw,
            'image_height': height,
            'image_width': width,
            # Separate storage and compute dtypes for flexibility
            'model_storage_dtype': model_storage_dtype,  # Actual model weight dtype (may be fp8)
            'compute_dtype': compute_dtype,  # Dtype for compute operations (always bf16/fp16)
            'dtype': dtype,  # Legacy: same as compute_dtype for compatibility
            # Include VAE for sampler to use for decoding
            'vae': vae,
        }
        
        print(f"[Gen2] ControlNet prepared: scale={control_context_scale}, layers={controlnet['control_layers']}")
        print(f"[Gen2] Image size: {width}x{height}, latent size: {lw}x{lh}")
        
        return (wrapped_model,)


# =============================================================================
# VideoX FlowMatch Scheduler Utilities
# =============================================================================

def filter_kwargs(cls, kwargs):
    """Filter kwargs to only include parameters accepted by cls.__init__"""
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def get_qwen_scheduler(sampler_name: str, shift: float):
    """
    Create a FlowMatch scheduler matching VideoX's get_qwen_scheduler.
    
    Args:
        sampler_name: One of "Flow", "Flow_Unipc", "Flow_DPM++"
        shift: Shift parameter for the scheduler
    
    Returns:
        Configured scheduler instance
    """
    # Try to import VideoX's custom schedulers, fall back to standard FlowMatch
    try:
        from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
        from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        HAS_VIDEOX_SCHEDULERS = True
    except ImportError:
        HAS_VIDEOX_SCHEDULERS = False
    
    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler if HAS_VIDEOX_SCHEDULERS else FlowMatchEulerDiscreteScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler if HAS_VIDEOX_SCHEDULERS else FlowMatchEulerDiscreteScheduler,
    }[sampler_name]
    
    # Match VideoX's scheduler defaults (see videox-fun comfyui/qwenimage/nodes.py)
    scheduler_kwargs = {
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": 0.9,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": 0.02,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler_kwargs["shift"] = shift
    scheduler = Chosen_Scheduler(**filter_kwargs(Chosen_Scheduler, scheduler_kwargs))
    return scheduler


# =============================================================================
# Gen2 Transformer Block Wrapper - VideoX forward logic with ComfyUI weights
# =============================================================================

class Gen2QwenImageTransformerBlockWrapper:
    """
    Wrapper that uses ComfyUI's transformer block WEIGHTS but applies VideoX's forward LOGIC.
    
    This gives us:
    1. ComfyUI's fast model loading (weights are shared, not copied)
    2. VideoX's exact RoPE calculation and attention processing
    3. Full control over the forward pass
    
    The key insight is that ComfyUI and VideoX have identical weight structures,
    just different forward implementations. We use the weights from ComfyUI's loaded
    model but execute the forward pass exactly as VideoX does.
    """
    
    def __init__(self, comfyui_block):
        """
        Args:
            comfyui_block: A ComfyUI QwenImageTransformerBlock with loaded weights
        """
        # Reference ComfyUI block's weight tensors directly (shared memory, no copy)
        # Block-level modules
        self.img_mod = comfyui_block.img_mod
        self.img_norm1 = comfyui_block.img_norm1
        self.img_norm2 = comfyui_block.img_norm2
        self.img_mlp = comfyui_block.img_mlp
        
        self.txt_mod = comfyui_block.txt_mod
        self.txt_norm1 = comfyui_block.txt_norm1
        self.txt_norm2 = comfyui_block.txt_norm2
        self.txt_mlp = comfyui_block.txt_mlp
        
        # Attention module - we'll use its weights but our own attention logic
        self.attn = comfyui_block.attn
        
        # Store dimension info
        self.dim = comfyui_block.dim
        self.num_attention_heads = comfyui_block.num_attention_heads
        self.attention_head_dim = comfyui_block.attention_head_dim
    
    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply modulation to input tensor (VideoX style).
        
        Args:
            x: Input tensor [B, S, D]
            mod_params: Modulation parameters [B, 3*D] (shift, scale, gate)
        
        Returns:
            (modulated_x, gate) tuple
        """
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        # x: [B, S, D], shift/scale/gate: [B, D] -> unsqueeze to [B, 1, D]
        modulated = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return modulated, gate.unsqueeze(1)
    
    def _attention_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor],
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        VideoX-style attention forward using ComfyUI's attention weights.
        
        Args:
            hidden_states: Image features [B, img_seq, D]
            encoder_hidden_states: Text features [B, txt_seq, D]
            encoder_hidden_states_mask: Text attention mask
            image_rotary_emb: VideoX-style RoPE tuple (vid_freqs, txt_freqs)
        
        Returns:
            (img_attn_output, txt_attn_output) tuple
        """
        attn = self.attn
        seq_txt = encoder_hidden_states.shape[1]
        
        # Compute QKV for image stream using ComfyUI's weights
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        
        # Compute QKV for text stream using ComfyUI's weights
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)
        
        # Reshape for multi-head attention [B, S, H, D]
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))
        
        # Apply QK normalization using ComfyUI's norm weights
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)
        
        # Apply VideoX-style RoPE (this is the key difference from ComfyUI!)
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
        
        # Concatenate for joint attention [txt, img]
        # Shape: [B, S, H, D] (BLHD format, matches VideoX's attention function)
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        
        # Build attention mask (mask text padding only, image tokens always valid)
        attn_mask = None
        if encoder_hidden_states_mask is not None:
            pad_mask = encoder_hidden_states_mask == 0
            if pad_mask.any():
                batch_size = pad_mask.shape[0]
                seq_total = joint_key.shape[1]
                seq_txt = encoder_hidden_states_mask.shape[1]
                attn_mask = torch.zeros(
                    (batch_size, 1, 1, seq_total),
                    device=joint_key.device,
                    dtype=joint_key.dtype,
                )
                attn_mask[:, :, :, :seq_txt] = torch.where(
                    pad_mask.unsqueeze(1).unsqueeze(1),
                    -torch.finfo(joint_key.dtype).max,
                    0.0,
                )
        
        # Use VideoX-compatible attention (Flash Attention / Sage Attention / SDPA fallback)
        # gen2_attention handles format conversion internally based on backend
        joint_hidden_states = gen2_attention(
            joint_query, joint_key, joint_value,
            attn_mask=attn_mask, dropout_p=0.0, causal=False
        )
        
        # Flatten heads: [B, S, H, D] -> [B, S, H*D]
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)
        
        # Split back to text and image
        txt_attn_output = joint_hidden_states[:, :seq_txt]
        img_attn_output = joint_hidden_states[:, seq_txt:]
        
        # Apply output projections using ComfyUI's weights
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout
        
        txt_attn_output = attn.to_add_out(txt_attn_output)
        
        return img_attn_output, txt_attn_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor],
        temb: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        VideoX-style forward pass using ComfyUI's weights.
        
        Args:
            hidden_states: Image features [B, img_seq, D]
            encoder_hidden_states: Text features [B, txt_seq, D]
            encoder_hidden_states_mask: Text attention mask
            temb: Timestep embedding [B, D]
            image_rotary_emb: VideoX-style RoPE tuple (vid_freqs, txt_freqs)
        
        Returns:
            (encoder_hidden_states, hidden_states) - text and image outputs
        """
        # Get modulation parameters
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]
        
        # Split into two sets (for norm1 and norm2)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
        
        # === Attention block ===
        # Image: norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)
        
        # Text: norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)
        
        # Joint attention with VideoX RoPE
        img_attn_output, txt_attn_output = self._attention_forward(
            img_modulated, txt_modulated, encoder_hidden_states_mask, image_rotary_emb
        )
        
        # Residual with gating
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
        
        # === MLP block ===
        # Image: norm2 + modulation + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)
        
        # Text: norm2 + modulation + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)
        
        # Clip for numerical stability (VideoX does this ONLY for fp16, NOT bfloat16)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clamp(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clamp(-65504, 65504)
        
        return encoder_hidden_states, hidden_states
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# =============================================================================
# Gen2 Model Wrapper - VideoX-compatible interface for ComfyUI models
# =============================================================================

class Gen2QwenImageModelWrapper:
    """
    Wrapper that provides VideoX-compatible forward interface for ComfyUI's QwenImage model.
    
    KEY DESIGN (Option E):
    1. Uses ComfyUI's fast model loading (weights are shared, not copied)
    2. Wraps each transformer block with Gen2QwenImageTransformerBlockWrapper
    3. Uses VideoX's QwenEmbedRope for exact RoPE calculation
    4. Operates on PACKED 3D latents throughout, matching VideoX exactly
    
    This gives us the best of both worlds:
    - ComfyUI's optimized model loading
    - VideoX's exact forward computation (RoPE + attention)
    
    INTERFACE UPDATE (V2):
    - Matches VideoX's QwenImageControlTransformer2DModel.forward() signature exactly
    - Supports guidance, attention_kwargs, control_context_scale, return_dict
    - Exposes .config property for compatibility with VideoX pipeline
    """
    
    def __init__(self, comfyui_model, control_model=None, latent_height=None, latent_width=None, control_layers=None):
        """
        Args:
            comfyui_model: ComfyUI's loaded QwenImageTransformer2DModel
            control_model: Optional QwenImageControlModel for ControlNet
            latent_height: Latent height for position calculations
            latent_width: Latent width for position calculations
            control_layers: List of layer indices where control hints are injected
        """
        self.model = comfyui_model
        self.control_model = control_model
        self.patch_size = comfyui_model.patch_size
        self.inner_dim = comfyui_model.inner_dim
        self.out_channels = comfyui_model.out_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.control_layers = control_layers or []
        
        # Create VideoX-style block wrappers (lazy - weights are shared, not copied)
        self._block_wrappers = None
        
        # Create VideoX-style RoPE embedder
        # Use same config as QwenImage model: theta=10000, axes_dim=[16, 56, 56]
        # CRITICAL: Must use scale_rope=True to match VideoX's training!
        # scale_rope=True creates symmetric position encoding centered at image center
        # scale_rope=False creates asymmetric encoding from corner - causes grid artifacts
        self.rope_embedder = QwenEmbedRope(
            theta=10000,
            axes_dim=[16, 56, 56],
            scale_rope=True,
        )
        
        # Create a config object that mimics VideoX's transformer config
        # This is needed for VideoX pipeline compatibility
        self._config = Gen2TransformerConfig(
            in_channels=64,  # QwenImage uses 64 input channels (16 * 4 after packing)
            out_channels=self.out_channels,
            guidance_embeds=False,  # QwenImage doesn't use guidance embeds
        )
    
    @property
    def config(self):
        """Expose config for VideoX pipeline compatibility."""
        return self._config
    
    def _ensure_block_wrappers(self):
        """Lazily create block wrappers on first use."""
        if self._block_wrappers is None:
            self._block_wrappers = [
                Gen2QwenImageTransformerBlockWrapper(block)
                for block in self.model.transformer_blocks
            ]
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        img_shapes: Optional[List] = None,
        txt_seq_lens: Optional[List[int]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        control_context: Optional[torch.Tensor] = None,
        control_context_scale: float = 1.0,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        VideoX-compatible forward pass on PACKED latents.
        
        This method signature EXACTLY matches VideoX's QwenImageControlTransformer2DModel.forward()
        so it can be used as a drop-in replacement in VideoX's pipeline.
        
        Uses our block wrappers with VideoX RoPE for exact compatibility.
        
        Args:
            hidden_states: PACKED 3D latents (batch, seq_len, channels*4)
            timestep: Timestep tensor (ALREADY DIVIDED BY 1000 by pipeline)
            encoder_hidden_states: Text embeddings (batch, seq_len, hidden_dim) or list of tensors
            encoder_hidden_states_mask: Attention mask for text (batch, seq_len) or list
            guidance: Guidance scale tensor (unused for QwenImage, kept for compatibility)
            img_shapes: List of [(frame, height, width)] tuples per batch - packed patch dimensions
            txt_seq_lens: List of text sequence lengths for RoPE
            attention_kwargs: Additional attention arguments (unused, kept for compatibility)
            control_context: PACKED control context (batch, seq_len, 132)
            control_context_scale: Scale for control hint injection
            return_dict: Whether to return dict (False returns raw tensor, VideoX uses False)
        
        Returns:
            PACKED 3D output (batch, seq_len, channels*4)
        """
        model = self.model
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Ensure block wrappers are created
        self._ensure_block_wrappers()
        
        # VideoX passes encoder_hidden_states as a LIST for CFG batching
        # Each element in the list is a tensor for one batch item
        # We need to handle both list and tensor formats
        if isinstance(encoder_hidden_states, list):
            # Stack list of tensors into a batch
            encoder_hidden_states = torch.stack([t.squeeze(0) if t.dim() > 2 else t for t in encoder_hidden_states])
        
        # Same for encoder_hidden_states_mask
        if isinstance(encoder_hidden_states_mask, list):
            encoder_hidden_states_mask = torch.stack([t.squeeze(0) if t.dim() > 1 else t for t in encoder_hidden_states_mask])
        
        # IMPORTANT: VideoX does NOT convert encoder_hidden_states_mask to attention mask!
        # The mask is only used for calculating txt_seq_lens for RoPE.
        # In VideoX's QwenDoubleStreamAttnProcessor2_0, attention_mask is always None.
        # The mask values (1 for valid, 0 for padding) are kept as-is for RoPE calculation.
        # We do NOT pass it to attention - attention runs on full sequence without masking.
        
        # Get image shape for RoPE generation
        # img_shapes is a list like [[(1, h_patches, w_patches)]] * batch_size
        if img_shapes and len(img_shapes) > 0 and len(img_shapes[0]) > 0:
            num_frames, h_patches, w_patches = img_shapes[0][0]
        else:
            # Fallback: derive from latent dimensions
            h_patches = self.latent_height // 2
            w_patches = self.latent_width // 2
            num_frames = 1
        
        # === KEY CHANGE: Use VideoX's QwenEmbedRope instead of ComfyUI's pe_embedder ===
        # This generates the exact same RoPE frequencies as VideoX
        img_shape = (num_frames, h_patches, w_patches)
        
        # CRITICAL: Use the passed txt_seq_lens parameter (actual token counts from mask.sum())
        # NOT the encoder_hidden_states.shape[1] which is the PADDED length!
        # VideoX calculates: txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        # For CFG batching, this is a list like [neg_len, pos_len]
        if txt_seq_lens is not None and len(txt_seq_lens) > 0:
            # Use the passed actual token counts
            rope_txt_seq_lens = txt_seq_lens
        else:
            # Fallback: use padded length (shouldn't happen in normal use)
            rope_txt_seq_lens = [encoder_hidden_states.shape[1]] * batch_size
        
        # Pass img_shapes directly to rope_embedder (VideoX format)
        # VideoX pipeline passes [[(1, h, w)], [(1, h, w)]] for CFG batch
        # QwenEmbedRope.forward extracts first batch item internally (all same shape)
        if img_shapes and len(img_shapes) > 0:
            rope_img_shapes = img_shapes  # Pass as-is, let rope_embedder handle
        else:
            rope_img_shapes = [[(num_frames, h_patches, w_patches)]]
        
        # Debug: Log RoPE parameters (only on first call via static flag)
        if not hasattr(self, '_rope_debug_logged'):
            self._rope_debug_logged = True
            print(f"[Gen2 RoPE] txt_seq_lens: {rope_txt_seq_lens} (actual token counts)")
            print(f"[Gen2 RoPE] encoder_hidden_states shape: {encoder_hidden_states.shape} (padded)")
            print(f"[Gen2 RoPE] img_shapes format: {rope_img_shapes[0] if rope_img_shapes else 'N/A'}")
        
        # QwenEmbedRope returns (vid_freqs, txt_freqs) tuple - exactly what VideoX uses
        image_rotary_emb = self.rope_embedder(rope_img_shapes, rope_txt_seq_lens, device)
        
        # Process inputs through embeddings (use ComfyUI's embedding layers)
        # hidden_states is already packed [B, seq, channels*4]
        hidden_states = model.img_in(hidden_states)
        
        # CRITICAL: Convert timestep to hidden_states dtype AFTER img_in (VideoX exact match)
        # This ensures time embedding calculation uses correct precision
        timestep = timestep.to(hidden_states.dtype)
        
        encoder_hidden_states_processed = model.txt_norm(encoder_hidden_states)
        encoder_hidden_states_processed = model.txt_in(encoder_hidden_states_processed)
        
        # Compute timestep embedding
        temb = model.time_text_embed(timestep, hidden_states, None)
        
        # Generate control hints if control model is available
        control_hints = None
        if self.control_model is not None and control_context is not None:
            # Ensure control model is on the correct device
            self.control_model.to(device=device, dtype=dtype)
            
            # CRITICAL: Use the passed txt_seq_lens (actual token counts from mask.sum())
            # NOT the encoder_hidden_states.shape[1] which is padded length!
            # Generate hints through control blocks (uses VideoX RoPE internally)
            control_hints = self.control_model.forward_control(
                x=hidden_states,
                control_context=control_context,
                encoder_hidden_states=encoder_hidden_states_processed,
                # VideoX does NOT use padding masks inside attention.
                # Keep mask only for txt_seq_lens (RoPE), but pass None to attention.
                encoder_hidden_states_mask=None,
                temb=temb,
                img_shape=img_shape,
                txt_seq_lens=rope_txt_seq_lens,  # Use same txt_seq_lens as main forward
            )
        
        # Prepare control layers mapping (use self.control_layers set during init)
        control_layer_set = set(self.control_layers) if self.control_layers else set()
        hint_idx = 0
        
        # === KEY CHANGE: Run through OUR block wrappers instead of ComfyUI's blocks ===
        # Our wrappers use ComfyUI's weights but VideoX's forward logic + RoPE
        for i, block_wrapper in enumerate(self._block_wrappers):
            encoder_hidden_states_processed, hidden_states = block_wrapper(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states_processed,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,  # VideoX-style tuple (vid_freqs, txt_freqs)
            )
            
            # Inject control hints at specified layers (VideoX style)
            if control_hints is not None and i in control_layer_set and hint_idx < len(control_hints):
                hint = control_hints[hint_idx]
                # hint shape: [batch, seq, hidden] - same as hidden_states
                hidden_states = hidden_states + control_context_scale * hint
                hint_idx += 1
        
        # Final normalization and projection (use ComfyUI's layers)
        hidden_states = model.norm_out(hidden_states, temb)
        output = model.proj_out(hidden_states)
        
        # Return packed 3D output (NO unpacking here - matches VideoX)
        return output
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

# =============================================================================
# Custom VAE Nodes - Full VideoX Compatible
# =============================================================================

# QwenImage VAE configuration constants
QWEN_VAE_CONFIG = {
    "attn_scales": [],
    "base_dim": 96,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "latents_mean": [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ],
    "latents_std": [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916
    ],
    "num_res_blocks": 2,
    "temperal_downsample": [False, True, True],
    "z_dim": 16
}


class Gen2_LoadQwenVAE:
    """
    Load QwenImage VAE with proper VideoX configuration.
    
    This node loads the VAE using VideoX's AutoencoderKLQwenImage class
    with correct latents_mean/std for proper normalization.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"default": "qwen_image_vae.safetensors"}),
                "precision": (["bf16", "fp16"], {"default": "bf16"}),
            }
        }
    
    RETURN_TYPES = ("GEN2_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "Gen2/QwenImage"
    
    def load(self, model_name, precision):
        # Import VideoX's VAE class
        # Note: _setup_videox_imports() adds videox-fun to sys.path at module load
        try:
            from videox_fun.models.qwenimage_vae import AutoencoderKLQwenImage
        except ImportError as e:
            # Try to provide helpful debug info
            current_dir = os.path.dirname(os.path.abspath(__file__))
            custom_nodes_dir = os.path.dirname(os.path.dirname(current_dir))
            videox_path = os.path.join(custom_nodes_dir, "videox-fun")
            raise ImportError(
                f"Cannot import AutoencoderKLQwenImage from videox_fun.\n"
                f"  Looking for videox-fun at: {videox_path}\n"
                f"  Exists: {os.path.exists(videox_path)}\n"
                f"  sys.path includes videox-fun: {videox_path in sys.path}\n"
                f"  Original error: {e}\n"
                f"Make sure videox-fun is installed in custom_nodes folder."
            )
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        
        # Load state dict
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
        
        # Check for Wan compiled VAE format
        if "conv1.weight" in vae_state_dict:
            use_wan_compiled_vae = True
            if not any(k.startswith("model.") for k in vae_state_dict.keys()):
                vae_state_dict = {f"model.{k}": v for k, v in vae_state_dict.items()}
        else:
            use_wan_compiled_vae = False
        
        # Filter kwargs to match class signature
        kwargs = dict(QWEN_VAE_CONFIG)
        sig = inspect.signature(AutoencoderKLQwenImage)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        # Create VAE model
        if use_wan_compiled_vae:
            try:
                from videox_fun.models.wan_vae import AutoencoderKLWanCompileQwenImage
                vae = AutoencoderKLWanCompileQwenImage(**accepted)
            except ImportError:
                vae = AutoencoderKLQwenImage(**accepted)
        else:
            vae = AutoencoderKLQwenImage(**accepted)
        
        vae.load_state_dict(vae_state_dict)
        vae = vae.eval().to(device=offload_device, dtype=weight_dtype)
        
        print(f"[Gen2] Loaded QwenImage VAE: {model_name}")
        print(f"  z_dim={vae.z_dim}, spatial_compression={vae.spatial_compression_ratio}")
                        
        # Return VAE with config
        return ({
            'model': vae,
            'config': QWEN_VAE_CONFIG,
            'dtype': weight_dtype,
            'device': offload_device,
        },)


# =============================================================================
# Sampler: VideoX Denoising Loop with ComfyUI Inputs
# =============================================================================

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """Calculate shift for FlowMatch scheduler (from VideoX)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps_v2(
    scheduler,
    num_inference_steps: int,
    device: torch.device,
    sigmas: Optional[List[float]] = None,
    mu: Optional[float] = None,
):
    """
    Retrieve timesteps from scheduler (simplified from VideoX).
    """
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    
    timesteps = scheduler.timesteps
    return timesteps, len(timesteps)


def pack_latents_v2(latents, batch_size, num_channels_latents, height, width, num_frame=None):
    """Pack latents from 5D to 3D sequence format (from VideoX)."""
    if num_frame is None:
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    else:
        latents = latents.view(batch_size, num_channels_latents, num_frame, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 3, 5, 1, 4, 6)
        latents = latents.reshape(batch_size, num_frame * (height // 2) * (width // 2), num_channels_latents * 4)
    return latents


def unpack_latents_v2(latents, height, width, vae_scale_factor, num_frame=None):
    """Unpack latents from 3D sequence to 5D format (from VideoX)."""
    batch_size, num_patches, channels = latents.shape
    if num_frame is None:
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)
    else:
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        
        latents = latents.view(batch_size, num_frame, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 4, 1, 2, 5, 3, 6)
        latents = latents.reshape(batch_size, channels // (2 * 2), num_frame, height, width)
    return latents


# =============================================================================
# Gen2 QwenImage Text Encoder Node (VideoX Style with Custom Tokenizer)
# =============================================================================

# VideoX encoding constants
VIDEOX_PROMPT_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
VIDEOX_DROP_IDX = 34  # Number of template tokens to drop (fixed, unlike ComfyUI's dynamic calculation)
VIDEOX_TOKENIZER_MAX_LENGTH = 1024

# Global tokenizer cache
_gen2_tokenizer = None

def get_gen2_tokenizer():
    """Load and cache our custom HuggingFace tokenizer."""
    global _gen2_tokenizer
    if _gen2_tokenizer is None:
        from transformers import Qwen2Tokenizer
        # Path to our tokenizer (relative to ComfyUI root)
        tokenizer_path = os.path.join(folder_paths.models_dir, "gen2", "qwen_2512_tokenizer")
        if os.path.exists(tokenizer_path):
            _gen2_tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
            print(f"[Gen2] Loaded custom tokenizer from: {tokenizer_path}")
        else:
            # Fallback: try to load from HuggingFace
            try:
                _gen2_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
                print(f"[Gen2] Loaded tokenizer from HuggingFace (fallback)")
            except Exception as e:
                raise RuntimeError(f"[Gen2] Failed to load tokenizer: {e}\nExpected path: {tokenizer_path}")
    return _gen2_tokenizer


class Gen2_QwenClipTextEncode:
    """
    Encode text prompts for QwenImage using VideoX's EXACT encoding process.
    
    KEY DIFFERENCES from ComfyUI's CLIPTextEncode:
    1. Uses our own HuggingFace tokenizer (not ComfyUI's wrapped version)
    2. Applies VideoX's exact template with FIXED drop_idx=34
    3. Extracts valid tokens and drops template prefix exactly like VideoX
    4. Returns embeddings with actual token length (no fixed padding)
    
    This ensures 100% compatibility with VideoX's text encoding.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, 
                         "tooltip": "The text prompt to encode"}),
                "max_sequence_length": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64,
                                        "tooltip": "Maximum sequence length (truncate if longer, VideoX default: 1024)"}),
                "embeds_dtype": (["auto", "fp16", "bf16"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("GEN2_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Gen2/QwenImage"
    DESCRIPTION = "Encode text for QwenImage using VideoX's exact encoding process"
    
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """Extract valid (non-padded) hidden states. Exact copy of VideoX's method."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result
    
    def encode(self, clip, text, max_sequence_length, embeds_dtype):
        """
        Encode text using VideoX's EXACT process:
        
        1. Apply template: "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        2. Tokenize with OUR tokenizer (not ComfyUI's)
        3. Get hidden states from text encoder
        4. Extract masked hidden states (valid tokens only)
        5. Drop first 34 tokens (template prefix) - FIXED, not dynamic!
        6. Return with proper attention mask
        """
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None")
        
        # Get our custom tokenizer
        tokenizer = get_gen2_tokenizer()
        
        # Get device
        device = mm.get_torch_device()
        
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
        
        # Apply VideoX template
        txt = [VIDEOX_PROMPT_TEMPLATE.format(t) for t in text]
        
        # Tokenize using OUR tokenizer with VideoX's exact parameters
        txt_tokens = tokenizer(
            txt, 
            max_length=VIDEOX_TOKENIZER_MAX_LENGTH + VIDEOX_DROP_IDX,
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids = txt_tokens.input_ids.to(device)
        attention_mask = txt_tokens.attention_mask.to(device)
        
        # Load CLIP model to GPU
        clip.load_model()
        
        # Access the underlying text encoder model
        # Structure: clip.cond_stage_model (QwenImageTEModel) -> qwen25_7b (Qwen25_7BVLIModel)
        # qwen25_7b: SDClipModel with transformer = Qwen25_7BVLI
        cond_stage = clip.cond_stage_model
        
        # Prefer VideoX-style path: call the text encoder directly and use last hidden state
        # This keeps dtype/normalization consistent with VideoX
        hidden_states = None
        text_encoder_dtype = None
        
        with torch.no_grad():
            try:
                text_encoder_dtype = getattr(cond_stage, "dtype", None)
                encoder_out = cond_stage(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = encoder_out.hidden_states[-1]
            except Exception:
                if hasattr(cond_stage, 'qwen25_7b'):
                    clip_model = cond_stage.qwen25_7b
                elif hasattr(cond_stage, 'clip'):
                    clip_model = getattr(cond_stage, cond_stage.clip)
                else:
                    clip_model = cond_stage
                
                transformer = clip_model.transformer if hasattr(clip_model, 'transformer') else clip_model.model
                text_encoder_dtype = getattr(transformer, "dtype", None)
                
                embeddings = transformer.get_input_embeddings()(input_ids, out_dtype=transformer.dtype if hasattr(transformer, "dtype") else torch.float16)
                hidden_states = transformer(
                    None,  # x (not used when embeds provided)
                    attention_mask=attention_mask.float(),
                    embeds=embeddings,
                    num_tokens=None,
                    intermediate_output=None,  # "last" layer
                    final_layer_norm_intermediate=True,
                    dtype=embeddings.dtype,
                    embeds_info=[]
                )
                
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
        
        # hidden_states: [batch, seq_len, hidden_dim]
        # Now apply VideoX's exact post-processing
        
        # Extract masked hidden states (valid tokens only)
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        
        # Drop first VIDEOX_DROP_IDX tokens (template prefix)
        split_hidden_states = [e[VIDEOX_DROP_IDX:] for e in split_hidden_states]
        
        # Create attention mask (all 1s for actual tokens after template removal)
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=device) for e in split_hidden_states]
        
        # Get max sequence length in batch
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        
        # Truncate to max_sequence_length if needed
        if max_seq_len > max_sequence_length:
            split_hidden_states = [e[:max_sequence_length] for e in split_hidden_states]
            attn_mask_list = [m[:max_sequence_length] for m in attn_mask_list]
            max_seq_len = max_sequence_length
        
        # Pad to max_seq_len in batch (VideoX style - pad with zeros)
        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) 
            for u in split_hidden_states
        ])
        encoder_attention_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) 
            for u in attn_mask_list
        ])
        
        # Get actual token count (before batch padding, after template removal)
        actual_seq_lens = encoder_attention_mask.sum(dim=1).tolist()
        
        # Convert to requested dtype if specified, otherwise use compute dtype from encoder
        # This ensures embeddings are always in a compute-friendly dtype (bf16/fp16)
        # even when text encoder uses quantized weights
        if embeds_dtype == "bf16":
            target_dtype = torch.bfloat16
        elif embeds_dtype == "fp16":
            target_dtype = torch.float16
        else:
            # "auto" mode: use get_compute_dtype to handle quantized encoders
            base_dtype = text_encoder_dtype if text_encoder_dtype is not None else prompt_embeds.dtype
            target_dtype = get_compute_dtype(base_dtype, fallback_dtype=torch.bfloat16)
        prompt_embeds = prompt_embeds.to(dtype=target_dtype)
        
        # Create GEN2_CONDITIONING format
        conditioning = {
            "embeds": prompt_embeds,  # [batch, seq_len, hidden]
            "attention_mask": encoder_attention_mask,  # [batch, seq_len]
            "txt_seq_len": actual_seq_lens,  # Actual token count per batch
            "pooled_output": None,
        }
        
        # Diagnostic output
        print(f"[Gen2 TextEncode] VideoX-style: seq_len={prompt_embeds.shape[1]}, actual_tokens={actual_seq_lens}")
        print(f"[Gen2 TextEncode] embeds dtype: {prompt_embeds.dtype}")
        print(f"[Gen2 TextEncode] embeds: mean={prompt_embeds.mean().item():.6f}, std={prompt_embeds.std().item():.6f}")
        
        return (conditioning,)


# =============================================================================
# Gen2 LoRA Loader Node
# =============================================================================

class Gen2_LoadQwenLora:
    """
    Load LoRA for QwenImage ControlNet (VideoX style).
    
    This node stores LoRA path and strength for use by Gen2_QwenImageControlSampler.
    Multiple LoRAs can be chained by connecting the output to another Gen2_LoadQwenLora node.
    
    The LoRA will be merged before sampling and unmerged after sampling,
    ensuring the base model weights are not permanently modified.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {"default": None}),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),  # Match VideoX range
            },
            "optional": {
                "lora": ("GEN2_LORA",),  # For chaining multiple LoRAs
            }
        }
    
    RETURN_TYPES = ("GEN2_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load_lora"
    CATEGORY = "Gen2/QwenImage"
    
    def load_lora(self, lora_name, strength, lora=None):
        """
        Store LoRA information for later use by sampler.
        
        Args:
            lora_name: Name of the LoRA file from ComfyUI's loras folder
            strength: LoRA strength multiplier
            lora: Optional previous LoRA info for chaining
        
        Returns:
            lora_info: Dict containing list of LoRA paths and strengths
        """
        # Start with previous LoRAs if provided
        if lora is not None:
            lora_paths = list(lora.get('lora_paths', []))
            lora_strengths = list(lora.get('lora_strengths', []))
        else:
            lora_paths = []
            lora_strengths = []
        
        # Add this LoRA
        if lora_name is not None:
            full_path = folder_paths.get_full_path("loras", lora_name)
            if full_path:
                lora_paths.append(full_path)
                lora_strengths.append(strength)
                print(f"[Gen2 LoRA] Added LoRA: {lora_name} (strength: {strength})")
            else:
                print(f"[Gen2 LoRA] Warning: LoRA not found: {lora_name}")
        
        lora_info = {
            'lora_paths': lora_paths,
            'lora_strengths': lora_strengths,
        }
        
        return (lora_info,)


class Gen2_QwenImageControlSampler:
    """
    QwenImage ControlNet Sampler using VideoX's EXACT denoising loop.
    
    Features:
    - Takes GEN2_CONDITIONING from Gen2_QwenClipTextEncode (VideoX-compatible)
    - Uses VideoX's exact FlowMatch denoising loop
    - Supports Flow, Flow_Unipc, and Flow_DPM++ schedulers
    - Proper True CFG with norm re-scaling
    - VAE is passed through from Gen2_ApplyQwenControlNetFun
    
    Usage:
    1. Load model with ComfyUI's Load Diffusion Model (type: qwen_image)
    2. Load VAE with Gen2_LoadQwenVAE
    3. Load ControlNet with Gen2_LoadQwenControlNetFun
    4. Apply ControlNet with Gen2_ApplyQwenControlNetFun
    5. Encode prompts with Gen2_QwenClipTextEncode (NOT standard CLIPTextEncode!)
    6. Use this sampler for denoising
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("GEN2_WRAPPED_MODEL",),
                "positive": ("GEN2_CONDITIONING",),  # From Gen2_QwenClipTextEncode
                "negative": ("GEN2_CONDITIONING",),  # From Gen2_QwenClipTextEncode
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shift": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1}),
                "sampler": (["Flow", "Flow_Unipc", "Flow_DPM++"], {"default": "Flow"}),
            },
            "optional": {
                "lora": ("GEN2_LORA",),  # Optional LoRA from Gen2_LoadQwenLora
                "attention_backend": (["AUTO", "FLASH_ATTENTION", "SAGE_ATTENTION", "SDPA"], {"default": "AUTO"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "Gen2/QwenImage"
    
    def sample(self, model, positive, negative, width, height, seed, steps, cfg, shift, sampler, lora=None, attention_backend="AUTO"):
        """
        Run sampling using VideoX's exact denoising loop.
        
        This method closely follows VideoX's QwenImageControlPipeline.__call__()
        but uses GEN2_CONDITIONING from Gen2_QwenClipTextEncode.
        
        Key difference from previous version:
        - positive/negative are now GEN2_CONDITIONING dicts with properly sized embeddings
        - No more pad_or_truncate - embeddings already have correct length from text encoder
        - txt_seq_len is included in the conditioning dict
        """
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # Get model components (all from the wrapped_model dict)
        transformer = model['wrapped_model']  # Gen2QwenImageModelWrapper
        
        # Get dtype information
        # - model_storage_dtype: actual weight dtype (may be fp8, int8 for quantized models)
        # - compute_dtype: dtype for compute operations (always bf16/fp16)
        model_storage_dtype = model.get('model_storage_dtype', model.get('dtype'))
        compute_dtype = model.get('compute_dtype', model.get('dtype'))
        
        # Verify dtype (for debugging)
        base_transformer = transformer.model if hasattr(transformer, "model") else transformer
        actual_model_dtype = next(base_transformer.parameters()).dtype
        
        # For quantized models, actual_model_dtype may differ from compute_dtype - this is expected
        if actual_model_dtype in QUANTIZED_DTYPES:
            print(f"[Gen2] Quantized model detected: storage={actual_model_dtype}, compute={compute_dtype}")
        elif actual_model_dtype != compute_dtype:
            print(f"[Gen2 WARNING] dtype mismatch! model={actual_model_dtype}, expected={compute_dtype}")
        
        # Get control context (prepared by Gen2_ApplyQwenControlNetFun)
        control_context_raw = model['control_context']
        control_context_scale = model['control_context_scale']
        
        # Compute latent dimensions from image dimensions
        vae_scale_factor = 8
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        
        # Get VAE from wrapped model
        vae = model['vae']
        
        # VAE components (for decoding only)
        vae_model = vae['model']
        vae_config = vae['config']
        vae_storage_dtype = vae['dtype']
        
        # VAE compute dtype (ensure it's not quantized)
        vae_compute_dtype = get_compute_dtype(vae_storage_dtype, fallback_dtype=compute_dtype)
        
        # Move VAE to device
        vae_model = vae_model.to(device)
        
        # =================================================================
        # 0. Apply LoRA (if provided) - VideoX style
        # =================================================================
        lora_applied = False
        lora_paths_applied = []
        lora_strengths_applied = []
        
        if lora is not None and len(lora.get('lora_paths', [])) > 0:
            lora_paths_applied = lora['lora_paths']
            lora_strengths_applied = lora['lora_strengths']
            
            # Get the actual transformer model from our wrapper
            actual_transformer = transformer.model if hasattr(transformer, "model") else transformer
            
            # For quantized models, LoRA merge uses compute_dtype for calculations
            # VideoX behavior: merge LoRA in fp32/bf16, then the model handles quantized weights
            lora_dtype = compute_dtype
            
            print(f"[Gen2] Merging {len(lora_paths_applied)} LoRA(s) (dtype={lora_dtype})...")
            for lora_path, lora_strength in zip(lora_paths_applied, lora_strengths_applied):
                gen2_merge_lora(actual_transformer, lora_path, lora_strength, device=device, dtype=lora_dtype)
            
            lora_applied = True
        
        # =================================================================
        # 1. Extract prompt embeddings from GEN2_CONDITIONING (VideoX style)
        # =================================================================
        # GEN2_CONDITIONING format: {"embeds": tensor, "attention_mask": tensor, "txt_seq_len": List[int]}
        # Embeddings are already properly sized by Gen2_QwenClipTextEncode (no padding beyond actual tokens)
        
        prompt_embeds = positive["embeds"].to(device=device)
        prompt_embeds_mask = positive["attention_mask"].to(device=device)
        pos_txt_seq_len = positive["txt_seq_len"]  # Actual token counts per batch
        
        negative_prompt_embeds = negative["embeds"].to(device=device)
        negative_prompt_embeds_mask = negative["attention_mask"].to(device=device)
        neg_txt_seq_len = negative["txt_seq_len"]  # Actual token counts per batch
        
        # Use text encoder dtype for compute (VideoX ties compute dtype to prompt embeds)
        compute_dtype = prompt_embeds.dtype
        prompt_embeds = prompt_embeds.to(dtype=compute_dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=compute_dtype)

        # Align control context dtype with compute dtype (VideoX uses text_encoder dtype)
        control_context = control_context_raw.to(device=device, dtype=compute_dtype)
        
        # For CFG, both positive and negative must have the same sequence length
        # VideoX pads to the max of (pos_len, neg_len) for consistent torch.stack()
        max_seq_len = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
        
        def pad_to_length(embeds, mask, target_len):
            """Pad embeddings and mask to target_len with zeros."""
            seq_len = embeds.shape[1]
            if seq_len < target_len:
                batch = embeds.shape[0]
                hidden = embeds.shape[2]
                pad_len = target_len - seq_len
                embeds = torch.cat([embeds, torch.zeros(batch, pad_len, hidden, device=embeds.device, dtype=embeds.dtype)], dim=1)
                mask = torch.cat([mask, torch.zeros(batch, pad_len, device=mask.device, dtype=mask.dtype)], dim=1)
            return embeds, mask
        
        # Pad both to same length (max of pos/neg) for CFG batching
        prompt_embeds, prompt_embeds_mask = pad_to_length(prompt_embeds, prompt_embeds_mask, max_seq_len)
        negative_prompt_embeds, negative_prompt_embeds_mask = pad_to_length(negative_prompt_embeds, negative_prompt_embeds_mask, max_seq_len)
        
        batch_size = prompt_embeds.shape[0]
        
        # VideoX behavior:
        # - UI "cfg" is passed as guidance_scale (only used if guidance_embeds=True).
        # - true_cfg_scale is fixed at default (4.0) unless explicitly overridden.
        guidance_scale_input = cfg
        true_cfg_scale = 4.0

        # Determine if we do true CFG (VideoX uses true_cfg_scale)
        has_neg_prompt = negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        
        # txt_seq_lens from GEN2_CONDITIONING (actual token counts, not padded length)
        # This is CRITICAL - these are the real token counts for RoPE calculation
        txt_seq_lens = pos_txt_seq_len
        negative_txt_seq_lens = neg_txt_seq_len if do_true_cfg else None
        
        # Determine which attention backend will be used (matches gen2_attention fallback chain)
        import os
        if attention_backend is not None and attention_backend != "AUTO":
            os.environ["VIDEOX_ATTENTION_TYPE"] = attention_backend
        attn_type = os.environ.get("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION")
        if attn_type == "SAGE_ATTENTION" and SAGE_ATTENTION_AVAILABLE:
            active_backend = "SageAttention"
        elif attn_type == "FLASH_ATTENTION" and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
            active_backend = "FlashAttn3" if FLASH_ATTN_3_AVAILABLE else "FlashAttn2"
        else:
            active_backend = "SDPA"
        
        print(f"[Gen2] Sampling: {width}x{height}, steps={steps}, cfg_input={cfg}, true_cfg_scale={true_cfg_scale}, shift={shift}, sampler={sampler}")
        print(f"[Gen2] Precision: model_storage={model_storage_dtype}, compute={compute_dtype}, vae={vae_storage_dtype}, control_ctx={control_context.dtype}")
        print(f"[Gen2] Attention backend: {active_backend}")
        print(f"[Gen2] Control scale: {control_context_scale}")
        print(f"[Gen2] Prompt embeds: pos={prompt_embeds.shape} (tokens={pos_txt_seq_len}), neg={negative_prompt_embeds.shape} (tokens={neg_txt_seq_len})")
        # Diagnostic: compare embedding statistics with VideoX
        print(f"[Gen2 Diag] pos_embeds: mean={prompt_embeds.mean().item():.6f}, std={prompt_embeds.std().item():.6f}")
        print(f"[Gen2 Diag] neg_embeds: mean={negative_prompt_embeds.mean().item():.6f}, std={negative_prompt_embeds.std().item():.6f}")
        
        # Check prompt embeddings for NaN/Inf
        if torch.isnan(prompt_embeds).any() or torch.isinf(prompt_embeds).any():
            print("[Gen2 WARNING] NaN/Inf in positive prompt embeddings!")
        if negative_prompt_embeds is not None:
            if torch.isnan(negative_prompt_embeds).any() or torch.isinf(negative_prompt_embeds).any():
                print("[Gen2 WARNING] NaN/Inf in negative prompt embeddings!")
        
        # Check control context for extreme values
        ctrl_min, ctrl_max = control_context.min().item(), control_context.max().item()
        print(f"[Gen2] Control context range: [{ctrl_min:.4f}, {ctrl_max:.4f}]")
        
        # =================================================================
        # 2. Prepare latents (random noise)
        # =================================================================
        num_channels_latents = 16  # QwenImage uses 16 channels
        
        # Create random latents and pack them
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(
            (batch_size, 1, num_channels_latents, latent_height, latent_width),
            generator=generator, device=device, dtype=compute_dtype
        )
        latents = pack_latents_v2(latents, batch_size, num_channels_latents, latent_height, latent_width)
        
        # Check initial latents
        lat_min, lat_max = latents.min().item(), latents.max().item()
        print(f"[Gen2] Initial latents range: [{lat_min:.4f}, {lat_max:.4f}]")
        
        # =================================================================
        # 3. Prepare img_shapes for RoPE
        # =================================================================
        # img_shapes contains the PACKED dimensions: (frames, packed_h, packed_w)
        # Packed dimensions are latent dimensions / 2 due to 2x2 packing
        packed_h = latent_height // 2
        packed_w = latent_width // 2
        img_shapes = [
            [(1, packed_h, packed_w)]
        ] * batch_size
        
        # =================================================================
        # 5. Setup scheduler (using VideoX's scheduler setup)
        # =================================================================
        scheduler = get_qwen_scheduler(sampler, shift)
        
        sigmas = np.linspace(1.0, 1 / steps, steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        
        timesteps, num_inference_steps = retrieve_timesteps_v2(
            scheduler, steps, device, sigmas=sigmas, mu=mu
        )
        
        # =================================================================
        # 6. Denoising loop (VideoX exact copy)
        # =================================================================
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        
        scheduler.set_begin_index(0)
        
        for i, t in enumerate(timesteps):
            if do_true_cfg:
                # Double batch for CFG
                latent_model_input = torch.cat([latents] * 2)
                
                # Combine negative and positive embeddings
                # VideoX passes these as lists
                prompt_embeds_mask_input = [
                    m for m in negative_prompt_embeds_mask
                ] + [m for m in prompt_embeds_mask] if prompt_embeds_mask.dim() > 1 else [
                    negative_prompt_embeds_mask, prompt_embeds_mask
                ]
                prompt_embeds_input = [
                    e for e in negative_prompt_embeds
                ] + [e for e in prompt_embeds] if prompt_embeds.dim() > 2 else [
                    negative_prompt_embeds, prompt_embeds
                ]
                
                img_shapes_input = img_shapes * 2
                txt_seq_lens_input = (negative_txt_seq_lens or txt_seq_lens) + txt_seq_lens
                control_context_input = torch.cat([control_context] * 2)
            else:
                latent_model_input = latents
                prompt_embeds_mask_input = prompt_embeds_mask
                prompt_embeds_input = prompt_embeds
                img_shapes_input = img_shapes
                txt_seq_lens_input = txt_seq_lens
                control_context_input = control_context
            
            # Scale model input if scheduler requires
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # Timestep (normalized by 1000 as VideoX does)
            timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
            
            # Call transformer (using VideoX's exact autocast context)
            # Note: torch.cuda.device sets the default device context, torch.no_grad disables gradients
            with torch.cuda.amp.autocast(dtype=compute_dtype), torch.cuda.device(device=device), torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=(
                        torch.full([1], guidance_scale_input, device=device, dtype=torch.float32).expand(latent_model_input.shape[0])
                        if getattr(transformer, "config", None) is not None and transformer.config.guidance_embeds
                        else None
                    ),
                    encoder_hidden_states_mask=prompt_embeds_mask_input,
                    encoder_hidden_states=prompt_embeds_input,
                    img_shapes=img_shapes_input,
                    txt_seq_lens=txt_seq_lens_input,
                    attention_kwargs=None,
                    control_context=control_context_input,
                    control_context_scale=control_context_scale,
                    return_dict=False,
                )
            
            # Check transformer output for NaN/Inf
            if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                print(f"[Gen2 WARNING] NaN/Inf in transformer output at step {i}, timestep={t.item():.4f}")
            
            # Apply CFG (VideoX's true_cfg with norm rescaling - EXACT MATCH)
            if do_true_cfg:
                neg_noise_pred, pos_noise_pred = noise_pred.chunk(2)
                comb_pred = neg_noise_pred + true_cfg_scale * (pos_noise_pred - neg_noise_pred)
                
                # Norm rescaling - VideoX does NOT use epsilon here
                # cond_norm / noise_norm exactly as VideoX does
                cond_norm = torch.norm(pos_noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                
                # Log CFG stats at first, middle, and last step
                if i == 0 or i == num_inference_steps // 2 or i == num_inference_steps - 1:
                    ratio = (cond_norm / noise_norm).mean().item()
                    print(f"[Gen2 CFG] step={i}: cond_norm_mean={cond_norm.mean().item():.4f}, noise_norm_mean={noise_norm.mean().item():.4f}, ratio={ratio:.4f}")
                
                # EXACT VideoX: comb_pred * (cond_norm / noise_norm)
                noise_pred = comb_pred * (cond_norm / noise_norm)
            
            # Check for NaN/Inf (debugging numerical instability)
            if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                print(f"[Gen2 WARNING] NaN/Inf detected in noise_pred at step {i}")
            
            # Scheduler step
            latents_dtype = latents.dtype
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # Check latents for NaN/Inf
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                print(f"[Gen2 WARNING] NaN/Inf detected in latents at step {i}")
            
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)
            
            pbar.update(1)
        
        # =================================================================
        # 7. Decode latents to image
        # =================================================================
        # Check final latents
        final_lat_min, final_lat_max = latents.min().item(), latents.max().item()
        final_lat_std = latents.std().item()
        print(f"[Gen2] Final latents range: [{final_lat_min:.4f}, {final_lat_max:.4f}], std={final_lat_std:.4f}")
        if torch.isnan(latents).any() or torch.isinf(latents).any():
            print("[Gen2 ERROR] NaN/Inf in final latents! Image will be corrupted.")
        
        # Calculate actual image dimensions from latent dimensions
        # latent_height/width are the 5D latent dims, image = latent * vae_scale_factor
        actual_height = latent_height * vae_scale_factor
        actual_width = latent_width * vae_scale_factor
        
        # Number of patches to keep (packed_h * packed_w)
        num_patches_to_keep = packed_h * packed_w
        
        # Unpack latents
        latents = unpack_latents_v2(
            latents[:, :num_patches_to_keep],
            actual_height, actual_width, vae_scale_factor, num_frame=1
        )
        latents = latents.to(vae_model.dtype)
        latents = latents[:, :, :1]  # Take first frame
        
        # Denormalize
        latents_mean_dec = torch.tensor(vae_config['latents_mean']).view(1, vae_config['z_dim'], 1, 1, 1).to(latents.device, latents.dtype)
        latents_std_dec = 1.0 / torch.tensor(vae_config['latents_std']).view(1, vae_config['z_dim'], 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std_dec + latents_mean_dec
        
        # Decode
        with torch.no_grad():
            decoded = vae_model.decode(latents, return_dict=False)[0]
        
        # Take first frame if 5D
        if decoded.ndim == 5:
            decoded = decoded[:, :, 0]
        
        # [-1, 1] -> [0, 1]
        decoded = (decoded + 1.0) / 2.0
        decoded = decoded.clamp(0, 1)
        
        # BCHW -> BHWC
        image = decoded.permute(0, 2, 3, 1).cpu().float()
        
        # Move VAE back
        vae_model = vae_model.to(vae['device'])
        
        # =================================================================
        # 8. Unmerge LoRA (if applied) - Restore original weights
        # =================================================================
        if lora_applied:
            print(f"[Gen2] Unmerging {len(lora_paths_applied)} LoRA(s) (dtype={lora_dtype})...")
            actual_transformer = transformer.model if hasattr(transformer, "model") else transformer
            for lora_path, lora_strength in zip(lora_paths_applied, lora_strengths_applied):
                gen2_unmerge_lora(actual_transformer, lora_path, lora_strength, device=device, dtype=lora_dtype)
        
        print(f"[Gen2 V2] Generated image: {tuple(image.shape)} ({actual_width}x{actual_height})")
        
        return (image,)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "Gen2_LoadQwenControlNetFun": Gen2_LoadQwenControlNetFun,
    "Gen2_LoadQwenVAE": Gen2_LoadQwenVAE,
    "Gen2_ApplyQwenControlNetFun": Gen2_ApplyQwenControlNetFun,
    "Gen2_QwenClipTextEncode": Gen2_QwenClipTextEncode,
    "Gen2_LoadQwenLora": Gen2_LoadQwenLora,
    "Gen2_QwenImageControlSampler": Gen2_QwenImageControlSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_LoadQwenControlNetFun": "Gen2 Load QwenImage ControlNet",
    "Gen2_LoadQwenVAE": "Gen2 Load QwenImage VAE",
    "Gen2_ApplyQwenControlNetFun": "Gen2 Apply QwenImage ControlNet",
    "Gen2_QwenClipTextEncode": "Gen2 QwenImage Text Encode",
    "Gen2_LoadQwenLora": "Gen2 Load QwenImage LoRA",
    "Gen2_QwenImageControlSampler": "Gen2 QwenImage Control Sampler",
}
