"""
Gen2 QwenImage Core - Data Type Utilities

Handles dtype detection for quantized models (fp8, GGUF) and compute dtype selection.
"""

import torch

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

