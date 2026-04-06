"""
Gen2 QwenImage Core - Attention Functions

Flash Attention / Sage Attention / SDPA fallback, matching VideoX's attention_utils.py.
Also contains the QwenDoubleStreamAttnProcessor2_0 attention processor.
"""

from typing import Optional, Tuple
import warnings

import torch
import torch.nn.functional as F

from .rope import apply_rotary_emb_qwen

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
sageattn = None
SAGE_ATTENTION_AVAILABLE = False
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
except:
    try:
        from sageattention import sageattn
        SAGE_ATTENTION_AVAILABLE = True
    except:
        pass

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


# =============================================================================
# Attention Functions
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
        attn,
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
        
        # Use VideoX-compatible attention
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

