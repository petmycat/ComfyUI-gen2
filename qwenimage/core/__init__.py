"""
Gen2 QwenImage Core - Internal logic modules.

This package contains the split logic from the monolithic qwenimage_controlnet_fun.py.
Import order matters: imports.py sets up videox path first.
"""

# Setup VideoX imports first (adds videox-fun to sys.path)
from .imports import _setup_videox_imports, DIFFUSERS_AVAILABLE

# Dtype utilities
from .dtype_utils import QUANTIZED_DTYPES, get_compute_dtype, get_autocast_dtype

# RoPE
from .rope import apply_rotary_emb_qwen, QwenEmbedRope

# Attention
from .attention import (
    FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE, SAGE_ATTENTION_AVAILABLE,
    gen2_flash_attention, gen2_attention, QwenDoubleStreamAttnProcessor2_0,
)

# LoRA
from .lora import (
    _build_lora_key_map, _parse_lora_weights,
    gen2_merge_lora, gen2_unmerge_lora,
)

# Conditioning
from .conditioning import Gen2TransformerConfig, extract_from_conditioning, get_txt_seq_len_from_mask

# Model
from .model import QwenImageTransformerBlock, QwenImageControlTransformerBlock, QwenImageControlModel

# Model wrapper
from .model_wrapper import Gen2QwenImageTransformerBlockWrapper, Gen2QwenImageModelWrapper

# Scheduler
from .scheduler import filter_kwargs, get_qwen_scheduler

# Latent utilities
from .latent_utils import (
    QWEN_VAE_CONFIG, calculate_shift, retrieve_timesteps_v2,
    pack_latents_v2, unpack_latents_v2,
)

# Tokenizer
from .tokenizer import (
    get_gen2_tokenizer,
    VIDEOX_PROMPT_TEMPLATE, VIDEOX_DROP_IDX, VIDEOX_TOKENIZER_MAX_LENGTH,
)

