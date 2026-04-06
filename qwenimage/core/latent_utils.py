"""
Gen2 QwenImage Core - Latent Utilities

Pack/unpack latents, VAE config, scheduler shift calculation, timestep retrieval.
"""

from typing import List, Optional

import torch
import numpy as np


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

