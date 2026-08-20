from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .types import V9_LORA_ALPHA, V9_LORA_RANK, V9_VIRTUAL_TOKEN_COUNT


def interpolate_embedding(frozen_initializer: torch.Tensor, learned_weight: torch.Tensor, strength: float) -> torch.Tensor:
    if frozen_initializer.shape != learned_weight.shape:
        raise ValueError("embedding initializer and weight shapes differ")
    if frozen_initializer.ndim != 2 or frozen_initializer.shape[0] != V9_VIRTUAL_TOKEN_COUNT:
        raise ValueError(f"V9 embeddings must be [{V9_VIRTUAL_TOKEN_COUNT}, H]")
    return frozen_initializer + float(strength) * (learned_weight - frozen_initializer)


def remap_atomic_token_ids(token_ids: torch.Tensor, atomic_token_id: int, lookup_token_id: int) -> torch.Tensor:
    return token_ids.masked_fill(token_ids == int(atomic_token_id), int(lookup_token_id))


def replace_trigger_embeddings(hidden: torch.Tensor, trigger_mask: torch.Tensor, virtual_token_indices: torch.Tensor, replacements: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or trigger_mask.shape != hidden.shape[:2] or virtual_token_indices.shape != trigger_mask.shape:
        raise ValueError("hidden must be [B,L,H] and trigger metadata must be [B,L]")
    if replacements.shape != (V9_VIRTUAL_TOKEN_COUNT, hidden.shape[-1]):
        raise ValueError(f"replacement embeddings must be [{V9_VIRTUAL_TOKEN_COUNT}, {hidden.shape[-1]}]")
    selected = replacements.to(hidden.device, hidden.dtype)[virtual_token_indices.clamp_min(0)]
    return torch.where(trigger_mask.to(hidden.device).unsqueeze(-1), selected, hidden)


def module_lora_update(inputs: torch.Tensor, down: torch.Tensor, up: torch.Tensor, strength: float, alpha: float = V9_LORA_ALPHA, rank: int = V9_LORA_RANK) -> torch.Tensor:
    if inputs.shape[-1] != down.shape[1] or down.shape[0] != rank or up.shape[1] != rank:
        raise ValueError(
            f"module-LoRA shape mismatch: input={inputs.shape[-1]}, down={tuple(down.shape)}, up={tuple(up.shape)}"
        )
    return F.linear(F.linear(inputs, down), up) * (float(alpha) / rank) * float(strength)


def apply_masked_module_lora(output: torch.Tensor, inputs: torch.Tensor, trigger_mask: torch.Tensor, down: torch.Tensor, up: torch.Tensor, strength: float, alpha: float = V9_LORA_ALPHA, rank: int = V9_LORA_RANK) -> torch.Tensor:
    if trigger_mask.shape != inputs.shape[:-1] or output.shape[:-1] != inputs.shape[:-1]:
        raise ValueError("module-LoRA mask/input/output token dimensions differ")
    update = module_lora_update(inputs, down, up, strength, alpha, rank)
    return output + update * trigger_mask.to(output.device, output.dtype).unsqueeze(-1)


def combine_native_conditioning_layers(states: Sequence[torch.Tensor]) -> torch.Tensor:
    if not states:
        raise ValueError("at least one native Ideogram4 conditioning layer is required")
    shape = states[0].shape
    if len(shape) != 3 or any(item.shape != shape for item in states):
        raise ValueError("native Ideogram4 conditioning layers must have identical [B,L,H] shapes")
    stacked = torch.stack(tuple(states), dim=0)
    arranged = stacked.permute(1, 2, 3, 0)
    return arranged.reshape(shape[0], shape[1], shape[2] * len(states))


def apply_conditioning_attention_mask(conditioning: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return conditioning
    if attention_mask.shape != conditioning.shape[:2]:
        raise ValueError("attention mask must match conditioning [B,L]")
    return conditioning * attention_mask.to(conditioning.device, conditioning.dtype).unsqueeze(-1)


__all__ = [
    "apply_conditioning_attention_mask", "apply_masked_module_lora", "combine_native_conditioning_layers",
    "interpolate_embedding", "module_lora_update", "remap_atomic_token_ids", "replace_trigger_embeddings",
]
