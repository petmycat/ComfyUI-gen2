from __future__ import annotations

import torch
import torch.nn.functional as F

from .types import PhaseCRouterBundle


def aggregate_projected_occurrences(projected_states: torch.Tensor, occurrence_count: int, token_count: int) -> torch.Tensor:
    if projected_states.ndim == 3:
        expected = (occurrence_count, token_count)
        if tuple(projected_states.shape[:2]) != expected:
            raise ValueError(f"Projected activator states must start with shape {expected}")
        occurrences = projected_states
    elif projected_states.ndim == 2:
        if projected_states.shape[0] != occurrence_count * token_count:
            raise ValueError("Projected activator state count does not match occurrence contract")
        occurrences = projected_states.reshape(occurrence_count, token_count, projected_states.shape[-1])
    else:
        raise ValueError("Projected activator states must be [occurrences,tokens,D] or [occurrences*tokens,D]")
    result = occurrences.float().sum(dim=0, keepdim=True)
    if not torch.isfinite(result).all().item():
        raise ValueError("Aggregated activator states contain NaN or infinity")
    return result


def normalize_canonical_timestep(timestep: torch.Tensor) -> torch.Tensor:
    if not isinstance(timestep, torch.Tensor):
        raise TypeError("Canonical timestep must be a tensor")
    if timestep.ndim == 1:
        value = timestep.unsqueeze(-1)
    elif timestep.ndim == 2 and timestep.shape[1] == 1:
        value = timestep
    elif timestep.ndim == 2:
        first = timestep[:, :1]
        if not torch.allclose(timestep.float(), first.expand_as(timestep).float(), rtol=0.0, atol=1.0e-6):
            raise ValueError("Router requires one canonical timestep per batch row")
        value = first
    else:
        raise ValueError("Canonical timestep must be [B], [B,1], or row-constant [B,L]")
    value = value.float()
    if not torch.isfinite(value).all().item() or ((value < 0.0) | (value > 1.0)).any().item():
        raise ValueError("Canonical timestep must be finite and normalized to [0,1]")
    return value


def effective_gates(q: torch.Tensor, style_strength: float | torch.Tensor) -> torch.Tensor:
    if q.ndim != 2 or not torch.isfinite(q).all().item():
        raise ValueError("q must be a finite [B,G] tensor")
    strength = torch.as_tensor(style_strength, device=q.device, dtype=q.dtype)
    if strength.ndim == 0:
        strength = strength.reshape(1, 1)
    elif strength.ndim == 1:
        strength = strength.unsqueeze(-1)
    elif strength.ndim != 2 or strength.shape[1] != 1:
        raise ValueError("style_strength must be scalar, [B], or [B,1]")
    if not torch.isfinite(strength).all().item() or ((strength < 0.0) | (strength > 1.0)).any().item():
        raise ValueError("style_strength must be finite and in [0,1]")
    if strength.shape[0] not in (1, q.shape[0]):
        raise ValueError("style_strength batch must be one or match q")
    if torch.all(strength == 0.5).item():
        return torch.ones_like(q)
    lower = (2.0 * strength).expand_as(q)
    upper = 1.0 + (2.0 * strength - 1.0) * q
    gates = torch.where(strength <= 0.5, lower, upper)
    if not torch.isfinite(gates).all().item():
        raise ValueError("Effective Phase C gates contain NaN or infinity")
    return gates


class PhaseCRouter:
    def __init__(self, bundle: PhaseCRouterBundle) -> None:
        self.bundle = bundle

    def encode_activator(self, projected_states: torch.Tensor) -> torch.Tensor:
        config = self.bundle.config
        expected = (config.activator_token_count, config.conditioning_dim)
        if projected_states.ndim != 3 or tuple(projected_states.shape[1:]) != expected:
            raise ValueError(f"Projected activator states must be [B,{expected[0]},{expected[1]}]")
        weights = self.bundle.weights
        device = projected_states.device
        normalized = F.layer_norm(
            projected_states.float(),
            (config.conditioning_dim,),
            weights.activator_norm_weight.to(device),
            weights.activator_norm_bias.to(device),
        )
        compressed = F.silu(
            F.linear(
                normalized,
                weights.activator_projection_weight.to(device),
                weights.activator_projection_bias.to(device),
            )
        )
        code = compressed.reshape(compressed.shape[0], config.activator_code_dim)
        if not torch.isfinite(code).all().item():
            raise ValueError("Phase C activator code contains NaN or infinity")
        return code

    def __call__(self, canonical_tau: torch.Tensor, activator_code: torch.Tensor) -> torch.Tensor:
        config = self.bundle.config
        if activator_code.ndim != 2 or activator_code.shape[1] != config.activator_code_dim:
            raise ValueError(f"Activator code must be [B,{config.activator_code_dim}]")
        tau = normalize_canonical_timestep(canonical_tau).reshape(-1)
        if tau.shape[0] == 1 and activator_code.shape[0] != 1:
            tau = tau.expand(activator_code.shape[0])
        if tau.shape[0] != activator_code.shape[0]:
            raise ValueError("Timestep batch must be one or match activator code")
        weights = self.bundle.weights
        device = activator_code.device
        context_in = weights.context_in.to(device)
        context_out = weights.context_out.to(device)
        universal = weights.universal_anchors.to(device)
        hidden = F.silu(torch.einsum("krc,bc->bkr", context_in, activator_code.float()))
        contextual = torch.einsum("kgr,bkr->bkg", context_out, hidden)
        position = tau * float(config.temporal_anchor_count - 1)
        left = torch.floor(position).long().clamp(0, config.temporal_anchor_count - 1)
        right = (left + 1).clamp(max=config.temporal_anchor_count - 1)
        alpha = (position - left.to(position.dtype)).unsqueeze(-1)
        rows = torch.arange(activator_code.shape[0], device=device)
        raw = (
            (1.0 - alpha) * (universal[left] + contextual[rows, left])
            + alpha * (universal[right] + contextual[rows, right])
        )
        q = config.q_max * torch.tanh(raw.float() / config.q_max)
        if not torch.isfinite(q).all().item() or (q.abs() > config.q_max + 1.0e-6).any().item():
            raise ValueError("Phase C router output is non-finite or outside q_max")
        return q


__all__ = [
    "PhaseCRouter",
    "aggregate_projected_occurrences",
    "effective_gates",
    "normalize_canonical_timestep",
]
