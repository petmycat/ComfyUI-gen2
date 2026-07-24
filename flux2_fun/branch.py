from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .types import CheckpointProfile


def _cast_modulation(value, dtype: torch.dtype, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    if hasattr(value, "shift") and hasattr(value, "scale") and hasattr(value, "gate"):
        value_type = type(value)
        return value_type(
            shift=value.shift.to(device=device, dtype=dtype),
            scale=value.scale.to(device=device, dtype=dtype),
            gate=value.gate.to(device=device, dtype=dtype),
        )
    if isinstance(value, tuple):
        return tuple(_cast_modulation(item, dtype, device) for item in value)
    return value


def _apply_modulation(tensor: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor | None, modulation_dims=None) -> torch.Tensor:
    if modulation_dims is None:
        result = tensor * scale
        return result + shift if shift is not None else result
    result = tensor.clone()
    for token_start, token_end, batch_index in modulation_dims:
        result[:, token_start:token_end] *= scale[:, batch_index:batch_index + 1]
        if shift is not None:
            result[:, token_start:token_end] += shift[:, batch_index:batch_index + 1]
    return result


class Flux2FunSwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, linear = value.chunk(2, dim=-1)
        return self.gate_fn(gate) * linear


class Flux2FunFeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int, operations, *, dtype=None, device=None) -> None:
        super().__init__()
        self.linear_in = operations.Linear(dim, inner_dim * 2, bias=False, dtype=dtype, device=device)
        self.act_fn = Flux2FunSwiGLU()
        self.linear_out = operations.Linear(inner_dim, dim, bias=False, dtype=dtype, device=device)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.act_fn(self.linear_in(value)))


class Flux2FunAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, operations, *, dtype=None, device=None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.to_k = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.to_v = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.add_q_proj = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.add_k_proj = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.add_v_proj = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
        self.norm_q = operations.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(head_dim, eps=1e-6, dtype=dtype, device=device)
        self.to_out = nn.ModuleList((operations.Linear(dim, dim, bias=False, dtype=dtype, device=device), nn.Dropout(0.0)))
        self.to_add_out = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pe: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        transformer_options: dict,
        attention_fn: Callable,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_q = self.norm_q(self._heads(self.to_q(hidden_states)))
        image_k = self.norm_k(self._heads(self.to_k(hidden_states)))
        image_v = self._heads(self.to_v(hidden_states))
        text_q = self.norm_added_q(self._heads(self.add_q_proj(encoder_hidden_states)))
        text_k = self.norm_added_k(self._heads(self.add_k_proj(encoder_hidden_states)))
        text_v = self._heads(self.add_v_proj(encoder_hidden_states))

        query = torch.cat((text_q, image_q), dim=2)
        key = torch.cat((text_k, image_k), dim=2)
        value = torch.cat((text_v, image_v), dim=2)
        attended = attention_fn(query, key, value, pe=pe, mask=attn_mask, transformer_options=transformer_options)
        text_length = encoder_hidden_states.shape[1]
        text_output, image_output = attended[:, :text_length], attended[:, text_length:]
        return self.to_out[0](image_output), self.to_add_out(text_output)


class Flux2FunControlBlock(nn.Module):
    def __init__(self, profile: CheckpointProfile, block_id: int, operations, *, dtype=None, device=None) -> None:
        super().__init__()
        dim = profile.hidden_size
        self.block_id = block_id
        self.norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.norm1_context = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.attn = Flux2FunAttention(dim, profile.num_heads, profile.head_dim, operations, dtype=dtype, device=device)
        self.norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.ff = Flux2FunFeedForward(dim, profile.mlp_hidden_dim, operations, dtype=dtype, device=device)
        self.norm2_context = operations.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.ff_context = Flux2FunFeedForward(dim, profile.mlp_hidden_dim, operations, dtype=dtype, device=device)
        if block_id == 0:
            self.before_proj = operations.Linear(dim, dim, bias=True, dtype=dtype, device=device)
        self.after_proj = operations.Linear(dim, dim, bias=True, dtype=dtype, device=device)

    def forward(
        self,
        control_state: torch.Tensor,
        base_image_state: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        vec,
        pe: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        transformer_options: dict,
        attention_fn: Callable,
        modulation_dims_img=None,
        modulation_dims_txt=None,
        capture: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.block_id == 0:
            projected_control = self.before_proj(control_state)
            if capture is not None:
                capture["block_0_before_proj"] = projected_control.detach()
            control_state = projected_control + base_image_state

        (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec
        image_norm = _apply_modulation(self.norm1(control_state), 1 + img_mod1.scale, img_mod1.shift, modulation_dims_img)
        text_norm = _apply_modulation(self.norm1_context(encoder_hidden_states), 1 + txt_mod1.scale, txt_mod1.shift, modulation_dims_txt)
        image_attn, text_attn = self.attn(
            image_norm,
            text_norm,
            pe,
            attn_mask,
            transformer_options,
            attention_fn,
        )
        control_state = control_state + _apply_modulation(image_attn, img_mod1.gate, None, modulation_dims_img)
        image_ff_input = _apply_modulation(self.norm2(control_state), 1 + img_mod2.scale, img_mod2.shift, modulation_dims_img)
        control_state = control_state + _apply_modulation(self.ff(image_ff_input), img_mod2.gate, None, modulation_dims_img)

        encoder_hidden_states = encoder_hidden_states + _apply_modulation(text_attn, txt_mod1.gate, None, modulation_dims_txt)
        text_ff_input = _apply_modulation(
            self.norm2_context(encoder_hidden_states), 1 + txt_mod2.scale, txt_mod2.shift, modulation_dims_txt
        )
        encoder_hidden_states = encoder_hidden_states + _apply_modulation(
            self.ff_context(text_ff_input), txt_mod2.gate, None, modulation_dims_txt
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = torch.nan_to_num(encoder_hidden_states, nan=0.0, posinf=65504, neginf=-65504)
        return encoder_hidden_states, control_state, self.after_proj(control_state)


class Flux2FunControlBranch(nn.Module):
    def __init__(
        self,
        profile: CheckpointProfile,
        operations,
        *,
        dtype=None,
        device=None,
        compute_dtype=None,
        attention_fn: Callable | None = None,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.compute_dtype = compute_dtype or dtype
        self.control_img_in = operations.Linear(profile.control_dim, profile.hidden_size, bias=True, dtype=dtype, device=device)
        self.control_transformer_blocks = nn.ModuleList(
            Flux2FunControlBlock(profile, block_id, operations, dtype=dtype, device=device)
            for block_id in range(profile.block_count)
        )
        if attention_fn is None:
            from comfy.ldm.flux.math import attention as attention_fn
        self.attention_fn = attention_fn

    def forward_control(
        self,
        base_image_state: torch.Tensor,
        control_context: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        vec,
        pe: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
        transformer_options: dict | None = None,
        modulation_dims_img=None,
        modulation_dims_txt=None,
        capture: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        transformer_options = transformer_options or {}
        compute_dtype = self.compute_dtype or control_context.dtype
        context = control_context.to(device=base_image_state.device, dtype=compute_dtype)
        image_state = base_image_state.to(dtype=compute_dtype)
        text_state = encoder_hidden_states.to(dtype=compute_dtype)
        compute_vec = _cast_modulation(vec, compute_dtype, base_image_state.device)
        control_state = self.control_img_in(context)
        if capture is not None:
            capture["control_img_in"] = control_state.detach()

        hints = []
        for block_id, block in enumerate(self.control_transformer_blocks):
            text_state, control_state, hint = block(
                control_state,
                image_state,
                text_state,
                compute_vec,
                pe,
                attn_mask,
                transformer_options,
                self.attention_fn,
                modulation_dims_img=modulation_dims_img,
                modulation_dims_txt=modulation_dims_txt,
                capture=capture,
            )
            hints.append(hint)
            if capture is not None:
                capture[f"block_{block_id}_output"] = control_state.detach()
                capture[f"hint_{block_id}"] = hint.detach()
        return tuple(hints)

    def forward_one_block(self, block_id: int, *args, **kwargs):
        if not 0 <= block_id < len(self.control_transformer_blocks):
            raise IndexError(f"Flux2 Fun control block index out of range: {block_id}")
        return self.control_transformer_blocks[block_id](*args, attention_fn=self.attention_fn, **kwargs)
