"""
Gen2 QwenImage Core - Model Definitions

Transformer blocks, control blocks, and control model matching VideoX architecture.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .imports import Attention, FeedForward
from .attention import QwenDoubleStreamAttnProcessor2_0, gen2_attention
from .rope import apply_rotary_emb_qwen, QwenEmbedRope


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
        img_shape: Tuple[int, int, int],
        txt_seq_lens: List[int],
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
            txt_seq_lens: List of actual text sequence lengths per batch item
        
        Returns:
            List of hint tensors, one per control layer
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Generate our own VideoX-compatible RoPE frequencies
        img_shapes = [[img_shape]] * batch_size
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
            kwargs["encoder_hidden_states"] = encoder_hidden_states
        
        # Extract hints (all but last element, which is final state)
        hints = list(torch.unbind(c))[:-1]
        
        return hints

