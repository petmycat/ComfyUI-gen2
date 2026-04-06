"""
Gen2 QwenImage Core - Model Wrappers

Wrappers that use ComfyUI's model weights but apply VideoX's exact forward logic.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from .rope import apply_rotary_emb_qwen, QwenEmbedRope
from .attention import gen2_attention
from .conditioning import Gen2TransformerConfig


# =============================================================================
# Block Wrapper - VideoX forward logic with ComfyUI weights
# =============================================================================

class Gen2QwenImageTransformerBlockWrapper:
    """
    Wrapper that uses ComfyUI's transformer block WEIGHTS but applies VideoX's forward LOGIC.
    
    This gives us:
    1. ComfyUI's fast model loading (weights are shared, not copied)
    2. VideoX's exact RoPE calculation and attention processing
    3. Full control over the forward pass
    """
    
    def __init__(self, comfyui_block):
        """
        Args:
            comfyui_block: A ComfyUI QwenImageTransformerBlock with loaded weights
        """
        # Reference ComfyUI block's weight tensors directly (shared memory, no copy)
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
        """Apply modulation to input tensor (VideoX style)."""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
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
        
        Returns:
            (encoder_hidden_states, hidden_states) - text and image outputs
        """
        # Get modulation parameters
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)
        
        # Split into two sets (for norm1 and norm2)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
        
        # === Attention block ===
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)
        
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
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)
        
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
# Model Wrapper - VideoX-compatible interface for ComfyUI models
# =============================================================================

class Gen2QwenImageModelWrapper:
    """
    Wrapper that provides VideoX-compatible forward interface for ComfyUI's QwenImage model.
    
    Uses ComfyUI's fast model loading (weights are shared, not copied),
    wraps each transformer block with Gen2QwenImageTransformerBlockWrapper,
    uses VideoX's QwenEmbedRope for exact RoPE calculation,
    and operates on PACKED 3D latents throughout, matching VideoX exactly.
    """
    
    def __init__(self, comfyui_model, control_model=None, latent_height=None, latent_width=None, control_layers=None):
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
        # CRITICAL: Must use scale_rope=True to match VideoX's training!
        self.rope_embedder = QwenEmbedRope(
            theta=10000,
            axes_dim=[16, 56, 56],
            scale_rope=True,
        )
        
        # Create a config object that mimics VideoX's transformer config
        self._config = Gen2TransformerConfig(
            in_channels=64,
            out_channels=self.out_channels,
            guidance_embeds=False,
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
        Matches VideoX's QwenImageControlTransformer2DModel.forward() signature exactly.
        """
        model = self.model
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Ensure block wrappers are created
        self._ensure_block_wrappers()
        
        # Handle list format for CFG batching
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = torch.stack([t.squeeze(0) if t.dim() > 2 else t for t in encoder_hidden_states])
        
        if isinstance(encoder_hidden_states_mask, list):
            encoder_hidden_states_mask = torch.stack([t.squeeze(0) if t.dim() > 1 else t for t in encoder_hidden_states_mask])
        
        # Get image shape for RoPE generation
        if img_shapes and len(img_shapes) > 0 and len(img_shapes[0]) > 0:
            num_frames, h_patches, w_patches = img_shapes[0][0]
        else:
            h_patches = self.latent_height // 2
            w_patches = self.latent_width // 2
            num_frames = 1
        
        # Use VideoX's QwenEmbedRope for exact RoPE calculation
        img_shape = (num_frames, h_patches, w_patches)
        
        # CRITICAL: Use the passed txt_seq_lens parameter (actual token counts from mask.sum())
        if txt_seq_lens is not None and len(txt_seq_lens) > 0:
            rope_txt_seq_lens = txt_seq_lens
        else:
            rope_txt_seq_lens = [encoder_hidden_states.shape[1]] * batch_size
        
        # Pass img_shapes directly to rope_embedder (VideoX format)
        if img_shapes and len(img_shapes) > 0:
            rope_img_shapes = img_shapes
        else:
            rope_img_shapes = [[(num_frames, h_patches, w_patches)]]
        
        # Debug: Log RoPE parameters (only on first call)
        if not hasattr(self, '_rope_debug_logged'):
            self._rope_debug_logged = True
            print(f"[Gen2 RoPE] txt_seq_lens: {rope_txt_seq_lens} (actual token counts)")
            print(f"[Gen2 RoPE] encoder_hidden_states shape: {encoder_hidden_states.shape} (padded)")
            print(f"[Gen2 RoPE] img_shapes format: {rope_img_shapes[0] if rope_img_shapes else 'N/A'}")
        
        # QwenEmbedRope returns (vid_freqs, txt_freqs) tuple
        image_rotary_emb = self.rope_embedder(rope_img_shapes, rope_txt_seq_lens, device)
        
        # Process inputs through embeddings (use ComfyUI's embedding layers)
        hidden_states = model.img_in(hidden_states)
        
        # Convert timestep to hidden_states dtype AFTER img_in (VideoX exact match)
        timestep = timestep.to(hidden_states.dtype)
        
        encoder_hidden_states_processed = model.txt_norm(encoder_hidden_states)
        encoder_hidden_states_processed = model.txt_in(encoder_hidden_states_processed)
        
        # Compute timestep embedding
        temb = model.time_text_embed(timestep, hidden_states, None)
        
        # Generate control hints if control model is available
        control_hints = None
        if self.control_model is not None and control_context is not None:
            self.control_model.to(device=device, dtype=dtype)
            
            control_hints = self.control_model.forward_control(
                x=hidden_states,
                control_context=control_context,
                encoder_hidden_states=encoder_hidden_states_processed,
                encoder_hidden_states_mask=None,
                temb=temb,
                img_shape=img_shape,
                txt_seq_lens=rope_txt_seq_lens,
            )
        
        # Prepare control layers mapping
        control_layer_set = set(self.control_layers) if self.control_layers else set()
        hint_idx = 0
        
        # Run through OUR block wrappers (ComfyUI weights + VideoX forward logic)
        for i, block_wrapper in enumerate(self._block_wrappers):
            encoder_hidden_states_processed, hidden_states = block_wrapper(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states_processed,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            
            # Inject control hints at specified layers (VideoX style)
            if control_hints is not None and i in control_layer_set and hint_idx < len(control_hints):
                hint = control_hints[hint_idx]
                hidden_states = hidden_states + control_context_scale * hint
                hint_idx += 1
        
        # Final normalization and projection (use ComfyUI's layers)
        hidden_states = model.norm_out(hidden_states, temb)
        output = model.proj_out(hidden_states)
        
        # Return packed 3D output (NO unpacking here - matches VideoX)
        return output
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

