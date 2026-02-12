"""
Gen2 QwenImage Core - Rotary Position Embedding (RoPE)

Implements VideoX's QwenEmbedRope and apply_rotary_emb_qwen exactly.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


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

