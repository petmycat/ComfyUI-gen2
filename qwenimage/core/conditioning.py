"""
Gen2 QwenImage Core - Conditioning Utilities

Helper classes and functions for handling ComfyUI CONDITIONING and config compatibility.
"""

import torch


class Gen2TransformerConfig:
    """
    Simple config object that mimics VideoX's transformer.config structure.
    Used for compatibility with VideoX pipeline which accesses config attributes.
    """
    def __init__(self, in_channels=64, out_channels=16, guidance_embeds=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.guidance_embeds = guidance_embeds
    
    def get(self, key, default=None):
        return getattr(self, key, default)


def extract_from_conditioning(conditioning):
    """
    Extract prompt_embeds and attention_mask from ComfyUI CONDITIONING.
    
    ComfyUI CONDITIONING format:
    [
        (
            encoder_hidden_states,  # [batch, seq_len, hidden_dim]
            {
                "pooled_output": ...,     # optional
                "attention_mask": ...,    # [batch, seq_len] - optional
                ...
            }
        )
    ]
    
    Returns:
        prompt_embeds: Tensor [batch, seq_len, hidden_dim]
        prompt_embeds_mask: Tensor [batch, seq_len] or None
    """
    if conditioning is None or len(conditioning) == 0:
        return None, None
    
    # Get the first conditioning entry
    cond_entry = conditioning[0]
    
    # Extract encoder_hidden_states (prompt_embeds)
    prompt_embeds = cond_entry[0]
    
    # Extract attention_mask from the dict (if available)
    cond_dict = cond_entry[1] if len(cond_entry) > 1 else {}
    prompt_embeds_mask = cond_dict.get("attention_mask", None)
    
    # If no mask provided, create one with all ones (all tokens valid)
    if prompt_embeds_mask is None:
        batch_size, seq_len = prompt_embeds.shape[:2]
        prompt_embeds_mask = torch.ones(batch_size, seq_len, device=prompt_embeds.device, dtype=torch.long)
    
    return prompt_embeds, prompt_embeds_mask


def get_txt_seq_len_from_mask(attention_mask):
    """
    Get actual text sequence length from attention mask.
    
    The attention mask has 1s for valid tokens and 0s for padding.
    The actual sequence length is the sum of 1s.
    
    Args:
        attention_mask: Tensor [batch, seq_len] with 1s and 0s
    
    Returns:
        List of sequence lengths for each batch item
    """
    if attention_mask is None:
        return None
    
    # Sum along sequence dimension to get valid token count per batch
    seq_lens = attention_mask.sum(dim=-1).tolist()
    
    # Handle single batch case
    if isinstance(seq_lens, (int, float)):
        seq_lens = [int(seq_lens)]
    else:
        seq_lens = [int(x) for x in seq_lens]
    
    return seq_lens

