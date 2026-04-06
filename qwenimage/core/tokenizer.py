"""
Gen2 QwenImage Core - Tokenizer Utilities

Loads and caches the HuggingFace Qwen2 tokenizer for VideoX-compatible text encoding.
"""

import os

import folder_paths


# VideoX encoding constants
VIDEOX_PROMPT_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
VIDEOX_DROP_IDX = 34  # Number of template tokens to drop (fixed, unlike ComfyUI's dynamic calculation)
VIDEOX_TOKENIZER_MAX_LENGTH = 1024

# Global tokenizer cache
_gen2_tokenizer = None


def get_gen2_tokenizer():
    """Load and cache our custom HuggingFace tokenizer."""
    global _gen2_tokenizer
    if _gen2_tokenizer is None:
        from transformers import Qwen2Tokenizer
        # Path to our tokenizer (relative to ComfyUI root)
        tokenizer_path = os.path.join(folder_paths.models_dir, "gen2", "qwen_2512_tokenizer")
        if os.path.exists(tokenizer_path):
            _gen2_tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_path)
            print(f"[Gen2] Loaded custom tokenizer from: {tokenizer_path}")
        else:
            # Fallback: try to load from HuggingFace
            try:
                _gen2_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
                print(f"[Gen2] Loaded tokenizer from HuggingFace (fallback)")
            except Exception as e:
                raise RuntimeError(f"[Gen2] Failed to load tokenizer: {e}\nExpected path: {tokenizer_path}")
    return _gen2_tokenizer

