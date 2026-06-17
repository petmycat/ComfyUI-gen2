"""
Gen2 QwenImage - Node Registration

Maps all QwenImage node classes to ComfyUI's node registry format.
"""

from .qwen_nodes import (
    Gen2_LoadQwenControlNetFun,
    Gen2_LoadQwenVAE,
    Gen2_ApplyQwenControlNetFun,
    Gen2_QwenClipTextEncode,
    Gen2_LoadQwenLora,
    Gen2_QwenImageControlSampler,
)


NODE_CLASS_MAPPINGS = {
    "Gen2_LoadQwenControlNetFun": Gen2_LoadQwenControlNetFun,
    "Gen2_LoadQwenVAE": Gen2_LoadQwenVAE,
    "Gen2_ApplyQwenControlNetFun": Gen2_ApplyQwenControlNetFun,
    "Gen2_QwenClipTextEncode": Gen2_QwenClipTextEncode,
    "Gen2_LoadQwenLora": Gen2_LoadQwenLora,
    "Gen2_QwenImageControlSampler": Gen2_QwenImageControlSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_LoadQwenControlNetFun": "Gen2 Load QwenImage ControlNet (outdated)",
    "Gen2_LoadQwenVAE": "Gen2 Load QwenImage VAE (outdated)",
    "Gen2_ApplyQwenControlNetFun": "Gen2 Apply QwenImage ControlNet (outdated)",
    "Gen2_QwenClipTextEncode": "Gen2 QwenImage Text Encode (outdated)",
    "Gen2_LoadQwenLora": "Gen2 Load QwenImage LoRA (outdated)",
    "Gen2_QwenImageControlSampler": "Gen2 QwenImage Control Sampler (outdated)",
}

