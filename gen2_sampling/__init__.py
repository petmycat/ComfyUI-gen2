"""
Gen2 Sampling Package

Flux.2 [klein] compatibility fix for ComfyUI >= v0.17.0.
"""

from .flux2_klein_fix import Gen2_Flux2KleinFix
from .ideogram4_aitk_lora import Gen2_Ideogram4AITKLoRALoader
from .lanpaint_soft_denoise import Gen2_LanPaintSoftDenoisePatch

NODE_CLASS_MAPPINGS = {
    "Gen2_Flux2KleinFix": Gen2_Flux2KleinFix,
    "Gen2_Ideogram4AITKLoRALoader": Gen2_Ideogram4AITKLoRALoader,
    "Gen2_LanPaintSoftDenoisePatch": Gen2_LanPaintSoftDenoisePatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_Flux2KleinFix": "Gen2 Flux.2 Klein Fix (#12905 revert)",
    "Gen2_Ideogram4AITKLoRALoader": "Ideogram4 AI Toolkit LoRA Loader",
    "Gen2_LanPaintSoftDenoisePatch": "Gen2 LanPaint Soft Denoise Patch",
}
