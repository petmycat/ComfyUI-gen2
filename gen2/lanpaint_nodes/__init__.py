"""Gen2 LanPaint integration."""

from .lanpaint_soft_denoise import Gen2_LanPaintSoftDenoisePatch

NODE_CLASS_MAPPINGS = {
    "Gen2_LanPaintSoftDenoisePatch": Gen2_LanPaintSoftDenoisePatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_LanPaintSoftDenoisePatch": "Gen2 LanPaint Soft Denoise Patch",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
