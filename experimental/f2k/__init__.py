"""Experimental Flux.2 Klein compatibility fix."""

from .flux2_klein_fix import Gen2_Flux2KleinFix

NODE_CLASS_MAPPINGS = {
    "Gen2_Flux2KleinFix": Gen2_Flux2KleinFix,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_Flux2KleinFix": "Gen2 Flux.2 Klein Fix (#12905 revert)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
