"""
Gen2 Sampling Package

Flux.2 [klein] compatibility fix for ComfyUI >= v0.17.0.
"""

from .flux2_klein_fix import Gen2_Flux2KleinFix

NODE_CLASS_MAPPINGS = {
    "Gen2_Flux2KleinFix": Gen2_Flux2KleinFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_Flux2KleinFix": "Gen2 Flux.2 Klein Fix (#12905 revert)",
}
