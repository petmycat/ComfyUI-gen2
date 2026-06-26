"""
Gen2 API Nodes - V3-style configurable input/output panels for API-driven workflows.

These are V3 io.ComfyNode classes. To stay compatible with the gen2 pack's
existing V1 __init__.py (which merges NODE_CLASS_MAPPINGS from all sub-packages
and is detected by ComfyUI's loader via the NODE_CLASS_MAPPINGS branch), we also
expose them through NODE_CLASS_MAPPINGS here. The server's node_info() detects
V3 classes (issubclass _ComfyNodeInternal) and downgrades them to V1 dicts for
the frontend automatically, so no comfy_entrypoint is needed at the pack level.

We still expose comfy_entrypoint() for Comfy Registry / future V3-only loading.
"""

from .input_panel import Gen2InputPanel
from .output_panel import Gen2OutputPanel

NODE_CLASS_MAPPINGS = {
    "Gen2_InputPanel": Gen2InputPanel,
    "Gen2_OutputPanel": Gen2OutputPanel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_InputPanel": "Gen2 Input Panel",
    "Gen2_OutputPanel": "Gen2 Output Panel",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
