"""
ComfyUI-gen2: Custom nodes for ComfyUI
General-purpose repository for custom nodes.
"""

# Import QwenImage nodes
try:
    from .qwenimage import NODE_CLASS_MAPPINGS as QWENIMAGE_NODE_CLASS_MAPPINGS
    from .qwenimage import NODE_DISPLAY_NAME_MAPPINGS as QWENIMAGE_NODE_DISPLAY_NAME_MAPPINGS
    QWENIMAGE_AVAILABLE = True
except ImportError as e:
    print(f"[Gen2] QwenImage nodes not available: {e}")
    QWENIMAGE_NODE_CLASS_MAPPINGS = {}
    QWENIMAGE_NODE_DISPLAY_NAME_MAPPINGS = {}
    QWENIMAGE_AVAILABLE = False

# Import Misc nodes (string utilities, pose, etc.)
try:
    from .misc_nodes import NODE_CLASS_MAPPINGS as MISC_NODE_CLASS_MAPPINGS
    from .misc_nodes import NODE_DISPLAY_NAME_MAPPINGS as MISC_NODE_DISPLAY_NAME_MAPPINGS
    MISC_AVAILABLE = True
except ImportError as e:
    print(f"[Gen2] Misc nodes not available: {e}")
    MISC_NODE_CLASS_MAPPINGS = {}
    MISC_NODE_DISPLAY_NAME_MAPPINGS = {}
    MISC_AVAILABLE = False

# Import Tiling nodes (tile splitter, masks, merger)
try:
    from .tiling import NODE_CLASS_MAPPINGS as TILING_NODE_CLASS_MAPPINGS
    from .tiling import NODE_DISPLAY_NAME_MAPPINGS as TILING_NODE_DISPLAY_NAME_MAPPINGS
    TILING_AVAILABLE = True
except ImportError as e:
    print(f"[Gen2] Tiling nodes not available: {e}")
    TILING_NODE_CLASS_MAPPINGS = {}
    TILING_NODE_DISPLAY_NAME_MAPPINGS = {}
    TILING_AVAILABLE = False

# Import Gen2 Sampling nodes (Flux.2 klein fix)
try:
    from .gen2_sampling import NODE_CLASS_MAPPINGS as SAMPLING_NODE_CLASS_MAPPINGS
    from .gen2_sampling import NODE_DISPLAY_NAME_MAPPINGS as SAMPLING_NODE_DISPLAY_NAME_MAPPINGS
    SAMPLING_AVAILABLE = True
except ImportError as e:
    print(f"[Gen2] Sampling nodes not available: {e}")
    SAMPLING_NODE_CLASS_MAPPINGS = {}
    SAMPLING_NODE_DISPLAY_NAME_MAPPINGS = {}
    SAMPLING_AVAILABLE = False

# Import Gen2 API nodes (V3 io.ComfyNode: Input Panel / Output Panel)
try:
    from .api_nodes import NODE_CLASS_MAPPINGS as API_NODE_CLASS_MAPPINGS
    from .api_nodes import NODE_DISPLAY_NAME_MAPPINGS as API_NODE_DISPLAY_NAME_MAPPINGS
    API_AVAILABLE = True
except ImportError as e:
    print(f"[Gen2] API panel nodes not available: {e}")
    API_NODE_CLASS_MAPPINGS = {}
    API_NODE_DISPLAY_NAME_MAPPINGS = {}
    API_AVAILABLE = False

# Import native-style Flux.2 Fun ControlNet nodes
try:
    from .flux2_fun import NODE_CLASS_MAPPINGS as FLUX2_FUN_NODE_CLASS_MAPPINGS
    from .flux2_fun import NODE_DISPLAY_NAME_MAPPINGS as FLUX2_FUN_NODE_DISPLAY_NAME_MAPPINGS
    FLUX2_FUN_AVAILABLE = True
except ImportError as e:
    print(f"[Gen2] Flux2 Fun ControlNet nodes not available: {e}")
    FLUX2_FUN_NODE_CLASS_MAPPINGS = {}
    FLUX2_FUN_NODE_DISPLAY_NAME_MAPPINGS = {}
    FLUX2_FUN_AVAILABLE = False

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **QWENIMAGE_NODE_CLASS_MAPPINGS,
    **MISC_NODE_CLASS_MAPPINGS,
    **TILING_NODE_CLASS_MAPPINGS,
    **SAMPLING_NODE_CLASS_MAPPINGS,
    **API_NODE_CLASS_MAPPINGS,
    **FLUX2_FUN_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **QWENIMAGE_NODE_DISPLAY_NAME_MAPPINGS,
    **MISC_NODE_DISPLAY_NAME_MAPPINGS,
    **TILING_NODE_DISPLAY_NAME_MAPPINGS,
    **SAMPLING_NODE_DISPLAY_NAME_MAPPINGS,
    **API_NODE_DISPLAY_NAME_MAPPINGS,
    **FLUX2_FUN_NODE_DISPLAY_NAME_MAPPINGS,
}

# Frontend extension: the Configure popup + dynamic slot management for the
# Input/Output panels. Loaded by the browser from /extensions/ComfyUI-gen2/.
WEB_DIRECTORY = "./web/js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
