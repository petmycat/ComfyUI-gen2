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

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **QWENIMAGE_NODE_CLASS_MAPPINGS,
    **MISC_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **QWENIMAGE_NODE_DISPLAY_NAME_MAPPINGS,
    **MISC_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
