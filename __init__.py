"""
ComfyUI-gen2: Custom nodes for ComfyUI
General-purpose repository for custom nodes.
"""

# Import Pose nodes
try:
    from .pose import NODE_CLASS_MAPPINGS as POSE_NODE_CLASS_MAPPINGS
    from .pose import NODE_DISPLAY_NAME_MAPPINGS as POSE_NODE_DISPLAY_NAME_MAPPINGS
    POSE_AVAILABLE = True
except ImportError as e:
    print(f"[Gen2] Pose nodes not available: {e}")
    POSE_NODE_CLASS_MAPPINGS = {}
    POSE_NODE_DISPLAY_NAME_MAPPINGS = {}
    POSE_AVAILABLE = False

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

# Import Utility nodes
try:
    from .nodes import NODE_CLASS_MAPPINGS as UTILS_NODE_CLASS_MAPPINGS
    from .nodes import NODE_DISPLAY_NAME_MAPPINGS as UTILS_NODE_DISPLAY_NAME_MAPPINGS
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"[Gen2] Utility nodes not available: {e}")
    UTILS_NODE_CLASS_MAPPINGS = {}
    UTILS_NODE_DISPLAY_NAME_MAPPINGS = {}
    UTILS_AVAILABLE = False

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **POSE_NODE_CLASS_MAPPINGS,
    **QWENIMAGE_NODE_CLASS_MAPPINGS,
    **UTILS_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **POSE_NODE_DISPLAY_NAME_MAPPINGS,
    **QWENIMAGE_NODE_DISPLAY_NAME_MAPPINGS,
    **UTILS_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
