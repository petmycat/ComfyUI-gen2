"""
Gen2 Miscellaneous Nodes - String utilities, Pose detection, etc.
"""

from .string_replace import NODE_CLASS_MAPPINGS as STRING_NODES, NODE_DISPLAY_NAME_MAPPINGS as STRING_NAMES

# Pose nodes depend on comfyui_controlnet_aux; import gracefully
try:
    from .pose import NODE_CLASS_MAPPINGS as POSE_NODES, NODE_DISPLAY_NAME_MAPPINGS as POSE_NAMES
except Exception as e:
    print(f"[Gen2] Warning: Could not load pose nodes: {e}")
    POSE_NODES = {}
    POSE_NAMES = {}

NODE_CLASS_MAPPINGS = {**STRING_NODES, **POSE_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**STRING_NAMES, **POSE_NAMES}
