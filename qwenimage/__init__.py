"""
Gen2 QwenImage Module
ComfyUI Custom Nodes for QwenImage ControlNet (VideoX Fun 2026 architecture)

These nodes integrate with ComfyUI's NATIVE ControlNet system:
- Use standard UnetLoader for QwenImage base model
- Use standard VAELoader, CLIPLoader  
- Use Gen2_LoadQwenControlNetFun to load VideoX Fun's ControlNet weights
- Use Gen2_ApplyQwenControlNetFun to apply ControlNet to CONDITIONING
- Use standard KSampler for sampling

Architecture Notes:
- VideoX Fun ControlNet generates hints at sparse layers: [0, 12, 24, 36, 48]
- Control input: VAE-encoded control image + mask + inpaint = 33 channels
- After packing: 33 * 4 = 132 features per sequence position
- Integrates with ComfyUI's native control infrastructure (no custom sampler needed)
"""

# ControlNet Fun (VideoX 2026 architecture)
try:
    from .qwenimage_controlnet_fun import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError as e:
    print(f"[Gen2] QwenImage ControlNet Fun nodes not available: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
