"""
Gen2 QwenImage Core - VideoX and Diffusers Import Setup

Sets up sys.path for VideoX-Fun imports and checks for diffusers availability.
This module should be imported first in the core package.
"""

import os
import sys


def _setup_videox_imports():
    """
    Add videox-fun custom node to sys.path so we can import videox_fun modules.
    VideoX is installed as a ComfyUI custom node, not a pip package.
    """
    # Find the custom_nodes directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate up: core -> qwenimage -> ComfyUI-gen2 -> custom_nodes
    custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    # Path to videox-fun
    videox_path = os.path.join(custom_nodes_dir, "videox-fun")
    
    if os.path.exists(videox_path) and videox_path not in sys.path:
        sys.path.insert(0, videox_path)
        return True
    
    # Try alternative names
    for name in ["videox_fun", "VideoX-Fun", "ComfyUI-VideoX-Fun"]:
        alt_path = os.path.join(custom_nodes_dir, name)
        if os.path.exists(alt_path) and alt_path not in sys.path:
            sys.path.insert(0, alt_path)
            return True
    
    return False


# Setup videox imports on module load
_setup_videox_imports()

# Check diffusers availability
try:
    from diffusers.models.attention import Attention, FeedForward
    from diffusers.models.normalization import RMSNorm
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.image_processor import VaeImageProcessor
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    Attention = None
    FeedForward = None
    RMSNorm = None
    FlowMatchEulerDiscreteScheduler = None
    VaeImageProcessor = None
    print("[Gen2] Warning: diffusers not available. QwenImage nodes will not work.")

