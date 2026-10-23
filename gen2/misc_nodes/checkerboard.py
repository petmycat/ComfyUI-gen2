import torch
import numpy as np


class Gen2_Checkerboard:
    """Generate a checkerboard pattern image with configurable block size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "block_size": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Gen2/Utils"

    def generate(self, width: int, height: int, block_size: int) -> tuple:
        # Create checkerboard: each block is block_size x block_size pixels
        rows = np.arange(height)
        cols = np.arange(width)
        grid = (cols[None, :] // block_size + rows[:, None] // block_size) % 2  # shape: (H, W), values 0 or 1
        # Expand to 3-channel RGB and convert to float32 [0, 1]
        checkerboard = np.stack([grid, grid, grid], axis=-1).astype(np.float32)
        # ComfyUI IMAGE format: (B, H, W, C) float32
        image = torch.from_numpy(checkerboard).unsqueeze(0)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "Gen2_Checkerboard": Gen2_Checkerboard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_Checkerboard": "Gen2 Checkerboard",
}

