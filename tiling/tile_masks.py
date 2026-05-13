"""
Gen2 Tile Masks - Generate per-tile inpaint masks from a tile layout.

For each tile in the layout:
- The mask frame matches the tile image (expanded size, base + 2*overlap on each axis)
- A rectangle of value 1.0 covers the tile's base region (its "owned" slice)
- The surrounding halo is 0.0
- mask_blend_pixels controls a symmetric Gaussian feather across the base/halo
  boundary, with total transition band width ~= mask_blend_pixels pixels.

Output order matches the order of tiles produced by Gen2_TileSplitter, so the
mask at index i belongs to the tile at index i.
"""

import torch
import scipy.ndimage


class Gen2_TileMasks:
    """
    Build per-tile masks that select each tile's owned base region.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_layout": ("GEN2_TILE_LAYOUT",),
                "mask_blend_pixels": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "make_masks"
    CATEGORY = "Gen2/Tiling"

    def make_masks(self, tile_layout: dict, mask_blend_pixels: int):
        if not isinstance(tile_layout, dict) or "splits" not in tile_layout:
            raise ValueError(
                "Gen2_TileMasks: tile_layout is not a valid GEN2_TILE_LAYOUT (missing 'splits')."
            )

        splits = tile_layout["splits"]
        blend_px = int(mask_blend_pixels)
        sigma = blend_px / 2.0 if blend_px > 0 else 0.0

        masks = []

        for s in splits:
            h = int(s["h"])
            w = int(s["w"])
            by = int(s["base_y_local"])
            bx = int(s["base_x_local"])
            bh = int(s["base_h_local"])
            bw = int(s["base_w_local"])

            mask = torch.zeros((1, h, w), dtype=torch.float32)
            mask[:, by : by + bh, bx : bx + bw] = 1.0

            if sigma > 0.0:
                mask_np = mask[0].numpy()
                mask_np = scipy.ndimage.gaussian_filter(
                    mask_np, sigma=sigma, mode="constant", cval=0.0
                )
                mask = torch.from_numpy(mask_np).unsqueeze(0).clamp(0.0, 1.0)

            masks.append(mask)

        return (masks,)


NODE_CLASS_MAPPINGS = {
    "Gen2_TileMasks": Gen2_TileMasks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_TileMasks": "Gen2 Tile Masks",
}
