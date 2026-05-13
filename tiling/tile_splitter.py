"""
Gen2 Tile Splitter - Auto-grid splitting of an image into tiles with a fixed-
thickness overlap halo around each tile's "owned" base region.

Partition strategy:
- n_rows = argmin_{n>=1, n<=MAX} |H/n - tile_size|
- n_cols = same for width
- base_h_floor = H // n_rows, h_remainder = H - n_rows * base_h_floor
- First (n_rows - 1) rows use base_h = base_h_floor; the last row absorbs the
  remainder ("last_tile_wins"), so its base_h is base_h_floor + h_remainder.
  Same logic for columns.
- The union of every tile's base region exactly tiles the original image with no
  overlap and no gap.

Overlap halo:
- overlap_px_h = round(overlap_pct * base_h_floor) (single value for the grid)
- overlap_px_w = round(overlap_pct * base_w_floor)
- Each tile's expanded crop = base region expanded by overlap_px on all 4 sides
- Where the expansion falls outside the image, content is replicate-padded
- Result: every interior tile has expanded size
  (base_h_floor + 2*overlap_px_h) x (base_w_floor + 2*overlap_px_w);
  edge tiles (last row/col) are up to (n - 1) pixels larger in the base axis.
"""

import torch
import torch.nn.functional as F


MAX_TILES_PER_DIM = 32


def _best_n(dim: int, target: int) -> int:
    """Pick n in [1, MAX_TILES_PER_DIM] that minimizes |dim/n - target|."""
    if dim <= target:
        return 1
    best_n = 1
    best_dist = abs(dim - target)
    for n in range(2, MAX_TILES_PER_DIM + 1):
        actual = dim / n
        dist = abs(actual - target)
        if dist < best_dist:
            best_n = n
            best_dist = dist
        if actual < target * 0.5:
            break
    return best_n


class Gen2_TileSplitter:
    """
    Split an image into a uniform grid of tiles, each carrying an overlap halo
    of surrounding context. Emits a GEN2_TILE_LAYOUT describing the grid and a
    list of tile images in row-major order.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_size": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "overlap_pct": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("GEN2_TILE_LAYOUT", "IMAGE")
    RETURN_NAMES = ("tile_layout", "tiles_image_list")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "split"
    CATEGORY = "Gen2/Tiling"

    def split(self, image: torch.Tensor, tile_size: int, overlap_pct: float):
        if image.ndim != 4:
            raise ValueError(
                f"Gen2_TileSplitter expected IMAGE of shape (B, H, W, C); got {tuple(image.shape)}"
            )
        _B, H, W, _C = image.shape

        n_rows = _best_n(int(H), int(tile_size))
        n_cols = _best_n(int(W), int(tile_size))

        base_h_floor = H // n_rows
        base_w_floor = W // n_cols
        h_remainder = H - n_rows * base_h_floor
        w_remainder = W - n_cols * base_w_floor

        overlap_px_h = int(round(float(overlap_pct) * base_h_floor))
        overlap_px_w = int(round(float(overlap_pct) * base_w_floor))

        # Pre-pad the image once (replicate at borders) so every expanded crop
        # can be a straight slice. F.pad expects (B, C, H, W).
        img_nchw = image.permute(0, 3, 1, 2)
        if overlap_px_h > 0 or overlap_px_w > 0:
            padded = F.pad(
                img_nchw,
                (overlap_px_w, overlap_px_w, overlap_px_h, overlap_px_h),
                mode="replicate",
            )
        else:
            padded = img_nchw
        padded_bhwc = padded.permute(0, 2, 3, 1).contiguous()

        splits = []
        tiles = []

        for r in range(n_rows):
            base_y = r * base_h_floor
            base_h = base_h_floor + (h_remainder if r == n_rows - 1 else 0)

            for c in range(n_cols):
                base_x = c * base_w_floor
                base_w = base_w_floor + (w_remainder if c == n_cols - 1 else 0)

                tile_h = base_h + 2 * overlap_px_h
                tile_w = base_w + 2 * overlap_px_w

                # In padded-space, image-space (0, 0) sits at (overlap_px_h, overlap_px_w),
                # so an image-space crop starting at (base_y - overlap_px_h, base_x - overlap_px_w)
                # of size (tile_h, tile_w) becomes padded-space (base_y, base_x, tile_h, tile_w).
                tile = padded_bhwc[
                    :,
                    base_y : base_y + tile_h,
                    base_x : base_x + tile_w,
                    :,
                ]
                tiles.append(tile)

                splits.append({
                    "row": r,
                    "col": c,
                    "y": base_y - overlap_px_h,
                    "x": base_x - overlap_px_w,
                    "h": int(tile_h),
                    "w": int(tile_w),
                    "base_y": int(base_y),
                    "base_x": int(base_x),
                    "base_h": int(base_h),
                    "base_w": int(base_w),
                    "base_y_local": int(overlap_px_h),
                    "base_x_local": int(overlap_px_w),
                    "base_h_local": int(base_h),
                    "base_w_local": int(base_w),
                })

        tile_layout = {
            "original_height": int(H),
            "original_width": int(W),
            "rows": int(n_rows),
            "cols": int(n_cols),
            "overlap_px_h": int(overlap_px_h),
            "overlap_px_w": int(overlap_px_w),
            "splits": splits,
        }

        return (tile_layout, tiles)


NODE_CLASS_MAPPINGS = {
    "Gen2_TileSplitter": Gen2_TileSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_TileSplitter": "Gen2 Tile Splitter",
}
