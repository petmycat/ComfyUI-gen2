"""
Gen2 Seam Fix - Build seam-targeted tiles and masks for a second-pass denoise
that smooths over the dimming/discontinuities left by the first-pass merge.

Workflow position:
    pass 1: Splitter -> TileMasks (feathered) -> sampler -> TileMerger (none)
            => merged image with dim cross at tile-base boundaries
    pass 2: Gen2_SeamFix -> sampler -> Gen2_SeamMerger
            => clean final image

Seam tile geometry (non-overlapping):
- For each pair of adjacent base regions in the input tile_layout, the boundary
  between them is a "seam" -- a 1-pixel line in image space.
- We carve the union of "seam neighborhoods" into:
    * one intersection tile per (horizontal seam x vertical seam) crossing
    * one arm tile per seam segment between consecutive intersections (or image
      edge <-> first/last intersection)
- Every tile is a rectangle of thickness = seam_strip_width + 2*overlap_px
  in its across-seam direction (so the diffusion sampler has overlap_px of
  context on each side of the seam strip).
- Tiles tile the cross-shaped seam neighborhood exactly with no overlap.

Tile content:
- Starts as the merged image cropped at the tile's image-space rect.
- The strip region (where the mask is non-zero) is then OVERWRITTEN with the
  original image's content at the same image-space location, giving the
  sampler a clean inpaint initialization at the strip while leaving the
  regenerated bases untouched as conditioning context.

Tile masks:
- Arm tile: single strip aligned with the seam axis.
- Intersection tile: vertical strip and horizontal strip are independently
  feathered, then combined via per-pixel max into a single cross-shaped mask.
- All masks are Gaussian-feathered with sigma = mask_blend_pixels / 2.

Output ordering: intersection tiles first (in row-major order of their seam
crossing), then arm tiles (grouped by seam axis: all vertical-seam arms in
row-major, then all horizontal-seam arms in row-major). Same order is used
for seam_tiles and seam_tiles_masks so positional pairing is unambiguous.
"""

from typing import List, Tuple

import torch
import scipy.ndimage


def _unique_sorted(values):
    return sorted(set(int(v) for v in values))


def _build_arm_segments(seam_coord: int, span_start: int, span_end: int,
                         crossing_coords: List[int], thickness: int):
    """Split a seam segment [span_start, span_end) (along the seam axis) into
    sub-segments delimited by crossing points (where a perpendicular seam
    intersects). Returns a list of (along_start, along_end) ranges for the
    arm tiles, EXCLUDING the intersection squares themselves.

    crossing_coords: locations of perpendicular seams along the seam axis.
    thickness: size of each intersection tile along the seam axis (also = the
        seam-tile thickness on the other axis).
    """
    half = thickness // 2
    # Compute "blocked" ranges around each crossing.
    blocked = []
    for c in crossing_coords:
        if span_start < c < span_end:
            blocked.append((max(span_start, c - half), min(span_end, c - half + thickness)))
    blocked.sort()

    # Carve [span_start, span_end) minus the blocked ranges.
    segments = []
    cursor = span_start
    for b_start, b_end in blocked:
        if b_start > cursor:
            segments.append((cursor, b_start))
        cursor = max(cursor, b_end)
    if cursor < span_end:
        segments.append((cursor, span_end))
    return segments


def _gaussian_feather_strip(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply scipy.ndimage.gaussian_filter to a 2D mask, preserving dtype."""
    if sigma <= 0.0:
        return mask
    m_np = mask.cpu().numpy()
    m_np = scipy.ndimage.gaussian_filter(m_np, sigma=sigma, mode="constant", cval=0.0)
    return torch.from_numpy(m_np).to(device=mask.device, dtype=mask.dtype).clamp(0.0, 1.0)


class Gen2_SeamFix:
    """Build seam-targeted tiles and masks from a first-pass merged image and
    its tile_layout, for use in a second-pass denoise that smooths the seams.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "merged_image": ("IMAGE",),
                "tile_layout": ("GEN2_TILE_LAYOUT",),
                "seam_strip_width": ("INT", {"default": 128, "min": 8,
                                              "max": 1024, "step": 8}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0,
                                               "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("GEN2_SEAM_LAYOUT", "IMAGE", "MASK")
    RETURN_NAMES = ("seam_layout", "seam_tiles", "seam_tiles_masks")
    OUTPUT_IS_LIST = (False, True, True)
    FUNCTION = "build"
    CATEGORY = "Gen2/Tiling"

    def build(self, original_image: torch.Tensor, merged_image: torch.Tensor,
              tile_layout: dict, seam_strip_width: int, mask_blend_pixels: int):
        if original_image.shape != merged_image.shape:
            raise ValueError(
                f"Gen2_SeamFix: original_image and merged_image must have the "
                f"same shape; got {tuple(original_image.shape)} vs "
                f"{tuple(merged_image.shape)}"
            )
        if not isinstance(tile_layout, dict) or "splits" not in tile_layout:
            raise ValueError("Gen2_SeamFix: tile_layout is not a valid GEN2_TILE_LAYOUT")

        H = int(tile_layout["original_height"])
        W = int(tile_layout["original_width"])
        overlap_px_h = int(tile_layout["overlap_px_h"])
        overlap_px_w = int(tile_layout["overlap_px_w"])
        splits = tile_layout["splits"]

        if int(original_image.shape[1]) != H or int(original_image.shape[2]) != W:
            raise ValueError(
                f"Gen2_SeamFix: image shape {tuple(original_image.shape)} doesn't match "
                f"tile_layout original dims ({H}, {W})"
            )

        # Identify vertical and horizontal seam locations from base regions.
        verticals_set = set()
        horizontals_set = set()
        for s in splits:
            bx_end = int(s["base_x"]) + int(s["base_w"])
            by_end = int(s["base_y"]) + int(s["base_h"])
            if bx_end < W:
                verticals_set.add(bx_end)
            if by_end < H:
                horizontals_set.add(by_end)
        verticals = _unique_sorted(verticals_set)
        horizontals = _unique_sorted(horizontals_set)

        thickness_v = seam_strip_width + 2 * overlap_px_w  # seam-tile thickness for vertical seams
        thickness_h = seam_strip_width + 2 * overlap_px_h  # thickness for horizontal seams
        intersection_size_h = thickness_h
        intersection_size_w = thickness_v
        sigma = mask_blend_pixels / 2.0

        device = merged_image.device
        dtype = merged_image.dtype

        seam_tiles_meta = []  # entries describing each tile in order

        # 1. Intersection tiles: one per (horizontal x vertical) crossing.
        for hy in horizontals:
            for vx in verticals:
                half_h = intersection_size_h // 2
                half_w = intersection_size_w // 2
                y0 = max(0, hy - half_h)
                x0 = max(0, vx - half_w)
                y1 = min(H, y0 + intersection_size_h)
                x1 = min(W, x0 + intersection_size_w)
                # Re-anchor if clipped at the far edge so we always preserve the
                # full thickness (shift the tile inward instead of shrinking).
                if y1 - y0 < intersection_size_h:
                    y0 = max(0, y1 - intersection_size_h)
                if x1 - x0 < intersection_size_w:
                    x0 = max(0, x1 - intersection_size_w)
                seam_tiles_meta.append({
                    "kind": "intersection",
                    "axis": None,
                    "y": int(y0), "x": int(x0),
                    "h": int(y1 - y0), "w": int(x1 - x0),
                    "strips": [
                        {"axis": "vertical",   "center_local": int(vx - x0),
                         "width": int(seam_strip_width)},
                        {"axis": "horizontal", "center_local": int(hy - y0),
                         "width": int(seam_strip_width)},
                    ],
                    "seam_axes_coords": [int(vx), int(hy)],
                })

        # 2. Arm tiles for vertical seams.
        for vx in verticals:
            arm_segments = _build_arm_segments(
                seam_coord=vx, span_start=0, span_end=H,
                crossing_coords=horizontals, thickness=intersection_size_h,
            )
            for y_start, y_end in arm_segments:
                half_w = intersection_size_w // 2
                x0 = max(0, vx - half_w)
                x1 = min(W, x0 + intersection_size_w)
                if x1 - x0 < intersection_size_w:
                    x0 = max(0, x1 - intersection_size_w)
                seam_tiles_meta.append({
                    "kind": "arm",
                    "axis": "vertical",
                    "y": int(y_start), "x": int(x0),
                    "h": int(y_end - y_start), "w": int(x1 - x0),
                    "strips": [
                        {"axis": "vertical", "center_local": int(vx - x0),
                         "width": int(seam_strip_width)},
                    ],
                    "seam_axes_coords": [int(vx)],
                })

        # 3. Arm tiles for horizontal seams.
        for hy in horizontals:
            arm_segments = _build_arm_segments(
                seam_coord=hy, span_start=0, span_end=W,
                crossing_coords=verticals, thickness=intersection_size_w,
            )
            for x_start, x_end in arm_segments:
                half_h = intersection_size_h // 2
                y0 = max(0, hy - half_h)
                y1 = min(H, y0 + intersection_size_h)
                if y1 - y0 < intersection_size_h:
                    y0 = max(0, y1 - intersection_size_h)
                seam_tiles_meta.append({
                    "kind": "arm",
                    "axis": "horizontal",
                    "y": int(y0), "x": int(x_start),
                    "h": int(y1 - y0), "w": int(x_end - x_start),
                    "strips": [
                        {"axis": "horizontal", "center_local": int(hy - y0),
                         "width": int(seam_strip_width)},
                    ],
                    "seam_axes_coords": [int(hy)],
                })

        # 4. Build seam tile images and masks.
        seam_tiles = []
        seam_masks = []
        for meta in seam_tiles_meta:
            y, x, h, w = meta["y"], meta["x"], meta["h"], meta["w"]
            merged_crop = merged_image[:, y:y + h, x:x + w, :].clone()
            original_crop = original_image[:, y:y + h, x:x + w, :]

            # Build per-strip binary masks, feather independently, combine via max.
            strip_masks = []
            for strip in meta["strips"]:
                m = torch.zeros((h, w), dtype=torch.float32, device=device)
                center = int(strip["center_local"])
                half = int(strip["width"]) // 2
                if strip["axis"] == "vertical":
                    lo = max(0, center - half)
                    hi = min(w, center + half)
                    m[:, lo:hi] = 1.0
                else:  # horizontal
                    lo = max(0, center - half)
                    hi = min(h, center + half)
                    m[lo:hi, :] = 1.0
                strip_masks.append(_gaussian_feather_strip(m, sigma))

            if len(strip_masks) == 1:
                combined_mask = strip_masks[0]
            else:
                combined_mask = strip_masks[0]
                for sm in strip_masks[1:]:
                    combined_mask = torch.maximum(combined_mask, sm)

            # Overwrite strip region with original-image content (sampler-friendly init).
            # Use a hard binary version of the strip union for the substitution so the
            # context outside the feather stays exactly equal to the merged image.
            hard_strip = torch.zeros((h, w), dtype=torch.float32, device=device)
            for strip in meta["strips"]:
                center = int(strip["center_local"])
                half = int(strip["width"]) // 2
                if strip["axis"] == "vertical":
                    lo = max(0, center - half)
                    hi = min(w, center + half)
                    hard_strip[:, lo:hi] = 1.0
                else:
                    lo = max(0, center - half)
                    hi = min(h, center + half)
                    hard_strip[lo:hi, :] = 1.0
            hs = hard_strip[None, :, :, None]
            seam_tile = merged_crop * (1.0 - hs) + original_crop * hs

            seam_tiles.append(seam_tile.to(dtype=dtype))
            seam_masks.append(combined_mask.unsqueeze(0))

        seam_layout = {
            "original_height": H,
            "original_width": W,
            "seam_strip_width": int(seam_strip_width),
            "mask_blend_pixels": int(mask_blend_pixels),
            "seam_tile_thickness_h": int(intersection_size_h),
            "seam_tile_thickness_w": int(intersection_size_w),
            "verticals": verticals,
            "horizontals": horizontals,
            "tiles": seam_tiles_meta,
        }

        return (seam_layout, seam_tiles, seam_masks)


NODE_CLASS_MAPPINGS = {
    "Gen2_SeamFix": Gen2_SeamFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_SeamFix": "Gen2 Seam Fix",
}
