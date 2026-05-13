"""
Gen2 Tile Merger - Recombine processed tiles back into a single image, using a
GEN2_TILE_LAYOUT to know where each tile belongs.

Blend modes:
- "none"       Base-only paste. Halos are discarded, base regions tile the
               original image exactly. The fast path for the masked-inpaint
               workflow (since halos are supposed to be unchanged context).
- "linear"     Normalized weighted average. Each tile contributes proportional
               to a linear-falloff mask (1.0 over base, dropping to 0 at the
               halo's outer edge).
- "gaussian"   Same as linear but with a Gaussian falloff whose sigma is
               controlled by blend_strength.
- "multi_band" Sequential Laplacian-pyramid blend in raster order, using
               Gaussian-shaped masks. Highest quality, slowest.

Seam mode (only applies when blend_mode != "none"):
- "middle"     No DP. Masks decay through the entire halo according to the
               blend_mode shape.
- "optimal"    For each pair of adjacent tiles, run a min-energy DP cut through
               their 2*overlap_px shared band and force the masks to follow it.

Histogram matching (only meaningful when blend_mode != "none"): applies a
Reinhard mean/std color transfer to every tile that has a left neighbor,
matching the left neighbor's overlap content. Reduces tile-to-tile color drift
when each tile is sampled independently.
"""

import torch

from .utils import (
    base_only_mask,
    linear_falloff_mask,
    gaussian_falloff_mask,
    reinhard_match,
    optimal_seam,
    multi_band_blend,
)


def _take_first(value):
    """Unwrap a single-element list (INPUT_IS_LIST wraps non-list inputs)."""
    if isinstance(value, list):
        if len(value) == 0:
            return None
        return value[0]
    return value


def _crop_for_canvas(s, H: int, W: int):
    """Compute (img slice, tile-local slice) so that an out-of-bounds halo
    portion (e.g. a corner tile's outer replicate padding) is silently dropped
    when pasting the tile onto the image-space canvas."""
    ty = int(s["y"])
    tx = int(s["x"])
    th = int(s["h"])
    tw = int(s["w"])

    img_y0 = max(0, ty)
    img_x0 = max(0, tx)
    img_y1 = min(H, ty + th)
    img_x1 = min(W, tx + tw)

    local_y0 = img_y0 - ty
    local_x0 = img_x0 - tx
    local_y1 = local_y0 + (img_y1 - img_y0)
    local_x1 = local_x0 + (img_x1 - img_x0)

    return (img_y0, img_x0, img_y1, img_x1), (local_y0, local_x0, local_y1, local_x1)


class Gen2_TileMerger:
    """
    Merge a list of processed tiles back into a single image using a
    GEN2_TILE_LAYOUT.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_layout": ("GEN2_TILE_LAYOUT",),
                "blend_mode": (["none", "linear", "gaussian", "multi_band"],
                               {"default": "none"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0,
                                              "max": 1.0, "step": 0.01}),
                "seam_mode": (["middle", "optimal"], {"default": "middle"}),
                "histogram_matching": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "processed_tiles_image_list": ("IMAGE",),
                "masks_list": ("MASK",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_image",)
    FUNCTION = "merge"
    CATEGORY = "Gen2/Tiling"

    def merge(self, tile_layout, blend_mode, blend_strength, seam_mode,
              histogram_matching, processed_tiles_image_list=None,
              masks_list=None):
        layout = _take_first(tile_layout)
        mode = _take_first(blend_mode)
        strength = float(_take_first(blend_strength))
        seam = _take_first(seam_mode)
        do_hist = bool(_take_first(histogram_matching))

        if layout is None or "splits" not in layout:
            raise ValueError(
                "Gen2_TileMerger: tile_layout is not a valid GEN2_TILE_LAYOUT"
            )
        if processed_tiles_image_list is None or len(processed_tiles_image_list) == 0:
            raise ValueError(
                "Gen2_TileMerger: processed_tiles_image_list is empty; nothing to merge"
            )

        tiles = list(processed_tiles_image_list)
        splits = layout["splits"]

        if len(tiles) != len(splits):
            raise ValueError(
                f"Gen2_TileMerger: tile count {len(tiles)} doesn't match layout "
                f"split count {len(splits)}"
            )

        H = int(layout["original_height"])
        W = int(layout["original_width"])
        overlap_px_h = int(layout["overlap_px_h"])
        overlap_px_w = int(layout["overlap_px_w"])
        rows = int(layout["rows"])
        cols = int(layout["cols"])

        ref = tiles[0]
        B = int(ref.shape[0])
        C = int(ref.shape[-1])
        device = ref.device
        dtype = ref.dtype

        grid_map = {(s["row"], s["col"]): i for i, s in enumerate(splits)}

        # --------------------------------------------------------------------
        # Optional histogram matching (left-neighbor reference)
        # --------------------------------------------------------------------
        if do_hist and mode != "none" and overlap_px_w > 0:
            tiles = [t.clone() for t in tiles]
            for r in range(rows):
                for c in range(1, cols):
                    li = grid_map[(r, c - 1)]
                    ri = grid_map[(r, c)]
                    left_tile = tiles[li]
                    right_tile = tiles[ri]
                    band = 2 * overlap_px_w
                    ref_slice = (slice(None), slice(left_tile.shape[2] - band, None))
                    src_slice = (slice(None), slice(0, band))
                    tiles[ri] = reinhard_match(right_tile, left_tile,
                                                src_slice, ref_slice)

        # --------------------------------------------------------------------
        # Build per-tile 2D weight masks (H_tile x W_tile)
        # --------------------------------------------------------------------
        if mode == "none" and masks_list is not None and len(masks_list) == len(splits):
            masks = []
            for m in masks_list:
                if m.ndim == 3:
                    masks.append(m[0].to(device=device, dtype=torch.float32))
                else:
                    masks.append(m.to(device=device, dtype=torch.float32))
        else:
            masks = []
            for s in splits:
                h = int(s["h"])
                w = int(s["w"])
                by = int(s["base_y_local"])
                bx = int(s["base_x_local"])
                bh = int(s["base_h_local"])
                bw = int(s["base_w_local"])

                if mode == "none":
                    masks.append(base_only_mask(h, w, by, bx, bh, bw, device=device))
                elif mode == "linear":
                    masks.append(linear_falloff_mask(h, w, by, bx, bh, bw,
                                                     device=device))
                else:  # gaussian or multi_band
                    masks.append(gaussian_falloff_mask(h, w, by, bx, bh, bw,
                                                        strength,
                                                        overlap_px_h,
                                                        overlap_px_w,
                                                        device=device))

        # --------------------------------------------------------------------
        # Optional optimal-seam DP between adjacent tile pairs
        # --------------------------------------------------------------------
        if seam == "optimal" and mode != "none":
            # Horizontal pairs
            for r in range(rows):
                for c in range(cols - 1):
                    self._apply_horizontal_seam(
                        tiles, splits, masks, grid_map, r, c, overlap_px_w
                    )
            # Vertical pairs
            for r in range(rows - 1):
                for c in range(cols):
                    self._apply_vertical_seam(
                        tiles, splits, masks, grid_map, r, c, overlap_px_h
                    )

        # --------------------------------------------------------------------
        # Composite tiles onto the canvas
        # --------------------------------------------------------------------
        if mode == "none":
            return (self._compose_paste(tiles, splits, masks, B, H, W, C, dtype, device),)
        if mode in ("linear", "gaussian"):
            return (self._compose_weighted(tiles, splits, masks, B, H, W, C, dtype, device),)
        if mode == "multi_band":
            return (self._compose_multi_band(tiles, splits, masks, B, H, W, C, dtype, device),)
        raise ValueError(f"Unknown blend_mode: {mode!r}")

    # ------------------------------------------------------------------
    # Seam helpers
    # ------------------------------------------------------------------

    def _apply_horizontal_seam(self, tiles, splits, masks, grid_map,
                                r: int, c: int, overlap_px_w: int):
        if overlap_px_w <= 0:
            return
        li = grid_map[(r, c)]
        ri = grid_map[(r, c + 1)]
        sL = splits[li]
        sR = splits[ri]

        band = 2 * overlap_px_w
        tL = tiles[li]
        tR = tiles[ri]

        # Use only the y-range where BOTH tiles have base content.
        y0 = max(sL["base_y_local"], sR["base_y_local"])
        y1L = sL["base_y_local"] + sL["base_h_local"]
        y1R = sR["base_y_local"] + sR["base_h_local"]
        y1 = min(y1L, y1R)
        if y1 <= y0:
            return

        l_x0 = sL["w"] - band
        l_x1 = sL["w"]
        r_x0 = 0
        r_x1 = band
        if l_x0 < 0 or r_x1 > sR["w"]:
            return

        ov_left = tL[0, y0:y1, l_x0:l_x1, :]
        ov_right = tR[0, y0:y1, r_x0:r_x1, :]
        seam = optimal_seam(ov_left, ov_right, "horizontal")

        masks[li][y0:y1, l_x0:l_x1] = masks[li][y0:y1, l_x0:l_x1] * seam
        masks[ri][y0:y1, r_x0:r_x1] = masks[ri][y0:y1, r_x0:r_x1] * (1.0 - seam)

    def _apply_vertical_seam(self, tiles, splits, masks, grid_map,
                              r: int, c: int, overlap_px_h: int):
        if overlap_px_h <= 0:
            return
        ti = grid_map[(r, c)]
        bi = grid_map[(r + 1, c)]
        sT = splits[ti]
        sB = splits[bi]

        band = 2 * overlap_px_h
        tT = tiles[ti]
        tB = tiles[bi]

        x0 = max(sT["base_x_local"], sB["base_x_local"])
        x1T = sT["base_x_local"] + sT["base_w_local"]
        x1B = sB["base_x_local"] + sB["base_w_local"]
        x1 = min(x1T, x1B)
        if x1 <= x0:
            return

        t_y0 = sT["h"] - band
        t_y1 = sT["h"]
        b_y0 = 0
        b_y1 = band
        if t_y0 < 0 or b_y1 > sB["h"]:
            return

        ov_top = tT[0, t_y0:t_y1, x0:x1, :]
        ov_bottom = tB[0, b_y0:b_y1, x0:x1, :]
        seam = optimal_seam(ov_top, ov_bottom, "vertical")

        masks[ti][t_y0:t_y1, x0:x1] = masks[ti][t_y0:t_y1, x0:x1] * seam
        masks[bi][b_y0:b_y1, x0:x1] = masks[bi][b_y0:b_y1, x0:x1] * (1.0 - seam)

    # ------------------------------------------------------------------
    # Composition strategies
    # ------------------------------------------------------------------

    def _compose_paste(self, tiles, splits, masks, B, H, W, C, dtype, device):
        """blend_mode='none': normalized weighted average using the provided
        (or default binary) masks.

        Reduces to an exact direct paste when the masks are binary (each pixel
        has exactly one mask=1 across all tiles, so sum=1 and division is a
        no-op). When the masks are feathered (e.g. Gen2_TileMasks with
        mask_blend_pixels > 0), normalization prevents the dimming that an
        alpha-overlay 'canvas*(1-m) + tile*m' formula would otherwise produce
        at tile-grid seams and image edges.

        Also robust against samplers that output incorrect content in tile
        halos: zero-mask halos contribute zero weight and are ignored.
        """
        canvas = torch.zeros((B, H, W, C), dtype=dtype, device=device)
        weight = torch.zeros((1, H, W, 1), dtype=dtype, device=device)
        for i, s in enumerate(splits):
            tile = tiles[i]
            mask = masks[i]
            (iy0, ix0, iy1, ix1), (ly0, lx0, ly1, lx1) = _crop_for_canvas(s, H, W)
            tile_crop = tile[:, ly0:ly1, lx0:lx1, :]
            mask_crop = mask[ly0:ly1, lx0:lx1]
            m = mask_crop[None, :, :, None]
            canvas[:, iy0:iy1, ix0:ix1, :] += tile_crop * m
            weight[:, iy0:iy1, ix0:ix1, :] += m
        eps = 1e-8
        canvas = canvas / weight.clamp(min=eps)
        return canvas

    def _compose_weighted(self, tiles, splits, masks, B, H, W, C, dtype, device):
        """blend_mode='linear' or 'gaussian': normalized weighted average across
        all tiles covering each pixel."""
        canvas = torch.zeros((B, H, W, C), dtype=dtype, device=device)
        weight = torch.zeros((1, H, W, 1), dtype=dtype, device=device)
        for i, s in enumerate(splits):
            tile = tiles[i]
            mask = masks[i]
            (iy0, ix0, iy1, ix1), (ly0, lx0, ly1, lx1) = _crop_for_canvas(s, H, W)
            tile_crop = tile[:, ly0:ly1, lx0:lx1, :]
            mask_crop = mask[ly0:ly1, lx0:lx1]
            m = mask_crop[None, :, :, None]
            canvas[:, iy0:iy1, ix0:ix1, :] += tile_crop * m
            weight[:, iy0:iy1, ix0:ix1, :] += m
        eps = 1e-8
        canvas = canvas / weight.clamp(min=eps)
        return canvas

    def _compose_multi_band(self, tiles, splits, masks, B, H, W, C, dtype, device):
        """blend_mode='multi_band': sequential Laplacian-pyramid blend onto the
        canvas, in raster order.

        The first tile is pasted at full strength (its halo content seeds the
        canvas for subsequent tiles to blend against). Each subsequent tile
        pyramid-blends with the existing canvas using its gaussian-shaped mask.
        """
        canvas = torch.zeros((B, H, W, C), dtype=dtype, device=device)
        first = True
        for i, s in enumerate(splits):
            tile = tiles[i]
            mask = masks[i]
            (iy0, ix0, iy1, ix1), (ly0, lx0, ly1, lx1) = _crop_for_canvas(s, H, W)
            tile_crop = tile[:, ly0:ly1, lx0:lx1, :]
            mask_crop = mask[ly0:ly1, lx0:lx1]

            if first:
                canvas[:, iy0:iy1, ix0:ix1, :] = tile_crop
                first = False
                continue

            a_nchw = canvas[:, iy0:iy1, ix0:ix1, :].permute(0, 3, 1, 2)
            b_nchw = tile_crop.permute(0, 3, 1, 2)
            m_nchw = mask_crop.view(1, 1, mask_crop.shape[0], mask_crop.shape[1])
            m_nchw = m_nchw.expand(a_nchw.shape[0], 1, -1, -1).contiguous()
            blended = multi_band_blend(a_nchw, b_nchw, m_nchw, levels=5)
            canvas[:, iy0:iy1, ix0:ix1, :] = blended.permute(0, 2, 3, 1)
        return canvas


NODE_CLASS_MAPPINGS = {
    "Gen2_TileMerger": Gen2_TileMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_TileMerger": "Gen2 Tile Merger",
}
