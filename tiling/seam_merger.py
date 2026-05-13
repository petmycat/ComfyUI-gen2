"""
Gen2 Seam Merger - Composite regenerated seam tiles back onto a first-pass
merged image to produce the final, seam-smoothed output.

Inputs:
- merged_image: the first-pass output (already has tile-base content; will have
  dim cross / discontinuities at the seams).
- seam_layout: GEN2_SEAM_LAYOUT from Gen2_SeamFix describing where each seam
  tile goes.
- processed_seam_tiles_list: the seam tiles after the sampler has regenerated
  the strip regions.
- seam_tiles_masks_list (optional): the masks from Gen2_SeamFix. If omitted,
  the merger reconstructs them from the seam_layout's strip definitions.
- blend_mode: how each seam tile gets composited onto the canvas.
- blend_strength: only meaningful for multi_band mode (controls pyramid depth).
- histogram_matching: if True, the strip-region content of each seam tile is
  Reinhard-matched to the surrounding (non-strip) merged-image context before
  compositing, so the regenerated strip blends color-consistently.

Output: final_image, an IMAGE of the same shape as merged_image.
"""

import math

import torch

from .utils import multi_band_blend


def _take_first(value):
    if isinstance(value, list):
        return value[0] if len(value) > 0 else None
    return value


def _rebuild_mask(meta: dict, mask_blend_pixels: int, device, sigma: float):
    """Rebuild a seam tile mask from its strip definitions (used when the
    caller didn't pass seam_tiles_masks_list)."""
    from .seam_fix import _gaussian_feather_strip

    h, w = int(meta["h"]), int(meta["w"])
    strip_masks = []
    for strip in meta["strips"]:
        m = torch.zeros((h, w), dtype=torch.float32, device=device)
        center = int(strip["center_local"])
        half = int(strip["width"]) // 2
        if strip["axis"] == "vertical":
            lo = max(0, center - half)
            hi = min(w, center + half)
            m[:, lo:hi] = 1.0
        else:
            lo = max(0, center - half)
            hi = min(h, center + half)
            m[lo:hi, :] = 1.0
        strip_masks.append(_gaussian_feather_strip(m, sigma))
    if len(strip_masks) == 1:
        return strip_masks[0]
    out = strip_masks[0]
    for sm in strip_masks[1:]:
        out = torch.maximum(out, sm)
    return out


def _hard_strip_mask(meta: dict, device):
    """Binary mask of the strip union (no feather). Used to identify the
    'strip region' vs the 'context region' for histogram matching."""
    h, w = int(meta["h"]), int(meta["w"])
    hard = torch.zeros((h, w), dtype=torch.bool, device=device)
    for strip in meta["strips"]:
        center = int(strip["center_local"])
        half = int(strip["width"]) // 2
        if strip["axis"] == "vertical":
            lo = max(0, center - half)
            hi = min(w, center + half)
            hard[:, lo:hi] = True
        else:
            lo = max(0, center - half)
            hi = min(h, center + half)
            hard[lo:hi, :] = True
    return hard


def _histogram_match_strip_to_context(seam_tile: torch.Tensor, merged_crop: torch.Tensor,
                                       hard_strip: torch.Tensor) -> torch.Tensor:
    """Adjust the strip region of seam_tile so its per-channel mean/std matches
    the merged_crop in the NON-strip region.

    seam_tile, merged_crop: (B, h, w, C) tensors.
    hard_strip: (h, w) bool, True = inside strip.
    """
    context_mask = ~hard_strip
    if context_mask.sum() < 2 or hard_strip.sum() < 1:
        return seam_tile

    context_pixels = merged_crop[:, context_mask, :]
    strip_pixels = seam_tile[:, hard_strip, :]

    ref_mean = context_pixels.mean(dim=1, keepdim=True)
    ref_std = context_pixels.std(dim=1, keepdim=True) + 1e-6
    src_mean = strip_pixels.mean(dim=1, keepdim=True)
    src_std = strip_pixels.std(dim=1, keepdim=True) + 1e-6

    matched = (strip_pixels - src_mean) / src_std * ref_std + ref_mean
    matched = matched.clamp(0.0, 1.0)

    out = seam_tile.clone()
    out[:, hard_strip, :] = matched
    return out


class Gen2_SeamMerger:
    """Composite regenerated seam tiles onto the first-pass merged image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merged_image": ("IMAGE",),
                "seam_layout": ("GEN2_SEAM_LAYOUT",),
                "blend_mode": (["linear", "gaussian", "multi_band"],
                               {"default": "gaussian"}),
                "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0,
                                              "max": 1.0, "step": 0.01}),
                "histogram_matching": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "processed_seam_tiles_list": ("IMAGE",),
                "seam_tiles_masks_list": ("MASK",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("final_image",)
    FUNCTION = "merge"
    CATEGORY = "Gen2/Tiling"

    def merge(self, merged_image, seam_layout, blend_mode, blend_strength,
              histogram_matching, processed_seam_tiles_list=None,
              seam_tiles_masks_list=None):
        merged = _take_first(merged_image)
        layout = _take_first(seam_layout)
        mode = _take_first(blend_mode)
        strength = float(_take_first(blend_strength))
        do_hist = bool(_take_first(histogram_matching))

        if merged is None:
            raise ValueError("Gen2_SeamMerger: merged_image is required")
        if layout is None or "tiles" not in layout:
            raise ValueError("Gen2_SeamMerger: seam_layout is not a valid GEN2_SEAM_LAYOUT")

        tiles_meta = layout["tiles"]
        if processed_seam_tiles_list is None or len(processed_seam_tiles_list) == 0:
            # Nothing to composite -- just pass the merged image through.
            return (merged.clone(),)

        if len(processed_seam_tiles_list) != len(tiles_meta):
            raise ValueError(
                f"Gen2_SeamMerger: tile count {len(processed_seam_tiles_list)} "
                f"doesn't match seam_layout count {len(tiles_meta)}"
            )

        device = merged.device
        sigma = float(layout.get("mask_blend_pixels", 32)) / 2.0

        canvas = merged.clone()

        # Determine pyramid level cap for multi_band from blend_strength.
        # strength=0 -> 1 level (effectively just direct blend at full res);
        # strength=1 -> 6 levels (deep low-frequency blend).
        levels_max = max(1, int(round(1 + strength * 5)))

        for i, meta in enumerate(tiles_meta):
            y = int(meta["y"])
            x = int(meta["x"])
            h = int(meta["h"])
            w = int(meta["w"])

            seam_tile = processed_seam_tiles_list[i]
            if seam_tile.shape[1] != h or seam_tile.shape[2] != w:
                raise ValueError(
                    f"Gen2_SeamMerger: tile #{i} shape {tuple(seam_tile.shape)} "
                    f"doesn't match seam_layout entry h={h} w={w}"
                )

            # Mask
            if seam_tiles_masks_list is not None and i < len(seam_tiles_masks_list):
                m = seam_tiles_masks_list[i]
                mask2d = m[0] if m.ndim == 3 else m
                mask2d = mask2d.to(device=device, dtype=torch.float32)
                if mask2d.shape != (h, w):
                    raise ValueError(
                        f"Gen2_SeamMerger: mask #{i} shape {tuple(mask2d.shape)} "
                        f"doesn't match seam_layout entry ({h}, {w})"
                    )
            else:
                mask2d = _rebuild_mask(meta, layout.get("mask_blend_pixels", 32),
                                       device, sigma)

            merged_crop = canvas[:, y:y + h, x:x + w, :]

            # Optional histogram match (strip region of seam_tile -> non-strip merged context)
            tile_for_blend = seam_tile.to(device=device, dtype=canvas.dtype)
            if do_hist:
                hard = _hard_strip_mask(meta, device)
                tile_for_blend = _histogram_match_strip_to_context(
                    tile_for_blend, merged_crop, hard
                )

            if mode == "multi_band":
                a_nchw = merged_crop.permute(0, 3, 1, 2)
                b_nchw = tile_for_blend.permute(0, 3, 1, 2)
                m_nchw = mask2d.view(1, 1, h, w).expand(a_nchw.shape[0], 1, -1, -1).contiguous()
                # Cap by what the input size can actually support.
                supported = max(1, int(math.floor(math.log2(min(h, w)))) - 1)
                levels = min(levels_max, supported)
                blended = multi_band_blend(a_nchw, b_nchw, m_nchw, levels=levels)
                canvas[:, y:y + h, x:x + w, :] = blended.permute(0, 2, 3, 1)
            else:
                m4 = mask2d.view(1, h, w, 1)
                canvas[:, y:y + h, x:x + w, :] = merged_crop * (1.0 - m4) + tile_for_blend * m4

        return (canvas,)


NODE_CLASS_MAPPINGS = {
    "Gen2_SeamMerger": Gen2_SeamMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_SeamMerger": "Gen2 Seam Merger",
}
