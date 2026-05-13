"""
Gen2 Tiling - blending and seam helpers used by Gen2_TileMerger.

All functions operate on torch tensors. Multi-band blending and the optimal-seam
DP use numpy under the hood (scipy/numpy is already a dep for the mask node).

Conventions:
- Tile images are (B, H, W, C) float32 in [0, 1] (ComfyUI's IMAGE format).
- Weight masks are (H, W) float32 in [0, 1] (a single 2D map; broadcast across
  batch and channels at use-site).
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Per-tile weight masks
# ---------------------------------------------------------------------------

def base_only_mask(h: int, w: int, base_y: int, base_x: int,
                   base_h: int, base_w: int,
                   device: torch.device = None) -> torch.Tensor:
    """Hard binary mask: 1.0 over the base region, 0.0 over the halo.

    This is the natural mask for the masked-inpaint workflow ('blend_mode=none').
    """
    mask = torch.zeros((h, w), dtype=torch.float32, device=device)
    mask[base_y:base_y + base_h, base_x:base_x + base_w] = 1.0
    return mask


def linear_falloff_mask(h: int, w: int, base_y: int, base_x: int,
                        base_h: int, base_w: int,
                        device: torch.device = None) -> torch.Tensor:
    """Linear ramp from 1.0 at the base-region boundary to 0.0 at the frame edge.

    Inside the base region the value is 1.0; in the halo it linearly drops to
    0.0 at the outermost row/column. If there is no halo on a given side
    (i.e. the base region touches the frame), the mask stays at 1.0 there.
    """
    pad_top = base_y
    pad_left = base_x
    pad_bottom = h - (base_y + base_h)
    pad_right = w - (base_x + base_w)

    y_ramp = torch.ones(h, dtype=torch.float32, device=device)
    if pad_top > 0:
        y_ramp[:pad_top] = torch.linspace(0.0, 1.0, steps=pad_top + 1,
                                          device=device)[1:]
    if pad_bottom > 0:
        y_ramp[-pad_bottom:] = torch.linspace(1.0, 0.0, steps=pad_bottom + 1,
                                              device=device)[:-1]

    x_ramp = torch.ones(w, dtype=torch.float32, device=device)
    if pad_left > 0:
        x_ramp[:pad_left] = torch.linspace(0.0, 1.0, steps=pad_left + 1,
                                           device=device)[1:]
    if pad_right > 0:
        x_ramp[-pad_right:] = torch.linspace(1.0, 0.0, steps=pad_right + 1,
                                             device=device)[:-1]

    return y_ramp[:, None] * x_ramp[None, :]


def gaussian_falloff_mask(h: int, w: int, base_y: int, base_x: int,
                          base_h: int, base_w: int,
                          blend_strength: float,
                          overlap_px_h: int, overlap_px_w: int,
                          device: torch.device = None) -> torch.Tensor:
    """Mask = 1.0 over base region, decaying with a Gaussian shape into the halo.

    The Gaussian sigma is `blend_strength * max(overlap_px_h, overlap_px_w) / 2`
    (so blend_strength=1.0 gives a sigma equal to half the overlap width, and
    blend_strength=0.0 reduces to a hard binary mask).

    Implementation detail: the binary "plateau" is inflated by 3*sigma into the
    halo before convolution, so the Gaussian falloff lands entirely inside the
    halo and the base region itself stays at 1.0 (no fall-off at the base
    boundary). This matters at image corners, where only a single tile's halo
    covers the corner pixels.
    """
    base_mask = base_only_mask(h, w, base_y, base_x, base_h, base_w, device=device)

    sigma = float(blend_strength) * max(overlap_px_h, overlap_px_w) / 2.0
    if sigma <= 0.0:
        return base_mask

    inflate = int(round(3.0 * sigma))
    inflate_y = min(inflate, max(0, base_y), max(0, h - (base_y + base_h)))
    inflate_x = min(inflate, max(0, base_x), max(0, w - (base_x + base_w)))
    e_y0 = base_y - inflate_y
    e_x0 = base_x - inflate_x
    e_y1 = base_y + base_h + inflate_y
    e_x1 = base_x + base_w + inflate_x
    inflated = torch.zeros((h, w), dtype=torch.float32, device=device)
    inflated[e_y0:e_y1, e_x0:e_x1] = 1.0

    kernel_radius = max(1, int(round(3.0 * sigma)))
    kernel_size = 2 * kernel_radius + 1
    coords = torch.arange(-kernel_radius, kernel_radius + 1,
                          dtype=torch.float32, device=device)
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    m = inflated.view(1, 1, h, w)
    kernel_h = kernel_1d.view(1, 1, 1, kernel_size)
    kernel_v = kernel_1d.view(1, 1, kernel_size, 1)
    m = F.conv2d(m, kernel_h, padding=(0, kernel_radius))
    m = F.conv2d(m, kernel_v, padding=(kernel_radius, 0))
    return m.view(h, w).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Reinhard histogram matching (mean/std color transfer)
# ---------------------------------------------------------------------------

def reinhard_match(source: torch.Tensor, reference: torch.Tensor,
                   region_source: Tuple[slice, slice],
                   region_reference: Tuple[slice, slice]) -> torch.Tensor:
    """Return a copy of `source` whose per-channel mean/std (measured only over
    `region_source`) is matched to that of `reference[region_reference]`.

    Both tensors are (B, H, W, C). Region slices are (y_slice, x_slice).
    """
    src_patch = source[:, region_source[0], region_source[1], :]
    ref_patch = reference[:, region_reference[0], region_reference[1], :]

    s_mean = src_patch.mean(dim=(1, 2), keepdim=True)
    s_std = src_patch.std(dim=(1, 2), keepdim=True) + 1e-6
    r_mean = ref_patch.mean(dim=(1, 2), keepdim=True)
    r_std = ref_patch.std(dim=(1, 2), keepdim=True) + 1e-6

    out = (source - s_mean) / s_std * r_std + r_mean
    return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Optimal seam via dynamic programming (naive per-pixel loop)
# ---------------------------------------------------------------------------

def optimal_seam(overlap_a: torch.Tensor, overlap_b: torch.Tensor,
                 direction: str) -> torch.Tensor:
    """Compute a min-energy seam through the overlap region between two tiles.

    Args:
        overlap_a, overlap_b: (H, W, C) tensors of the overlapping content from
            tile A (left/top) and tile B (right/bottom). Both have the same shape.
        direction: 'horizontal' for a left/right neighbor pair (seam runs
            top-to-bottom), 'vertical' for a top/bottom pair (seam runs
            left-to-right).

    Returns:
        A (H, W) float mask where 1.0 means "A wins" and 0.0 means "B wins".
    """
    if overlap_a.shape != overlap_b.shape:
        raise ValueError(
            f"optimal_seam: overlap shapes differ {tuple(overlap_a.shape)} vs "
            f"{tuple(overlap_b.shape)}"
        )

    energy = (overlap_a - overlap_b).abs().mean(dim=-1).cpu().numpy()
    h, w = energy.shape
    mask = np.zeros((h, w), dtype=np.float32)

    if direction == "horizontal":
        dp = energy.copy()
        back = np.zeros_like(dp, dtype=np.int32)
        for i in range(1, h):
            for j in range(w):
                lo = max(0, j - 1)
                hi = min(w, j + 2)
                slab = dp[i - 1, lo:hi]
                k = int(np.argmin(slab))
                dp[i, j] += slab[k]
                back[i, j] = lo + k
        cut = int(np.argmin(dp[-1]))
        for i in range(h - 1, -1, -1):
            mask[i, :cut] = 1.0
            if i > 0:
                cut = int(back[i, cut])
    elif direction == "vertical":
        dp = energy.copy()
        back = np.zeros_like(dp, dtype=np.int32)
        for j in range(1, w):
            for i in range(h):
                lo = max(0, i - 1)
                hi = min(h, i + 2)
                slab = dp[lo:hi, j - 1]
                k = int(np.argmin(slab))
                dp[i, j] += slab[k]
                back[i, j] = lo + k
        cut = int(np.argmin(dp[:, -1]))
        for j in range(w - 1, -1, -1):
            mask[:cut, j] = 1.0
            if j > 0:
                cut = int(back[cut, j])
    else:
        raise ValueError(f"optimal_seam: direction must be 'horizontal' or 'vertical', got {direction!r}")

    return torch.from_numpy(mask).to(overlap_a.device)


# ---------------------------------------------------------------------------
# Multi-band (Laplacian pyramid) blending
# ---------------------------------------------------------------------------

def _build_gaussian_pyramid(x: torch.Tensor, levels: int) -> list:
    """Build a Gaussian pyramid via 2x average-pooling."""
    pyramid = [x]
    for _ in range(levels):
        pyramid.append(F.avg_pool2d(pyramid[-1], kernel_size=2))
    return pyramid


def multi_band_blend(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor,
                     levels: int = 5) -> torch.Tensor:
    """Laplacian-pyramid blend of two (B, C, H, W) images using a (B, 1, H, W)
    mask. mask=1 selects b, mask=0 selects a.

    `levels` is auto-capped so the coarsest level still has >= 2 pixels per side.
    """
    if a.shape != b.shape:
        raise ValueError(f"multi_band_blend: a and b must have the same shape; got {tuple(a.shape)} vs {tuple(b.shape)}")
    if mask.shape[-2:] != a.shape[-2:]:
        raise ValueError("multi_band_blend: mask spatial shape must match images")

    h, w = a.shape[-2], a.shape[-1]
    max_levels = max(1, int(np.floor(np.log2(min(h, w)))) - 1)
    levels = min(levels, max_levels)

    g_a = _build_gaussian_pyramid(a, levels)
    g_b = _build_gaussian_pyramid(b, levels)
    g_m = _build_gaussian_pyramid(mask, levels)

    lap_a = []
    lap_b = []
    for i in range(levels):
        hi, wi = g_a[i].shape[-2], g_a[i].shape[-1]
        up_a = F.interpolate(g_a[i + 1], size=(hi, wi), mode='bilinear', align_corners=False)
        up_b = F.interpolate(g_b[i + 1], size=(hi, wi), mode='bilinear', align_corners=False)
        lap_a.append(g_a[i] - up_a)
        lap_b.append(g_b[i] - up_b)
    lap_a.append(g_a[-1])
    lap_b.append(g_b[-1])

    blended = []
    for i in range(levels + 1):
        m = g_m[i]
        blended.append(lap_a[i] * (1.0 - m) + lap_b[i] * m)

    out = blended[-1]
    for i in range(levels - 1, -1, -1):
        hi, wi = blended[i].shape[-2], blended[i].shape[-1]
        out = F.interpolate(out, size=(hi, wi), mode='bilinear', align_corners=False)
        out = out + blended[i]
    return out.clamp(0.0, 1.0)
