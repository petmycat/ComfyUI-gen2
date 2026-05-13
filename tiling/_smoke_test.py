"""
Standalone smoke test for Gen2_TileSplitter and Gen2_TileMasks.

Run from anywhere:
    python f:\\devComfy\\ComfyUI\\custom_nodes\\ComfyUI-gen2\\tiling\\_smoke_test.py

Or as a module from the gen2 directory:
    python -m tiling._smoke_test

This test does NOT require ComfyUI to be importable. It uses only torch + scipy.

Scenarios covered:
  1. Clean partition: 2048x2048, tile_size=1024, overlap_pct=0.25
       -> 2x2 grid, 4 tiles of 1536x1536 with 1024x1024 base regions
  2. last_tile_wins: 2500x2500, tile_size=1024, overlap_pct=0.25
       -> 2x2 grid... wait, check via algorithm: 2500/1=2500 (dist=1476),
          2500/2=1250 (dist=226), 2500/3=833.33 (dist=190.67). n=3 wins.
       -> 3x3 grid; first 2 rows/cols have base_h=833, last row/col absorbs
          remainder so its base_h=833+1=834
  3. No halo: 2048x2048, tile_size=1024, overlap_pct=0.0
       -> 2x2 grid of 1024x1024 tiles, no padding, no halo

Each scenario verifies:
  - tile count and dimensions
  - tile_layout metadata
  - each tile's base region matches the corresponding crop of the original image
  - inner halos pull real image content from neighboring base regions
  - outer halos are replicate-padded (constant along the padded direction)
  - masks have the right shape and 1.0 plateau exactly over the base region
  - re-stamping each tile's base region into a fresh canvas reconstructs the
    original image bit-for-bit
"""

import os
import sys

import torch

# Make this script runnable standalone: put the parent of `tiling/` on the path
# so the modules can resolve their `from .utils import ...` relative imports.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tiling.tile_splitter import Gen2_TileSplitter
from tiling.tile_masks import Gen2_TileMasks
from tiling.tile_merger import Gen2_TileMerger
from tiling.seam_fix import Gen2_SeamFix
from tiling.seam_merger import Gen2_SeamMerger


def _make_pattern_image(h: int, w: int) -> torch.Tensor:
    """Return a (1, H, W, 3) image where each pixel encodes (y, x) so we can
    spot-check that the splitter is reading the right pixels."""
    ys = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
    xs = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
    r = (ys / max(h - 1, 1))
    g = (xs / max(w - 1, 1))
    b = ((ys + xs) / max(h + w - 2, 1))
    img = torch.stack([r, g, b], dim=-1).unsqueeze(0)
    return img


def _assert_eq(actual, expected, label: str):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _assert_tensor_eq(actual: torch.Tensor, expected: torch.Tensor, label: str,
                      atol: float = 1e-6):
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label}: shape mismatch, expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
        )
    diff = (actual - expected).abs().max().item()
    if diff > atol:
        raise AssertionError(f"{label}: max abs diff {diff} > {atol}")


def scenario_clean_partition():
    label = "[scenario 1: 2048x2048, tile_size=1024, overlap_pct=0.25]"
    print(label)

    H, W = 2048, 2048
    img = _make_pattern_image(H, W)

    splitter = Gen2_TileSplitter()
    layout, tiles = splitter.split(img, tile_size=1024, overlap_pct=0.25)

    _assert_eq(layout["original_height"], 2048, f"{label} original_height")
    _assert_eq(layout["original_width"], 2048, f"{label} original_width")
    _assert_eq(layout["rows"], 2, f"{label} rows")
    _assert_eq(layout["cols"], 2, f"{label} cols")
    _assert_eq(layout["overlap_px_h"], 256, f"{label} overlap_px_h")
    _assert_eq(layout["overlap_px_w"], 256, f"{label} overlap_px_w")
    _assert_eq(len(layout["splits"]), 4, f"{label} num splits")
    _assert_eq(len(tiles), 4, f"{label} num tiles")

    for i, (s, tile) in enumerate(zip(layout["splits"], tiles)):
        _assert_eq(tuple(tile.shape), (1, 1536, 1536, 3), f"{label} tile[{i}] shape")
        _assert_eq(s["h"], 1536, f"{label} split[{i}] h")
        _assert_eq(s["w"], 1536, f"{label} split[{i}] w")
        _assert_eq(s["base_h"], 1024, f"{label} split[{i}] base_h")
        _assert_eq(s["base_w"], 1024, f"{label} split[{i}] base_w")
        _assert_eq(s["base_y_local"], 256, f"{label} split[{i}] base_y_local")
        _assert_eq(s["base_x_local"], 256, f"{label} split[{i}] base_x_local")
        _assert_eq(s["base_h_local"], 1024, f"{label} split[{i}] base_h_local")
        _assert_eq(s["base_w_local"], 1024, f"{label} split[{i}] base_w_local")

        # Base region of each tile must equal the corresponding image crop
        base_crop_in_tile = tile[:, 256:1280, 256:1280, :]
        base_crop_in_image = img[:, s["base_y"]:s["base_y"] + 1024,
                                    s["base_x"]:s["base_x"] + 1024, :]
        _assert_tensor_eq(base_crop_in_tile, base_crop_in_image,
                          f"{label} tile[{i}] base region")

    # Spot-check interior halos pull real image content from neighbors.
    # Tile 0 = (row=0, col=0). Its RIGHT halo strip [256:1280, 1280:1536]
    # (in tile-local coords) should equal image[0:1024, 1024:1280] (the leftmost
    # 256 px of column 1's base region).
    tile0 = tiles[0]
    right_halo = tile0[:, 256:1280, 1280:1536, :]
    expected_right = img[:, 0:1024, 1024:1280, :]
    _assert_tensor_eq(right_halo, expected_right, f"{label} tile[0] right halo")

    # Tile 0's BOTTOM halo strip [1280:1536, 256:1280] should equal
    # image[1024:1280, 0:1024].
    bottom_halo = tile0[:, 1280:1536, 256:1280, :]
    expected_bottom = img[:, 1024:1280, 0:1024, :]
    _assert_tensor_eq(bottom_halo, expected_bottom, f"{label} tile[0] bottom halo")

    # Spot-check outer halos are replicate-padded.
    # Tile 0's TOP halo [0:256, 256:1280] should be the first row of image
    # replicated downward 256 times. Verify by checking that all 256 rows are
    # identical.
    top_halo = tile0[:, 0:256, 256:1280, :]
    first_row = top_halo[:, 0:1, :, :]
    if not torch.equal(top_halo, first_row.expand_as(top_halo)):
        raise AssertionError(f"{label} tile[0] top halo is not constant-replicate")
    # And the first row of the top halo should equal image's first row.
    _assert_tensor_eq(top_halo[:, 0, :, :], img[:, 0, 0:1024, :],
                      f"{label} tile[0] top halo first-row == image row 0")

    # Tile 0's TOP-LEFT corner halo [0:256, 0:256] should be a constant block
    # equal to image[0, 0].
    corner = tile0[:, 0:256, 0:256, :]
    corner_val = img[:, 0:1, 0:1, :]
    if not torch.equal(corner, corner_val.expand_as(corner)):
        raise AssertionError(f"{label} tile[0] top-left corner halo is not a constant block")

    # Mask node
    masker = Gen2_TileMasks()
    (masks,) = masker.make_masks(layout, mask_blend_pixels=0)
    _assert_eq(len(masks), 4, f"{label} num masks")
    for i, m in enumerate(masks):
        _assert_eq(tuple(m.shape), (1, 1536, 1536), f"{label} mask[{i}] shape")
        plateau = m[:, 256:1280, 256:1280]
        if not torch.equal(plateau, torch.ones_like(plateau)):
            raise AssertionError(f"{label} mask[{i}] plateau is not all 1.0")
        # Sum of 1.0 cells must equal exactly 1024*1024.
        expected_sum = 1024 * 1024
        actual_sum = m.sum().item()
        if abs(actual_sum - expected_sum) > 1e-3:
            raise AssertionError(
                f"{label} mask[{i}] sum {actual_sum} != {expected_sum}"
            )

    # Reconstruction: paste each tile's base region back into a canvas at
    # (base_y, base_x) and verify it equals the original image.
    canvas = torch.zeros_like(img)
    for s, tile in zip(layout["splits"], tiles):
        base = tile[:, 256:1280, 256:1280, :]
        canvas[:, s["base_y"]:s["base_y"] + 1024,
                  s["base_x"]:s["base_x"] + 1024, :] = base
    _assert_tensor_eq(canvas, img, f"{label} reconstruction")

    print(f"{label} OK")


def scenario_last_tile_wins():
    label = "[scenario 2: 2500x2500, tile_size=1024, overlap_pct=0.25]"
    print(label)

    H, W = 2500, 2500
    img = _make_pattern_image(H, W)

    splitter = Gen2_TileSplitter()
    layout, tiles = splitter.split(img, tile_size=1024, overlap_pct=0.25)

    # n=3 wins (closest tile dim to 1024); base_h_floor = 833, remainder = 1.
    _assert_eq(layout["rows"], 3, f"{label} rows")
    _assert_eq(layout["cols"], 3, f"{label} cols")
    _assert_eq(layout["overlap_px_h"], int(round(0.25 * 833)),
               f"{label} overlap_px_h")
    _assert_eq(layout["overlap_px_w"], int(round(0.25 * 833)),
               f"{label} overlap_px_w")
    _assert_eq(len(layout["splits"]), 9, f"{label} num splits")
    _assert_eq(len(tiles), 9, f"{label} num tiles")

    overlap_px = layout["overlap_px_h"]
    # Tiles in last row OR last col have base_h or base_w = 834 (= 833 + 1).
    for s, tile in zip(layout["splits"], tiles):
        expected_base_h = 834 if s["row"] == 2 else 833
        expected_base_w = 834 if s["col"] == 2 else 833
        _assert_eq(s["base_h"], expected_base_h,
                   f"{label} split(r={s['row']},c={s['col']}) base_h")
        _assert_eq(s["base_w"], expected_base_w,
                   f"{label} split(r={s['row']},c={s['col']}) base_w")
        expected_tile_h = expected_base_h + 2 * overlap_px
        expected_tile_w = expected_base_w + 2 * overlap_px
        _assert_eq(tuple(tile.shape),
                   (1, expected_tile_h, expected_tile_w, 3),
                   f"{label} tile(r={s['row']},c={s['col']}) shape")

    # Base regions must form an exact partition (no overlap, no gap).
    covered = torch.zeros((H, W), dtype=torch.int32)
    for s in layout["splits"]:
        covered[s["base_y"]:s["base_y"] + s["base_h"],
                s["base_x"]:s["base_x"] + s["base_w"]] += 1
    if int(covered.min().item()) != 1 or int(covered.max().item()) != 1:
        raise AssertionError(
            f"{label} base regions do not partition image: "
            f"min={int(covered.min().item())} max={int(covered.max().item())}"
        )

    # Reconstruction via base regions
    canvas = torch.zeros_like(img)
    for s, tile in zip(layout["splits"], tiles):
        by = s["base_y_local"]
        bx = s["base_x_local"]
        bh = s["base_h_local"]
        bw = s["base_w_local"]
        base = tile[:, by:by + bh, bx:bx + bw, :]
        canvas[:, s["base_y"]:s["base_y"] + bh,
                  s["base_x"]:s["base_x"] + bw, :] = base
    _assert_tensor_eq(canvas, img, f"{label} reconstruction")

    # Masks
    masker = Gen2_TileMasks()
    (masks,) = masker.make_masks(layout, mask_blend_pixels=0)
    _assert_eq(len(masks), 9, f"{label} num masks")
    union = torch.zeros((H, W), dtype=torch.float32)
    for s, m in zip(layout["splits"], masks):
        bh = s["base_h_local"]
        bw = s["base_w_local"]
        by = s["base_y_local"]
        bx = s["base_x_local"]
        plateau = m[0, by:by + bh, bx:bx + bw]
        if not torch.equal(plateau, torch.ones_like(plateau)):
            raise AssertionError(
                f"{label} mask(r={s['row']},c={s['col']}) plateau != 1.0"
            )
        union[s["base_y"]:s["base_y"] + bh,
              s["base_x"]:s["base_x"] + bw] += plateau
    if not torch.equal(union, torch.ones_like(union)):
        raise AssertionError(
            f"{label} masks don't union to a single full-image cover "
            f"(min={union.min().item()}, max={union.max().item()})"
        )

    print(f"{label} OK")


def scenario_no_halo():
    label = "[scenario 3: 2048x2048, tile_size=1024, overlap_pct=0.0]"
    print(label)

    H, W = 2048, 2048
    img = _make_pattern_image(H, W)

    splitter = Gen2_TileSplitter()
    layout, tiles = splitter.split(img, tile_size=1024, overlap_pct=0.0)

    _assert_eq(layout["overlap_px_h"], 0, f"{label} overlap_px_h")
    _assert_eq(layout["overlap_px_w"], 0, f"{label} overlap_px_w")
    _assert_eq(len(tiles), 4, f"{label} num tiles")

    # No halo => every tile is exactly its base region.
    for i, (s, tile) in enumerate(zip(layout["splits"], tiles)):
        _assert_eq(tuple(tile.shape), (1, 1024, 1024, 3),
                   f"{label} tile[{i}] shape")
        _assert_eq(s["base_y_local"], 0, f"{label} split[{i}] base_y_local")
        _assert_eq(s["base_x_local"], 0, f"{label} split[{i}] base_x_local")
        expected = img[:, s["base_y"]:s["base_y"] + 1024,
                          s["base_x"]:s["base_x"] + 1024, :]
        _assert_tensor_eq(tile, expected, f"{label} tile[{i}] == image crop")

    # Mask with mask_blend_pixels=0 => all-ones masks (since base == frame).
    masker = Gen2_TileMasks()
    (masks,) = masker.make_masks(layout, mask_blend_pixels=0)
    for i, m in enumerate(masks):
        _assert_eq(tuple(m.shape), (1, 1024, 1024), f"{label} mask[{i}] shape")
        if not torch.equal(m, torch.ones_like(m)):
            raise AssertionError(f"{label} mask[{i}] is not all ones")

    print(f"{label} OK")


def scenario_mask_blend():
    label = "[scenario 4: mask_blend_pixels Gaussian feather is symmetric]"
    print(label)

    H, W = 2048, 2048
    img = _make_pattern_image(H, W)

    splitter = Gen2_TileSplitter()
    layout, _ = splitter.split(img, tile_size=1024, overlap_pct=0.25)

    masker = Gen2_TileMasks()
    (masks,) = masker.make_masks(layout, mask_blend_pixels=16)

    # Center of the base plateau should still be ~1.0; halo far from boundary
    # should still be ~0.0; samples exactly on the boundary should be ~0.5.
    m0 = masks[0][0]  # (1536, 1536)
    deep_center = m0[600, 600].item()
    far_halo = m0[20, 20].item()
    on_boundary = m0[256, 600].item()  # boundary y=256, well inside base in x

    if deep_center < 0.999:
        raise AssertionError(f"{label} deep center {deep_center} expected ~1.0")
    if far_halo > 0.001:
        raise AssertionError(f"{label} far halo {far_halo} expected ~0.0")
    if not (0.45 <= on_boundary <= 0.55):
        raise AssertionError(
            f"{label} on-boundary value {on_boundary} expected ~0.5"
        )

    print(f"{label} OK")


def scenario_merger_roundtrip():
    """Split-then-merge should reproduce the original image when tiles are
    passed through unchanged. blend_mode='none' is exact; the other modes are
    approximate (small numerical error from blending math)."""
    label = "[scenario 5: merger round-trip identity, 2048x2048, overlap 0.25]"
    print(label)

    H, W = 2048, 2048
    img = _make_pattern_image(H, W)

    splitter = Gen2_TileSplitter()
    layout, tiles = splitter.split(img, tile_size=1024, overlap_pct=0.25)

    masker = Gen2_TileMasks()
    (masks,) = masker.make_masks(layout, mask_blend_pixels=0)

    merger = Gen2_TileMerger()

    # blend_mode='none' with the explicit masks_list must be exact.
    (out_none,) = merger.merge(
        tile_layout=[layout],
        blend_mode=["none"],
        blend_strength=[1.0],
        seam_mode=["middle"],
        histogram_matching=[False],
        processed_tiles_image_list=list(tiles),
        masks_list=list(masks),
    )
    _assert_tensor_eq(out_none, img, f"{label} blend_mode='none' identity")

    # blend_mode='none' WITHOUT masks_list (merger builds default base masks).
    (out_none_default,) = merger.merge(
        tile_layout=[layout],
        blend_mode=["none"],
        blend_strength=[1.0],
        seam_mode=["middle"],
        histogram_matching=[False],
        processed_tiles_image_list=list(tiles),
        masks_list=None,
    )
    _assert_tensor_eq(out_none_default, img,
                      f"{label} blend_mode='none' default-mask identity")

    # blend_mode='linear': halos agree with base regions so normalized blending
    # should also reconstruct the original (within float tolerance).
    (out_linear,) = merger.merge(
        tile_layout=[layout],
        blend_mode=["linear"],
        blend_strength=[1.0],
        seam_mode=["middle"],
        histogram_matching=[False],
        processed_tiles_image_list=list(tiles),
        masks_list=None,
    )
    _assert_tensor_eq(out_linear, img, f"{label} blend_mode='linear' identity",
                      atol=1e-4)

    # blend_mode='gaussian'
    (out_gauss,) = merger.merge(
        tile_layout=[layout],
        blend_mode=["gaussian"],
        blend_strength=[0.5],
        seam_mode=["middle"],
        histogram_matching=[False],
        processed_tiles_image_list=list(tiles),
        masks_list=None,
    )
    _assert_tensor_eq(out_gauss, img, f"{label} blend_mode='gaussian' identity",
                      atol=1e-4)

    # blend_mode='multi_band' is more aggressive (sequential pyramid blend), so
    # we allow a slightly looser tolerance and check max-pixel diff.
    (out_mb,) = merger.merge(
        tile_layout=[layout],
        blend_mode=["multi_band"],
        blend_strength=[0.5],
        seam_mode=["middle"],
        histogram_matching=[False],
        processed_tiles_image_list=list(tiles),
        masks_list=None,
    )
    mb_diff = (out_mb - img).abs().max().item()
    if mb_diff > 5e-2:
        raise AssertionError(
            f"{label} blend_mode='multi_band' max diff {mb_diff} > 5e-2"
        )

    print(f"{label} OK")


def scenario_merger_last_tile_wins():
    label = "[scenario 6: merger round-trip with last_tile_wins, 2500x2500]"
    print(label)

    H, W = 2500, 2500
    img = _make_pattern_image(H, W)

    splitter = Gen2_TileSplitter()
    layout, tiles = splitter.split(img, tile_size=1024, overlap_pct=0.25)
    merger = Gen2_TileMerger()

    (out,) = merger.merge(
        tile_layout=[layout],
        blend_mode=["none"],
        blend_strength=[1.0],
        seam_mode=["middle"],
        histogram_matching=[False],
        processed_tiles_image_list=list(tiles),
        masks_list=None,
    )
    _assert_tensor_eq(out, img, f"{label} blend_mode='none' identity")

    print(f"{label} OK")


def scenario_merger_optimal_seam():
    """Optimal-seam DP must run without error and still reproduce the original
    when fed identical (un-edited) tiles."""
    label = "[scenario 7: merger optimal-seam DP, 2048x2048]"
    print(label)

    H, W = 2048, 2048
    img = _make_pattern_image(H, W)

    splitter = Gen2_TileSplitter()
    layout, tiles = splitter.split(img, tile_size=1024, overlap_pct=0.25)
    merger = Gen2_TileMerger()

    (out,) = merger.merge(
        tile_layout=[layout],
        blend_mode=["gaussian"],
        blend_strength=[0.5],
        seam_mode=["optimal"],
        histogram_matching=[False],
        processed_tiles_image_list=list(tiles),
        masks_list=None,
    )
    _assert_tensor_eq(out, img, f"{label} optimal-seam identity", atol=1e-3)

    print(f"{label} OK")


def scenario_merger_none_feathered_masks():
    """Regression: blend_mode='none' fed feathered masks (mask_blend_pixels>0)
    must NOT dim the result at tile-grid seams.

    With the old alpha-overlay 'canvas*(1-m)+tile*m' formula, identity round-
    trip showed max diff ~0.72 with a clear seam-cross pattern. With the fix
    (normalized weighted average), identity should be exact regardless of how
    much feathering Gen2_TileMasks applies.
    """
    label = "[scenario 8: merger blend_mode='none' with feathered masks]"
    print(label)

    H, W = 2048, 2048
    img = _make_pattern_image(H, W)

    splitter = Gen2_TileSplitter()
    layout, tiles = splitter.split(img, tile_size=1024, overlap_pct=0.25)

    masker = Gen2_TileMasks()
    merger = Gen2_TileMerger()

    for mask_blend_pixels in (0, 4, 16, 32):
        (masks,) = masker.make_masks(layout, mask_blend_pixels=mask_blend_pixels)
        (out,) = merger.merge(
            tile_layout=[layout],
            blend_mode=["none"],
            blend_strength=[1.0],
            seam_mode=["middle"],
            histogram_matching=[False],
            processed_tiles_image_list=list(tiles),
            masks_list=list(masks),
        )
        _assert_tensor_eq(
            out,
            img,
            f"{label} mask_blend_pixels={mask_blend_pixels} identity",
            atol=1e-4,
        )

    print(f"{label} OK")


def _make_alt_image(h, w):
    """Different gradient than _make_pattern_image so we can distinguish them."""
    ys = torch.arange(h, dtype=torch.float32).view(h, 1).expand(h, w)
    xs = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
    r = 1.0 - ys / max(h - 1, 1)
    g = 1.0 - xs / max(w - 1, 1)
    b = (ys - xs).abs() / max(h - 1, w - 1, 1)
    return torch.stack([r, g, b], dim=-1).unsqueeze(0)


def scenario_seam_fix_geometry():
    label = "[scenario 9: SeamFix geometry, 2048x2048, tile_size=1024, overlap_pct=0.25]"
    print(label)

    H, W = 2048, 2048
    original = _make_pattern_image(H, W)
    merged = _make_alt_image(H, W)

    splitter = Gen2_TileSplitter()
    layout, _ = splitter.split(original, tile_size=1024, overlap_pct=0.25)

    sf = Gen2_SeamFix()
    seam_layout, seam_tiles, seam_masks = sf.build(
        original_image=original, merged_image=merged, tile_layout=layout,
        seam_strip_width=128, mask_blend_pixels=32,
    )

    _assert_eq(seam_layout["verticals"], [1024], f"{label} vertical seams")
    _assert_eq(seam_layout["horizontals"], [1024], f"{label} horizontal seams")
    _assert_eq(seam_layout["seam_tile_thickness_h"], 640, f"{label} thickness h")
    _assert_eq(seam_layout["seam_tile_thickness_w"], 640, f"{label} thickness w")

    tiles_meta = seam_layout["tiles"]
    _assert_eq(len(tiles_meta), 5, f"{label} total tile count")
    _assert_eq(len(seam_tiles), 5, f"{label} seam_tiles list length")
    _assert_eq(len(seam_masks), 5, f"{label} seam_masks list length")

    intersections = [m for m in tiles_meta if m["kind"] == "intersection"]
    arms = [m for m in tiles_meta if m["kind"] == "arm"]
    v_arms = [m for m in arms if m["axis"] == "vertical"]
    h_arms = [m for m in arms if m["axis"] == "horizontal"]
    _assert_eq(len(intersections), 1, f"{label} intersection count")
    _assert_eq(len(v_arms), 2, f"{label} vertical arm count")
    _assert_eq(len(h_arms), 2, f"{label} horizontal arm count")

    inter = intersections[0]
    _assert_eq((inter["y"], inter["x"], inter["h"], inter["w"]),
               (704, 704, 640, 640), f"{label} intersection rect")

    v_rects = sorted([(m["y"], m["x"], m["h"], m["w"]) for m in v_arms])
    _assert_eq(v_rects, [(0, 704, 704, 640), (1344, 704, 704, 640)],
               f"{label} vertical-arm rects")
    h_rects = sorted([(m["y"], m["x"], m["h"], m["w"]) for m in h_arms])
    _assert_eq(h_rects, [(704, 0, 640, 704), (704, 1344, 640, 704)],
               f"{label} horizontal-arm rects")

    coverage = torch.zeros((H, W), dtype=torch.int32)
    for m in tiles_meta:
        coverage[m["y"]:m["y"] + m["h"], m["x"]:m["x"] + m["w"]] += 1
    if int(coverage.max().item()) > 1:
        raise AssertionError(f"{label} seam tiles overlap (max coverage {int(coverage.max().item())})")

    for i, (m, t, mask) in enumerate(zip(tiles_meta, seam_tiles, seam_masks)):
        _assert_eq(tuple(t.shape), (1, m["h"], m["w"], 3), f"{label} tile[{i}] shape")
        _assert_eq(tuple(mask.shape), (1, m["h"], m["w"]), f"{label} mask[{i}] shape")

    print(f"{label} OK")


def scenario_seam_fix_content():
    label = "[scenario 10: SeamFix tile content (strip replacement)]"
    print(label)

    H, W = 2048, 2048
    original = _make_pattern_image(H, W)
    merged = _make_alt_image(H, W)

    splitter = Gen2_TileSplitter()
    layout, _ = splitter.split(original, tile_size=1024, overlap_pct=0.25)
    sf = Gen2_SeamFix()
    seam_layout, seam_tiles, seam_masks = sf.build(
        original_image=original, merged_image=merged, tile_layout=layout,
        seam_strip_width=128, mask_blend_pixels=32,
    )

    for i, meta in enumerate(seam_layout["tiles"]):
        y, x, h, w = meta["y"], meta["x"], meta["h"], meta["w"]
        hard = torch.zeros((h, w), dtype=torch.bool)
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

        t = seam_tiles[i]
        merged_crop = merged[:, y:y + h, x:x + w, :]
        original_crop = original[:, y:y + h, x:x + w, :]

        outside = ~hard
        if outside.any():
            _assert_tensor_eq(t[:, outside, :], merged_crop[:, outside, :],
                              f"{label} tile[{i}] outside-strip == merged")
        if hard.any():
            _assert_tensor_eq(t[:, hard, :], original_crop[:, hard, :],
                              f"{label} tile[{i}] inside-strip == original")

    print(f"{label} OK")


def scenario_seam_merger_passthrough():
    """If we feed the SeamFix tiles straight back into SeamMerger with hard
    masks (mask_blend_pixels=0), the final image should be:
      - merged image where no seam strip covers it
      - original image inside seam strips (because SeamFix replaced them)
    """
    label = "[scenario 11: SeamMerger passthrough composition]"
    print(label)

    H, W = 2048, 2048
    original = _make_pattern_image(H, W)
    merged = _make_alt_image(H, W)

    splitter = Gen2_TileSplitter()
    layout, _ = splitter.split(original, tile_size=1024, overlap_pct=0.25)
    sf = Gen2_SeamFix()
    seam_layout, seam_tiles, seam_masks = sf.build(
        original_image=original, merged_image=merged, tile_layout=layout,
        seam_strip_width=128, mask_blend_pixels=0,
    )

    sm = Gen2_SeamMerger()
    (final,) = sm.merge(
        merged_image=[merged],
        seam_layout=[seam_layout],
        blend_mode=["linear"],
        blend_strength=[0.5],
        histogram_matching=[False],
        processed_seam_tiles_list=list(seam_tiles),
        seam_tiles_masks_list=list(seam_masks),
    )

    expected = merged.clone()
    for meta in seam_layout["tiles"]:
        y, x, h, w = meta["y"], meta["x"], meta["h"], meta["w"]
        for strip in meta["strips"]:
            center = int(strip["center_local"])
            half = int(strip["width"]) // 2
            if strip["axis"] == "vertical":
                lo = max(0, center - half)
                hi = min(w, center + half)
                expected[:, y:y + h, x + lo:x + hi, :] = original[:, y:y + h, x + lo:x + hi, :]
            else:
                lo = max(0, center - half)
                hi = min(h, center + half)
                expected[:, y + lo:y + hi, x:x + w, :] = original[:, y + lo:y + hi, x:x + w, :]

    _assert_tensor_eq(final, expected, f"{label} final == expected", atol=1e-5)
    print(f"{label} OK")


def scenario_seam_merger_modes_run():
    """Smoke-run all three blend_modes + histogram_matching to make sure
    nothing crashes; final image must be in [0, 1]."""
    label = "[scenario 12: SeamMerger modes/histogram run cleanly]"
    print(label)

    H, W = 2048, 2048
    original = _make_pattern_image(H, W)
    merged = _make_alt_image(H, W)

    splitter = Gen2_TileSplitter()
    layout, _ = splitter.split(original, tile_size=1024, overlap_pct=0.25)
    sf = Gen2_SeamFix()
    seam_layout, seam_tiles, seam_masks = sf.build(
        original_image=original, merged_image=merged, tile_layout=layout,
        seam_strip_width=128, mask_blend_pixels=32,
    )

    sm = Gen2_SeamMerger()
    for mode in ("linear", "gaussian", "multi_band"):
        for hist in (False, True):
            (final,) = sm.merge(
                merged_image=[merged],
                seam_layout=[seam_layout],
                blend_mode=[mode],
                blend_strength=[0.5],
                histogram_matching=[hist],
                processed_seam_tiles_list=list(seam_tiles),
                seam_tiles_masks_list=list(seam_masks),
            )
            _assert_eq(tuple(final.shape), tuple(merged.shape),
                       f"{label} mode={mode} hist={hist} shape")
            mn = final.min().item()
            mx = final.max().item()
            if mn < -1e-3 or mx > 1.0 + 1e-3:
                raise AssertionError(
                    f"{label} mode={mode} hist={hist} out of range: min={mn} max={mx}"
                )

    print(f"{label} OK")


def main():
    scenario_clean_partition()
    scenario_last_tile_wins()
    scenario_no_halo()
    scenario_mask_blend()
    scenario_merger_roundtrip()
    scenario_merger_last_tile_wins()
    scenario_merger_optimal_seam()
    scenario_merger_none_feathered_masks()
    scenario_seam_fix_geometry()
    scenario_seam_fix_content()
    scenario_seam_merger_passthrough()
    scenario_seam_merger_modes_run()
    print("\nAll scenarios passed.")


if __name__ == "__main__":
    main()
