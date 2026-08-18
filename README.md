# ComfyUI-Gen2 Custom Nodes

A general-purpose ComfyUI custom node pack collecting the sampling fixes, model-compatibility patches, and quality-of-life utilities we use day to day. The pack is organized into independent sections that each load on their own — if one section's optional dependency is missing, the rest keep working.

## What's in here

- **Flux.2 Fun ControlNet** — branch-only support for Alibaba PAI's official `FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors`, using clone-local ComfyUI block replacements, native model management, exact 260-channel control context packing, reference tokens, schedules, and experimental multi-control composition.
- **Gen2 Sampling** — clone-local model patches and compatibility loaders, including the Flux.2 [klein] fix and an Ideogram4 AI-Toolkit joint MODEL/Qwen text-encoder LoRA loader.
- **Ideogram4 Trigger Activator V9** — strict four-slot `[4,H]` trigger embeddings plus 36 independent rank-4 `mlp.down_proj` module-LoRA hooks, with native Ideogram4/Qwen3-VL identity gating and standard CONDITIONING output.
- **Tiling** — tile-based workflow nodes: auto-grid splitting with overlap halos, per-tile masks, seam-aware merging, and a two-pass seam-fix denoise (great for high-res inpaint and tiled upscaling).
- **API Panels** — a pair of configurable nodes (`Gen2 Input Panel` / `Gen2 Output Panel`) that replace a workflow's scattered `INPUT_*`/`OUTPUT_*` constant nodes. Click **Configure** to define named, typed parameters; each name becomes a typed slot and the API-export key. Designed for driving ComfyUI via the API export. Works on both the legacy LiteGraph and Nodes 2.0 (Vue) frontends.
- **Utilities** — small QoL nodes: string replace, checkerboard generator, DWpose-with-thresholds.
- **QwenImage ControlNet** *(outdated)* — the pack's original purpose: a self-contained QwenImage ControlNet pipeline matching VideoX-Fun's diffusers output. Kept for backward compatibility; node names are suffixed **(outdated)**.

Most sections integrate with ComfyUI's own model loading (Load Diffusion Model / Load CLIP / Load VAE) and only override the specific inference behavior they need.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/petmycat/ComfyUI-gen2.git
cd ComfyUI-gen2
pip install -r requirements.txt
```

This release requires **ComfyUI v0.28.0 or newer**. The Flux.2 Fun integration is pinned and validated against ComfyUI commit `700821e1364eaab0e8f21c538a2131719fec57bf`.

Sections have independent, optional prerequisites — install only what you use:

- **Flux.2 Fun ControlNet** — no Python package beyond ComfyUI. Put the official `FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors` in `ComfyUI/models/controlnet/`.
- **Gen2 Sampling / Utilities (string, checkerboard)** — no extra Python dependencies.
- **Tiling** — needs `scipy` (Gaussian mask feathering). Without it you'll see `[Gen2] Tiling nodes not available: ...` and everything else still loads.
- **DWpose utility** — needs `comfyui_controlnet_aux`.
- **QwenImage ControlNet (outdated)** — needs [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) (and [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) for GGUF models), plus the tokenizer from [Qwen-Image-2512](https://huggingface.co/alibaba-pai/Qwen-Image-2512) placed in `ComfyUI/models/gen2/qwen_2512_tokenizer/`.

## Credits

- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** - The modular diffusion GUI this pack extends.
- **[VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)** - Flux.2 Fun mathematical oracle pinned to commit `248ab0ac0ebc48f0b4ae43ceb2d7ded24cc907bb`, and the source of the legacy QwenImage section.
- **[Alibaba PAI FLUX.2 Fun ControlNet Union](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union)** - Official 2602 checkpoint. Model weights remain subject to their upstream license and usage terms.

## Nodes

### Flux.2 Fun ControlNet

| Node | Description |
|------|-------------|
| **Load Flux2 Fun ControlNet** | Strictly loads the official 2602 checkpoint from `models/controlnet`. It validates the 76-key architecture profile and always verifies SHA256 `516532a885d12ae84bb3c6b24ef4816ac05ffa1c9c7b93476f74652eb0a7a794`. |
| **Prepare Flux2 Fun Control** | Uses ComfyUI's Flux.2 VAE and packs `[control_latents(128), preserved_mask(4), inpaint_latents(128)]`. Connect the sampling `LATENT` to `target_latent` so the control canvas exactly matches the main image tokens. White input mask means repaint; absent image branches are direct zero latents. |
| **Apply Flux2 Fun Control** | Clones a Flux.2 Dev `MODEL`, registers the managed control branch once, and composes clone-local replacements at double blocks 0/2/4/6. Supports strength and start/end percentages. |
| **Combine Flux2 Fun Controls (experimental)** | Creates a deterministic immutable control group. Multi-control remains experimental until the complete GPU matrix passes. |

Only **Flux.2 Dev + the 2602 checkpoint** is accepted. Flux.1, Flux.2 Klein, the older Union checkpoint, and shape-incompatible derivatives are rejected. The implementation does not replace `Flux.forward_orig`, does not globally monkey-patch ComfyUI, and never resizes token tensors heuristically.

Reference images are supported through ComfyUI's `reference_image_num_tokens`; the control context receives zero-valued reference positions and runs over the complete image-token sequence. `index_timestep_zero` modulation regions are consumed from the real ComfyUI block payload rather than reimplemented globally.

A template workflow is provided at `workflows/flux2_fun_control_2602.json`, with an API fragment at `assets/flux2_fun_control_2602_api.json`. The development machine does not contain the production GPU/weight environment, so real-weight end-to-end parity is explicitly **not yet claimed**; the opt-in harness and acceptance matrix live under `tests/`.

### Compatibility matrix

| Component | Status |
|---|---|
| ComfyUI v0.28.0 / commit `700821e...` | Targeted runtime |
| Newer ComfyUI | Expected, but rerun the GPU matrix after core Flux changes |
| ComfyUI v0.10.0 | Not supported |
| Official 2602 BF16 checkpoint | Supported |
| BF16 / FP16 control compute | Implemented; production GPU validation required |
| FP8 base + BF16/FP16 control | Architecture supported; production GPU validation required |
| Multiple controls | Experimental |

### Gen2 Sampling

| Node | Description |
|------|-------------|
| **Gen2 Flux.2 Klein Fix (#12905 revert)** | Restores pre-`44f1246` (PR #12905) Flux.2 [klein] sampling on ComfyUI ≥ v0.17.0. That commit's KV-cache refactor changed Klein output for normal `index` reference-latent workflows (e.g. masked inpaint) even though its new code paths are gated off. This node clones the model and, **only during its own sampling**, swaps the flux `forward_orig`/`_forward` back to the last-good (v0.16.4) implementation, then restores them — no ComfyUI core files are edited, so it survives `git pull`. Wire it `Load Diffusion Model → Gen2 Flux.2 Klein Fix → guider`. Don't combine it on the same model with `FluxKVCache` or the newer flux2 edit (`index_timestep_zero`) features, which are exactly what it reverts. |
| **Ideogram4 AI Toolkit LoRA Loader** | Stock-compatible MODEL + CLIP LoRA loader for ordinary joint weight-space Ideogram4 transformer/Qwen LoRAs. It is **not** the loader for V9 trigger TE artifacts. V9 TE files contain 36 trigger-masked, encode-scoped `mlp.down_proj` module-LoRA pairs and must use the V9 nodes below. |

### Ideogram4 Trigger Activator V9

The V9 path exposes exactly five nodes: embedding loader, TE module-LoRA loader, activator composer, trigger text encoder, and diagnostics. Each trigger occurrence expands into four virtual slots, and the default `semantic_only` mode applies both the interpolated `[4,4096]` embedding and all 36 rank-4/alpha-4 TE module-LoRAs. Legacy single-token/shared residual/tap artifacts and `full`/`tap_only` modes are rejected.

Formal encoding is fail-closed: only a native backend identifiable as Ideogram4 + Qwen3-VL with 36 hookable `mlp.down_proj` layers and compatible MRoPE interfaces is accepted. Flux/Klein Qwen3-8B is explicitly unsupported even when its hidden size and layer count match. Real A2 artifact parity and visual smoke remain a release gate until matching artifacts, tokenizer/checkpoint, and golden tensors are available; synthetic contract/lifecycle tests are included locally.

### QwenImage ControlNet (outdated)

> These nodes are kept for backward compatibility and are no longer the focus of
> this pack. Their display names are suffixed **(outdated)** in the node menu.

| Node | Description |
|------|-------------|
| **Gen2 Load QwenImage ControlNet (outdated)** | Load ControlNet weights |
| **Gen2 Load QwenImage VAE (outdated)** | Load VAE with VideoX-compatible config |
| **Gen2 Apply QwenImage ControlNet (outdated)** | Prepare control context and wrap model |
| **Gen2 QwenImage Text Encode (outdated)** | VideoX-style text encoding (use instead of CLIPTextEncode) |
| **Gen2 Load QwenImage LoRA (outdated)** | Load LoRA for VideoX-style merging |
| **Gen2 QwenImage Control Sampler (outdated)** | VideoX-compatible sampling with True CFG |

### Utilities

| Node | Description |
|------|-------------|
| **Gen2 DWpose with Threshold** | DWpose detector with configurable confidence thresholds for body/hand/face keypoints |
| **Gen2 StringReplace** | Replace all occurrences of a search string with a replacement string (case-sensitive) |
| **Gen2 Checkerboard** | Generate a checkerboard pattern image (1px black & white squares) at specified width × height |

### Tiling

A pair of nodes for tile-based image workflows (e.g. high-res inpaint, tiled upscaling). Tiles share a fixed-thickness overlap halo of surrounding context, while each tile's "owned" base region exactly partitions the original image — so the per-tile masks union to a full-image cover with no double-coverage.

| Node | Description |
|------|-------------|
| **Gen2 Tile Splitter** | Auto-grid an image into uniform tiles with an overlap halo. Inputs: `image`, `tile_size`, `overlap_pct`. Outputs: `tile_layout` (`GEN2_TILE_LAYOUT`) and `tiles_image_list` (list of tile images, row-major). |
| **Gen2 Tile Masks** | Build per-tile masks selecting each tile's owned base region. Inputs: `tile_layout`, `mask_blend_pixels`. Output: `masks_list` (list of `MASK`, same order as the splitter's tiles). Each mask has value `1.0` over the base region and `0.0` over the halo, with optional Gaussian feathering. |
| **Gen2 Tile Merger** | Recombine processed tiles back into a single image. Inputs: `tile_layout`, `blend_mode`, `blend_strength`, `seam_mode`, `histogram_matching`, optional `processed_tiles_image_list`, optional `masks_list`. Output: `merged_image`. |
| **Gen2 Seam Fix** | Build seam-targeted tiles + strip masks from a first-pass merged image, for a second-pass denoise that smooths the seams between regenerated bases. Inputs: `original_image`, `merged_image`, `tile_layout`, `seam_strip_width`, `mask_blend_pixels`. Outputs: `seam_layout` (`GEN2_SEAM_LAYOUT`), `seam_tiles`, `seam_tiles_masks`. |
| **Gen2 Seam Merger** | Composite regenerated seam tiles back onto the first-pass merged image. Inputs: `merged_image`, `seam_layout`, `blend_mode`, `blend_strength`, `histogram_matching`, optional `processed_seam_tiles_list`, optional `seam_tiles_masks_list`. Output: `final_image`. |

#### Splitter algorithm

- `n_rows = argmin_{n≥1} \|H/n − tile_size\|`, same for `n_cols`. No user knob — the grid is chosen automatically to minimize the distance from the target tile size.
- Base regions exactly partition the original image: `base_h = floor(H/n_rows)` for the first `n_rows−1` rows; the last row absorbs any remainder. Same for columns.
- `overlap_px = round(overlap_pct × base_dim_floor)` on each side. Every tile's expanded crop is `(base_h + 2·overlap_px_h) × (base_w + 2·overlap_px_w)`. Where the expansion falls outside the image, content is `replicate`-padded.
- Example: 2048×2048 image with `tile_size=1024`, `overlap_pct=0.25` → 2×2 grid, four 1536×1536 tiles each carrying a 1024×1024 base region surrounded by a 256 px halo.

#### `GEN2_TILE_LAYOUT`

A dict carrying everything downstream nodes need to reconstruct the partition:

```
{
  "original_height": int, "original_width": int,
  "rows": int, "cols": int,
  "overlap_px_h": int, "overlap_px_w": int,
  "splits": [
    {
      "row": int, "col": int,
      "y": int, "x": int, "h": int, "w": int,            # expanded crop, image-space
      "base_y": int, "base_x": int, "base_h": int, "base_w": int,   # owned region, image-space
      "base_y_local": int, "base_x_local": int,          # owned region, expanded-tile-local
      "base_h_local": int, "base_w_local": int,
    },
    ...
  ]
}
```

#### Merger blend modes

| `blend_mode` | Behavior |
|---|---|
| `none`        | **Base-only paste.** Halos are discarded; each tile contributes only its base region. The fast path for the masked-inpaint workflow (since halos are supposed to be unchanged context). When the optional `masks_list` is wired in, those masks are used directly (any per-tile shape, e.g. feathered base masks from `Gen2 Tile Masks`). |
| `linear`      | **Normalized weighted average.** Per-tile linear-falloff mask, with the weighted average computed across all tiles covering each pixel. Identity-preserving in the round-trip test. |
| `gaussian`    | Same as linear but with a Gaussian falloff whose sigma is `blend_strength × max(overlap_px) / 2`. The Gaussian plateau is inflated by `3σ` so the base region remains at `1.0` and the falloff lives entirely inside the halo. |
| `multi_band`  | **Sequential Laplacian-pyramid blend.** Raster-order traversal: the first tile is pasted at full strength, then each subsequent tile pyramid-blends into the canvas using its Gaussian mask. Highest quality on high-contrast tile seams; slowest mode. |

`seam_mode = "optimal"` runs a per-pixel DP min-cut through the `2·overlap_px` shared band between each adjacent tile pair and adjusts the masks to follow it. `seam_mode = "middle"` uses the natural geometric falloff with no DP. The DP loop is naive (Python-level inner loop) — fine for typical overlap sizes, slow for very large overlaps.

`histogram_matching = True` applies a Reinhard mean/std color transfer to each tile (except the first column) using the left neighbor's `2·overlap_px` left-edge band as the reference. Useful when each tile is sampled independently and tile-to-tile color drift is visible.

#### Two-pass seam-fix workflow

Even with the merger's normalized weighted average, regenerated tile bases can disagree at the seams (each tile was sampled independently, so the same image-space pixel can look different inside tile A's halo vs tile B's base). The merge smoothly blends them, but the blend itself can show ghosting or texture mismatch at the seam line. `Gen2_SeamFix` + `Gen2_SeamMerger` add a second-pass denoise targeted at exactly those seam strips:

```
Pass 1 (existing):
    Splitter → TileMasks → sampler → TileMerger (none)  → merged_image
                                                            │
Pass 2 (new):                                              │
    original_image ┐                                       │
    merged_image ──┼→ SeamFix → sampler → SeamMerger ──→ final_image
    tile_layout ───┘            (seam tiles)   ↑
                                merged_image ──┘
```

`Gen2_SeamFix` carves the union of seam neighborhoods into:

- One **intersection tile** per (vertical seam × horizontal seam) crossing — a square of side `seam_strip_width + 2·overlap_px` centered on the crossing.
- One **arm tile** per seam segment between consecutive intersections (or image edge ↔ first/last intersection) — a long rectangle of the same thickness, running along the seam axis.

For a 2×2 grid (2048×2048, `tile_size=1024`, `overlap_pct=0.25`, `seam_strip_width=128`):
- 1 intersection at image `[704:1344, 704:1344]` (640×640)
- 4 arms: top `[0:704, 704:1344]`, bottom `[1344:2048, 704:1344]`, left `[704:1344, 0:704]`, right `[704:1344, 1344:2048]`

All 5 tiles tile the cross-shaped seam neighborhood with **no overlap**, so the seam merger doesn't need to renormalize across overlapping tiles.

Each seam tile's image content = the merged image cropped at the tile rect, with the strip region **overwritten by the original image's content** at the same image-space location. That gives the sampler a clean inpaint init at the strip and the regenerated base content as conditioning context.

Each seam tile's mask is a strip (single rectangle for arms, vertical-strip-OR-horizontal-strip cross for intersections), Gaussian-feathered with `mask_blend_pixels`.

`Gen2_SeamMerger` composites each regenerated seam tile back onto the merged image using one of:

- `linear` / `gaussian` — direct alpha composite, `final = merged·(1−mask) + seam·mask`. Identical for both modes on non-overlapping seam tiles (the mode names are kept for symmetry with `Gen2_TileMerger`).
- `multi_band` — Laplacian-pyramid blend per seam tile against the canvas. Safe here (unlike in the first-pass merger) because the seam tile's non-strip region equals the canvas content, so there's no low-frequency disagreement to ring on. `blend_strength` controls pyramid depth (0 → 1 level, 1 → 6 levels).

`histogram_matching = True` Reinhard-matches each seam tile's strip-region content to the surrounding (non-strip) merged-image context, ensuring the regenerated strip blends color-consistently with the existing canvas.

#### `GEN2_SEAM_LAYOUT`

A dict carrying the seam-tile geometry:

```
{
  "original_height": int, "original_width": int,
  "seam_strip_width": int, "mask_blend_pixels": int,
  "seam_tile_thickness_h": int, "seam_tile_thickness_w": int,
  "verticals": [int, ...],    # x-coords of vertical seams
  "horizontals": [int, ...],  # y-coords of horizontal seams
  "tiles": [
    {
      "kind": "intersection" | "arm",
      "axis": "vertical" | "horizontal" | None,    # None for intersection
      "y": int, "x": int, "h": int, "w": int,      # image-space crop
      "strips": [                                  # 1 entry for arm, 2 for intersection
        {"axis": "vertical" | "horizontal",
         "center_local": int, "width": int},
        ...
      ],
      "seam_axes_coords": [int, ...]               # coords of the seam(s) this tile fixes
    },
    ...
  ]
}
```

#### Verifying the tiling nodes

A standalone smoke test lives at `tiling/_smoke_test.py`. It covers clean partitions, the `last_tile_wins` remainder case, the no-halo edge case, Gaussian-feathered masks, and merger round-trip identity for `none`/`linear`/`gaussian`/`multi_band` blend modes and `optimal` seam:

```
python ComfyUI/custom_nodes/ComfyUI-gen2/tiling/_smoke_test.py
```

The test asserts every tile's base region equals the corresponding image crop, halos pull the right content (real image for interior sides, replicate-pad for outer sides), masks have a `1.0` plateau exactly over the base region, stamping each tile's base region back into a fresh canvas reconstructs the original image bit-for-bit, and splitting then merging an image reproduces the original (exactly for `blend_mode="none"`, within float tolerance for the other modes).

## API Panels

A pair of configurable nodes that collapse a workflow's scattered `INPUT_*`/`OUTPUT_*` constant nodes into one panel each — for driving ComfyUI as a backend via the API export.

| Node | Description |
|------|-------------|
| **Gen2 Input Panel** | Click **Configure** to define named, typed output parameters (STRING / INT / FLOAT / BOOLEAN / IMAGE). Each name becomes a typed output slot **and** the API-export key. Wire its `PANEL_LINK` output into a Gen2 Output Panel to bind the pair. |
| **Gen2 Output Panel** | Click **Configure** to define named, typed input parameters. IMAGE inputs are saved to the output folder and their URLs returned via `/history` (like SaveImage). Wire a Gen2 Input Panel's `PANEL_LINK` output into this node's `PANEL_LINK` input. |

Up to 32 parameters per panel. Supported types: `STRING`, `INT`, `FLOAT`, `BOOLEAN`, `IMAGE`.

### Parameter properties

Each parameter has a **name** (the API-export key + slot label), a **type**, and a **default** (can be empty/null = no default). INT and FLOAT parameters additionally accept **min**, **max**, and **step**:

- **min / max** — the accepted value range. At execute time, an out-of-range value **interrupts the workflow** with a clear error message (e.g. `Gen2_InputPanel: parameter 'strength' = 1.5 is above max 1.0`), rather than silently clamping.
- **step** — the UI snapping increment and documentation value (e.g. `0.05` means `1.00, 1.05, 1.10, …`).
- **default** — the value used when no runtime value is provided. `null` means "no default" (the parameter must be provided). **Exporting the workflow/API always yields the default values, not whatever was set during a run** — the node's per-parameter widgets serialize defaults, so your exported workflow JSON is a clean template.

### How it looks in the API export

After configuring an Input Panel with `seed` (INT, default 0, min 0, max 999, step 1), `lora` (STRING, default `"f2k_q1Q2me1X2E_v1.safetensors"`), `loraStrength` (FLOAT, default 1.0, min 0, max 2, step 0.05), `genMode` (BOOLEAN, default false), `imageUrl` (IMAGE, default null), the API-export JSON is:

```json
"Gen2_InputPanel": {
  "inputs": {
    "_config": "[{\"name\":\"seed\",\"type\":\"INT\",\"default\":0,\"min\":0,\"max\":999,\"step\":1},...]",
    "seed": 0,
    "lora": "f2k_q1Q2me1X2E_v1.safetensors",
    "loraStrength": 1.0,
    "genMode": false,
    "imageUrl": null
  },
  "class_type": "Gen2_InputPanel"
}
```

Note the values are the **defaults**, not runtime values. An external frontend scans for `class_type == "Gen2_InputPanel"` and reads/writes the parameter keys directly — no more scanning node titles for `INPUT_*` prefixes.

### JSON schema output

The Gen2 Output Panel shows a **read-only JSON schema textbox** on its node body. When connected to an Input Panel via `PANEL_LINK` and the workflow is run, the textbox is populated with a JSON string listing every parameter's name, type, default, and range/step (for numeric types). Click the textbox (or the **Copy** button) to copy it for use elsewhere:

```json
[
  {
    "name": "strength",
    "type": "FLOAT",
    "default": 0.8,
    "min": 0.0,
    "max": 1.0,
    "step": 0.05
  },
  {
    "name": "prompt",
    "type": "STRING",
    "default": null
  }
]
```

The schema is sourced from the Input Panel's config (carried via `PANEL_LINK`), so it always reflects the input side's definitions.

### Compatibility

The backend uses the V3 node API (`io.ComfyNode` + `accept_all_inputs=True`); the frontend extension (`web/js/gen2Panels.js`, shipped via `WEB_DIRECTORY`) provides the Configure popup, per-parameter widgets (with range/step snapping for numbers, upload for images), and the JSON schema textbox. Works on both the legacy LiteGraph canvas and the Nodes 2.0 (Vue) frontend. Requires ComfyUI ≥ v0.10.0 (tested on v0.10.0 and v0.26.0).

## Dtype Support

Supports multiple precision modes:
- **bf16/fp16** - Full precision models
- **fp8** - Quantized models (automatic compute dtype detection)
- **GGUF** - Quantized models via ComfyUI-GGUF

## TODO

- [ ] Add node parameter explanations for better user support (document what each parameter does in every node)
- [ ] Integrate custom Load VAE node into ComfyUI system and add latent image input to sampler node
- [ ] Decouple ControlNet node and sampler node
- [ ] Add start and end step parameters to sampler node
- [x] Reorganize code for better maintenance — split into `qwenimage/` (core + nodes) and `misc_nodes/` (pose, string utils)
- [x] Tiling: `Gen2_TileSplitter` and `Gen2_TileMasks` with the auto-partition + halo-expansion algorithm
- [x] Tiling: `Gen2_TileMerger` with `none` / `linear` / `gaussian` / `multi_band` blend modes, optimal-seam DP, and Reinhard histogram matching
- [x] Tiling: `Gen2_SeamFix` + `Gen2_SeamMerger` for a two-pass seam-denoise workflow that smooths the boundaries between independently-sampled tile bases

## License

This repository's original code is licensed under Apache License 2.0. ComfyUI, VideoX-Fun, and the Alibaba PAI Flux.2/control-model weights retain their own licenses and usage terms; downloading or using model weights is not covered by this repository's Apache license.

