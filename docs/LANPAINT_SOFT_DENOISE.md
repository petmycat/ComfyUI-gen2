# LanPaint soft-denoise adapter

## Scope

`Gen2 LanPaint Soft Denoise Patch` targets ComfyUI v0.28.0 and the current LanPaint runtime contract. It supports 4D image latents `[B,C,H,W]`. Five-dimensional video latents are rejected explicitly until their mask and source-noising behavior is validated end to end.

LanPaint remains an optional, independent custom-node pack. ComfyUI-gen2 does not import LanPaint modules, edit its files, or install a permanent monkeypatch.

## Workflow

Use the built-in `InpaintModelConditioning` node with `noise_mask` enabled for the binary mask path:

```text
binary MASK + source IMAGE + VAE + conditioning
    -> InpaintModelConditioning
    -> LATENT with noise_mask
    -> existing LanPaint sampler latent input
```

Patch only the model branch used by LanPaint:

```text
MODEL + soft MASK
    -> Gen2 LanPaint Soft Denoise Patch
    -> existing LanPaint sampler model input
```

The binary source mask should contain hard zero-or-one values. If it is already the same pixel size as the source image, `InpaintModelConditioning` preserves those values. If it resizes the mask, its bilinear interpolation may create intermediate values; LanPaint later prepares the latent mask and thresholds it at `> 0.5`.

## Runtime contract

The adapter is attached with the keyed ComfyUI wrapper:

```text
WrappersMP.SAMPLER_SAMPLE / gen2_lanpaint_soft_denoise
```

During the wrapper call, `executor.class_obj` is the active `SAMPLER`. The adapter temporarily replaces only that object's `sampler_function`, calls the next wrapper through `executor(...)`, and restores the original callable in `finally`.

The sampler function must receive a callable runtime object exposing:

```text
PaintMethod
latent_image
noise
inner_model
sigmas
LanPaint_early_stop
```

This is feature detection rather than an import, class-name, path, or version check. All four current LanPaint nodes use this route:

```text
LanPaint KSampler
LanPaint KSampler (Advanced)
LanPaint Sampler Custom
LanPaint Sampler Custom (Advanced)
```

If the contract changes, the adapter fails with a compatibility error rather than falling back to ordinary sampling.

## Mask ownership

The original `denoise_mask` is passed unchanged to LanPaint. LanPaint alone owns its thresholding, known/unknown partition, BiG score routing, Langevin coefficients, source replacement, early-stop mask semantics, and prompt mode.

The soft mask is prepared independently using bilinear interpolation, continuous `[0,1]` values, target device/dtype, strict batch handling, and channel broadcasting. A local hard envelope prepared with nearest-exact interpolation and `> 0.5` constrains the soft mask so interpolation cannot edit outside LanPaint's effective editable region.

If `model_options["denoise_mask_function"]` exists, the adapter clones model options, removes the function from the copy passed to LanPaint, and applies it to the soft mask instead. This keeps Differential Diffusion scheduling on the soft edit strength without changing LanPaint's binary partition.

## Per-evaluation behavior

Before each LanPaint evaluation, the adapter calculates LanPaint's forward-noised source state through the current model's `model_sampling.noise_scaling`, then blends:

```text
x_work = x * soft_edit + source_noised * (1 - soft_edit)
```

LanPaint is called with `x_work` and the original hard `denoise_mask`. Current LanPaint mutates the supplied state in place; after it returns, the adapter copies `x_work` back to the outer sampler tensor.

The denoised prediction is then blended with the clean source latent:

```text
out = out * soft_edit + latent_image * (1 - soft_edit)
```

Removing the Gen2 node removes the keyed wrapper from that model branch and restores normal LanPaint behavior.

## Maintenance

When LanPaint changes:

1. Verify its replacement `KSAMPLER.sample` still calls `sampler_function(model_k, ...)`.
2. Verify the runtime attributes listed above.
3. Verify the hard mask is still thresholded at `> 0.5` and remains editable-one polarity.
4. Verify its in-place `input_x.copy_(x)` behavior.
5. Verify source noising still uses `inner_model.inner_model.model_sampling.noise_scaling`.
6. Rerun the unit tests and the four-node GPU workflow matrix.

CPU tests:

```bash
python -m unittest tests.test_lanpaint_soft_denoise -v
```

A real ComfyUI/LanPaint workflow should additionally compare an unpatched baseline against `soft mask = binary mask`, with identical model, seed, sigmas, sampler and LanPaint settings.
