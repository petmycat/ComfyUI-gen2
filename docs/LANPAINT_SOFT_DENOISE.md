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

## Adaptive references and outer-step state

The patch maintains two clean references for one sampler run:

- `original_source_reference`: an immutable clone of LanPaint's source latent.
- `adaptive_reference`: a latest-generated or EMA reference, updated only after a completed outer sampler step.

The state is created inside each decorated `sampler_function` invocation and cannot leak into another sampling run. The adapter wraps the sampler callback and commits the staged candidate there. This is deliberately different from incrementing on every proxy call: multi-evaluation samplers can invoke the model several times during one outer step, while LanPaint can also run internal Langevin iterations. Only the callback boundary represents a completed outer step.

Before each LanPaint evaluation, the active clean reference is forward-noised through the current model's `model_sampling.noise_scaling`. Input and output blend strengths are independently scheduled with `constant`, `linear`, or `cosine` interpolation. `schedule_end_step = 0` means the last outer step.

The effective edit mask is:

```text
effective_edit = 1 - blend_strength * (1 - soft_edit)
```

Therefore strength `1` reproduces the full soft-mask effect and strength `0` bypasses that blend side. LanPaint is still called with the original hard `denoise_mask`; its in-place update of `x_work` is copied back to the outer sampler tensor.

Adaptive updates can use either raw LanPaint output or post-blend output and can be restricted to the feather band, all nonzero soft-mask pixels, or the full hard editable region. With `lock_original_outside_adaptive_region` enabled, every update re-anchors pixels outside that region to the immutable original source.

## Experimental mode recipes

Mode C, the default experimental configuration:

```text
reference_mode = ema_generated
reference_ema_momentum = 0.7
enable_input_blend = true
input strengths = 1 -> 1
enable_output_blend = true
output strengths = 1 -> 0
blend_schedule_type = linear
adaptive_region_mode = soft_band_only
adaptive_update_source = raw_generated_output
adaptive_reference_init = original_source
```

Mode D uses the same EMA/input settings with `enable_output_blend = false`.

Mode E uses:

```text
reference_mode = latest_generated
reference_warmup_steps = 2 to 4
adaptive_reference_init = first_generated_after_warmup
enable_output_blend = true
output strengths = 1 -> 0
```

To reproduce the original adapter behavior, select `original_source`, enable both blends, set all four strengths to `1`, and use `constant` scheduling.

Removing the Gen2 node removes the keyed wrapper from that model branch and restores normal LanPaint behavior.

## Maintenance

When LanPaint changes:

1. Verify its replacement `KSAMPLER.sample` still calls `sampler_function(model_k, ...)`.
2. Verify the runtime attributes listed above.
3. Verify the hard mask is still thresholded at `> 0.5` and remains editable-one polarity.
4. Verify its in-place `input_x.copy_(x)` behavior.
5. Verify source noising still uses `inner_model.inner_model.model_sampling.noise_scaling`.
6. Verify the sampler callback still reports one completion event per outer sigma step, in either dictionary or positional callback form.
7. Rerun the unit tests and the four-node GPU workflow matrix, including Mode C/D/E visual comparisons.

CPU tests:

```bash
python -m unittest tests.test_lanpaint_soft_denoise -v
```

A real ComfyUI/LanPaint workflow should additionally compare an unpatched baseline against `soft mask = binary mask`, with identical model, seed, sigmas, sampler and LanPaint settings.
