# Flux.2 Fun ControlNet integration

## Fixed sources

The machine-readable source manifest is `flux2_fun/source_manifest.json`.

- VideoX-Fun mathematical oracle: `248ab0ac0ebc48f0b4ae43ceb2d7ded24cc907bb`
- ComfyUI runtime: v0.28.0, `700821e1364eaab0e8f21c538a2131719fec57bf`
- Official checkpoint snapshot: `b3dcd7836a0e926248dac3ccba8fc0853495764b`
- Checkpoint SHA256: `516532a885d12ae84bb3c6b24ef4816ac05ffa1c9c7b93476f74652eb0a7a794`
- Negative reference only: `bryanmcguire/comfyui-flux2fun-controlnet@9285022d86abcf4af29fd2aabff828fb4d408bdf`

## Supported scope

Only Flux.2 Dev and `FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors` are supported. The loader requires the official 76-tensor branch profile: hidden size 6144, 48 heads of dimension 128, four control blocks, and an MLP width of 18432.

The control branch contains only `control_img_in`, the four control transformer blocks, block-zero `before_proj`, and four `after_proj` layers. It does not load or duplicate the base transformer.

## Preprocessing contract

- ComfyUI IMAGE inputs are resized to a shared multiple-of-16 canvas.
- White input mask means repaint.
- The inpaint image is zeroed in repaint regions before VAE encoding.
- ComfyUI's Flux.2 VAE must return `[B,128,H,W]`, already 2×2-patchified and BN-normalized.
- The preserved-region mask is nearest-resized and 2×2 patchified to four channels per token.
- Final order is `[control 128, preserved mask 4, inpaint 128]`.
- Missing image branches are direct zero latents; black images are not encoded as substitutes.
- Token tensors are never resized. Shape or token-count disagreement is a hard error.

VideoX-Fun's current no-reference path can access `image_latents.size()` when the value is absent. This integration treats that as an upstream bug and appends no reference padding when there are no reference tokens.

## Runtime behavior

`Apply Flux2 Fun Control` clones the input MODEL and patches only that clone. It installs composable replacements at base double blocks 0, 2, 4, and 6, preserving pre-existing replacements and injecting Fun hints afterward. Per-forward state is created by a keyed diffusion-model wrapper and is not stored as a mutable control list in persistent `transformer_options`.

The managed control model is attached as a namespaced additional model. Sampling code does not manually move the model between CPU/GPU and does not call `torch.cuda.empty_cache()`.

For references, zero 260-channel tokens are appended to the prepared context, the branch runs over the complete image sequence and positional encoding, and complete-length hints are injected. ComfyUI-provided modulation objects determine `index_timestep_zero` regions.

## Known prohibited reference patterns

The negative reference repository was useful for checkpoint shape discovery, but this implementation intentionally does not copy its global `Flux.forward_orig` replacement, mutable `transformer_options` lists, repeated device migration, synthetic `HooksContainer`, unchecked `strict=False` loading, fixed architecture guesses, or `empty_cache()` calls.

## Validation

Run CPU tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Optional real-weight parity uses `tests/flux2_fun_oracle_harness.py` and requires `GEN2_FLUX2_FUN_ORACLE=1`. It compares packed context, input projection, block outputs, after-projection hints, and a denoising forward, writing a machine-readable JSON report.

The required production acceptance cases are listed in `tests/flux2_fun_gpu_matrix.json`. They have not been executed on the development machine, which lacks the real production GPU/ComfyUI environment.
