from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F

import comfy.model_patcher
import comfy.patcher_extension


LOGGER = logging.getLogger(__name__)
WRAPPER_KEY = "gen2_lanpaint_soft_denoise"
ATTACHMENT_KEY = "gen2_lanpaint_soft_denoise_mask"
_COMPATIBILITY_SUFFIX = (
    " LanPaint may have changed its sampler interface. "
    "Update ComfyUI-gen2 or disable the Gen2 LanPaint Soft Denoise Patch."
)
_REQUIRED_RUNTIME_ATTRIBUTES = (
    "PaintMethod",
    "latent_image",
    "noise",
    "inner_model",
    "sigmas",
    "LanPaint_early_stop",
)


def _runtime_error(message: str) -> RuntimeError:
    return RuntimeError(message + _COMPATIBILITY_SUFFIX)


def _validate_image_latent(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise _runtime_error("Gen2 LanPaint Soft Denoise Patch expected the sampler state to be a tensor.")
    if x.ndim == 5:
        raise _runtime_error(
            "Gen2 LanPaint Soft Denoise Patch does not yet support 5D video latents; only [B,C,H,W] image latents are supported."
        )
    if x.ndim != 4:
        raise _runtime_error(
            f"Gen2 LanPaint Soft Denoise Patch expected a 4D [B,C,H,W] latent, but received shape {tuple(x.shape)}."
        )


def _normalize_mask_rank(mask: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise _runtime_error(f"Gen2 LanPaint Soft Denoise Patch expected {name} to be a tensor.")
    if mask.ndim == 2:
        return mask.unsqueeze(0).unsqueeze(0)
    if mask.ndim == 3:
        return mask.unsqueeze(1)
    if mask.ndim == 4:
        return mask
    raise _runtime_error(
        f"Gen2 LanPaint Soft Denoise Patch expected {name} with shape [H,W], [B,H,W], or [B,C,H,W], "
        f"but received {tuple(mask.shape)}."
    )


def _align_mask_batch(mask: torch.Tensor, target_batch: int, name: str) -> torch.Tensor:
    source_batch = int(mask.shape[0])
    if source_batch == target_batch:
        return mask
    if source_batch == 1:
        return mask.expand((target_batch,) + tuple(mask.shape[1:]))
    raise _runtime_error(
        f"Gen2 LanPaint Soft Denoise Patch cannot align {name} batch {source_batch} to latent batch {target_batch}; "
        "use a single mask or one mask per latent batch item."
    )


def _align_mask_channels(mask: torch.Tensor, target_channels: int, name: str) -> torch.Tensor:
    source_channels = int(mask.shape[1])
    if source_channels == target_channels:
        return mask
    if source_channels == 1:
        return mask.expand(mask.shape[0], target_channels, *mask.shape[2:])
    raise _runtime_error(
        f"Gen2 LanPaint Soft Denoise Patch cannot align {name} channels {source_channels} to latent channels {target_channels}."
    )


def prepare_soft_mask(source_mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    _validate_image_latent(x)
    mask = _normalize_mask_rank(source_mask, "soft mask")
    if not torch.isfinite(mask).all():
        raise _runtime_error("Gen2 LanPaint Soft Denoise Patch received NaN or infinite values in the soft mask.")
    mask = mask.to(device=x.device, dtype=x.dtype)
    mask = _align_mask_batch(mask, int(x.shape[0]), "soft mask")
    if tuple(mask.shape[-2:]) != tuple(x.shape[-2:]):
        mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)
    mask = _align_mask_channels(mask, int(x.shape[1]), "soft mask")
    mask = mask.clamp(0.0, 1.0)
    if not torch.isfinite(mask).all():
        raise _runtime_error("Gen2 LanPaint Soft Denoise Patch produced invalid values while preparing the soft mask.")
    return mask


def prepare_hard_envelope(denoise_mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    _validate_image_latent(x)
    mask = _normalize_mask_rank(denoise_mask, "LanPaint denoise mask")
    if not torch.isfinite(mask).all():
        raise _runtime_error("Gen2 LanPaint Soft Denoise Patch received NaN or infinite values in the LanPaint denoise mask.")
    mask = mask.to(device=x.device, dtype=x.dtype)
    mask = _align_mask_batch(mask, int(x.shape[0]), "LanPaint denoise mask")
    if tuple(mask.shape[-2:]) != tuple(x.shape[-2:]):
        mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest-exact")
    mask = _align_mask_channels(mask, int(x.shape[1]), "LanPaint denoise mask")
    return (mask > 0.5).to(dtype=x.dtype)


def validate_lanpaint_runtime(model_k) -> None:
    missing = [name for name in _REQUIRED_RUNTIME_ATTRIBUTES if not hasattr(model_k, name)]
    if missing:
        raise _runtime_error(
            "Gen2 LanPaint Soft Denoise Patch was used without a compatible LanPaint sampler runtime. "
            f"Missing attributes: {missing}."
        )
    if not callable(model_k):
        raise _runtime_error("Gen2 LanPaint Soft Denoise Patch expected the LanPaint runtime model to be callable.")


def _source_noised_latent(model_k, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    try:
        model_sampling = model_k.inner_model.inner_model.model_sampling
        sigma_shape = [sigma.shape[0]] + [1] * (x.ndim - 1)
        sigma_broadcast = sigma.reshape(sigma_shape)
        source = model_sampling.noise_scaling(sigma_broadcast, model_k.noise, model_k.latent_image)
    except Exception as error:
        raise _runtime_error(
            "Gen2 LanPaint Soft Denoise Patch could not access LanPaint's model-specific source-noising behavior."
        ) from error
    if not isinstance(source, torch.Tensor) or source.shape != x.shape:
        shape = None if not isinstance(source, torch.Tensor) else tuple(source.shape)
        raise _runtime_error(
            "Gen2 LanPaint Soft Denoise Patch received an incompatible forward-noised source latent "
            f"with shape {shape}; expected {tuple(x.shape)}."
        )
    return source.to(device=x.device, dtype=x.dtype)


class LanPaintSoftDenoiseProxy:
    def __init__(self, model_k, source_soft_mask: torch.Tensor) -> None:
        self.model_k = model_k
        self.source_soft_mask = source_soft_mask

    def __getattr__(self, name):
        return getattr(self.model_k, name)

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        denoise_mask=None,
        model_options=None,
        seed=None,
        **kwargs,
    ):
        _validate_image_latent(x)
        if denoise_mask is None:
            raise _runtime_error(
                "Gen2 LanPaint Soft Denoise Patch requires a noise mask from InpaintModelConditioning with noise_mask enabled."
            )

        lanpaint_model_options = comfy.model_patcher.create_model_options_clone(model_options or {})
        mask_function = lanpaint_model_options.pop("denoise_mask_function", None)

        hard_edit = prepare_hard_envelope(denoise_mask, x)
        soft_edit = prepare_soft_mask(self.source_soft_mask, x)
        if mask_function is not None:
            try:
                soft_edit = mask_function(
                    sigma,
                    soft_edit,
                    extra_options={"model": self.model_k.inner_model, "sigmas": self.model_k.sigmas},
                )
            except Exception as error:
                raise _runtime_error(
                    "Gen2 LanPaint Soft Denoise Patch failed while applying denoise_mask_function to the soft mask."
                ) from error
            soft_edit = prepare_soft_mask(soft_edit, x)

        soft_edit = soft_edit * hard_edit
        if not torch.isfinite(soft_edit).all():
            raise _runtime_error("Gen2 LanPaint Soft Denoise Patch produced NaN or infinite soft-mask values.")

        source_noised = _source_noised_latent(self.model_k, x, sigma)
        x_work = x * soft_edit + source_noised * (1.0 - soft_edit)

        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "LanPaint soft denoise: source_mask=%s prepared=%s latent=%s soft=[%.6f, %.6f] hard=[%.6f, %.6f] scheduled=%s",
                tuple(self.source_soft_mask.shape),
                tuple(soft_edit.shape),
                tuple(x.shape),
                float(soft_edit.min().detach().cpu()),
                float(soft_edit.max().detach().cpu()),
                float(hard_edit.min().detach().cpu()),
                float(hard_edit.max().detach().cpu()),
                mask_function is not None,
            )

        out = self.model_k(
            x_work,
            sigma,
            denoise_mask=denoise_mask,
            model_options=lanpaint_model_options,
            seed=seed,
            **kwargs,
        )
        if not isinstance(out, torch.Tensor) or out.shape != x.shape:
            shape = None if not isinstance(out, torch.Tensor) else tuple(out.shape)
            raise _runtime_error(
                f"Gen2 LanPaint Soft Denoise Patch received an incompatible LanPaint output shape {shape}; expected {tuple(x.shape)}."
            )

        x.copy_(x_work)
        return out * soft_edit + self.model_k.latent_image.to(device=x.device, dtype=x.dtype) * (1.0 - soft_edit)


def make_lanpaint_soft_sampler_function(
    original_sampler_function: Callable,
    source_soft_mask: torch.Tensor,
) -> Callable:
    def wrapped_sampler_function(model_k, noise, sigmas, *args, **kwargs):
        validate_lanpaint_runtime(model_k)
        proxy = LanPaintSoftDenoiseProxy(model_k, source_soft_mask)
        return original_sampler_function(proxy, noise, sigmas, *args, **kwargs)

    return wrapped_sampler_function


def make_sampler_sample_wrapper(source_soft_mask: torch.Tensor) -> Callable:
    def sampler_sample_wrapper(executor, *args, **kwargs):
        sampler_object = executor.class_obj
        if sampler_object is None or not hasattr(sampler_object, "sampler_function"):
            raise _runtime_error("Gen2 LanPaint Soft Denoise Patch could not access the active sampler function.")

        original_sampler_function = sampler_object.sampler_function
        sampler_object.sampler_function = make_lanpaint_soft_sampler_function(
            original_sampler_function,
            source_soft_mask,
        )
        try:
            return executor(*args, **kwargs)
        finally:
            sampler_object.sampler_function = original_sampler_function

    return sampler_sample_wrapper


def apply_lanpaint_soft_denoise_patch(model, soft_mask: torch.Tensor):
    if not isinstance(soft_mask, torch.Tensor):
        raise TypeError("Gen2 LanPaint Soft Denoise Patch requires a ComfyUI MASK tensor.")
    patched_model = model.clone()
    patched_model.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        WRAPPER_KEY,
    )
    patched_model.remove_attachments(ATTACHMENT_KEY)
    patched_model.set_attachments(ATTACHMENT_KEY, soft_mask)
    patched_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        WRAPPER_KEY,
        make_sampler_sample_wrapper(soft_mask),
    )
    return patched_model


class Gen2_LanPaintSoftDenoisePatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "soft_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Gen2/Sampling"
    DESCRIPTION = (
        "Clone and locally patch a MODEL so LanPaint keeps its binary noise mask while a separate soft mask "
        "controls continuous source preservation before and after each denoising evaluation."
    )

    def patch(self, model, soft_mask):
        return (apply_lanpaint_soft_denoise_patch(model, soft_mask),)
