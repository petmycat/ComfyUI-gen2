from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

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
_REFERENCE_MODES = ("original_source", "latest_generated", "ema_generated")
_SCHEDULE_TYPES = ("constant", "linear", "cosine")
_ADAPTIVE_REGION_MODES = ("soft_band_only", "soft_nonzero", "hard_edit_region")
_ADAPTIVE_UPDATE_SOURCES = ("raw_generated_output", "blended_output")
_ADAPTIVE_REFERENCE_INITS = ("original_source", "first_generated_after_warmup")
_OUTPUT_REFERENCE_MODES = ("legacy", "latest_only", "mixed_original_latest", "original_only")
_REFERENCE_SELECTOR_CURVES = ("linear", "smoothstep", "smootherstep", "power")
_MIXED_OUTPUT_SCHEDULES = ("constant", "linear", "cosine", "smoothstep", "smootherstep")
_OUTPUT_MASK_CURVES = ("legacy", "linear", "smoothstep", "smootherstep", "power")
_MIXED_OUTPUT_SCHEDULE_DOMAINS = ("normalized_step", "normalized_sigma")


@dataclass(frozen=True)
class LanPaintSoftDenoiseConfig:
    reference_mode: str = "ema_generated"
    reference_warmup_steps: int = 0
    reference_ema_momentum: float = 0.7
    enable_input_blend: bool = True
    input_blend_strength_start: float = 1.0
    input_blend_strength_end: float = 1.0
    enable_output_blend: bool = True
    output_blend_strength_start: float = 1.0
    output_blend_strength_end: float = 0.0
    blend_schedule_type: str = "linear"
    schedule_start_step: int = 0
    schedule_end_step: int = 0
    adaptive_region_mode: str = "soft_band_only"
    adaptive_update_source: str = "raw_generated_output"
    lock_original_outside_adaptive_region: bool = True
    adaptive_reference_init: str = "original_source"
    debug_logging: bool = False
    output_reference_mode: str = "legacy"
    reference_selector_curve: str = "smoothstep"
    reference_selector_low: float = 0.0
    reference_selector_high: float = 1.0
    reference_selector_gamma: float = 1.0
    latest_reference_ratio_at_soft_min: float = 0.0
    latest_reference_ratio_at_soft_max: float = 1.0
    invert_reference_selector: bool = False
    mixed_output_blend_strength_start: float = 1.0
    mixed_output_blend_strength_end: float = 0.2
    mixed_output_blend_schedule: str = "cosine"
    mixed_output_blend_schedule_start: float = 0.15
    mixed_output_blend_schedule_end: float = 0.75
    mixed_output_schedule_domain: str = "normalized_sigma"
    output_blend_mask_curve: str = "legacy"
    output_blend_mask_low: float = 0.0
    output_blend_mask_high: float = 1.0
    output_blend_mask_gamma: float = 1.0

    def validate(self) -> None:
        choices = (
            ("reference_mode", self.reference_mode, _REFERENCE_MODES),
            ("blend_schedule_type", self.blend_schedule_type, _SCHEDULE_TYPES),
            ("adaptive_region_mode", self.adaptive_region_mode, _ADAPTIVE_REGION_MODES),
            ("adaptive_update_source", self.adaptive_update_source, _ADAPTIVE_UPDATE_SOURCES),
            ("adaptive_reference_init", self.adaptive_reference_init, _ADAPTIVE_REFERENCE_INITS),
            ("output_reference_mode", self.output_reference_mode, _OUTPUT_REFERENCE_MODES),
            ("reference_selector_curve", self.reference_selector_curve, _REFERENCE_SELECTOR_CURVES),
            ("mixed_output_blend_schedule", self.mixed_output_blend_schedule, _MIXED_OUTPUT_SCHEDULES),
            ("mixed_output_schedule_domain", self.mixed_output_schedule_domain, _MIXED_OUTPUT_SCHEDULE_DOMAINS),
            ("output_blend_mask_curve", self.output_blend_mask_curve, _OUTPUT_MASK_CURVES),
        )
        for name, value, allowed in choices:
            if value not in allowed:
                raise ValueError(f"Invalid {name} {value!r}; expected one of {allowed}.")
        if self.reference_warmup_steps < 0:
            raise ValueError("reference_warmup_steps must be non-negative.")
        if not 0.0 <= self.reference_ema_momentum <= 0.99:
            raise ValueError("reference_ema_momentum must be between 0.0 and 0.99.")
        strengths = (
            self.input_blend_strength_start,
            self.input_blend_strength_end,
            self.output_blend_strength_start,
            self.output_blend_strength_end,
        )
        if any(not 0.0 <= value <= 1.0 for value in strengths):
            raise ValueError("Input and output blend strengths must be between 0.0 and 1.0.")
        if self.schedule_start_step < 0 or self.schedule_end_step < 0:
            raise ValueError("Blend schedule steps must be non-negative.")
        if self.schedule_end_step != 0 and self.schedule_end_step < self.schedule_start_step:
            raise ValueError("schedule_end_step must be zero (automatic) or greater than or equal to schedule_start_step.")
        if not 0.0 <= self.reference_selector_low < self.reference_selector_high <= 1.0:
            raise ValueError("reference_selector_low/high must satisfy 0.0 <= low < high <= 1.0.")
        if not self.reference_selector_gamma > 0.0:
            raise ValueError("reference_selector_gamma must be greater than zero.")
        if not 0.0 <= self.latest_reference_ratio_at_soft_min <= 1.0:
            raise ValueError("latest_reference_ratio_at_soft_min must be between 0.0 and 1.0.")
        if not 0.0 <= self.latest_reference_ratio_at_soft_max <= 1.0:
            raise ValueError("latest_reference_ratio_at_soft_max must be between 0.0 and 1.0.")
        if not 0.0 <= self.mixed_output_blend_strength_start <= 1.0:
            raise ValueError("mixed_output_blend_strength_start must be between 0.0 and 1.0.")
        if not 0.0 <= self.mixed_output_blend_strength_end <= 1.0:
            raise ValueError("mixed_output_blend_strength_end must be between 0.0 and 1.0.")
        if not 0.0 <= self.mixed_output_blend_schedule_start < self.mixed_output_blend_schedule_end <= 1.0:
            raise ValueError("mixed output schedule start/end must satisfy 0.0 <= start < end <= 1.0.")
        if not 0.0 <= self.output_blend_mask_low < self.output_blend_mask_high <= 1.0:
            raise ValueError("output_blend_mask_low/high must satisfy 0.0 <= low < high <= 1.0.")
        if not self.output_blend_mask_gamma > 0.0:
            raise ValueError("output_blend_mask_gamma must be greater than zero.")


@dataclass
class AdaptiveReferenceState:
    config: LanPaintSoftDenoiseConfig
    total_steps: int
    current_step: int = 0
    original_source_reference: torch.Tensor | None = None
    adaptive_reference: torch.Tensor | None = None
    pending_candidate: torch.Tensor | None = None
    pending_region: torch.Tensor | None = None
    pending_sigma: float | None = None
    pending_input_strength: float = 0.0
    pending_output_strength: float = 0.0
    pending_soft_range: tuple[float, float] = (0.0, 0.0)
    pending_input_range: tuple[float, float] = (1.0, 1.0)
    pending_output_range: tuple[float, float] = (1.0, 1.0)
    pending_schedule_progress: float = 0.0
    pending_selector_summary: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pending_output_mask_summary: tuple[float, float, float] = (1.0, 1.0, 1.0)
    pending_preservation_summary: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def initialize(self, source: torch.Tensor) -> None:
        source = source.detach().clone()
        self.original_source_reference = source
        if self.config.adaptive_reference_init == "original_source":
            self.adaptive_reference = source.clone()

    def active_reference(self) -> torch.Tensor:
        if self.original_source_reference is None:
            raise _runtime_error("Gen2 LanPaint Soft Denoise Patch reference state was not initialized.")
        if self.config.reference_mode == "original_source":
            return self.original_source_reference
        if self.current_step < self.config.reference_warmup_steps or self.adaptive_reference is None:
            return self.original_source_reference
        return self.adaptive_reference

    def stage_update(
        self,
        candidate: torch.Tensor,
        region: torch.Tensor,
        sigma: torch.Tensor,
        input_strength: float,
        output_strength: float,
        soft_edit: torch.Tensor,
        effective_input: torch.Tensor,
        effective_output: torch.Tensor,
        schedule_progress: float = 0.0,
        reference_selector: torch.Tensor | None = None,
        output_spatial_mask: torch.Tensor | None = None,
        preservation_amount: torch.Tensor | None = None,
    ) -> None:
        self.pending_candidate = candidate.detach().clone()
        self.pending_region = region.detach()
        self.pending_sigma = float(sigma.detach().float().mean().cpu())
        self.pending_input_strength = input_strength
        self.pending_output_strength = output_strength
        self.pending_soft_range = _tensor_range(soft_edit)
        self.pending_input_range = _tensor_range(effective_input)
        self.pending_output_range = _tensor_range(effective_output)
        self.pending_schedule_progress = float(schedule_progress)
        if reference_selector is not None:
            self.pending_selector_summary = _tensor_summary(reference_selector)
        if output_spatial_mask is not None:
            self.pending_output_mask_summary = _tensor_summary(output_spatial_mask)
        if preservation_amount is not None:
            self.pending_preservation_summary = _tensor_summary(preservation_amount)

    def commit_outer_step(self, completed_step: int | None = None) -> bool:
        step = self.current_step if completed_step is None else int(completed_step)
        updated = False
        if (
            self.config.reference_mode != "original_source"
            and step >= self.config.reference_warmup_steps
            and self.pending_candidate is not None
            and self.pending_region is not None
        ):
            updated = self._update_reference(self.pending_candidate, self.pending_region)

        if self.config.debug_logging or LOGGER.isEnabledFor(logging.DEBUG):
            reference = self.active_reference()
            log = LOGGER.info if self.config.debug_logging else LOGGER.debug
            log(
                "[Gen2 LanPaint debug] outer_step=%d total_steps=%d sigma=%s mode=%s warmup=%s "
                "input_strength=%.6f output_strength=%.6f updated=%s update_source=%s region=%s "
                "soft=[%.6f,%.6f] effective_input=[%.6f,%.6f] effective_output=[%.6f,%.6f] reference=[%.6f,%.6f] "
                "output_reference_mode=%s progress=%.6f selector=[%.6f,%.6f,%.6f] output_mask=[%.6f,%.6f,%.6f] preservation=[%.6f,%.6f,%.6f]",
                step,
                self.total_steps,
                "unknown" if self.pending_sigma is None else f"{self.pending_sigma:.6f}",
                self.config.reference_mode,
                step < self.config.reference_warmup_steps,
                self.pending_input_strength,
                self.pending_output_strength,
                updated,
                self.config.adaptive_update_source,
                self.config.adaptive_region_mode,
                *self.pending_soft_range,
                *self.pending_input_range,
                *self.pending_output_range,
                *_tensor_range(reference),
                self.config.output_reference_mode,
                self.pending_schedule_progress,
                *self.pending_selector_summary,
                *self.pending_output_mask_summary,
                *self.pending_preservation_summary,
            )

        self.current_step = max(self.current_step + 1, step + 1)
        self.pending_candidate = None
        self.pending_region = None
        self.pending_sigma = None
        return updated

    def _update_reference(self, candidate: torch.Tensor, region: torch.Tensor) -> bool:
        if self.original_source_reference is None:
            raise _runtime_error("Gen2 LanPaint Soft Denoise Patch cannot update an uninitialized reference.")
        candidate = candidate.to(
            device=self.original_source_reference.device,
            dtype=self.original_source_reference.dtype,
        )
        region = region.to(device=candidate.device, dtype=candidate.dtype)
        if not torch.any(region > 0):
            return False

        if self.adaptive_reference is None:
            previous = self.original_source_reference
            updated_region = candidate
        else:
            previous = self.adaptive_reference
            if self.config.reference_mode == "ema_generated":
                momentum = self.config.reference_ema_momentum
                updated_region = previous * momentum + candidate * (1.0 - momentum)
            else:
                updated_region = candidate

        outside = self.original_source_reference if self.config.lock_original_outside_adaptive_region else previous
        self.adaptive_reference = (updated_region * region + outside * (1.0 - region)).detach()
        return True


def _runtime_error(message: str) -> RuntimeError:
    return RuntimeError(message + _COMPATIBILITY_SUFFIX)


def _tensor_range(tensor: torch.Tensor) -> tuple[float, float]:
    return (
        float(tensor.detach().float().min().cpu()),
        float(tensor.detach().float().max().cpu()),
    )


def _tensor_summary(tensor: torch.Tensor) -> tuple[float, float, float]:
    detached = tensor.detach().float()
    return (
        float(detached.min().cpu()),
        float(detached.max().cpu()),
        float(detached.mean().cpu()),
    )


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


def schedule_strength(
    start_value: float,
    end_value: float,
    step: int,
    total_steps: int,
    schedule_type: str,
    start_step: int,
    end_step: int,
) -> float:
    if schedule_type == "constant":
        return float(start_value)
    resolved_end = total_steps - 1 if end_step == 0 else end_step
    resolved_end = max(resolved_end, start_step)
    if step <= start_step:
        progress = 0.0
    elif step >= resolved_end:
        progress = 1.0
    else:
        progress = (step - start_step) / max(resolved_end - start_step, 1)
    if schedule_type == "cosine":
        progress = 0.5 - 0.5 * math.cos(math.pi * progress)
    return float(start_value + (end_value - start_value) * progress)


def effective_edit_mask(soft_edit: torch.Tensor, strength: float, enabled: bool) -> torch.Tensor:
    if not enabled or strength <= 0.0:
        return torch.ones_like(soft_edit)
    return 1.0 - float(strength) * (1.0 - soft_edit)


def _remap_unit_interval(
    value: torch.Tensor,
    curve: str,
    low: float,
    high: float,
    gamma: float,
) -> torch.Tensor:
    if curve == "legacy":
        return value.clamp(0.0, 1.0)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("Mask remap low/high must satisfy 0.0 <= low < high <= 1.0.")
    normalized = ((value.clamp(0.0, 1.0) - low) / (high - low)).clamp(0.0, 1.0)
    if curve == "linear":
        return normalized
    if curve == "smoothstep":
        return normalized * normalized * (3.0 - 2.0 * normalized)
    if curve == "smootherstep":
        return normalized * normalized * normalized * (normalized * (normalized * 6.0 - 15.0) + 10.0)
    if curve == "power":
        return normalized.pow(float(gamma))
    raise ValueError(f"Unknown mask remap curve: {curve}")


def build_reference_selector(
    soft_mask: torch.Tensor,
    curve: str,
    low: float,
    high: float,
    gamma: float,
    latest_min: float,
    latest_max: float,
    invert: bool = False,
) -> torch.Tensor:
    selector = _remap_unit_interval(soft_mask, curve, low, high, gamma)
    if invert:
        selector = 1.0 - selector
    return (latest_min + (latest_max - latest_min) * selector).clamp(0.0, 1.0)


def build_mixed_output_reference(
    original_reference: torch.Tensor,
    latest_reference: torch.Tensor,
    reference_selector: torch.Tensor,
) -> torch.Tensor:
    if original_reference.shape != latest_reference.shape or original_reference.shape != reference_selector.shape:
        raise ValueError("Mixed output reference tensors must have identical shapes.")
    return torch.lerp(original_reference, latest_reference, reference_selector)


def schedule_progress_from_sigma(sigma: torch.Tensor, sigmas: torch.Tensor, step: int, total_steps: int, domain: str) -> float:
    if domain == "normalized_step":
        return min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    if domain != "normalized_sigma":
        raise ValueError(f"Unknown mixed output schedule domain: {domain}")
    sigma_value = float(sigma.detach().float().mean().cpu())
    sigma_values = sigmas.detach().float().reshape(-1)
    first = float(sigma_values[0].cpu())
    last = float(sigma_values[-1].cpu())
    denominator = first - last
    if abs(denominator) < 1e-12:
        return min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    progress = (first - sigma_value) / denominator
    return min(max(progress, 0.0), 1.0)


def schedule_fraction(progress: float, start: float, end: float, curve: str) -> float:
    if progress <= start:
        fraction = 0.0
    elif progress >= end:
        fraction = 1.0
    else:
        fraction = (progress - start) / max(end - start, 1e-12)
    if curve == "constant":
        fraction = 0.0
    elif curve == "cosine":
        fraction = 0.5 - 0.5 * math.cos(math.pi * fraction)
    elif curve == "smoothstep":
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    elif curve == "smootherstep":
        fraction = fraction * fraction * fraction * (fraction * (fraction * 6.0 - 15.0) + 10.0)
    elif curve != "linear":
        raise ValueError(f"Unknown schedule curve: {curve}")
    return float(fraction)


def scheduled_mixed_output_strength(config: LanPaintSoftDenoiseConfig, progress: float) -> float:
    fraction = schedule_fraction(
        progress,
        config.mixed_output_blend_schedule_start,
        config.mixed_output_blend_schedule_end,
        config.mixed_output_blend_schedule,
    )
    return float(
        config.mixed_output_blend_strength_start
        + (config.mixed_output_blend_strength_end - config.mixed_output_blend_strength_start) * fraction
    )


def adaptive_update_region(
    mode: str,
    base_soft_edit: torch.Tensor,
    hard_edit: torch.Tensor,
) -> torch.Tensor:
    if mode == "soft_band_only":
        region = (base_soft_edit > 0.0) & (base_soft_edit < 1.0)
    elif mode == "soft_nonzero":
        region = base_soft_edit > 0.0
    elif mode == "hard_edit_region":
        region = hard_edit > 0.5
    else:
        raise ValueError(f"Unknown adaptive region mode: {mode}")
    return (region & (hard_edit > 0.5)).to(dtype=base_soft_edit.dtype)


def validate_lanpaint_runtime(model_k) -> None:
    missing = [name for name in _REQUIRED_RUNTIME_ATTRIBUTES if not hasattr(model_k, name)]
    if missing:
        raise _runtime_error(
            "Gen2 LanPaint Soft Denoise Patch was used without a compatible LanPaint sampler runtime. "
            f"Missing attributes: {missing}."
        )
    if not callable(model_k):
        raise _runtime_error("Gen2 LanPaint Soft Denoise Patch expected the LanPaint runtime model to be callable.")


def _source_noised_latent(
    model_k,
    x: torch.Tensor,
    sigma: torch.Tensor,
    clean_reference: torch.Tensor,
) -> torch.Tensor:
    try:
        model_sampling = model_k.inner_model.inner_model.model_sampling
        sigma_shape = [sigma.shape[0]] + [1] * (x.ndim - 1)
        sigma_broadcast = sigma.reshape(sigma_shape)
        source = model_sampling.noise_scaling(sigma_broadcast, model_k.noise, clean_reference)
    except Exception as error:
        raise _runtime_error(
            "Gen2 LanPaint Soft Denoise Patch could not access LanPaint's model-specific source-noising behavior."
        ) from error
    if not isinstance(source, torch.Tensor) or source.shape != x.shape:
        shape = None if not isinstance(source, torch.Tensor) else tuple(source.shape)
        raise _runtime_error(
            "Gen2 LanPaint Soft Denoise Patch received an incompatible forward-noised reference latent "
            f"with shape {shape}; expected {tuple(x.shape)}."
        )
    return source.to(device=x.device, dtype=x.dtype)


class LanPaintSoftDenoiseProxy:
    def __init__(
        self,
        model_k,
        source_soft_mask: torch.Tensor,
        config: LanPaintSoftDenoiseConfig | None = None,
        state: AdaptiveReferenceState | None = None,
    ) -> None:
        self.model_k = model_k
        self.source_soft_mask = source_soft_mask
        self.config = config or LanPaintSoftDenoiseConfig(
            reference_mode="original_source",
            output_blend_strength_end=1.0,
            blend_schedule_type="constant",
        )
        self.config.validate()
        self.state = state or AdaptiveReferenceState(self.config, max(len(model_k.sigmas) - 1, 1))

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

        original_source = self.model_k.latent_image.to(device=x.device, dtype=x.dtype)
        if original_source.shape != x.shape:
            raise _runtime_error(
                f"Gen2 LanPaint Soft Denoise Patch source latent shape {tuple(original_source.shape)} "
                f"does not match sampler state {tuple(x.shape)}."
            )
        if self.state.original_source_reference is None:
            self.state.initialize(original_source)

        lanpaint_model_options = comfy.model_patcher.create_model_options_clone(model_options or {})
        mask_function = lanpaint_model_options.pop("denoise_mask_function", None)

        hard_edit = prepare_hard_envelope(denoise_mask, x)
        base_soft_edit = prepare_soft_mask(self.source_soft_mask, x) * hard_edit
        soft_edit = base_soft_edit
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
            soft_edit = prepare_soft_mask(soft_edit, x) * hard_edit

        step = self.state.current_step
        input_strength = schedule_strength(
            self.config.input_blend_strength_start,
            self.config.input_blend_strength_end,
            step,
            self.state.total_steps,
            self.config.blend_schedule_type,
            self.config.schedule_start_step,
            self.config.schedule_end_step,
        )
        output_strength = schedule_strength(
            self.config.output_blend_strength_start,
            self.config.output_blend_strength_end,
            step,
            self.state.total_steps,
            self.config.blend_schedule_type,
            self.config.schedule_start_step,
            self.config.schedule_end_step,
        )
        effective_input = effective_edit_mask(soft_edit, input_strength, self.config.enable_input_blend)
        effective_output = effective_edit_mask(soft_edit, output_strength, self.config.enable_output_blend)

        active_reference = self.state.active_reference().to(device=x.device, dtype=x.dtype)
        reference_noised = _source_noised_latent(self.model_k, x, sigma, active_reference)
        x_work = x * effective_input + reference_noised * (1.0 - effective_input)

        raw_generated = self.model_k(
            x_work,
            sigma,
            denoise_mask=denoise_mask,
            model_options=lanpaint_model_options,
            seed=seed,
            **kwargs,
        )
        if not isinstance(raw_generated, torch.Tensor) or raw_generated.shape != x.shape:
            shape = None if not isinstance(raw_generated, torch.Tensor) else tuple(raw_generated.shape)
            raise _runtime_error(
                f"Gen2 LanPaint Soft Denoise Patch received an incompatible LanPaint output shape {shape}; expected {tuple(x.shape)}."
            )

        x.copy_(x_work)
        original_reference = self.state.original_source_reference.to(device=x.device, dtype=x.dtype)
        output_reference = active_reference
        output_strength_for_step = output_strength
        output_spatial_mask = soft_edit
        reference_selector = torch.zeros_like(soft_edit)
        preservation_amount = 1.0 - effective_output
        schedule_progress = schedule_progress_from_sigma(
            sigma,
            self.model_k.sigmas,
            step,
            self.state.total_steps,
            self.config.mixed_output_schedule_domain,
        )
        if self.config.output_reference_mode == "original_only":
            output_reference = original_reference
        elif self.config.output_reference_mode == "mixed_original_latest":
            reference_selector = build_reference_selector(
                soft_edit,
                self.config.reference_selector_curve,
                self.config.reference_selector_low,
                self.config.reference_selector_high,
                self.config.reference_selector_gamma,
                self.config.latest_reference_ratio_at_soft_min,
                self.config.latest_reference_ratio_at_soft_max,
                self.config.invert_reference_selector,
            )
            output_reference = build_mixed_output_reference(
                original_reference,
                active_reference,
                reference_selector,
            )
            output_strength_for_step = scheduled_mixed_output_strength(self.config, schedule_progress)
            output_spatial_mask = _remap_unit_interval(
                soft_edit,
                self.config.output_blend_mask_curve,
                self.config.output_blend_mask_low,
                self.config.output_blend_mask_high,
                self.config.output_blend_mask_gamma,
            )
            preservation_amount = (
                output_strength_for_step * (1.0 - output_spatial_mask)
                if self.config.enable_output_blend
                else torch.zeros_like(output_spatial_mask)
            ).clamp(0.0, 1.0)
            effective_output = 1.0 - preservation_amount
        elif self.config.output_reference_mode == "latest_only":
            output_reference = active_reference
        elif self.config.output_reference_mode == "legacy":
            output_reference = active_reference

        if self.config.enable_output_blend and output_strength_for_step > 0.0:
            blended_output = raw_generated * effective_output + output_reference * (1.0 - effective_output)
        else:
            blended_output = raw_generated

        candidate = (
            raw_generated
            if self.config.adaptive_update_source == "raw_generated_output"
            else blended_output
        )
        region = adaptive_update_region(self.config.adaptive_region_mode, base_soft_edit, hard_edit)
        self.state.stage_update(
            candidate,
            region,
            sigma,
            input_strength,
            output_strength_for_step if self.config.enable_output_blend else 0.0,
            soft_edit,
            effective_input,
            effective_output,
            schedule_progress,
            reference_selector,
            output_spatial_mask,
            preservation_amount,
        )
        return blended_output


def make_lanpaint_soft_sampler_function(
    original_sampler_function: Callable,
    source_soft_mask: torch.Tensor,
    config: LanPaintSoftDenoiseConfig | None = None,
) -> Callable:
    resolved_config = config or LanPaintSoftDenoiseConfig()
    resolved_config.validate()

    def wrapped_sampler_function(model_k, noise, sigmas, *args, **kwargs):
        validate_lanpaint_runtime(model_k)
        state = AdaptiveReferenceState(resolved_config, max(len(sigmas) - 1, 1))
        proxy = LanPaintSoftDenoiseProxy(model_k, source_soft_mask, resolved_config, state)
        original_callback = kwargs.get("callback")
        last_committed_step = None

        def outer_step_callback(*callback_args, **callback_kwargs):
            nonlocal last_committed_step
            completed_step = None
            if callback_args and isinstance(callback_args[0], dict):
                completed_step = callback_args[0].get("i")
            elif callback_args and isinstance(callback_args[0], int):
                completed_step = callback_args[0]
            elif "i" in callback_kwargs:
                completed_step = callback_kwargs["i"]

            if completed_step is not None and completed_step != last_committed_step:
                state.commit_outer_step(completed_step)
                last_committed_step = completed_step

            if original_callback is not None:
                return original_callback(*callback_args, **callback_kwargs)
            return None

        kwargs["callback"] = outer_step_callback
        return original_sampler_function(proxy, noise, sigmas, *args, **kwargs)

    return wrapped_sampler_function


def make_sampler_sample_wrapper(
    source_soft_mask: torch.Tensor,
    config: LanPaintSoftDenoiseConfig | None = None,
) -> Callable:
    def sampler_sample_wrapper(executor, *args, **kwargs):
        sampler_object = executor.class_obj
        if sampler_object is None or not hasattr(sampler_object, "sampler_function"):
            raise _runtime_error("Gen2 LanPaint Soft Denoise Patch could not access the active sampler function.")

        original_sampler_function = sampler_object.sampler_function
        sampler_object.sampler_function = make_lanpaint_soft_sampler_function(
            original_sampler_function,
            source_soft_mask,
            config,
        )
        try:
            return executor(*args, **kwargs)
        finally:
            sampler_object.sampler_function = original_sampler_function

    return sampler_sample_wrapper


def apply_lanpaint_soft_denoise_patch(
    model,
    soft_mask: torch.Tensor,
    config: LanPaintSoftDenoiseConfig | None = None,
):
    if not isinstance(soft_mask, torch.Tensor):
        raise TypeError("Gen2 LanPaint Soft Denoise Patch requires a ComfyUI MASK tensor.")
    resolved_config = config or LanPaintSoftDenoiseConfig()
    resolved_config.validate()
    patched_model = model.clone()
    patched_model.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        WRAPPER_KEY,
    )
    patched_model.remove_attachments(ATTACHMENT_KEY)
    patched_model.set_attachments(ATTACHMENT_KEY, {"soft_mask": soft_mask, "config": resolved_config})
    patched_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        WRAPPER_KEY,
        make_sampler_sample_wrapper(soft_mask, resolved_config),
    )
    return patched_model


class Gen2_LanPaintSoftDenoisePatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "soft_mask": ("MASK",),
            },
            "optional": {
                "reference_mode": (_REFERENCE_MODES, {"default": "ema_generated"}),
                "reference_warmup_steps": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "reference_ema_momentum": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 0.99, "step": 0.01}),
                "enable_input_blend": ("BOOLEAN", {"default": True}),
                "input_blend_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "input_blend_strength_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enable_output_blend": ("BOOLEAN", {"default": True}),
                "output_blend_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output_blend_strength_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_schedule_type": (_SCHEDULE_TYPES, {"default": "linear"}),
                "schedule_start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "schedule_end_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "adaptive_region_mode": (_ADAPTIVE_REGION_MODES, {"default": "soft_band_only"}),
                "adaptive_update_source": (_ADAPTIVE_UPDATE_SOURCES, {"default": "raw_generated_output"}),
                "lock_original_outside_adaptive_region": ("BOOLEAN", {"default": True}),
                "adaptive_reference_init": (_ADAPTIVE_REFERENCE_INITS, {"default": "original_source"}),
                "debug_logging": ("BOOLEAN", {"default": False}),
                "output_reference_mode": (_OUTPUT_REFERENCE_MODES, {"default": "legacy", "tooltip": "legacy preserves the existing output path; mixed_original_latest enables the new opt-in spatial mixture."}),
                "reference_selector_curve": (_REFERENCE_SELECTOR_CURVES, {"default": "smoothstep", "tooltip": "Shapes how the soft mask chooses original versus latest reference."}),
                "reference_selector_low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01}),
                "reference_selector_high": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "reference_selector_gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.05}),
                "latest_reference_ratio_at_soft_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "latest_reference_ratio_at_soft_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_reference_selector": ("BOOLEAN", {"default": False}),
                "mixed_output_blend_strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mixed_output_blend_strength_end": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mixed_output_blend_schedule": (_MIXED_OUTPUT_SCHEDULES, {"default": "cosine"}),
                "mixed_output_blend_schedule_start": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 0.99, "step": 0.01}),
                "mixed_output_blend_schedule_end": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 1.0, "step": 0.01}),
                "mixed_output_schedule_domain": (_MIXED_OUTPUT_SCHEDULE_DOMAINS, {"default": "normalized_sigma", "tooltip": "Uses sigma trajectory progress for the new mixed-output schedule; normalized_step is available for direct step timing."}),
                "output_blend_mask_curve": (_OUTPUT_MASK_CURVES, {"default": "legacy", "tooltip": "Spatially reshapes output preservation without changing input blending."}),
                "output_blend_mask_low": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01}),
                "output_blend_mask_high": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "output_blend_mask_gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Gen2/Sampling"
    DESCRIPTION = (
        "Experimental clone-local LanPaint dual-mask patch with original, latest-generated, and EMA adaptive "
        "references plus independently scheduled input/output blending. schedule_end_step=0 uses the last outer step."
    )

    def patch(
        self,
        model,
        soft_mask,
        reference_mode="ema_generated",
        reference_warmup_steps=0,
        reference_ema_momentum=0.7,
        enable_input_blend=True,
        input_blend_strength_start=1.0,
        input_blend_strength_end=1.0,
        enable_output_blend=True,
        output_blend_strength_start=1.0,
        output_blend_strength_end=0.0,
        blend_schedule_type="linear",
        schedule_start_step=0,
        schedule_end_step=0,
        adaptive_region_mode="soft_band_only",
        adaptive_update_source="raw_generated_output",
        lock_original_outside_adaptive_region=True,
        adaptive_reference_init="original_source",
        debug_logging=False,
        output_reference_mode="legacy",
        reference_selector_curve="smoothstep",
        reference_selector_low=0.0,
        reference_selector_high=1.0,
        reference_selector_gamma=1.0,
        latest_reference_ratio_at_soft_min=0.0,
        latest_reference_ratio_at_soft_max=1.0,
        invert_reference_selector=False,
        mixed_output_blend_strength_start=1.0,
        mixed_output_blend_strength_end=0.2,
        mixed_output_blend_schedule="cosine",
        mixed_output_blend_schedule_start=0.15,
        mixed_output_blend_schedule_end=0.75,
        mixed_output_schedule_domain="normalized_sigma",
        output_blend_mask_curve="legacy",
        output_blend_mask_low=0.0,
        output_blend_mask_high=1.0,
        output_blend_mask_gamma=1.0,
    ):
        config = LanPaintSoftDenoiseConfig(
            reference_mode=str(reference_mode),
            reference_warmup_steps=int(reference_warmup_steps),
            reference_ema_momentum=float(reference_ema_momentum),
            enable_input_blend=bool(enable_input_blend),
            input_blend_strength_start=float(input_blend_strength_start),
            input_blend_strength_end=float(input_blend_strength_end),
            enable_output_blend=bool(enable_output_blend),
            output_blend_strength_start=float(output_blend_strength_start),
            output_blend_strength_end=float(output_blend_strength_end),
            blend_schedule_type=str(blend_schedule_type),
            schedule_start_step=int(schedule_start_step),
            schedule_end_step=int(schedule_end_step),
            adaptive_region_mode=str(adaptive_region_mode),
            adaptive_update_source=str(adaptive_update_source),
            lock_original_outside_adaptive_region=bool(lock_original_outside_adaptive_region),
            adaptive_reference_init=str(adaptive_reference_init),
            debug_logging=bool(debug_logging),
            output_reference_mode=str(output_reference_mode),
            reference_selector_curve=str(reference_selector_curve),
            reference_selector_low=float(reference_selector_low),
            reference_selector_high=float(reference_selector_high),
            reference_selector_gamma=float(reference_selector_gamma),
            latest_reference_ratio_at_soft_min=float(latest_reference_ratio_at_soft_min),
            latest_reference_ratio_at_soft_max=float(latest_reference_ratio_at_soft_max),
            invert_reference_selector=bool(invert_reference_selector),
            mixed_output_blend_strength_start=float(mixed_output_blend_strength_start),
            mixed_output_blend_strength_end=float(mixed_output_blend_strength_end),
            mixed_output_blend_schedule=str(mixed_output_blend_schedule),
            mixed_output_blend_schedule_start=float(mixed_output_blend_schedule_start),
            mixed_output_blend_schedule_end=float(mixed_output_blend_schedule_end),
            mixed_output_schedule_domain=str(mixed_output_schedule_domain),
            output_blend_mask_curve=str(output_blend_mask_curve),
            output_blend_mask_low=float(output_blend_mask_low),
            output_blend_mask_high=float(output_blend_mask_high),
            output_blend_mask_gamma=float(output_blend_mask_gamma),
        )
        return (apply_lanpaint_soft_denoise_patch(model, soft_mask, config),)
