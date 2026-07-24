from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from .types import PreparedFlux2FunContext


FLUX2_LATENT_CHANNELS = 128
MASK_TOKEN_CHANNELS = 4
CONTROL_CONTEXT_CHANNELS = 260


def _to_bhwc_image(image: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(f"{name} must be a rank-4 ComfyUI IMAGE tensor [B,H,W,C].")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3, or 4 channels in its last dimension, got {tuple(image.shape)}.")
    image = image[..., :3].float().clamp(0.0, 1.0)
    if image.shape[-1] == 1:
        image = image.repeat(1, 1, 1, 3)
    return image


def _resize_bhwc(image: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    if image.shape[1:3] == (height, width):
        return image
    bchw = image.movedim(-1, 1)
    kwargs = {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}
    return F.interpolate(bchw, size=(height, width), mode=mode, **kwargs).movedim(1, -1)


def _normalize_mask(mask: torch.Tensor, batch: int, height: int, width: int) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError("mask must be a torch.Tensor.")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[1] == 1:
            mask = mask[:, 0]
        else:
            raise ValueError(f"mask rank-4 input must have one channel, got {tuple(mask.shape)}.")
    if mask.ndim != 3:
        raise ValueError(f"mask must have shape [B,H,W], got {tuple(mask.shape)}.")
    mask = mask.float().clamp(0.0, 1.0).unsqueeze(1)
    mask = F.interpolate(mask, size=(height, width), mode="nearest")[:, 0]
    mask = (mask >= 0.5).to(mask.dtype)
    return align_batch(mask, batch, "mask")


def align_batch(tensor: torch.Tensor, target_batch: int, name: str) -> torch.Tensor:
    source_batch = int(tensor.shape[0])
    if source_batch == target_batch:
        return tensor
    if source_batch == 1:
        return tensor.repeat((target_batch,) + (1,) * (tensor.ndim - 1))
    if target_batch % source_batch == 0:
        repeats = target_batch // source_batch
        return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))
    raise ValueError(
        f"Cannot align {name} batch {source_batch} to target batch {target_batch}; "
        "only equal, singleton, or exact integer-multiple repetition is allowed."
    )


def pack_mask_tokens(preserved_mask: torch.Tensor, latent_height: int, latent_width: int) -> torch.Tensor:
    if preserved_mask.ndim == 3:
        preserved_mask = preserved_mask.unsqueeze(1)
    if preserved_mask.ndim != 4 or preserved_mask.shape[1] != 1:
        raise ValueError("preserved_mask must have shape [B,1,H,W] or [B,H,W].")
    mask_latent = F.interpolate(preserved_mask.float(), size=(latent_height * 2, latent_width * 2), mode="nearest")
    b, _, h, w = mask_latent.shape
    if h % 2 or w % 2:
        raise ValueError(f"Mask latent spatial dimensions must be even before patchify, got {(h, w)}.")
    return mask_latent.reshape(b, 1, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, 4)


def validate_flux2_vae_contract(vae: object, encoded: torch.Tensor) -> None:
    latent_channels = getattr(vae, "latent_channels", None)
    downscale = getattr(vae, "downscale_ratio", None)
    if latent_channels != FLUX2_LATENT_CHANNELS or downscale != 16:
        raise ValueError(
            "Gen2 Flux2 Fun requires ComfyUI's Flux.2 VAE contract "
            f"(latent_channels=128, downscale_ratio=16); got latent_channels={latent_channels}, downscale_ratio={downscale}."
        )
    if encoded.ndim != 4 or encoded.shape[1] != FLUX2_LATENT_CHANNELS:
        raise ValueError(
            "Flux.2 VAE encode() must return already patchified and BN-normalized [B,128,H,W] latents; "
            f"got {tuple(encoded.shape)}. Do not apply an additional normalization or patchify step."
        )


def pack_control_context(control_latents: torch.Tensor, preserved_mask: torch.Tensor, inpaint_latents: torch.Tensor) -> torch.Tensor:
    if control_latents.ndim != 4 or inpaint_latents.ndim != 4:
        raise ValueError("control_latents and inpaint_latents must be [B,128,H,W].")
    if control_latents.shape != inpaint_latents.shape:
        raise ValueError(
            f"Control and inpaint latent shapes must match exactly; got {tuple(control_latents.shape)} and {tuple(inpaint_latents.shape)}."
        )
    if control_latents.shape[1] != FLUX2_LATENT_CHANNELS:
        raise ValueError(f"Flux.2 latent channel count must be 128, got {control_latents.shape[1]}.")
    b, _, latent_height, latent_width = control_latents.shape
    control_tokens = control_latents.flatten(2).transpose(1, 2)
    inpaint_tokens = inpaint_latents.flatten(2).transpose(1, 2)
    mask_tokens = pack_mask_tokens(preserved_mask, latent_height, latent_width)
    if not (control_tokens.shape[1] == mask_tokens.shape[1] == inpaint_tokens.shape[1]):
        raise ValueError(
            "Flux2 Fun token count mismatch; token tensors are never resized heuristically: "
            f"control={control_tokens.shape[1]}, mask={mask_tokens.shape[1]}, inpaint={inpaint_tokens.shape[1]}."
        )
    packed = torch.cat((control_tokens, mask_tokens.to(control_tokens), inpaint_tokens.to(control_tokens)), dim=-1)
    if packed.shape != (b, latent_height * latent_width, CONTROL_CONTEXT_CHANNELS):
        raise RuntimeError(f"Unexpected packed Flux2 Fun context shape: {tuple(packed.shape)}.")
    return packed


def prepare_control_context(
    vae: object,
    control_image: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    inpaint_image: torch.Tensor | None = None,
    target_latent: dict | None = None,
) -> PreparedFlux2FunContext:
    if control_image is None and inpaint_image is None:
        raise ValueError("Prepare Flux2 Fun Control requires control_image or inpaint_image.")

    images: list[tuple[str, torch.Tensor]] = []
    if control_image is not None:
        images.append(("control_image", _to_bhwc_image(control_image, "control_image")))
    if inpaint_image is not None:
        images.append(("inpaint_image", _to_bhwc_image(inpaint_image, "inpaint_image")))

    batch = max(int(image.shape[0]) for _, image in images)
    height = max(int(image.shape[1]) for _, image in images)
    width = max(int(image.shape[2]) for _, image in images)
    downscale = int(getattr(vae, "downscale_ratio", 0) or 0)
    if downscale != 16:
        raise ValueError(f"Flux.2 VAE downscale_ratio must be 16, got {downscale}.")
    if target_latent is not None:
        samples = target_latent.get("samples") if isinstance(target_latent, dict) else None
        if not isinstance(samples, torch.Tensor) or samples.ndim != 4 or samples.shape[1] != FLUX2_LATENT_CHANNELS:
            shape = tuple(samples.shape) if isinstance(samples, torch.Tensor) else None
            raise ValueError(
                "target_latent must be a Flux.2 LATENT with samples [B,128,H,W]; "
                f"got {shape}."
            )
        batch = int(samples.shape[0])
        height = int(samples.shape[-2]) * downscale
        width = int(samples.shape[-1]) * downscale
    else:
        height = (height // downscale) * downscale
        width = (width // downscale) * downscale
    if height <= 0 or width <= 0:
        raise ValueError(f"Images are too small for Flux.2 VAE encoding: {(height, width)}.")

    prepared_images = {
        name: align_batch(_resize_bhwc(image, height, width), batch, name)
        for name, image in images
    }
    repaint_mask = (
        _normalize_mask(mask, batch, height, width)
        if mask is not None
        else torch.ones((batch, height, width), dtype=torch.float32)
    )
    preserved_mask = 1.0 - repaint_mask

    encoded: dict[str, torch.Tensor] = {}
    if "control_image" in prepared_images:
        encoded["control"] = vae.encode(prepared_images["control_image"])
        validate_flux2_vae_contract(vae, encoded["control"])
    if "inpaint_image" in prepared_images:
        masked_image = prepared_images["inpaint_image"] * preserved_mask.unsqueeze(-1)
        encoded["inpaint"] = vae.encode(masked_image)
        validate_flux2_vae_contract(vae, encoded["inpaint"])

    template = next(iter(encoded.values()))
    control_latents = encoded.get("control", torch.zeros_like(template))
    inpaint_latents = encoded.get("inpaint", torch.zeros_like(template))
    if control_latents.shape != inpaint_latents.shape:
        raise ValueError(
            "VAE produced different control and inpaint latent shapes; no token resizing is permitted: "
            f"{tuple(control_latents.shape)} versus {tuple(inpaint_latents.shape)}."
        )

    packed = pack_control_context(control_latents, preserved_mask.unsqueeze(1), inpaint_latents).contiguous()
    return PreparedFlux2FunContext(
        packed=packed,
        main_tokens=int(packed.shape[1]),
        latent_height=int(control_latents.shape[-2]),
        latent_width=int(control_latents.shape[-1]),
        batch_size=int(packed.shape[0]),
    )


def append_reference_zeros(context: PreparedFlux2FunContext, reference_tokens: int, target_batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    packed = align_batch(context.packed, target_batch, "prepared Flux2 Fun context").to(device=device, dtype=dtype)
    if reference_tokens < 0:
        raise ValueError("reference_tokens cannot be negative.")
    if reference_tokens == 0:
        return packed
    zeros = torch.zeros((target_batch, reference_tokens, context.channels), device=device, dtype=dtype)
    return torch.cat((packed, zeros), dim=1)
