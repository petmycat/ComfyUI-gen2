from __future__ import annotations

import logging
from dataclasses import dataclass, replace as dataclass_replace
from typing import Callable

import torch

from .preprocess import append_reference_zeros
from .types import CONTROL_BLOCK_LAYERS, Flux2FunControlDescriptor, Flux2FunControlGroup


LOGGER = logging.getLogger(__name__)
DISPATCHER_ATTACHMENT_KEY = "gen2_flux2_fun_dispatcher"
ADDITIONAL_MODELS_KEY = "gen2_flux2_fun_control_models"
STATE_KEY = "gen2_flux2_fun_forward_state"
WRAPPER_KEY = "gen2_flux2_fun_forward_wrapper"
OWNER_MARKER = "gen2_flux2_fun"
MULTIGPU_CALLBACK_KEY = "gen2_flux2_fun_multigpu_rebind"


@dataclass
class ForwardState:
    descriptors: tuple[Flux2FunControlDescriptor, ...]
    hints: tuple[torch.Tensor, ...] | None = None


def _reference_tokens(transformer_options: dict) -> int:
    value = transformer_options.get("reference_image_num_tokens", ())
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    return sum(int(item) for item in value)


def _modulation_dims(img_tokens: int, main_tokens: int, vec) -> list[tuple[int, int, int]] | None:
    if img_tokens == main_tokens:
        return None
    first_mod = vec[0][0]
    scale = first_mod.scale
    if scale.ndim < 3 or scale.shape[1] == 1:
        return None
    if scale.shape[1] != 2:
        raise ValueError(
            "Flux2 Fun expected one global image modulation region or two ComfyUI timestep-zero regions; "
            f"got {scale.shape[1]}."
        )
    return [(0, main_tokens, 0), (main_tokens, img_tokens, 1)]


def _schedule_active(descriptor: Flux2FunControlDescriptor, timestep: torch.Tensor, model_sampling) -> bool:
    if descriptor.strength == 0.0:
        return False
    current = float(timestep.detach().float().max().cpu())
    start_sigma = model_sampling.percent_to_sigma(descriptor.start_percent)
    end_sigma = model_sampling.percent_to_sigma(descriptor.end_percent)
    start_t = float(model_sampling.timestep(torch.as_tensor(start_sigma)).detach().float().cpu())
    end_t = float(model_sampling.timestep(torch.as_tensor(end_sigma)).detach().float().cpu())
    lower, upper = sorted((start_t, end_t))
    return lower - 1e-6 <= current <= upper + 1e-6


class Flux2FunDispatcher:
    owner = OWNER_MARKER

    def __init__(self, descriptors: tuple[Flux2FunControlDescriptor, ...], model_sampling) -> None:
        self.descriptors = descriptors
        self.model_sampling = model_sampling

    def with_descriptors(self, descriptors: tuple[Flux2FunControlDescriptor, ...]) -> "Flux2FunDispatcher":
        return Flux2FunDispatcher(self.descriptors + descriptors, self.model_sampling)

    def diffusion_wrapper(self, executor, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None, transformer_options=None, **kwargs):
        options = dict(transformer_options or {})
        active = tuple(descriptor for descriptor in self.descriptors if _schedule_active(descriptor, timestep, self.model_sampling))
        options[STATE_KEY] = ForwardState(active)
        try:
            return executor(x, timestep, context, y, guidance, ref_latents, control, options, **kwargs)
        finally:
            options.pop(STATE_KEY, None)

    def _compute_hints(self, args: dict, state: ForwardState) -> tuple[torch.Tensor, ...]:
        img = args["img"]
        txt = args["txt"]
        vec = args["vec"]
        pe = args.get("pe")
        attn_mask = args.get("attn_mask")
        transformer_options = args.get("transformer_options") or {}
        reference_tokens = _reference_tokens(transformer_options)
        accumulated = [torch.zeros_like(img) for _ in CONTROL_BLOCK_LAYERS]

        for descriptor in state.descriptors:
            context = descriptor.context
            if context.main_tokens + reference_tokens != img.shape[1]:
                raise ValueError(
                    "Flux2 Fun token mismatch: prepared main tokens "
                    f"{context.main_tokens} + reference tokens {reference_tokens} != current image tokens {img.shape[1]}. "
                    "Token tensors are never resized heuristically."
                )
            control_context = append_reference_zeros(
                context,
                reference_tokens,
                img.shape[0],
                device=img.device,
                dtype=descriptor.model.compute_dtype,
            )
            modulation_dims_img = _modulation_dims(img.shape[1], context.main_tokens, vec)
            hints = descriptor.model.model.forward_control(
                img,
                control_context,
                txt,
                vec,
                pe,
                attn_mask=attn_mask,
                transformer_options=transformer_options,
                modulation_dims_img=modulation_dims_img,
            )
            if len(hints) != len(CONTROL_BLOCK_LAYERS):
                raise RuntimeError(f"Flux2 Fun branch returned {len(hints)} hints; expected {len(CONTROL_BLOCK_LAYERS)}.")
            for index, hint in enumerate(hints):
                accumulated[index] = accumulated[index] + hint.to(device=img.device, dtype=img.dtype) * descriptor.strength
        return tuple(accumulated)

    def replacement(self, block_index: int, existing: Callable | None = None):
        hint_index = CONTROL_BLOCK_LAYERS.index(block_index)

        def replace(args: dict, extra: dict):
            options = args.get("transformer_options") or {}
            state = options.get(STATE_KEY)
            if state is None or not state.descriptors:
                return existing(args, extra) if existing is not None else extra["original_block"](args)
            if block_index == CONTROL_BLOCK_LAYERS[0]:
                state.hints = self._compute_hints(args, state)
            if state.hints is None:
                raise RuntimeError(f"Flux2 Fun forward state was not initialized before double block {block_index}.")
            out = existing(args, extra) if existing is not None else extra["original_block"](args)
            out = dict(out)
            out["img"] = out["img"] + state.hints[hint_index]
            if block_index == CONTROL_BLOCK_LAYERS[-1]:
                state.hints = None
            return out

        replace.gen2_flux2_fun_owner = OWNER_MARKER
        replace.gen2_flux2_fun_block = block_index
        replace.gen2_flux2_fun_upstream = existing
        return replace


def validate_base_model(model) -> object:
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception as error:
        raise ValueError("Flux2 Fun Apply requires a ComfyUI MODEL exposing diffusion_model.") from error
    params = getattr(diffusion_model, "params", None)
    expected = {
        "hidden_size": 6144,
        "num_heads": 48,
        "global_modulation": True,
        "mlp_silu_act": True,
        "patch_size": 1,
        "out_channels": 128,
    }
    mismatches = {
        name: (getattr(params, name, None), value)
        for name, value in expected.items()
        if getattr(params, name, None) != value
    }
    if mismatches:
        raise ValueError(f"Only Flux.2 Dev is supported by Flux2 Fun 2602; incompatible base model fields: {mismatches}.")
    if len(getattr(diffusion_model, "double_blocks", ())) <= CONTROL_BLOCK_LAYERS[-1]:
        raise ValueError("Flux.2 base model does not expose double blocks 0, 2, 4, and 6.")
    return diffusion_model


def _existing_replacement(model, block_index: int):
    options = model.model_options.get("transformer_options", {})
    replacements = options.get("patches_replace", {}).get("dit", {})
    return replacements.get(("double_block", block_index))


def _rebind_multigpu_dispatcher(_source, cloned) -> None:
    dispatcher = cloned.get_attachment(DISPATCHER_ATTACHMENT_KEY)
    if not isinstance(dispatcher, Flux2FunDispatcher):
        return
    additional = cloned.get_additional_models_with_key(ADDITIONAL_MODELS_KEY)
    by_identity = {item.clone_base_uuid: item for item in additional}
    descriptors = []
    for descriptor in dispatcher.descriptors:
        identity = descriptor.model.patcher.clone_base_uuid
        patcher = by_identity.get(identity)
        if patcher is None:
            raise RuntimeError("Flux2 Fun multigpu clone is missing its managed control branch.")
        model_handle = dataclass_replace(descriptor.model, model=patcher.model, patcher=patcher)
        descriptors.append(dataclass_replace(descriptor, model=model_handle))
    rebound = Flux2FunDispatcher(tuple(descriptors), cloned.get_model_object("model_sampling"))
    cloned.set_attachments(DISPATCHER_ATTACHMENT_KEY, rebound)
    cloned.remove_wrappers_with_key("diffusion_model", WRAPPER_KEY)
    cloned.add_wrapper_with_key("diffusion_model", WRAPPER_KEY, rebound.diffusion_wrapper)
    for block_index in CONTROL_BLOCK_LAYERS:
        current = _existing_replacement(cloned, block_index)
        upstream = (
            getattr(current, "gen2_flux2_fun_upstream", None)
            if getattr(current, "gen2_flux2_fun_owner", None) == OWNER_MARKER
            else current
        )
        cloned.set_model_patch_replace(rebound.replacement(block_index, upstream), "dit", "double_block", block_index)
    cloned.remove_callbacks_with_key("on_deepclone_multigpu", MULTIGPU_CALLBACK_KEY)
    cloned.add_callback_with_key("on_deepclone_multigpu", MULTIGPU_CALLBACK_KEY, _rebind_multigpu_dispatcher)


def install_flux2_fun_dispatcher(model, group: Flux2FunControlGroup):
    validate_base_model(model)
    cloned = model.clone()
    existing_dispatcher = cloned.get_attachment(DISPATCHER_ATTACHMENT_KEY)
    if existing_dispatcher is not None and not isinstance(existing_dispatcher, Flux2FunDispatcher):
        raise RuntimeError("The MODEL already contains an unknown object under the Gen2 Flux2 Fun dispatcher key.")

    model_sampling = cloned.get_model_object("model_sampling")
    dispatcher = (
        existing_dispatcher.with_descriptors(group.descriptors)
        if existing_dispatcher is not None
        else Flux2FunDispatcher(group.descriptors, model_sampling)
    )
    cloned.set_attachments(DISPATCHER_ATTACHMENT_KEY, dispatcher)
    cloned.remove_wrappers_with_key("diffusion_model", WRAPPER_KEY)
    cloned.add_wrapper_with_key("diffusion_model", WRAPPER_KEY, dispatcher.diffusion_wrapper)

    for block_index in CONTROL_BLOCK_LAYERS:
        existing = _existing_replacement(cloned, block_index)
        if existing is not None and getattr(existing, "gen2_flux2_fun_owner", None) == OWNER_MARKER:
            existing = getattr(existing, "gen2_flux2_fun_upstream", None)
        elif existing is not None:
            LOGGER.warning(
                "[Gen2] Flux2 Fun is composing after an existing double-block replacement at block %d (%r).",
                block_index,
                existing,
            )
        cloned.set_model_patch_replace(dispatcher.replacement(block_index, existing), "dit", "double_block", block_index)

    additional = list(cloned.get_additional_models_with_key(ADDITIONAL_MODELS_KEY))
    known = {getattr(item, "clone_base_uuid", id(item)) for item in additional}
    for descriptor in dispatcher.descriptors:
        patcher = descriptor.model.patcher
        identity = getattr(patcher, "clone_base_uuid", id(patcher))
        if identity not in known:
            additional.append(patcher)
            known.add(identity)
    cloned.set_additional_models(ADDITIONAL_MODELS_KEY, additional)
    cloned.remove_callbacks_with_key("on_deepclone_multigpu", MULTIGPU_CALLBACK_KEY)
    cloned.add_callback_with_key("on_deepclone_multigpu", MULTIGPU_CALLBACK_KEY, _rebind_multigpu_dispatcher)
    return cloned
