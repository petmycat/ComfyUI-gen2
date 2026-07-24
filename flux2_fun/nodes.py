from __future__ import annotations

from pathlib import Path

import folder_paths

from .model_management import load_managed_control_branch
from .preprocess import prepare_control_context
from .runtime import install_flux2_fun_dispatcher
from .types import Flux2FunControlDescriptor, Flux2FunControlGroup


CATEGORY = "Gen2/Flux2 Fun ControlNet"


class Gen2_LoadFlux2FunControlNet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet_name": (folder_paths.get_filename_list("controlnet"),),
                "precision": (["auto", "bf16", "fp16"],),
            }
        }

    RETURN_TYPES = ("FLUX2_FUN_MODEL",)
    RETURN_NAMES = ("control_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load the official FLUX.2-dev-Fun-Controlnet-Union-2602 branch-only checkpoint."

    def load(self, controlnet_name: str, precision: str):
        path = folder_paths.get_full_path("controlnet", controlnet_name)
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"Flux2 Fun ControlNet file not found: {controlnet_name}")
        handle = load_managed_control_branch(path, precision)
        return (handle,)


class Gen2_PrepareFlux2FunControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"vae": ("VAE",)},
            "optional": {
                "control_image": ("IMAGE",),
                "mask": ("MASK",),
                "inpaint_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("FLUX2_FUN_CONTEXT",)
    RETURN_NAMES = ("control_context",)
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = "Prepare exact [control latents, preserved mask, inpaint latents] 260-channel Flux.2 Fun tokens."

    def prepare(self, vae, control_image=None, mask=None, inpaint_image=None):
        return (prepare_control_context(vae, control_image, mask, inpaint_image),)


class Gen2_ApplyFlux2FunControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "control_model": ("FLUX2_FUN_MODEL",),
                "control_context": ("FLUX2_FUN_CONTEXT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"control_group": ("FLUX2_FUN_CONTROL_GROUP",)},
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "Clone and locally patch a Flux.2 Dev MODEL; no global Flux monkey-patching is performed."

    def apply(
        self,
        model,
        control_model,
        control_context,
        strength: float,
        start_percent: float,
        end_percent: float,
        control_group=None,
    ):
        descriptor = Flux2FunControlDescriptor(
            model=control_model,
            context=control_context,
            strength=float(strength),
            start_percent=float(start_percent),
            end_percent=float(end_percent),
        )
        group = Flux2FunControlGroup((descriptor,))
        if control_group is not None:
            group = Flux2FunControlGroup.from_value(control_group)
            group = Flux2FunControlGroup(group.descriptors + (descriptor,))
        return (install_flux2_fun_dispatcher(model, group),)


class Gen2_CombineFlux2FunControls:
    @classmethod
    def INPUT_TYPES(cls):
        control_fields = {
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }
        return {
            "required": {
                "model_a": ("FLUX2_FUN_MODEL",),
                "context_a": ("FLUX2_FUN_CONTEXT",),
                "strength_a": control_fields["strength"],
                "start_a": control_fields["start"],
                "end_a": control_fields["end"],
                "model_b": ("FLUX2_FUN_MODEL",),
                "context_b": ("FLUX2_FUN_CONTEXT",),
                "strength_b": control_fields["strength"],
                "start_b": control_fields["start"],
                "end_b": control_fields["end"],
            },
            "optional": {"existing_group": ("FLUX2_FUN_CONTROL_GROUP",)},
        }

    RETURN_TYPES = ("FLUX2_FUN_CONTROL_GROUP",)
    RETURN_NAMES = ("control_group",)
    FUNCTION = "combine"
    CATEGORY = CATEGORY
    DESCRIPTION = "Experimental: combine immutable Flux2 Fun descriptors in deterministic order."

    def combine(
        self,
        model_a,
        context_a,
        strength_a,
        start_a,
        end_a,
        model_b,
        context_b,
        strength_b,
        start_b,
        end_b,
        existing_group=None,
    ):
        descriptors = () if existing_group is None else Flux2FunControlGroup.from_value(existing_group).descriptors
        descriptors += (
            Flux2FunControlDescriptor(model_a, context_a, float(strength_a), float(start_a), float(end_a)),
            Flux2FunControlDescriptor(model_b, context_b, float(strength_b), float(start_b), float(end_b)),
        )
        return (Flux2FunControlGroup(descriptors),)


NODE_CLASS_MAPPINGS = {
    "Gen2_LoadFlux2FunControlNet": Gen2_LoadFlux2FunControlNet,
    "Gen2_PrepareFlux2FunControl": Gen2_PrepareFlux2FunControl,
    "Gen2_ApplyFlux2FunControl": Gen2_ApplyFlux2FunControl,
    "Gen2_CombineFlux2FunControls": Gen2_CombineFlux2FunControls,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gen2_LoadFlux2FunControlNet": "Load Flux2 Fun ControlNet",
    "Gen2_PrepareFlux2FunControl": "Prepare Flux2 Fun Control",
    "Gen2_ApplyFlux2FunControl": "Apply Flux2 Fun Control",
    "Gen2_CombineFlux2FunControls": "Combine Flux2 Fun Controls (experimental)",
}
