"""Load node implementations without importing ComfyUI or the pack entry point.

The stubs model only the host boundary; tensor operations under test remain real
CPU PyTorch. Every temporary sys.modules entry is restored after the import.
"""

import copy
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy  # Load dependencies before isolating sys.modules.
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_implementation(relative_path, name, stubs=None):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**(stubs or {}), name: module}):
        spec.loader.exec_module(module)
    return module


def lanpaint_module():
    comfy = types.ModuleType("comfy")
    comfy.model_patcher = types.ModuleType("comfy.model_patcher")
    comfy.model_patcher.create_model_options_clone = copy.deepcopy
    comfy.patcher_extension = types.ModuleType("comfy.patcher_extension")
    comfy.patcher_extension.WrappersMP = types.SimpleNamespace(SAMPLER_SAMPLE="sampler_sample")
    return load_implementation(
        "gen2/lanpaint_nodes/lanpaint_soft_denoise.py", "_gen2_test_lanpaint",
        {"comfy": comfy, "comfy.model_patcher": comfy.model_patcher,
         "comfy.patcher_extension": comfy.patcher_extension},
    )


def klein_module():
    comfy = types.ModuleType("comfy")
    comfy.ldm = types.ModuleType("comfy.ldm")
    comfy.ldm.flux = types.ModuleType("comfy.ldm.flux")
    comfy.ldm.flux.model = types.ModuleType("comfy.ldm.flux.model")
    comfy.ldm.flux.layers = types.ModuleType("comfy.ldm.flux.layers")

    class Flux:
        def forward_orig(self):
            return "native forward_orig"

        def _forward(self):
            return "native _forward"

    comfy.ldm.flux.model.Flux = Flux
    # Full transformer execution is outside these CPU wrapper lifecycle tests.
    comfy.ldm.flux.layers.timestep_embedding = lambda value, dim: value
    names = {
        "comfy": comfy, "comfy.ldm": comfy.ldm,
        "comfy.ldm.flux": comfy.ldm.flux,
        "comfy.ldm.flux.model": comfy.ldm.flux.model,
        "comfy.ldm.flux.layers": comfy.ldm.flux.layers,
    }
    module = load_implementation(
        "experimental/f2k/flux2_klein_fix.py", "_gen2_test_f2k", names,
    )
    return module, Flux


class ModelPatcherDouble:
    """Minimal independent options/attachment clone for node-boundary tests."""

    def __init__(self):
        self.wrappers = {}
        self.attachments = {}
        self.model_options = {}

    def clone(self):
        result = type(self)()
        result.wrappers = {kind: dict(items) for kind, items in self.wrappers.items()}
        result.attachments = dict(self.attachments)
        result.model_options = dict(self.model_options)
        return result

    def remove_wrappers_with_key(self, kind, key):
        self.wrappers.get(kind, {}).pop(key, None)

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {})[key] = wrapper

    def remove_attachments(self, key):
        self.attachments.pop(key, None)

    def set_attachments(self, key, value):
        self.attachments[key] = value

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


class LanPaintRuntimeDouble:
    """Deterministic runtime with explicit source noising and captured calls."""

    def __init__(self, source=None, generated=10.0):
        self.latent_image = torch.full((1, 1, 1, 3), 2.0) if source is None else source
        self.noise = torch.full_like(self.latent_image, 3.0)
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.PaintMethod = "test"
        self.LanPaint_early_stop = False
        self.generated = generated
        self.calls = []
        sampling = types.SimpleNamespace(
            noise_scaling=lambda sigma, noise, latent: latent + sigma * noise,
        )
        self.inner_model = types.SimpleNamespace(inner_model=types.SimpleNamespace(model_sampling=sampling))

    def __call__(self, x, sigma, **kwargs):
        self.calls.append((x.detach().clone(), sigma.detach().clone(), kwargs))
        return torch.full_like(x, self.generated)
