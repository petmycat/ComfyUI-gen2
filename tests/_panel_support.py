"""Isolate the ComfyUI boundary; exercise real panel code, PIL and CPU tensors."""

import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy
import torch
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]


class NodeOutput:
    def __init__(self, *args, ui=None):
        self.args = args
        self.result = args
        self.ui = ui


class Port:
    def __init__(self, id=None, **kwargs):
        self.id = id
        self.__dict__.update(kwargs)


def load_panels():
    package_name = "_gen2_panel_unit_tests"
    directory = ROOT / "gen2" / "api_panel_nodes"
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    folders = types.ModuleType("folder_paths")
    folders.get_annotated_filepath = lambda value: value
    folders.get_output_directory = lambda: "."
    helpers = types.ModuleType("node_helpers")
    helpers.pillow = lambda function, *args: function(*args)
    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.ComfyExtension = type("ComfyExtension", (), {})
    latest.io = types.SimpleNamespace(
        ComfyNode=type("ComfyNode", (), {}), NodeOutput=NodeOutput,
        Schema=lambda **kwargs: types.SimpleNamespace(**kwargs), Input=Port, Output=Port,
        AnyType=types.SimpleNamespace(Input=Port, Output=Port),
        String=types.SimpleNamespace(Input=Port),
        Hidden=types.SimpleNamespace(unique_id="unique_id", prompt="prompt", extra_pnginfo="extra_pnginfo"),
    )
    comfy_api.latest = latest
    modules = {}
    with patch.dict(sys.modules, {
        package_name: package, "folder_paths": folders, "node_helpers": helpers,
        "comfy_api": comfy_api, "comfy_api.latest": latest,
    }):
        for name in ("_config", "input_panel", "output_panel", "workflow_contract"):
            full_name = f"{package_name}.{name}"
            spec = importlib.util.spec_from_file_location(full_name, directory / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    return types.SimpleNamespace(**modules, folders=folders)


panels = load_panels()
