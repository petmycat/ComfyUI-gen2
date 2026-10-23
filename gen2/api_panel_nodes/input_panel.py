"""
Gen2 InputPanel - a single configurable node that replaces a workflow's scattered
INPUT_* constant nodes.

Configure it in the ComfyUI frontend (a "Configure" button opens a popup) by
defining N parameters, each with a type (STRING / INT / FLOAT / BOOLEAN / IMAGE)
and a custom name. The name becomes both the output slot label and the key in
the API-export JSON.

Each INT/FLOAT parameter declares a range (min/max), step, and default. Literal
values are checked before queuing and linked values are checked at execution.
Workflow/API exports preserve current widget values separately from defaults.

Backend design (V3 API):
- accept_all_inputs=True lets user-named parameters reach execute() via **kwargs.
- validate_inputs checks configuration and literal values before execution.
- The schema declares a fixed bank of wildcard (*) outputs. The frontend renames
  the configured ones. Output 0 is PANEL_LINK, a mandatory tie to
  Gen2_OutputPanel that also carries the config.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
import node_helpers
from comfy_api.latest import io
from comfy_api.latest import ComfyExtension
from typing_extensions import override

from ._config import (
    MAX_PARAMS,
    parse_input_config,
    schema_entries,
    validate_runtime_value,
)


def _load_image_tensor(image_ref: str) -> torch.Tensor:
    """Load an image from the input folder by filename, returning (B,H,W,C)
    float32 in [0,1] — the ComfyUI IMAGE format.

    Mirrors core LoadImage.load_image but returns only the image (no mask).
    Accepts plain filenames ("foo.png") or annotated paths ("foo.png [input]").
    """
    from PIL import ImageSequence

    image_path = folder_paths.get_annotated_filepath(image_ref)
    output_images = []
    w, h = None, None
    with node_helpers.pillow(Image.open, image_path) as img:
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == "I":
                i = i.point(lambda v: v * (1 / 255))
            i = i.convert("RGB")
            if len(output_images) == 0:
                w, h = i.size
            if i.size != (w, h):
                continue
            arr = np.array(i).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(arr)[None,])
            if getattr(img, "format", None) == "MPO":
                break
    if not output_images:
        raise ValueError(f"Gen2_InputPanel: could not load image {image_ref!r}")
    if len(output_images) > 1:
        return torch.cat(output_images, dim=0)
    return output_images[0]


def _validated_values(params: list[dict[str, Any]], values: dict[str, Any], *, pending_links=False) -> dict[str, Any]:
    unknown = set(values) - {p["name"] for p in params}
    if unknown:
        raise ValueError(f"Unknown Input Panel parameter(s): {', '.join(sorted(unknown))}.")
    result = {}
    for param in params:
        name = param["name"]
        value = values.get(name, param["default"])
        # ComfyUI represents unresolved linked inputs as None during validation.
        # Execution still validates those values strictly after their producers run.
        if pending_links and value is None:
            continue
        value = validate_runtime_value(param, value)
        if param["type"] == "IMAGE" and not os.path.isfile(folder_paths.get_annotated_filepath(value)):
            raise ValueError(f"Parameter {name!r} image does not exist: {value!r}.")
        result[name] = value
    return result


def _build_param_outputs() -> list[io.Output]:
    """Build the fixed bank of wildcard outputs: PANEL_LINK + MAX_PARAMS *."""
    outputs: list[io.Output] = [io.AnyType.Output(id="PANEL_LINK", display_name="panel_link")]
    for i in range(MAX_PARAMS):
        outputs.append(io.AnyType.Output(id=f"param_{i}", display_name=f"param_{i}"))
    return outputs


class Gen2InputPanel(io.ComfyNode):
    """Configurable input parameter panel.

    Click "Configure" in the ComfyUI frontend to define parameters. Each
    parameter becomes a typed output slot whose name is the API-export key.
    Wire PANEL_LINK into Gen2_OutputPanel to bind the pair.
    """

    class ConfigValues(dict):
        """TypedDict-style hint for the _config widget value (a JSON string)."""
        pass

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gen2_InputPanel",
            display_name="Gen2 Input Panel",
            category="Gen2/API",
            description=(
                "A single configurable panel that exposes named, typed output "
                "slots for workflow parameters. Click Configure to define "
                "parameters (name + type + default + range/step for numbers). "
                "Wire its PANEL_LINK output into a Gen2 Output Panel. Designed "
                "for driving ComfyUI via the API export: each parameter name "
                "becomes a key in the node's inputs."
            ),
            inputs=[
                io.String.Input(
                    "_config",
                    default="[]",
                    multiline=False,
                ),
            ],
            outputs=_build_param_outputs(),
            hidden=[io.Hidden.unique_id],
            accept_all_inputs=True,
            not_idempotent=True,
        )

    @classmethod
    def validate_inputs(cls, _config: str = "[]", **kwargs) -> bool | str:
        try:
            _validated_values(parse_input_config(_config), kwargs, pending_links=True)
        except (ValueError, OSError) as exc:
            return str(exc)
        return True

    @classmethod
    def fingerprint_inputs(cls, _config: str = "[]", **kwargs):
        """Invalidate cached image outputs when an upload replaces the same file."""
        fingerprints = []
        for param in parse_input_config(_config):
            if param["type"] != "IMAGE":
                continue
            value = kwargs.get(param["name"], param["default"])
            if not isinstance(value, str) or not value.strip():
                # A linked filename is unavailable here; its underlying file
                # may still have changed even if its producing node is cached.
                return float("nan")
            digest = hashlib.sha256()
            try:
                with open(folder_paths.get_annotated_filepath(value.strip()), "rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
            except OSError:
                return float("nan")  # Never reuse a previously valid missing image.
            fingerprints.append((param["id"], digest.hexdigest()))
        return tuple(fingerprints)

    @classmethod
    def execute(cls, _config: str = "[]", **kwargs) -> io.NodeOutput:
        params = parse_input_config(_config)
        current_values = _validated_values(params, kwargs)
        runtime_values: list[Any] = []

        for p in params:
            name = p["name"]
            ptype = p["type"]
            value = current_values[name]
            runtime_values.append(_load_image_tensor(value) if ptype == "IMAGE" else value)

        panel_link = {
            "version": 1,
            "inputs": {
                "schema": schema_entries(params, "input"),
                "latest_values": current_values,
            },
        }
        out: list[Any] = [panel_link]
        for i in range(MAX_PARAMS):
            out.append(runtime_values[i] if i < len(runtime_values) else None)
        # ComfyUI's UI payload fields are list-valued. A scalar bool reaches
        # execution result merging as a non-iterable and aborts the workflow.
        return io.NodeOutput(*out, ui={"gen2_input_executed": [True]})


class Gen2InputPanelExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Gen2InputPanel]


async def comfy_entrypoint() -> Gen2InputPanelExtension:
    return Gen2InputPanelExtension()
