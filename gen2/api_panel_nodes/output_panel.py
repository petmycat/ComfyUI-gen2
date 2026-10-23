"""Gen2 Output Panel: collect named workflow results and publish one document."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from ._config import (
    MAX_PARAMS, parse_input_config, parse_output_config, schema_entries,
    validate_output_value, validate_runtime_value,
)


def _build_param_inputs() -> list[io.Input]:
    inputs: list[io.Input] = [
        io.AnyType.Input(
            "PANEL_LINK",
            display_name="panel_link",
            tooltip="Connect from a Gen2 Input Panel to include input schema and latest values.",
        )
    ]
    for i in range(MAX_PARAMS):
        inputs.append(io.AnyType.Input(f"param_{i}", display_name=f"param_{i}", optional=True))
    return inputs


def _input_document_from_panel_link(panel_link: Any) -> dict[str, Any]:
    if not isinstance(panel_link, dict):
        raise ValueError("PANEL_LINK must come from a Gen2 Input Panel.")
    if panel_link.get("version", 1) != 1:
        raise ValueError("PANEL_LINK has an unsupported contract version.")

    inputs = panel_link.get("inputs")
    if isinstance(inputs, dict):
        schema = inputs.get("schema")
        values = inputs.get("latest_values")
        if not isinstance(schema, list) or not isinstance(values, dict):
            raise ValueError("PANEL_LINK inputs must contain schema and latest_values.")
        params = parse_input_config(schema)
        if set(values) != {p["name"] for p in params}:
            raise ValueError("PANEL_LINK latest_values must match the input schema names.")
        values = {p["name"]: validate_runtime_value(p, values[p["name"]]) for p in params}
        json.dumps(schema, allow_nan=False)
        return {"schema": schema, "latest_values": values}

    # Legacy PANEL_LINK carried only {"params": [...]}.
    params = panel_link.get("params")
    if isinstance(params, list):
        # Preserve the legacy schema-only envelope; it had no executed values.
        json.dumps(params, allow_nan=False)
        return {"schema": params, "latest_values": {}}
    raise ValueError("PANEL_LINK is missing its input document.")


def _validate_image(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Gen2_OutputPanel output {name!r} expects an IMAGE tensor.")
    if value.ndim != 4 or any(size <= 0 for size in value.shape) or value.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Gen2_OutputPanel output {name!r} needs a non-empty BHWC image with 1, 3, or 4 channels.")
    if not value.is_floating_point() or not torch.isfinite(value).all().item():
        raise ValueError(f"Gen2_OutputPanel output {name!r} needs finite floating-point pixels.")
    return value


def _validated_outputs(params: list[dict[str, Any]], values: dict[str, Any]) -> list[Any]:
    allowed = {"PANEL_LINK"} | {p["name"] for p in params} | {f"param_{i}" for i in range(MAX_PARAMS)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown Output Panel parameter(s): {', '.join(sorted(unknown))}.")
    for i in range(len(params), MAX_PARAMS):
        if values.get(f"param_{i}") is not None:
            raise ValueError(f"Output Panel param_{i} has a value but is not configured.")
    result = []
    for i, param in enumerate(params):
        value = values[f"param_{i}"] if f"param_{i}" in values else values.get(param["name"])
        if value is not None and param["type"] == "IMAGE":
            value = _validate_image(value, param["name"])
        else:
            if isinstance(value, np.generic):
                value = value.item()
            value = validate_output_value(param, value)
        result.append(value)
    return result


def _save_images(image: torch.Tensor, filename_prefix: str) -> list[dict[str, str]]:
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_dir, int(image.shape[2]), int(image.shape[1])
    )
    results: list[dict[str, str]] = []
    for img in image:
        pixels = np.clip(255.0 * img.detach().float().cpu().numpy(), 0, 255).astype(np.uint8)
        if pixels.shape[-1] == 1:
            pixels = pixels[..., 0]
        pil = Image.fromarray(pixels)
        file = f"{filename}_{counter:05}_.png"
        pil.save(os.path.join(full_output_folder, file), compress_level=4)
        results.append({"filename": file, "subfolder": subfolder, "type": "output"})
        counter += 1
    return results


class Gen2OutputPanel(io.ComfyNode):
    class ConfigValues(dict):
        pass

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gen2_OutputPanel",
            display_name="Gen2 Output Panel",
            category="Gen2/API",
            description=(
                "Collect named workflow outputs. Configure each output with a name and type. "
                "PANEL_LINK adds the paired Input Panel schema and latest values to the result document."
            ),
            inputs=[io.String.Input("_config", default="[]", multiline=False), *_build_param_inputs()],
            outputs=[],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            accept_all_inputs=True,
            not_idempotent=True,
        )

    @classmethod
    def validate_inputs(cls, _config: str = "[]", **kwargs) -> bool | str:
        try:
            _validated_outputs(parse_output_config(_config), kwargs)
            # The linked Input Panel has not executed during queue validation.
            if kwargs.get("PANEL_LINK") is not None:
                _input_document_from_panel_link(kwargs["PANEL_LINK"])
        except (ValueError, TypeError) as exc:
            return str(exc)
        return True

    @classmethod
    def execute(cls, _config: str = "[]", **kwargs) -> io.NodeOutput:
        output_params = parse_output_config(_config)
        input_document = _input_document_from_panel_link(kwargs.get("PANEL_LINK"))
        # Validate every value before saving any files so a bad later scalar
        # cannot leave partial image results behind.
        values = _validated_outputs(output_params, kwargs)
        ui_images: list[dict[str, Any]] = []
        collected: dict[str, Any] = {}

        for i, param in enumerate(output_params):
            name = param["name"]
            ptype = param["type"]
            val = values[i]

            if ptype == "IMAGE":
                if val is None:
                    collected[name] = None
                    continue
                # Names remain unchanged in JSON; only the on-disk prefix is a
                # filename component, never a path or a ComfyUI format token.
                safe_name = re.sub(r"[^\w.-]+", "_", name).strip(".")[:80] or "image"
                saved = _save_images(val, filename_prefix=f"Gen2OutputPanel_{safe_name}")
                collected[name] = saved
                ui_images.extend({"name": name, **image_ref} for image_ref in saved)
                continue

            collected[name] = val

        output_schema = schema_entries(output_params, "output")
        document = {
            "version": 1,
            "inputs": input_document,
            "outputs": {"schema": output_schema, "latest_values": collected},
        }
        document_json = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)

        # Keep the original fields for integrations that already read them.
        # ComfyUI concatenates each UI field as a list across executions.
        # Keep images flat, and wrap structured/scalar compatibility fields.
        ui_payload = {
            "images": ui_images,
            "params": [collected],
            "schema_json": [document_json],
            "document": [document],
            "document_json": [document_json],
        }
        return io.NodeOutput(ui=ui_payload)


class Gen2OutputPanelExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Gen2OutputPanel]


async def comfy_entrypoint() -> Gen2OutputPanelExtension:
    return Gen2OutputPanelExtension()
