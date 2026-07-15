"""Gen2 Output Panel: collect named workflow results and publish one document."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

from ._config import MAX_PARAMS, parse_output_config, schema_entries


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
    empty = {"schema": [], "latest_values": {}}
    if not isinstance(panel_link, dict):
        return empty

    inputs = panel_link.get("inputs")
    if isinstance(inputs, dict):
        schema = inputs.get("schema") if isinstance(inputs.get("schema"), list) else []
        values = inputs.get("latest_values") if isinstance(inputs.get("latest_values"), dict) else {}
        return {"schema": schema, "latest_values": values}

    # Legacy PANEL_LINK carried only {"params": [...]}.
    params = panel_link.get("params")
    if isinstance(params, list):
        return {"schema": params, "latest_values": {}}
    return empty


def _save_images(image: torch.Tensor, filename_prefix: str) -> list[dict[str, str]]:
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_dir, int(image.shape[2]), int(image.shape[1])
    )
    results: list[dict[str, str]] = []
    for img in image:
        pixels = 255.0 * img.cpu().numpy()
        pil = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
        file = f"{filename}_{counter:05}_.png"
        pil.save(os.path.join(full_output_folder, file), compress_level=4)
        results.append({"filename": file, "subfolder": subfolder, "type": "output"})
        counter += 1
    return results


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise ValueError(f"Gen2_OutputPanel cannot document value of type {type(value).__name__}.")


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
    def validate_inputs(cls, _config: str = "[]", **kwargs) -> bool:
        parse_output_config(_config)
        return True

    @classmethod
    def execute(cls, _config: str = "[]", **kwargs) -> io.NodeOutput:
        output_params = parse_output_config(_config)
        input_document = _input_document_from_panel_link(kwargs.get("PANEL_LINK"))
        ui_images: list[dict[str, Any]] = []
        collected: dict[str, Any] = {}

        for i, param in enumerate(output_params):
            name = param["name"]
            ptype = param["type"]
            slot_name = f"param_{i}"
            val = kwargs[slot_name] if slot_name in kwargs else kwargs.get(name)

            if ptype == "IMAGE":
                if val is None:
                    collected[name] = None
                    continue
                if not isinstance(val, torch.Tensor):
                    raise ValueError(f"Gen2_OutputPanel output {name!r} expects an IMAGE tensor.")
                saved = _save_images(val, filename_prefix=f"Gen2OutputPanel_{name}")
                collected[name] = saved
                ui_images.extend({"name": name, **image_ref} for image_ref in saved)
                continue

            collected[name] = _json_safe(val)

        output_schema = schema_entries(output_params, "output")
        document = {
            "version": 1,
            "inputs": input_document,
            "outputs": {"schema": output_schema, "latest_values": collected},
        }
        document_json = json.dumps(document, indent=2, ensure_ascii=False)

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
