"""
Gen2 OutputPanel - the companion to Gen2 InputPanel.

It collects a workflow's outputs into one node so an external frontend driving
ComfyUI via the API can read results from a single place (the /history
response). Click "Configure" in the ComfyUI frontend to define N named inputs
(type + name); each name is the API-export key. Wire the paired InputPanel's
PANEL_LINK output into this node's PANEL_LINK input to bind the pair and let
the output side inherit the input side's config.

IMAGE inputs are saved to the ComfyUI output folder (like SaveImage) and their
filenames/URLs are returned via the node's UI payload.

A JSON schema string describing all parameters (names, types, defaults, ranges,
steps) is also returned in the UI payload, so the frontend can display it in a
copyable read-only textbox on the node body.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy_api.latest import io
from comfy_api.latest import ComfyExtension
from typing_extensions import override

from ._config import (
    MAX_PARAMS,
    SUPPORTED_TYPES,
    parse_config,
    build_schema_json,
)


def _build_param_inputs() -> list[io.Input]:
    """Build the fixed bank of wildcard inputs: PANEL_LINK + MAX_PARAMS *."""
    inputs: list[io.Input] = [
        io.AnyType.Input("PANEL_LINK", display_name="panel_link", tooltip="Connect from a Gen2 Input Panel's PANEL_LINK output.")
    ]
    for i in range(MAX_PARAMS):
        inputs.append(io.AnyType.Input(f"param_{i}", display_name=f"param_{i}", optional=True))
    return inputs


def _config_from_panel_link(panel_link: Any) -> list[dict]:
    """If PANEL_LINK carries the InputPanel's config, inherit it (full params)."""
    if isinstance(panel_link, dict) and "params" in panel_link:
        inherited = panel_link["params"]
        if isinstance(inherited, list):
            result = []
            for p in inherited:
                if not isinstance(p, dict) or not p.get("name"):
                    continue
                e = {
                    "name": str(p.get("name", "")),
                    "type": str(p.get("type", "STRING")).upper(),
                    "default": p.get("default"),
                    "min": p.get("min"),
                    "max": p.get("max"),
                    "step": p.get("step"),
                }
                if e["type"] == "INT" and p.get("controlMode"):
                    e["controlMode"] = p["controlMode"]
                result.append(e)
            return result
    return []


def _save_image(image: torch.Tensor, filename_prefix: str = "Gen2OutputPanel") -> dict:
    """Save a single IMAGE tensor (B,H,W,C float32 in [0,1]) to the output dir.

    Mirrors core SaveImage.save_images but for one image at a time. Returns the
    {filename, subfolder, type} dict that the frontend turns into a view URL.
    """
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_dir, int(image.shape[2]), int(image.shape[1])
    )
    results = []
    for batch_number, img in enumerate(image):
        i = 255.0 * img.cpu().numpy()
        pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        file = f"{filename}_{counter:05}_.png"
        pil.save(os.path.join(full_output_folder, file), compress_level=4)
        results.append({"filename": file, "subfolder": subfolder, "type": "output"})
        counter += 1
    return results[0] if results else {}


class Gen2OutputPanel(io.ComfyNode):
    """Configurable output collection panel.

    Click "Configure" in the ComfyUI frontend to define named inputs. Wire the
    paired Gen2 Input Panel's PANEL_LINK output into this node's PANEL_LINK
    input. IMAGE inputs are saved to the output folder and their URLs returned
    via /history; other types are passed through as-is. A JSON schema of the
    parameter definitions is shown in a copyable textbox on the node.
    """

    class ConfigValues(dict):
        pass

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gen2_OutputPanel",
            display_name="Gen2 Output Panel",
            category="Gen2/API",
            description=(
                "Collects a workflow's outputs into one node. Click Configure "
                "to define named inputs. IMAGE inputs are saved to the output "
                "folder and their URLs returned via /history. Wire a Gen2 Input "
                "Panel's PANEL_LINK output into this node's PANEL_LINK input. "
                "A JSON schema of the parameter definitions is shown on the node."
            ),
            inputs=[
                io.String.Input("_config", default="[]", multiline=False),
                *_build_param_inputs(),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id, io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
            accept_all_inputs=True,
            not_idempotent=True,
        )

    @classmethod
    def validate_inputs(cls, _config: str = "[]", **kwargs) -> bool:
        return True

    @classmethod
    def execute(cls, _config: str = "[]", **kwargs) -> io.NodeOutput:
        # Prefer config inherited from the InputPanel via PANEL_LINK (full param
        # defs including ranges/steps); fall back to this node's own _config.
        panel_link = kwargs.get("PANEL_LINK")
        params = _config_from_panel_link(panel_link)
        if not params:
            params = parse_config(_config)

        ui_images: list[dict] = []
        collected: dict[str, Any] = {}

        for i, p in enumerate(params[:MAX_PARAMS]):
            name = p["name"]
            ptype = p["type"]
            # Param values arrive via the wildcard inputs param_0..param_31, but
            # with accept_all_inputs they may also surface as kwargs keyed by the
            # frontend-visible name. Prefer the wildcard slot, then the name.
            val = kwargs.get(f"param_{i}")
            if val is None:
                val = kwargs.get(name)

            if ptype == "IMAGE" and isinstance(val, torch.Tensor):
                saved = _save_image(val, filename_prefix=f"Gen2OutputPanel_{name}")
                if saved:
                    ui_images.append({"name": name, **saved})
                    collected[name] = saved
                continue

            collected[name] = val

        # JSON schema string for the frontend textbox (from the full param defs).
        schema_json = build_schema_json(params)

        ui_payload = {"images": ui_images, "params": collected, "schema_json": schema_json}
        return io.NodeOutput(ui=ui_payload)


class Gen2OutputPanelExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Gen2OutputPanel]


async def comfy_entrypoint() -> Gen2OutputPanelExtension:
    return Gen2OutputPanelExtension()
