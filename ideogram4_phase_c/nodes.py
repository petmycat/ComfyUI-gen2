from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import folder_paths
from comfy_api.latest import io

from .artifacts import load_router_bundle
from .runtime import apply_phase_c_strength
from .types import PhaseCRouterBundle

MODEL_CATEGORY = "gen2"
CATEGORY = "Gen2/Ideogram4 Phase C V2"
CUSTOM_ROUTER = io.Custom("GEN2_IDEOGRAM4_PHASE_C_V2_ROUTER")


def _filenames(suffix: str) -> list[str]:
    names = [name for name in folder_paths.get_filename_list(MODEL_CATEGORY) if name.lower().endswith(suffix)]
    return names or [f"<no {suffix} files>"]


def _json_names() -> list[str]:
    return ["<embedded>", *_filenames(".json")]


def _resolve(name: str, suffix: str) -> str:
    if name.startswith("<no ") or name == "<embedded>":
        raise FileNotFoundError(f"A real {suffix} file must be selected")
    path = Path(folder_paths.get_full_path_or_raise(MODEL_CATEGORY, name))
    if path.suffix.lower() != suffix:
        raise ValueError(f"Selected file must end with {suffix}: {name}")
    return str(path)


class LoadIdeogram4PhaseCV2Router(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gen2_LoadIdeogram4PhaseCV2Router",
            display_name="Load Ideogram4 Phase C V2 Router",
            category=CATEGORY,
            description="Loads and strictly validates a Phase C V2 router, config, and V3 group registry.",
            inputs=[
                io.Combo.Input("artifact_layout", options=["separate_files", "self_contained"], default="separate_files"),
                io.Combo.Input("router_name", options=_filenames(".safetensors")),
                io.Combo.Input("router_config_name", options=_json_names(), default="<embedded>"),
                io.Combo.Input("group_registry_name", options=_json_names(), default="<embedded>"),
                io.Combo.Input("source_manifest_name", options=_json_names(), default="<embedded>"),
                io.Boolean.Input("verify_hashes", default=True, advanced=True),
            ],
            outputs=[CUSTOM_ROUTER.Output("router"), io.String.Output("diagnostics")],
        )

    @classmethod
    def execute(
        cls,
        artifact_layout: str,
        router_name: str,
        router_config_name: str,
        group_registry_name: str,
        source_manifest_name: str,
        verify_hashes: bool,
    ) -> io.NodeOutput:
        router_path = _resolve(router_name, ".safetensors")
        if artifact_layout == "self_contained":
            if (
                router_config_name != "<embedded>"
                or group_registry_name != "<embedded>"
                or source_manifest_name != "<embedded>"
            ):
                raise ValueError("Self-contained layout requires <embedded> config, registry, and source selections")
            config_path = registry_path = source_path = None
        else:
            config_path = _resolve(router_config_name, ".json")
            registry_path = _resolve(group_registry_name, ".json")
            source_path = _resolve(source_manifest_name, ".json")
        bundle = load_router_bundle(
            router_path,
            artifact_layout=artifact_layout,
            router_config_path=config_path,
            group_registry_path=registry_path,
            source_manifest_path=source_path,
            verify_hashes=bool(verify_hashes),
        )
        diagnostics = json.dumps(bundle.diagnostics_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        return io.NodeOutput(bundle, diagnostics)


class ApplyIdeogram4PhaseCV2Strength(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gen2_ApplyIdeogram4PhaseCV2Strength",
            display_name="Apply Ideogram4 Phase C V2 Strength",
            category=CATEGORY,
            description="Converts the exact registered V3 MODEL LoRA into content-, timestep-, and group-gated residuals.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("conditioning"),
                CUSTOM_ROUTER.Input("router"),
                io.Float.Input("style_strength", default=0.5, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("debug_logging", default=False, advanced=True),
            ],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("conditioning")],
        )

    @classmethod
    def execute(
        cls,
        model: Any,
        conditioning: list[list[Any]],
        router: PhaseCRouterBundle,
        style_strength: float,
        debug_logging: bool,
    ) -> io.NodeOutput:
        if not isinstance(router, PhaseCRouterBundle):
            raise TypeError("router input is not a validated Phase C V2 router bundle")
        patched, conditioning_out = apply_phase_c_strength(
            model, conditioning, router, style_strength, debug_logging
        )
        return io.NodeOutput(patched, conditioning_out)


NODE_CLASS_MAPPINGS = {
    "Gen2_LoadIdeogram4PhaseCV2Router": LoadIdeogram4PhaseCV2Router,
    "Gen2_ApplyIdeogram4PhaseCV2Strength": ApplyIdeogram4PhaseCV2Strength,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    node_id: node.define_schema().display_name for node_id, node in NODE_CLASS_MAPPINGS.items()
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
