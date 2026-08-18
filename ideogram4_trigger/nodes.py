from __future__ import annotations

import json
import math
from typing import Any

import folder_paths
from comfy_api.latest import io

from .artifacts import ArtifactValidationError, load_embedding_artifact, load_te_adapter_artifact, peek_artifact_type
from .compatibility import inspect_ideogram4_clip
from .encoder import RUNTIME_MODES, encode_ideogram4_trigger
from .types import Ideogram4TriggerActivator, TriggerDiagnostics, TriggerEmbeddingArtifact, TriggerTEAdapterArtifact

MODEL_CATEGORY = "gen2"
CATEGORY = "Gen2/Ideogram4 Trigger V9"
CUSTOM_EMBEDDING = io.Custom("GEN2_IDEOGRAM4_V9_TRIGGER_EMBEDDING")
CUSTOM_TE_ADAPTER = io.Custom("GEN2_IDEOGRAM4_V9_TRIGGER_TE_MODULE_LORA")
CUSTOM_ACTIVATOR = io.Custom("GEN2_IDEOGRAM4_V9_TRIGGER_ACTIVATOR")
CUSTOM_DIAGNOSTICS = io.Custom("GEN2_IDEOGRAM4_V9_TRIGGER_DIAGNOSTICS")


def _artifact_names(expected_type: str) -> list[str]:
    names: list[str] = []
    for name in folder_paths.get_filename_list(MODEL_CATEGORY):
        try:
            path = folder_paths.get_full_path_or_raise(MODEL_CATEGORY, name)
            if peek_artifact_type(path) == expected_type:
                names.append(name)
        except (ArtifactValidationError, FileNotFoundError, OSError):
            continue
    return names or ["<no matching artifact>"]


def _resolve_artifact(name: str) -> str:
    if name == "<no matching artifact>":
        raise FileNotFoundError("No matching Ideogram4 V9 artifact exists under ComfyUI/models/gen2/")
    return folder_paths.get_full_path_or_raise(MODEL_CATEGORY, name)


def _finite_strength(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def compose_activator(embedding: TriggerEmbeddingArtifact, te_adapter: TriggerTEAdapterArtifact, embedding_strength: float = 1.0, internal_strength: float = 1.0) -> Ideogram4TriggerActivator:
    if embedding.manifest.compatibility_fingerprint != te_adapter.manifest.compatibility_fingerprint:
        raise ValueError("Embedding and TE module-LoRA artifacts have incompatible fingerprints")
    return Ideogram4TriggerActivator(
        embedding, te_adapter, _finite_strength("embedding_strength", embedding_strength),
        _finite_strength("internal_strength", internal_strength),
    )


class LoadIdeogram4TriggerEmbedding(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gen2_LoadIdeogram4V9TriggerEmbedding", display_name="Load Ideogram4 V9 Trigger Embedding",
            category=CATEGORY, description="Loads a strict V9 [4,H] trigger embedding artifact.",
            inputs=[io.Combo.Input("artifact_name", options=_artifact_names("embedding"))],
            outputs=[CUSTOM_EMBEDDING.Output("embedding")],
        )

    @classmethod
    def execute(cls, artifact_name: str) -> io.NodeOutput:
        return io.NodeOutput(load_embedding_artifact(_resolve_artifact(artifact_name)))


class LoadIdeogram4TriggerTEAdapter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gen2_LoadIdeogram4V9TriggerTEAdapter", display_name="Load Ideogram4 V9 TE Module-LoRA",
            category=CATEGORY, description="Loads exactly 36 independent rank-4 mlp.down_proj module-LoRA pairs.",
            inputs=[io.Combo.Input("artifact_name", options=_artifact_names("te_adapter"))],
            outputs=[CUSTOM_TE_ADAPTER.Output("te_adapter")],
        )

    @classmethod
    def execute(cls, artifact_name: str) -> io.NodeOutput:
        return io.NodeOutput(load_te_adapter_artifact(_resolve_artifact(artifact_name)))


class ComposeIdeogram4TriggerActivator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gen2_ComposeIdeogram4V9TriggerActivator", display_name="Compose Ideogram4 V9 Trigger Activator",
            category=CATEGORY, description="Composes mandatory fingerprint-matched V9 embedding and TE artifacts.",
            inputs=[
                CUSTOM_EMBEDDING.Input("embedding"), CUSTOM_TE_ADAPTER.Input("te_adapter"),
                io.Float.Input("embedding_strength", default=1.0, min=-10.0, max=10.0, step=0.05),
                io.Float.Input("internal_strength", default=1.0, min=-10.0, max=10.0, step=0.05),
            ], outputs=[CUSTOM_ACTIVATOR.Output("activator")],
        )

    @classmethod
    def execute(cls, embedding: TriggerEmbeddingArtifact, te_adapter: TriggerTEAdapterArtifact, embedding_strength: float, internal_strength: float) -> io.NodeOutput:
        return io.NodeOutput(compose_activator(embedding, te_adapter, embedding_strength, internal_strength))


class Ideogram4TriggerTextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gen2_Ideogram4V9TriggerTextEncode", display_name="Ideogram4 V9 Trigger Text Encode",
            category=CATEGORY,
            description="Strict native Ideogram4/Qwen3-VL V9 four-slot encoder returning standard CONDITIONING.",
            inputs=[
                io.Clip.Input("clip"), CUSTOM_ACTIVATOR.Input("activator"),
                io.String.Input("text", multiline=True, dynamic_prompts=True, default="[trigger]"),
                io.Combo.Input("mode", options=list(RUNTIME_MODES), default="semantic_only"),
                io.String.Input("placeholder", default="[trigger]"),
                io.String.Input("literal", default="<r1X1dOn9mA2>"),
                io.Int.Input("max_length", default=0, min=0, max=1048576, step=1, advanced=True),
            ], outputs=[io.Conditioning.Output("conditioning"), CUSTOM_DIAGNOSTICS.Output("diagnostics")],
        )

    @classmethod
    def execute(cls, clip: Any, activator: Ideogram4TriggerActivator, text: str, mode: str, placeholder: str, literal: str, max_length: int) -> io.NodeOutput:
        conditioning, diagnostics = encode_ideogram4_trigger(
            clip, activator, text, mode, placeholder, literal, max_length or None,
        )
        return io.NodeOutput(conditioning, diagnostics)


class Ideogram4TriggerActivatorDiagnostics(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Gen2_Ideogram4V9TriggerDiagnostics", display_name="Ideogram4 V9 Trigger Diagnostics",
            category=CATEGORY, description="Reports strict backend identity, artifacts, four-slot binding and hook lifecycle.",
            inputs=[CUSTOM_ACTIVATOR.Input("activator"), io.Clip.Input("clip", optional=True), CUSTOM_DIAGNOSTICS.Input("runtime_diagnostics", optional=True)],
            outputs=[io.String.Output("report")],
        )

    @classmethod
    def execute(cls, activator: Ideogram4TriggerActivator, clip: Any | None = None, runtime_diagnostics: TriggerDiagnostics | None = None) -> io.NodeOutput:
        payload = activator.diagnostics_dict()
        if clip is not None:
            payload["compatibility"] = inspect_ideogram4_clip(clip).as_dict()
        if runtime_diagnostics is not None:
            payload["runtime"] = runtime_diagnostics.as_dict()
        return io.NodeOutput(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


NODE_CLASS_MAPPINGS = {
    "Gen2_LoadIdeogram4V9TriggerEmbedding": LoadIdeogram4TriggerEmbedding,
    "Gen2_LoadIdeogram4V9TriggerTEAdapter": LoadIdeogram4TriggerTEAdapter,
    "Gen2_ComposeIdeogram4V9TriggerActivator": ComposeIdeogram4TriggerActivator,
    "Gen2_Ideogram4V9TriggerTextEncode": Ideogram4TriggerTextEncode,
    "Gen2_Ideogram4V9TriggerDiagnostics": Ideogram4TriggerActivatorDiagnostics,
}
NODE_DISPLAY_NAME_MAPPINGS = {node_id: node.define_schema().display_name for node_id, node in NODE_CLASS_MAPPINGS.items()}
