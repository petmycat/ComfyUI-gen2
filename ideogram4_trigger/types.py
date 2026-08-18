from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch


EXPECTED_HIDDEN_SIZE = 4096
IDEOGRAM4_LAYER_COUNT = 36
IDEOGRAM4_CAPTURE_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
V9_VIRTUAL_TOKEN_COUNT = 4
V9_LORA_RANK = 4
V9_LORA_ALPHA = 4.0


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: Path
    file_sha256: str
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    schema: str
    schema_version: int
    artifact_type: str
    artifact_schema: str
    phase_fingerprint: str
    source_fingerprint: str
    config_fingerprint: str
    tensors: Mapping[str, Mapping[str, Any]]
    extra: Any = None

    def __post_init__(self) -> None:
        specs = {
            key: MappingProxyType(
                {"shape": tuple(spec["shape"]), "dtype": spec["dtype"], "sha256": spec["sha256"]}
            )
            for key, spec in self.tensors.items()
        }
        object.__setattr__(self, "tensors", MappingProxyType(specs))
        object.__setattr__(self, "extra", _freeze_json(self.extra))

    @property
    def compatibility_fingerprint(self) -> tuple[str, str, str]:
        return self.phase_fingerprint, self.source_fingerprint, self.config_fingerprint


@dataclass(frozen=True, slots=True)
class TriggerEmbedding:
    weight: torch.Tensor
    frozen_initializer: torch.Tensor
    manifest: ArtifactManifest
    identity: ArtifactIdentity


@dataclass(frozen=True, slots=True)
class ModuleLoRA:
    down: torch.Tensor
    up: torch.Tensor
    layer_index: int
    module_name: str
    rank: int = V9_LORA_RANK
    alpha: float = V9_LORA_ALPHA


@dataclass(frozen=True, slots=True)
class TEModuleLoRAArtifact:
    layers: Mapping[int, ModuleLoRA]
    manifest: ArtifactManifest
    identity: ArtifactIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", MappingProxyType(dict(self.layers)))


ArtifactComponent = TriggerEmbedding | TEModuleLoRAArtifact
TriggerEmbeddingArtifact = TriggerEmbedding
TriggerTEAdapterArtifact = TEModuleLoRAArtifact


@dataclass(frozen=True, slots=True)
class ActivatorArtifacts:
    embedding: TriggerEmbedding | None = None
    te_adapter: TEModuleLoRAArtifact | None = None

    def components(self) -> tuple[ArtifactComponent, ...]:
        return tuple(item for item in (self.embedding, self.te_adapter) if item is not None)


@dataclass(frozen=True, slots=True)
class Ideogram4TriggerActivator:
    embedding: TriggerEmbedding
    te_adapter: TEModuleLoRAArtifact
    embedding_strength: float = 1.0
    internal_strength: float = 1.0

    def diagnostics_dict(self) -> dict[str, Any]:
        return {
            "topology": "v9-four-slot-down-proj-module-lora",
            "components": {"embedding": True, "te_adapter": True, "tap_adapters": False},
            "strengths": {"embedding": self.embedding_strength, "internal": self.internal_strength},
            "fingerprints": {
                "embedding": {
                    "phase": self.embedding.manifest.phase_fingerprint,
                    "source": self.embedding.manifest.source_fingerprint,
                    "config": self.embedding.manifest.config_fingerprint,
                },
                "te_adapter": {
                    "phase": self.te_adapter.manifest.phase_fingerprint,
                    "source": self.te_adapter.manifest.source_fingerprint,
                    "config": self.te_adapter.manifest.config_fingerprint,
                },
                "compatible": self.embedding.manifest.compatibility_fingerprint
                == self.te_adapter.manifest.compatibility_fingerprint,
            },
            "file_sha256": {
                "embedding": self.embedding.identity.file_sha256,
                "te_adapter": self.te_adapter.identity.file_sha256,
            },
            "hidden_size": int(self.embedding.weight.shape[-1]),
            "embedding_tokens": int(self.embedding.weight.shape[0]),
            "virtual_slots": V9_VIRTUAL_TOKEN_COUNT,
            "te_layers": len(self.te_adapter.layers),
            "rank": V9_LORA_RANK,
            "alpha": V9_LORA_ALPHA,
            "capture_layers": list(IDEOGRAM4_CAPTURE_LAYERS),
        }


@dataclass(frozen=True, slots=True)
class TriggerDiagnostics:
    mode: str
    rendered_text: str
    literal: str
    occurrence_count: int
    slot_count: int
    token_spans: tuple[tuple[int, int], ...]
    token_indices: tuple[int, ...]
    virtual_token_indices: tuple[int, ...]
    occurrence_indices: tuple[int, ...]
    atomic_token_id: int | None
    lookup_token_id: int | None
    sequence_length: int
    runtime_source: str
    backend_identity: str
    device: str
    dtype: str
    hooks_installed: int
    hooks_cleaned: bool
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rendered_text": self.rendered_text,
            "literal": self.literal,
            "occurrence_count": self.occurrence_count,
            "slot_count": self.slot_count,
            "token_spans": [list(span) for span in self.token_spans],
            "token_indices": list(self.token_indices),
            "virtual_token_indices": list(self.virtual_token_indices),
            "occurrence_indices": list(self.occurrence_indices),
            "atomic_token_id": self.atomic_token_id,
            "lookup_token_id": self.lookup_token_id,
            "sequence_length": self.sequence_length,
            "runtime_source": self.runtime_source,
            "backend_identity": self.backend_identity,
            "device": self.device,
            "dtype": self.dtype,
            "hooks_installed": self.hooks_installed,
            "hooks_cleaned": self.hooks_cleaned,
            "tap_adapters": False,
            "notes": list(self.notes),
        }
