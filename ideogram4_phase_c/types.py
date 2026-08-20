from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch


@dataclass(frozen=True, slots=True)
class PhaseCRouterConfig:
    conditioning_dim: int
    activator_token_count: int
    activator_occurrence_count: int
    activator_token_dim: int
    temporal_anchor_count: int
    contextual_rank: int
    q_max: float
    run_fingerprint: str | None
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", MappingProxyType(dict(self.raw)))

    @property
    def activator_code_dim(self) -> int:
        return self.activator_token_count * self.activator_token_dim


@dataclass(frozen=True, slots=True)
class PhaseCRegistryModule:
    module_index: int
    module_name: str
    block_index: int | None
    kind: str
    group_id: str
    group_index: int
    rank: int
    down_shape: tuple[int, ...]
    up_shape: tuple[int, ...]
    down_fp32_norm: float
    up_fp32_norm: float
    module_active: bool


@dataclass(frozen=True, slots=True)
class PhaseCGroupRegistry:
    schema: str
    schema_version: int
    group_count: int
    module_count: int
    groups: tuple[Mapping[str, Any], ...]
    modules: tuple[PhaseCRegistryModule, ...]
    fingerprint: str
    canonical_json: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "groups",
            tuple(MappingProxyType(dict(group)) for group in self.groups),
        )


@dataclass(frozen=True, slots=True)
class PhaseCRouterWeights:
    activator_norm_weight: torch.Tensor
    activator_norm_bias: torch.Tensor
    activator_projection_weight: torch.Tensor
    activator_projection_bias: torch.Tensor
    universal_anchors: torch.Tensor
    context_in: torch.Tensor
    context_out: torch.Tensor

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "activator_norm.weight": self.activator_norm_weight,
            "activator_norm.bias": self.activator_norm_bias,
            "activator_projection.weight": self.activator_projection_weight,
            "activator_projection.bias": self.activator_projection_bias,
            "universal_anchors": self.universal_anchors,
            "context_in": self.context_in,
            "context_out": self.context_out,
        }


@dataclass(frozen=True, slots=True)
class PhaseCRouterBundle:
    source_layout: str
    router_path: Path
    router_file_sha256: str
    config_path: Path | None
    config_sha256: str
    registry_path: Path | None
    registry_sha256: str
    source_manifest_path: Path | None
    source_manifest_sha256: str
    expected_embedding_sha256: str
    expected_te_adapter_sha256: str
    config: PhaseCRouterConfig
    registry: PhaseCGroupRegistry
    weights: PhaseCRouterWeights

    def diagnostics_dict(self) -> dict[str, Any]:
        return {
            "source_layout": self.source_layout,
            "router_path": str(self.router_path),
            "router_file_sha256": self.router_file_sha256,
            "config_path": None if self.config_path is None else str(self.config_path),
            "config_sha256": self.config_sha256,
            "registry_path": None if self.registry_path is None else str(self.registry_path),
            "registry_sha256": self.registry_sha256,
            "source_manifest_path": None if self.source_manifest_path is None else str(self.source_manifest_path),
            "source_manifest_sha256": self.source_manifest_sha256,
            "expected_embedding_sha256": self.expected_embedding_sha256,
            "expected_te_adapter_sha256": self.expected_te_adapter_sha256,
            "run_fingerprint": self.config.run_fingerprint,
            "registry_fingerprint": self.registry.fingerprint,
            "group_count": self.registry.group_count,
            "module_count": self.registry.module_count,
            "conditioning_dim": self.config.conditioning_dim,
            "activator_token_count": self.config.activator_token_count,
            "activator_occurrence_count": self.config.activator_occurrence_count,
            "activator_token_dim": self.config.activator_token_dim,
            "temporal_anchor_count": self.config.temporal_anchor_count,
            "contextual_rank": self.config.contextual_rank,
            "q_max": self.config.q_max,
        }


__all__ = [
    "PhaseCGroupRegistry",
    "PhaseCRegistryModule",
    "PhaseCRouterBundle",
    "PhaseCRouterConfig",
    "PhaseCRouterWeights",
]
