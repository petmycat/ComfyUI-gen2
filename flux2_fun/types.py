from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Tuple

import torch


CONTROL_BLOCK_LAYERS = (0, 2, 4, 6)


@dataclass(frozen=True)
class CheckpointProfile:
    name: str
    tensor_count: int
    hidden_size: int
    control_dim: int
    block_count: int
    mlp_hidden_dim: int
    num_heads: int
    head_dim: int
    block_layers: Tuple[int, ...]
    sha256: str
    snapshot: str
    shapes: Mapping[str, Tuple[int, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shapes", MappingProxyType(dict(self.shapes)))


@dataclass(frozen=True)
class Flux2FunModelHandle:
    model: torch.nn.Module
    patcher: Any
    profile: CheckpointProfile
    storage_dtype: torch.dtype
    compute_dtype: torch.dtype
    checkpoint_path: str
    checkpoint_sha256: str | None


@dataclass(frozen=True)
class PreparedFlux2FunContext:
    packed: torch.Tensor
    main_tokens: int
    latent_height: int
    latent_width: int
    batch_size: int
    channels: int = 260
    profile_name: str = "flux2-dev-fun-controlnet-union-2602"


@dataclass(frozen=True)
class Flux2FunControlDescriptor:
    model: Flux2FunModelHandle
    context: PreparedFlux2FunContext
    strength: float
    start_percent: float
    end_percent: float

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise ValueError("Flux2 Fun control strength must be non-negative.")
        if not 0.0 <= self.start_percent <= 1.0:
            raise ValueError("Flux2 Fun start_percent must be in [0, 1].")
        if not 0.0 <= self.end_percent <= 1.0:
            raise ValueError("Flux2 Fun end_percent must be in [0, 1].")
        if self.start_percent > self.end_percent:
            raise ValueError("Flux2 Fun start_percent cannot exceed end_percent.")


@dataclass(frozen=True)
class Flux2FunControlGroup:
    descriptors: Tuple[Flux2FunControlDescriptor, ...]

    def __post_init__(self) -> None:
        if not self.descriptors:
            raise ValueError("Flux2 Fun control group cannot be empty.")
        object.__setattr__(self, "descriptors", tuple(self.descriptors))

    @classmethod
    def from_value(cls, value: Flux2FunControlDescriptor | "Flux2FunControlGroup") -> "Flux2FunControlGroup":
        if isinstance(value, cls):
            return value
        if isinstance(value, Flux2FunControlDescriptor):
            return cls((value,))
        raise TypeError(f"Unsupported Flux2 Fun control value: {type(value).__name__}")
