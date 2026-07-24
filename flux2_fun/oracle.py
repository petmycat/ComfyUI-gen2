from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch


ORACLE_TENSOR_NAMES = (
    "packed_context",
    "control_img_in",
    "block_0_before_proj",
    "block_0_output",
    "hint_0",
    "block_1_output",
    "hint_1",
    "block_2_output",
    "hint_2",
    "block_3_output",
    "hint_3",
    "denoising_forward",
)
REQUIRED_ORACLE_TENSOR_NAMES = frozenset(ORACLE_TENSOR_NAMES)


@dataclass(frozen=True)
class TensorComparison:
    name: str
    shape: tuple[int, ...]
    dtype: str
    reference_dtype: str
    max_abs: float
    mean_abs: float
    atol: float
    rtol: float
    passed: bool


def compare_tensors(name: str, actual: torch.Tensor, reference: torch.Tensor, *, atol: float, rtol: float) -> TensorComparison:
    if tuple(actual.shape) != tuple(reference.shape):
        return TensorComparison(
            name=name,
            shape=tuple(actual.shape),
            dtype=str(actual.dtype),
            reference_dtype=str(reference.dtype),
            max_abs=float("inf"),
            mean_abs=float("inf"),
            atol=atol,
            rtol=rtol,
            passed=False,
        )
    delta = (actual.detach().float().cpu() - reference.detach().float().cpu()).abs()
    return TensorComparison(
        name=name,
        shape=tuple(actual.shape),
        dtype=str(actual.dtype),
        reference_dtype=str(reference.dtype),
        max_abs=float(delta.max().item()) if delta.numel() else 0.0,
        mean_abs=float(delta.mean().item()) if delta.numel() else 0.0,
        atol=atol,
        rtol=rtol,
        passed=bool(torch.allclose(actual.detach().float().cpu(), reference.detach().float().cpu(), atol=atol, rtol=rtol)),
    )


def write_oracle_report(path: str | Path, comparisons: Iterable[TensorComparison], metadata: dict | None = None) -> None:
    payload = {
        "schema_version": 1,
        "metadata": metadata or {},
        "comparisons": [asdict(item) for item in comparisons],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_tensor_bundle(path: str | Path, tensors: dict[str, torch.Tensor], metadata: dict | None = None) -> None:
    tensor_names = set(tensors)
    unknown = sorted(tensor_names - REQUIRED_ORACLE_TENSOR_NAMES)
    if unknown:
        raise ValueError(f"Unknown Flux2 Fun oracle tensor names: {unknown}")
    missing = sorted(REQUIRED_ORACLE_TENSOR_NAMES - tensor_names)
    if missing:
        raise ValueError(f"Missing required Flux2 Fun oracle tensors: {missing}")
    payload = {
        "schema_version": 1,
        "metadata": metadata or {},
        "tensors": {name: tensor.detach().cpu() for name, tensor in tensors.items()},
    }
    torch.save(payload, path)
