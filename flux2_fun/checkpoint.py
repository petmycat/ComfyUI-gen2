from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Mapping

import torch

from .types import CONTROL_BLOCK_LAYERS, CheckpointProfile


OFFICIAL_2602_FILENAME = "FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors"
OFFICIAL_2602_SHA256 = "516532a885d12ae84bb3c6b24ef4816ac05ffa1c9c7b93476f74652eb0a7a794"
OFFICIAL_2602_SNAPSHOT = "b3dcd7836a0e926248dac3ccba8fc0853495764b"

_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "model.",
)
_BLOCK_KEY_RE = re.compile(r"^control_transformer_blocks\.(\d+)\.")


def normalize_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for original_key, value in state_dict.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in _PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
                    break
        if key in normalized:
            raise ValueError(f"Checkpoint key normalization collision: {original_key!r} -> {key!r}")
        normalized[key] = value
    return normalized


def infer_checkpoint_profile(state_dict: Mapping[str, torch.Tensor]) -> CheckpointProfile:
    state_dict = normalize_state_dict_keys(state_dict)
    required = {
        "control_img_in.weight",
        "control_img_in.bias",
        "control_transformer_blocks.0.before_proj.weight",
        "control_transformer_blocks.0.before_proj.bias",
    }
    missing = sorted(required - set(state_dict))
    if missing:
        raise ValueError(f"Not a Flux.2 Fun 2602 control checkpoint; missing required keys: {missing}")

    input_weight = state_dict["control_img_in.weight"]
    if input_weight.ndim != 2:
        raise ValueError("control_img_in.weight must be a rank-2 tensor.")
    hidden_size, control_dim = map(int, input_weight.shape)

    block_ids = sorted({int(match.group(1)) for key in state_dict if (match := _BLOCK_KEY_RE.match(key))})
    if block_ids != list(range(len(block_ids))):
        raise ValueError(f"Control block indices must be contiguous from zero, got {block_ids}.")
    block_count = len(block_ids)
    if block_count == 0:
        raise ValueError("Checkpoint contains no control transformer blocks.")

    head_dim = int(state_dict["control_transformer_blocks.0.attn.norm_q.weight"].numel())
    if head_dim <= 0 or hidden_size % head_dim:
        raise ValueError(f"Cannot infer a valid attention head count from hidden={hidden_size}, head_dim={head_dim}.")
    num_heads = hidden_size // head_dim

    ff_in = state_dict["control_transformer_blocks.0.ff.linear_in.weight"]
    ff_out = state_dict["control_transformer_blocks.0.ff.linear_out.weight"]
    if ff_in.ndim != 2 or ff_out.ndim != 2 or ff_in.shape[0] % 2:
        raise ValueError("Invalid Flux.2 SwiGLU feed-forward shapes.")
    mlp_hidden_dim = int(ff_in.shape[0] // 2)
    if tuple(ff_out.shape) != (hidden_size, mlp_hidden_dim):
        raise ValueError(
            f"Feed-forward output shape {tuple(ff_out.shape)} does not match inferred {(hidden_size, mlp_hidden_dim)}."
        )

    shapes = {key: tuple(int(dim) for dim in value.shape) for key, value in state_dict.items()}
    return CheckpointProfile(
        name="flux2-dev-fun-controlnet-union-2602",
        tensor_count=len(state_dict),
        hidden_size=hidden_size,
        control_dim=control_dim,
        block_count=block_count,
        mlp_hidden_dim=mlp_hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        block_layers=CONTROL_BLOCK_LAYERS,
        sha256=OFFICIAL_2602_SHA256,
        snapshot=OFFICIAL_2602_SNAPSHOT,
        shapes=shapes,
    )


def validate_official_2602_profile(profile: CheckpointProfile) -> None:
    expected = {
        "tensor_count": 76,
        "hidden_size": 6144,
        "control_dim": 260,
        "block_count": 4,
        "mlp_hidden_dim": 18432,
        "num_heads": 48,
        "head_dim": 128,
        "block_layers": CONTROL_BLOCK_LAYERS,
    }
    mismatches = {
        name: (getattr(profile, name), value)
        for name, value in expected.items()
        if getattr(profile, name) != value
    }
    if mismatches:
        details = ", ".join(f"{name}={actual!r} (expected {expected_value!r})" for name, (actual, expected_value) in mismatches.items())
        raise ValueError(f"Unsupported Flux.2 Fun checkpoint profile: {details}.")

    before_proj_keys = sorted(key for key in profile.shapes if ".before_proj." in key)
    expected_before = [
        "control_transformer_blocks.0.before_proj.bias",
        "control_transformer_blocks.0.before_proj.weight",
    ]
    if before_proj_keys != expected_before:
        raise ValueError(f"Official 2602 requires before_proj only on control block 0, got {before_proj_keys}.")


def validate_state_dict_shapes(state_dict: Mapping[str, torch.Tensor], profile: CheckpointProfile) -> None:
    normalized = normalize_state_dict_keys(state_dict)
    expected_keys = set(profile.shapes)
    actual_keys = set(normalized)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    wrong_shapes = sorted(
        (key, tuple(normalized[key].shape), profile.shapes[key])
        for key in expected_keys & actual_keys
        if tuple(normalized[key].shape) != profile.shapes[key]
    )
    if missing or unexpected or wrong_shapes:
        raise ValueError(
            "Strict Flux.2 Fun checkpoint validation failed: "
            f"missing={missing}, unexpected={unexpected}, wrong_shapes={wrong_shapes}."
        )


def strict_load_control_branch(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor], profile: CheckpointProfile) -> None:
    normalized = normalize_state_dict_keys(state_dict)
    validate_state_dict_shapes(normalized, profile)
    incompatible = model.load_state_dict(normalized, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Strict Flux.2 Fun weight loading returned incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
        )


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
