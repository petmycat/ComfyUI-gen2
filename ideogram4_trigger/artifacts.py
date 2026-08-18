from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

import torch
from safetensors import safe_open

from .types import (
    EXPECTED_HIDDEN_SIZE,
    IDEOGRAM4_LAYER_COUNT,
    V9_LORA_ALPHA,
    V9_LORA_RANK,
    V9_VIRTUAL_TOKEN_COUNT,
    ActivatorArtifacts,
    ArtifactComponent,
    ArtifactIdentity,
    ArtifactManifest,
    ModuleLoRA,
    TEModuleLoRAArtifact,
    TriggerEmbedding,
)

ARTIFACT_SCHEMA = "ai-toolkit.trigger-binding-artifact"
ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_TYPES = frozenset({"embedding", "te_adapter", "diffusion_lora"})
ARTIFACT_MANIFEST_SCHEMAS = {
    artifact_type: f"{ARTIFACT_SCHEMA}.{artifact_type}.v{ARTIFACT_SCHEMA_VERSION}"
    for artifact_type in ARTIFACT_TYPES
}

_METADATA_SCHEMA = "trigger_binding.schema"
_METADATA_VERSION = "trigger_binding.schema_version"
_METADATA_TYPE = "trigger_binding.artifact_type"
_METADATA_MANIFEST = "trigger_binding.manifest"
_METADATA_MANIFEST_SHA256 = "trigger_binding.manifest_sha256"
_REQUIRED_METADATA_KEYS = frozenset(
    {_METADATA_SCHEMA, _METADATA_VERSION, _METADATA_TYPE, _METADATA_MANIFEST, _METADATA_MANIFEST_SHA256}
)
_REQUIRED_MANIFEST_KEYS = frozenset(
    {
        "schema", "schema_version", "artifact_type", "artifact_schema",
        "phase_fingerprint", "source_fingerprint", "config_fingerprint", "tensors",
    }
)
_ALLOWED_MANIFEST_KEYS = _REQUIRED_MANIFEST_KEYS | {"extra"}
_TENSOR_SPEC_KEYS = frozenset({"shape", "dtype", "sha256"})
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TE_KEY = re.compile(r"^layer_(0|[1-9][0-9]*)__mlp__down_proj\.(down|up)\.weight$")

PathLike = str | os.PathLike[str]
T = TypeVar("T", bound=ArtifactComponent)


class ArtifactValidationError(ValueError):
    pass


def canonical_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("Artifact manifest is not canonical JSON data") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    normalized = tensor.detach().cpu().contiguous()
    if normalized.layout != torch.strided:
        raise ArtifactValidationError("Artifact tensors must use strided layout")
    return sha256_bytes(normalized.reshape(-1).view(torch.uint8).numpy().tobytes()) if normalized.numel() else sha256_bytes(b"")


def _require_plain_int(value: Any, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ArtifactValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ArtifactValidationError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _validate_json_tree(value: Any, label: str = "manifest") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactValidationError(f"{label} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactValidationError(f"{label} has a non-string object key")
            _validate_json_tree(item, f"{label}.{key}")
        return
    raise ArtifactValidationError(f"{label} contains unsupported JSON value {type(value).__name__}")


def _parse_manifest(metadata: Mapping[str, str] | None) -> ArtifactManifest:
    if metadata is None:
        raise ArtifactValidationError("Missing required safetensors metadata")
    missing = sorted(_REQUIRED_METADATA_KEYS.difference(metadata))
    if missing:
        raise ArtifactValidationError(f"Missing required safetensors metadata: {missing}")
    if metadata[_METADATA_SCHEMA] != ARTIFACT_SCHEMA or metadata[_METADATA_VERSION] != str(ARTIFACT_SCHEMA_VERSION):
        raise ArtifactValidationError("Artifact metadata schema/version mismatch")
    manifest_json = metadata[_METADATA_MANIFEST]
    if sha256_bytes(manifest_json.encode("utf-8")) != _require_sha256(metadata[_METADATA_MANIFEST_SHA256], "Artifact manifest metadata hash"):
        raise ArtifactValidationError("Artifact manifest metadata hash mismatch")
    try:
        raw = json.loads(manifest_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("Artifact manifest is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ArtifactValidationError("Artifact manifest must be a JSON object")
    _validate_json_tree(raw)
    if canonical_json_dumps(raw) != manifest_json:
        raise ArtifactValidationError("Artifact manifest is not canonically encoded")
    if not _REQUIRED_MANIFEST_KEYS.issubset(raw) or set(raw).difference(_ALLOWED_MANIFEST_KEYS):
        raise ArtifactValidationError("Artifact manifest has missing or unknown top-level keys")
    artifact_type = raw["artifact_type"]
    if raw["schema"] != ARTIFACT_SCHEMA or raw["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError("Artifact manifest schema/version mismatch")
    if artifact_type not in ARTIFACT_TYPES or metadata[_METADATA_TYPE] != artifact_type:
        raise ArtifactValidationError("Artifact type is unsupported or inconsistent")
    if raw["artifact_schema"] != ARTIFACT_MANIFEST_SCHEMAS[artifact_type]:
        raise ArtifactValidationError("Artifact-specific manifest schema mismatch")
    tensors = raw["tensors"]
    if not isinstance(tensors, dict) or not tensors:
        raise ArtifactValidationError("Artifact manifest tensor map must be non-empty")
    normalized: dict[str, dict[str, Any]] = {}
    for key, spec in tensors.items():
        if not isinstance(key, str) or not isinstance(spec, dict) or set(spec) != _TENSOR_SPEC_KEYS:
            raise ArtifactValidationError(f"Invalid tensor manifest entry for {key!r}")
        if not isinstance(spec["shape"], list):
            raise ArtifactValidationError(f"Artifact tensor {key!r} shape must be a list")
        dtype = spec["dtype"]
        if not isinstance(dtype, str) or not dtype or dtype.startswith("torch."):
            raise ArtifactValidationError(f"Artifact tensor {key!r} has invalid dtype name")
        normalized[key] = {
            "shape": [_require_plain_int(item, f"Artifact tensor {key!r} shape dimension") for item in spec["shape"]],
            "dtype": dtype,
            "sha256": _require_sha256(spec["sha256"], f"Artifact tensor {key!r} hash"),
        }
    return ArtifactManifest(
        schema=ARTIFACT_SCHEMA,
        schema_version=ARTIFACT_SCHEMA_VERSION,
        artifact_type=artifact_type,
        artifact_schema=raw["artifact_schema"],
        phase_fingerprint=_require_sha256(raw["phase_fingerprint"], "phase_fingerprint"),
        source_fingerprint=_require_sha256(raw["source_fingerprint"], "source_fingerprint"),
        config_fingerprint=_require_sha256(raw["config_fingerprint"], "config_fingerprint"),
        tensors=normalized,
        extra=raw.get("extra"),
    )


def _stat_signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _validated_path(path: PathLike) -> Path:
    artifact_path = Path(path).expanduser().resolve()
    if artifact_path.suffix.lower() != ".safetensors" or not artifact_path.is_file():
        raise ArtifactValidationError(f"Artifact must be an existing .safetensors file: {artifact_path}")
    return artifact_path


def _load_tensors(path: PathLike, expected_type: str, verify_hashes: bool, expected_file_sha256: str | None):
    artifact_path = _validated_path(path)
    before = _stat_signature(artifact_path)
    file_hash = sha256_file(artifact_path)
    if expected_file_sha256 is not None and file_hash != _require_sha256(expected_file_sha256, "Expected artifact file hash"):
        raise ArtifactValidationError("Artifact file SHA-256 mismatch")
    try:
        with safe_open(str(artifact_path), framework="pt", device="cpu") as handle:
            manifest = _parse_manifest(handle.metadata())
            if manifest.artifact_type != expected_type:
                raise ArtifactValidationError(f"Expected artifact type {expected_type!r}, got {manifest.artifact_type!r}")
            if set(handle.keys()) != set(manifest.tensors):
                raise ArtifactValidationError("Artifact tensor keys mismatch manifest")
            tensors = {key: handle.get_tensor(key).detach().cpu().contiguous() for key in sorted(handle.keys())}
    except ArtifactValidationError:
        raise
    except Exception as exc:
        raise ArtifactValidationError(f"Unable to read safetensors artifact: {artifact_path}") from exc
    after = _stat_signature(artifact_path)
    if before != after:
        raise ArtifactValidationError("Artifact file changed while it was being loaded")
    for key, tensor in tensors.items():
        spec = manifest.tensors[key]
        if tuple(tensor.shape) != tuple(spec["shape"]) or str(tensor.dtype).removeprefix("torch.") != spec["dtype"]:
            raise ArtifactValidationError(f"Artifact tensor {key!r} shape/dtype mismatch")
        if verify_hashes and tensor_sha256(tensor) != spec["sha256"]:
            raise ArtifactValidationError(f"Artifact tensor {key!r} SHA-256 mismatch")
    return tensors, manifest, ArtifactIdentity(artifact_path, file_hash, after[0], after[1], after[2])


def _require_exact_keys(tensors: Mapping[str, torch.Tensor], expected: Sequence[str]) -> None:
    if set(tensors) != set(expected):
        raise ArtifactValidationError(f"Artifact component keys mismatch: actual={sorted(tensors)}, expected={sorted(expected)}")


def _require_bfloat16(tensor: torch.Tensor, key: str) -> None:
    if tensor.dtype != torch.bfloat16:
        raise ArtifactValidationError(f"Artifact tensor {key!r} must use bfloat16")


def load_trigger_embedding(path: PathLike, verify_hashes: bool = True, expected_file_sha256: str | None = None, hidden_size: int = EXPECTED_HIDDEN_SIZE) -> TriggerEmbedding:
    tensors, manifest, identity = _load_tensors(path, "embedding", verify_hashes, expected_file_sha256)
    _require_exact_keys(tensors, ("frozen_initializer", "weight"))
    for key in ("frozen_initializer", "weight"):
        _require_bfloat16(tensors[key], key)
        if tuple(tensors[key].shape) != (V9_VIRTUAL_TOKEN_COUNT, hidden_size):
            raise ArtifactValidationError(f"Artifact tensor {key!r} must have shape [{V9_VIRTUAL_TOKEN_COUNT}, {hidden_size}]")
    return TriggerEmbedding(tensors["weight"], tensors["frozen_initializer"], manifest, identity)


def _expected_te_keys(layer_count: int = IDEOGRAM4_LAYER_COUNT) -> tuple[str, ...]:
    return tuple(
        f"layer_{layer}__mlp__down_proj.{branch}.weight"
        for layer in range(layer_count)
        for branch in ("down", "up")
    )


def load_te_adapter(path: PathLike, verify_hashes: bool = True, expected_file_sha256: str | None = None, layer_count: int = IDEOGRAM4_LAYER_COUNT) -> TEModuleLoRAArtifact:
    tensors, manifest, identity = _load_tensors(path, "te_adapter", verify_hashes, expected_file_sha256)
    _require_exact_keys(tensors, _expected_te_keys(layer_count))
    grouped: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        match = _TE_KEY.fullmatch(key)
        if match is None:
            raise ArtifactValidationError(f"Invalid V9 TE tensor key: {key!r}")
        layer, branch = int(match.group(1)), match.group(2)
        if layer >= layer_count:
            raise ArtifactValidationError(f"TE adapter layer {layer} is outside 0..{layer_count - 1}")
        _require_bfloat16(tensor, key)
        grouped.setdefault(layer, {})[branch] = tensor
    layers: dict[int, ModuleLoRA] = {}
    for layer in range(layer_count):
        down, up = grouped[layer]["down"], grouped[layer]["up"]
        if down.ndim != 2 or down.shape[0] != V9_LORA_RANK:
            raise ArtifactValidationError(f"Layer {layer} down tensor must be [{V9_LORA_RANK}, in_features]")
        if up.ndim != 2 or tuple(up.shape) != (up.shape[0], V9_LORA_RANK):
            raise ArtifactValidationError(f"Layer {layer} up tensor must be [out_features, {V9_LORA_RANK}]")
        layers[layer] = ModuleLoRA(down, up, layer, f"language_model.layers.{layer}.mlp.down_proj")
    return TEModuleLoRAArtifact(layers, manifest, identity)


def validate_component_compatibility(components: ActivatorArtifacts | Sequence[ArtifactComponent]) -> tuple[str, str, str] | None:
    values = components.components() if isinstance(components, ActivatorArtifacts) else tuple(components)
    if not values:
        return None
    expected = values[0].manifest.compatibility_fingerprint
    if any(item.manifest.compatibility_fingerprint != expected for item in values[1:]):
        raise ArtifactValidationError("Activator artifacts have incompatible phase/source/config fingerprints")
    return expected


class ArtifactCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[Any, ...], ArtifactComponent] = {}
        self._lock = threading.RLock()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _get_or_load(self, path: PathLike, loader: Callable[..., T], kind: str, verify_hashes: bool, expected_file_sha256: str | None, options: tuple[Any, ...], kwargs: Mapping[str, Any]) -> T:
        artifact_path = _validated_path(path)
        stat = _stat_signature(artifact_path)
        key = (str(artifact_path), kind, verify_hashes, expected_file_sha256, options, stat)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                return cast(T, cached)
            loaded = loader(artifact_path, verify_hashes=verify_hashes, expected_file_sha256=expected_file_sha256, **kwargs)
            if _stat_signature(artifact_path) != stat:
                raise ArtifactValidationError("Artifact file changed during cache population")
            self._entries = {k: v for k, v in self._entries.items() if k[:2] != key[:2]}
            self._entries[key] = loaded
            return loaded

    def load_embedding(self, path: PathLike, verify_hashes: bool = True, expected_file_sha256: str | None = None, hidden_size: int = EXPECTED_HIDDEN_SIZE) -> TriggerEmbedding:
        return self._get_or_load(path, load_trigger_embedding, "embedding", verify_hashes, expected_file_sha256, (hidden_size,), {"hidden_size": hidden_size})

    def load_te_adapter(self, path: PathLike, verify_hashes: bool = True, expected_file_sha256: str | None = None, layer_count: int = IDEOGRAM4_LAYER_COUNT) -> TEModuleLoRAArtifact:
        return self._get_or_load(path, load_te_adapter, "te_adapter", verify_hashes, expected_file_sha256, (layer_count,), {"layer_count": layer_count})


DEFAULT_ARTIFACT_CACHE = ArtifactCache()


def peek_artifact_type(path: PathLike) -> str:
    with safe_open(str(_validated_path(path)), framework="pt", device="cpu") as handle:
        return _parse_manifest(handle.metadata()).artifact_type


def load_embedding_artifact(path: PathLike) -> TriggerEmbedding:
    return DEFAULT_ARTIFACT_CACHE.load_embedding(path)


def load_te_adapter_artifact(path: PathLike) -> TEModuleLoRAArtifact:
    return DEFAULT_ARTIFACT_CACHE.load_te_adapter(path)


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMAS", "ARTIFACT_SCHEMA", "ARTIFACT_SCHEMA_VERSION", "ArtifactCache",
    "ArtifactValidationError", "DEFAULT_ARTIFACT_CACHE", "canonical_json_dumps", "load_embedding_artifact",
    "load_te_adapter", "load_te_adapter_artifact", "load_trigger_embedding", "peek_artifact_type",
    "sha256_bytes", "sha256_file", "tensor_sha256", "validate_component_compatibility",
]
