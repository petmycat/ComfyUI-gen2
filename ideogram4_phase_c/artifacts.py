from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open

from .types import (
    PhaseCGroupRegistry,
    PhaseCRegistryModule,
    PhaseCRouterBundle,
    PhaseCRouterConfig,
    PhaseCRouterWeights,
)

ROUTER_CONFIG_SCHEMA = "ai-toolkit.ideogram4-v3-phase-c-v2-router-config"
ROUTER_CONFIG_VERSION = 2
REGISTRY_SCHEMA = "ai-toolkit.residual-gate-registry"
REGISTRY_VERSION = 1
BUNDLE_SCHEMA = "gen2.ideogram4-phase-c-v2-router-bundle"
BUNDLE_VERSION = 1
BUNDLE_METADATA_SCHEMA = "gen2.phase_c.schema"
BUNDLE_METADATA_VERSION = "gen2.phase_c.schema_version"
BUNDLE_METADATA_CONFIG = "gen2.phase_c.router_config"
BUNDLE_METADATA_CONFIG_SHA256 = "gen2.phase_c.router_config_sha256"
BUNDLE_METADATA_REGISTRY = "gen2.phase_c.group_registry"
BUNDLE_METADATA_REGISTRY_SHA256 = "gen2.phase_c.group_registry_sha256"
BUNDLE_METADATA_SOURCE = "gen2.phase_c.source_manifest"
BUNDLE_METADATA_SOURCE_SHA256 = "gen2.phase_c.source_manifest_sha256"
SOURCE_SCHEMA = "ai-toolkit.ideogram4-v3-phase-c-v2-source"
SOURCE_VERSION = 2
REQUIRED_ROUTER_KEYS = frozenset(
    {
        "activator_norm.weight",
        "activator_norm.bias",
        "activator_projection.weight",
        "activator_projection.bias",
        "universal_anchors",
        "context_in",
        "context_out",
    }
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PathLike = str | os.PathLike[str]


class PhaseCArtifactError(ValueError):
    pass


def canonical_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PhaseCArtifactError("Phase C JSON data is not canonicalizable") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_json_tree(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PhaseCArtifactError(f"{label} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PhaseCArtifactError(f"{label} contains a non-string key")
            _validate_json_tree(item, f"{label}.{key}")
        return
    raise PhaseCArtifactError(f"{label} contains unsupported value {type(value).__name__}")


def _load_json(path: PathLike) -> tuple[Path, dict[str, Any], str, str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.suffix.lower() != ".json":
        raise PhaseCArtifactError(f"Expected an existing JSON file: {resolved}")
    before = resolved.stat()
    try:
        raw_text = resolved.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhaseCArtifactError(f"Unable to read Phase C JSON: {resolved}") from exc
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise PhaseCArtifactError(f"Phase C JSON changed while loading: {resolved}")
    if not isinstance(payload, dict):
        raise PhaseCArtifactError(f"Phase C JSON must contain an object: {resolved}")
    _validate_json_tree(payload, resolved.name)
    return resolved, payload, sha256_file(resolved), canonical_json_dumps(payload)


def _required_int(payload: Mapping[str, Any], key: str, minimum: int = 1) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise PhaseCArtifactError(f"Router config {key!r} must be an integer >= {minimum}")
    return value


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise PhaseCArtifactError(f"Router config {key!r} must be finite")
    return float(value)


def parse_router_config(payload: Mapping[str, Any]) -> PhaseCRouterConfig:
    if payload.get("schema") != ROUTER_CONFIG_SCHEMA or payload.get("schema_version") != ROUTER_CONFIG_VERSION:
        raise PhaseCArtifactError("Router config schema/version mismatch")
    revision = payload.get("contract_revision")
    if revision != 4:
        raise PhaseCArtifactError("Router config contract_revision must be 4")
    if payload.get("canonical_api") not in (None, "toolkit.residual_gating.ResidualGateRouter"):
        raise PhaseCArtifactError("Router config canonical_api is unsupported")
    if payload.get("runtime_api") not in (None, "toolkit.residual_gating.ResidualGateRuntime"):
        raise PhaseCArtifactError("Router runtime_api is unsupported")
    if payload.get("conditioning_source") != "projected_private_activator_states":
        raise PhaseCArtifactError("Router conditioning source must be projected_private_activator_states")
    if payload.get("conditioning_normalization") not in (None, "shared_layer_norm"):
        raise PhaseCArtifactError("Router conditioning normalization is unsupported")
    if payload.get("activator_mask_schema") not in (None, "a1-a2-trigger-mask-v1"):
        raise PhaseCArtifactError("Router activator mask schema is unsupported")
    token_count = _required_int(payload, "activator_token_count")
    occurrence_count = _required_int(payload, "activator_occurrence_count")
    token_dim = _required_int(payload, "activator_token_dim")
    if occurrence_count != 3 or token_count != 4:
        raise PhaseCArtifactError("Router activator contract must be exactly three occurrences of four virtual tokens")
    if payload.get("activator_occurrence_mode") not in (None, "additive"):
        raise PhaseCArtifactError("Router occurrence mode is unsupported")
    if payload.get("activator_pre_router_aggregation") != "sum_by_virtual_token_index":
        raise PhaseCArtifactError("Router occurrence aggregation must be sum_by_virtual_token_index")
    if payload.get("temporal_interpolation") != "linear":
        raise PhaseCArtifactError("Router temporal interpolation must be linear")
    code_dim = payload.get("activator_code_dim")
    if code_dim is not None and code_dim != token_count * token_dim:
        raise PhaseCArtifactError("Router activator_code_dim is inconsistent")
    q_max = _required_float(payload, "q_max")
    if not 0.0 < q_max <= 0.5:
        raise PhaseCArtifactError("Router q_max must be in (0,0.5]")
    anchor_count = _required_int(payload, "temporal_anchor_count", 2)
    anchor_locations = payload.get("anchor_locations")
    expected_locations = [index / float(anchor_count - 1) for index in range(anchor_count)]
    if anchor_locations is not None and (
        not isinstance(anchor_locations, list)
        or len(anchor_locations) != anchor_count
        or any(
            not isinstance(actual, (int, float))
            or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-9)
            for actual, expected in zip(anchor_locations, expected_locations)
        )
    ):
        raise PhaseCArtifactError("Router anchor_locations do not match the canonical linear grid")
    run_fingerprint = payload.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or HEX_SHA256.fullmatch(run_fingerprint) is None:
        raise PhaseCArtifactError("Router run_fingerprint must be a SHA-256 digest")
    return PhaseCRouterConfig(
        conditioning_dim=_required_int(payload, "conditioning_dim"),
        activator_token_count=token_count,
        activator_occurrence_count=occurrence_count,
        activator_token_dim=token_dim,
        temporal_anchor_count=anchor_count,
        contextual_rank=_required_int(payload, "contextual_rank"),
        q_max=q_max,
        run_fingerprint=run_fingerprint,
        raw=payload,
    )


def _parse_shape(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value
    ):
        raise PhaseCArtifactError(f"{label} must be a non-empty positive integer shape")
    return tuple(value)


def parse_group_registry(payload: Mapping[str, Any], canonical_json: str | None = None) -> PhaseCGroupRegistry:
    if payload.get("schema") != REGISTRY_SCHEMA or payload.get("schema_version") != REGISTRY_VERSION:
        raise PhaseCArtifactError("Group registry schema/version mismatch")
    group_count = _required_int(payload, "group_count")
    module_count = _required_int(payload, "module_count")
    groups = payload.get("groups")
    modules = payload.get("modules")
    if not isinstance(groups, list) or len(groups) != group_count:
        raise PhaseCArtifactError("Group registry group_count mismatch")
    if not isinstance(modules, list) or len(modules) != module_count:
        raise PhaseCArtifactError("Group registry module_count mismatch")
    expected_indices = list(range(group_count))
    actual_indices = sorted(group.get("group_index") for group in groups if isinstance(group, dict))
    if actual_indices != expected_indices:
        raise PhaseCArtifactError("Group registry indices must be contiguous from zero")
    parsed_modules: list[PhaseCRegistryModule] = []
    names: set[str] = set()
    for row in modules:
        if not isinstance(row, dict):
            raise PhaseCArtifactError("Group registry module entries must be objects")
        name = row.get("module_name")
        group_index = row.get("group_index")
        if not isinstance(name, str) or not name or name in names:
            raise PhaseCArtifactError("Group registry module names must be unique and non-empty")
        if not isinstance(group_index, int) or isinstance(group_index, bool) or group_index not in expected_indices:
            raise PhaseCArtifactError(f"Group registry module {name!r} has invalid group_index")
        names.add(name)
        down_norm = row.get("down_fp32_norm")
        up_norm = row.get("up_fp32_norm")
        if not isinstance(down_norm, (int, float)) or not isinstance(up_norm, (int, float)):
            raise PhaseCArtifactError(f"Group registry module {name!r} lacks FP32 norm metadata")
        if not math.isfinite(float(down_norm)) or not math.isfinite(float(up_norm)):
            raise PhaseCArtifactError(f"Group registry module {name!r} has non-finite norms")
        parsed_modules.append(
            PhaseCRegistryModule(
                module_index=int(row.get("module_index", -1)),
                module_name=name,
                block_index=row.get("block_index"),
                kind=str(row.get("kind", "")),
                group_id=str(row.get("group_id", "")),
                group_index=group_index,
                rank=int(row.get("rank", 0)),
                down_shape=_parse_shape(row.get("down_shape"), f"{name}.down_shape"),
                up_shape=_parse_shape(row.get("up_shape"), f"{name}.up_shape"),
                down_fp32_norm=float(down_norm),
                up_fp32_norm=float(up_norm),
                module_active=bool(row.get("module_active", False)),
            )
        )
    canonical = canonical_json or canonical_json_dumps(dict(payload))
    fingerprint = sha256_bytes(canonical.encode("utf-8"))
    return PhaseCGroupRegistry(
        schema=REGISTRY_SCHEMA,
        schema_version=REGISTRY_VERSION,
        group_count=group_count,
        module_count=module_count,
        groups=tuple(groups),
        modules=tuple(parsed_modules),
        fingerprint=fingerprint,
        canonical_json=canonical,
    )


def _load_router_tensors(path: PathLike) -> tuple[Path, dict[str, torch.Tensor], Mapping[str, str], str]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.suffix.lower() != ".safetensors":
        raise PhaseCArtifactError(f"Expected an existing router safetensors file: {resolved}")
    before = resolved.stat()
    try:
        with safe_open(str(resolved), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if keys != REQUIRED_ROUTER_KEYS:
                raise PhaseCArtifactError(
                    f"Router tensor keys mismatch: missing={sorted(REQUIRED_ROUTER_KEYS - keys)}, "
                    f"unexpected={sorted(keys - REQUIRED_ROUTER_KEYS)}"
                )
            metadata = dict(handle.metadata() or {})
            tensors = {key: handle.get_tensor(key).detach().cpu().contiguous() for key in sorted(keys)}
    except PhaseCArtifactError:
        raise
    except Exception as exc:
        raise PhaseCArtifactError(f"Unable to read router safetensors: {resolved}") from exc
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise PhaseCArtifactError("Router safetensors changed while loading")
    for key, tensor in tensors.items():
        if tensor.dtype != torch.float32:
            raise PhaseCArtifactError(f"Router tensor {key!r} must be float32")
        if not torch.isfinite(tensor).all().item():
            raise PhaseCArtifactError(f"Router tensor {key!r} contains NaN or infinity")
    return resolved, tensors, metadata, sha256_file(resolved)


def _validate_shapes(
    tensors: Mapping[str, torch.Tensor], config: PhaseCRouterConfig, registry: PhaseCGroupRegistry
) -> PhaseCRouterWeights:
    dimensions = {
        "activator_norm.weight": (config.conditioning_dim,),
        "activator_norm.bias": (config.conditioning_dim,),
        "activator_projection.weight": (config.activator_token_dim, config.conditioning_dim),
        "activator_projection.bias": (config.activator_token_dim,),
        "universal_anchors": (config.temporal_anchor_count, registry.group_count),
        "context_in": (config.temporal_anchor_count, config.contextual_rank, config.activator_code_dim),
        "context_out": (config.temporal_anchor_count, registry.group_count, config.contextual_rank),
    }
    for key, expected in dimensions.items():
        if tuple(tensors[key].shape) != expected:
            raise PhaseCArtifactError(
                f"Router tensor {key!r} must have shape {expected}, got {tuple(tensors[key].shape)}"
            )
    active_registry = config.raw.get("active_registry")
    if active_registry is not None:
        embedded = canonical_json_dumps(active_registry)
        if embedded != registry.canonical_json:
            raise PhaseCArtifactError("Router config active_registry differs from group_registry")
    return PhaseCRouterWeights(
        tensors["activator_norm.weight"],
        tensors["activator_norm.bias"],
        tensors["activator_projection.weight"],
        tensors["activator_projection.bias"],
        tensors["universal_anchors"],
        tensors["context_in"],
        tensors["context_out"],
    )


def _parse_source_manifest(payload: Mapping[str, Any], config: PhaseCRouterConfig) -> tuple[str, str]:
    if payload.get("schema") != SOURCE_SCHEMA or payload.get("schema_version") != SOURCE_VERSION:
        raise PhaseCArtifactError("Phase C source manifest schema/version mismatch")
    if payload.get("contract_revision") != 4 or payload.get("status") != "resolved":
        raise PhaseCArtifactError("Phase C source manifest contract/status mismatch")
    if payload.get("run_fingerprint") != config.run_fingerprint:
        raise PhaseCArtifactError("Source manifest and router config run_fingerprint differ")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise PhaseCArtifactError("Phase C source manifest lacks inputs")

    def input_hash(key: str) -> str:
        reference = inputs.get(key)
        digest = reference.get("sha256") if isinstance(reference, Mapping) else None
        if not isinstance(digest, str) or HEX_SHA256.fullmatch(digest) is None:
            raise PhaseCArtifactError(f"Phase C source manifest lacks a valid {key} SHA-256")
        return digest

    return input_hash("best_embedding"), input_hash("best_te_adapter")


def load_router_bundle(
    router_path: PathLike,
    *,
    artifact_layout: str,
    router_config_path: PathLike | None = None,
    group_registry_path: PathLike | None = None,
    source_manifest_path: PathLike | None = None,
    verify_hashes: bool = True,
) -> PhaseCRouterBundle:
    router_file, tensors, metadata, router_hash = _load_router_tensors(router_path)
    if artifact_layout == "separate_files":
        if router_config_path is None or group_registry_path is None or source_manifest_path is None:
            raise PhaseCArtifactError("Separate layout requires router config, group registry, and source manifest files")
        config_file, config_payload, config_hash, config_json = _load_json(router_config_path)
        registry_file, registry_payload, registry_hash, registry_json = _load_json(group_registry_path)
        source_file, source_payload, source_hash, source_json = _load_json(source_manifest_path)
    elif artifact_layout == "self_contained":
        required = {
            BUNDLE_METADATA_SCHEMA,
            BUNDLE_METADATA_VERSION,
            BUNDLE_METADATA_CONFIG,
            BUNDLE_METADATA_CONFIG_SHA256,
            BUNDLE_METADATA_REGISTRY,
            BUNDLE_METADATA_REGISTRY_SHA256,
            BUNDLE_METADATA_SOURCE,
            BUNDLE_METADATA_SOURCE_SHA256,
        }
        missing = sorted(required.difference(metadata))
        if missing:
            raise PhaseCArtifactError(f"Self-contained router metadata is incomplete: {missing}")
        if metadata[BUNDLE_METADATA_SCHEMA] != BUNDLE_SCHEMA or metadata[BUNDLE_METADATA_VERSION] != str(BUNDLE_VERSION):
            raise PhaseCArtifactError("Self-contained router metadata schema/version mismatch")
        config_json = metadata[BUNDLE_METADATA_CONFIG]
        registry_json = metadata[BUNDLE_METADATA_REGISTRY]
        source_json = metadata[BUNDLE_METADATA_SOURCE]
        config_hash = sha256_bytes(config_json.encode("utf-8"))
        registry_hash = sha256_bytes(registry_json.encode("utf-8"))
        source_hash = sha256_bytes(source_json.encode("utf-8"))
        if verify_hashes and config_hash != metadata[BUNDLE_METADATA_CONFIG_SHA256]:
            raise PhaseCArtifactError("Embedded router config SHA-256 mismatch")
        if verify_hashes and registry_hash != metadata[BUNDLE_METADATA_REGISTRY_SHA256]:
            raise PhaseCArtifactError("Embedded group registry SHA-256 mismatch")
        if verify_hashes and source_hash != metadata[BUNDLE_METADATA_SOURCE_SHA256]:
            raise PhaseCArtifactError("Embedded source manifest SHA-256 mismatch")
        try:
            config_payload = json.loads(config_json)
            registry_payload = json.loads(registry_json)
            source_payload = json.loads(source_json)
        except json.JSONDecodeError as exc:
            raise PhaseCArtifactError("Embedded Phase C metadata is not valid JSON") from exc
        if (
            canonical_json_dumps(config_payload) != config_json
            or canonical_json_dumps(registry_payload) != registry_json
            or canonical_json_dumps(source_payload) != source_json
        ):
            raise PhaseCArtifactError("Embedded Phase C metadata must be canonical JSON")
        config_file = None
        registry_file = None
        source_file = None
    else:
        raise PhaseCArtifactError(f"Unsupported artifact_layout {artifact_layout!r}")
    config = parse_router_config(config_payload)
    registry = parse_group_registry(registry_payload, registry_json)
    embedding_hash, te_adapter_hash = _parse_source_manifest(source_payload, config)
    weights = _validate_shapes(tensors, config, registry)
    expected_registry_hash = config.raw.get("active_registry_fingerprint")
    if verify_hashes and expected_registry_hash is not None and expected_registry_hash != registry.fingerprint:
        raise PhaseCArtifactError("Router config active_registry_fingerprint mismatch")
    return PhaseCRouterBundle(
        source_layout=artifact_layout,
        router_path=router_file,
        router_file_sha256=router_hash,
        config_path=config_file,
        config_sha256=config_hash,
        registry_path=registry_file,
        registry_sha256=registry_hash,
        source_manifest_path=source_file,
        source_manifest_sha256=source_hash,
        expected_embedding_sha256=embedding_hash,
        expected_te_adapter_sha256=te_adapter_hash,
        config=config,
        registry=registry,
        weights=weights,
    )


__all__ = [
    "BUNDLE_SCHEMA",
    "BUNDLE_VERSION",
    "PhaseCArtifactError",
    "canonical_json_dumps",
    "load_router_bundle",
    "parse_group_registry",
    "parse_router_config",
    "sha256_bytes",
    "sha256_file",
]
