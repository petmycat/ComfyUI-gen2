"""Configuration contracts shared by the Gen2 Input and Output panels."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Literal

MAX_PARAMS = 32
SUPPORTED_TYPES = ("STRING", "INT", "FLOAT", "BOOLEAN", "IMAGE", "COMBO", "SEED")
NUMERIC_TYPES = ("INT", "FLOAT", "SEED")
INT_TYPES = ("INT", "SEED")
SEED_MIN = 0
COMFY_SEED_MAX = 0xFFFFFFFFFFFFFFFF
SEED_MAX = (2**53) - 1
SEED_STEP = 1
CONTROL_MODES = ("fixed", "randomize", "increment", "decrement")

PanelMode = Literal["input", "output"]


def _stable_legacy_id(index: int, name: str, ptype: str) -> str:
    token = hashlib.sha1(f"{index}:{name}:{ptype}".encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")[:24] or "param"
    return f"legacy-{slug}-{token}"


def _load_entries(config_raw: Any) -> list[Any]:
    if config_raw is None:
        return []
    if isinstance(config_raw, list):
        entries = config_raw
    else:
        text = str(config_raw).strip()
        if not text:
            return []
        try:
            entries = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Panel configuration must be valid JSON.") from exc
    if not isinstance(entries, list):
        raise ValueError("Panel configuration must be a JSON array.")
    if len(entries) > MAX_PARAMS:
        raise ValueError(f"Panel configuration supports at most {MAX_PARAMS} parameters.")
    return entries


def _normalise_identity(entry: dict[str, Any], index: int, seen_names: set[str], seen_ids: set[str]) -> tuple[str, str, str]:
    name = str(entry.get("name", "")).strip()
    if not name:
        raise ValueError(f"Parameter {index + 1} needs a name.")
    if name in seen_names:
        raise ValueError(f"Parameter name {name!r} is duplicated.")

    ptype = str(entry.get("type", "STRING")).strip().upper()
    if ptype not in SUPPORTED_TYPES:
        raise ValueError(f"Parameter {name!r} has unsupported type {ptype!r}.")

    param_id = str(entry.get("id", "")).strip() or _stable_legacy_id(index, name, ptype)
    if param_id in seen_ids:
        raise ValueError(f"Parameter id {param_id!r} is duplicated.")

    seen_names.add(name)
    seen_ids.add(param_id)
    return param_id, name, ptype


def _number(value: Any, *, field: str, name: str, integer: bool) -> int | float:
    if value is None or value == "":
        raise ValueError(f"Parameter {name!r} needs {field}.")
    if isinstance(value, bool):
        raise ValueError(f"Parameter {name!r} {field} must be numeric.")
    try:
        number = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Parameter {name!r} {field} must be numeric.") from exc
    if not integer and not math.isfinite(number):
        raise ValueError(f"Parameter {name!r} {field} must be finite.")
    if integer and isinstance(value, float) and not value.is_integer():
        raise ValueError(f"Parameter {name!r} {field} must be an integer.")
    if integer and isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"Parameter {name!r} {field} must be an integer.") from exc
        if not parsed.is_integer() or int(parsed) != number:
            raise ValueError(f"Parameter {name!r} {field} must be an integer.")
    return number


def parse_input_config(config_raw: Any) -> list[dict[str, Any]]:
    entries = _load_entries(config_raw)
    clean: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_ids: set[str] = set()

    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError(f"Parameter {index + 1} must be an object.")
        param_id, name, ptype = _normalise_identity(raw, index, seen_names, seen_ids)
        default = raw.get("default")
        if ptype == "SEED" and default is None:
            default = SEED_MIN
        if default is None or (isinstance(default, str) and not default.strip()):
            raise ValueError(f"Parameter {name!r} needs a default value.")

        entry: dict[str, Any] = {"id": param_id, "name": name, "type": ptype, "default": default}
        if ptype in NUMERIC_TYPES:
            integer = ptype in INT_TYPES
            entry["default"] = _number(default, field="default", name=name, integer=integer)
            entry["min"] = _number(
                raw.get("min", SEED_MIN if ptype == "SEED" else None),
                field="min", name=name, integer=integer,
            )
            entry["max"] = _number(
                raw.get("max", SEED_MAX if ptype == "SEED" else None),
                field="max", name=name, integer=integer,
            )
            entry["step"] = _number(
                raw.get("step", SEED_STEP if ptype == "SEED" else None),
                field="step", name=name, integer=integer,
            )
            if entry["min"] > entry["max"]:
                raise ValueError(f"Parameter {name!r} min must not exceed max.")
            if not entry["min"] <= entry["default"] <= entry["max"]:
                raise ValueError(f"Parameter {name!r} default must be between min and max.")
            if entry["step"] <= 0:
                raise ValueError(f"Parameter {name!r} step must be greater than 0.")
            if ptype == "SEED":
                if entry["min"] < SEED_MIN or entry["max"] > SEED_MAX:
                    raise ValueError(
                        f"Parameter {name!r} seed range must be within the exact frontend range "
                        f"{SEED_MIN}..{SEED_MAX} (ComfyUI supports up to {COMFY_SEED_MAX})."
                    )
                mode = str(raw.get("controlMode", "randomize")).strip().lower()
                if mode not in CONTROL_MODES:
                    raise ValueError(f"Parameter {name!r} has invalid after-run mode {mode!r}.")
                entry["controlMode"] = mode
        elif ptype == "BOOLEAN":
            if not isinstance(default, bool):
                raise ValueError(f"Parameter {name!r} default must be true or false.")
        elif ptype in ("STRING", "COMBO", "IMAGE"):
            if not isinstance(default, str) or not default.strip():
                raise ValueError(f"Parameter {name!r} default must be a non-empty string.")
            entry["default"] = default.strip() if ptype == "IMAGE" else default
        clean.append(entry)
    return clean


def parse_output_config(config_raw: Any) -> list[dict[str, str]]:
    entries = _load_entries(config_raw)
    clean: list[dict[str, str]] = []
    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError(f"Parameter {index + 1} must be an object.")
        param_id, name, ptype = _normalise_identity(raw, index, seen_names, seen_ids)
        clean.append({"id": param_id, "name": name, "type": ptype})
    return clean


def parse_config(config_raw: Any, mode: PanelMode = "input") -> list[dict[str, Any]]:
    """Backward-compatible entry point used by older integrations."""
    return parse_input_config(config_raw) if mode == "input" else parse_output_config(config_raw)


def validate_value(name: str, ptype: str, value: Any, param: dict[str, Any]) -> None:
    if ptype not in NUMERIC_TYPES or value is None:
        return
    lo = param.get("min")
    hi = param.get("max")
    if lo is not None and value < lo:
        raise ValueError(f"Gen2_InputPanel: parameter {name!r} = {value} is below min {lo}")
    if hi is not None and value > hi:
        raise ValueError(f"Gen2_InputPanel: parameter {name!r} = {value} is above max {hi}")


def schema_entries(params: list[dict[str, Any]], mode: PanelMode = "input") -> list[dict[str, Any]]:
    if mode == "output":
        return [{"id": p["id"], "name": p["name"], "type": p["type"]} for p in params]
    out: list[dict[str, Any]] = []
    for p in params:
        entry = {"id": p["id"], "name": p["name"], "type": p["type"], "default": p.get("default")}
        if p["type"] in NUMERIC_TYPES:
            entry.update(min=p.get("min"), max=p.get("max"), step=p.get("step"))
        if p["type"] == "SEED":
            entry["controlMode"] = p.get("controlMode", "randomize")
        out.append(entry)
    return out


def build_schema_json(params: list[dict[str, Any]], mode: PanelMode = "input") -> str:
    return json.dumps(schema_entries(params, mode), indent=2, ensure_ascii=False)
