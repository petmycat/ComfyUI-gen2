"""
Shared helpers for Gen2 Input/Output Panel nodes.

The parameter config is a JSON list of dicts. Each entry:
  {
    "name": str,            # parameter name (output slot label + API-export key)
    "type": "STRING"|"INT"|"FLOAT"|"BOOLEAN"|"IMAGE",
    "default": Any|None,    # default value; null = no default
    "min": float|None,      # INT/FLOAT only: minimum (inclusive)
    "max": float|None,      # INT/FLOAT only: maximum (inclusive)
    "step": float|None      # INT/FLOAT only: step for UI snapping + docs
  }

Only name and type are required; the rest are optional and only meaningful for
the types noted above.
"""

from __future__ import annotations

import json
from typing import Any

MAX_PARAMS = 32
SUPPORTED_TYPES = ("STRING", "INT", "FLOAT", "BOOLEAN", "IMAGE")
NUMERIC_TYPES = ("INT", "FLOAT")


def parse_config(config_raw: Any) -> list[dict]:
    """Parse a _config widget value (JSON string or list) into clean param dicts.

    Robust to None / empty / malformed input (returns []). Normalizes types,
    fills missing optional keys with None, and coerces min/max/step to floats
    for numeric params.
    """
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
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(entries, list):
        return []

    clean = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        ptype = str(e.get("type", "STRING")).strip().upper()
        if not name:
            continue
        if ptype not in SUPPORTED_TYPES:
            ptype = "STRING"

        entry: dict[str, Any] = {"name": name, "type": ptype, "default": e.get("default")}

        if ptype in NUMERIC_TYPES:
            for k in ("min", "max", "step"):
                v = e.get(k)
                if v is not None and v != "":
                    try:
                        entry[k] = float(v) if ptype == "FLOAT" else int(v)
                    except (TypeError, ValueError):
                        entry[k] = None
                else:
                    entry[k] = None
            # control_after_generate: INT-only, one of fixed/randomize/increment/decrement
            if ptype == "INT":
                cm = str(e.get("controlMode", "fixed")).strip().lower()
                if cm in ("fixed", "randomize", "increment", "decrement"):
                    entry["controlMode"] = cm
                else:
                    entry["controlMode"] = "fixed"
        clean.append(entry)
    return clean


def validate_value(name: str, ptype: str, value: Any, param: dict) -> None:
    """Raise ValueError if value is out of the declared range for INT/FLOAT.

    Called from InputPanel.execute() so an out-of-range value interrupts the
    workflow with a clear error message (instead of silently clamping).
    """
    if ptype not in NUMERIC_TYPES or value is None:
        return
    lo = param.get("min")
    hi = param.get("max")
    if lo is not None and value < lo:
        raise ValueError(
            f"Gen2_InputPanel: parameter {name!r} = {value} is below min {lo}"
        )
    if hi is not None and value > hi:
        raise ValueError(
            f"Gen2_InputPanel: parameter {name!r} = {value} is above max {hi}"
        )


def build_schema_json(params: list[dict]) -> str:
    """Build the JSON schema string describing all parameters.

    Output shape (pretty-printed for easy copy/paste):
      [
        {"name": "seed", "type": "INT", "default": 0, "min": 0, "max": 999, "step": 1},
        {"name": "prompt", "type": "STRING", "default": null},
        ...
      ]
    """
    out = []
    for p in params:
        entry = {"name": p["name"], "type": p["type"], "default": p.get("default")}
        if p["type"] in NUMERIC_TYPES:
            entry["min"] = p.get("min")
            entry["max"] = p.get("max")
            entry["step"] = p.get("step")
        if p["type"] == "INT" and p.get("controlMode") and p["controlMode"] != "fixed":
            entry["controlMode"] = p["controlMode"]
        out.append(entry)
    return json.dumps(out, indent=2, ensure_ascii=False)
