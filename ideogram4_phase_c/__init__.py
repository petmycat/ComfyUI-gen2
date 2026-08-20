from __future__ import annotations

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
REGISTRATION_ERROR: str | None = None

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except (ImportError, AttributeError) as exc:
    REGISTRATION_ERROR = f"{type(exc).__name__}: {exc}"
    print(f"[Gen2] Ideogram4 Phase C V2 nodes unavailable: {REGISTRATION_ERROR}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "REGISTRATION_ERROR"]
