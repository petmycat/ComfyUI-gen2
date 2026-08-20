from __future__ import annotations

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
REGISTRATION_ERROR: str | None = None

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except (ImportError, AttributeError) as exc:
    REGISTRATION_ERROR = str(exc)
    print(f"[Gen2] Ideogram4 Phase C V2 nodes unavailable: {exc}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "REGISTRATION_ERROR"]
