from __future__ import annotations

import os

MODEL_CATEGORY = "gen2"
MODEL_DIRECTORY = ""
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
REGISTRATION_ERROR: str | None = None

try:
    import folder_paths

    if not hasattr(folder_paths, "models_dir") or not hasattr(folder_paths, "add_model_folder_path"):
        raise ImportError("incomplete folder_paths module")
    MODEL_DIRECTORY = os.path.join(folder_paths.models_dir, MODEL_CATEGORY)
    folder_paths.add_model_folder_path(MODEL_CATEGORY, MODEL_DIRECTORY, is_default=True)
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except (ImportError, AttributeError) as exc:
    REGISTRATION_ERROR = str(exc)
    print(f"[Gen2] Ideogram4 V9 Trigger nodes unavailable: {exc}")

__all__ = [
    "MODEL_CATEGORY", "MODEL_DIRECTORY", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
    "REGISTRATION_ERROR",
]
