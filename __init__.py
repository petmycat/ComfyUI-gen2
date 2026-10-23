"""Register maintained Gen2 nodes and the experimental Flux.2 Klein fix."""

from importlib import import_module


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Load each feature independently so an unavailable dependency does not hide
# unrelated nodes. The ignored legacy archive is intentionally not registered.
for module_name, label in (
    ("gen2.api_panel_nodes", "API panels"),
    ("gen2.misc_nodes", "Misc utilities"),
    ("gen2.lanpaint_nodes", "LanPaint"),
    ("experimental.f2k", "Experimental Flux.2 Klein fix"),
):
    try:
        module = import_module(f".{module_name}", __name__)
    except ImportError as exc:
        print(f"[Gen2] {label} nodes not available: {exc}")
    else:
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

# ComfyUI serves this directory at the pack's existing extension URL.
WEB_DIRECTORY = "./gen2/api_panel_nodes/web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
