"""
Gen2 Tiling - Splitter, mask, and (later) merger nodes.

Splits an input image into a uniform grid of tiles with a fixed-thickness
overlap halo and emits a GEN2_TILE_LAYOUT describing the partition. A separate
mask node generates per-tile masks that select each tile's owned base region
for inpaint-style workflows.
"""

from .tile_splitter import (
    NODE_CLASS_MAPPINGS as SPLITTER_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as SPLITTER_NAMES,
)
from .tile_masks import (
    NODE_CLASS_MAPPINGS as MASK_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as MASK_NAMES,
)
from .tile_merger import (
    NODE_CLASS_MAPPINGS as MERGER_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as MERGER_NAMES,
)
from .seam_fix import (
    NODE_CLASS_MAPPINGS as SEAM_FIX_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as SEAM_FIX_NAMES,
)
from .seam_merger import (
    NODE_CLASS_MAPPINGS as SEAM_MERGER_NODES,
    NODE_DISPLAY_NAME_MAPPINGS as SEAM_MERGER_NAMES,
)

NODE_CLASS_MAPPINGS = {
    **SPLITTER_NODES,
    **MASK_NODES,
    **MERGER_NODES,
    **SEAM_FIX_NODES,
    **SEAM_MERGER_NODES,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **SPLITTER_NAMES,
    **MASK_NAMES,
    **MERGER_NAMES,
    **SEAM_FIX_NAMES,
    **SEAM_MERGER_NAMES,
}
