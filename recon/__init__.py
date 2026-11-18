"""BraTS reconstruction package.

Heavy ML imports (torch / MONAI) stay lazy so the demo viewer can
``import recon.io`` / ``import recon.slices`` without loading them.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CONN",
    "PostConfig",
    "REGION_COLORS",
    "brain_mask_from_image",
    "build_scene",
    "check_mesh_volume",
    "compute_tract_proximity",
    "load_case",
    "mesh_from_mask",
    "postprocess",
    "predict_probs",
    "reconstruct_tumor",
    "remove_small_components",
    "warp_atlas_tracts",
]

_LAZY: dict[str, tuple[str, str]] = {
    "CONN": ("recon.postprocess", "CONN"),
    "PostConfig": ("recon.postprocess", "PostConfig"),
    "REGION_COLORS": ("recon.scene", "REGION_COLORS"),
    "brain_mask_from_image": ("recon.postprocess", "brain_mask_from_image"),
    "build_scene": ("recon.scene", "build_scene"),
    "check_mesh_volume": ("recon.evaluate", "check_mesh_volume"),
    "compute_tract_proximity": ("recon.tracts", "compute_tract_proximity"),
    "load_case": ("recon.io", "load_case"),
    "mesh_from_mask": ("recon.mesh", "mesh_from_mask"),
    "postprocess": ("recon.postprocess", "postprocess"),
    "predict_probs": ("recon.infer", "predict_probs"),
    "reconstruct_tumor": ("recon.scene", "reconstruct_tumor"),
    "remove_small_components": ("recon.postprocess", "remove_small_components"),
    "warp_atlas_tracts": ("recon.tracts", "warp_atlas_tracts"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
