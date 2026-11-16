"""Bright-vessel extraction, light meshing, and a three-view QC PNG.

Operates on an angiography or venography volume already in patient space.
Does not call the tumor model or write under ``outputs/`` on its own — callers
pass the QC image path explicitly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy import ndimage

from recon.mesh import mesh_from_mask
from recon.postprocess import CONN

logger = logging.getLogger(__name__)

# Seed threshold in robust MAD units above the in-head intensity centre.
SIGMA_HIGH = 3.5
# Growth threshold in robust MAD units; voxels above this join if 26-connected to a seed.
SIGMA_LOW = 1.5
# Drop 26-connected blobs smaller than this physical volume (mm³).
MIN_COMPONENT_MM3 = 20.0
# Keep at most this many components, largest first.
KEEP_LARGEST = 40
# Convert median absolute deviation to a normal-like scale factor.
MAD_TO_SIGMA = 1.4826
# Erode a provided head mask by this many millimetres before thresholding.
HEAD_MARGIN_MM = 8.0
# Narrow Gaussian width (mm) for the vessel-enhancing fine scale.
ENHANCE_SMOOTH_MM = 0.3
# Wide Gaussian width (mm) subtracted to suppress bright brain banding.
ENHANCE_BACKGROUND_MM = 4.0
# Reject a mask that covers more than this fraction of the analysis region.
MAX_VESSEL_FRACTION = 0.03
# Light Gaussian blur before marching cubes so thin tubes survive.
MESH_SIGMA = 0.3
# Do not decimate meshes that already have this many faces or fewer.
MESH_MAX_FACES = 150_000
# Taubin iterations for the vessel surface (paired with MESH_SIGMA).
MESH_TAUBIN_ITERS = 5


def _voxel_mm3(affine) -> float:
    return abs(float(np.linalg.det(np.asarray(affine, dtype=np.float64)[:3, :3])))


def _spacing_mm(affine) -> tuple[float, float, float]:
    from nibabel.affines import voxel_sizes

    return tuple(float(abs(v)) for v in voxel_sizes(np.asarray(affine, dtype=np.float64)))


def _mm_to_voxel_sigmas(affine, mm: float) -> tuple[float, float, float]:
    return tuple(float(mm) / max(spacing, 1e-6) for spacing in _spacing_mm(affine))


def _erode_mask_mm(mask, affine, margin_mm: float) -> np.ndarray:
    """Erode ``mask`` by ``margin_mm`` using the affine voxel sizes."""
    mask = np.asarray(mask, dtype=bool)
    if margin_mm <= 0 or not np.any(mask):
        return mask
    spacing = _spacing_mm(affine)
    dist = ndimage.distance_transform_edt(mask, sampling=spacing)
    return dist > float(margin_mm)


def _inside_mask(volume, head_mask, affine, head_margin_mm: float) -> np.ndarray:
    data = np.asarray(volume, dtype=np.float32)
    if head_mask is not None:
        eroded = _erode_mask_mm(head_mask, affine, head_margin_mm)
        return eroded & np.isfinite(data)
    return np.isfinite(data) & (data != 0)


def enhance_vessels(
    volume,
    affine,
    *,
    background_mm: float = ENHANCE_BACKGROUND_MM,
    smooth_mm: float = ENHANCE_SMOOTH_MM,
) -> np.ndarray:
    """Multi-scale vesselness-lite: signed fine smooth minus background smooth."""
    data = np.asarray(volume, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Vessel volume must be 3D, got {data.shape}")
    fine = ndimage.gaussian_filter(data, sigma=_mm_to_voxel_sigmas(affine, smooth_mm))
    coarse = ndimage.gaussian_filter(data, sigma=_mm_to_voxel_sigmas(affine, background_mm))
    return (fine - coarse).astype(np.float32)


def segment_vessels(
    volume,
    affine,
    *,
    head_mask=None,
    sigma_high: float = SIGMA_HIGH,
    sigma_low: float = SIGMA_LOW,
    min_component_mm3: float = MIN_COMPONENT_MM3,
    keep_largest: int = KEEP_LARGEST,
    head_margin_mm: float = HEAD_MARGIN_MM,
    enhance: bool = False,
) -> np.ndarray:
    """Boolean vessel mask via hysteresis thresholding and size filters.

    Seeds are voxels brighter than ``centre + sigma_high * spread``. The kept
    set is every voxel above ``centre + sigma_low * spread`` that is
    26-connected to a seed. When ``head_mask`` is set it is eroded by
    ``head_margin_mm`` first so scalp fat at the rim cannot grow into the mask.

    With ``enhance=True`` (opt-in), thresholds are computed on the signed
    :func:`enhance_vessels` response while the inside mask still comes from the
    original volume (eroded head mask, or nonzero voxels).
    """
    data = np.asarray(volume, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Vessel volume must be 3D, got {data.shape}")
    if float(sigma_low) > float(sigma_high):
        raise ValueError(f"sigma_low ({sigma_low}) must be <= sigma_high ({sigma_high})")
    inside = _inside_mask(data, head_mask, affine, head_margin_mm)
    out = np.zeros(data.shape, dtype=bool)
    if not np.any(inside):
        return out

    work = enhance_vessels(data, affine) if enhance else data
    samples = work[inside]
    centre = float(np.median(samples))
    mad = float(np.median(np.abs(samples - centre)))
    spread = MAD_TO_SIGMA * mad
    if not np.isfinite(spread) or spread == 0.0:
        raise ValueError(
            f"Vessel intensity spread is {spread!r}; cannot set hysteresis thresholds "
            "(median absolute deviation is zero or non-finite inside the analysis region)"
        )
    high = centre + float(sigma_high) * spread
    low = centre + float(sigma_low) * spread
    seeds = inside & (work > high)
    candidates = inside & (work > low)
    if not np.any(seeds):
        return out
    grown = ndimage.binary_propagation(seeds, structure=CONN, mask=candidates)
    if not np.any(grown):
        return out

    labeled, count = ndimage.label(grown, structure=CONN)
    if count == 0:
        return out

    voxel_mm3 = _voxel_mm3(affine)
    if voxel_mm3 <= 0:
        raise ValueError("Affine determinant must be non-zero for vessel volumes")
    min_voxels = float(min_component_mm3) / voxel_mm3
    sizes = ndimage.sum(grown, labeled, index=np.arange(1, count + 1))
    sizes = np.asarray(sizes, dtype=np.float64)
    order = np.argsort(sizes)[::-1]
    kept = []
    for index in order:
        if sizes[index] < min_voxels:
            continue
        kept.append(int(index) + 1)
        if len(kept) >= int(keep_largest):
            break
    if not kept:
        return out
    mask = np.isin(labeled, kept)
    region = int(np.count_nonzero(inside))
    fraction = float(np.count_nonzero(mask)) / float(region) if region else 0.0
    if fraction > MAX_VESSEL_FRACTION:
        logger.warning(
            "Vessel mask covers %.4f of the analysis region (limit %.4f); returning empty mask",
            fraction,
            MAX_VESSEL_FRACTION,
        )
        return out
    return mask


def vessel_stats(mask, affine) -> dict:
    """Physical volume and component summary for a vessel mask."""
    mask = np.asarray(mask, dtype=bool)
    labeled, count = ndimage.label(mask, structure=CONN) if np.any(mask) else (mask.astype(np.int32), 0)
    total = int(np.count_nonzero(mask))
    volume_cm3 = total * _voxel_mm3(affine) / 1000.0
    if total == 0 or count == 0:
        return {
            "volume_cm3": float(volume_cm3),
            "component_count": 0,
            "largest_component_fraction": 0.0,
        }
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, count + 1))
    largest = float(np.max(sizes))
    return {
        "volume_cm3": float(volume_cm3),
        "component_count": int(count),
        "largest_component_fraction": float(largest / total),
    }


def mesh_vessels(mask, affine):
    """Light-smoothed vessel surface, or ``None`` when the mask is empty."""
    return mesh_from_mask(
        mask,
        affine,
        sigma=MESH_SIGMA,
        taubin_iters=MESH_TAUBIN_ITERS,
        decimate_to=MESH_MAX_FACES,
    )


def _mip(volume, axis: int) -> np.ndarray:
    data = np.asarray(volume, dtype=np.float32)
    finite = np.where(np.isfinite(data), data, np.nan)
    with np.errstate(all="ignore"):
        projected = np.nanmax(finite, axis=axis)
    return np.nan_to_num(projected, nan=0.0)


def _outline(mask2d: np.ndarray) -> np.ndarray:
    mask2d = np.asarray(mask2d, dtype=bool)
    if not np.any(mask2d):
        return np.zeros_like(mask2d, dtype=bool)
    eroded = ndimage.binary_erosion(mask2d, iterations=1)
    return mask2d & ~eroded


def _window_uint8(plane: np.ndarray) -> np.ndarray:
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros(plane.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def write_vessel_qc(volume, mask, affine, out_png, study_id: str) -> Path:
    """Three MIP panels (axial, coronal, sagittal) with coloured vessel outlines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    del affine  # geometry is not needed for intensity MIPs
    views = (
        ("Axial", 2),
        ("Coronal", 1),
        ("Sagittal", 0),
    )
    mask = np.asarray(mask, dtype=bool)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    fig.suptitle(str(study_id), fontsize=11)
    for ax, (title, axis) in zip(axes, views):
        gray = _window_uint8(_mip(volume, axis))
        rgb = np.stack([gray, gray, gray], axis=-1)
        outline = _outline(_mip(mask.astype(np.float32), axis) > 0)
        # Display with origin lower so superior/anterior read upward when present.
        panel = np.transpose(rgb, (1, 0, 2))
        edge = np.transpose(outline, (1, 0))
        overlay = panel.astype(np.float32)
        overlay[edge] = (220.0, 60.0, 60.0)
        ax.imshow(np.clip(overlay, 0, 255).astype(np.uint8), origin="lower")
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    dest = Path(out_png)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return dest
