"""Synthetic bright-tube tests for vessel segmentation, stats, mesh, and QC."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.postprocess import CONN
from recon.vessels import (
    ENHANCE_BACKGROUND_MM,
    ENHANCE_SMOOTH_MM,
    HEAD_MARGIN_MM,
    KEEP_LARGEST,
    MAD_TO_SIGMA,
    MAX_VESSEL_FRACTION,
    MESH_MAX_FACES,
    MESH_SIGMA,
    MIN_COMPONENT_MM3,
    SIGMA_HIGH,
    SIGMA_LOW,
    enhance_vessels,
    mesh_vessels,
    segment_vessels,
    vessel_stats,
    write_vessel_qc,
)


def _curved_tube(shape=(96, 96, 96), radius: int = 1) -> np.ndarray:
    """Centreline arc dilated to about three voxels across (diameter 2*radius+1)."""
    mask = np.zeros(shape, dtype=bool)
    t = np.linspace(0.0, 1.6 * np.pi, 500)
    cx, cy, cz = [n // 2 for n in shape]
    xs = np.rint(cx + 22.0 * np.cos(t)).astype(int)
    ys = np.rint(cy + 18.0 * np.sin(t)).astype(int)
    zs = np.rint(cz - 28.0 + 56.0 * (t / t.max())).astype(int)
    for x, y, z in zip(xs, ys, zs):
        if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
            mask[x, y, z] = True
    struct = ndimage.generate_binary_structure(3, 1)
    if radius > 0:
        mask = ndimage.binary_dilation(mask, structure=struct, iterations=int(radius))
    return mask


def _attach_dim_branch(tube: np.ndarray, length: int = 18) -> np.ndarray:
    """One-voxel-wide branch starting on the tube and stepping +x."""
    branch = np.zeros(tube.shape, dtype=bool)
    coords = np.argwhere(tube)
    # Pick a mid-tube voxel with room to grow in +x.
    order = np.argsort(coords[:, 2])
    start = None
    for index in order[len(order) // 3 :]:
        x, y, z = (int(v) for v in coords[index])
        if x + length < tube.shape[0] and not tube[x + 1, y, z]:
            start = (x, y, z)
            break
    if start is None:
        raise RuntimeError("Could not place a dim side branch on the tube")
    x, y, z = start
    for step in range(1, length + 1):
        branch[x + step, y, z] = True
    return branch


def test_module_constants_are_named():
    assert SIGMA_HIGH == 3.5
    assert SIGMA_LOW == 1.5
    assert MIN_COMPONENT_MM3 == 20.0
    assert KEEP_LARGEST == 40
    assert MAD_TO_SIGMA == 1.4826
    assert HEAD_MARGIN_MM == 8.0
    assert ENHANCE_SMOOTH_MM == 0.3
    assert ENHANCE_BACKGROUND_MM == 4.0
    assert MAX_VESSEL_FRACTION == 0.03
    assert MESH_SIGMA == 0.3
    assert MESH_MAX_FACES == 150_000


def test_segment_vessels_recovers_bright_tube_and_drops_small_blob(tmp_path):
    shape = (96, 96, 96)
    affine = np.eye(4, dtype=np.float64)
    rng = np.random.default_rng(0)
    # Quiet background for a stable MAD; clamp non-structure noise ≤ 0 so it cannot bridge.
    volume = rng.normal(0.0, 0.25, size=shape).astype(np.float32)
    tube = _curved_tube(shape, radius=1)
    branch = _attach_dim_branch(tube)
    disconnected = np.zeros(shape, dtype=bool)
    disconnected[12, 70:85, 70] = True
    speck = np.zeros(shape, dtype=bool)
    speck[8:10, 8:10, 8:10] = True
    structure = tube | branch | disconnected | speck
    volume[~structure] = np.minimum(volume[~structure], 0.0)
    volume[tube] = 12.0
    # Dim side branch: between low and high thresholds for this quiet background.
    volume[branch] = 1.0
    # Same-intensity disconnected blob must not grow without a seed.
    volume[disconnected] = 1.0
    # Bright speck well under 20 mm³ at 1 mm isotropic (~8 voxels).
    volume[speck] = 20.0

    mask = segment_vessels(volume, affine, enhance=False)
    overlap = float(np.count_nonzero(mask & tube)) / float(np.count_nonzero(tube))
    assert overlap >= 0.85
    assert float(np.count_nonzero(mask & branch)) / float(np.count_nonzero(branch)) >= 0.85
    assert np.count_nonzero(mask & disconnected) == 0

    expected = tube | branch
    dilated = ndimage.binary_dilation(expected, structure=CONN, iterations=1)
    outside = float(np.count_nonzero(mask & ~dilated)) / max(float(np.count_nonzero(mask)), 1.0)
    assert outside < 0.02

    assert np.count_nonzero(mask & speck) == 0

    stats = vessel_stats(mask, affine)
    assert stats["component_count"] >= 1
    assert stats["volume_cm3"] > 0
    assert 0.0 < stats["largest_component_fraction"] <= 1.0

    mesh = mesh_vessels(mask, affine)
    assert mesh is not None
    assert len(mesh.faces) > 0

    qc = tmp_path / "vessel_qc.png"
    write_vessel_qc(volume, mask, affine, qc, study_id="study_abcdef123456")
    assert qc.is_file()
    assert qc.stat().st_size > 0


def test_all_noise_gives_empty_mask_and_none_mesh():
    rng = np.random.default_rng(1)
    # Bounded noise: robust high threshold sits above the intensity range, so no seeds.
    volume = rng.uniform(-1.0, 1.0, size=(48, 48, 48)).astype(np.float32)
    affine = np.eye(4, dtype=np.float64)
    mask = segment_vessels(volume, affine)
    assert not np.any(mask)
    assert vessel_stats(mask, affine)["component_count"] == 0
    assert mesh_vessels(mask, affine) is None


def test_anisotropic_affine_sets_component_volume_threshold():
    shape = (40, 40, 40)
    affine = np.diag([2.0, 1.0, 1.0, 1.0]).astype(np.float64)
    rng = np.random.default_rng(2)
    head = np.zeros(shape, dtype=bool)
    head[5:35, 5:35, 5:35] = True
    # Tiny continuous jitter so MAD > 0 without bridging the two bright blobs.
    volume = np.zeros(shape, dtype=np.float32)
    volume[head] = 1.0 + rng.uniform(-0.02, 0.02, size=int(np.count_nonzero(head))).astype(
        np.float32
    )
    # 8 voxels × 2 mm³ = 16 mm³ < 20 mm³ → dropped.
    volume[10:12, 10:12, 10:12] = 20.0
    # 16 voxels × 2 mm³ = 32 mm³ ≥ 20 mm³ → kept.
    volume[20:24, 20:22, 20:22] = 20.0
    mask = segment_vessels(
        volume, affine, head_mask=head, head_margin_mm=0.0, enhance=False
    )
    assert np.count_nonzero(mask[10:12, 10:12, 10:12]) == 0
    assert np.count_nonzero(mask[20:24, 20:22, 20:22]) == 16
    stats = vessel_stats(mask, affine)
    assert abs(stats["volume_cm3"] - (16 * 2.0) / 1000.0) < 1e-9


def test_enhance_recovers_dim_tube_on_bright_ramp():
    shape = (64, 64, 64)
    affine = np.eye(4, dtype=np.float64)
    # Strong smooth ramp; bright end is 10.
    ramp = np.linspace(0.0, 10.0, shape[0], dtype=np.float32)
    volume = np.broadcast_to(ramp[:, None, None], shape).copy()
    # Thin tube in the darker half: locally brighter than the ramp, below the bright end.
    # Stay several coarse-kernel widths inside the FOV so signed DoG is not rim-dominated.
    tube = np.zeros(shape, dtype=bool)
    x0, y0 = 24, 32
    local = float(volume[x0, y0, 0])
    assert local < 10.0
    for z in range(16, 48):
        tube[x0, y0, z] = True
    volume[tube] = local + 1.5
    assert float(volume[tube].max()) < 10.0

    enhanced = enhance_vessels(volume, affine)
    assert enhanced.shape == volume.shape
    # Signed DoG: background / dark side of edges go negative.
    assert float(enhanced.min()) < 0.0
    assert float(enhanced[tube].mean()) > float(np.median(enhanced))

    head = np.zeros(shape, dtype=bool)
    head[16:48, 16:48, 16:48] = True
    with_enh = segment_vessels(
        volume,
        affine,
        enhance=True,
        sigma_low=1.5,
        head_mask=head,
        head_margin_mm=0.0,
    )
    without = segment_vessels(
        volume,
        affine,
        enhance=False,
        sigma_low=1.5,
        head_mask=head,
        head_margin_mm=0.0,
    )
    with_frac = float(np.count_nonzero(with_enh & tube)) / float(np.count_nonzero(tube))
    without_frac = float(np.count_nonzero(without & tube)) / float(np.count_nonzero(tube))
    assert with_frac >= 0.80
    assert without_frac < 0.30


def test_constant_background_does_not_fill_fov():
    shape = (32, 32, 32)
    affine = np.eye(4, dtype=np.float64)
    volume = np.full(shape, 5.0, dtype=np.float32)
    try:
        mask = segment_vessels(volume, affine, enhance=True)
    except ValueError as exc:
        assert "spread" in str(exc).lower()
        return
    assert np.count_nonzero(mask) == 0


def test_mask_above_fraction_limit_returns_empty(caplog):
    import logging

    shape = (40, 40, 40)
    affine = np.eye(4, dtype=np.float64)
    rng = np.random.default_rng(3)
    volume = rng.normal(1.0, 0.1, size=shape).astype(np.float32)
    # Large bright block (~12.5% of the FOV) that would pass hysteresis alone.
    volume[5:25, 5:25, 5:25] = 20.0
    with caplog.at_level(logging.WARNING, logger="recon.vessels"):
        mask = segment_vessels(
            volume,
            affine,
            enhance=False,
            head_margin_mm=0.0,
            min_component_mm3=1.0,
        )
    assert np.count_nonzero(mask) == 0
    assert any("returning empty mask" in r.message for r in caplog.records)
    # Sanity: the raw bright block really exceeds the named limit.
    inside = np.isfinite(volume) & (volume != 0)
    raw_frac = float(np.count_nonzero(volume >= 20.0)) / float(np.count_nonzero(inside))
    assert raw_frac > MAX_VESSEL_FRACTION
