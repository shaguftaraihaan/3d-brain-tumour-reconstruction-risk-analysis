"""Sweep sigma_low for vessel QC PNGs on one study file.

Example:
    python scripts/vessel_qc_sweep.py --study study_ff9c6c72e9d1 --vessel vessels_mra.nii.gz
    python scripts/vessel_qc_sweep.py --study study_ff9c6c72e9d1 --vessel vessels_mra.nii.gz --enhance
    python scripts/vessel_qc_sweep.py --study study_cabf8c832a83 --vessel vessels_ven.nii.gz \\
        --brain-mask --sigma-high 6 --sigma-low 5,4,3,2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.ingest_real import DEFAULT_OUTPUT  # noqa: E402
from recon.vessels import (  # noqa: E402
    ARTERY_DILATE_MM,
    SIGMA_HIGH,
    VEIN_DILATE_MM,
    brain_region_mask,
    segment_vessels,
    vessel_stats,
    write_vessel_qc,
)

SIGMA_LOW_SWEEP = (3.0, 2.5, 2.0, 1.5)
QC_DIR = ROOT / "reports" / "vessels_qc"


def _parse_sigma_low(text: str | None) -> tuple[float, ...]:
    if text is None or not str(text).strip():
        return SIGMA_LOW_SWEEP
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise argparse.ArgumentTypeError("--sigma-low must list at least one number")
    return tuple(values)


def _dilate_for_vessel(vessel_name: str) -> float:
    name = vessel_name.lower()
    if "ven" in name:
        return VEIN_DILATE_MM
    return ARTERY_DILATE_MM


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Write vessel QC PNGs across sigma_low values")
    parser.add_argument("--study", required=True, help="Hashed study id (study_xxxxxxxxxxxx)")
    parser.add_argument(
        "--vessel",
        required=True,
        help="Vessel NIfTI file name under 04_ready (e.g. vessels_mra.nii.gz)",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT, help="Real-study output root")
    parser.add_argument("--qc-dir", type=Path, default=QC_DIR, help="Directory for QC PNGs")
    parser.add_argument(
        "--sigma-high",
        type=float,
        default=SIGMA_HIGH,
        help="Fixed hysteresis seed threshold (default 3.5)",
    )
    parser.add_argument(
        "--sigma-low",
        type=str,
        default=None,
        help="Comma-separated growth thresholds to sweep (default 3,2.5,2,1.5)",
    )
    parser.add_argument(
        "--brain-mask",
        action="store_true",
        help="Restrict analysis to dilated 04_ready/brain_mask.nii.gz on the vessel grid",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Run enhance_vessels before thresholding; QC filenames include _enh",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    study_id = str(args.study).strip()
    vessel_name = Path(args.vessel).name
    ready = Path(args.out_root) / study_id / "04_ready"
    path = ready / vessel_name
    if not path.is_file():
        print(f"Missing vessel volume: {path}", file=sys.stderr)
        return 1
    try:
        sigma_lows = _parse_sigma_low(args.sigma_low)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        print(f"Invalid --sigma-low: {exc}", file=sys.stderr)
        return 1

    image = nib.load(str(path))
    volume = np.asarray(image.get_fdata(dtype=np.float32))
    affine = np.asarray(image.affine, dtype=np.float64)

    region = None
    dilate_mm = None
    if args.brain_mask:
        mask_path = ready / "brain_mask.nii.gz"
        if not mask_path.is_file():
            print(f"Missing brain mask: {mask_path}", file=sys.stderr)
            return 1
        brain_image = nib.load(str(mask_path))
        brain = np.asarray(brain_image.get_fdata(dtype=np.float32)) > 0
        brain_affine = np.asarray(brain_image.affine, dtype=np.float64)
        dilate_mm = _dilate_for_vessel(vessel_name)
        region = brain_region_mask(brain, brain_affine, volume.shape, affine, dilate_mm)

    stem = vessel_name.replace(".nii.gz", "").replace(".nii", "")
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    tag = "_enh" if args.enhance else ""
    region_tag = "_br" if region is not None else ""
    print(
        f"study={study_id} vessel={vessel_name} sigma_high={args.sigma_high} "
        f"enhance={args.enhance} brain_mask={region is not None}"
        + (f" dilate_mm={dilate_mm:g}" if dilate_mm is not None else ""),
        flush=True,
    )
    for sigma_low in sigma_lows:
        if region is not None:
            mask = segment_vessels(
                volume,
                affine,
                head_mask=region,
                head_margin_mm=0.0,
                sigma_high=args.sigma_high,
                sigma_low=sigma_low,
                enhance=args.enhance,
            )
            stats = vessel_stats(mask, affine, region=region)
            denom = int(np.count_nonzero(region))
        else:
            mask = segment_vessels(
                volume,
                affine,
                sigma_high=args.sigma_high,
                sigma_low=sigma_low,
                enhance=args.enhance,
            )
            stats = vessel_stats(mask, affine)
            denom = int(np.count_nonzero(np.isfinite(volume) & (volume != 0)))
        fraction_pct = (
            100.0 * float(np.count_nonzero(mask)) / float(denom) if denom else 0.0
        )
        out = args.qc_dir / f"{study_id}_{stem}{tag}{region_tag}_{sigma_low:g}.png"
        write_vessel_qc(volume, mask, affine, out, study_id=study_id)
        print(
            f"sigma_low={sigma_low:g} volume_cm3={stats['volume_cm3']:.4f} "
            f"component_count={stats['component_count']} "
            f"fraction_pct={fraction_pct:.3f} -> {out}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
