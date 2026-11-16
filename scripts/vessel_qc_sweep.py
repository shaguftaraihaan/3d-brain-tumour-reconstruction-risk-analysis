"""Sweep sigma_low for vessel QC PNGs on one study file.

Example:
    python scripts/vessel_qc_sweep.py --study study_ff9c6c72e9d1 --vessel vessels_mra.nii.gz
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
from recon.vessels import SIGMA_HIGH, segment_vessels, vessel_stats, write_vessel_qc  # noqa: E402

SIGMA_LOW_SWEEP = (3.0, 2.5, 2.0, 1.5)
QC_DIR = ROOT / "reports" / "vessels_qc"


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
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    study_id = str(args.study).strip()
    vessel_name = Path(args.vessel).name
    path = Path(args.out_root) / study_id / "04_ready" / vessel_name
    if not path.is_file():
        print(f"Missing vessel volume: {path}", file=sys.stderr)
        return 1
    image = nib.load(str(path))
    volume = np.asarray(image.get_fdata(dtype=np.float32))
    affine = np.asarray(image.affine, dtype=np.float64)
    stem = vessel_name.replace(".nii.gz", "").replace(".nii", "")
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    print(f"study={study_id} vessel={vessel_name} sigma_high={args.sigma_high}", flush=True)
    for sigma_low in SIGMA_LOW_SWEEP:
        mask = segment_vessels(volume, affine, sigma_high=args.sigma_high, sigma_low=sigma_low)
        stats = vessel_stats(mask, affine)
        out = args.qc_dir / f"{study_id}_{stem}_{sigma_low:g}.png"
        write_vessel_qc(volume, mask, affine, out, study_id=study_id)
        print(
            f"sigma_low={sigma_low:g} volume_cm3={stats['volume_cm3']:.4f} "
            f"component_count={stats['component_count']} -> {out}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
