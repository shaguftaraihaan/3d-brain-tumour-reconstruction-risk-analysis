"""Compare stock DIPY SynthSeg vs this project's run_synthseg on one BraTS T1.

Diagnosis only. Does not write caches or change pipeline code.

Example:
    python scripts/debug_synthseg.py
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.anatomy import run_synthseg  # noqa: E402
from recon.scene import _write_canonical_t1  # noqa: E402

CASE = ROOT / "data" / "cases" / "BraTS20_Training_002"
CHECK_LABELS = (10, 11, 12, 13, 16, 17, 18, 7, 8)
CHECK_NAMES = {
    10: "thalamus-L",
    11: "caudate-L",
    12: "putamen-L",
    13: "pallidum-L",
    16: "brainstem",
    17: "hippocampus-L",
    18: "amygdala-L",
    7: "cerebellar-wm-L",
    8: "cerebellar-cortex-L",
}


def _labels_from_predict(result) -> np.ndarray:
    if isinstance(result, tuple):
        labels = result[0]
    else:
        labels = result
    return np.rint(np.squeeze(np.asarray(labels))).astype(np.int16)


def _report(title: str, labels: np.ndarray) -> None:
    ids = sorted(int(v) for v in np.unique(labels) if int(v) != 0)
    print(f"\n=== {title} ===", flush=True)
    print(f"distinct labels (excl. 0): {len(ids)}", flush=True)
    print(f"label ids: {ids}", flush=True)
    for label_id in CHECK_LABELS:
        count = int(np.count_nonzero(labels == label_id))
        print(f"  label {label_id:3d} ({CHECK_NAMES[label_id]}): {count} voxels", flush=True)


def main() -> int:
    if not CASE.is_dir():
        raise FileNotFoundError(f"Case folder not found: {CASE}")

    with tempfile.TemporaryDirectory() as tmp:
        t1_path = Path(tmp) / "t1n_ras.nii.gz"
        affine, shape = _write_canonical_t1(CASE, t1_path, anatomy_input="t1")
        import nibabel as nib

        image = nib.load(str(t1_path))
        t1 = np.asarray(image.get_fdata(dtype=np.float32))
        affine = np.asarray(image.affine, dtype=np.float64)
        print(f"case: {CASE}", flush=True)
        print(f"T1 shape={shape} dtype={t1.dtype} affine=\n{affine}", flush=True)

        from dipy.nn.torch.synthseg import SynthSeg

        print("\n--- Run A: stock DIPY SynthSeg (no monkeypatch) ---", flush=True)
        stock = SynthSeg(verbose=False, use_cuda=False)
        result_a = stock.predict(t1, affine)
        labels_a = _labels_from_predict(result_a)
        _report("Run A stock DIPY", labels_a)

        print("\n--- Run B: project run_synthseg on CPU ---", flush=True)
        labels_b, names_b, cortical_b = run_synthseg(t1, affine, torch.device("cpu"))
        print(f"run_synthseg names={len(names_b)} has_cortical={cortical_b}", flush=True)
        _report("Run B run_synthseg", labels_b)

        print("\n--- DIPY private predict source ---", flush=True)
        print(inspect.getsource(SynthSeg._SynthSeg__predict), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
