"""Diagnose why DIPY SynthSeg returns few labels: orientation / intensity / resample.

Does not modify recon/ or web/. Console output is also saved under
``reports/synthseg_diagnosis.txt`` (not meant to be committed).

Example:
    python scripts/debug_synthseg2.py
"""

from __future__ import annotations

import inspect
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.scene import _write_canonical_t1  # noqa: E402

CASE = ROOT / "data" / "cases" / "BraTS20_Training_002"
REAL_READY = ROOT / "outputs" / "real" / "study_ff9c6c72e9d1" / "04_ready"
REPORT = ROOT / "reports" / "synthseg_diagnosis.txt"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _labels_from_predict(result) -> np.ndarray:
    if isinstance(result, tuple):
        labels = result[0]
    else:
        labels = result
    return np.rint(np.squeeze(np.asarray(labels))).astype(np.int16)


def _label_stats(labels: np.ndarray) -> tuple[int, list[int]]:
    ids = sorted(int(v) for v in np.unique(labels) if int(v) != 0)
    return len(ids), ids


def _reorient(data: np.ndarray, affine: np.ndarray, axcodes: str):
    from nibabel.orientations import (
        apply_orientation,
        axcodes2ornt,
        inv_ornt_aff,
        io_orientation,
        ornt_transform,
    )

    current = io_orientation(np.asarray(affine, dtype=np.float64))
    target = axcodes2ornt(tuple(axcodes))
    transform = ornt_transform(current, target)
    new_data = apply_orientation(data, transform)
    new_affine = np.asarray(affine, dtype=np.float64) @ inv_ornt_aff(transform, data.shape)
    return np.asarray(new_data, dtype=np.float32), new_affine


def _rescale_percentile(data: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    nonzero = data[np.isfinite(data) & (data != 0)]
    if nonzero.size == 0:
        return data.astype(np.float32, copy=True)
    lo, hi = np.percentile(nonzero, (1.0, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(data, lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    return (scaled * (new_max - new_min) + new_min).astype(np.float32)


def _resample_1mm(data: np.ndarray, affine: np.ndarray):
    import nibabel as nib
    from nibabel.processing import resample_to_output

    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    out = resample_to_output(image, voxel_sizes=1.0, order=1)
    return np.asarray(out.get_fdata(dtype=np.float32)), np.asarray(out.affine, dtype=np.float64)


def _print_source_head(title: str, obj, n_lines: int = 60) -> None:
    print(f"\n--- {title} (first {n_lines} lines) ---", flush=True)
    try:
        src = inspect.getsource(obj)
    except OSError as exc:
        print(f"(could not get source: {exc})", flush=True)
        return
    lines = src.splitlines()
    for line in lines[:n_lines]:
        print(line, flush=True)
    if len(lines) > n_lines:
        print(f"... ({len(lines) - n_lines} more lines)", flush=True)


def _print_doc_and_reorient_lines(title: str, obj) -> None:
    print(f"\n--- {title} ---", flush=True)
    doc = inspect.getdoc(obj) or "(no docstring)"
    print("docstring:", flush=True)
    print(doc, flush=True)
    try:
        src = inspect.getsource(obj)
    except OSError as exc:
        print(f"(could not get source: {exc})", flush=True)
        return
    print("lines mentioning reorient / rescale / affine / orient / normalize / resample:", flush=True)
    for i, line in enumerate(src.splitlines(), start=1):
        low = line.lower()
        if any(
            key in low
            for key in (
                "reorient",
                "orient",
                "rescale",
                "affine",
                "normalize",
                "resample",
                "voxsize",
                "percentile",
                "clip",
            )
        ):
            print(f"  L{i}: {line.rstrip()}", flush=True)


def step1_inspect(SynthSeg) -> None:
    import dipy
    import torch

    print("=== Step 1: versions and signatures ===", flush=True)
    print(f"dipy.__version__ = {dipy.__version__}", flush=True)
    print(f"torch.__version__ = {torch.__version__}", flush=True)
    print(f"SynthSeg.__init__ signature: {inspect.signature(SynthSeg.__init__)}", flush=True)
    print(f"SynthSeg.predict signature: {inspect.signature(SynthSeg.predict)}", flush=True)
    print(f"SynthSeg public/relevant attrs: {[n for n in dir(SynthSeg) if 'prep' in n.lower() or 'predict' in n.lower() or 'norm' in n.lower() or 'resample' in n.lower() or 'conform' in n.lower()]}", flush=True)

    _print_source_head("SynthSeg._SynthSeg__predict", SynthSeg._SynthSeg__predict, 60)
    _print_doc_and_reorient_lines("SynthSeg._prepare_img", SynthSeg._prepare_img)

    from dipy.nn.utils import normalize, transform_img

    _print_doc_and_reorient_lines("dipy.nn.utils.transform_img (used inside predict)", transform_img)
    _print_doc_and_reorient_lines("dipy.nn.utils.normalize (used inside predict)", normalize)
    _print_source_head("SynthSeg.predict (first 60 lines; contains transform/normalize)", SynthSeg.predict, 60)


def _run_variant(model, name: str, data: np.ndarray, affine: np.ndarray, rows: list) -> dict:
    print(f"\n>>> {name}", flush=True)
    started = time.perf_counter()
    try:
        result = model.predict(data, affine)
        labels = _labels_from_predict(result)
        n_labels, ids = _label_stats(labels)
        runtime = time.perf_counter() - started
        row = {
            "name": name,
            "n_labels": n_labels,
            "ids": ids,
            "runtime_s": runtime,
            "error": None,
        }
        print(
            f"variant={name} n_nonzero_labels={n_labels} ids={ids} runtime_s={runtime:.1f}",
            flush=True,
        )
    except Exception as exc:
        runtime = time.perf_counter() - started
        row = {
            "name": name,
            "n_labels": -1,
            "ids": [],
            "runtime_s": runtime,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"variant={name} FAILED after {runtime:.1f}s: {row['error']}", flush=True)
        traceback.print_exc()
    rows.append(row)
    return row


def step2_variants(t1: np.ndarray, affine: np.ndarray, SynthSeg) -> list[dict]:
    print("\n=== Step 2: variants on BraTS T1 ===", flush=True)
    print(f"input shape={t1.shape} dtype={t1.dtype}", flush=True)
    from nibabel.orientations import aff2axcodes

    print(f"input axcodes={''.join(aff2axcodes(affine))}", flush=True)

    model = SynthSeg(verbose=False, use_cuda=False)
    rows: list[dict] = []

    _run_variant(model, "V0_as_is", t1, affine, rows)

    for codes, tag in (("LIA", "V1_LIA"), ("LPS", "V1b_LPS"), ("RAS", "V1c_RAS"), ("ASR", "V1d_ASR")):
        try:
            data_r, aff_r = _reorient(t1, affine, codes)
            print(f"{tag} reoriented to {codes}: shape={data_r.shape} axcodes={''.join(aff2axcodes(aff_r))}", flush=True)
            _run_variant(model, tag, data_r, aff_r, rows)
        except Exception as exc:
            print(f"{tag} reorient/predict failed: {exc}", flush=True)
            traceback.print_exc()
            rows.append(
                {
                    "name": tag,
                    "n_labels": -1,
                    "ids": [],
                    "runtime_s": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    for new_max, tag in ((255.0, "V2_0_255"), (1.0, "V2b_0_1")):
        try:
            scaled = _rescale_percentile(t1, 0.0, new_max)
            _run_variant(model, tag, scaled, affine, rows)
        except Exception as exc:
            print(f"{tag} failed: {exc}", flush=True)
            traceback.print_exc()
            rows.append(
                {
                    "name": tag,
                    "n_labels": -1,
                    "ids": [],
                    "runtime_s": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    try:
        data_rs, aff_rs = _resample_1mm(t1, affine)
        print(f"V3 resampled shape={data_rs.shape}", flush=True)
        _run_variant(model, "V3_resample_1mm", data_rs, aff_rs, rows)
    except Exception as exc:
        print(f"V3 failed: {exc}", flush=True)
        traceback.print_exc()
        rows.append(
            {
                "name": "V3_resample_1mm",
                "n_labels": -1,
                "ids": [],
                "runtime_s": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    orient_rows = [r for r in rows if r["name"].startswith("V1") and r["n_labels"] >= 0]
    intens_rows = [r for r in rows if r["name"].startswith("V2") and r["n_labels"] >= 0]
    best_orient = max(orient_rows, key=lambda r: r["n_labels"]) if orient_rows else None
    best_intens = max(intens_rows, key=lambda r: r["n_labels"]) if intens_rows else None
    print(
        f"best orientation so far: {None if best_orient is None else best_orient['name']} "
        f"({None if best_orient is None else best_orient['n_labels']} labels)",
        flush=True,
    )
    print(
        f"best intensity so far: {None if best_intens is None else best_intens['name']} "
        f"({None if best_intens is None else best_intens['n_labels']} labels)",
        flush=True,
    )

    if best_orient is not None and best_intens is not None:
        try:
            codes = {"V1_LIA": "LIA", "V1b_LPS": "LPS", "V1c_RAS": "RAS", "V1d_ASR": "ASR"}[
                best_orient["name"]
            ]
            data_r, aff_r = _reorient(t1, affine, codes)
            new_max = 255.0 if best_intens["name"] == "V2_0_255" else 1.0
            combined = _rescale_percentile(data_r, 0.0, new_max)
            tag = f"V4_{best_orient['name']}+{best_intens['name']}"
            _run_variant(model, tag, combined, aff_r, rows)
        except Exception as exc:
            print(f"V4 failed: {exc}", flush=True)
            traceback.print_exc()
            rows.append(
                {
                    "name": "V4_combine",
                    "n_labels": -1,
                    "ids": [],
                    "runtime_s": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    else:
        print("V4 skipped: missing successful orientation or intensity variant", flush=True)

    init_sig = inspect.signature(SynthSeg.__init__)
    flag_params = [
        name
        for name, param in init_sig.parameters.items()
        if name not in ("self", "verbose", "use_cuda")
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    print(f"V5 constructor flags beyond verbose/use_cuda: {flag_params or '(none)'}", flush=True)
    for flag in flag_params:
        try:
            kwargs = {flag: True, "verbose": False, "use_cuda": False}
            flagged = SynthSeg(**kwargs)
            _run_variant(flagged, f"V5_{flag}=True", t1, affine, rows)
        except Exception as exc:
            print(f"V5_{flag} failed: {exc}", flush=True)
            traceback.print_exc()
            rows.append(
                {
                    "name": f"V5_{flag}=True",
                    "n_labels": -1,
                    "ids": [],
                    "runtime_s": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    # Real-patient T1 (skull-stripped only available under 04_ready).
    real_t1 = REAL_READY / "t1.nii.gz"
    print("\n=== Real-patient T1 (study_ff9c6c72e9d1) ===", flush=True)
    if real_t1.is_file():
        import nibabel as nib

        print(
            "Only brain-stripped T1 found at "
            f"{real_t1} (preprocess.json skull_strip=hd-bet; no pre-strip volume on disk).",
            flush=True,
        )
        image = nib.load(str(real_t1))
        rt1 = np.asarray(image.get_fdata(dtype=np.float32))
        raff = np.asarray(image.affine, dtype=np.float64)
        _run_variant(model, "REAL_V0_as_is", rt1, raff, rows)
        ok = [r for r in rows if r["n_labels"] >= 0 and not r["name"].startswith("REAL_")]
        best = max(ok, key=lambda r: r["n_labels"]) if ok else None
        if best is not None and best["name"] != "V0_as_is":
            try:
                data_b, aff_b = rt1, raff
                name = best["name"]
                if name.startswith("V1"):
                    codes = {"V1_LIA": "LIA", "V1b_LPS": "LPS", "V1c_RAS": "RAS", "V1d_ASR": "ASR"}[name]
                    data_b, aff_b = _reorient(rt1, raff, codes)
                elif name.startswith("V2"):
                    data_b = _rescale_percentile(rt1, 0.0, 255.0 if name == "V2_0_255" else 1.0)
                elif name.startswith("V3"):
                    data_b, aff_b = _resample_1mm(rt1, raff)
                elif name.startswith("V4_"):
                    # Re-apply same orientation+intensity combo encoded in the name when possible.
                    parts = name.replace("V4_", "").split("+")
                    if len(parts) == 2:
                        o_name, i_name = parts
                        codes = {"V1_LIA": "LIA", "V1b_LPS": "LPS", "V1c_RAS": "RAS", "V1d_ASR": "ASR"}.get(
                            o_name
                        )
                        if codes:
                            data_b, aff_b = _reorient(rt1, raff, codes)
                        data_b = _rescale_percentile(
                            data_b, 0.0, 255.0 if i_name == "V2_0_255" else 1.0
                        )
                _run_variant(model, f"REAL_best_{best['name']}", data_b, aff_b, rows)
            except Exception as exc:
                print(f"REAL best variant failed: {exc}", flush=True)
                traceback.print_exc()
        else:
            print("REAL best variant skipped (no better than V0 or none succeeded)", flush=True)
    else:
        print(f"Missing real T1: {real_t1}", flush=True)

    return rows


def step3_table(rows: list[dict]) -> None:
    print("\n=== Step 3: final table (sorted by n_labels desc) ===", flush=True)
    ranked = sorted(rows, key=lambda r: r["n_labels"], reverse=True)
    print(f"{'variant':40s} {'n_labels':>8s} {'runtime_s':>10s}  notes", flush=True)
    for row in ranked:
        note = row["error"] or f"ids={row['ids']}"
        print(
            f"{row['name']:40s} {row['n_labels']:8d} {row['runtime_s']:10.1f}  {note}",
            flush=True,
        )
    winners = [r for r in ranked if r["n_labels"] >= 20]
    if winners:
        # Prefer the first among the original BraTS variants in run order that reached 20.
        order = [r["name"] for r in rows]
        first = min(winners, key=lambda r: order.index(r["name"]))
        print(
            f"\nFirst variant to reach >= 20 distinct non-zero labels: {first['name']} "
            f"({first['n_labels']} labels)",
            flush=True,
        )
    else:
        print("\nNone of the variants reached at least 20 distinct non-zero labels.", flush=True)


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report_handle = REPORT.open("w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, report_handle)
    sys.stderr = _Tee(old_err, report_handle)
    try:
        if not CASE.is_dir():
            raise FileNotFoundError(f"Case folder not found: {CASE}")

        from dipy.nn.torch.synthseg import SynthSeg

        step1_inspect(SynthSeg)

        with tempfile.TemporaryDirectory() as tmp:
            t1_path = Path(tmp) / "t1n_ras.nii.gz"
            _write_canonical_t1(CASE, t1_path, anatomy_input="t1")
            import nibabel as nib

            image = nib.load(str(t1_path))
            t1 = np.asarray(image.get_fdata(dtype=np.float32))
            affine = np.asarray(image.affine, dtype=np.float64)
            print(f"\ncase: {CASE}", flush=True)
            print(f"T1 shape={t1.shape} affine=\n{affine}", flush=True)
            rows = step2_variants(t1, affine, SynthSeg)
            step3_table(rows)

        print(f"\nWrote {REPORT}", flush=True)
        return 0
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        report_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
