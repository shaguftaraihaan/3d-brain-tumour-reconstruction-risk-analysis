#!/usr/bin/env python3
"""Export a backend-free static demo of the case explorer.

Writes ``dist/static_demo/`` with the viewer assets plus pre-baked GLB, JSON,
report HTML, and FLAIR / overlay PNGs for a short list of public BraTS cases.

Example:
    python scripts/export_static_demo.py \\
        --cases BraTS20_Training_002,BraTS20_Training_004,BraTS20_Training_022 \\
        --stride 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CASES = [
    "BraTS20_Training_002",
    "BraTS20_Training_004",
    "BraTS20_Training_022",
]
TARGET_BYTES = 100 * 1024 * 1024


def _parse_cases(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_CASES)
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _fmt_mb(num: int) -> str:
    return f"{num / (1024 * 1024):.1f} MB"


def _rewrite_static_html(src: Path, dest: Path) -> None:
    text = src.read_text(encoding="utf-8")
    text = text.replace('href="/static/', 'href="./')
    text = text.replace('src="/static/', 'src="./')
    text = text.replace('"/static/vendor/three/', '"./vendor/three/')
    text = text.replace(
        "<head>",
        "<head>\n"
        '  <meta name="brats-static-demo" content="1">\n'
        "  <script>window.__STATIC_DEMO__ = true;</script>",
        1,
    )
    footer = (
        '\n  <footer id="static-demo-footer" class="static-demo-footer">'
        "Research prototype, not for diagnosis. "
        "Demo cases from the public BraTS dataset."
        "</footer>\n"
    )
    if "static-demo-footer" not in text:
        text = text.replace("</body>", footer + "</body>")
    dest.write_text(text, encoding="utf-8")


def _copy_viewer_assets(dest_root: Path) -> None:
    static = ROOT / "web" / "static"
    # Core viewer files (relative paths after HTML rewrite).
    for name in ("viewer.js", "viewer.css", "explorer-ui.js", "hero.js", "landing.css", "landing.html", "favicon.png"):
        src = static / name
        if src.is_file():
            shutil.copy2(src, dest_root / name)
    img = static / "img"
    if img.is_dir():
        shutil.copytree(img, dest_root / "img", dirs_exist_ok=True)
    vendor = static / "vendor"
    if vendor.is_dir():
        shutil.copytree(vendor, dest_root / "vendor", dirs_exist_ok=True)
    _rewrite_static_html(static / "index.html", dest_root / "index.html")
    # Optional landing page with relative links into the viewer.
    landing = dest_root / "landing.html"
    if landing.is_file():
        text = landing.read_text(encoding="utf-8")
        text = text.replace('href="/static/', 'href="./').replace('src="/static/', 'src="./')
        text = text.replace('href="/viewer', 'href="./index.html')
        landing.write_text(text, encoding="utf-8")


def _save_png(path: Path, png_bytes: bytes, *, colors: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if colors is None:
        path.write_bytes(png_bytes)
        return
    image = Image.open(io_bytes(png_bytes))
    if image.mode == "RGBA":
        # Keep alpha; quantize RGB only via adaptive palette when opaque enough.
        alpha = image.split()[-1]
        quantized = image.convert("RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        out = quantized.convert("RGBA")
        out.putalpha(alpha)
        out.save(path, format="PNG", optimize=True, compress_level=9)
    else:
        image.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).save(
            path, format="PNG", optimize=True, compress_level=9
        )


def io_bytes(data: bytes):
    import io

    return io.BytesIO(data)


def _indices(length: int, stride: int) -> list[int]:
    if length <= 0:
        return []
    stride = max(1, int(stride))
    values = list(range(0, length, stride))
    if values[-1] != length - 1:
        values.append(length - 1)
    return values


def _export_case(app, case_id: str, dest: Path, *, stride: int, colors: int | None) -> dict:
    from recon.slices import ORIENT_LETTERS, brain_window, render_overlay_png, render_slice_png, slice_extent_mm
    from nibabel.affines import voxel_sizes
    from web import app as app_mod

    out_dir = Path(app.config["OUTPUTS"]) / case_id
    case_dest = dest / "cases" / case_id
    case_dest.mkdir(parents=True, exist_ok=True)
    slices_dest = case_dest / "slices"
    slices_dest.mkdir(parents=True, exist_ok=True)

    for name in ("scene.glb", "manifest.json"):
        src = out_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing {src}")
        shutil.copy2(src, case_dest / name)

    client = app.test_client()
    location = client.get(f"/api/cases/{case_id}/location.json")
    if location.status_code != 200:
        raise RuntimeError(f"location.json failed for {case_id}: {location.status_code}")
    (case_dest / "location.json").write_bytes(location.data)

    uncertainty = client.get(f"/api/cases/{case_id}/uncertainty")
    if uncertainty.status_code != 200:
        raise RuntimeError(f"uncertainty failed for {case_id}: {uncertainty.status_code}")
    unc = uncertainty.get_json()
    # Static demo never ships probability maps.
    unc["has_prob"] = False
    unc["reasons"] = dict(unc.get("reasons") or {})
    unc["reasons"]["prob"] = "Probability map needs the live model server."
    (case_dest / "uncertainty.json").write_text(json.dumps(unc, indent=2), encoding="utf-8")

    report = client.get(f"/api/cases/{case_id}/report.html")
    if report.status_code != 200:
        raise RuntimeError(f"report failed for {case_id}: {report.status_code}")
    (case_dest / "report.html").write_bytes(report.data)

    volume, affine = app_mod._cached_volume(app, case_id, "flair")
    brain = app_mod._brain_mask(app, case_id, volume)
    lo, hi = brain_window(volume, brain if brain is not None and brain.shape == volume.shape else None)
    fg = app_mod._combined_foreground(app, case_id, volume.shape)
    pred = app_mod._cached_pred(app, case_id)
    try:
        second = app_mod._cached_second_mask(app, case_id)
    except FileNotFoundError:
        second = None

    shape = [int(n) for n in volume.shape]
    spacing = [float(abs(v)) for v in voxel_sizes(np.asarray(affine, dtype=np.float64))]
    extents = {str(axis): list(slice_extent_mm(affine, volume.shape, axis)) for axis in (0, 1, 2)}
    window = {"lo": float(lo), "hi": float(hi)}
    windows = {
        name: {"lo": float(lo), "hi": float(hi), "suffix": suf}
        for name, suf in (("flair", "t2f"), ("t1", "t1n"), ("t1c", "t1c"), ("t2", "t2w"))
    }
    exported = {str(axis): _indices(shape[axis], stride) for axis in (0, 1, 2)}
    volume_meta = {
        "modality": "t2f",
        "shape": shape,
        "bbox": {"lo": [0, 0, 0], "hi": [n - 1 for n in shape]},
        "affine": np.asarray(affine, dtype=np.float64).round(6).tolist(),
        "spacing": spacing,
        "extent_mm": extents,
        "letters": {str(axis): dict(letters) for axis, letters in ORIENT_LETTERS.items()},
        "window": window,
        "windows": windows,
        "static": {
            "stride": int(stride),
            "modality": "flair",
            "exported_indices": exported,
            "slice_pattern": "slices/flair_a{axis}_i{index}.png",
            "overlay_model_pattern": "slices/overlay_model_a{axis}_i{index}.png",
            "overlay_second_pattern": "slices/overlay_second_a{axis}_i{index}.png",
        },
    }
    (case_dest / "volume.json").write_text(json.dumps(volume_meta, indent=2), encoding="utf-8")

    # Volumes summary + Dice already live in location / uncertainty; also write a tiny summary.
    location_payload = json.loads((case_dest / "location.json").read_text(encoding="utf-8"))
    summary = {
        "case_id": case_id,
        "volumes_ml": location_payload.get("volumes_ml"),
        "dice": unc.get("dice"),
        "window": window,
        "stride": int(stride),
    }
    (case_dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    count = 0
    for axis in (0, 1, 2):
        for index in exported[str(axis)]:
            flair = render_slice_png(volume, axis, index, lo, hi, alpha=fg)
            _save_png(slices_dest / f"flair_a{axis}_i{index}.png", flair, colors=colors)
            model = render_overlay_png(pred, axis, index, palette="model")
            _save_png(slices_dest / f"overlay_model_a{axis}_i{index}.png", model, colors=None)
            if second is not None:
                expert = render_overlay_png(second, axis, index, palette="expert")
                _save_png(slices_dest / f"overlay_second_a{axis}_i{index}.png", expert, colors=None)
            count += 1

    # Free large arrays between cases.
    with app_mod._VOLUME_CACHE_LOCK:
        app.config["VOLUMES"].clear()
    return {"case_id": case_id, "slices_written": count, "window": window, "stride": stride}


def export_static_demo(
    cases: list[str],
    dest: Path,
    *,
    stride: int = 2,
    colors: int | None = 64,
    target_bytes: int = TARGET_BYTES,
) -> dict:
    from web.app import create_app

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    _copy_viewer_assets(dest)
    css = dest / "viewer.css"
    if css.is_file():
        extra = (
            "\n.static-demo-footer{position:fixed;left:0;right:0;bottom:0;z-index:40;"
            "padding:0.35rem 0.75rem;font:12px/1.4 ui-sans-serif,system-ui,sans-serif;"
            "color:#c9d4e3;background:rgba(8,12,20,0.92);border-top:1px solid #243044;"
            "pointer-events:none;}\n"
            "body{padding-bottom:1.6rem;}\n"
        )
        css.write_text(css.read_text(encoding="utf-8") + extra, encoding="utf-8")

    app = create_app()
    # Exporter needs full mode helpers (dice / second mask) even when BRATS_DEMO is set.
    exported = []
    cuts: list[str] = []
    active_cases = list(cases)
    active_stride = max(1, int(stride))
    active_colors = colors

    while True:
        # Fresh case tree each attempt.
        cases_root = dest / "cases"
        if cases_root.exists():
            shutil.rmtree(cases_root)
        cases_root.mkdir(parents=True)
        exported = []
        for case_id in active_cases:
            print(f"exporting {case_id} stride={active_stride} colors={active_colors}", flush=True)
            exported.append(
                _export_case(app, case_id, dest, stride=active_stride, colors=active_colors)
            )
        (dest / "cases.json").write_text(
            json.dumps({"cases": active_cases, "static": True, "stride": active_stride}, indent=2),
            encoding="utf-8",
        )
        # Tiny vercel-friendly 404 → index for SPA-ish deep links is unnecessary;
        # query-string case routing is enough.
        total = _dir_size(dest)
        print(f"dist size {_fmt_mb(total)} ({total} bytes)", flush=True)
        if total <= target_bytes:
            break
        # Size budget: escalate cuts.
        if active_stride < 2:
            active_stride = 2
            cuts.append("raised stride to 2")
            continue
        if active_stride < 3:
            active_stride = 3
            cuts.append("raised stride to 3 (viewer snaps to nearest exported index)")
            continue
        if len(active_cases) > 3:
            active_cases = active_cases[:3]
            cuts.append("limited to 3 cases")
            continue
        if active_colors is None or active_colors > 32:
            active_colors = 32
            cuts.append("reduced FLAIR PNG palette to 32 colors")
            continue
        if active_stride < 4:
            active_stride = 4
            cuts.append("raised stride to 4")
            continue
        cuts.append(f"still over budget at {_fmt_mb(total)}; leaving as-is")
        break

    meta = {
        "cases": active_cases,
        "stride": active_stride,
        "colors": active_colors,
        "bytes": _dir_size(dest),
        "cuts": cuts,
        "exported": exported,
    }
    (dest / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default=",".join(DEFAULT_CASES),
        help="Comma-separated case ids (default: three public BraTS demo cases).",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "static_demo")
    parser.add_argument("--stride", type=int, default=2, help="Export every Nth slice (default 2).")
    parser.add_argument(
        "--colors",
        type=int,
        default=64,
        help="Quantize FLAIR PNGs to this many colors (0 disables).",
    )
    parser.add_argument("--target-mb", type=float, default=100.0)
    args = parser.parse_args(argv)
    colors = None if int(args.colors) <= 0 else int(args.colors)
    meta = export_static_demo(
        _parse_cases(args.cases),
        args.out,
        stride=max(1, int(args.stride)),
        colors=colors,
        target_bytes=int(args.target_mb * 1024 * 1024),
    )
    print(json.dumps({k: meta[k] for k in ("cases", "stride", "colors", "bytes", "cuts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
