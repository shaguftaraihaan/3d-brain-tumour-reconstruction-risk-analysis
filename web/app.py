"""Local viewer and case API.

Upload saves a BraTS folder and runs ``build_scene``. The viewer then loads
``scene.glb``, ``manifest.json``, and PNG MRI slices cut on the same RAS grid
as the meshes.

Run:
    python web/app.py
"""

from __future__ import annotations

import io
import logging
import os
import re
import secrets
import sys
import threading
from pathlib import Path

import nibabel as nib
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REQUIRED = ("t2f", "t1n", "t1c", "t2w")
_ROLE_NAMES = {"t2f": "FLAIR", "t1n": "T1", "t1c": "T1 with contrast", "t2w": "T2"}
# Four flipped views at overlap 0.5: the setting the tuned thresholds were fit on, ~3x faster than eight.
UPLOAD_TTA = "fast"
UPLOAD_OVERLAP = 0.5
_BUILDING: set[str] = set()
_BUILDING_LOCK = threading.Lock()
_JOBS: dict[str, object] = {}
_JOBS_LOCK = threading.Lock()
_VOLUME_CACHE_LOCK = threading.Lock()
logger = logging.getLogger(__name__)

# Bump when viewer static assets change so deploys never serve stale JS/CSS.
_STATIC_ASSET_VERSION = "explorer4"


def create_app(outputs_dir: Path | None = None, cases_dir: Path | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"), static_url_path="/static")
    app.config["OUTPUTS"] = Path(outputs_dir) if outputs_dir else ROOT / "outputs"
    app.config["CASES"] = Path(cases_dir) if cases_dir else ROOT / "data" / "cases"
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
    app.config["VOLUMES"] = {}
    app.config["DEMO_PASSWORD"] = os.environ.get("DEMO_PASSWORD") or ""
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.after_request
    def _cache_headers(response):
        path = request.path or ""
        if path.startswith("/static/"):
            # Versioned viewer assets + GLBs must revalidate after every deploy.
            if path.endswith((".js", ".css", ".mjs", ".map", ".glb", ".gltf")):
                response.headers["Cache-Control"] = "no-cache, must-revalidate"
                response.headers["Pragma"] = "no-cache"
            else:
                response.headers.setdefault("Cache-Control", "public, max-age=3600")
        elif path.endswith(".glb") or path.endswith("scene.glb"):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    @app.before_request
    def _optional_basic_auth():
        password = app.config.get("DEMO_PASSWORD") or ""
        if not password:
            return None
        # Health checks and public static assets must not require credentials.
        if request.path == "/healthz":
            return None
        if request.path.startswith("/static/"):
            return None
        auth = request.authorization
        user_ok = bool(auth and auth.username == "demo")
        pass_ok = bool(auth and secrets.compare_digest(auth.password or "", password))
        if user_ok and pass_ok:
            return None
        return Response(
            "Authentication required",
            401,
            {"WWW-Authenticate": 'Basic realm="Tumor in context demo"'},
        )

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True}), 200

    @app.get("/")
    def landing_page():
        return send_file(Path(app.static_folder) / "landing.html")

    @app.get("/viewer")
    def viewer_page():
        return send_file(Path(app.static_folder) / "index.html")

    @app.errorhandler(FileNotFoundError)
    def missing(exc):
        return jsonify({"error": str(exc)}), 404

    @app.get("/api/capabilities")
    def capabilities():
        """Whether upload/inference can run in this deployment."""
        return jsonify(_capabilities())

    @app.post("/api/identify")
    def identify():
        """Role of each uploaded file name, so the page can show plain scan names."""
        from recon.io import guess_case_id, identify_modality

        payload = request.get_json(silent=True) or {}
        names = [str(name) for name in payload.get("names", [])][:200]
        folder = payload.get("folder") or None
        caps = _capabilities()
        return jsonify(
            {
                "roles": [identify_modality(name) for name in names],
                "required": list(_REQUIRED),
                "case_id": guess_case_id(names, folder),
                **caps,
            }
        )

    @app.get("/api/cases")
    def list_cases():
        """IDs of finished cases under outputs/ (no paths or patient info)."""
        outputs = Path(app.config["OUTPUTS"])
        ids: list[str] = []
        if outputs.is_dir():
            for path in sorted(outputs.iterdir()):
                if not path.is_dir():
                    continue
                name = path.name
                if not _CASE_ID.match(name):
                    continue
                # Never advertise real-patient study folders even if present on disk.
                if name == "real" or name.startswith("study_"):
                    continue
                if (path / "scene.glb").is_file() and (path / "manifest.json").is_file():
                    ids.append(name)
        return jsonify({"cases": ids})

    @app.post("/api/cases")
    def create_case():
        caps = _capabilities()
        if not caps["upload"]:
            return jsonify({"error": caps["message"], "demo": True}), 503
        try:
            case_id = _safe_case_id(request.form.get("case_id") or _guess_case_id())
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        with _BUILDING_LOCK:
            if case_id in _BUILDING:
                return jsonify({"error": "This scan is already being built. Please wait for it to finish."}), 409
            _BUILDING.add(case_id)
        try:
            return _enqueue_case(app, case_id)
        except Exception:
            with _BUILDING_LOCK:
                _BUILDING.discard(case_id)
            raise

    @app.get("/api/cases/<case_id>/job")
    def case_job(case_id: str):
        try:
            case_id = _safe_case_id(case_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        with _JOBS_LOCK:
            progress = _JOBS.get(case_id)
        if progress is not None:
            payload = progress.snapshot()
            payload["case_id"] = case_id
            if payload["status"] == "done":
                payload["viewer"] = f"/viewer?case={case_id}"
            return jsonify(payload)
        out = app.config["OUTPUTS"] / case_id
        if (out / "scene.glb").is_file() and (out / "manifest.json").is_file():
            return jsonify(
                {
                    "case_id": case_id,
                    "status": "done",
                    "stage": "Done",
                    "viewer": f"/viewer?case={case_id}",
                    "stages": {},
                    "error": None,
                    "warning": None,
                    "elapsed": 0,
                }
            )
        return jsonify({"error": f"No job for case {case_id}"}), 404

    @app.get("/api/cases/<case_id>/manifest.json")
    def manifest(case_id: str):
        path = _output_file(app, case_id, "manifest.json")
        return send_file(path, mimetype="application/json", max_age=0)

    @app.get("/api/cases/<case_id>/scene.glb")
    def scene_glb(case_id: str):
        path = _output_file(app, case_id, "scene.glb")
        response = send_file(path, mimetype="model/gltf-binary", max_age=0)
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    @app.get("/api/cases/<case_id>/volume")
    def volume_meta(case_id: str):
        from recon.slices import SEQUENCE_ORDER, resolve_sequence

        modality = request.args.get("modality", "flair")
        try:
            suffix = resolve_sequence(modality)
            data, affine = _cached_volume(app, case_id, modality)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        lo, hi = _foreground_bbox(data)
        brain = _brain_mask(app, case_id, data)
        windows = {}
        for name in SEQUENCE_ORDER:
            try:
                # Only the requested modality stays cached; others are loaded briefly
                # for window stats so demo RSS fits Render's 512 MB free tier.
                if resolve_sequence(name) == suffix:
                    volume = data
                else:
                    volume, _ = _read_volume(app, case_id, name)
            except (FileNotFoundError, ValueError):
                continue
            mask = brain if brain is not None and brain.shape == volume.shape else None
            wlo, whi = _window_for(volume, mask)
            windows[name] = {"lo": wlo, "hi": whi, "suffix": resolve_sequence(name)}
            if resolve_sequence(name) != suffix:
                del volume
        key = modality if modality in windows else "flair"
        chosen = windows.get(key, next(iter(windows.values()), {"lo": 0.0, "hi": 1.0}))
        from nibabel.affines import voxel_sizes
        from recon.slices import ORIENT_LETTERS, slice_extent_mm

        spacing = [float(abs(v)) for v in voxel_sizes(np.asarray(affine, dtype=np.float64))]
        extents = {str(axis): list(slice_extent_mm(affine, data.shape, axis)) for axis in (0, 1, 2)}
        return jsonify(
            {
                "modality": suffix,
                "shape": [int(v) for v in data.shape],
                "bbox": {"lo": lo, "hi": hi},
                "affine": np.asarray(affine, dtype=np.float64).round(6).tolist(),
                "spacing": spacing,
                "extent_mm": extents,
                "letters": {str(axis): dict(letters) for axis, letters in ORIENT_LETTERS.items()},
                "window": {"lo": chosen["lo"], "hi": chosen["hi"]},
                "windows": windows,
            }
        )

    @app.get("/api/cases/<case_id>/slice")
    def slice_png(case_id: str):
        modality = request.args.get("modality", "flair")
        try:
            axis = int(request.args.get("axis", "2"))
            index = int(request.args.get("index", "0"))
        except (TypeError, ValueError):
            return jsonify({"error": "axis and index must be integers"}), 400
        try:
            data, _affine = _cached_volume(app, case_id, modality)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if axis not in (0, 1, 2):
            return jsonify({"error": "axis must be 0, 1, or 2"}), 400
        if index < 0 or index >= data.shape[axis]:
            return jsonify({"error": f"index {index} is outside axis {axis} (length {data.shape[axis]})"}), 400
        brain = _brain_mask(app, case_id, data)
        default_lo, default_hi = _window_for(data, brain if brain is not None and brain.shape == data.shape else None)
        lo = _optional_float(request.args.get("lo"), default_lo)
        hi = _optional_float(request.args.get("hi"), default_hi)
        alpha = _combined_foreground(app, case_id, data.shape)
        png = _cached_slice_png(app, case_id, modality, data, axis, index, lo, hi, alpha=alpha)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=120)

    @app.get("/api/cases/<case_id>/overlay")
    def overlay_png(case_id: str):
        try:
            axis = int(request.args.get("axis", "2"))
            index = int(request.args.get("index", "0"))
        except (TypeError, ValueError):
            return jsonify({"error": "axis and index must be integers"}), 400
        source = (request.args.get("source") or "model").strip().lower()
        if source not in {"model", "second", "prob"}:
            return jsonify({"error": "source must be model, second, or prob"}), 400
        try:
            if source == "model":
                label = _cached_pred(app, case_id)
            elif source == "second":
                label = _cached_second_mask(app, case_id)
            else:
                label = None
                prob = _cached_wt_prob(app, case_id)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        if source == "prob":
            if axis not in (0, 1, 2):
                return jsonify({"error": "axis must be 0, 1, or 2"}), 400
            if index < 0 or index >= prob.shape[axis]:
                return jsonify({"error": f"index {index} is outside axis {axis} (length {prob.shape[axis]})"}), 400
            png = _cached_prob_png(app, case_id, prob, axis, index)
            return send_file(io.BytesIO(png), mimetype="image/png", max_age=120)
        if axis not in (0, 1, 2):
            return jsonify({"error": "axis must be 0, 1, or 2"}), 400
        if index < 0 or index >= label.shape[axis]:
            return jsonify({"error": f"index {index} is outside axis {axis} (length {label.shape[axis]})"}), 400
        tag = "overlay" if source == "model" else "overlay_expert"
        palette = "model" if source == "model" else "expert"
        png = _cached_overlay_png(app, case_id, label, axis, index, tag=tag, palette=palette)
        return send_file(io.BytesIO(png), mimetype="image/png", max_age=120)

    @app.get("/api/cases/<case_id>/location.json")
    def location_json(case_id: str):
        try:
            case_id = _safe_case_id(case_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        path = app.config["OUTPUTS"] / case_id / "location.json"
        if path.is_file():
            return send_file(path, mimetype="application/json", max_age=0)
        return jsonify(_build_location_report(app, case_id))

    @app.get("/api/cases/<case_id>/uncertainty")
    def uncertainty_meta(case_id: str):
        try:
            case_id = _safe_case_id(case_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        out = app.config["OUTPUTS"] / case_id
        has_model = (out / "pred_seg.nii.gz").is_file()
        has_second = False
        try:
            second = _find_second_mask(app, case_id)
            has_second = second is not None
        except FileNotFoundError:
            has_second = False
        has_prob = (out / "wt_prob.nii.gz").is_file()
        dice = None
        if has_model and has_second:
            try:
                dice = _mask_dice(app, case_id)
            except Exception as exc:  # noqa: BLE001 — surface as null, not 500
                logger.warning("dice failed for %s: %s", case_id, exc)
                dice = None
        reasons = {
            "model": None if has_model else "No model segmentation for this case.",
            "second": None
            if has_second
            else "No expert ground-truth mask (seg.nii.gz) for this case.",
            "prob": None
            if has_prob
            else "Needs the model, not available in this demo",
        }
        return jsonify(
            {
                "has_model": has_model,
                "has_second": has_second,
                "has_prob": has_prob,
                "dice": dice,
                "reasons": reasons,
                "sources": {
                    "model": "pred_seg.nii.gz" if has_model else None,
                    "second": "seg.nii.gz" if has_second else None,
                    "prob": "wt_prob.nii.gz" if has_prob else None,
                },
            }
        )

    @app.get("/api/cases/<case_id>/report")
    @app.get("/api/cases/<case_id>/report.html")
    @app.get("/api/cases/<case_id>/report.pdf")
    def report_html(case_id: str):
        try:
            case_id = _safe_case_id(case_id)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            html = _build_report_html(app, case_id)
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        return Response(html, mimetype="text/html; charset=utf-8")

    return app


def _enqueue_case(app: Flask, case_id: str):
    """Save the uploaded scans, validate the grid, and start a background build."""
    dest = app.config["CASES"] / case_id
    dest.mkdir(parents=True, exist_ok=True)
    saved = {}
    for suffix in (*_REQUIRED, "seg", "synthseg"):
        upload = request.files.get(suffix)
        if upload is None or not upload.filename:
            continue
        saved[suffix] = _save_nifti(upload, dest, suffix)
    missing = [suffix for suffix in _REQUIRED if suffix not in saved]
    if missing:
        names = ", ".join(_ROLE_NAMES[suffix] for suffix in missing)
        with _BUILDING_LOCK:
            _BUILDING.discard(case_id)
        return jsonify({"error": f"Missing scans: {names}"}), 400
    if "seg" not in saved:
        _write_empty_seg(saved["t1n"], dest / "seg.nii.gz")

    try:
        from recon.io import validate_case_grid

        validate_case_grid(dest)
    except (FileNotFoundError, ValueError) as exc:
        with _BUILDING_LOCK:
            _BUILDING.discard(case_id)
        return jsonify({"error": str(exc)}), 400

    from recon.progress import BuildProgress

    anatomy = _form_flag("anatomy", default=True)
    tracts = _form_flag("tracts", default=False)
    try:
        model, checkpoint = _upload_model()
    except FileNotFoundError as exc:
        with _BUILDING_LOCK:
            _BUILDING.discard(case_id)
        return jsonify({"error": str(exc), "demo": True}), 503
    synthseg = saved.get("synthseg")
    outputs = app.config["OUTPUTS"]
    progress = BuildProgress()
    with _JOBS_LOCK:
        _JOBS[case_id] = progress
    thread = threading.Thread(
        target=_run_build_job,
        kwargs={
            "case_id": case_id,
            "dest": dest,
            "outputs": outputs,
            "anatomy": anatomy,
            "tracts": tracts,
            "checkpoint": checkpoint,
            "model": model,
            "synthseg": synthseg,
            "progress": progress,
        },
        daemon=True,
        name=f"build-{case_id}",
    )
    thread.start()
    snap = progress.snapshot()
    return jsonify(
        {
            "case_id": case_id,
            "job": f"/api/cases/{case_id}/job",
            "status": snap["status"],
            "stage": snap["stage"],
            "viewer": f"/viewer?case={case_id}",
        }
    ), 202


def _run_build_job(
    *,
    case_id,
    dest,
    outputs,
    anatomy,
    tracts,
    checkpoint,
    model,
    synthseg,
    progress,
):
    """Worker: never touches Flask ``request``."""
    try:
        from recon.scene import build_scene

        build_scene(
            dest,
            anatomy=anatomy,
            tracts=tracts,
            checkpoint=checkpoint,
            output_root=outputs,
            synthseg=synthseg,
            model=model,
            tta=UPLOAD_TTA,
            overlap=UPLOAD_OVERLAP,
            progress=progress,
        )
        if progress.status == "running":
            progress.finish()
    except Exception as exc:
        if progress.status == "running":
            progress.fail(str(exc))
        logger.exception("reconstruction failed for %s", case_id)
    finally:
        with _BUILDING_LOCK:
            _BUILDING.discard(case_id)


def _form_flag(name: str, *, default: bool = False) -> bool:
    """HTML checkbox. Anatomy defaults on so a 3D brain is built even if the box is omitted."""
    raw = request.form.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "on", "yes"}


def _safe_case_id(value: str) -> str:
    text = (value or "").strip()
    if not _CASE_ID.match(text):
        raise ValueError(f"Case id must be letters, numbers, dot, underscore, or hyphen: {value!r}")
    return text


_DEMO_UNAVAILABLE = (
    "Upload and inference are not available in this demo. "
    "Open a public BraTS case from the list instead."
)


# Path only — do not import recon.infer here (that pulls torch/MONAI).
_ABHINAV_CHECKPOINT = ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt"


def _checkpoint_candidates() -> list[Path]:
    """Paths that may hold a SegResNet weight file for upload reconstruction."""
    candidates: list[Path] = []
    env = os.environ.get("BRATS_CHECKPOINT")
    if env:
        candidates.append(Path(env))
    candidates.append(_ABHINAV_CHECKPOINT)
    candidates.append(ROOT / "models" / "best_model.pt")
    candidates.append(ROOT / "checkpoints" / "brats_scale_full" / "best_model.pt")
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _find_checkpoint() -> Path | None:
    for path in _checkpoint_candidates():
        if path.is_file():
            return path
    return None


def _capabilities() -> dict:
    """Upload/inference availability for the HF demo and local installs."""
    checkpoint = _find_checkpoint()
    upload = checkpoint is not None
    return {
        "upload": upload,
        "inference": upload,
        "checkpoint": bool(checkpoint),
        "message": None if upload else _DEMO_UNAVAILABLE,
    }


def _upload_model() -> tuple[str, Path]:
    """``abhinav_v3`` when its checkpoint is present, else the local checkpoint.

    ``BRATS_UPLOAD_MODEL=ours`` forces the local checkpoint.
    Raises ``FileNotFoundError`` when no weight file exists (callers should check
    ``_capabilities`` first so the demo returns 503 instead of a 500).
    """
    # Import constants only when upload/inference is about to run.
    from recon.infer import ABHINAV_V3, OURS

    wanted = os.environ.get("BRATS_UPLOAD_MODEL", ABHINAV_V3)
    if wanted == ABHINAV_V3 and _ABHINAV_CHECKPOINT.is_file():
        return ABHINAV_V3, _ABHINAV_CHECKPOINT
    found = _find_checkpoint()
    if found is None:
        raise FileNotFoundError(_DEMO_UNAVAILABLE)
    if found == _ABHINAV_CHECKPOINT:
        return ABHINAV_V3, found
    return OURS, found


def _guess_case_id() -> str:
    from recon.io import guess_case_id

    names = [upload.filename for upload in request.files.values() if upload and upload.filename]
    return guess_case_id(names)


def _save_nifti(storage, dest: Path, suffix: str) -> Path:
    filename = (storage.filename or "").lower()
    ext = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
    path = dest / f"{suffix}{ext}"
    storage.save(path)
    return path


def _write_empty_seg(reference: Path, dest: Path) -> None:
    image = nib.load(str(reference))
    shape = image.shape[:3]
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), image.affine), str(dest))


def _output_file(app: Flask, case_id: str, name: str) -> Path:
    case_id = _safe_case_id(case_id)
    path = app.config["OUTPUTS"] / case_id / name
    if not path.is_file():
        raise FileNotFoundError(f"No {name} for case {case_id}")
    return path


def _case_dir(app: Flask, case_id: str) -> Path:
    case_id = _safe_case_id(case_id)
    manifest = app.config["OUTPUTS"] / case_id / "manifest.json"
    if manifest.is_file():
        import json

        meta = json.loads(manifest.read_text(encoding="utf-8"))
        recorded = meta.get("case_dir")
        if recorded and Path(recorded).is_dir():
            return Path(recorded)
    fallback = app.config["CASES"] / case_id
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(f"No images for case {case_id}")


def _find_modality(case_dir: Path, suffix: str) -> Path:
    from recon.io import _find_volume

    return _find_volume(case_dir, suffix)


def _foreground_bbox(volume) -> tuple[list[int], list[int]]:
    """Inclusive voxel box of values above the background minimum.

    Collapses one axis at a time instead of ``np.argwhere`` so a full-brain
    mask does not allocate an (N, 3) index array (hundreds of MB on BraTS).
    """
    data = np.asarray(volume)
    if data.size == 0:
        return [0, 0, 0], [0, 0, 0]
    finite = np.isfinite(data)
    if not np.any(finite):
        return [0, 0, 0], [int(n) - 1 for n in data.shape]
    mask = finite & (data > np.nanmin(data))
    if not np.any(mask):
        return [0, 0, 0], [int(n) - 1 for n in data.shape]
    lo: list[int] = []
    hi: list[int] = []
    for axis in range(mask.ndim):
        other = tuple(i for i in range(mask.ndim) if i != axis)
        collapsed = np.any(mask, axis=other)
        idx = np.flatnonzero(collapsed)
        lo.append(int(idx[0]))
        hi.append(int(idx[-1]))
    return lo, hi


def _read_volume(app: Flask, case_id: str, modality: str):
    """Load one RAS float32 volume without touching the in-memory cache."""
    from recon.slices import resolve_sequence

    suffix = resolve_sequence(modality)
    path = _find_modality(_case_dir(app, case_id), suffix)
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.squeeze(np.asanyarray(image.dataobj)).astype(np.float32, copy=False)
    if data.ndim != 3:
        raise ValueError(f"{path.name} is not a 3D volume")
    return data, np.asarray(image.affine, dtype=np.float64)


def _evict_other_cases(cache: dict, case_id: str) -> None:
    """Drop cached arrays for every case except ``case_id`` (demo RSS budget)."""
    for existing in list(cache):
        if isinstance(existing, tuple) and len(existing) >= 1 and existing[0] != case_id:
            cache.pop(existing, None)


def _cached_volume(app: Flask, case_id: str, modality: str):
    from recon.slices import resolve_sequence

    suffix = resolve_sequence(modality)
    key = (case_id, suffix)
    cache = app.config["VOLUMES"]
    with _VOLUME_CACHE_LOCK:
        hit = cache.get(key)
    if hit is not None:
        return hit
    # Load outside the lock so concurrent readers of other keys are not blocked
    # on disk I/O; re-check under the lock before insert.
    loaded = _read_volume(app, case_id, modality)
    with _VOLUME_CACHE_LOCK:
        existing = cache.get(key)
        if existing is not None:
            return existing
        # One active case at a time, and at most one MRI volume for that case.
        _evict_other_cases(cache, case_id)
        for prior in list(cache):
            if (
                isinstance(prior, tuple)
                and len(prior) == 2
                and prior[0] == case_id
                and prior[1] in {"t2f", "t1n", "t1c", "t2w"}
                and prior[1] != suffix
            ):
                cache.pop(prior, None)
        cache[key] = loaded
        return loaded


def _cached_pred(app: Flask, case_id: str):
    cache = app.config["VOLUMES"]
    key = (case_id, "pred")
    with _VOLUME_CACHE_LOCK:
        hit = cache.get(key)
    if hit is not None:
        return hit
    path = _output_file(app, case_id, "pred_seg.nii.gz")
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.rint(np.squeeze(np.asanyarray(image.dataobj))).astype(np.int16, copy=False)
    if data.ndim != 3:
        raise ValueError("pred_seg is not a 3D volume")
    with _VOLUME_CACHE_LOCK:
        existing = cache.get(key)
        if existing is not None:
            return existing
        _evict_other_cases(cache, case_id)
        cache[key] = data
        return data


def _brain_mask(app: Flask, case_id: str, volume) -> np.ndarray:
    cache = app.config["VOLUMES"]
    key = (case_id, "brain")
    if key in cache:
        return cache[key]
    for name in ("anatomy_seg.nii.gz", "pred_seg.nii.gz"):
        path = app.config["OUTPUTS"] / _safe_case_id(case_id) / name
        if not path.is_file():
            continue
        image = nib.as_closest_canonical(nib.load(str(path)))
        labels = np.rint(np.squeeze(np.asanyarray(image.dataobj))).astype(np.int16, copy=False)
        mask = labels != 0
        if np.any(mask):
            cache[key] = mask
            return mask
    data = np.asarray(volume)
    mask = np.isfinite(data) & (data > np.nanmin(data))
    cache[key] = mask
    return mask


def _window_for(volume, brain_mask) -> tuple[float, float]:
    from recon.slices import brain_window

    return brain_window(volume, brain_mask)


def _optional_float(raw, default: float) -> float:
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _slice_cache_dir(app: Flask, case_id: str) -> Path:
    dest = app.config["OUTPUTS"] / _safe_case_id(case_id) / "slices"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _combined_foreground(app: Flask, case_id: str, shape) -> np.ndarray:
    """Opaque MRI voxels. Prefer anatomy ∪ tumor; else sequences that are not 0.

    Normalized BraTS volumes are not exactly 0 outside the brain, so using
    ``volume != 0`` alone leaves a black rectangle. Label masks follow the
    brain. Cases without parcellation still use exact zeros (synthetic tests).
    """
    from recon.slices import SEQUENCE_ORDER
    from scipy import ndimage

    cache = app.config["VOLUMES"]
    key = (case_id, "fg5")
    if key in cache:
        return cache[key]
    fg = np.zeros(tuple(int(n) for n in shape[:3]), dtype=bool)
    has_parc = False
    out_dir = app.config["OUTPUTS"] / _safe_case_id(case_id)
    for name in ("anatomy_seg.nii.gz", "synthseg_raw.nii.gz", "pred_seg.nii.gz"):
        path = out_dir / name
        if not path.is_file():
            continue
        image = nib.as_closest_canonical(nib.load(str(path)))
        labels = np.rint(np.squeeze(np.asarray(image.get_fdata(dtype=np.float32)))).astype(np.int16)
        if labels.shape != fg.shape:
            continue
        if name != "pred_seg.nii.gz":
            has_parc = has_parc or np.any(labels)
        fg |= labels != 0
    if has_parc:
        fg = ndimage.binary_dilation(fg)
        cache[key] = fg
        return fg
    for name in SEQUENCE_ORDER:
        try:
            volume, _ = _cached_volume(app, case_id, name)
        except (FileNotFoundError, ValueError):
            continue
        if volume.shape == fg.shape:
            fg |= np.isfinite(volume) & (volume != 0)
    if not np.any(fg):
        fg = np.ones(fg.shape, dtype=bool)
    cache[key] = fg
    return fg


def _cached_slice_png(app, case_id, modality, data, axis, index, lo, hi, alpha=None) -> bytes:
    from recon.slices import cache_path, render_slice_png, resolve_sequence, window_token

    suffix = resolve_sequence(modality)
    extra = window_token(lo, hi) + ("_rgba_fg5" if alpha is not None else "")
    path = cache_path(_slice_cache_dir(app, case_id), suffix, axis, index, extra)
    if path.is_file():
        return path.read_bytes()
    png = render_slice_png(data, axis, index, lo, hi, alpha=alpha)
    path.write_bytes(png)
    return png


def _cached_overlay_png(app, case_id, label, axis, index, *, tag: str = "overlay", palette: str = "model") -> bytes:
    from recon.slices import cache_path, render_overlay_png

    path = cache_path(_slice_cache_dir(app, case_id), tag, axis, index)
    if path.is_file():
        return path.read_bytes()
    png = render_overlay_png(label, axis, index, palette=palette)
    path.write_bytes(png)
    return png


def _cached_prob_png(app, case_id, prob, axis, index) -> bytes:
    from recon.slices import cache_path, orient_ras_slice

    path = cache_path(_slice_cache_dir(app, case_id), "wt_prob", axis, index)
    if path.is_file():
        return path.read_bytes()
    plane = np.clip(np.asarray(orient_ras_slice(prob, axis, index), dtype=np.float32), 0.0, 1.0)
    rgba = np.zeros(plane.shape + (4,), dtype=np.uint8)
    # Magenta→yellow heat; transparent where probability is near zero.
    rgba[..., 0] = np.clip(255 * plane, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(255 * (plane * 0.55), 0, 255).astype(np.uint8)
    rgba[..., 2] = np.clip(255 * (1.0 - plane) * plane * 2.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(255 * plane * 0.85, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    png = buffer.getvalue()
    path.write_bytes(png)
    return png


def _load_manifest(app: Flask, case_id: str) -> dict:
    import json

    path = _output_file(app, case_id, "manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _tumor_volumes_ml(manifest: dict) -> dict[str, float]:
    structures = manifest.get("structures") or {}
    edema = float((structures.get("edema") or {}).get("volume_cm3") or 0.0)
    necrosis = float((structures.get("necrosis") or {}).get("volume_cm3") or 0.0)
    enhancing = float((structures.get("enhancing") or {}).get("volume_cm3") or 0.0)
    return {
        "wt_ml": edema + necrosis + enhancing,
        "tc_ml": necrosis + enhancing,
        "et_ml": enhancing,
        "edema_ml": edema,
        "necrosis_ml": necrosis,
    }


def _build_location_report(app: Flask, case_id: str) -> dict:
    """Synthesize a location report from manifest when location.json is absent."""
    VOLUME_FLOOR_ML = 20.9
    manifest = _load_manifest(app, case_id)
    volumes = _tumor_volumes_ml(manifest)
    tumor_location = manifest.get("tumor_location") or {"WT": [], "ET": [], "margin_mm": 10.0}
    warning = (manifest.get("anatomy_warning") or "").strip()
    nearest = list(tumor_location.get("WT") or [])
    has_nearest = bool(nearest)
    if warning:
        qc_status = "unreliable"
        reasons = [warning.splitlines()[0]]
        withheld = True
        withheld_reason = warning.splitlines()[0]
    elif not has_nearest:
        qc_status = "caution"
        reasons = ["No overlapping or nearby structures were recorded for this case."]
        withheld = True
        withheld_reason = reasons[0]
    else:
        qc_status = "good"
        reasons = []
        withheld = False
        withheld_reason = None
    below_floor = volumes["wt_ml"] > 0 and volumes["wt_ml"] < VOLUME_FLOOR_ML
    if below_floor:
        qc_status = "unreliable" if qc_status != "good" else "caution"
        reasons.append(f"Whole-tumor volume {volumes['wt_ml']:.1f} mL is below the {VOLUME_FLOOR_ML} mL size floor.")
        if not withheld:
            withheld = True
            withheld_reason = reasons[-1]
    headline = None
    if not withheld and nearest:
        top = nearest[0]
        if top.get("relation") == "overlaps":
            headline = f"Whole tumor overlaps {top['name']}."
        else:
            headline = f"Whole tumor is {float(top.get('distance_mm') or 0):.1f} mm from {top['name']}."
    return {
        "case_id": case_id,
        "qc": {"status": qc_status, "reasons": reasons},
        "plausibility": {
            "floor_ml": VOLUME_FLOOR_ML,
            "wt_ml": round(volumes["wt_ml"], 3),
            "passed": not below_floor and volumes["wt_ml"] > 0,
        },
        "volumes_ml": {key: round(value, 3) for key, value in volumes.items()},
        "tumor_location": tumor_location,
        "nearest": nearest,
        "headline": headline,
        "withheld": withheld,
        "withheld_reason": withheld_reason,
        "sequences": ["FLAIR", "T1", "T1c", "T2"],
        "segmentation_labels": ["necrosis", "edema", "enhancing"],
        "model_note": "model-estimated, unverified",
    }


def _find_second_mask(app: Flask, case_id: str) -> Path | None:
    case_dir = _case_dir(app, case_id)
    for name in ("seg.nii.gz", "seg.nii"):
        path = case_dir / name
        if not path.is_file():
            continue
        image = nib.as_closest_canonical(nib.load(str(path)))
        data = np.rint(np.squeeze(np.asanyarray(image.dataobj))).astype(np.int16, copy=False)
        if np.any(data):
            return path
    return None


def _cached_second_mask(app: Flask, case_id: str):
    cache = app.config["VOLUMES"]
    key = (case_id, "second")
    with _VOLUME_CACHE_LOCK:
        hit = cache.get(key)
    if hit is not None:
        return hit
    path = _find_second_mask(app, case_id)
    if path is None:
        raise FileNotFoundError(f"No second mask for case {case_id}")
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.rint(np.squeeze(np.asanyarray(image.dataobj))).astype(np.int16, copy=False)
    if data.ndim != 3:
        raise ValueError("second mask is not a 3D volume")
    with _VOLUME_CACHE_LOCK:
        existing = cache.get(key)
        if existing is not None:
            return existing
        _evict_other_cases(cache, case_id)
        cache[key] = data
        return data


def _cached_wt_prob(app: Flask, case_id: str):
    cache = app.config["VOLUMES"]
    key = (case_id, "wt_prob")
    with _VOLUME_CACHE_LOCK:
        hit = cache.get(key)
    if hit is not None:
        return hit
    path = _output_file(app, case_id, "wt_prob.nii.gz")
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.squeeze(np.asarray(image.get_fdata(dtype=np.float32)))
    if data.ndim != 3:
        raise ValueError("wt_prob is not a 3D volume")
    clipped = np.clip(data, 0.0, 1.0)
    with _VOLUME_CACHE_LOCK:
        existing = cache.get(key)
        if existing is not None:
            return existing
        _evict_other_cases(cache, case_id)
        cache[key] = clipped
        return clipped


def _mask_dice(app: Flask, case_id: str) -> dict:
    from recon.evaluate import region_overlap, regions_from_label

    pred = _cached_pred(app, case_id)
    second = _cached_second_mask(app, case_id)
    if pred.shape != second.shape:
        raise ValueError(f"Mask shapes differ: pred {pred.shape} vs second {second.shape}")
    pred_regions = regions_from_label(pred)
    gt_regions = regions_from_label(second)
    out = {}
    for name in ("WT", "TC", "ET"):
        dice, _precision, _recall = region_overlap(pred_regions[name], gt_regions[name])
        out[name.lower()] = round(float(dice), 4)
    return out


def _build_report_html(app: Flask, case_id: str) -> str:
    """Self-contained hospital-style HTML report (print to PDF from the browser)."""
    import base64
    import html as html_lib
    from datetime import datetime, timezone

    from recon.evaluate import regions_from_label

    manifest = _load_manifest(app, case_id)
    location = _build_location_report(app, case_id)
    volumes = location["volumes_ml"]
    wt = float(volumes.get("wt_ml") or 0.0)
    tc = float(volumes.get("tc_ml") or 0.0)
    et = float(volumes.get("et_ml") or 0.0)
    # Approximate edema / necrosis from exclusive compartment volumes when present.
    edema = max(wt - tc, 0.0)
    necrosis = max(tc - et, 0.0)
    dice = None
    try:
        if _find_second_mask(app, case_id) is not None:
            dice = _mask_dice(app, case_id)
    except Exception:  # noqa: BLE001
        dice = None

    def pct(part: float) -> str:
        if wt <= 0:
            return "—"
        return f"{100.0 * part / wt:.1f}%"

    def esc(value) -> str:
        return html_lib.escape(str(value if value is not None else "—"))

    now = datetime.now(timezone.utc).astimezone()
    stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    report_id = f"TIC-{case_id}-{now.strftime('%Y%m%d%H%M')}"
    qc = location["qc"]["status"]
    plaus = "passed" if location["plausibility"]["passed"] else "failed"
    source = "Public BraTS training case" if case_id.startswith("BraTS") else "Uploaded study"
    sequences = ", ".join(location.get("sequences") or ["FLAIR", "T1", "T1c", "T2"])

    if location.get("withheld"):
        findings_location = (
            f"Location not reported: {location.get('withheld_reason') or 'registration quality insufficient'}."
        )
    elif location.get("headline"):
        findings_location = location["headline"]
    else:
        findings_location = "Location not reported: no nearby-structure summary available."

    lesion = "A lesion is present on the submitted MRI sequences." if wt > 0 else "No lesion volume was measured."
    findings = (
        f"{lesion} Estimated volumes: whole tumor {wt:.1f} mL, tumor core {tc:.1f} mL, "
        f"enhancing {et:.1f} mL, edema (WT−TC) {edema:.1f} mL, necrosis (TC−ET) {necrosis:.1f} mL. "
        f"{findings_location}"
    )
    impression = (
        f"Automated segmentation suggests a lesion with whole-tumor volume about {wt:.1f} mL "
        f"and enhancing component about {et:.1f} mL. "
        f"Registration QC is {qc}; size-floor check {plaus}. "
        "Findings are research estimates only. Model-estimated, unverified."
    )

    image_blocks = []
    try:
        data, _affine = _cached_volume(app, case_id, "flair")
        pred = _cached_pred(app, case_id)
        wt_mask = regions_from_label(pred)["WT"]
        if np.any(wt_mask):
            zs = np.where(np.any(wt_mask, axis=(0, 1)))[0]
            indices = [int(zs[len(zs) // 4]), int(zs[len(zs) // 2]), int(zs[(3 * len(zs)) // 4])]
        else:
            mid = data.shape[2] // 2
            indices = [max(mid - 8, 0), mid, min(mid + 8, data.shape[2] - 1)]
        brain = _brain_mask(app, case_id, data)
        lo, hi = _window_for(data, brain if brain is not None and brain.shape == data.shape else None)
        alpha = _combined_foreground(app, case_id, data.shape)
        for index in indices:
            mri = Image.open(
                io.BytesIO(_cached_slice_png(app, case_id, "flair", data, 2, index, lo, hi, alpha=alpha))
            ).convert("RGBA")
            overlay = Image.open(io.BytesIO(_cached_overlay_png(app, case_id, pred, 2, index))).convert("RGBA")
            composed = Image.alpha_composite(mri, overlay).convert("RGB")
            buf = io.BytesIO()
            composed.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            image_blocks.append(
                f'<figure><img src="data:image/png;base64,{b64}" alt="Axial FLAIR z={index}">'
                f"<figcaption>Axial FLAIR, slice {index}, tumor outlined</figcaption></figure>"
            )
        image_note = "Three axial key images with tumor outline. A 3D snapshot is not embedded in this demo report."
    except Exception as exc:  # noqa: BLE001
        image_note = f"Key images unavailable: {exc}"
        image_blocks = []

    dice_rows = ""
    if dice:
        dice_rows = (
            f"<tr><td>Dice vs expert mask</td><td>WT {dice.get('wt', '—')}</td>"
            f"<td>TC {dice.get('tc', '—')}</td><td>ET {dice.get('et', '—')}</td></tr>"
        )

    qc_reasons = "".join(f"<li>{esc(reason)}</li>" for reason in (location["qc"].get("reasons") or []))
    if not qc_reasons:
        qc_reasons = "<li>No additional QC flags.</li>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Research Imaging Summary — {esc(case_id)}</title>
<style>
  @page {{ size: A4; margin: 16mm; }}
  body {{ font: 11pt/1.45 "IBM Plex Sans", "Segoe UI", sans-serif; color: #1a1f27; margin: 0; background: #fff; }}
  main {{ max-width: 780px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 18pt; margin: 0 0 4px; }}
  h2 {{ font-size: 12pt; margin: 22px 0 8px; border-bottom: 1px solid #ccc; padding-bottom: 4px; page-break-after: avoid; }}
  .banner {{ background: #fff3cd; border: 2px solid #c9a227; color: #5c4500; padding: 10px 12px; font-weight: 700;
             text-transform: uppercase; letter-spacing: 0.03em; margin: 12px 0 18px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 8px 0 12px; page-break-inside: avoid; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f3f5f8; }}
  .meta {{ color: #555; font-size: 10pt; }}
  .images {{ display: flex; flex-wrap: wrap; gap: 10px; page-break-inside: avoid; }}
  .images figure {{ margin: 0; width: 230px; }}
  .images img {{ width: 100%; border: 1px solid #bbb; }}
  .sign {{ border: 1px solid #999; padding: 14px; margin-top: 18px; page-break-inside: avoid; }}
  .muted {{ color: #666; }}
  @media print {{
    body {{ background: #fff; }}
    main {{ padding: 0; max-width: none; }}
    a {{ color: inherit; text-decoration: none; }}
  }}
</style>
</head>
<body>
<main>
  <header>
    <h1>Tumor in context, Research Imaging Summary</h1>
    <p class="meta">Report ID: {esc(report_id)} · Generated: {esc(stamp)}</p>
    <p class="banner">RESEARCH PROTOTYPE, NOT A CLINICAL REPORT, NOT FOR DIAGNOSIS</p>
  </header>

  <h2>Study details</h2>
  <table>
    <tr><th>Study / case ID</th><td>{esc(case_id)}</td></tr>
    <tr><th>Data source</th><td>{esc(source)}</td></tr>
    <tr><th>Sequences used</th><td>{esc(sequences)}</td></tr>
    <tr><th>Software / model</th><td>Tumor in context demo viewer · BraTS-trained SegResNet (when available)</td></tr>
    <tr><th>Patient name</th><td>not recorded</td></tr>
    <tr><th>Patient ID / DOB / sex</th><td>not recorded</td></tr>
  </table>

  <h2>Technique</h2>
  <p>MRI volumes were reoriented to RAS. Tumor labels (necrosis, edema, enhancing) come from the stored
  model segmentation for this case. Structure proximity uses the exported scene atlas when registration QC allows.</p>

  <h2>Findings</h2>
  <p>{esc(findings)}</p>

  <h2>Measurements</h2>
  <table>
    <thead><tr><th>Region</th><th>Volume (mL)</th><th>Percent of whole tumor</th><th>Notes</th></tr></thead>
    <tbody>
      <tr><td>Whole tumor (WT)</td><td>{wt:.1f}</td><td>100%</td><td>Necrosis + edema + enhancing</td></tr>
      <tr><td>Tumor core (TC)</td><td>{tc:.1f}</td><td>{pct(tc)}</td><td>Necrosis + enhancing</td></tr>
      <tr><td>Enhancing (ET)</td><td>{et:.1f}</td><td>{pct(et)}</td><td></td></tr>
      <tr><td>Edema (approx.)</td><td>{edema:.1f}</td><td>{pct(edema)}</td><td>WT − TC</td></tr>
      <tr><td>Necrosis (approx.)</td><td>{necrosis:.1f}</td><td>{pct(necrosis)}</td><td>TC − ET</td></tr>
      {dice_rows}
    </tbody>
  </table>

  <h2>Key images</h2>
  <p class="muted">{esc(image_note)}</p>
  <div class="images">{"".join(image_blocks)}</div>

  <h2>Impression</h2>
  <p>{esc(impression)}</p>

  <h2>Limitations and QC</h2>
  <p>QC status: <strong>{esc(qc)}</strong>. Plausibility (size floor {location['plausibility']['floor_ml']} mL): <strong>{esc(plaus)}</strong>.</p>
  <ul>{qc_reasons}
    <li>Model trained on BraTS public data; not validated for real-patient care.</li>
    <li>Atlas tracts are not patient tractography.</li>
    <li>Location may be withheld when registration quality is insufficient.</li>
  </ul>

  <div class="sign">
    <p><strong>Sign-off</strong></p>
    <p>Reviewed by: ____________________ &nbsp;&nbsp; Date: __________ &nbsp;&nbsp; Status: not reviewed</p>
  </div>
</main>
</body>
</html>
"""


def _slice_png(volume: np.ndarray, axis: int, index: int) -> bytes:
    """PNG of one radiological RAS slice (same orientation as the 2D panels)."""
    from recon.slices import brain_window, render_slice_png

    lo, hi = brain_window(volume)
    return render_slice_png(volume, axis, index, lo, hi)


app = create_app()


def _warm_imports() -> None:
    """Import torch/MONAI and the scene code once, so the first request is not the slow one."""
    import recon.io  # noqa: F401
    import recon.scene  # noqa: F401


if __name__ == "__main__":
    # Demo / Render images skip warm-import so torch stays unloaded until upload.
    if os.environ.get("BRATS_DEMO", "").strip() not in {"1", "true", "yes"} and _find_checkpoint() is not None:
        threading.Thread(target=_warm_imports, daemon=True).start()
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="0.0.0.0", port=port)
