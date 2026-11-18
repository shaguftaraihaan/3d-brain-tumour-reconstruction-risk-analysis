"""Optional DEMO_PASSWORD Basic Auth and torch-free demo startup."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_basic_auth_requires_credentials(monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", "render-demo-secret")
    from web.app import create_app

    client = create_app().test_client()
    assert client.get("/").status_code == 401
    assert client.get("/viewer").status_code == 401
    assert client.get("/api/cases").status_code == 401
    # Health checks stay open.
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json["ok"] is True
    # Static assets stay open (must not leak case volumes).
    static = client.get("/static/landing.html")
    assert static.status_code == 200


def test_basic_auth_accepts_demo_user(monkeypatch):
    monkeypatch.setenv("DEMO_PASSWORD", "render-demo-secret")
    from web.app import create_app

    client = create_app().test_client()
    token = base64.b64encode(b"demo:render-demo-secret").decode("ascii")
    headers = {"Authorization": f"Basic {token}"}
    assert client.get("/", headers=headers).status_code == 200
    assert client.get("/viewer", headers=headers).status_code == 200
    assert client.get("/api/cases", headers=headers).status_code == 200


def test_basic_auth_disabled_without_env(monkeypatch):
    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    from web.app import create_app

    client = create_app().test_client()
    assert client.get("/").status_code == 200
    assert client.get("/healthz").status_code == 200


def test_demo_paths_do_not_import_torch_or_monai(monkeypatch, tmp_path):
    """Serving landing + public BraTS case APIs must not load torch/MONAI."""
    import json

    import nibabel as nib
    import numpy as np

    monkeypatch.delenv("DEMO_PASSWORD", raising=False)
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch.") or name == "monai" or name.startswith("monai."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    outputs = tmp_path / "outputs"
    cases = tmp_path / "cases"
    shape = (6, 5, 4)
    data = np.zeros(shape, dtype=np.float32)
    data[1, 2, 3] = 40.0
    case_dir = cases / "BraTS20_Training_002"
    case_dir.mkdir(parents=True)
    for name in ("t2f", "t1n", "t1c", "t2w"):
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(case_dir / f"{name}.nii.gz"))
    pred = np.zeros(shape, dtype=np.uint8)
    pred[1, 2, 3] = 1
    out = outputs / "BraTS20_Training_002"
    out.mkdir(parents=True)
    nib.save(nib.Nifti1Image(pred, np.eye(4)), str(out / "pred_seg.nii.gz"))
    (out / "scene.glb").write_bytes(b"glb")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "case_id": "BraTS20_Training_002",
                "case_dir": str(case_dir),
                "structures": {},
                "tract_proximity": {},
            }
        ),
        encoding="utf-8",
    )

    from web.app import create_app

    client = create_app(outputs, cases).test_client()
    assert client.get("/").status_code == 200
    assert client.get("/viewer").status_code == 200
    assert client.get("/api/cases").status_code == 200
    assert client.get("/api/capabilities").status_code == 200
    assert client.get("/api/cases/BraTS20_Training_002/manifest.json").status_code == 200
    assert client.get("/api/cases/BraTS20_Training_002/scene.glb").status_code == 200
    volume = client.get("/api/cases/BraTS20_Training_002/volume?modality=flair")
    assert volume.status_code == 200
    slice_png = client.get("/api/cases/BraTS20_Training_002/slice?modality=flair&axis=0&index=1")
    assert slice_png.status_code == 200

    loaded = [name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "monai" or name.startswith("monai.")]
    assert loaded == [], f"demo path imported ML stack: {loaded}"
