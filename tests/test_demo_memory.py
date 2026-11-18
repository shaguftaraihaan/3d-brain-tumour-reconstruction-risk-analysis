"""Demo RSS must stay under 400 MB after concurrent slice/scene traffic."""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_healthz_reports_rss(monkeypatch):
    monkeypatch.setenv("BRATS_DEMO", "1")
    from web.app import create_app

    client = create_app().test_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json["ok"] is True
    assert "rss_mb" in response.json
    assert response.json["demo"] is True


def test_demo_does_not_import_torch_or_monai(monkeypatch):
    monkeypatch.setenv("BRATS_DEMO", "1")
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch.") or name == "monai" or name.startswith("monai."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    from web.app import create_app

    client = create_app().test_client()
    assert client.get("/healthz").status_code == 200
    assert client.get("/api/cases/BraTS20_Training_002/manifest.json").status_code == 200
    assert client.get("/api/cases/BraTS20_Training_002/volume?modality=flair").status_code == 200
    assert client.get("/api/cases/BraTS20_Training_002/slice?modality=flair&axis=2&index=77").status_code == 200
    assert "torch" not in sys.modules
    assert "monai" not in sys.modules


def test_demo_capabilities_disable_upload(monkeypatch):
    monkeypatch.setenv("BRATS_DEMO", "1")
    from web.app import create_app

    caps = create_app().test_client().get("/api/capabilities")
    assert caps.status_code == 200
    assert caps.json["upload"] is False
    assert "not available in this demo" in caps.json["message"]


def _gunicorn_rss_mb(port: int) -> float:
    out = subprocess.check_output(["ps", "-ax", "-o", "pid=,rss=,command="], text=True)
    best = 0.0
    for line in out.splitlines():
        if "gunicorn" not in line or str(port) not in line:
            continue
        pid, rss, *_rest = line.split(None, 2)
        best = max(best, int(rss) / 1024)
    return best


@pytest.mark.skipif(not os.environ.get("VIEWER_BASE_URL"), reason="Set VIEWER_BASE_URL to a live gunicorn demo")
def test_live_gunicorn_concurrent_load_under_400mb():
    base = os.environ["VIEWER_BASE_URL"].rstrip("/")
    # Infer port for ps matching.
    port = 8000
    if "://" in base:
        host = base.split("://", 1)[1]
        if ":" in host:
            port = int(host.split(":", 1)[1].split("/", 1)[0])
    cases = ["BraTS20_Training_002", "BraTS20_Training_004", "BraTS20_Training_022"]
    paths = []
    for case in cases:
        paths.extend(
            [
                f"/api/cases/{case}/manifest.json",
                f"/api/cases/{case}/volume?modality=flair",
                f"/api/cases/{case}/scene.glb",
                f"/api/cases/{case}/slice?modality=flair&axis=2&index=77",
                f"/api/cases/{case}/overlay?source=model&axis=2&index=77",
            ]
        )
    batch = (paths * 2)[:20]

    def hit(path: str):
        with urllib.request.urlopen(base + path, timeout=180) as response:
            # Read in chunks so we do not hold twenty full GLBs in the client.
            total = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
            return response.status, total

    idle = urllib.request.urlopen(base + "/healthz", timeout=30).read()
    assert b"rss_mb" in idle
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(hit, batch))
    assert all(status == 200 for status, _size in results)
    health = urllib.request.urlopen(base + "/healthz", timeout=30).read().decode()
    import json

    payload = json.loads(health)
    rss = float(payload["rss_mb"])
    worker = _gunicorn_rss_mb(port)
    assert rss <= 400, payload
    assert worker <= 400, f"gunicorn worker RSS {worker} MB"
