"""Regression: ACS slices must not stay black; 3D meshes must not be degenerate.

Runs against a live server when VIEWER_BASE_URL is set; otherwise starts a
Flask test client is not enough for WebGL — skip unless VIEWER_BASE_URL is set.
Also includes a static check that ACS helpers and cache headers exist.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_acs_helpers_and_cache_bust_are_wired():
    explorer = (ROOT / "web" / "static" / "explorer-ui.js").read_text(encoding="utf-8")
    viewer = (ROOT / "web" / "static" / "viewer.js").read_text(encoding="utf-8")
    html = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
    app = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    assert "acsStripIsPainted" in explorer and "acsStripIsPainted" in viewer
    assert "crossOrigin = \"anonymous\"" not in explorer.split("function loadImage")[1].split("export function")[0]
    assert "attribute vec3 color" not in viewer
    assert "vertexColors: false" in viewer
    assert "glassFallbackMaterial" in viewer
    assert "viewer.js?v=explorer7" in html
    assert "explorer-ui.js?v=explorer7" in viewer
    assert 'id="upload-panel"' in html
    assert "no-cache, must-revalidate" in app
    assert "_VOLUME_CACHE_LOCK" in app


def test_static_asset_cache_headers():
    from web.app import create_app

    client = create_app().test_client()
    js = client.get("/static/viewer.js")
    assert js.status_code == 200
    assert "no-cache" in (js.headers.get("Cache-Control") or "")
    css = client.get("/static/viewer.css")
    assert "no-cache" in (css.headers.get("Cache-Control") or "")


@pytest.mark.skipif(not os.environ.get("VIEWER_BASE_URL"), reason="Set VIEWER_BASE_URL for live Chrome/WebGL check")
def test_live_viewer_acs_and_mesh_bounds_real_or_headless():
    """Fails when ACS canvases are blank or visible meshes have degenerate bounds."""
    from playwright.sync_api import sync_playwright

    base = os.environ["VIEWER_BASE_URL"].rstrip("/")
    channel = os.environ.get("VIEWER_BROWSER_CHANNEL")  # e.g. "chrome"
    headed = os.environ.get("VIEWER_HEADED", "").strip() in {"1", "true", "yes"}
    launch_args = {}
    if channel:
        launch_args["channel"] = channel
    if not headed and not channel:
        launch_args["args"] = [
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
        ]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not headed, **launch_args)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.goto(f"{base}/viewer?case=BraTS20_Training_002", wait_until="domcontentloaded", timeout=180000)
        page.wait_for_function("window.__viewerReady === true", timeout=180000)
        page.wait_for_timeout(1500)
        page.wait_for_function("window.__acsPainted === true", timeout=30000)
        page.wait_for_timeout(500)
        diag = page.evaluate("() => window.__viewerDiagnostics ? window.__viewerDiagnostics() : null")
        assert diag, "viewer diagnostics missing"
        assert diag["acsPainted"] is True, diag
        assert diag.get("glbOk") is True, diag
        assert not diag["degenerateVisible"], f"degenerate visible meshes: {diag['degenerateVisible']}"
        for mesh in diag["meshes"]:
            if mesh["visible"]:
                assert mesh["verts"] >= 24, mesh
        shader_errors = [text for text in errors if "Shader Error" in text or "VALIDATE_STATUS false" in text]
        assert not shader_errors, shader_errors
        upload = page.locator("#upload-scan-btn")
        assert upload.count() == 1
        browser.close()
