"""Real-Chrome verification for the backend-free static demo export."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "static_demo"


@pytest.mark.skipif(not DIST.is_dir(), reason="Run scripts/export_static_demo.py first")
def test_static_demo_export_under_100mb_and_has_cases():
    total = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    assert total <= 100 * 1024 * 1024, f"dist is {total / (1024 * 1024):.1f} MB"
    index = (DIST / "index.html").read_text(encoding="utf-8")
    explorer = (DIST / "explorer.html").read_text(encoding="utf-8")
    assert "CerebraVision" in index
    assert "hero.js" in index and "viewer.js" not in index
    assert "./vendor/three/three.module.js" in index
    assert "cdn." not in index.lower()
    assert "explorer.html?case=BraTS20_Training_002" in index
    assert "__STATIC_DEMO__" in explorer
    assert 'id="home-link"' in explorer and 'href="./index.html"' in explorer
    cases = json.loads((DIST / "cases.json").read_text(encoding="utf-8"))["cases"]
    assert len(cases) >= 1
    for case_id in cases:
        case_dir = DIST / "cases" / case_id
        assert (case_dir / "scene.glb").is_file()
        assert (case_dir / "manifest.json").is_file()
        assert (case_dir / "volume.json").is_file()
        assert (case_dir / "report.html").is_file()
        assert (case_dir / "uncertainty.json").is_file()


@pytest.mark.skipif(
    not os.environ.get("STATIC_DEMO_BASE_URL"),
    reason="Set STATIC_DEMO_BASE_URL to a python -m http.server root for dist/static_demo",
)
def test_static_demo_real_chrome_per_case():
    from playwright.sync_api import sync_playwright

    base = os.environ["STATIC_DEMO_BASE_URL"].rstrip("/")
    out = ROOT / "docs" / "ui_checks" / "static_demo"
    out.mkdir(parents=True, exist_ok=True)
    cases = json.loads((DIST / "cases.json").read_text(encoding="utf-8"))["cases"]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=False)
        for case_id in cases:
            page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
            failed = []
            console_errors = []
            page.on(
                "requestfailed",
                lambda req: failed.append(f"{req.method} {req.url} :: {req.failure}"),
            )
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text)
                if msg.type == "error" and "favicon.ico" not in (msg.location or {}).get("url", "")
                else None,
            )
            page.on("pageerror", lambda err: console_errors.append(f"pageerror:{err}"))
            page.goto(f"{base}/explorer.html?case={case_id}", wait_until="domcontentloaded", timeout=180000)
            page.wait_for_function("window.__viewerReady === true", timeout=180000)
            page.wait_for_timeout(2000)
            page.wait_for_function("window.__acsPainted === true", timeout=30000)
            diag = page.evaluate("() => window.__viewerDiagnostics()")
            assert diag["glbOk"] is True, diag
            assert not diag["degenerateVisible"], diag
            assert diag["acsPainted"] is True, diag
            assert diag.get("glassBroken") is False, diag

            # Second mask + Dice
            second = page.locator('#uncertainty-modes [data-unc="second"]')
            assert second.count() == 1
            assert second.is_enabled()
            second.click()
            page.wait_for_timeout(800)
            dice = page.locator("#uncertainty-dice").inner_text()
            assert "WT" in dice and "0." in dice, dice

            # Report download href resolves
            from urllib.parse import urljoin

            href = page.locator("#download-report").get_attribute("href")
            assert href and "report.html" in href
            report = page.request.get(urljoin(page.url, href))
            assert report.status == 200
            body = report.body()[:400].lower()
            assert b"<html" in body or b"<!doctype" in body

            upload = page.locator("#upload-scan-btn")
            assert upload.is_disabled()

            page.screenshot(path=str(out / f"{case_id}_viewer.png"), full_page=False)
            assert not failed, failed
            assert not console_errors, console_errors
            page.close()

        # Upload controls are hard-disabled on the picker screen; drops must not navigate away.
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page.goto(f"{base}/explorer.html", wait_until="domcontentloaded", timeout=120000)
        page.wait_for_selector("#demo-note:not([hidden])", timeout=30000)
        page.wait_for_selector("#case-list button", timeout=30000)
        assert page.locator("#case-picker").evaluate("el => el.classList.contains('is-primary')")
        assert page.locator("#pick-folder").is_disabled()
        assert page.locator("#pick-files").is_disabled()
        assert page.locator("#upload-submit").is_disabled()
        assert page.locator("#drop-zone").get_attribute("aria-disabled") == "true"
        assert page.locator("#advanced").is_hidden()
        note = page.locator("#demo-note").inner_text()
        assert "trained model" in note and "3 BraTS example cases" in note
        before = page.url
        page.evaluate(
            """() => {
              const event = new Event('drop', { bubbles: true, cancelable: true });
              Object.defineProperty(event, 'dataTransfer', {
                value: { files: [], items: [], types: ['Files'] },
              });
              window.dispatchEvent(new Event('dragover', { bubbles: true, cancelable: true }));
              window.dispatchEvent(event);
              document.body.dispatchEvent(event);
            }"""
        )
        page.wait_for_timeout(400)
        assert page.url == before
        page.locator("#case-list button").first.click()
        page.wait_for_url("**/explorer.html?case=*", timeout=30000)
        page.wait_for_function("window.__viewerReady === true", timeout=180000)
        page.close()
        browser.close()
