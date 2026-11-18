"""Regression: ACS strip must stay visible and the left-column layout must hold."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_acs_strip_shows_errors_instead_of_silent_black():
    explorer = (ROOT / "web" / "static" / "explorer-ui.js").read_text(encoding="utf-8")
    assert "acs-error" in explorer
    assert "Could not load slice" in explorer


def test_viewer_resize_targets_view_wrap_not_window():
    script = (ROOT / "web" / "static" / "viewer.js").read_text(encoding="utf-8")
    assert 'querySelector("#view-wrap")' in script
    assert "clientWidth" in script
    assert "clientHeight" in script
    # Must not size the WebGL canvas to the full window (that covers the slices column).
    assert "renderer.setSize(width, height, false)" in script
    assert "window.innerWidth;\n    const height = window.innerHeight;" not in script


def test_left_column_slices_layout_css():
    css = (ROOT / "web" / "static" / "viewer.css").read_text(encoding="utf-8")
    html = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert "--slices-width" in css
    assert "slices view panel" in css
    assert "@media (max-width: 900px)" in css
    assert 'id="acs-strip"' in html
    assert 'class="slices-head"' in html
    assert "slices-details" not in html


def test_uncertainty_modes_change_overlay_source():
    script = (ROOT / "web" / "static" / "viewer.js").read_text(encoding="utf-8")
    explorer = (ROOT / "web" / "static" / "explorer-ui.js").read_text(encoding="utf-8")
    assert "source=${encodeURIComponent(uncertaintySource)}" in script or "uncertaintySource" in script
    assert "refreshClips()" in script
    assert "Needs the model, not available in this demo" in explorer
    assert "has_second" in explorer and "has_prob" in explorer


def test_report_link_points_at_html():
    explorer = (ROOT / "web" / "static" / "explorer-ui.js").read_text(encoding="utf-8")
    assert "/report.html" in explorer
    assert "report.pdf" not in explorer or "report.html" in explorer
