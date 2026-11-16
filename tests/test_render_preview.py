"""Preview output-folder lookup and --all-missing skip logic (no browser)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.scene import resolve_output_dir
from scripts.render_preview import cases_missing_preview, main


def _built(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps({"case_id": folder.name}), encoding="utf-8")
    (folder / "scene.glb").write_bytes(b"glb")


def test_resolve_output_dir_brats_and_real(tmp_path):
    outputs = tmp_path / "outputs"
    brats = outputs / "BraTS20_Training_001"
    real = outputs / "real" / "study_abc123def456"
    _built(brats)
    _built(real)

    assert resolve_output_dir(outputs, "BraTS20_Training_001") == brats
    assert resolve_output_dir(outputs, "study_abc123def456") == real


def test_all_missing_skips_case_with_preview(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    has_preview = outputs / "BraTS20_Training_001"
    missing = outputs / "real" / "study_need_preview"
    _built(has_preview)
    _built(missing)
    (has_preview / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    assert cases_missing_preview(outputs) == [missing]

    calls: list[str] = []

    def fake_render(case_id, host="http://127.0.0.1:8000", *, outputs=None, force=False):
        calls.append(case_id)
        dest = resolve_output_dir(outputs, case_id) / "preview.png"
        dest.write_bytes(b"png")
        return dest

    monkeypatch.setattr("scripts.render_preview.render_preview", fake_render)
    code = main(["--all-missing", "--outputs", str(outputs)])
    assert code == 0
    assert calls == ["study_need_preview"]
