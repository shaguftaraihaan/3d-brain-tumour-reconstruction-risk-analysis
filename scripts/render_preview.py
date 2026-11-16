"""Capture the default glass-brain view to ``preview.png`` beside scene.glb.

Example:
    python scripts/render_preview.py --case BraTS20_Training_002
    python scripts/render_preview.py --all-missing
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.scene import (  # noqa: E402
    default_output_root,
    iter_built_output_dirs,
    resolve_output_dir,
)


def _chrome() -> Path:
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        Path("/usr/bin/google-chrome"),
        Path("/usr/bin/chromium"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Google Chrome is required to render preview.png")


def cases_missing_preview(outputs: Path | str) -> list[Path]:
    """Built case folders that do not yet have ``preview.png``."""
    return [folder for folder in iter_built_output_dirs(outputs) if not (folder / "preview.png").is_file()]


def render_preview(
    case_id: str,
    host: str = "http://127.0.0.1:8000",
    *,
    outputs: Path | str | None = None,
    force: bool = False,
) -> Path:
    outputs = Path(outputs) if outputs is not None else default_output_root()
    out_dir = resolve_output_dir(outputs, case_id)
    dest = out_dir / "preview.png"
    if dest.is_file() and not force:
        raise FileExistsError(f"preview already exists: {dest} (pass --force to overwrite)")
    chrome = _chrome()
    profile = Path(tempfile.mkdtemp(prefix="brats-preview-"))
    url = f"{host}/viewer?case={case_id}&preview=1"
    cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--user-data-dir={profile}",
        "--window-size=1600,900",
        "--virtual-time-budget=40000",
        f"--screenshot={dest}",
        url,
    ]
    subprocess.run(cmd, check=True)
    if not dest.is_file():
        raise RuntimeError(f"Chrome did not write {dest}")
    return dest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render the default glass-brain preview")
    parser.add_argument("--case", default=None, help="Case id under outputs/ or outputs/real/")
    parser.add_argument("--all-missing", action="store_true", help="Render every built case that lacks preview.png")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing preview.png")
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--outputs", type=Path, default=None, help="Outputs root (default: repo outputs/)")
    args = parser.parse_args(argv)
    outputs = args.outputs if args.outputs is not None else default_output_root()

    if args.all_missing:
        missing = cases_missing_preview(outputs)
        if not missing:
            print("No built cases missing preview.png", flush=True)
            return 0
        failures = 0
        for out_dir in missing:
            case_id = out_dir.name
            try:
                path = render_preview(case_id, args.host, outputs=outputs, force=args.force)
                print(f"{case_id}: ok ({path})", flush=True)
            except Exception as exc:
                failures += 1
                print(f"{case_id}: {exc}", flush=True)
        return 1 if failures else 0

    case_id = args.case or "BraTS20_Training_002"
    path = render_preview(case_id, args.host, outputs=outputs, force=args.force)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
