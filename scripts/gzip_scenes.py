"""Create scene.glb.gz for every built case under outputs/ and outputs/real/.

Example:
    python scripts/gzip_scenes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.scene import default_output_root, write_scene_glb_gz  # noqa: E402


def _case_dirs(outputs: Path) -> list[Path]:
    found: list[Path] = []
    if not outputs.is_dir():
        return found
    for child in sorted(outputs.iterdir()):
        if child.name == "real" or not child.is_dir():
            continue
        if (child / "scene.glb").is_file():
            found.append(child)
    real_root = outputs / "real"
    if real_root.is_dir():
        for child in sorted(real_root.iterdir()):
            if child.is_dir() and (child / "scene.glb").is_file():
                found.append(child)
    return found


def main(argv: list[str] | None = None) -> int:
    outputs = default_output_root() if not argv else Path(argv[0])
    cases = _case_dirs(outputs)
    if not cases:
        print(f"No scene.glb files under {outputs}")
        return 0
    for out_dir in cases:
        glb = out_dir / "scene.glb"
        before = glb.stat().st_size
        gz = write_scene_glb_gz(glb)
        after = gz.stat().st_size
        rel = out_dir.relative_to(outputs) if out_dir.is_relative_to(outputs) else out_dir
        print(f"{rel}: {before} -> {after} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
