"""Set ``anatomy_method`` on older manifests that lack the key.

Heuristic: if a manifest has none of Hippocampus, Amygdala, Pallidum, or
Brainstem among its structure names, treat it as an MNI-atlas case and set
``anatomy_method`` to ``mni-atlas``. Otherwise print the case id for manual
review and leave the file alone.

Example:
    python scripts/backfill_anatomy_method.py --dry-run
    python scripts/backfill_anatomy_method.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.scene import default_output_root  # noqa: E402

SYNTHSEG_MARKERS = ("Hippocampus", "Amygdala", "Pallidum", "Brainstem")


def _case_dirs(outputs: Path) -> list[Path]:
    found: list[Path] = []
    if not outputs.is_dir():
        return found
    for child in sorted(outputs.iterdir()):
        if child.name == "real" or not child.is_dir():
            continue
        if (child / "manifest.json").is_file():
            found.append(child)
    real_root = outputs / "real"
    if real_root.is_dir():
        for child in sorted(real_root.iterdir()):
            if child.is_dir() and (child / "manifest.json").is_file():
                found.append(child)
    return found


def _has_synthseg_marker(structures: dict) -> bool:
    names = list(structures.keys())
    return any(marker in name for name in names for marker in SYNTHSEG_MARKERS)


def backfill_manifest(path: Path, *, dry_run: bool = False) -> str:
    """Return ``updated``, ``review``, ``skip``, or ``missing``."""
    if not path.is_file():
        return "missing"
    text = path.read_text(encoding="utf-8")
    manifest = json.loads(text)
    if manifest.get("anatomy_method"):
        return "skip"
    case_id = manifest.get("case_id") or path.parent.name
    structures = manifest.get("structures") or {}
    if _has_synthseg_marker(structures):
        print(f"review: {case_id}")
        return "review"
    if dry_run:
        print(f"would set anatomy_method=mni-atlas: {case_id}")
        return "updated"
    manifest["anatomy_method"] = "mni-atlas"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"updated: {case_id}")
    return "updated"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root (default: outputs/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change; write nothing",
    )
    args = parser.parse_args(argv)
    root = args.output if args.output is not None else default_output_root()
    case_dirs = _case_dirs(root)
    counts = {"updated": 0, "review": 0, "skip": 0, "missing": 0}
    for out_dir in case_dirs:
        status = backfill_manifest(out_dir / "manifest.json", dry_run=args.dry_run)
        counts[status] = counts.get(status, 0) + 1
    print(
        f"done: updated={counts['updated']} review={counts['review']} "
        f"skip={counts['skip']} missing={counts['missing']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
