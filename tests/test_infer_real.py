"""Scene rebuild from existing pred_seg keeps the original anatomy method stamp."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recon.postprocess import PostConfig
from recon.scene import build_scene
from scripts.infer_real import rebuild_scene_study


def _write(path: Path, data, affine):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine), str(path))


def test_rebuild_scene_keeps_mni_atlas_method_and_sources():
    shape = (16, 16, 16)
    affine = np.eye(4, dtype=np.float64)
    parc = np.zeros(shape, dtype=np.int16)
    parc[1:5, 1:5, 1:5] = 16
    parc[1:7, 7:13, 1:7] = 101
    parc[7:13, 1:7, 1:7] = 3
    parc[7:13, 7:13, 1:7] = 42
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_root = root / "real"
        study_id = "study_atlasrebuild01"
        ready = out_root / study_id / "04_ready"
        study_out = out_root / study_id
        brain = np.ones(shape, dtype=np.float32)
        for suffix in ("t2f", "t1n", "t1c", "t2w"):
            _write(ready / f"{suffix}.nii.gz", brain, affine)
        probs = np.zeros((3,) + shape, dtype=np.float32)
        probs[2, 4:10, 4:10, 4:10] = 0.95
        probs[1, 5:9, 5:9, 5:9] = 0.95
        probs[0, 6:8, 6:8, 6:8] = 0.95
        image = np.ones((4,) + shape, dtype=np.float32)
        case = {
            "image": image,
            "affine": affine,
            "case_id": study_id,
            "case_dir": ready,
            "probs": probs,
        }
        config = PostConfig(
            threshold_et=0.5,
            threshold_tc=0.5,
            threshold_wt=0.5,
            min_voxels_et=0,
            min_voxels_tc=0,
            min_voxels_wt=0,
            et_min_voxels=0,
            brain_closing_iterations=1,
        )
        import recon.scene as scene_mod

        original = scene_mod._raw_parcellation

        def fake_raw(*args, **kwargs):
            return parc, "mni-atlas"

        scene_mod._raw_parcellation = fake_raw
        try:
            first = build_scene(
                case,
                anatomy=True,
                vessels=False,
                output_root=out_root,
                config=config,
                model="abhinav_v3",
            )
        finally:
            scene_mod._raw_parcellation = original

        assert first["anatomy_method"] == "mni-atlas"
        assert (study_out / "pred_seg.nii.gz").is_file()
        assert (study_out / "synthseg_raw.nii.gz").is_file()
        # Corrupt the stamp the way a bad synthseg= path would.
        bad = json.loads((study_out / "manifest.json").read_text(encoding="utf-8"))
        bad["anatomy_method"] = "synthseg"
        for name, entry in (bad.get("structures") or {}).items():
            if entry.get("source") == "atlas-estimated" and entry.get("group") != "Cortex":
                entry["source"] = "synthseg"
        (study_out / "manifest.json").write_text(json.dumps(bad, indent=2), encoding="utf-8")

        stats = rebuild_scene_study(
            study_id,
            out_root=out_root,
            model="abhinav_v3",
            anatomy=True,
            anatomy_method="atlas",
            vessels=False,
        )
        rebuilt = json.loads((study_out / "manifest.json").read_text(encoding="utf-8"))

    assert stats["anatomy_method"] == "mni-atlas"
    assert rebuilt["anatomy_method"] == "mni-atlas"
    assert rebuilt["structures"]["Brainstem"]["source"] == "atlas-estimated"
    assert rebuilt["structures"]["Left Cortex"]["source"] == "atlas-estimated"
    legend = {entry["name"]: entry for entry in rebuilt["legend"]}
    assert legend["Brainstem"]["source"] == "atlas-estimated"
