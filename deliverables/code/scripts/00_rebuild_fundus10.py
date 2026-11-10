#!/usr/bin/env python
"""Rebuild Fundus10 rows from the raw ORIGINAL images (9 labels, Pterygium dropped).

Input : <DATA>/_raw/Fundus10/images/original/<class>/*.jpg
Output: outputs/manifest/fundus10_clean.csv

Notes
-----
* Pterygium is dropped (plan 1.2: out of scope).
* Class folders are read with os.scandir, so names that contain "[" "]"
  (e.g. "Central Serous Chorioretinopathy [Color Fundus]") work correctly.
* No split is produced here. 05_build_manifest.py owns the single
  group-aware 70/15/15 policy (plan 6.4 step 5 intentionally centralised).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths import DATA_ROOT  # noqa: E402
SRC_DIR = DATA_ROOT / "_raw" / "Fundus10" / "images" / "original"
OUT_CSV = PROJECT_ROOT / "outputs" / "manifest" / "fundus10_clean.csv"

LABELS = json.loads((PROJECT_ROOT / "data_contract" / "labels.json").read_text(encoding="utf-8"))["labels"]

CLASS_TO_LABEL = {
    "Central Serous Chorioretinopathy [Color Fundus]": "csc",
    "Diabetic Retinopathy": "dr",
    "Disc Edema": "disc_edema",
    "Glaucoma": "glaucoma",
    "Healthy": "normal",
    "Macular Scar": "macular_scar",
    "Myopia": "myopia",
    "Retinal Detachment": "retinal_det",
    "Retinitis Pigmentosa": "retinitis_pig",
}
DROP_CLASSES = {"Pterygium"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main() -> int:
    if not SRC_DIR.is_dir():
        raise SystemExit(f"missing Fundus10 source dir: {SRC_DIR}")

    rows = []
    per_class = {}
    unknown = []
    with os.scandir(SRC_DIR) as it:
        class_dirs = sorted((e for e in it if e.is_dir()), key=lambda e: e.name)
    for entry in class_dirs:
        cls = entry.name
        if cls in DROP_CLASSES:
            per_class[cls] = -1  # marker: dropped
            continue
        label_key = CLASS_TO_LABEL.get(cls)
        if label_key is None:
            unknown.append(cls)
            continue
        n = 0
        with os.scandir(entry.path) as it:
            for f in it:
                if not f.is_file() or Path(f.name).suffix.lower() not in IMAGE_EXTS:
                    continue
                rows.append(
                    {
                        "path": f.path,
                        "dataset": "Fundus10",
                        "source_label": cls,
                        "source_grade": "",
                        "label_key": label_key,
                        "patient_key": Path(f.name).stem,
                    }
                )
                n += 1
        per_class[cls] = n

    if unknown:
        raise SystemExit(f"unmapped Fundus10 classes: {unknown}")

    rows.sort(key=lambda r: r["path"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_CSV}  rows={len(rows)}")
    for cls, n in sorted(per_class.items()):
        flag = " (DROPPED)" if n < 0 else ""
        print(f"  {cls:<50} {max(n, 0):>5}{flag}")
    print(f"expected 5318 rows (5335 original - 17 Pterygium), got {len(rows)}")
    return 0 if len(rows) == 5318 else 1


if __name__ == "__main__":
    raise SystemExit(main())
