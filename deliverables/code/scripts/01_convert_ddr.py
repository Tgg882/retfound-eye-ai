#!/usr/bin/env python
"""Convert the DDR grading txt lists into clean rows (drop ungradable label 5).

Input : <DATA>/DDR-dataset/DR_grading/{train,valid,test}.txt  + flat jpg folder
Output: outputs/manifest/ddr_clean.csv

Rules (plan 5.2 / 6.5)
  label 0      -> dr = 0
  label 1..4   -> dr = 1
  label 5      -> dropped (ungradable), 1151 files
  group_id     -> filename stem.  NOTE: the first "-" token is NOT a patient id
                  (8250 files share the prefix "007"), so plan 6.5 step 5
                  cannot be applied; DDR has no usable patient id in this mirror.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths import DATA_ROOT  # noqa: E402
DDR_ROOT = DATA_ROOT / "DDR-dataset" / "DR_grading"
OUT_CSV = PROJECT_ROOT / "outputs" / "manifest" / "ddr_clean.csv"

LEGACY_SPLIT = {"train": "train", "valid": "val", "test": "test"}  # txt name -> legacy split
SPLIT_DIR = {"train": "train", "valid": "valid", "test": "test"}  # on-disk folder names


def main() -> int:
    rows = []
    dropped = {"label5": 0, "missing_file": 0}
    per_label = {}
    for txt_name, split in LEGACY_SPLIT.items():
        split_dir = DDR_ROOT / SPLIT_DIR[txt_name]
        txt_path = DDR_ROOT / f"{txt_name}.txt"
        if not txt_path.is_file():
            raise SystemExit(f"missing label file: {txt_path}")
        with txt_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise SystemExit(f"bad line in {txt_path}: {line!r}")
                fname, grade = parts[0], int(parts[1])
                if grade == 5:
                    dropped["label5"] += 1
                    continue
                if grade not in (0, 1, 2, 3, 4):
                    raise SystemExit(f"unexpected DDR grade {grade} in {line!r}")
                img = split_dir / fname
                if not img.is_file():
                    dropped["missing_file"] += 1
                    continue
                per_label[grade] = per_label.get(grade, 0) + 1
                rows.append(
                    {
                        "path": str(img),
                        "dataset": "DDR",
                        "source_label": str(grade),
                        "source_grade": grade,
                        "label_key": "dr",
                        "patient_key": Path(fname).stem,
                        "split_legacy": split,
                    }
                )

    if dropped["missing_file"]:
        raise SystemExit(f"DDR rows without image file: {dropped['missing_file']}")

    rows.sort(key=lambda r: r["path"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT_CSV}  rows={len(rows)}  dropped_label5={dropped['label5']}")
    print("per grade:", dict(sorted(per_label.items())))
    print(f"expected 12522 rows, got {len(rows)}")
    return 0 if len(rows) == 12522 else 1


if __name__ == "__main__":
    raise SystemExit(main())
