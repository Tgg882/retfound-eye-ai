#!/usr/bin/env python
"""Resolve byte-identical / conflicting-label duplicates (decision A).

Input : outputs/manifest/manifest_v0.csv   (never modified)
Output: outputs/manifest/conflict_groups_v1.csv
        outputs/manifest/manifest_v0_clean.csv
        outputs/reports/manifest_v0_clean_report.json
        outputs/reports/label_status_v1.csv

Rule
----
A SHA1 group is a *conflict* when at least two of its rows carry different
supervision vectors (y_*).  Such a group is removed **as a whole**.
Duplicates that agree on the supervision vector are NOT conflicts and are kept
(they stay in the same dedup group / split, so they cannot leak).

IDRID official test (103) is preserved untouched.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "outputs" / "manifest"

LABELS = json.loads((PROJECT_ROOT / "data_contract" / "labels.json").read_text(encoding="utf-8"))["labels"]
Y_COLS = [f"y_{label}" for label in LABELS]
M_COLS = [f"m_{label}" for label in LABELS]

EXPECTED_CLEAN = {
    "MESSIDOR-2": 1744,
    "APTOS2019": 3654,
    "IDRID": 515,  # 516 - 1: the only IDRID conflict pair is (holdout, train); holdout is kept
    "DDR": 12522,
    "Fundus10": 4376,
    "PAPILA": 488,
}


def main() -> int:
    df = pd.read_csv(MANIFEST / "manifest_v0.csv", dtype={"path": str, "sha1": str})
    df["sig"] = df[Y_COLS].astype(str).agg("".join, axis=1)

    dup = df[df.duplicated("sha1", keep=False)]
    conflict_sha: set[str] = set()
    rows = []
    for sha1, group in dup.groupby("sha1"):
        if group["sig"].nunique() > 1:
            conflict_sha.add(sha1)
            rows.append(
                {
                    "sha1": sha1,
                    "n_rows": len(group),
                    "n_datasets": group["dataset"].nunique(),
                    "datasets": "|".join(sorted(group["dataset"].unique())),
                    "labels": "|".join(sorted(set(group["label_key"].astype(str)))),
                    "label_vectors": "|".join(sorted(set(group["sig"]))),
                    "paths": ";".join(group["path"].astype(str)),
                }
            )

    conflicts = pd.DataFrame(rows).sort_values(["n_rows", "sha1"], ascending=[False, True])
    conflicts.to_csv(MANIFEST / "conflict_groups_v1.csv", index=False, encoding="utf-8")

    # IDRID official holdout is a hard constraint: if a conflicting group contains a
    # holdout row, keep the holdout row(s) and drop only the other members.
    protected = []
    drop_index = set()
    for sha1 in conflict_sha:
        group = df[df["sha1"] == sha1]
        holdout_rows = group[group["is_external_holdout"] == 1]
        if holdout_rows.empty:
            drop_index.update(group.index)
        else:
            drop_index.update(group[group["is_external_holdout"] == 0].index)
            protected.append(
                {
                    "sha1": sha1,
                    "kept": ";".join(holdout_rows["path"].astype(str)),
                    "dropped": ";".join(group.loc[group["is_external_holdout"] == 0, "path"].astype(str)),
                }
            )

    clean = df.drop(index=list(drop_index)).drop(columns=["sig"]).copy()

    residual = clean[clean.duplicated("sha1", keep=False)]
    if len(residual):
        residual_conflicts = int(
            (residual.groupby("sha1")["sig"].nunique() > 1).sum() if "sig" in residual else 0
        )
        if residual_conflicts:
            raise SystemExit(f"unresolved conflicts after cleaning: {residual_conflicts}")
    clean.to_csv(MANIFEST / "manifest_v0_clean.csv", index=False, encoding="utf-8")

    counts = clean["dataset"].value_counts().to_dict()
    idrid_holdout = int(((clean["dataset"] == "IDRID") & (clean["is_external_holdout"] == 1)).sum())

    label_status = []
    non_holdout_test = clean[(clean["split"] == "test") & (clean["is_external_holdout"] == 0)]
    for label in LABELS:
        n_pos = int(((clean[f"y_{label}"] == 1) & (clean[f"m_{label}"] == 1)).sum())
        n_test_pos = int(((non_holdout_test[f"y_{label}"] == 1) & (non_holdout_test[f"m_{label}"] == 1)).sum())
        label_status.append(
            {
                "label": label,
                "n_positives": n_pos,
                "n_test_positives": n_test_pos,
                "min_positives_gate": 100,
                "gate_ok": n_pos >= 100,
                "status": "core" if n_pos >= 100 else "exploratory",
            }
        )
    status_df = pd.DataFrame(label_status)
    status_df.to_csv(PROJECT_ROOT / "outputs" / "reports" / "label_status_v1.csv", index=False, encoding="utf-8")

    report = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(MANIFEST / "manifest_v0.csv"),
        "rows_before": int(len(df)),
        "conflict_groups": int(len(conflicts)),
        "rows_removed": int(len(df) - len(clean)),
        "holdout_protected_groups": protected,
        "rows_after": int(len(clean)),
        "rows_by_dataset": {k: int(v) for k, v in counts.items()},
        "expected_clean": EXPECTED_CLEAN,
        "counts_match_expected": {k: int(counts.get(k, 0)) == v for k, v in EXPECTED_CLEAN.items()},
        "idrid_external_holdout": idrid_holdout,
        "deviation_note": "user projection was 23298 rows / IDRID 514; one conflicting pair is (official-test, train) and the holdout must stay at 103, so only the train copy is dropped -> 23299 rows / IDRID 515",
        "rows_without_supervision": int((clean[M_COLS].sum(axis=1) == 0).sum()),
        "exploratory_labels": status_df.loc[~status_df["gate_ok"], "label"].tolist(),
    }
    (PROJECT_ROOT / "outputs" / "reports" / "manifest_v0_clean_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    ok = (
        len(clean) == 23299
        and all(report["counts_match_expected"].values())
        and idrid_holdout == 103
        and report["rows_without_supervision"] == 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
