"""Schema + hard-rule gates for the CFP-9 manifest (plan G0/G1/G2/G4).

Default target: outputs/manifest/manifest_v0.csv
Override with MANIFEST_PATH=... to check manifest_v1 once the embedding
dedup layer has rebuilt the splits.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(
    os.environ.get("MANIFEST_PATH", PROJECT_ROOT / "outputs" / "manifest" / "manifest_v0.csv")
)
LABELS = json.loads((PROJECT_ROOT / "data_contract" / "labels.json").read_text(encoding="utf-8"))["labels"]
Y_COLS = [f"y_{label}" for label in LABELS]
M_COLS = [f"m_{label}" for label in LABELS]

pa = pytest.importorskip("pandera.pandas")

FORBIDDEN = ("04_Fundus10", "06_Fundus_CFP", "_raw\\Fundus_CFP", "_raw/Fundus_CFP")
STAGE = "v0" if "v0" in MANIFEST_PATH.stem else "v1"


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.is_file():
        pytest.skip(f"manifest not built yet: {MANIFEST_PATH}")
    return pd.read_csv(MANIFEST_PATH)


def build_schema(stage: str = "v0"):
    flags = ["good", "usable", "reject"] + (["pending"] if stage == "v0" else [])
    columns = {
        "path": pa.Column(str, pa.Check(lambda s: s.is_unique), nullable=False),
        "dataset": pa.Column(str, nullable=False),
        "modality": pa.Column(str, pa.Check.isin(["cfp"]), nullable=False),
        "split": pa.Column(str, pa.Check.isin(["train", "val", "test"]), nullable=False),
        "group_id": pa.Column(str, nullable=False),
        "quality_score": pa.Column(
            float, pa.Check.in_range(0, 1), nullable=(stage == "v0")
        ),
        "quality_flag": pa.Column(str, pa.Check.isin(flags), nullable=False),
        "sample_weight": pa.Column(float, pa.Check.greater_than(0), nullable=False),
        "is_external_holdout": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
        **{
            col: pa.Column(int, pa.Check.isin([0, 1]), nullable=False)
            for col in Y_COLS + M_COLS
        },
    }
    return pa.DataFrameSchema(columns, strict=False)


def hard_rule_violations(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    bad_paths = df["path"].str.contains("|".join(re.escape(m) for m in FORBIDDEN), regex=True)
    if bad_paths.any():
        problems.append(f"forbidden source rows: {int(bad_paths.sum())}")
    if (df["source_label"].astype(str) == "Pterygium").any():
        problems.append("Pterygium rows present")
    if ((df["dataset"] == "DDR") & (df["source_grade"] == 5)).any():
        problems.append("DDR grade 5 rows present")

    y = df[Y_COLS].to_numpy()
    m = df[M_COLS].to_numpy()
    violations = int(((m == 0) & (y != 0)).sum())
    if violations:
        problems.append(f"mask violations (m=0, y=1): {violations}")
    if (m.sum(axis=1) == 0).any():
        problems.append(f"rows without supervision: {int((m.sum(axis=1) == 0).sum())}")

    cross = df.groupby("group_id")["split"].nunique()
    if (cross > 1).any():
        problems.append(f"groups crossing splits: {int((cross > 1).sum())}")
    return problems


def test_schema(manifest: pd.DataFrame) -> None:
    build_schema(STAGE).validate(manifest, lazy=True)


def test_hard_rules(manifest: pd.DataFrame) -> None:
    problems = hard_rule_violations(manifest)
    assert not problems, problems


def test_idrid_external_holdout(manifest: pd.DataFrame) -> None:
    idrid = manifest[manifest["dataset"] == "IDRID"]
    holdout = idrid[idrid["is_external_holdout"] == 1]
    assert len(holdout) == 103
    assert (holdout["split"] == "test").all()
    assert (idrid["is_external_holdout"] == 0).sum() == len(idrid) - 103


def test_holdout_never_leaks_into_train(manifest: pd.DataFrame) -> None:
    holdout_groups = set(manifest.loc[manifest["is_external_holdout"] == 1, "group_id"])
    leaked = manifest[(manifest["group_id"].isin(holdout_groups)) & (manifest["split"] != "test")]
    assert leaked.empty


def test_expected_dataset_counts(manifest: pd.DataFrame) -> None:
    counts = manifest["dataset"].value_counts().to_dict()
    if STAGE == "v0":
        expected = {
            "MESSIDOR-2": 1744,
            "APTOS2019": 3662,
            "IDRID": 516,
            "DDR": 12522,
            "Fundus10": 5318,
            "PAPILA": 488,
        }
        assert counts == expected
    else:
        report_path = PROJECT_ROOT / "outputs" / "reports" / f"manifest_{MANIFEST_PATH.stem.replace('manifest_', '')}_report.json"
        assert report_path.is_file(), report_path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert counts == report["rows_by_dataset"]


def test_hash_coverage(manifest: pd.DataFrame) -> None:
    assert manifest["sha1"].notna().all()
    failure_rate = manifest["phash"].isna().mean()
    assert failure_rate < 0.01, f"pHash decode failures: {failure_rate:.3%}"


def test_every_label_has_test_positives(manifest: pd.DataFrame) -> None:
    test = manifest[(manifest["split"] == "test") & (manifest["is_external_holdout"] == 0)]
    counts = {label: int(((test[f"y_{label}"] == 1) & (test[f"m_{label}"] == 1)).sum()) for label in LABELS}
    assert all(counts[label] > 0 for label in LABELS), counts


def test_v1_quality_fields(manifest: pd.DataFrame) -> None:
    if STAGE == "v0":
        pytest.skip("v0 quality is pending")
    assert manifest["quality_score"].notna().all()
    assert manifest["quality_flag"].isin(["good", "usable", "reject"]).all()
