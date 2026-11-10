"""Mask/supervision consistency per dataset (plan 5.3 / hard rule 5.4)."""
from __future__ import annotations

import json
import os
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
DR_DATASETS = ["MESSIDOR-2", "APTOS2019", "IDRID", "DDR"]


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.is_file():
        pytest.skip(f"manifest not built yet: {MANIFEST_PATH}")
    return pd.read_csv(MANIFEST_PATH)


@pytest.mark.parametrize("dataset", DR_DATASETS)
def test_dr_datasets_only_supervise_dr(manifest: pd.DataFrame, dataset: str) -> None:
    sub = manifest[manifest["dataset"] == dataset]
    assert sub["m_dr"].eq(1).all()
    others = [c for c in M_COLS if c != "m_dr"]
    assert sub[others].eq(0).all().all()
    assert sub.loc[sub["source_grade"] == 0, "y_dr"].eq(0).all()
    assert sub.loc[sub["source_grade"] >= 1, "y_dr"].eq(1).all()


def test_papila_only_supervises_glaucoma(manifest: pd.DataFrame) -> None:
    sub = manifest[manifest["dataset"] == "PAPILA"]
    assert sub["m_glaucoma"].eq(1).all()
    assert sub["y_glaucoma"].isin([0, 1]).all()
    others = [c for c in M_COLS if c != "m_glaucoma"]
    assert sub[others].eq(0).all().all()


def test_fundus10_is_one_hot_and_fully_supervised(manifest: pd.DataFrame) -> None:
    sub = manifest[manifest["dataset"] == "Fundus10"]
    assert sub[M_COLS].eq(1).all().all()
    assert sub[Y_COLS].sum(axis=1).eq(1).all()
    assert sub.loc[sub["label_key"] == "normal", "y_normal"].eq(1).all()


def test_no_row_claims_normal_and_dr_simultaneously() -> None:
    """Guard: DR datasets must never set normal=1 (plan 5.4 rule 1)."""
    if not MANIFEST_PATH.is_file():
        pytest.skip("manifest not built yet")
    df = pd.read_csv(MANIFEST_PATH)
    dr_rows = df[df["dataset"].isin(DR_DATASETS)]
    assert dr_rows["y_normal"].eq(0).all()


def test_quality_is_pending_in_v0(manifest: pd.DataFrame) -> None:
    if "v0" not in MANIFEST_PATH.name:
        pytest.skip("v1 is expected to carry real quality flags")
    assert manifest["quality_flag"].eq("pending").all()
    assert manifest["quality_score"].isna().all()
