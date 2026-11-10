from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from manifest_provenance import manifest_provenance, sidecar_path, write_public_sidecar  # noqa: E402
from paths import path_key  # noqa: E402


def _write_source(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "path": [r"C:\old\dataset\01_MESSIDOR2\train\class_0\a.jpg", r"C:\old\dataset\02_APTOS2019\val\class_1\b.jpg"],
            "split": ["train", "val"],
            "y_dr": [1, 0],
            "m_dr": [1, 1],
        }
    )
    df.to_csv(path, index=False)
    return df


def test_public_sidecar_maps_to_canonical_sha(tmp_path):
    source = tmp_path / "manifest_v1.csv"
    df = _write_source(source)
    public = tmp_path / "manifest_v1.public.csv"
    df.assign(path=df["path"].map(path_key)).to_csv(public, index=False)

    side = write_public_sidecar(source, public)
    assert side == tmp_path / "manifest_v1.public.json"
    prov = manifest_provenance(public)
    assert prov["provenance_mode"] == "public_sidecar"
    assert prov["canonical_sha256"] == manifest_provenance(source)["canonical_sha256"]
    assert prov["file_sha256"] != prov["canonical_sha256"]


def test_sidecar_rejects_tampered_public_file(tmp_path):
    source = tmp_path / "manifest_v1.csv"
    df = _write_source(source)
    public = tmp_path / "manifest_v1.public.csv"
    df.assign(path=df["path"].map(path_key)).to_csv(public, index=False)
    write_public_sidecar(source, public)

    tampered = pd.read_csv(public)
    tampered.loc[0, "y_dr"] = 0
    tampered.to_csv(public, index=False)

    with pytest.raises(ValueError, match="public manifest SHA256"):
        manifest_provenance(public)


def test_sidecar_rejects_semantic_change(tmp_path):
    source = tmp_path / "manifest_v1.csv"
    df = _write_source(source)
    public = tmp_path / "manifest_v1.public.csv"
    changed = df.assign(path=df["path"].map(path_key))
    changed.loc[0, "split"] = "val"
    changed.to_csv(public, index=False)

    with pytest.raises(ValueError, match="semantic columns"):
        write_public_sidecar(source, public)


def test_sidecar_path_for_plain_manifest(tmp_path):
    assert sidecar_path(tmp_path / "manifest_v1.csv") == tmp_path / "manifest_v1.csv.provenance.json"
