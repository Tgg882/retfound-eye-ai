#!/usr/bin/env python
"""Create portable, relative-path release metadata.

This script does not modify the frozen manifest or any training result.  It
rewrites only the ``path`` / ``preprocessed_path`` columns in release copies so
that a teammate can place the source datasets under ``dataset/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from manifest_provenance import sha256_file, write_public_sidecar  # noqa: E402
from paths import path_key  # noqa: E402


def main() -> int:
    manifest = pd.read_csv(PROJECT_ROOT / "outputs" / "manifest" / "manifest_v1.csv", dtype={"path": str})
    prep = pd.read_csv(
        PROJECT_ROOT / "outputs" / "manifest" / "manifest_v0_clean_preprocessed.csv",
        dtype={"path": str, "preprocessed_path": str},
    )
    index = pd.read_csv(PROJECT_ROOT / "outputs" / "features" / "retfound_cfp_224_index.csv", dtype={"path": str})

    manifest["path"] = manifest["path"].map(path_key)
    prep["path"] = prep["path"].map(path_key)
    prep["preprocessed_path"] = prep["preprocessed_path"].map(path_key)
    index["path"] = index["path"].map(path_key)

    manifest_keys = set(manifest["path"])
    prep_keys = set(prep["path"])
    index_keys = set(index["path"])
    if not manifest_keys <= prep_keys:
        raise SystemExit(f"prep map missing {len(manifest_keys - prep_keys)} manifest keys")
    if not manifest_keys <= index_keys:
        raise SystemExit(f"feature index missing {len(manifest_keys - index_keys)} manifest keys")

    out = PROJECT_ROOT / "deliverables" / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest_v1.public.csv": out / "manifest_v1.public.csv",
        "manifest_v1_preprocessed.public.csv": out / "manifest_v1_preprocessed.public.csv",
        "retfound_cfp_224.public_index.csv": out / "retfound_cfp_224.public_index.csv",
    }
    manifest.to_csv(paths["manifest_v1.public.csv"], index=False, encoding="utf-8")
    prep[["path", "preprocessed_path"]].to_csv(paths["manifest_v1_preprocessed.public.csv"], index=False, encoding="utf-8")
    index.to_csv(paths["retfound_cfp_224.public_index.csv"], index=False, encoding="utf-8")
    sidecar = write_public_sidecar(
        PROJECT_ROOT / "outputs" / "manifest" / "manifest_v1.csv",
        paths["manifest_v1.public.csv"],
        source_display="outputs/manifest/manifest_v1.csv",
    )

    check = {
        "manifest_rows": int(len(manifest)),
        "prep_rows": int(len(prep)),
        "feature_index_rows": int(len(index)),
        "manifest_unique_keys": int(manifest["path"].nunique()),
        "absolute_path_rows": {
            name: int(df["path"].astype(str).str.match(r"^[A-Za-z]:[\\/]").sum())
            for name, df in {"manifest": manifest, "prep": prep, "index": index}.items()
        },
        "sha256": {**{name: sha256_file(path) for name, path in paths.items()},
                   sidecar.name: sha256_file(sidecar)},
    }
    (out / "public_release_check.json").write_text(json.dumps(check, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(check, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
