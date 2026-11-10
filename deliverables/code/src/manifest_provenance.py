"""Provenance helpers for portable public copies of a frozen manifest.

A public manifest rewrites only the ``path`` column so that paths are relative to
``dataset/``.  Its file SHA therefore differs from the frozen manifest even
though every semantic column is unchanged.  A sidecar records and verifies this
transformation; training then compares checkpoints against the canonical frozen
manifest SHA without weakening the guard.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from paths import path_key

SIDECAR_SCHEMA = "cfp9-manifest-public-provenance-v1"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def semantic_sha256(df: pd.DataFrame) -> str:
    """Hash every column except the machine-specific ``path`` column."""
    cols = [c for c in df.columns if c != "path"]
    payload = df[cols].to_csv(index=False, lineterminator="\n").encode("utf-8")
    return _digest_bytes(payload)


def path_sequence_sha256(paths) -> str:
    """Hash the ordered path keys used to bind public rows to source rows."""
    payload = "\n".join(path_key(p) for p in paths).encode("utf-8")
    return _digest_bytes(payload)


def sidecar_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    if path.name.endswith(".public.csv"):
        return path.with_name(path.name[: -len(".public.csv")] + ".public.json")
    return path.with_suffix(path.suffix + ".provenance.json")


def public_sidecar_payload(source_manifest: str | Path, public_manifest: str | Path,
                           source_display: str | None = None) -> dict:
    source_path = Path(source_manifest)
    public_path = Path(public_manifest)
    source = pd.read_csv(source_path, dtype={"path": str})
    public = pd.read_csv(public_path, dtype={"path": str})
    if list(source.columns) != list(public.columns):
        raise ValueError("public manifest columns differ from the frozen source")
    if len(source) != len(public):
        raise ValueError(f"row count differs: source={len(source)} public={len(public)}")
    source_keys = [path_key(p) for p in source["path"]]
    if source_keys != list(public["path"].astype(str)):
        raise ValueError("public path column is not the ordered path_key projection of the source")
    source_sem = semantic_sha256(source)
    public_sem = semantic_sha256(public)
    if source_sem != public_sem:
        raise ValueError("semantic columns (all columns except path) differ")
    return {
        "schema": SIDECAR_SCHEMA,
        "source_manifest": source_display or source_path.name,
        "source_manifest_sha256": sha256_file(source_path),
        "public_manifest": public_path.name,
        "public_manifest_sha256": sha256_file(public_path),
        "row_count": int(len(public)),
        "path_column": "path",
        "transformation": "path column replaced by paths.path_key(path); all other columns unchanged",
        "semantic_columns_except_path_unchanged": True,
        "semantic_sha256_except_path": public_sem,
        "path_key_sequence_sha256": path_sequence_sha256(public["path"]),
    }


def write_public_sidecar(source_manifest: str | Path, public_manifest: str | Path,
                         source_display: str | None = None) -> Path:
    payload = public_sidecar_payload(source_manifest, public_manifest, source_display)
    out = sidecar_path(public_manifest)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def manifest_provenance(manifest_path: str | Path) -> dict:
    """Return file and canonical SHA256 values for a frozen or public manifest."""
    path = Path(manifest_path)
    file_sha = sha256_file(path)
    side = sidecar_path(path)
    if not side.is_file():
        return {
            "file_sha256": file_sha,
            "canonical_sha256": file_sha,
            "provenance_mode": "direct",
            "sidecar": None,
        }
    info = json.loads(side.read_text(encoding="utf-8"))
    if info.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"unsupported manifest sidecar schema: {info.get('schema')!r}")
    if str(info.get("public_manifest")) != path.name:
        raise ValueError(f"sidecar points to {info.get('public_manifest')!r}, not {path.name!r}")
    if info.get("public_manifest_sha256") != file_sha:
        raise ValueError("public manifest SHA256 does not match its provenance sidecar")
    canonical = str(info.get("source_manifest_sha256", ""))
    if len(canonical) != 64:
        raise ValueError("sidecar has no valid source_manifest_sha256")
    return {
        "file_sha256": file_sha,
        "canonical_sha256": canonical,
        "provenance_mode": "public_sidecar",
        "sidecar": side.name,
        "source_manifest": info.get("source_manifest"),
        "row_count": info.get("row_count"),
        "semantic_sha256_except_path": info.get("semantic_sha256_except_path"),
        "path_key_sequence_sha256": info.get("path_key_sequence_sha256"),
    }
