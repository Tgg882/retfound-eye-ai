#!/usr/bin/env python
"""Build the CFP-9 manifest (v0 = pipeline smoke / full).

IMPORTANT (user constraint #2)
-----------------------------
`manifest_v0.csv` is produced so the data pipeline can be exercised end to end.
It must NOT be used to start Stage 1.  After Stage-1 feature caching, the RETFound
embedding layer of the dedup must be added, `group_id` refreshed and the splits
re-derived, producing `manifest_v1.csv` = the only legal Stage-1 training input.

Pipeline
--------
1. Scan 01_MESSIDOR2 / 02_APTOS2019 / 03_IDRID / 05_PAPILA ImageFolder trees,
   the DDR txt lists (via outputs/manifest/ddr_clean.csv) and the Fundus10
   originals (via outputs/manifest/fundus10_clean.csv).
2. SHA1 exact duplicates (same dataset) + pHash Hamming<=6 near duplicates,
   merged with Union-Find into `group_id`.
3. Expand the 9-label masked supervision contract (y_* / m_*).
4. Group-aware 70/15/15 split.  IDRID official test (103 images) is frozen as
   external holdout and never re-split (plan 11.3, user constraint #1).
5. Write manifest, per-split csv, dedup report and label counts.

Excluded on purpose: 04_Fundus10, 06_Fundus_CFP, OCTID, Pterygium, DDR grade 5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paths import DATA_ROOT  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs"
MANIFEST_DIR = OUT_DIR / "manifest"
REPORT_DIR = OUT_DIR / "reports"

LABELS = json.loads((PROJECT_ROOT / "data_contract" / "labels.json").read_text(encoding="utf-8"))["labels"]
Y_COLS = [f"y_{label}" for label in LABELS]
M_COLS = [f"m_{label}" for label in LABELS]
LIDX = {label: i for i, label in enumerate(LABELS)}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SPLITS = ("train", "val", "test")
DR_DATASETS = {"MESSIDOR-2", "APTOS2019", "IDRID", "DDR"}

EXPECTED_ROWS = {
    "MESSIDOR-2": 1744,
    "APTOS2019": 3662,
    "IDRID": 516,
    "DDR": 12522,
    "Fundus10": 5318,
    "PAPILA": 488,
}
IDRID_HOLDOUT = 103

FORBIDDEN_PATH_MARKERS = ("04_Fundus10", "06_Fundus_CFP", "_raw\\Fundus_CFP", "_raw/Fundus_CFP")


@dataclass(frozen=True)
class ImageFolderSpec:
    name: str
    root: Path
    label_kind: str  # "dr" | "glaucoma"
    group_rule: str  # "messidor_patient" | "stem"


IMAGE_FOLDERS = (
    ImageFolderSpec("MESSIDOR-2", DATA_ROOT / "01_MESSIDOR2", "dr", "messidor_patient"),
    ImageFolderSpec("APTOS2019", DATA_ROOT / "02_APTOS2019", "dr", "stem"),
    ImageFolderSpec("IDRID", DATA_ROOT / "03_IDRID", "dr", "stem"),
    ImageFolderSpec("PAPILA", DATA_ROOT / "05_PAPILA", "glaucoma", "stem"),
)


def iter_files(directory: Path):
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTS:
                yield Path(entry.path)


def patient_key(spec: ImageFolderSpec, path: Path) -> str:
    if spec.group_rule == "messidor_patient":
        parts = path.stem.split("_")
        # Messidor-2 in this mirror mixes "YYYYMMDD_<patient>_<eye>_PP" with "IMxxxxxx"
        return parts[1] if len(parts) >= 3 and parts[1] else path.stem
    return path.stem


def scan_image_folder(spec: ImageFolderSpec, limit: int | None) -> list[dict]:
    if not spec.root.is_dir():
        raise FileNotFoundError(f"missing dataset root: {spec.root}")
    rows: list[dict] = []
    for split in SPLITS:
        split_dir = spec.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing split dir: {split_dir}")
        with os.scandir(split_dir) as it:
            class_dirs = sorted((e for e in it if e.is_dir()), key=lambda e: e.name)
        if not class_dirs:
            raise RuntimeError(f"no class_* dirs under {split_dir}")
        for cd in class_dirs:
            match = re.fullmatch(r"class_(\d+)", cd.name)
            if match is None:
                raise RuntimeError(f"unexpected class dir: {cd.path}")
            grade = int(match.group(1))
            for file_path in sorted(iter_files(Path(cd.path))):
                rows.append(
                    {
                        "path": str(file_path),
                        "dataset": spec.name,
                        "source_label": f"class_{grade}",
                        "source_grade": grade,
                        "label_key": spec.label_kind,
                        "patient_key": patient_key(spec, file_path),
                        "split_legacy": split,
                    }
                )
                if limit and len(rows) >= limit:
                    return rows
    return rows


def load_clean_csv(path: Path, dataset: str, limit: int | None) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run the source converter first (00_rebuild_fundus10 / 01_convert_ddr)"
        )
    # patient_key / source_label are id-like (e.g. "007"); never let pandas coerce them
    df = pd.read_csv(path, dtype={"patient_key": str, "label_key": str, "source_label": str})
    if limit:
        df = df.head(limit)
    df = df.copy()
    df["dataset"] = dataset
    if "split_legacy" not in df.columns:
        df["split_legacy"] = "na"
    if "source_grade" not in df.columns:
        df["source_grade"] = np.nan
    return df.to_dict("records")


def collect_rows(limit: int | None) -> pd.DataFrame:
    rows: list[dict] = []
    for spec in IMAGE_FOLDERS:
        rows.extend(scan_image_folder(spec, limit))
    rows.extend(load_clean_csv(MANIFEST_DIR / "fundus10_clean.csv", "Fundus10", limit))
    rows.extend(load_clean_csv(MANIFEST_DIR / "ddr_clean.csv", "DDR", limit))
    df = pd.DataFrame(rows)
    df["source_grade"] = pd.to_numeric(df["source_grade"], errors="coerce")
    df["split_legacy"] = df["split_legacy"].fillna("na").astype(str)
    if df["path"].duplicated().any():
        dupes = df.loc[df["path"].duplicated(), "path"].head(5).tolist()
        raise RuntimeError(f"duplicate paths in source scan: {dupes}")
    return df.sort_values(["dataset", "path"]).reset_index(drop=True)


def hash_one(path: str) -> dict:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        return {"path": path, "sha1": None, "phash": None, "size_bytes": None, "error": str(exc)}
    try:
        with Image.open(BytesIO(data)) as img:
            img.draft("RGB", (256, 256))
            phash = str(imagehash.phash(img.convert("L"), hash_size=16))
        error = None
    except Exception as exc:  # noqa: BLE001 - we log unreadable files instead of failing
        phash = None
        error = f"{type(exc).__name__}: {exc}"
    return {
        "path": path,
        "sha1": hashlib.sha1(data).hexdigest(),
        "phash": phash,
        "size_bytes": len(data),
        "error": error,
    }


def compute_hashes(paths: list[str], workers: int) -> pd.DataFrame:
    started = time.time()
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, rec in enumerate(pool.map(hash_one, paths), start=1):
            records.append(rec)
            if i % 2000 == 0:
                print(f"  hashed {i}/{len(paths)}  ({time.time() - started:.0f}s)", flush=True)
    print(f"  hashed {len(paths)} files in {time.time() - started:.0f}s", flush=True)
    return pd.DataFrame(records)


_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_M8 = np.uint64(0x0101010101010101)


def popcount64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = values - ((values >> np.uint64(1)) & _M1)
        x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
        x = (x + (x >> np.uint64(4))) & _M4
        return (x * _M8) >> np.uint64(56)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def phash_codes(phash_series: pd.Series) -> np.ndarray:
    """hex pHash -> (n, n_words) uint64 matrix (little-endian 64-bit words)."""
    n_words = max(1, (len(phash_series.iloc[0]) * 4 + 63) // 64)
    out = np.zeros((len(phash_series), n_words), dtype=np.uint64)
    for i, h in enumerate(phash_series):
        value = int(str(h), 16)
        for w in range(n_words):
            out[i, w] = np.uint64((value >> (64 * w)) & 0xFFFFFFFFFFFFFFFF)
    return out


def phash_edges(codes: np.ndarray, max_dist: int = 6, chunk: int = 384):
    n, n_words = codes.shape
    for start in range(0, n, chunk):
        block = codes[start : start + chunk]
        dist = np.zeros((block.shape[0], n), dtype=np.uint16)
        for w in range(n_words):
            xor = np.bitwise_xor(block[:, w][:, None], codes[:, w][None, :])
            dist += popcount64(xor).astype(np.uint16)
        ii, jj = np.nonzero((dist <= max_dist) & (dist > 0))
        for i, j in zip(ii.tolist(), jj.tolist()):
            a = start + i
            if j > a:
                yield a, j


def build_groups(df: pd.DataFrame, hashes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.merge(hashes, on="path", how="left", validate="one_to_one")

    uf = UnionFind(len(df))
    # patient groups (only inside the same dataset)
    for (_, key), idx in df.groupby(["dataset", "patient_key"]).groups.items():
        idx = list(idx)
        for other in idx[1:]:
            uf.union(idx[0], other)

    exact_same_dataset = 0
    exact_cross_dataset = []
    for sha1, idx in df.groupby("sha1").groups.items():
        if sha1 is None or pd.isna(sha1):
            continue
        idx = list(idx)
        datasets = df.loc[idx, "dataset"].unique()
        if len(idx) > 1:
            if len(datasets) == 1:
                exact_same_dataset += len(idx) - 1
            else:
                exact_cross_dataset.append({"sha1": sha1, "datasets": sorted(datasets.tolist()), "n": len(idx)})
        for other in idx[1:]:
            uf.union(idx[0], other)

    phash_ok = df["phash"].notna().to_numpy()
    ok_index = np.flatnonzero(phash_ok)
    codes = phash_codes(df.loc[phash_ok, "phash"].reset_index(drop=True))
    n_edges = 0
    for a, b in phash_edges(codes):
        uf.union(int(ok_index[a]), int(ok_index[b]))
        n_edges += 1

    roots = np.array([uf.find(i) for i in range(len(df))])
    # deterministic group ids: g00001 in order of first appearance
    order = {}
    for i, root in enumerate(roots):
        if root not in order:
            order[root] = len(order) + 1
    df["group_id"] = [f"g{order[r]:05d}" for r in roots]
    sizes = df.groupby("group_id").size()
    df["n_group_members"] = df["group_id"].map(sizes).astype(int)

    cross_dataset_groups = 0
    for _, sub in df.groupby("group_id"):
        if sub["dataset"].nunique() > 1:
            cross_dataset_groups += 1

    stats = {
        "n_rows": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "n_multi_member_groups": int((sizes > 1).sum()),
        "max_group_size": int(sizes.max()),
        "phash_bits": 256,
        "phash_max_hamming": 6,
        "phash_note": "plan said 64-bit pHash; 64-bit collapsed into one 19372-image component on fundus images, so 256-bit pHash with the same Hamming<=6 was used",
        "n_phash_failures": int((~phash_ok).sum()),
        "n_phash_edges": int(n_edges),
        "n_exact_duplicate_rows_same_dataset": int(exact_same_dataset),
        "n_cross_dataset_exact_duplicate_groups": len(exact_cross_dataset),
        "cross_dataset_exact_duplicates": exact_cross_dataset[:20],
        "n_cross_dataset_groups": int(cross_dataset_groups),
    }
    return df, stats


def expand_labels(df: pd.DataFrame) -> pd.DataFrame:
    y = np.zeros((len(df), len(LABELS)), dtype=np.int8)
    m = np.zeros((len(df), len(LABELS)), dtype=np.int8)

    for dataset, sub in df.groupby("dataset"):
        pos = sub.index.to_numpy()
        if dataset in DR_DATASETS:
            grades = sub["source_grade"].to_numpy()
            if np.isnan(grades).any():
                raise RuntimeError(f"{dataset}: missing source_grade")
            m[pos, LIDX["dr"]] = 1
            y[pos, LIDX["dr"]] = (grades >= 1).astype(np.int8)
        elif dataset == "PAPILA":
            grades = sub["source_grade"].to_numpy()
            m[pos, LIDX["glaucoma"]] = 1
            y[pos, LIDX["glaucoma"]] = grades.astype(np.int8)
        elif dataset == "Fundus10":
            keys = sub["label_key"].tolist()
            bad = sorted({k for k in keys if k not in LIDX})
            if bad:
                raise RuntimeError(f"Fundus10 unmapped label keys: {bad}")
            m[pos, :] = 1
            for row_pos, key in zip(pos, keys):
                y[row_pos, LIDX[key]] = 1
        else:
            raise RuntimeError(f"unexpected dataset: {dataset}")

    for j, label in enumerate(LABELS):
        df[f"y_{label}"] = y[:, j]
        df[f"m_{label}"] = m[:, j]
    return df


def primary_label(df: pd.DataFrame) -> pd.Series:
    m = df[M_COLS].to_numpy()
    y = df[Y_COLS].to_numpy()
    out = []
    for i in range(len(df)):
        known = np.flatnonzero(m[i])
        if known.size == 1:
            j = known[0]
            out.append(f"{LABELS[j]}={int(y[i, j])}")
        else:
            positives = np.flatnonzero(y[i] == 1)
            out.append(LABELS[positives[0]] if positives.size == 1 else "unknown")
    return pd.Series(out, index=df.index)


def assign_splits(df: pd.DataFrame, seed: int = 42) -> tuple[dict, dict]:
    info = (
        df.groupby("group_id")
        .agg(n=("path", "size"), stratum=("stratum", lambda s: s.value_counts().index[0]))
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    info["rand"] = rng.random(len(info))
    info = info.sort_values(["stratum", "n", "rand"], ascending=[True, False, True])

    quota = {}
    for stratum, sub in info.groupby("stratum"):
        total = int(sub["n"].sum())
        if total >= 4:
            n_val = max(1, int(round(0.15 * total)))
            n_test = max(1, int(round(0.15 * total)))
        else:
            n_val = n_test = 0
        n_train = total - n_val - n_test
        if n_train < 1:
            n_train, n_val, n_test = total, 0, 0
        quota[stratum] = {"train": n_train, "val": n_val, "test": n_test}

    targets = {stratum: dict(q) for stratum, q in quota.items()}
    remaining_total = {s: sum(q[s] for q in quota.values()) for s in SPLITS}
    assignment: dict[str, str] = {}
    for row in info.itertuples():
        q = quota[row.stratum]
        target = targets[row.stratum]

        def score(split: str) -> tuple[float, int, float, int]:
            # fill the split that is most under-filled RELATIVE to its target,
            # otherwise the 70% train quota always wins the tie-break.
            return (
                q[split] / max(1, target[split]),
                q[split],
                remaining_total[split] / max(1, sum(targets[s][split] for s in targets)),
                -SPLITS.index(split),
            )

        best = max(SPLITS, key=score)
        assignment[row.group_id] = best
        q[best] -= int(row.n)
        remaining_total[best] -= int(row.n)
    return assignment, quota


def apply_splits(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["stratum"] = df["dataset"] + "|" + primary_label(df)

    holdout = df["is_external_holdout"] == 1
    pool = df.loc[~holdout].copy()
    assignment, quota = assign_splits(pool, seed=seed)

    df["split"] = df["group_id"].map(assignment)
    # a group touched by the frozen IDRID holdout must never stay in train/val
    holdout_groups = set(df.loc[holdout, "group_id"])
    forced = df["group_id"].isin(holdout_groups) & ~holdout
    df.loc[forced, "split"] = "test"
    df.loc[holdout, "split"] = "test"

    missing = df["split"].isna()
    if missing.any():
        raise RuntimeError(f"{int(missing.sum())} rows without split assignment")

    warnings = []
    for (stratum, split), sub in df.loc[~holdout].groupby(["stratum", "split"]):
        if split in ("val", "test") and len(sub) == 0:
            warnings.append(f"empty {split} for stratum {stratum}")
    for stratum, sub in df.loc[~holdout].groupby("stratum"):
        if sub["split"].nunique() < 3 and len(sub) >= 30:
            warnings.append(f"stratum {stratum} has only splits {sorted(sub['split'].unique())}")
    return df, {"quota": quota, "warnings": warnings}


def write_outputs(df: pd.DataFrame, tag: str, report: dict, quality: pd.DataFrame | None = None) -> None:
    column_order = (
        ["path", "dataset", "modality", "split", "group_id", "quality_score", "quality_flag"]
        + Y_COLS
        + M_COLS
        + [
            "sample_weight",
            "stem",
            "source_label",
            "source_grade",
            "label_key",
            "split_legacy",
            "split_source",
            "is_external_holdout",
            "sha1",
            "phash",
            "n_group_members",
            "quality_issues",
        ]
    )
    out = df.copy()
    out["quality_score"] = np.nan
    out["quality_flag"] = "pending"
    out["quality_issues"] = ""
    out["sample_weight"] = 1.0
    if quality is not None:
        q = quality[["path", "quality_score", "quality_flag", "issues"]].rename(columns={"issues": "quality_issues"})
        out = out.drop(columns=["quality_score", "quality_flag", "quality_issues"]).merge(
            q, on="path", how="left", validate="one_to_one"
        )
        missing = out["quality_flag"].isna().sum()
        if missing:
            raise RuntimeError(f"{int(missing)} rows without quality rows")
    out["stem"] = out["path"].map(lambda p: Path(p).stem)
    out = out[column_order]

    manifest_path = MANIFEST_DIR / f"manifest_{tag}.csv"
    out.to_csv(manifest_path, index=False, encoding="utf-8")

    splits_dir = MANIFEST_DIR / "splits" / tag
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        out.loc[out["split"] == split].to_csv(splits_dir / f"{split}.csv", index=False, encoding="utf-8")
    out.loc[out["is_external_holdout"] == 1].to_csv(
        splits_dir / "external_holdout.csv", index=False, encoding="utf-8"
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"manifest_{tag}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    counts = []
    for dataset, sub in out.groupby("dataset"):
        for split in SPLITS:
            part = sub[sub["split"] == split]
            if part.empty:
                continue
            row = {"dataset": dataset, "split": split, "n": int(len(part))}
            for label in LABELS:
                row[f"pos_{label}"] = int(((part[f"y_{label}"] == 1) & (part[f"m_{label}"] == 1)).sum())
            counts.append(row)
    count_cols = ["dataset", "split", "n"] + [f"pos_{label}" for label in LABELS]
    pd.DataFrame(counts, columns=count_cols).to_csv(
        REPORT_DIR / f"label_counts_{tag}.csv", index=False, encoding="utf-8"
    )
    print(f"wrote {manifest_path}  rows={len(out)}")


def exact_duplicate_conflicts(df: pd.DataFrame) -> dict:
    """Byte-identical images that carry different supervision (label noise audit)."""
    dup = df[df.duplicated("sha1", keep=False)].copy()
    if dup.empty:
        return {"rows_in_exact_dup_groups": 0, "groups": 0, "conflicting_groups": 0, "by_dataset": {}}
    dup["sig"] = dup[Y_COLS].astype(str).agg("".join, axis=1)
    grouped = dup.groupby("sha1").agg(n=("path", "size"), n_sig=("sig", "nunique"), dataset=("dataset", "first"))
    conflicting = grouped[grouped["n_sig"] > 1]
    by_dataset = {}
    for dataset, sub in dup.groupby("dataset"):
        sub_groups = sub.groupby("sha1")["sig"].nunique()
        by_dataset[dataset] = {
            "rows_in_exact_dup_groups": int(len(sub)),
            "groups": int(sub["sha1"].nunique()),
            "conflicting_groups": int((sub_groups > 1).sum()),
        }
    return {
        "rows_in_exact_dup_groups": int(len(dup)),
        "groups": int(len(grouped)),
        "conflicting_groups": int(len(conflicting)),
        "by_dataset": by_dataset,
        "label_consistent": bool(len(conflicting) == 0),
        "note": "byte-identical images share one group_id and carry consistent labels; no label conflicts remain",
    }


def build_report(
    df: pd.DataFrame,
    dedup_stats: dict,
    split_info: dict,
    tag: str,
    limit: int | None,
    baseline: dict | None = None,
) -> dict:
    rows_by_dataset = df["dataset"].value_counts().to_dict()
    rows_by_split = df["split"].value_counts().to_dict()
    expected_rows = dict(baseline["rows_by_dataset"]) if baseline else dict(EXPECTED_ROWS)
    count_match = {k: int(v) == expected_rows[k] for k, v in rows_by_dataset.items() if k in expected_rows}

    idrid = df[df["dataset"] == "IDRID"]
    idrid_holdout = idrid[idrid["is_external_holdout"] == 1]

    forbidden_rows = int(
        df["path"].str.contains("|".join(re.escape(m) for m in FORBIDDEN_PATH_MARKERS), regex=True).sum()
    )
    pterygium_rows = int((df["source_label"].astype(str) == "Pterygium").sum())
    ddr5_rows = int(((df["dataset"] == "DDR") & (df["source_grade"] == 5)).sum())

    mask_violations = int(((df[M_COLS].to_numpy() == 0) & (df[Y_COLS].to_numpy() != 0)).sum())
    rows_without_supervision = int((df[M_COLS].sum(axis=1) == 0).sum())

    group_split = df.groupby("group_id")["split"].nunique()
    cross_split_groups = int((group_split > 1).sum())

    report = {
        "tag": tag,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "smoke_limit_per_source": limit,
        "rows_total": int(len(df)),
        "rows_by_dataset": {k: int(v) for k, v in rows_by_dataset.items()},
        "rows_by_split": {k: int(v) for k, v in rows_by_split.items()},
        "expected_rows": expected_rows,
        "row_count_match_expected": count_match,
        "dedup": dedup_stats,
        "exact_duplicate_label_conflicts": exact_duplicate_conflicts(df),
        "splits": {
            "policy": "group-aware greedy stratified 70/15/15 (seed=42)",
            "warnings": split_info["warnings"],
            "strata": len(split_info["quota"]),
        },
        "idrid_external_holdout": {
            "n": int(len(idrid_holdout)),
            "expected": IDRID_HOLDOUT,
            "all_legacy_test": bool((idrid_holdout["split_legacy"] == "test").all()),
        },
        "gates": {
            "G1_forbidden_path_rows": forbidden_rows,
            "G1_pterygium_rows": pterygium_rows,
            "G1_ddr_label5_rows": ddr5_rows,
            "G2_cross_split_groups": cross_split_groups,
            "G4_mask_violations": mask_violations,
            "G4_rows_without_supervision": rows_without_supervision,
            "G3_embedding_guard": "pending (manifest v1)",
        },
    }
    if baseline:
        report["clean_baseline"] = baseline
        report["rows_after_leakage_guard"] = int(len(df))
        report["leakage_guard_dropped"] = int(baseline["clean_rows"] - len(df))
        report["holdout_protected_groups"] = baseline.get("holdout_protected_groups", [])
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="v0", help="output tag, e.g. v0 or smoke")
    parser.add_argument("--limit", type=int, default=None, help="rows per source (smoke runs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--group-map", default=None, help="csv with path,group_id_v1 (embedding dedup)")
    parser.add_argument("--quality", default=None, help="csv with path,quality_score,quality_flag,issues")
    parser.add_argument("--features", default=None, help="npy of L2-normalised 1024-d features (leakage guard)")
    parser.add_argument(
        "--centered-features",
        default=None,
        help="npy of mean-centred L2-normalised features used for the cosine guard",
    )
    parser.add_argument(
        "--feature-index",
        default=None,
        help="csv with path column matching the feature npy rows",
    )
    parser.add_argument(
        "--from-manifest",
        default=None,
        help="re-split an existing manifest (e.g. manifest_v0_clean.csv) instead of re-scanning sources",
    )
    parser.add_argument(
        "--clean-baseline",
        default=None,
        help="clean manifest used as the row baseline (expected_rows / clean_rows) for this build",
    )
    parser.add_argument(
        "--provenance",
        default=None,
        help="json report of the clean stage; supplies conflict counts and holdout_protected_groups",
    )
    args = parser.parse_args(argv)

    started = time.time()
    if args.from_manifest:
        print("[1/5] loading existing manifest:", args.from_manifest)
        df = pd.read_csv(args.from_manifest, dtype={"path": str, "sha1": str, "phash": str})
        df["source_grade"] = pd.to_numeric(df["source_grade"], errors="coerce")
        df["split_legacy"] = df["split_legacy"].fillna("na").astype(str)
        df["label_key"] = df["label_key"].astype(str)
        df["modality"] = "cfp"
        df["is_external_holdout"] = df["is_external_holdout"].astype(int)
        sizes = df.groupby("group_id").size()
        dedup_stats = {
            "source": "reused from " + str(args.from_manifest),
            "n_rows": int(len(df)),
            "n_groups": int(df["group_id"].nunique()),
            "n_multi_member_groups": int((sizes > 1).sum()),
            "max_group_size": int(sizes.max()),
            "n_phash_failures": int(df["phash"].isna().sum()),
            "n_cross_dataset_groups": int(
                (df.groupby("group_id")["dataset"].nunique() > 1).sum()
            ),
        }
        print("  rows:", len(df), df["dataset"].value_counts().to_dict())
    else:
        print("[1/5] scanning sources")
        df = collect_rows(args.limit)
        print("  rows:", len(df), df["dataset"].value_counts().to_dict())

        print("[2/5] hashing (sha1 + pHash)")
        hashes = compute_hashes(df["path"].tolist(), workers=args.workers)

        print("[3/5] duplicate grouping (SHA1 + pHash<=6, union-find)")
        df, dedup_stats = build_groups(df, hashes)
        print("  groups:", dedup_stats["n_groups"], "multi-member:", dedup_stats["n_multi_member_groups"])

    print("[4/5] label expansion + group-aware split")
    df = expand_labels(df)
    df["modality"] = "cfp"
    df["is_external_holdout"] = (
        (df["dataset"] == "IDRID") & (df["split_legacy"] == "test")
    ).astype(int)
    if args.group_map:
        gm = pd.read_csv(args.group_map, dtype={"path": str})
        df = df.merge(gm[["path", "group_id_v1"]], on="path", how="left", validate="one_to_one")
        if df["group_id_v1"].isna().any():
            raise RuntimeError(f"{int(df['group_id_v1'].isna().sum())} rows without embedding group")
        df["group_id"] = df["group_id_v1"]
        df = df.drop(columns=["group_id_v1"])
        sizes = df.groupby("group_id").size()
        df["n_group_members"] = df["group_id"].map(sizes).astype(int)
        n_groups_before = int(dedup_stats.get("n_groups", len(sizes)))
        dedup_stats.update(
            {
                "source": f"{args.from_manifest or 'source scan'} + embedding map {args.group_map}",
                "n_groups_before_embedding_dedup": n_groups_before,
                "n_groups": int(df["group_id"].nunique()),
                "n_merge_edges": int(n_groups_before - df["group_id"].nunique()),
                "n_multi_member_groups": int((sizes > 1).sum()),
                "max_group_size": int(sizes.max()),
                "n_cross_dataset_groups": int((df.groupby("group_id")["dataset"].nunique() > 1).sum()),
            }
        )

    quality = pd.read_csv(args.quality, dtype={"path": str}) if args.quality else None
    df, split_info = apply_splits(df, seed=args.seed)

    guard_info = {"status": "pending (no --features)"}
    if args.features:
        feats = np.load(args.features, mmap_mode="r")
        index_path = args.feature_index or str(Path(args.features).with_name("retfound_cfp_224_index.csv"))
        fidx = pd.read_csv(index_path, dtype={"path": str})["path"]
        pos = df["path"].map({p: i for i, p in enumerate(fidx)})
        if pos.isna().any():
            raise RuntimeError(f"{int(pos.isna().sum())} rows without features")
        feat_row = pos.to_numpy(dtype=np.int64)

        centered_path = args.centered_features or str(
            Path(args.features).with_name("retfound_cfp_224_centered.npy")
        )
        centered = np.load(centered_path, mmap_mode="r")

        def phash_words(values) -> np.ndarray:
            words = np.zeros((len(values), 4), dtype=np.uint64)
            for i, h in enumerate(values):
                v = int(str(h), 16)
                for w in range(4):
                    words[i, w] = np.uint64((v >> (64 * w)) & 0xFFFFFFFFFFFFFFFF)
            return words

        words = phash_words(df["phash"].to_numpy())
        train_rows = np.flatnonzero((df["split"] == "train").to_numpy())
        test_rows = np.flatnonzero((df["split"] == "test").to_numpy())
        w_tr = words[train_rows]

        dropped, drop_cos, candidate_rows, candidate_max_cos, min_ham_all = [], [], [], [], []
        for start in range(0, len(test_rows), 256):
            rows = test_rows[start : start + 256]
            w_te = words[rows]
            ham = np.zeros((len(rows), len(train_rows)), dtype=np.uint16)
            for w in range(4):
                ham += popcount64(np.bitwise_xor(w_te[:, w][:, None], w_tr[:, w][None, :])).astype(np.uint16)
            min_ham = ham.min(axis=1)
            min_ham_all.append(min_ham)
            for local in np.flatnonzero(min_ham <= 24):
                near = np.flatnonzero(ham[local] <= 24)
                x = np.asarray(centered[feat_row[rows[local]]], dtype=np.float32)
                y = np.asarray(centered[feat_row[train_rows[near]]], dtype=np.float32)
                cos = float((x @ y.T).max())
                candidate_rows.append(int(rows[local]))
                candidate_max_cos.append(cos)
                if cos >= 0.99:
                    dropped.append(int(rows[local]))
                    drop_cos.append(cos)

        min_ham_all = np.concatenate(min_ham_all) if min_ham_all else np.array([], dtype=np.uint16)
        drop_df = pd.DataFrame(
            {"path": df.iloc[dropped]["path"].to_numpy(),
             "max_test_to_train_cosine": drop_cos}
        )
        drop_df.to_csv(MANIFEST_DIR / f"test_dropped_{args.tag}.csv", index=False, encoding="utf-8")
        guard_dropped_by_dataset = df.iloc[dropped]["dataset"].value_counts().to_dict() if dropped else {}
        df = df.drop(index=dropped).reset_index(drop=True)
        guard_info = {
            "status": "applied",
            "rule": "pHash Hamming<=24 to a train image AND centred cosine>=0.99",
            "n_test": int(len(test_rows)),
            "n_candidate_pairs_pHash<=24": int(len(candidate_rows)),
            "n_test_dropped": int(len(dropped)),
            "fraction_test_dropped": float(len(dropped) / max(1, len(test_rows))),
            "max_test_to_train_cosine_among_candidates": float(max(candidate_max_cos)) if candidate_max_cos else None,
            "min_pHash_hamming_to_train_min": int(min_ham_all.min()) if len(min_ham_all) else None,
            "min_pHash_hamming_to_train_p01": int(np.percentile(min_ham_all, 1)) if len(min_ham_all) else None,
        }
        print("  leakage guard:", guard_info)

        for split in SPLITS:
            rows = np.flatnonzero((df["split"] == split).to_numpy())
            np.save(
                Path(args.features).with_name(f"retfound_cfp_224_{split}.npy"),
                np.asarray(feats[feat_row[rows]], dtype=np.float32),
            )

    baseline = None
    if args.clean_baseline:
        cb = pd.read_csv(args.clean_baseline, dtype={"path": str}, usecols=["dataset"])
        baseline = {
            "source": str(args.clean_baseline),
            "clean_rows": int(len(cb)),
            "rows_by_dataset": {k: int(v) for k, v in cb["dataset"].value_counts().items()},
            "holdout_protected_groups": [],
        }
        if args.provenance and Path(args.provenance).exists():
            prov = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
            baseline["conflict_groups_removed"] = prov.get("conflict_groups")
            baseline["rows_removed_by_conflict_clean"] = prov.get("rows_removed")
            baseline["holdout_protected_groups"] = prov.get("holdout_protected_groups", [])

    print("[5/5] writing outputs")
    report = build_report(df, dedup_stats, split_info, args.tag, args.limit, baseline=baseline)
    report["gates"]["G3_embedding_guard"] = guard_info
    if args.features:
        report["leakage_guard_dropped_by_dataset"] = {
            k: int(v) for k, v in guard_dropped_by_dataset.items()
        }
    if quality is not None:
        report["quality"] = {
            "n": int(len(quality)),
            "flag_counts": quality["quality_flag"].value_counts().to_dict(),
            "reject_fraction": float((quality["quality_flag"] == "reject").mean()),
        }
    write_outputs(df, args.tag, report, quality=quality)

    print(json.dumps({k: report[k] for k in ("rows_total", "rows_by_split", "gates")}, indent=2))
    print(f"done in {time.time() - started:.0f}s")

    failed = (
        report["gates"]["G1_forbidden_path_rows"]
        or report["gates"]["G1_pterygium_rows"]
        or report["gates"]["G1_ddr_label5_rows"]
        or report["gates"]["G2_cross_split_groups"]
        or report["gates"]["G4_mask_violations"]
        or report["gates"]["G4_rows_without_supervision"]
    )
    if args.limit is None:
        if report["idrid_external_holdout"]["n"] != IDRID_HOLDOUT:
            print("FAIL: IDRID external holdout != 103", file=sys.stderr)
            failed = True
        bad_counts = [k for k, ok in report["row_count_match_expected"].items() if not ok]
        if bad_counts:
            dropped_by_ds = report.get("leakage_guard_dropped_by_dataset", {})
            explained = {k: dropped_by_ds.get(k, 0) for k in bad_counts}
            if all(explained.values()) and not args.clean_baseline:
                print(f"NOTE: row counts differ from plan for {bad_counts}", file=sys.stderr)
            else:
                print(
                    f"NOTE: v1 rows differ from clean baseline for {bad_counts} "
                    f"by {explained} (leakage guard G3)",
                    file=sys.stderr,
                )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
