#!/usr/bin/env python
"""R2-3 RETFound embedding dedup layer (plan 6.3 layers 3-4).

Input : outputs/manifest/manifest_v0_clean_preprocessed.csv  (23,299 rows, sorted by path)
        outputs/preprocessed/<dataset>/<stem>.png
Output: outputs/features/retfound_cfp_224.npy          (float32, L2-normalised, 1024-d)
        outputs/features/retfound_cfp_224_index.csv    (row order of the npy)
        outputs/manifest/dedup_groups_v1.csv           (path, group_id_v0, group_id_v1)
        outputs/reports/dedup_report_v1.json

Rule: cosine similarity >= 0.98 between two images -> same group (union-find,
merged with the SHA1/pHash group_id already present in the clean manifest).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RETFOUND_DIR = PROJECT_ROOT / "RETFound-main"
CKPT = PROJECT_ROOT / "model" / "FoundmaenatureCFP" / "RETFound_mae_natureCFP.pth"
PREP_MAP = PROJECT_ROOT / "outputs" / "manifest" / "manifest_v0_clean_preprocessed.csv"
FEAT = PROJECT_ROOT / "outputs" / "features" / "retfound_cfp_224.npy"
FEAT_INDEX = PROJECT_ROOT / "outputs" / "features" / "retfound_cfp_224_index.csv"
FEAT_CENTERED = PROJECT_ROOT / "outputs" / "features" / "retfound_cfp_224_centered.npy"
GROUP_MAP = PROJECT_ROOT / "outputs" / "manifest" / "dedup_groups_v1.csv"
REPORT = PROJECT_ROOT / "outputs" / "reports" / "dedup_report_v1.json"

SIM_THRESHOLD = 0.99
REUSE = '--reuse-features' in sys.argv
PHASH_MAX_HAMMING = 24
KNN_K = 21
BATCH = 32
WORKERS = 6

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from encoder import build_encoder  # noqa: E402


class FundusDataset(Dataset):
    def __init__(self, paths: list[str]) -> None:
        self.paths = paths
        self.tf = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            tensor = self.tf(img.convert("RGB"))
        return tensor


def load_encoder():
    model, info = build_encoder(target="cuda")
    print(f"  load_state_dict: {info}")
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()
    return model, info


@torch.no_grad()
def extract(model) -> tuple[np.ndarray, dict]:
    df = pd.read_csv(PREP_MAP, dtype={"path": str})
    paths = df["preprocessed_path"].tolist()
    loader = DataLoader(
        FundusDataset(paths), batch_size=BATCH, shuffle=False, num_workers=WORKERS, pin_memory=True
    )
    feats = []
    started = time.time()
    for i, batch in enumerate(loader, start=1):
        batch = batch.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(batch)
        feats.append(F.normalize(out.float(), dim=1).cpu().numpy())
        if i % 100 == 0:
            done = i * BATCH
            print(f"  {done}/{len(paths)}  ({time.time() - started:.0f}s)", flush=True)
    arr = np.concatenate(feats, axis=0).astype(np.float32)
    return arr, {"n": int(arr.shape[0]), "dim": int(arr.shape[1]), "seconds": round(time.time() - started, 1)}


class UnionFind:
    def __init__(self, keys) -> None:
        self.parent = {k: k for k in keys}

    def find(self, key):
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.manual_seed(42)

    if REUSE and FEAT.is_file():
        print("[1/4] reusing cached features")
        feats = np.load(FEAT).astype(np.float32)
        feat_info = {"n": int(feats.shape[0]), "dim": int(feats.shape[1]), "seconds": 0.0, "reused": True}
        load_info = {"reused": True}
    else:
        print("[1/4] loading RETFound encoder")
        model, load_info = load_encoder()
        print("[2/4] extracting 1024-d features")
        feats, feat_info = extract(model)
        FEAT.parent.mkdir(parents=True, exist_ok=True)
        np.save(FEAT, feats)
        prep = pd.read_csv(PREP_MAP, dtype={"path": str})[["path"]]
        prep.to_csv(FEAT_INDEX, index=False, encoding="utf-8")

    print("[3/4] centred cosine kNN + pHash confirmation")
    from sklearn.neighbors import NearestNeighbors

    # Raw RETFound features share a dominant "fundus" direction (nearest-neighbour
    # cosine median 0.9996), so raw cosine collapses the graph.  Mean-centre before
    # the cosine search, and require pHash confirmation (Hamming <= 24) so that
    # transitive closure cannot chain unrelated images.
    mean = feats.mean(axis=0, keepdims=True)
    centered = feats - mean
    centered /= np.linalg.norm(centered, axis=1, keepdims=True)
    np.save(FEAT_CENTERED, centered.astype(np.float32))

    nn = NearestNeighbors(n_neighbors=KNN_K, metric="cosine", algorithm="brute", n_jobs=8)
    nn.fit(centered)
    dist, ind = nn.kneighbors(centered, n_neighbors=KNN_K)
    sims = 1.0 - dist[:, 1:]

    clean = pd.read_csv(PROJECT_ROOT / "outputs" / "manifest" / "manifest_v0_clean.csv", dtype={"path": str, "phash": str})
    order = pd.read_csv(FEAT_INDEX, dtype={"path": str})
    clean_idx = clean.set_index("path").loc[order["path"]]
    v0_group = clean_idx["group_id"].tolist()
    phash_int = [int(str(h), 16) for h in clean_idx["phash"].tolist()]

    def hamming(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    print("[4/4] union-find merge (cos + pHash)")
    uf = UnionFind(set(v0_group))
    n_edges = 0
    for col in range(sims.shape[1]):
        for a in np.flatnonzero(sims[:, col] >= SIM_THRESHOLD):
            b = int(ind[a, col + 1])
            if hamming(phash_int[a], phash_int[b]) <= PHASH_MAX_HAMMING:
                if uf.union(v0_group[a], v0_group[b]):
                    n_edges += 1

    roots = {}
    group_ids = []
    for g in v0_group:
        root = uf.find(g)
        if root not in roots:
            roots[root] = f"h{len(roots) + 1:05d}"
        group_ids.append(roots[root])

    group_df = pd.DataFrame({"path": order["path"], "group_id_v0": v0_group, "group_id_v1": group_ids})
    group_df.to_csv(GROUP_MAP, index=False, encoding="utf-8")

    sizes = group_df.groupby("group_id_v1").size()
    report = {
        "feature": feat_info,
        "load_state_dict": load_info,
        "rule": f"cosine(centred) >= {SIM_THRESHOLD} AND pHash Hamming <= {PHASH_MAX_HAMMING}",
        "raw_cosine_note": "raw (uncentred) cosine >= 0.98 collapsed 22887 groups into 5219 (max 64); centred cosine alone still chains, hence the pHash confirmation",
        "n_edges_merged": int(n_edges),
        "n_groups_v0": int(group_df["group_id_v0"].nunique()),
        "n_groups_v1": int(group_df["group_id_v1"].nunique()),
        "max_group_size": int(sizes.max()),
        "n_multi_member_groups": int((sizes > 1).sum()),
        "nearest_similarity_p50": float(np.percentile(sims[:, 0], 50)),
        "nearest_similarity_p99": float(np.percentile(sims[:, 0], 99)),
        "nearest_similarity_max": float(sims[:, 0].max()),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
