"""Stage 1 - frozen RETFound encoder, train the 9-label head on cached features.

Usage
  python src/train_head.py --config configs/stage1_head.yaml --smoke --steps 2
  python src/train_head.py --config configs/stage1_head.yaml --epochs 2 --run-dir outputs/runs/stage1_pilot/seed_42
  python src/train_head.py --config configs/stage1_head.yaml --seed 42

Only `manifest_v1.csv` is accepted for training (test / external holdout rows are
never used for fitting or model selection).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import LABELS, M_COLS, Y_COLS, FeatureDataset, load_manifest  # noqa: E402
from head import CFP9Head  # noqa: E402
from losses import balanced_masked_bce  # noqa: E402
from manifest_provenance import manifest_provenance  # noqa: E402
from paths import path_key  # noqa: E402
from samplers import sqrt_balanced_sampler  # noqa: E402

FEAT = PROJECT_ROOT / "outputs" / "features" / "retfound_cfp_224.npy"
FEAT_INDEX = PROJECT_ROOT / "outputs" / "features" / "retfound_cfp_224_index.csv"
SMOKE_REPORT = PROJECT_ROOT / "outputs" / "reports" / "stage1_smoke.json"
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs" / "stage1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def evaluate(head, features, feat_row, Y, M, rows, device, min_pos: int = 30, batch: int = 1024):
    """Return per-label AUROC, macro AUROC over labels with n_pos >= min_pos, used labels.

    `rows` index the manifest; `feat_row` maps manifest rows to cached-feature rows
    (manifest v1 is shorter than the feature file because of the G3 leakage guard).
    """
    head.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            idx = rows[start : start + batch]
            x = torch.from_numpy(np.array(features[feat_row[idx]], dtype=np.float32, copy=True)).to(device)
            logits.append(head(x).float().cpu().numpy())
    logits = np.concatenate(logits) if logits else np.zeros((0, len(LABELS)), dtype=np.float32)
    targets = Y[rows]
    masks = M[rows]

    per_label = {}
    for j, label in enumerate(LABELS):
        known = masks[:, j] == 1
        y = targets[known, j]
        p = logits[known, j]
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        auc = float(roc_auc_score(y, p)) if n_pos > 0 and n_neg > 0 else float("nan")
        per_label[label] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auroc": auc,
            "used_for_macro": bool(n_pos >= min_pos and np.isfinite(auc)),
        }
    used = [v["auroc"] for v in per_label.values() if v["used_for_macro"]]
    macro = float(np.mean(used)) if used else float("nan")
    used_labels = [label for label, v in per_label.items() if v["used_for_macro"]]
    return per_label, macro, used_labels


def make_checkpoint(head, optim, scheduler, epoch, cfg, seed, manifest_path, manifest_sha, metrics, manifest_prov):
    return {
        "head": head.state_dict(),
        "optimizer": optim.state_dict() if optim is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "config": cfg,
        "seed": int(seed),
        "labels": list(LABELS),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "manifest_file_sha256": manifest_prov["file_sha256"],
        "manifest_provenance_mode": manifest_prov["provenance_mode"],
        "epoch_metrics": metrics,
        "torch_version": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "outputs" / "manifest" / "manifest_v1.csv"))
    parser.add_argument("--features", default=str(FEAT), help="RETFound feature .npy")
    parser.add_argument("--feature-index", default=str(FEAT_INDEX), help="feature index CSV")
    parser.add_argument("--smoke", action="store_true", help="run --steps optimizer steps only, never writes run dirs")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs (e.g. 2 for a pilot)")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument("--run-dir", default=None, help="override outputs/runs/stage1/seed_<seed>")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    min_pos = int(cfg.get("macro_min_pos", 30))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = Path(args.manifest)
    if manifest_path.name not in {"manifest_v1.csv", "manifest_v1.public.csv"}:
        raise SystemExit(f"refusing to train on {manifest_path.name}: only manifest_v1*.csv is allowed")
    manifest = load_manifest(manifest_path)
    manifest_prov = manifest_provenance(manifest_path)
    manifest_sha = manifest_prov["canonical_sha256"]

    features = np.load(args.features, mmap_mode="r")
    index = pd.read_csv(args.feature_index, dtype={"path": str})["path"].map(path_key)
    pos = manifest["path"].map(path_key).map({p: i for i, p in enumerate(index)})
    if pos.isna().any():
        raise SystemExit(f"{int(pos.isna().sum())} rows without cached features")
    feat_row = pos.to_numpy(dtype=np.int64)

    Y = manifest[Y_COLS].to_numpy(dtype=np.float32)
    M = manifest[M_COLS].to_numpy(dtype=np.float32)

    train_mask = (manifest["split"] == "train").to_numpy() & (manifest["is_external_holdout"] == 0).to_numpy()
    if cfg.get("drop_train_rejects", True):
        train_mask &= (manifest["quality_flag"] != "reject").to_numpy()
    train_rows = np.flatnonzero(train_mask)
    val_rows = np.flatnonzero((manifest["split"] == "val").to_numpy())
    if set(manifest.iloc[train_rows]["split"].unique()) != {"train"}:
        raise SystemExit("train rows contain non-train splits")
    if manifest.iloc[train_rows]["is_external_holdout"].any():
        raise SystemExit("external holdout leaked into train rows")

    train_ds = FeatureDataset(features, manifest, train_rows, feat_row[train_rows])
    ys, ms = Y[train_rows], M[train_rows]
    label_counts = torch.tensor((ys * ms).sum(axis=0), dtype=torch.float32, device=device)
    sw = manifest.iloc[train_rows]["sample_weight"].to_numpy(dtype=np.float64)

    sampler = None
    if cfg.get("sqrt_balanced_sampler", True):
        sampler = sqrt_balanced_sampler(
            manifest.iloc[train_rows].reset_index(drop=True), np.arange(len(train_rows)), seed=seed
        )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 256)),
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        drop_last=False,
    )

    head = CFP9Head(dim=1024, hidden=int(cfg.get("hidden", 512)), n=9, p=float(cfg.get("dropout", 0.3))).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-2)))
    epochs = int(args.epochs if args.epochs is not None else cfg.get("epochs", 30))
    scheduler = None
    if cfg.get("scheduler", "cosine") == "cosine":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim, max_lr=float(cfg.get("lr", 1e-3)), total_steps=max(1, epochs * len(loader))
        )

    eps = float(cfg.get("label_smoothing", 0.05))
    alpha = float(cfg.get("alpha", 0.25))

    # ---------------------------------------------------------------- smoke
    if args.smoke:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        head.train()
        log, it, started = [], iter(loader), time.time()
        for step in range(1, args.steps + 1):
            try:
                x, y, m, w = next(it)
            except StopIteration:
                it = iter(loader)
                x, y, m, w = next(it)
            x, y, m, w = x.to(device), y.to(device), m.to(device), w.to(device)
            loss = balanced_masked_bce(head(x), y, m, label_counts, sample_weight=w, eps=eps, alpha=alpha)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()
            finite = bool(torch.isfinite(loss))
            log.append({"step": step, "loss": float(loss.detach()), "finite": finite})
            print(f"step {step}: loss={float(loss.detach()):.6f} finite={finite}")
        per_label, macro, used_labels = evaluate(head, features, feat_row, Y, M, val_rows, device, min_pos=min_pos)
        peak = float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0
        report = {
            "mode": "smoke",
            "steps": args.steps,
            "seconds": round(time.time() - started, 1),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "manifest_file_sha256": manifest_prov["file_sha256"],
            "manifest_provenance_mode": manifest_prov["provenance_mode"],
            "seed": seed,
            "loss_log": log,
            "all_losses_finite": bool(all(e["finite"] for e in log)),
            "n_train": int(len(train_rows)),
            "n_val": int(len(val_rows)),
            "train_mask_known_fraction": float(ms.mean()),
            "sample_weight_unique": int(len(np.unique(sw))),
            "device": str(device),
            "peak_vram_gb": peak,
            "macro_min_pos": min_pos,
            "val_macro_auroc": macro,
            "val_macro_labels": used_labels,
            "val_per_label": per_label,
        }
        SMOKE_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("mode", "steps", "all_losses_finite", "val_macro_auroc", "peak_vram_gb")}, indent=2))
        return 0 if report["all_losses_finite"] else 1

    # ------------------------------------------------------------ full run
    run_dir = Path(args.run_dir) if args.run_dir else RUN_ROOT / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = []
    best_macro, best_epoch, best_metrics = float("-inf"), None, None
    started = time.time()
    for epoch in range(1, epochs + 1):
        head.train()
        ep_start, total, batches, nonfinite = time.time(), 0.0, 0, 0
        for x, y, m, w in loader:
            x, y, m, w = x.to(device), y.to(device), m.to(device), w.to(device)
            loss = balanced_masked_bce(head(x), y, m, label_counts, sample_weight=w, eps=eps, alpha=alpha)
            if not torch.isfinite(loss):
                nonfinite += 1
                optim.zero_grad(set_to_none=True)
                continue
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach())
            batches += 1
        train_loss = total / max(batches, 1)
        per_label, macro, used_labels = evaluate(head, features, feat_row, Y, M, val_rows, device, min_pos=min_pos)
        peak = float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": float(optim.param_groups[0]["lr"]),
            "val_macro_auroc": macro,
            "val_macro_n_labels": len(used_labels),
            "nonfinite_batches": nonfinite,
            "seconds": round(time.time() - ep_start, 2),
            "peak_vram_gb": peak,
        }
        row.update({f"auroc_{k}": v["auroc"] for k, v in per_label.items()})
        row.update({f"n_pos_{k}": v["n_pos"] for k, v in per_label.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(metrics_path, index=False, encoding="utf-8")

        improved = bool(np.isfinite(macro) and macro > best_macro)
        if improved:
            best_macro, best_epoch, best_metrics = macro, epoch, per_label
            torch.save(
                make_checkpoint(head, optim, scheduler, epoch, cfg, seed, manifest_path, manifest_sha, row, manifest_prov),
                run_dir / "best.pt",
            )
        torch.save(
            make_checkpoint(head, optim, scheduler, epoch, cfg, seed, manifest_path, manifest_sha, row, manifest_prov),
            run_dir / "last.pt",
        )
        print(
            f"epoch {epoch:02d}/{epochs} loss={train_loss:.4f} val_macro={macro:.4f} "
            f"best={best_macro:.4f}@{best_epoch}"
        )

    # best.pt reload check (same weights -> same val AUROC)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    reload_head = CFP9Head(dim=1024, hidden=int(cfg.get("hidden", 512)), n=9, p=float(cfg.get("dropout", 0.3))).to(device)
    reload_head.load_state_dict(ckpt["head"])
    reload_per_label, reload_macro, reload_labels = evaluate(reload_head, features, feat_row, Y, M, val_rows, device, min_pos=min_pos)
    compare = {
        k: abs(reload_per_label[k]["auroc"] - best_metrics[k]["auroc"])
        for k in LABELS
        if np.isfinite(reload_per_label[k]["auroc"]) and np.isfinite(best_metrics[k]["auroc"])
    }
    max_delta = float(max(compare.values())) if compare else 0.0

    run = {
        "mode": "full",
        "config": cfg,
        "config_path": str(args.config),
        "seed": seed,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "manifest_file_sha256": manifest_prov["file_sha256"],
        "manifest_provenance_mode": manifest_prov["provenance_mode"],
        "features": str(args.features),
        "features_bytes": int(Path(args.features).stat().st_size),
        "labels": list(LABELS),
        "macro_min_pos": min_pos,
        "n_train": int(len(train_rows)),
        "n_val": int(len(val_rows)),
        "train_mask_known_fraction": float(ms.mean()),
        "sample_weight_unique": int(len(np.unique(sw))),
        "sample_weight_mean": float(sw.mean()),
        "epochs": epochs,
        "batch_size": int(cfg.get("batch_size", 256)),
        "steps_per_epoch": len(loader),
        "best_epoch": best_epoch,
        "best_val_macro_auroc": best_macro,
        "best_macro_labels": reload_labels,
        "best_val_per_label": reload_per_label,
        "reload_check": {"max_abs_auroc_delta": max_delta, "reproduced": bool(max_delta <= 1e-6)},
        "device": str(device),
        "torch_version": torch.__version__,
        "git_commit": git_commit(),
        "seconds": round(time.time() - started, 1),
        "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0,
        "run_dir": str(run_dir),
    }
    (run_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("seed", "best_epoch", "best_val_macro_auroc", "reload_check", "seconds")}, indent=2))
    return 0 if run["reload_check"]["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
