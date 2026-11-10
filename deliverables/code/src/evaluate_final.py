"""Final one-shot evaluation for Stage 1 v2 (cached features + head) and Stage 2 (full model).

Test and the IDRID external holdout may only be touched once, after the val self-check
passes. Thresholds (sens@90%spec) and per-label temperature scaling are fitted on the
val split only and then applied unchanged to test.

  # 1) val self-check (must reproduce the train-time numbers)
  python src/evaluate_final.py --split val

  # 2) the single allowed test run
  python src/evaluate_final.py --split test --final

  # 3) IDRID external holdout (103 images), separate report
  python src/evaluate_final.py --split external_holdout --final

No seed is selected on test: every seed is reported, and the headline number is the
3-seed mean +- std with a group-aware bootstrap CI of that mean.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import LABELS, M_COLS, Y_COLS, align_features, load_manifest  # noqa: E402
from encoder import build_encoder  # noqa: E402
from head import CFP9Head  # noqa: E402
from manifest_provenance import manifest_provenance  # noqa: E402
from paths import display_path, path_key  # noqa: E402
from train_finetune import ImageManifestDataset, attach_forward, build_transforms  # noqa: E402

FEAT = PROJECT_ROOT / "outputs" / "features" / "retfound_cfp_224.npy"
FEAT_INDEX = PROJECT_ROOT / "outputs" / "features" / "retfound_cfp_224_index.csv"
PREPROCESSED = PROJECT_ROOT / "outputs" / "manifest" / "manifest_v0_clean_preprocessed.csv"
STAGE1_CFG = PROJECT_ROOT / "configs" / "stage1_head_v2.yaml"
STAGE2_CFG = PROJECT_ROOT / "configs" / "stage2_finetune.yaml"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "runs"

SEEDS = ("42", "43", "44")


def checkpoint_path(root: Path, stage: str, seed: str) -> Path:
    """Support both the authoring layout and a flat release ``checkpoints/`` folder."""
    flat = root / (f"stage1_v2_seed{seed}_best.pt" if stage == "stage1" else f"stage2_seed{seed}_best.pt")
    if flat.is_file():
        return flat
    return root / (f"{stage}_v2" if stage == "stage1" else stage) / f"seed_{seed}" / "best.pt"
MIN_POS = 30          # macro / "reliable label" threshold on the split being reported
CALIB_MIN_POS = 30    # temperature scaling needs both classes with enough support
SPEC_TARGET = 0.90
CORE_LABELS = ["normal", "dr", "glaucoma", "macular_scar", "myopia"]
EXPLORATORY = ["csc", "disc_edema", "retinal_det", "retinitis_pig"]


# --------------------------------------------------------------------- helpers
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() <= 0 or (1 - y).sum() <= 0:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_aupr(y: np.ndarray, s: np.ndarray) -> float:
    if y.sum() <= 0:
        return float("nan")
    return float(average_precision_score(y, s))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        conf, acc = p[sel].mean(), y[sel].mean()
        ece += (sel.sum() / len(y)) * abs(acc - conf)
    return float(ece)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Per-label temperature by grid search on val NLL (no test data involved)."""
    if y.sum() <= 0 or (1 - y).sum() <= 0:
        return 1.0
    best_t, best_nll = 1.0, float("inf")
    for lo, hi, n in ((0.05, 20.0, 120), (0.8, 1.25, 91)):
        for t in np.exp(np.linspace(np.log(lo), np.log(hi), n)):
            p = sigmoid(logits / t)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            if nll < best_nll:
                best_nll, best_t = float(nll), float(t)
    return best_t


def spec90_threshold(neg_scores: np.ndarray) -> float:
    """Largest threshold whose val specificity is still >= 0.90."""
    if len(neg_scores) == 0:
        return float("nan")
    return float(np.quantile(neg_scores, SPEC_TARGET, method="higher"))


def label_block(y: np.ndarray, m: np.ndarray, s: np.ndarray, p: np.ndarray, min_pos: int) -> dict:
    """Per-label metrics for one model on one split, restricted to rows with m == 1 (as in training)."""
    out = {}
    for j, label in enumerate(LABELS):
        known = m[:, j] == 1
        yj, sj, pj = y[known, j], s[known, j], p[known, j]
        n_pos, n_neg = int(yj.sum()), int(len(yj) - yj.sum())
        out[label] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auroc": safe_auroc(yj, sj),
            "aupr": safe_aupr(yj, sj),
            "ece": expected_calibration_error(yj, pj),
            "brier": brier(yj, pj),
            "used_for_macro": bool(n_pos >= min_pos),
            "exploratory": label in EXPLORATORY,
        }
    kept = [out[l] for l in LABELS]
    used = [v["auroc"] for v in kept if v["used_for_macro"] and np.isfinite(v["auroc"])]
    out["_macro_auroc"] = float(np.mean(used)) if used else float("nan")
    used_pr = [v["aupr"] for v in kept if v["used_for_macro"] and np.isfinite(v["aupr"])]
    out["_macro_aupr"] = float(np.mean(used_pr)) if used_pr else float("nan")
    used_ece = [v["ece"] for v in kept if v["used_for_macro"] and np.isfinite(v["ece"])]
    out["_macro_ece"] = float(np.mean(used_ece)) if used_ece else float("nan")
    out["_macro_labels"] = [l for l, v in zip(LABELS, kept) if v["used_for_macro"]]
    return out


def grouped_index(groups: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(groups, kind="stable")
    uniq, starts = np.unique(groups[order], return_index=True)
    return np.split(order, starts[1:])


def bootstrap_metrics(y, m, scores, group_rows, labeled_idx, reps, rng):
    """Bootstrap DR AUROC / macro AUROC / DR AUPR for one model; whole groups resampled.

    Per label the metric is restricted to rows with m == 1, exactly like training/val scoring.
    """
    n_groups = len(group_rows)
    dr_j = LABELS.index("dr")
    dr_auc = np.empty(reps, dtype=np.float32)
    macro = np.empty(reps, dtype=np.float32)
    dr_ap = np.empty(reps, dtype=np.float32)
    for b in range(reps):
        pick = rng.integers(0, n_groups, n_groups)
        rows = np.concatenate([group_rows[g] for g in pick])
        yy, mm = y[rows], m[rows]
        kd = mm[:, dr_j] == 1
        dr_auc[b] = safe_auroc(yy[kd, dr_j], scores[rows][kd, dr_j])
        dr_ap[b] = safe_aupr(yy[kd, dr_j], scores[rows][kd, dr_j])
        sc = scores[rows]
        vals = []
        for j in labeled_idx:
            k = mm[:, j] == 1
            v = safe_auroc(yy[k, j], sc[k, j])
            if np.isfinite(v):
                vals.append(v)
        macro[b] = float(np.mean(vals)) if vals else np.nan
    return {"dr_auroc": dr_auc, "macro_auroc": macro, "dr_aupr": dr_ap}


def ci(arr: np.ndarray) -> list[float]:
    a = arr[np.isfinite(arr)]
    if len(a) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ------------------------------------------------------------------ prediction
def stage1_predict(ckpt_path: Path, cfg: dict, features, feat_row, rows, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if list(ckpt.get("labels", [])) != list(LABELS):
        raise SystemExit(f"{ckpt_path.name}: label order mismatch")
    head = CFP9Head(dim=1024, hidden=int(cfg.get("hidden", 1024)), n=len(LABELS),
                    p=float(cfg.get("dropout", 0.3))).to(device).eval()
    head.load_state_dict(ckpt["head"], strict=True)
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), 2048):
            idx = rows[start:start + 2048]
            x = torch.from_numpy(np.array(features[feat_row[idx]], dtype=np.float32, copy=True)).to(device)
            out.append(head(x).float().cpu().numpy())
    return np.concatenate(out), {"checkpoint": str(ckpt_path), "epoch": ckpt.get("epoch"),
                                 "manifest_sha256": ckpt.get("manifest_sha256")}


def stage2_predict(ckpt_path: Path, cfg: dict, records: pd.DataFrame, device, batch: int, workers: int):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if list(ckpt.get("labels", [])) != list(LABELS):
        raise SystemExit(f"{ckpt_path.name}: label order mismatch")
    # The full Stage 2 checkpoint contains every encoder weight.  Instantiate the
    # architecture only; do not require or load the separate RETFound pretrained file.
    base, enc_info = build_encoder(drop_path_rate=0.0, target="cpu", load_weights=False)
    head = CFP9Head(dim=1024, hidden=int(cfg.get("hidden", 1024)), n=len(LABELS),
                    p=float(cfg.get("dropout", 0.3)))
    base.head = head
    attach_forward(base, bool(cfg.get("l2_normalize_features", True)))
    base.load_state_dict(ckpt["model"], strict=True)
    base = base.to(device).eval()

    ds = ImageManifestDataset(records, build_transforms(int(cfg.get("img_size", 224)), 0.0, train=False,
                                                       center_crop=bool(cfg.get("center_crop", True))))
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for x, _, _, _ in loader:
            out.append(base(x.to(device, non_blocking=True)).float().cpu().numpy())
    logits = np.concatenate(out)
    ckpt_epoch = ckpt.get("epoch")
    ckpt_manifest = ckpt.get("manifest_sha256")
    del base, head, ckpt
    gc.collect()
    torch.cuda.empty_cache()
    return logits, {"checkpoint": str(ckpt_path), "epoch": ckpt_epoch,
                    "manifest_sha256": ckpt_manifest, "encoder_load": enc_info}


def rows_of(manifest: pd.DataFrame, split: str) -> np.ndarray:
    if split == "val":
        return np.flatnonzero((manifest["split"] == "val").to_numpy())
    if split == "test":
        return np.flatnonzero(((manifest["split"] == "test") & (manifest["is_external_holdout"] == 0)).to_numpy())
    if split == "external_holdout":
        return np.flatnonzero((manifest["is_external_holdout"] == 1).to_numpy())
    raise SystemExit(f"unknown split {split}")



# --------------------------------------------------------------------- markdown
def _f(x, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


def _pm(vals):
    a = np.array([v for v in vals if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return "n/a", "n/a"
    return _f(a.mean()), _f(a.std(ddof=1)) if len(a) > 1 else "n/a"


def write_markdown(payload: dict, path: Path, ext: dict | None = None) -> None:
    split_name = {"test": "内部 test split", "val": "val（自检）", "external_holdout": "IDRID external holdout"}[payload["mode"]]
    res = payload["results"]
    seeds = tuple(sorted({k.rsplit("_seed", 1)[1] for k in res if "_seed" in k}, key=int))
    seed_label = f"{len(seeds)}-seed"
    L = []
    L.append(f"# CFP-9 最终评估 — {split_name}")
    L.append("")
    L.append(f"- 生成时间：{payload['generated_at']}")
    L.append(f"- manifest：`{payload['manifest']['path']}`，SHA256 `{payload['manifest']['sha256']}`")
    L.append(f"- 行数：{payload['manifest']['n_target']}（total {payload['manifest']['n_rows_total']}），"
             f"bootstrap 重采样单位：group_id，{payload['bootstrap']['n_groups']} 组 × {payload['bootstrap']['reps']} 次")
    L.append(f"- git commit：`{payload['provenance']['git_commit']}`，torch {payload['provenance']['torch_version']}")
    L.append(f"- 阈值与温度：仅在 **val** 上拟合，原样应用于本 split；macro 只用 n_pos ≥ {MIN_POS} 的标签")
    L.append("")

    L.append("## 1. 结论摘要")
    L.append("")
    for stage, name in (("stage2", "Stage 2（微调 ViT-L）"), ("stage1", "Stage 1 v2（冻结编码器）")):
        k = f"{stage}_dr_auroc"
        m = f"{stage}_macro_auroc"
        L.append(f"- **{name}**：DR AUROC {_f(payload['stage_summary'][k]['point'])} "
                 f"[{_f(payload['stage_summary'][k]['ci95'][0])}, {_f(payload['stage_summary'][k]['ci95'][1])}]，"
                 f"macro AUROC {_f(payload['stage_summary'][m]['point'])} "
                 f"[{_f(payload['stage_summary'][m]['ci95'][0])}, {_f(payload['stage_summary'][m]['ci95'][1])}]"
                 f"（{len(seeds)} seed 均值 + group-aware 95% CI）")
    pd_ = payload["paired_delta_stage2_minus_stage1"]
    L.append(f"- **主比较 Stage 2 − Stage 1**：DR AUROC {pd_['dr_auroc']['seedmean_delta_point']:+.4f} "
             f"[{pd_['dr_auroc']['seedmean_delta_ci95'][0]:+.4f}, {pd_['dr_auroc']['seedmean_delta_ci95'][1]:+.4f}]，"
             f"macro AUROC {pd_['macro_auroc']['seedmean_delta_point']:+.4f} "
             f"[{pd_['macro_auroc']['seedmean_delta_ci95'][0]:+.4f}, {pd_['macro_auroc']['seedmean_delta_ci95'][1]:+.4f}]")
    L.append("")

    for stage, name in (("stage1", "2. Stage 1 v2（cached features + head）"),
                        ("stage2", "3. Stage 2（preprocessed image + 完整模型）")):
        L.append(f"## {name}")
        L.append("")
        L.append("| seed | DR AUROC | macro AUROC | DR AUPR | macro AUPR | DR sens@90%spec | DR ECE (raw→cal) | macro ECE (raw→cal) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for sd in seeds:
            r = res[f"{stage}_seed{sd}"]
            L.append(f"| {sd} | {_f(r['dr_auroc'])} | {_f(r['macro_auroc'])} | {_f(r['dr_aupr'])} | {_f(r['macro_aupr'])} | "
                     f"{_f(r['dr_sens_at_spec90'])} | {_f(r['dr_ece_raw'],3)}→{_f(r['dr_ece'],3)} | "
                     f"{_f(r['macro_ece_raw'],3)}→{_f(r['macro_ece'],3)} |")
        mu_dr, sd_dr = _pm([res[f"{stage}_seed{s}"]["dr_auroc"] for s in seeds])
        mu_mc, sd_mc = _pm([res[f"{stage}_seed{s}"]["macro_auroc"] for s in seeds])
        mu_ap, sd_ap = _pm([res[f"{stage}_seed{s}"]["dr_aupr"] for s in seeds])
        mu_mp, sd_mp = _pm([res[f"{stage}_seed{s}"]["macro_aupr"] for s in seeds])
        mu_sn, sd_sn = _pm([res[f"{stage}_seed{s}"]["dr_sens_at_spec90"] for s in seeds])
        L.append(f"| **mean±std** | {mu_dr} ± {sd_dr} | {mu_mc} ± {sd_mc} | {mu_ap} ± {sd_ap} | "
                 f"{mu_mp} ± {sd_mp} | {mu_sn} ± {sd_sn} | | |")
        L.append("")

    L.append("## 4. Stage 2 − Stage 1（配对 bootstrap，同一重采样）")
    L.append("")
    L.append(f"| 指标 | seed42 Δ [95% CI] (p) | {seed_label} 均值 Δ [95% CI] (p) |")
    L.append("|---|---|---|")
    for metric, label in (("dr_auroc", "DR AUROC"), ("macro_auroc", "macro AUROC"), ("dr_aupr", "DR AUPR")):
        v = pd_[metric]
        L.append(f"| {label} | {v['seed42_delta_point']:+.4f} [{v['seed42_delta_ci95'][0]:+.4f}, {v['seed42_delta_ci95'][1]:+.4f}] "
                 f"(p={v['seed42_p_two_sided']:.4f}) | {v['seedmean_delta_point']:+.4f} "
                 f"[{v['seedmean_delta_ci95'][0]:+.4f}, {v['seedmean_delta_ci95'][1]:+.4f}] (p={v['seedmean_p_two_sided']:.4f}) |")
    L.append("")
    L.append("> seed42 vs seed42 仅作配对参考；主结论用 3-seed 均值。seed 由训练期 val 决定，未在 test 上挑选。")
    L.append("")

    L.append("## 5. 逐标签（3-seed 均值）")
    L.append("")
    L.append("| label | S1 AUROC | S2 AUROC | Δ AUROC | S2 AUPR | S2 sens@90%spec | 角色 |")
    L.append("|---|---|---|---|---|---|---|")
    for label in LABELS:
        a1, _ = _pm([res[f"stage1_seed{s}"]["per_label"][label]["auroc"] for s in seeds])
        a2, _ = _pm([res[f"stage2_seed{s}"]["per_label"][label]["auroc"] for s in seeds])
        p2, _ = _pm([res[f"stage2_seed{s}"]["per_label"][label]["aupr"] for s in seeds])
        s2, _ = _pm([res[f"stage2_seed{s}"]["per_label"][label]["sensitivity_at_val_thr"] for s in seeds])
        v1 = np.array([res[f"stage1_seed{s}"]["per_label"][label]["auroc"] for s in seeds], dtype=float)
        v2 = np.array([res[f"stage2_seed{s}"]["per_label"][label]["auroc"] for s in seeds], dtype=float)
        m1 = np.nanmean(v1) if np.isfinite(v1).any() else np.nan
        m2 = np.nanmean(v2) if np.isfinite(v2).any() else np.nan
        delta = m2 - m1
        role = "exploratory" if label in EXPLORATORY else "core"
        L.append(f"| {label} | {a1} | {a2} | {delta:+.4f} | {p2} | {s2} | {role} |")
    L.append("")
    L.append(f"csc 为 exploratory（val n_pos 仅 12），不进入核心 macro 结论。")
    L.append("")

    L.append(f"## 6. 逐数据集 DR AUROC（{seed_label} 均值）")
    L.append("")
    datasets = sorted({d for s in seeds for d in res[f"stage2_seed{s}"]["per_dataset"]})
    L.append("| dataset | n | S1 DR AUROC | S2 DR AUROC | S2 macro AUROC |")
    L.append("|---|---|---|---|---|")
    for d in datasets:
        n = res["stage2_seed42"]["per_dataset"].get(d, {}).get("n", "n/a")
        s1, _ = _pm([res[f"stage1_seed{s}"]["per_dataset"].get(d, {}).get("dr_auroc", np.nan) for s in seeds])
        s2, _ = _pm([res[f"stage2_seed{s}"]["per_dataset"].get(d, {}).get("dr_auroc", np.nan) for s in seeds])
        m2, _ = _pm([res[f"stage2_seed{s}"]["per_dataset"].get(d, {}).get("macro_auroc", np.nan) for s in seeds])
        L.append(f"| {d} | {n} | {s1} | {s2} | {m2} |")
    L.append("")

    L.append("## 7. 阈值（val 拟合，原样应用）")
    L.append("")
    L.append("| label | thr@spec90 (S1 s42) | val sens | val spec | thr@spec90 (S2 s42) | val sens | val spec |")
    L.append("|---|---|---|---|---|---|---|")
    for label in LABELS:
        a = res["stage1_seed42"]["val_metrics_for_thresholds"][label]
        b = res["stage2_seed42"]["val_metrics_for_thresholds"][label]
        L.append(f"| {label} | {_f(a['thr_spec90'],3)} | {_f(a['val_sensitivity'],3)} | {_f(a['val_specificity'],3)} | "
                 f"{_f(b['thr_spec90'],3)} | {_f(b['val_sensitivity'],3)} | {_f(b['val_specificity'],3)} |")
    L.append("")

    if ext is not None:
        ex = ext["results"]
        L.append("## 8. IDRID external holdout（官方 103 张，单独一节）")
        L.append("")
        L.append(f"- 行数 {ext['manifest']['n_target']}，bootstrap {ext['bootstrap']['n_groups']} 组 × {ext['bootstrap']['reps']} 次")
        L.append(f"- 阈值/温度同样来自 val，未在 holdout 上调整")
        L.append("")
        L.append("| seed | S1 DR AUROC | S1 DR AUPR | S2 DR AUROC | S2 DR AUPR | S2 sens@90%spec | S2 ECE (raw→cal) |")
        L.append("|---|---|---|---|---|---|---|")
        for sd in seeds:
            a, b = ex[f"stage1_seed{sd}"], ex[f"stage2_seed{sd}"]
            L.append(f"| {sd} | {_f(a['dr_auroc'])} | {_f(a['dr_aupr'])} | {_f(b['dr_auroc'])} | {_f(b['dr_aupr'])} | "
                     f"{_f(b['dr_sens_at_spec90'])} | {_f(b['dr_ece_raw'],3)}→{_f(b['dr_ece'],3)} |")
        for stage in ("stage1", "stage2"):
            mu, sd_ = _pm([ex[f"{stage}_seed{s}"]["dr_auroc"] for s in seeds])
            ap, ap_sd = _pm([ex[f"{stage}_seed{s}"]["dr_aupr"] for s in seeds])
            L.append(f"| **{stage} mean±std** | {mu} ± {sd_} | {ap} ± {ap_sd} | | | | |")
        ex_pd = ext["paired_delta_stage2_minus_stage1"]["dr_auroc"]
        L.append("")
        L.append(f"配对（同一 holdout 重采样）：seed42 Δ = {ex_pd['seed42_delta_point']:+.4f} "
                 f"[{ex_pd['seed42_delta_ci95'][0]:+.4f}, {ex_pd['seed42_delta_ci95'][1]:+.4f}] "
                 f"(p={ex_pd['seed42_p_two_sided']:.3f})；{seed_label} 均值 Δ = {ex_pd['seedmean_delta_point']:+.4f} "
                 f"[{ex_pd['seedmean_delta_ci95'][0]:+.4f}, {ex_pd['seedmean_delta_ci95'][1]:+.4f}] "
                 f"(p={ex_pd['seedmean_p_two_sided']:.3f})。")
        L.append("")
        L.append(f"holdout 仅 DR 有标注（n_pos={ext['results']['stage2_seed42']['per_label']['dr']['n_pos']}），"
                 "其余 8 个标签 n_pos=0，故 macro 与 DR 相同；样本量小，CI 较宽。")
        L.append("")

    L.append("## 复现信息")
    L.append("")
    L.append("| model | checkpoint | epoch | sha256 |")
    L.append("|---|---|---|---|")
    for key, prov in payload["provenance"]["checkpoints"].items():
        L.append(f"| {key} | `{prov['checkpoint']}` | {prov.get('ckpt_epoch')} | `{(prov.get('checkpoint_sha256') or 'not hashed')[:16]}` |")
    L.append("")
    L.append(f"- Stage1 config：`{payload['provenance']['stage1_config']}`；Stage2 config：`{payload['provenance']['stage2_config']}`")
    L.append(f"- features：`{payload['provenance']['features']}`")
    if payload.get("self_check_vs_run_json"):
        L.append("")
        L.append("### val 自检（与训练 run.json 对比）")
        L.append("")
        L.append("| model | run.json | recomputed | Δ | ok |")
        L.append("|---|---|---|---|---|")
        for k, v in payload["self_check_vs_run_json"].items():
            L.append(f"| {k} | {v['expected_run_json']:.6f} | {v['recomputed']:.6f} | {v['abs_diff']:.2e} | {v['match']} |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    global PREPROCESSED, FEAT, FEAT_INDEX, REPORT_DIR, CHECKPOINT_ROOT, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(PROJECT_ROOT / "outputs" / "manifest" / "manifest_v1.csv"))
    ap.add_argument("--preprocessed", default=str(PREPROCESSED),
                    help="CSV with path -> preprocessed_path (release copy is supported)")
    ap.add_argument("--features", default=str(FEAT), help="RETFound feature .npy")
    ap.add_argument("--feature-index", default=str(FEAT_INDEX), help="feature index CSV")
    ap.add_argument("--checkpoint-root", default=str(CHECKPOINT_ROOT),
                    help="nested outputs/runs or flat release checkpoints folder")
    ap.add_argument("--run-json-root", default=str(CHECKPOINT_ROOT),
                    help="optional root for val self-check run.json files")
    ap.add_argument("--val-reference", default=str(PROJECT_ROOT / "deliverables" / "model" / "val_summary_3seed.csv"),
                    help="shipped CSV fallback for the val self-check when run.json is absent")
    ap.add_argument("--report-dir", default=str(REPORT_DIR), help="output report directory")
    ap.add_argument("--seeds", default="42,43,44", help="comma-separated seeds; 42 is required for paired headline")
    ap.add_argument("--split", choices=["val", "test", "external_holdout"], default=None,
                    help="required unless --rebuild-md")
    ap.add_argument("--final", action="store_true", help="required for test / external_holdout; writes final_eval.*")
    ap.add_argument("--force", action="store_true", help="overwrite an existing final_eval.json (discouraged)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261109)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hash-checkpoints", action="store_true")
    ap.add_argument("--rebuild-md", action="store_true",
                    help="only regenerate final_eval.md from the existing JSONs (no inference)")
    args = ap.parse_args(argv)

    PREPROCESSED = Path(args.preprocessed)
    FEAT = Path(args.features)
    FEAT_INDEX = Path(args.feature_index)
    REPORT_DIR = Path(args.report_dir)
    CHECKPOINT_ROOT = Path(args.checkpoint_root)
    SEEDS = tuple(x.strip() for x in args.seeds.split(",") if x.strip())
    if not SEEDS or "42" not in SEEDS:
        raise SystemExit("--seeds must include 42 (seed42 is the paired reference)")

    if args.rebuild_md:
        main_json = REPORT_DIR / "final_eval.json"
        if not main_json.is_file():
            raise SystemExit("final_eval.json not found")
        payload = json.loads(main_json.read_text(encoding="utf-8"))
        ext_path = REPORT_DIR / "final_eval_external_holdout.json"
        ext = json.loads(ext_path.read_text(encoding="utf-8")) if ext_path.is_file() else None
        write_markdown(payload, REPORT_DIR / "final_eval.md", ext)
        print("rebuilt final_eval.md" + (" (with IDRID holdout section)" if ext else ""))
        return 0

    if args.split is None:
        raise SystemExit("--split is required (val / test / external_holdout) unless --rebuild-md")

    manifest_path = Path(args.manifest)
    if manifest_path.name not in {"manifest_v1.csv", "manifest_v1.public.csv"}:
        raise SystemExit(f"refusing to evaluate on {manifest_path.name}: only manifest_v1*.csv is allowed")
    if args.split != "val" and not args.final:
        raise SystemExit("test / external_holdout require --final (one-shot rule)")
    out_json = REPORT_DIR / ("final_eval.json" if args.split == "test" else f"final_eval_{args.split}.json")
    if args.split != "val" and out_json.exists() and not args.force:
        raise SystemExit(f"{out_json.name} already exists; the test split may only be evaluated once (use --force to override)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    manifest_prov = manifest_provenance(manifest_path)
    manifest_sha = manifest_prov["canonical_sha256"]
    manifest = load_manifest(manifest_path)
    prep = pd.read_csv(PREPROCESSED, dtype={"path": str, "preprocessed_path": str})[["path", "preprocessed_path"]].copy()
    manifest = manifest.copy()
    manifest["path_key"] = manifest["path"].map(path_key)
    prep["path_key"] = prep["path"].map(path_key)
    manifest = manifest.merge(prep[["path_key", "preprocessed_path"]], on="path_key", how="left", validate="one_to_one")

    val_rows = rows_of(manifest, "val")
    target_rows = rows_of(manifest, args.split)
    if len(target_rows) == 0:
        raise SystemExit(f"no rows for split={args.split}")
    if manifest.iloc[target_rows]["is_external_holdout"].nunique() != 1:
        raise SystemExit("split selection mixes external holdout and internal rows")

    features = np.load(FEAT, mmap_mode="r")
    feat_row = align_features(manifest, FEAT, FEAT_INDEX)
    import yaml
    s1_cfg = yaml.safe_load(STAGE1_CFG.read_text(encoding="utf-8"))
    s2_cfg = yaml.safe_load(STAGE2_CFG.read_text(encoding="utf-8"))

    Y = manifest[Y_COLS].to_numpy(dtype=np.float32)
    M = manifest[M_COLS].to_numpy(dtype=np.float32)
    groups = manifest["group_id"].astype(str).to_numpy()

    models: dict[str, dict] = {}
    provenance: dict[str, dict] = {}

    for stage, cfg in (("stage1", s1_cfg), ("stage2", s2_cfg)):
        for seed in SEEDS:
            key = f"{stage}_seed{seed}"
            ckpt_path = checkpoint_path(CHECKPOINT_ROOT, stage, seed)
            if not ckpt_path.is_file():
                raise SystemExit(f"missing checkpoint: {ckpt_path}")
            t0 = time.time()
            same_split = args.split == "val"
            if stage == "stage1":
                val_logits, info = stage1_predict(ckpt_path, cfg, features, feat_row, val_rows, device)
                tgt_logits = val_logits if same_split else stage1_predict(
                    ckpt_path, cfg, features, feat_row, target_rows, device)[0]
            else:
                val_logits, info = stage2_predict(ckpt_path, cfg, manifest.iloc[val_rows].reset_index(drop=True), device,
                                                  args.eval_batch_size, args.workers)
                tgt_logits = val_logits if same_split else stage2_predict(
                    ckpt_path, cfg, manifest.iloc[target_rows].reset_index(drop=True), device,
                    args.eval_batch_size, args.workers)[0]
            ckpt_manifest = info.get("manifest_sha256")
            if ckpt_manifest and ckpt_manifest != manifest_sha:
                raise SystemExit(f"{ckpt_path.name}: checkpoint manifest SHA does not match current manifest")
            prov = {
                "checkpoint": display_path(ckpt_path),
                "checkpoint_bytes": int(ckpt_path.stat().st_size),
                "ckpt_epoch": info.get("epoch"),
                "ckpt_manifest_sha256": ckpt_manifest,
                "predict_seconds": round(time.time() - t0, 1),
            }
            if args.split == "test" or args.hash_checkpoints:
                prov["checkpoint_sha256"] = sha256_file(ckpt_path)
            models[key] = {"val_logits": val_logits, "target_logits": tgt_logits}
            provenance[key] = prov
            print(f"[{key}] predicted val={len(val_rows)} target={len(target_rows)} in {prov['predict_seconds']}s")

    # ---------------- thresholds & calibration fitted on val, applied to target
    val_y, val_m = Y[val_rows], M[val_rows]
    tgt_y, tgt_m = Y[target_rows], M[target_rows]

    for key, rec in models.items():
        vlog, tlog = rec["val_logits"], rec["target_logits"]
        temps, thresholds, val_metrics = {}, {}, {}
        for j, label in enumerate(LABELS):
            known_v = val_m[:, j] == 1
            yv, lv = val_y[known_v, j], vlog[known_v, j]
            t = fit_temperature(lv, yv) if int(yv.sum()) >= CALIB_MIN_POS else 1.0
            temps[label] = t
            neg = lv[yv == 0]
            thresholds[label] = spec90_threshold(neg)
            thr = thresholds[label]
            val_metrics[label] = {
                "n_pos": int(yv.sum()),
                "auroc_raw": safe_auroc(yv, lv),
                "temperature": t,
                "thr_spec90": thr,
                "decision_rule": "positive if score > thr",
                "val_sensitivity": float((lv[yv == 1] > thr).mean()) if yv.sum() else float("nan"),
                "val_specificity": float((neg <= thr).mean()) if len(neg) else float("nan"),
            }
        rec["temps"], rec["thresholds"], rec["val_per_label"] = temps, thresholds, val_metrics

    # ---------------- target-split metrics per model
    results = {}
    for key, rec in models.items():
        tlog = rec["target_logits"]
        p_raw = sigmoid(tlog)
        p_cal = np.stack([sigmoid(tlog[:, j] / rec["temps"][l]) for j, l in enumerate(LABELS)], axis=1)
        block = label_block(tgt_y, tgt_m, tlog, p_cal, MIN_POS)
        raw_block = label_block(tgt_y, tgt_m, tlog, p_raw, MIN_POS)
        for label in LABELS:
            block[label]["ece_raw"] = raw_block[label]["ece"]
            block[label]["brier_raw"] = raw_block[label]["brier"]
        sens, spec = {}, {}
        for j, label in enumerate(LABELS):
            known = tgt_m[:, j] == 1
            yt = tgt_y[known, j]
            thr = rec["thresholds"][label]
            pos, neg = tlog[known, j][yt == 1], tlog[known, j][yt == 0]
            sens[label] = float((pos > thr).mean()) if len(pos) else float("nan")
            spec[label] = float((neg <= thr).mean()) if len(neg) else float("nan")
        for label in LABELS:
            if isinstance(block[label], dict):
                block[label]["sensitivity_at_val_thr"] = sens[label]
                block[label]["specificity_at_val_thr"] = spec[label]
        per_dataset = {}
        ds_col = manifest["dataset"].to_numpy()
        for ds in sorted(set(ds_col[target_rows])):
            mask = ds_col[target_rows] == ds
            yy = tgt_y[mask]
            per_dataset[ds] = {
                "n": int(mask.sum()),
                "dr_auroc": safe_auroc(yy[:, LABELS.index("dr")], tlog[mask, LABELS.index("dr")]),
                "macro_auroc": label_block(yy, tgt_m[mask], tlog[mask], p_cal[mask], MIN_POS)["_macro_auroc"],
            }
        results[key] = {"per_label": block, "per_dataset": per_dataset,
                        "thresholds_at_spec90": rec["thresholds"], "temperatures": rec["temps"],
                        "val_metrics_for_thresholds": rec["val_per_label"],
                        "dr_auroc": block["dr"]["auroc"], "dr_aupr": block["dr"]["aupr"],
                        "macro_auroc": block["_macro_auroc"], "macro_aupr": block["_macro_aupr"],
                        "macro_ece": block["_macro_ece"], "macro_ece_raw": raw_block["_macro_ece"],
                        "dr_ece": block["dr"]["ece"], "dr_ece_raw": block["dr"]["ece_raw"],
                        "dr_brier": block["dr"]["brier"],
                        "dr_sens_at_spec90": sens["dr"], "dr_spec_at_spec90": spec["dr"],
                        "macro_sens_at_spec90": float(np.nanmean([sens[l] for l in block["_macro_labels"]]))}

    # ---------------- group-aware bootstrap over the target split
    rng = np.random.default_rng(args.bootstrap_seed)
    group_rows = grouped_index(groups[target_rows])
    labeled_idx = [LABELS.index(l) for l in results["stage1_seed42"]["per_label"]["_macro_labels"]]
    boot = {}
    for key in models:
        boot[key] = bootstrap_metrics(tgt_y, tgt_m, models[key]["target_logits"], group_rows, labeled_idx,
                                      args.bootstrap, rng)
        print(f"[bootstrap] {key} done ({len(group_rows)} groups, {args.bootstrap} reps)")

    def seed_mean(arrs):
        return np.nanmean(np.stack(arrs), axis=0)

    stage_summary = {}
    for stage in ("stage1", "stage2"):
        for metric in ("dr_auroc", "macro_auroc", "dr_aupr"):
            arrs = [boot[f"{stage}_seed{s}"][metric] for s in SEEDS]
            mean_curve = seed_mean(arrs)
            point = float(np.mean([results[f"{stage}_seed{s}"][metric] for s in SEEDS]))
            stage_summary[f"{stage}_{metric}"] = {"point": point, "ci95": ci(mean_curve),
                                                  "per_seed": [float(results[f"{stage}_seed{s}"][metric]) for s in SEEDS]}

    paired = {}
    for metric in ("dr_auroc", "macro_auroc", "dr_aupr"):
        s1a = boot["stage1_seed42"][metric]
        s2a = boot["stage2_seed42"][metric]
        d42 = s2a - s1a
        dmean = seed_mean([boot[f"stage2_seed{s}"][metric] for s in SEEDS]) - seed_mean(
            [boot[f"stage1_seed{s}"][metric] for s in SEEDS])
        paired[metric] = {
            "seed42_delta_point": float(results["stage2_seed42"][metric] - results["stage1_seed42"][metric]),
            "seed42_delta_ci95": ci(d42),
            "seed42_p_two_sided": float(2 * min((d42 <= 0).mean(), (d42 >= 0).mean())),
            "seedmean_delta_point": float(np.mean([results[f"stage2_seed{s}"][metric] for s in SEEDS])
                                          - np.mean([results[f"stage1_seed{s}"][metric] for s in SEEDS])),
            "seedmean_delta_ci95": ci(dmean),
            "seedmean_p_two_sided": float(2 * min((dmean <= 0).mean(), (dmean >= 0).mean())),
        }

    # ---------------- self-check against the train-time run.json (val only)
    self_check = None
    if args.split == "val":
        self_check = {}
        val_ref = Path(args.val_reference)
        val_ref_df = pd.read_csv(val_ref) if val_ref.is_file() else None
        for stage in ("stage1", "stage2"):
            for seed in SEEDS:
                run_json = Path(args.run_json_root) / (f"{stage}_v2" if stage == "stage1" else stage) / f"seed_{seed}" / "run.json"
                got = results[f"{stage}_seed{seed}"]["macro_auroc"]
                expected_dr = None
                source = None
                if run_json.is_file():
                    expected = json.loads(run_json.read_text(encoding="utf-8"))
                    want = float(expected["best_val_macro_auroc"])
                    expected_dr = expected.get("best_val_per_label", {}).get("dr", {}).get("auroc")
                    source = str(run_json)
                elif val_ref_df is not None:
                    stage_key = "stage1_v2" if stage == "stage1" else "stage2"
                    hit = val_ref_df[(val_ref_df["stage"] == stage_key) & (val_ref_df["seed"].astype(str) == seed)]
                    if len(hit) != 1:
                        self_check[f"{stage}_seed{seed}"] = {
                            "expected_run_json": None, "recomputed": got, "abs_diff": None,
                            "match": None, "skipped": f"no unique row in {val_ref}",
                        }
                        continue
                    want = float(hit.iloc[0]["val_macro_auroc"])
                    expected_dr = float(hit.iloc[0]["val_dr_auroc"])
                    source = str(val_ref)
                else:
                    self_check[f"{stage}_seed{seed}"] = {
                        "expected_run_json": None, "recomputed": got, "abs_diff": None,
                        "match": None, "skipped": f"run.json and val reference not available: {run_json}",
                    }
                    continue
                self_check[f"{stage}_seed{seed}"] = {
                    "expected_run_json": want, "recomputed": got, "abs_diff": abs(got - want),
                    "match": bool(abs(got - want) <= 1e-6),
                    "expected_dr": expected_dr,
                    "recomputed_dr": results[f"{stage}_seed{seed}"]["dr_auroc"],
                    "reference": source,
                }

    payload = {
        "mode": args.split,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": {"path": display_path(manifest_path), "sha256": manifest_sha,
                     "file_sha256": manifest_prov["file_sha256"],
                     "provenance_mode": manifest_prov["provenance_mode"],
                     "n_rows_total": int(len(manifest)), "n_val": int(len(val_rows)),
                     "n_target": int(len(target_rows)),
                     "target_datasets": {k: int(v) for k, v in pd.Series(manifest.iloc[target_rows]["dataset"]).value_counts().items()}},
        "labels": list(LABELS),
        "label_roles": {"core_for_macro": CORE_LABELS, "exploratory": EXPLORATORY,
                        "macro_rule": f"labels with n_pos >= {MIN_POS} on the reported split"},
        "calibration": {"fitted_on": "val", "method": "per-label temperature scaling", "min_pos": CALIB_MIN_POS},
        "sensitivity_thresholds": {"fitted_on": "val", "target_specificity": SPEC_TARGET},
        "provenance": {
            "git_commit": git_commit(),
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "stage1_config": display_path(STAGE1_CFG),
            "stage2_config": display_path(STAGE2_CFG),
            "features": display_path(FEAT),
            "preprocessed_manifest": display_path(PREPROCESSED),
            "checkpoints": provenance,
        },
        "results": results,
        "stage_summary": stage_summary,
        "paired_delta_stage2_minus_stage1": paired,
        "bootstrap": {"reps": args.bootstrap, "seed": args.bootstrap_seed,
                      "resampling_unit": "group_id (whole dedup group)", "n_groups": int(len(group_rows))},
        "self_check_vs_run_json": self_check,
        "seconds": round(time.time() - started, 1),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- flat CSV
    rows = []
    for key, res in results.items():
        base = {"model": key, "dr_auroc": res["dr_auroc"], "dr_aupr": res["dr_aupr"],
                "macro_auroc": res["macro_auroc"], "macro_aupr": res["macro_aupr"],
                "dr_sens_at_spec90": res["dr_sens_at_spec90"], "dr_ece": res["dr_ece"], "macro_ece": res["macro_ece"]}
        rows.append({**base, "scope": "overall"})
        for label, v in res["per_label"].items():
            if isinstance(v, dict):
                rows.append({"model": key, "scope": f"label:{label}", "n_pos": v["n_pos"], "auroc": v["auroc"],
                             "aupr": v["aupr"], "ece": v["ece"], "sens_at_spec90": v["sensitivity_at_val_thr"],
                             "macro_auroc": np.nan, "dr_auroc": np.nan})
        for ds, v in res["per_dataset"].items():
            rows.append({"model": key, "scope": f"dataset:{ds}", "n_pos": v["n"],
                         "dr_auroc": v["dr_auroc"], "macro_auroc": v["macro_auroc"]})
    csv_path = out_json.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
    ext = None
    ext_path = REPORT_DIR / "final_eval_external_holdout.json"
    if args.split == "test" and ext_path.is_file():
        ext = json.loads(ext_path.read_text(encoding="utf-8"))
    md_path = (REPORT_DIR / "final_eval.md") if args.split == "test" else out_json.with_suffix(".md")
    write_markdown(payload, md_path, ext)
    print(f"wrote {out_json.name}, {csv_path.name}, {md_path.name}")
    if self_check:
        ok = all(v["match"] for v in self_check.values())
        print(json.dumps({k: {"expected": round(v["expected_run_json"], 6), "got": round(v["recomputed"], 6),
                              "ok": v["match"]} for k, v in self_check.items()}, indent=2))
        print(f"VAL SELF-CHECK: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
