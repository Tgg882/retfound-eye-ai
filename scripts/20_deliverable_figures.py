"""Build the deliverable figures + distribution summary (read-only w.r.t. test).

Test per-image scores were never persisted (the one-shot rule forbids re-running test
inference), so ROC / PR / calibration curves are drawn from the **val** split, which is
also where thresholds and temperature were fitted. Per-dataset AUROC and the bootstrap
CIs are drawn directly from outputs/reports/final_eval*.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import LABELS, M_COLS, Y_COLS, align_features, load_manifest  # noqa: E402
from evaluate_final import (FEAT, FEAT_INDEX, STAGE1_CFG, STAGE2_CFG, rows_of,  # noqa: E402
                            sigmoid, stage1_predict, stage2_predict)

FIG_D = ROOT / "deliverables" / "dataset" / "figures"
FIG_E = ROOT / "deliverables" / "evaluation" / "figures"
TAB_E = ROOT / "deliverables" / "evaluation" / "tables"
for d in (FIG_D, FIG_E, TAB_E):
    d.mkdir(parents=True, exist_ok=True)

manifest = load_manifest(ROOT / "outputs" / "manifest" / "manifest_v1.csv")
prep = pd.read_csv(ROOT / "outputs" / "manifest" / "manifest_v0_clean_preprocessed.csv",
                   dtype={"path": str, "preprocessed_path": str})[["path", "preprocessed_path"]]
manifest = manifest.merge(prep, on="path", how="left", validate="one_to_one")
Y = manifest[Y_COLS].to_numpy(dtype=np.float32)
M = manifest[M_COLS].to_numpy(dtype=np.float32)

# ------------------------------------------------------------------ dataset figs
order = ["DDR", "Fundus10", "APTOS2019", "MESSIDOR-2", "IDRID", "PAPILA"]
ct = pd.crosstab(manifest["dataset"], manifest["split"]).reindex(order)
ct.to_csv(TAB_E / "split_by_dataset.csv")

fig, ax = plt.subplots(figsize=(8, 4.2))
left = np.zeros(len(ct))
for split, color in (("train", "#4C78A8"), ("val", "#F58518"), ("test", "#54A24B")):
    ax.barh(ct.index, ct[split].values, left=left, label=split, color=color)
    left += ct[split].values
for i, total in enumerate(ct.sum(axis=1).values):
    ax.text(total + 120, i, f"{int(total):,}", va="center", fontsize=9)
ax.set_xlabel("images"); ax.set_title("Dataset composition by split (manifest v1, n=23,294)")
ax.legend(title="split"); ax.set_xlim(0, ct.sum(axis=1).max() * 1.15)
fig.tight_layout(); fig.savefig(FIG_D / "dataset_composition.png", dpi=160); plt.close(fig)

rate = pd.DataFrame(index=LABELS, columns=order, dtype=float)
for l in LABELS:
    j = LABELS.index(l)
    known = M[:, j] == 1
    for ds in order:
        sel = known & (manifest["dataset"].to_numpy() == ds)
        rate.loc[l, ds] = float(Y[sel, j].mean()) if sel.sum() else np.nan
fig, ax = plt.subplots(figsize=(7.6, 5.2))
im = ax.imshow(rate.values.astype(float), cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=25, ha="right")
ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
for i in range(len(LABELS)):
    for j in range(len(order)):
        v = rate.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if v < 0.6 else "white")
        else:
            ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="grey")
ax.set_title("Positive rate among annotated rows (manifest v1)")
fig.colorbar(im, ax=ax, fraction=0.03, label="positive rate")
fig.tight_layout(); fig.savefig(FIG_D / "label_positive_rate_heatmap.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.8))
counts = [int((manifest["split"] == s).sum()) for s in ("train", "val", "test")]
hold = int(manifest["is_external_holdout"].sum())
bars = ax.bar(["train", "val", "test (internal)", "IDRID official\ntest holdout"], counts + [hold],
              color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
for b, v in zip(bars, counts + [hold]):
    ax.text(b.get_x() + b.get_width() / 2, v + 180, f"{v:,}", ha="center", fontsize=9)
ax.set_ylabel("images"); ax.set_title("Split sizes (holdout is a subset of the 3,577 test rows)")
fig.tight_layout(); fig.savefig(FIG_D / "split_counts.png", dpi=160); plt.close(fig)

qc = pd.crosstab(manifest["dataset"], manifest["quality_flag"]).reindex(order).fillna(0)
for c in ("good", "usable", "reject"):
    if c not in qc:
        qc[c] = 0
fig, ax = plt.subplots(figsize=(8, 4.2))
bottom = np.zeros(len(qc))
for c, color in (("good", "#54A24B"), ("usable", "#EECA3B"), ("reject", "#E45756")):
    ax.barh(qc.index, qc[c].values, left=bottom, label=c, color=color)
    bottom += qc[c].values
ax.set_xlabel("images"); ax.set_title("Quality flags by dataset (good / usable / reject)")
ax.legend(title="quality_flag"); ax.set_xlim(0, qc.sum(axis=1).max() * 1.12)
fig.tight_layout(); fig.savefig(FIG_D / "quality_flags.png", dpi=160); plt.close(fig)

# ------------------------------------------------------------- distribution csv
rows = []
for ds in order:
    for split in ("train", "val", "test"):
        sel = (manifest["dataset"] == ds).to_numpy() & (manifest["split"] == split).to_numpy()
        rec = {"dataset": ds, "split": split, "n": int(sel.sum())}
        for l in LABELS:
            j = LABELS.index(l)
            known = sel & (M[:, j] == 1)
            rec[f"n_pos_{l}"] = int(Y[known, j].sum())
            rec[f"pos_rate_{l}"] = round(float(Y[known, j].mean()), 5) if known.sum() else np.nan
            rec[f"masked_frac_{l}"] = round(float(1 - M[sel, j].mean()), 5) if sel.sum() else np.nan
        rec["quality_good"] = int(((manifest["quality_flag"] == "good") & sel).sum())
        rec["quality_usable"] = int(((manifest["quality_flag"] == "usable") & sel).sum())
        rec["quality_reject"] = int(((manifest["quality_flag"] == "reject") & sel).sum())
        rows.append(rec)
tot = {"dataset": "ALL", "split": "all", "n": int(len(manifest))}
for l in LABELS:
    j = LABELS.index(l)
    known = M[:, j] == 1
    tot[f"n_pos_{l}"] = int(Y[known, j].sum())
    tot[f"pos_rate_{l}"] = round(float(Y[known, j].mean()), 5)
    tot[f"masked_frac_{l}"] = round(float(1 - M[:, j].mean()), 5)
for c in ("good", "usable", "reject"):
    tot[f"quality_{c}"] = int((manifest["quality_flag"] == c).sum())
rows.append(tot)
pd.DataFrame(rows).to_csv(ROOT / "deliverables" / "dataset" / "distribution_summary.csv", index=False)

# ------------------------------------------------------------- evaluation figs
val_rows = rows_of(manifest, "val")
feat = np.load(FEAT, mmap_mode="r")
feat_row = align_features(manifest, FEAT, FEAT_INDEX)
s1_cfg = yaml.safe_load(STAGE1_CFG.read_text(encoding="utf-8"))
s2_cfg = yaml.safe_load(STAGE2_CFG.read_text(encoding="utf-8"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
j_dr = LABELS.index("dr")
known = M[val_rows, j_dr] == 1
yv = Y[val_rows][known, j_dr]
print("val DR known rows:", int(known.sum()), "pos:", int(yv.sum()))

s1_logits, _ = stage1_predict(ROOT / "outputs/runs/stage1_v2/seed_42/best.pt", s1_cfg, feat, feat_row, val_rows, device)
s2_logits, _ = stage2_predict(ROOT / "outputs/runs/stage2/seed_42/best.pt", s2_cfg,
                              manifest.iloc[val_rows].reset_index(drop=True), device, 16, 0)

ev = json.loads((ROOT / "outputs/reports/final_eval.json").read_text(encoding="utf-8"))
temps = ev["results"]["stage2_seed42"]["temperatures"]
s2_cal = sigmoid(s2_logits[known, j_dr] / temps["dr"])

curves = {
    "Stage 1 v2 seed42 (frozen)": sigmoid(s1_logits[known, j_dr]),
    "Stage 2 seed42 (fine-tuned)": sigmoid(s2_logits[known, j_dr]),
}
fig, ax = plt.subplots(figsize=(5.4, 5))
for name, sc in curves.items():
    fpr, tpr, _ = roc_curve(yv, sc)
    ax.plot(fpr, tpr, label=f"{name}  AUC={np.trapz(tpr, fpr):.4f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("DR ROC — val split (test scores not persisted)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(FIG_E / "roc_val_dr.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 5))
base = float(yv.mean())
for name, sc in curves.items():
    prec, rec, _ = precision_recall_curve(yv, sc)
    ax.plot(rec, prec, label=f"{name}  AP={np.trapz(prec[::-1], rec[::-1]):.4f}")
ax.axhline(base, ls="--", color="k", lw=0.8, label=f"prevalence {base:.3f}")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("DR precision-recall — val split")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig(FIG_E / "pr_val_dr.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.4, 5))
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
for name, sc, style in (("Stage 2 raw", curves["Stage 2 seed42 (fine-tuned)"], "o-"),
                        ("Stage 2 temperature-scaled", s2_cal, "s-")):
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(sc, edges[1:-1]), 0, 9)
    xs = [sc[idx == b].mean() for b in range(10) if (idx == b).any()]
    ys = [yv[idx == b].mean() for b in range(10) if (idx == b).any()]
    ax.plot(xs, ys, style, ms=4, lw=1.2, label=name)
ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
ax.set_title("DR calibration — val split (S2 seed42)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG_E / "calibration_val_dr.png", dpi=160); plt.close(fig)

# per-dataset heatmap (test, from the frozen json)
datasets = sorted(ev["results"]["stage2_seed42"]["per_dataset"])
mat = np.array([[ev["results"][f"stage1_seed{s}"]["per_dataset"].get(d, {}).get("dr_auroc", np.nan) for d in datasets]
                for s in ("42", "43", "44")] +
               [[ev["results"][f"stage2_seed{s}"]["per_dataset"].get(d, {}).get("dr_auroc", np.nan) for d in datasets]
                for s in ("42", "43", "44")])
fig, ax = plt.subplots(figsize=(7.6, 4.6))
im = ax.imshow(mat, cmap="viridis", vmin=0.6, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, rotation=20, ha="right")
ax.set_yticks(range(6)); ax.set_yticklabels([f"S1 s{s}" for s in ("42", "43", "44")] + [f"S2 s{s}" for s in ("42", "43", "44")])
for i in range(6):
    for j in range(len(datasets)):
        v = mat[i, j]
        ax.text(j, i, "n/a" if not np.isfinite(v) else f"{v:.3f}", ha="center", va="center", fontsize=8,
                color="white" if (np.isfinite(v) and v < 0.85) else "black")
ax.set_title("DR AUROC per dataset — internal test split")
fig.colorbar(im, ax=ax, fraction=0.03)
fig.tight_layout(); fig.savefig(FIG_E / "per_dataset_auroc_heatmap.png", dpi=160); plt.close(fig)

# bootstrap CI forest (test + holdout)
hold = json.loads((ROOT / "outputs/reports/final_eval_external_holdout.json").read_text(encoding="utf-8"))
labels_, pts, los, his, cols = [], [], [], [], []
for key, name in (("stage1_dr_auroc", "S1 3-seed"), ("stage2_dr_auroc", "S2 3-seed")):
    for src, suffix, color in ((ev, " (test)", "#4C78A8"), (hold, " (IDRID holdout)", "#B279A2")):
        s = src["stage_summary"][key]
        labels_.append(name + suffix); pts.append(s["point"]); los.append(s["ci95"][0]); his.append(s["ci95"][1]); cols.append(color)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
y = np.arange(len(labels_))[::-1]
ax.errorbar(pts, y, xerr=[np.array(pts) - np.array(los), np.array(his) - np.array(pts)],
            fmt="o", capsize=4, lw=1.4, color="black", mfc="white")
for xi, yi, c in zip(pts, y, cols):
    ax.plot(xi, yi, "o", ms=8, color=c)
for xi, yi, lo, hi in zip(pts, y, los, his):
    ax.text(hi + 0.004, yi, f"{xi:.4f} [{lo:.4f}, {hi:.4f}]", va="center", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(labels_)
ax.set_xlabel("DR AUROC (bootstrap 95% CI)")
ax.set_title("DR AUROC with group-aware 95% CI")
ax.set_xlim(min(los) - 0.03, max(his) + 0.13)
fig.tight_layout(); fig.savefig(FIG_E / "bootstrap_ci_dr.png", dpi=160); plt.close(fig)
print("figures written")
