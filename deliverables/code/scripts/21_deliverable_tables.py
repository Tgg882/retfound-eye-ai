"""Build deliverable tables strictly from frozen manifest / final_eval JSON.

No inference, no test re-run: every number is a re-aggregation of existing files.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "evaluation" / "tables"
DATA = ROOT / "deliverables" / "dataset"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

LABELS = ["normal", "dr", "glaucoma", "csc", "disc_edema",
          "macular_scar", "myopia", "retinal_det", "retinitis_pig"]
EXPLORATORY = {"csc"}

m = pd.read_csv(ROOT / "outputs" / "manifest" / "manifest_v1.csv", dtype={"path": str})

# ---------------------------------------------------------------- distribution
rows = []
for ds in list(dict.fromkeys(m["dataset"])) + ["ALL"]:
    sub_ds = m if ds == "ALL" else m[m["dataset"] == ds]
    for sp in ["train", "val", "test", "all"]:
        sub = sub_ds if sp == "all" else sub_ds[sub_ds["split"] == sp]
        if len(sub) == 0:
            continue
        row = {"dataset": ds, "split": sp, "n": int(len(sub))}
        pos_counts = {}
        for lab in LABELS:
            y, mask = sub[f"y_{lab}"].to_numpy(), sub[f"m_{lab}"].to_numpy()
            n_pos = int(y.sum())
            n_known = int(mask.sum())
            pos_counts[lab] = (n_pos, n_known)
            row[f"n_pos_{lab}"] = n_pos
            row[f"pos_rate_{lab}"] = round(n_pos / n_known, 6) if n_known else np.nan
            row[f"masked_frac_{lab}"] = round(1.0 - n_known / len(sub), 6)
        for q in ["good", "usable", "reject"]:
            row[f"quality_{q}"] = int((sub["quality_flag"] == q).sum())
        known_labels = {k: v for k, v in pos_counts.items() if v[1] > 0 and v[0] > 0}
        rarest = min(known_labels, key=lambda k: known_labels[k][0]) if known_labels else ""
        row["rarest_annotated_label"] = rarest
        row["rarest_pos"] = known_labels[rarest][0] if rarest else 0
        row["imbalance_neg_per_pos"] = (
            round((known_labels[rarest][1] - known_labels[rarest][0]) / known_labels[rarest][0], 4)
            if rarest else np.nan
        )
        row["n_multilabel_positive"] = int((sub[[f"y_{l}" for l in LABELS]].sum(axis=1) > 1).sum())
        row["n_all_zero"] = int((sub[[f"y_{l}" for l in LABELS]].sum(axis=1) == 0).sum())
        rows.append(row)

dist = pd.DataFrame(rows)
dist.to_csv(DATA / "distribution_summary.csv", index=False, encoding="utf-8")

# ------------------------------------------------------------------- eval json
ev = json.loads((ROOT / "outputs/reports/final_eval.json").read_text(encoding="utf-8"))
hold = json.loads((ROOT / "outputs/reports/final_eval_external_holdout.json").read_text(encoding="utf-8"))
val = json.loads((ROOT / "outputs/reports/final_eval_val.json").read_text(encoding="utf-8"))
SEEDS = ["42", "43", "44"]


def summary_table(payload, scope):
    out = []
    for key, s in payload["stage_summary"].items():
        per = np.array(s["per_seed"], dtype=float)
        out.append({
            "scope": scope, "metric": key, "mean_3seed": s["point"],
            "std_3seed": float(per.std(ddof=1)), "ci95_low": s["ci95"][0], "ci95_high": s["ci95"][1],
            "seed42": per[0], "seed43": per[1], "seed44": per[2],
        })
    return pd.DataFrame(out)


summ = pd.concat([summary_table(ev, "internal_test"), summary_table(hold, "idrid_holdout_103")],
                 ignore_index=True)
summ.to_csv(OUT / "stage_summary_3seed.csv", index=False, encoding="utf-8")

# paired deltas
pairs = []
for scope, payload in (("internal_test", ev), ("idrid_holdout_103", hold)):
    for metric, d in payload["paired_delta_stage2_minus_stage1"].items():
        pairs.append({
            "scope": scope, "metric": metric,
            "seed42_delta": d["seed42_delta_point"], "seed42_ci_low": d["seed42_delta_ci95"][0],
            "seed42_ci_high": d["seed42_delta_ci95"][1], "seed42_p": d["seed42_p_two_sided"],
            "seedmean_delta": d["seedmean_delta_point"], "seedmean_ci_low": d["seedmean_delta_ci95"][0],
            "seedmean_ci_high": d["seedmean_delta_ci95"][1], "seedmean_p": d["seedmean_p_two_sided"],
        })
pd.DataFrame(pairs).to_csv(OUT / "paired_stage2_minus_stage1.csv", index=False, encoding="utf-8")

# per-label / per-dataset means and std across seeds
def collect(payload, section):
    recs = []
    for model in payload["results"]:
        stage = "stage1" if model.startswith("stage1") else "stage2"
        seed = model.split("_")[-1]
        for name, d in payload["results"][model][section].items():
            rec = {"scope": payload["mode"], "stage": stage, "seed": seed, "name": name}
            rec.update({k: v for k, v in d.items() if not isinstance(v, dict)})
            recs.append(rec)
    raw = pd.DataFrame(recs)
    keys = ["scope", "stage", "name"]
    agg = raw.groupby(keys, dropna=False).agg(
        n_pos=("n_pos", "first") if "n_pos" in raw else ("auroc", "size"),
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", lambda x: x.std(ddof=1)),
        aupr_mean=("aupr", "mean"), aupr_std=("aupr", "mean") if False else ("aupr", lambda x: x.std(ddof=1)),
        ece_mean=("ece", "mean"), sens_mean=("sensitivity_at_val_thr", "mean"),
        spec_mean=("specificity_at_val_thr", "mean"),
    ).reset_index()
    return raw, agg


def per_label_table(payload, scope):
    recs = []
    for model, r in payload["results"].items():
        stage = "stage1" if model.startswith("stage1") else "stage2"
        for lab, d in r["per_label"].items():
            if lab.startswith("_"):
                continue
            recs.append({"scope": scope, "stage": stage, "label": lab, **{k: v for k, v in d.items() if not isinstance(v, dict)}})
    raw = pd.DataFrame(recs)
    agg = raw.groupby(["scope", "stage", "label"], dropna=False).agg(
        n_pos=("n_pos", "first"), auroc=("auroc", "mean"), auroc_std=("auroc", lambda x: x.std(ddof=1)),
        aupr=("aupr", "mean"), ece=("ece", "mean"), sens=("sensitivity_at_val_thr", "mean"),
        spec=("specificity_at_val_thr", "mean"), used_for_macro=("used_for_macro", "first"),
    ).reset_index()
    wide = agg.pivot(index="label", columns="stage")
    out = pd.DataFrame(index=agg["label"].unique())
    for metric in ["auroc", "aupr", "ece", "sens", "spec", "n_pos"]:
        for stage in ["stage1", "stage2"]:
            out[f"{stage}_{metric}"] = wide[(metric, stage)]
    out["delta_auroc"] = out["stage2_auroc"] - out["stage1_auroc"]
    expl = set(payload.get("label_roles", {}).get("exploratory", []))
    core = set(payload.get("label_roles", {}).get("core_for_macro", []))
    out["role"] = ["exploratory" if l in expl else "core" for l in out.index]
    out["core_for_macro"] = [bool(l in core) for l in out.index]
    out = out.reset_index().rename(columns={"index": "label"})
    out.insert(0, "scope", scope)
    return out


labs = pd.concat([per_label_table(ev, "internal_test"), per_label_table(hold, "idrid_holdout_103")],
                 ignore_index=True)
labs.to_csv(OUT / "per_label_3seed.csv", index=False, encoding="utf-8")


def per_dataset_table(payload, scope):
    recs = []
    for model, r in payload["results"].items():
        stage = "stage1" if model.startswith("stage1") else "stage2"
        for ds, d in r["per_dataset"].items():
            recs.append({"scope": scope, "stage": stage, "dataset": ds, **d})
    raw = pd.DataFrame(recs)
    agg = raw.groupby(["scope", "stage", "dataset"], dropna=False).agg(
        n=("n", "first"), dr_auroc=("dr_auroc", "mean"), dr_auroc_std=("dr_auroc", lambda x: x.std(ddof=1)),
        macro_auroc=("macro_auroc", "mean"), macro_auroc_std=("macro_auroc", lambda x: x.std(ddof=1)),
    ).reset_index()
    wide = agg.pivot(index=["dataset", "n"], columns="stage")
    out = pd.DataFrame(index=wide.index)
    for metric in ["dr_auroc", "macro_auroc"]:
        for stage in ["stage1", "stage2"]:
            out[f"{stage}_{metric}"] = wide[(metric, stage)]
    out["delta_dr_auroc"] = out["stage2_dr_auroc"] - out["stage1_dr_auroc"]
    out.insert(0, "scope", scope)
    return out.reset_index()


pd.concat([per_dataset_table(ev, "internal_test"), per_dataset_table(hold, "idrid_holdout_103")],
          ignore_index=True).to_csv(OUT / "per_dataset_3seed.csv", index=False, encoding="utf-8")

# ------------------------------------------------------------- checkpoint list
inv = []
for key, d in ev["provenance"]["checkpoints"].items():
    stage, seed = key.split("_seed")
    inv.append({"role": "baseline" if stage == "stage1" else ("primary" if seed == "42" else "stability"),
                "stage": "Stage 1 v2" if stage == "stage1" else "Stage 2", "seed": int(seed),
                "best_epoch": d["ckpt_epoch"], "path": d["checkpoint"].replace("\\", "/"),
                "bytes": d["checkpoint_bytes"], "sha256": d["checkpoint_sha256"],
                "manifest_sha256": d["ckpt_manifest_sha256"]})
pd.DataFrame(inv).sort_values(["stage", "seed"]).to_csv(
    ROOT / "deliverables/model/checkpoint_inventory.csv", index=False, encoding="utf-8")

print("wrote distribution_summary", len(dist), "rows")
print("wrote eval tables:", sorted(p.name for p in OUT.glob("*.csv")))

