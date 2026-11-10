#!/usr/bin/env python
"""R2-2 quality gate (EyeQ / cleanvision fallback, plan 6.2).

Input : outputs/manifest/manifest_v0_clean_preprocessed.csv
Output: outputs/reports/quality.csv
        outputs/reports/quality_report.json

Fallback metrics on the 256x256 preprocessed image:
  * blur        : variance of Laplacian (threshold = 1st percentile of the train split)
  * dark/light  : mean grey outside [10, 245]
  * grayscale   : saturation std < 3
  * reject      : any hard issue, or >= 2 issues in total
  * good / usable: score >= 0.5 / otherwise
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREP_MAP = PROJECT_ROOT / "outputs" / "manifest" / "manifest_v0_clean_preprocessed.csv"
MANIFEST = PROJECT_ROOT / "outputs" / "manifest" / "manifest_v0_clean.csv"
OUT_CSV = PROJECT_ROOT / "outputs" / "reports" / "quality.csv"
REPORT = PROJECT_ROOT / "outputs" / "reports" / "quality_report.json"


def metrics_one(task: dict) -> dict:
    try:
        raw = np.fromfile(task["preprocessed_path"], dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("decode failed")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean = float(gray.mean())
        sat_std = float(hsv[:, :, 1].std())
        return {
            "path": task["path"],
            "laplacian_var": lap,
            "mean_brightness": mean,
            "sat_std": sat_std,
            "decode_ok": True,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "path": task["path"],
            "laplacian_var": np.nan,
            "mean_brightness": np.nan,
            "sat_std": np.nan,
            "decode_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    prep = pd.read_csv(PREP_MAP, dtype={"path": str})
    manifest = pd.read_csv(MANIFEST, dtype={"path": str})[["path", "split"]]
    df = prep.merge(manifest, on="path", how="left", validate="one_to_one")

    tasks = df[["path", "preprocessed_path"]].to_dict("records")
    with Pool(processes=8) as pool:
        results = pool.map(metrics_one, tasks, chunksize=64)
    metrics = pd.DataFrame(results)

    df = df.merge(metrics, on="path", how="left", validate="one_to_one")
    train = df[(df["split"] == "train") & df["decode_ok"]]
    blur_threshold = float(np.percentile(train["laplacian_var"], 1)) if len(train) else 0.0
    sharp_ref = float(np.percentile(train["laplacian_var"], 50)) if len(train) else 1.0

    def classify(row) -> tuple[float, str, str]:
        issues = []
        if not row["decode_ok"]:
            return 0.0, "reject", "decode_failed"
        if row["laplacian_var"] < blur_threshold:
            issues.append("blur")
        if row["mean_brightness"] < 10:
            issues.append("dark")
        if row["mean_brightness"] > 245:
            issues.append("light")
        if row["sat_std"] < 3:
            issues.append("grayscale")
        sharp = min(1.0, row["laplacian_var"] / sharp_ref) if sharp_ref > 0 else 0.0
        bright = max(0.0, 1.0 - abs(row["mean_brightness"] - 128.0) / 128.0)
        color = min(1.0, row["sat_std"] / 15.0)
        score = float(0.5 * sharp + 0.3 * bright + 0.2 * color)
        hard = {"blur", "dark", "light", "grayscale"} & set(issues)
        if hard or len(issues) >= 2:
            flag = "reject"
        elif score >= 0.5:
            flag = "good"
        else:
            flag = "usable"
        return score, flag, "|".join(issues)

    out = df.apply(lambda r: pd.Series(classify(r), index=["quality_score", "quality_flag", "issues"]), axis=1)
    quality = pd.concat([df[["path", "dataset", "split", "preprocessed_path", "laplacian_var", "mean_brightness", "sat_std"]], out], axis=1)
    quality.to_csv(OUT_CSV, index=False, encoding="utf-8")

    report = {
        "n": int(len(quality)),
        "n_decode_failed": int((~quality["split"].notna()).sum()),
        "blur_threshold_p1_train": blur_threshold,
        "flag_counts": quality["quality_flag"].value_counts().to_dict(),
        "reject_fraction": float((quality["quality_flag"] == "reject").mean()),
        "reject_by_dataset": quality[quality["quality_flag"] == "reject"]["dataset"].value_counts().to_dict(),
        "issue_counts": {
            issue: int(quality["issues"].fillna("").str.contains(issue).sum())
            for issue in ("blur", "dark", "light", "grayscale", "decode_failed")
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["reject_fraction"] < 0.10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
