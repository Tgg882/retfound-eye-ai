#!/usr/bin/env python
"""R2-1 preprocessing: retinal crop + 256x256 (AutoMorph fallback, plan 6.1).

Input : outputs/manifest/manifest_v0_clean.csv
Output: outputs/preprocessed/<dataset>/<stem>.png
        outputs/manifest/manifest_v0_clean_preprocessed.csv
        outputs/reports/preprocess_failures.csv
        outputs/reports/preprocess_report.json

Fallback algorithm (no AutoMorph / Docker):
  grayscale -> threshold >10 -> largest external contour -> bbox + 2% margin
  -> resize 256x256 cubic.  If mask coverage < 30% use a centered square crop.

Every failure is written to preprocess_failures.csv; nothing is dropped silently.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from paths import path_key, resolve_data_path  # noqa: E402

MANIFEST = PROJECT_ROOT / "outputs" / "manifest" / "manifest_v0_clean.csv"
OUT_ROOT = PROJECT_ROOT / "outputs" / "preprocessed"
OUT_MAP = PROJECT_ROOT / "outputs" / "manifest" / "manifest_v0_clean_preprocessed.csv"
FAIL_CSV = PROJECT_ROOT / "outputs" / "reports" / "preprocess_failures.csv"
REPORT = PROJECT_ROOT / "outputs" / "reports" / "preprocess_report.json"
SIZE = 256
MIN_COVERAGE = 0.30


def center_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


def process_one(task: dict) -> dict:
    src = Path(task.get("src_path", task["path"]))
    dst = Path(task["out_path"])
    try:
        raw = np.fromfile(str(src), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = (gray > 10).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        method = "largest_contour"
        coverage = 0.0
        crop = img
        if contours:
            contour = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(contour)
            mx, my = int(0.02 * w), int(0.02 * h)
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1, y1 = min(w, x + cw + mx), min(h, y + ch + my)
            crop = img[y0:y1, x0:x1]
            area = crop.shape[0] * crop.shape[1]
            coverage = float(cv2.contourArea(contour) / area) if area else 0.0
        else:
            method = "center_crop_no_contour"
        if crop.size == 0:
            raise ValueError("empty crop")
        if coverage < MIN_COVERAGE:
            crop = center_square(img)
            method = f"{method}+center_crop"
        out = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".png", out)
        if not ok:
            raise ValueError("png encode failed")
        buf.tofile(str(dst))
        return {
            "path": task["path"],
            "dataset": task["dataset"],
            "preprocessed_path": str(dst),
            "mask_coverage": coverage,
            "method": method,
            "src_h": h,
            "src_w": w,
            "ok": True,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - logged, never silently dropped
        return {
            "path": task["path"],
            "dataset": task["dataset"],
            "preprocessed_path": "",
            "mask_coverage": float("nan"),
            "method": "",
            "src_h": 0,
            "src_w": 0,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--out-map", default=str(OUT_MAP))
    ap.add_argument("--fail-csv", default=str(FAIL_CSV))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--data-root", default=None, help="override CFP9_DATA_ROOT")
    ap.add_argument("--relative-paths", action="store_true",
                    help="write preprocessed_path relative to the project root")
    args = ap.parse_args()
    df = pd.read_csv(args.manifest, dtype={"path": str})
    out_root = Path(args.out_root)
    tasks = []
    for row in df.itertuples():
        stem = Path(row.path).stem
        tasks.append(
            {
                "path": row.path,
                "src_path": str(resolve_data_path(row.path, args.data_root)),
                "dataset": row.dataset,
                "out_path": str(out_root / row.dataset / f"{stem}.png"),
            }
        )
    print(f"preprocessing {len(tasks)} images -> {out_root}")

    results = []
    with Pool(processes=8) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, tasks, chunksize=32), start=1):
            results.append(res)
            if i % 2000 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)

    res_df = pd.DataFrame(results).sort_values("path").reset_index(drop=True)
    failures = res_df[~res_df["ok"]]
    failures.to_csv(args.fail_csv, index=False, encoding="utf-8")
    if args.relative_paths:
        res_df["preprocessed_path"] = res_df["preprocessed_path"].map(path_key)
    res_df[["path", "dataset", "preprocessed_path", "mask_coverage", "method", "src_h", "src_w"]].to_csv(
        args.out_map, index=False, encoding="utf-8"
    )

    ok = res_df[res_df["ok"]]
    cov = ok["mask_coverage"].to_numpy(dtype=float)
    report = {
        "n_total": int(len(res_df)),
        "n_ok": int(len(ok)),
        "n_failed": int(len(failures)),
        "coverage_p05": float(np.percentile(cov, 5)),
        "coverage_p50": float(np.percentile(cov, 50)),
        "coverage_p95": float(np.percentile(cov, 95)),
        "fraction_coverage_0.30_0.95": float(((cov >= 0.30) & (cov <= 0.95)).mean()),
        "fraction_center_crop": float(ok["method"].str.contains("center_crop").mean()),
        "min_source_side": int(min(ok["src_h"].min(), ok["src_w"].min())),
        "failures_by_dataset": failures["dataset"].value_counts().to_dict() if len(failures) else {},
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
