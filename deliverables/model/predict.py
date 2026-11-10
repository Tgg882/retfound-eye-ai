#!/usr/bin/env python
"""Single-image CFP-9 inference (Stage 2 primary; Stage 1 supported).

  python predict.py --image path/to/fundus.jpg
  python predict.py --image path/to/fundus.jpg --ckpt checkpoints/stage1_v2_seed42_best.pt --stage 1 \
      --encoder-ckpt /path/to/RETFound_mae_natureCFP.pth

The image path is preprocessed exactly like scripts/02_preprocess.py (largest-contour
field-of-view crop + 2% margin -> 256x256), then Resize(256), CenterCrop(224),
ImageNet normalisation.  Use --preprocessed only if the input is already the 256x256
PNG produced by that script.

Thresholds are applied to raw logits (fit on val at 90% specificity). Temperature is
used only for calibrated probabilities. Nothing is refit here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE / "checkpoints" / "stage2_seed42_best.pt"
DEFAULT_THR = HERE / "thresholds.json"
DEFAULT_CAL = HERE / "calibration.json"


class CFP9Head(nn.Module):
    """9-label head; identical definition to src/head.py."""

    def __init__(self, dim: int = 1024, hidden: int = 1024, n: int = 9, p: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(p), nn.Linear(hidden, n)
        )

    def forward(self, x):
        return self.net(x)


def find_retfound(explicit: str | None) -> Path:
    cands = []
    if explicit:
        cands.append(Path(explicit))
    cands += [HERE / "third_party" / "RETFound-main",
              HERE.parent / "code" / "third_party" / "RETFound-main",
              HERE.parents[1] / "RETFound-main",
              Path.cwd() / "RETFound-main"]
    for c in cands:
        if (c / "models_vit.py").is_file():
            return c
    raise SystemExit("RETFound-main not found; pass --retfound-code /path/to/RETFound-main")


def find_encoder(explicit: str | None) -> Path | None:
    cands = []
    if explicit:
        cands.append(Path(explicit))
    cands += [HERE / "checkpoints" / "RETFound_mae_natureCFP.pth",
              HERE.parents[1] / "model" / "FoundmaenatureCFP" / "RETFound_mae_natureCFP.pth",
              HERE.parents[1] / "RETFound-main" / "RETFound_mae_natureCFP.pth"]
    for c in cands:
        if c.is_file():
            return c
    return None


def attach_forward(base) -> None:
    def _forward(self, x):
        f = self.forward_head(self.forward_features(x), pre_logits=True)
        if f.ndim == 3:
            f = f.squeeze(1)
        return self.head(torch.nn.functional.normalize(f.float(), dim=1))
    base.forward = types.MethodType(_forward, base)


def build_model(ckpt_path: Path, retfound_dir: Path, device, encoder_ckpt: Path | None):
    sys.path.insert(0, str(retfound_dir))
    from models_vit import RETFound_mae

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    labels = list(ckpt.get("labels", []))
    configured = list(json.loads((HERE / "label_mapping.json").read_text(encoding="utf-8"))["labels"])
    if labels != configured:
        raise SystemExit(f"checkpoint label order mismatch: {labels} != {configured}")

    cfg = ckpt.get("config", {})
    base = RETFound_mae(global_pool=True, num_classes=0, img_size=224, drop_path_rate=0.0)
    head = CFP9Head(dim=1024, hidden=int(cfg.get("hidden", 1024)), n=len(configured),
                    p=float(cfg.get("dropout", 0.3)))
    base.head = head

    if "model" in ckpt:
        base.load_state_dict(ckpt["model"], strict=True)
        source = "full_checkpoint"
    else:
        head.load_state_dict(ckpt["head"], strict=True)
        if encoder_ckpt is None or not encoder_ckpt.is_file():
            raise SystemExit("Stage 1 checkpoint needs RETFound encoder weights; pass --encoder-ckpt")
        enc = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
        st = dict(enc["model"] if isinstance(enc, dict) and "model" in enc else enc)
        if "fc_norm.weight" not in st and "norm.weight" in st:
            st["fc_norm.weight"], st["fc_norm.bias"] = st["norm.weight"], st["norm.bias"]
        msg = base.load_state_dict(st, strict=False)
        # The MAE checkpoint also carries decoder/mask-token weights; training used
        # the same strict=False encoder load (src/encoder.py).
        if msg.missing_keys:
            print(f"warning: {len(msg.missing_keys)} missing encoder keys", file=sys.stderr)
        if msg.unexpected_keys:
            print(f"note: ignored {len(msg.unexpected_keys)} MAE decoder/head keys", file=sys.stderr)
        source = f"stage1_head+encoder:{encoder_ckpt} (missing={len(msg.missing_keys)}, unexpected_ignored={len(msg.unexpected_keys)})"
    attach_forward(base)
    return base.to(device).eval(), ckpt, source


def crop_fundus(img: np.ndarray) -> np.ndarray:
    """Byte-for-byte logic of scripts/02_preprocess.py (AutoMorph fallback)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop, coverage = img, 0.0
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(contour)
        mx, my = int(0.02 * w), int(0.02 * h)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + cw + mx), min(h, y + ch + my)
        crop = img[y0:y1, x0:x1]
        area = crop.shape[0] * crop.shape[1]
        coverage = float(cv2.contourArea(contour) / area) if area else 0.0
    if crop.size == 0 or coverage < 0.30:
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        crop = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (256, 256), interpolation=cv2.INTER_CUBIC)


def transform_image(image_path: Path, preprocessed: bool) -> torch.Tensor:
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot decode image: {image_path}")
    if not preprocessed:
        bgr = crop_fundus(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return tf(Image.fromarray(rgb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--stage", choices=["1", "2"], default=None)
    ap.add_argument("--encoder-ckpt", default=None, help="RETFound_mae_natureCFP.pth (Stage 1 only)")
    ap.add_argument("--retfound-code", default=None)
    ap.add_argument("--thresholds", default=str(DEFAULT_THR))
    ap.add_argument("--calibration", default=str(DEFAULT_CAL))
    ap.add_argument("--preprocessed", action="store_true", help="input is already the 256x256 PNG")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-calibration", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    stage = args.stage or ("1" if "stage1" in ckpt_path.name else "2")
    m = re.search(r"seed_?(\d+)", ckpt_path.name)
    seed = m.group(1) if m else "42"
    model_key = f"stage{stage}_seed{seed}"
    thr_all = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    cal_all = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    labels = json.loads((HERE / "label_mapping.json").read_text(encoding="utf-8"))["labels"]
    encoder = find_encoder(args.encoder_ckpt) if stage == "1" else None
    model, ckpt, source = build_model(ckpt_path, find_retfound(args.retfound_code), device, encoder)

    x = transform_image(Path(args.image), args.preprocessed).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(x).float().cpu().numpy()[0]

    out = {"image": str(args.image), "checkpoint": str(ckpt_path), "model": model_key,
           "checkpoint_epoch": ckpt.get("epoch"), "weights_source": source,
           "preprocessed_input": bool(args.preprocessed), "labels": {}, "positive_labels": []}
    for j, label in enumerate(labels):
        thr = thr_all["models"][model_key][label]["threshold"]
        temp = 1.0 if args.no_calibration else cal_all["models"][model_key][label]
        p_raw = float(1.0 / (1.0 + np.exp(-logits[j])))
        p_cal = float(1.0 / (1.0 + np.exp(-logits[j] / temp)))
        positive = bool(float(logits[j]) > thr)
        out["labels"][label] = {"logit": float(logits[j]), "prob": p_raw,
                                "prob_calibrated": p_cal, "threshold_logit": thr,
                                "threshold_prob": float(1.0 / (1.0 + np.exp(-thr))),
                                "positive": positive}
        if positive:
            out["positive_labels"].append(label)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


