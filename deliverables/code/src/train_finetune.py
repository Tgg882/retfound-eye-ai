"""Stage 2 - unfreeze RETFound ViT-L and fine-tune encoder + 9-label head.

Initialisation is *required* to come from a Stage 1 checkpoint (`stage1_checkpoint`
in the config); the head is loaded with strict=True and both the label order and
the manifest SHA256 are verified before any image is read.

  python src/train_finetune.py --config configs/stage2_finetune.yaml --dry-run
  python src/train_finetune.py --config configs/stage2_finetune.yaml --smoke --steps 2
  python src/train_finetune.py --config configs/stage2_finetune.yaml --max-steps 30 --run-dir outputs/runs/stage2_pilot/seed_42
  python src/train_finetune.py --config configs/stage2_finetune.yaml --seed 42

Only `manifest_v1.csv` is accepted; test and external-holdout rows never enter the
optimizer or the model-selection metric.
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
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import LABELS, M_COLS, Y_COLS, load_manifest  # noqa: E402
from encoder import build_encoder  # noqa: E402
from head import CFP9Head  # noqa: E402
from losses import balanced_masked_bce  # noqa: E402
from manifest_provenance import manifest_provenance  # noqa: E402
from paths import path_key, resolve_project_path  # noqa: E402
from samplers import sqrt_balanced_sampler  # noqa: E402

PREPROCESSED = PROJECT_ROOT / "outputs" / "manifest" / "manifest_v0_clean_preprocessed.csv"
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs" / "stage2"


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


def build_transforms(img_size: int, hflip: float, train: bool, center_crop: bool = True):
    """Must match the R2-3 cache (`scripts/04_dedup.py`): Resize(256) + CenterCrop(224).

    Using a plain Resize(224) instead shifts the field of view by ~12.5% and puts the
    Stage 1 head on a different feature distribution than it was trained on.
    """
    steps = []
    if train and hflip > 0:
        steps.append(transforms.RandomHorizontalFlip(p=hflip))
    if center_crop:
        steps += [
            transforms.Resize(int(img_size / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
        ]
    else:
        steps.append(transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(steps)


class ImageManifestDataset(Dataset):
    """Serves preprocessed 256px PNG crops; rows are already filtered by the caller."""

    def __init__(self, records: pd.DataFrame, transform) -> None:
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records.iloc[i]
        with Image.open(resolve_project_path(rec["preprocessed_path"])) as img:
            x = self.transform(img.convert("RGB"))
        y = torch.tensor(rec[Y_COLS].to_numpy(dtype=np.float32))
        m = torch.tensor(rec[M_COLS].to_numpy(dtype=np.float32))
        w = torch.tensor(float(rec["sample_weight"]), dtype=torch.float32)
        return x, y, m, w


def evaluate(model, loader, device, min_pos: int = 30, amp_dtype=None):
    model.eval()
    logits, targets, masks = [], [], []
    with torch.no_grad():
        for x, y, m, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(x)
            logits.append(out.float().cpu().numpy())
            targets.append(y.numpy())
            masks.append(m.numpy())
    logits = np.concatenate(logits)
    targets = np.concatenate(targets)
    masks = np.concatenate(masks)

    per_label = {}
    for j, label in enumerate(LABELS):
        known = masks[:, j] == 1
        y, p = targets[known, j], logits[known, j]
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        auc = float(roc_auc_score(y, p)) if n_pos > 0 and n_neg > 0 else float("nan")
        per_label[label] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auroc": auc,
            "used_for_macro": bool(n_pos >= min_pos and np.isfinite(auc)),
        }
    used = [v["auroc"] for v in per_label.values() if v["used_for_macro"]]
    macro = float(np.mean(used)) if used else float("nan")
    return per_label, macro, [l for l, v in per_label.items() if v["used_for_macro"]]


def load_stage1_head(cfg: dict, manifest_prov: dict, head: nn.Module, device):
    """Load the Stage 1 head (strict) and verify labels + manifest provenance."""
    manifest_sha = manifest_prov["canonical_sha256"]
    raw = cfg.get("stage1_checkpoint")
    if not raw:
        raise SystemExit("config is missing `stage1_checkpoint`: Stage 2 must start from Stage 1")
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise SystemExit(f"stage1_checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt_labels = list(ckpt.get("labels", []))
    if ckpt_labels != list(LABELS):
        raise SystemExit(f"label order mismatch: ckpt={ckpt_labels} vs contract={list(LABELS)}")
    if ckpt.get("manifest_sha256") != manifest_sha:
        raise SystemExit(
            "manifest sha256 mismatch: stage1 checkpoint was built on a different manifest "
            f"({str(ckpt.get('manifest_sha256'))[:16]} != {manifest_sha[:16]})"
        )
    try:
        head.load_state_dict(ckpt["head"], strict=True)
    except RuntimeError as exc:
        raise SystemExit(f"stage1 head is incompatible with this config: {exc}") from exc

    info = {
        "stage1_checkpoint": str(path),
        "stage1_checkpoint_sha256": sha256_file(path),
        "stage1_manifest_sha256": ckpt.get("manifest_sha256"),
        "canonical_manifest_sha256": manifest_sha,
        "manifest_file_sha256": manifest_prov["file_sha256"],
        "manifest_provenance_mode": manifest_prov["provenance_mode"],
        "stage1_epoch": ckpt.get("epoch"),
        "stage1_val_macro_auroc": (ckpt.get("epoch_metrics") or {}).get("val_macro_auroc"),
        "head_loaded": True,
        "head_load_mode": "strict=True",
        "labels_verified": True,
        "manifest_sha256_verified": True,
    }
    head.to(device)
    return info


def enable_true_grad_checkpointing(model) -> None:
    """RETFound overrides forward_features and ignores timm's grad_checkpointing flag.

    Re-implement the block loop with torch.utils.checkpoint so `grad_checkpointing: true`
    actually trades compute for activation memory.
    """
    import torch.utils.checkpoint as cp

    def _forward_features(self, x):
        b = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = cp.checkpoint(blk, x, use_reentrant=False)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1, keepdim=True)
            return self.fc_norm(x)
        return self.norm(x)[:, 0]

    import types

    model.forward_features = types.MethodType(_forward_features, model)


def attach_forward(model, l2_normalize: bool) -> None:
    """Route pooled encoder features through the same L2 normalisation used by the cache."""

    def _forward(self, x):
        # NOTE: RETFound's forward_features already applies fc_norm, and timm's
        # forward_head applies it again. The R2-3 cache was produced with the full
        # model forward, so Stage 2 must use the same path or the Stage 1 head sees a
        # different feature distribution (measured: val macro 0.879 vs 0.904).
        f = self.forward_head(self.forward_features(x), pre_logits=True)
        if f.ndim == 3:
            f = f.squeeze(1)
        if l2_normalize:
            f = torch.nn.functional.normalize(f.float(), dim=1)
        return self.head(f)

    import types

    model.forward = types.MethodType(_forward, model)


def build_model(cfg: dict, device, manifest_prov: dict):
    base, enc_info = build_encoder(drop_path_rate=float(cfg.get("drop_path", 0.0)), target="cpu")
    head = CFP9Head(dim=1024, hidden=int(cfg.get("hidden", 1024)), n=len(LABELS), p=float(cfg.get("dropout", 0.3)))
    base.head = head
    stage1_info = load_stage1_head(cfg, manifest_prov, head, device)
    stage1_info["l2_normalize_features"] = bool(cfg.get("l2_normalize_features", True))
    attach_forward(base, stage1_info["l2_normalize_features"])
    if cfg.get("grad_checkpointing", True):
        base.set_grad_checkpointing(True)  # timm flag (no-op for this class)
        enable_true_grad_checkpointing(base)
    base = base.to(device)
    return base, head, enc_info, stage1_info


def layer_decay_param_groups(model, head, base_lr: float, weight_decay: float, layer_decay: float):
    """BEiT/RETFound layer-wise lr decay: lr_layer = base_lr * decay^(L - layer_id)."""
    blocks = list(model.blocks)
    n_layers = len(blocks) + 2  # 0 = stem, 1..N = blocks, N+1 = fc_norm/head
    head_ids = {id(p) for p in head.parameters()}

    buckets: dict[tuple[float, float], list] = {}
    scales: dict[int, float] = {}

    def add(named, layer_id: int) -> None:
        scale = round(layer_decay ** (n_layers - 1 - layer_id), 8)
        scales[layer_id] = scale
        for _, p in named:
            if not p.requires_grad:
                continue
            wd = 0.0 if p.ndim <= 1 else weight_decay
            buckets.setdefault((scale, wd), []).append(p)

    stem = [(n, p) for n, p in model.named_parameters()
            if not n.startswith("blocks.") and not n.startswith("fc_norm") and id(p) not in head_ids]
    add(stem, 0)
    for i, blk in enumerate(blocks):
        add(list(blk.named_parameters()), i + 1)
    add(list(model.fc_norm.named_parameters()), n_layers - 1)
    add(list(head.named_parameters()), n_layers - 1)

    groups = [
        {"params": params, "lr": base_lr * scale, "layer_scale": scale, "weight_decay": wd}
        for (scale, wd), params in buckets.items()
    ]
    groups.sort(key=lambda g: (-g["layer_scale"], g["weight_decay"]))
    return groups


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int, min_ratio: float):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + np.cos(np.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def amp_probe(device, amp_dtype):
    if device.type != "cuda" or amp_dtype is None:
        return None
    with torch.autocast("cuda", dtype=amp_dtype):
        out = torch.ones(8, 8, device=device) @ torch.ones(8, 8, device=device)
    return str(out.dtype).replace("torch.", "")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "outputs" / "manifest" / "manifest_v1.csv"))
    parser.add_argument("--preprocessed", default=str(PREPROCESSED), help="path -> preprocessed_path CSV")
    parser.add_argument("--stage1-checkpoint", default=None, help="override config stage1_checkpoint")
    parser.add_argument("--smoke", action="store_true", help="run --steps micro-batches with fwd+bwd+step")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="stop a full run after N optimizer steps (pilot)")
    parser.add_argument("--warmup-steps", type=int, default=None, help="pilot only: override warmup length in optimizer steps")
    parser.add_argument("--lr", type=float, default=None, help="pilot only: override the base learning rate")
    parser.add_argument("--dry-run", action="store_true", help="build model + dataset, then exit (no training)")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.lr is not None:
        cfg["lr"] = float(args.lr)
    if args.stage1_checkpoint is not None:
        cfg["stage1_checkpoint"] = args.stage1_checkpoint
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

    prep = pd.read_csv(args.preprocessed, dtype={"path": str, "preprocessed_path": str})[["path", "preprocessed_path"]].copy()
    manifest = manifest.copy()
    manifest["path_key"] = manifest["path"].map(path_key)
    prep["path_key"] = prep["path"].map(path_key)
    manifest = manifest.merge(prep[["path_key", "preprocessed_path"]], on="path_key", how="left", validate="one_to_one")
    missing = int(manifest["preprocessed_path"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} manifest rows have no preprocessed image")

    train_mask = (manifest["split"] == "train").to_numpy() & (manifest["is_external_holdout"] == 0).to_numpy()
    if cfg.get("drop_train_rejects", True):
        train_mask &= (manifest["quality_flag"] != "reject").to_numpy()
    train_rows = np.flatnonzero(train_mask)
    val_rows = np.flatnonzero((manifest["split"] == "val").to_numpy())
    if manifest.iloc[train_rows]["is_external_holdout"].any():
        raise SystemExit("external holdout leaked into train rows")

    img_size = int(cfg.get("img_size", 224))
    center_crop = bool(cfg.get("center_crop", True))
    train_ds = ImageManifestDataset(
        manifest.iloc[train_rows], build_transforms(img_size, float(cfg.get("hflip", 0.5)), train=True, center_crop=center_crop)
    )
    val_ds = ImageManifestDataset(
        manifest.iloc[val_rows], build_transforms(img_size, 0.0, train=False, center_crop=center_crop)
    )

    sampler = None
    if cfg.get("sqrt_balanced_sampler", True):
        sampler = sqrt_balanced_sampler(
            manifest.iloc[train_rows].reset_index(drop=True), np.arange(len(train_rows)), seed=seed
        )
    bs = int(cfg.get("batch_size", 4))
    eval_bs = int(cfg.get("eval_batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, shuffle=sampler is None,
                              num_workers=int(cfg.get("num_workers", 4)), pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False,
                            num_workers=int(cfg.get("num_workers", 4)), pin_memory=True, drop_last=False)

    amp_name = str(cfg.get("amp", "bf16"))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp_name)
    use_scaler = amp_dtype == torch.float16
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model, head, enc_info, stage1_info = build_model(cfg, device, manifest_prov)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vram_after_load = float(torch.cuda.memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0
    print(f"encoder: {enc_info}")
    print(f"stage1 head: {stage1_info['stage1_checkpoint']} (strict, manifest verified)")
    print(f"trainable params: {n_train_params/1e6:.1f}M  | vram after load: {vram_after_load:.2f}GB")

    protocol = {
        "epochs": int(args.epochs if args.epochs is not None else cfg.get("epochs", 50)),
        "warmup_epochs": int(cfg.get("warmup_epochs", 10)),
        "base_lr": float(cfg.get("lr", 5e-4)),
        "min_lr": float(cfg.get("min_lr", 1e-6)),
        "layer_decay": float(cfg.get("layer_decay", 0.65)),
        "drop_path": float(cfg.get("drop_path", 0.2)),
        "weight_decay": float(cfg.get("weight_decay", 0.05)),
        "grad_clip": float(cfg.get("grad_clip", 1.0)),
        "micro_batch": bs,
        "grad_accum": int(cfg.get("grad_accum", 4)),
        "effective_batch": bs * int(cfg.get("grad_accum", 4)),
        "amp": amp_name,
        "img_size": img_size,
    }

    if args.dry_run:
        x, y, m, _ = next(iter(train_loader))
        print(f"dry-run: train={len(train_ds)} val={len(val_ds)} batch={tuple(x.shape)} labels={tuple(y.shape)}")
        if device.type == "cuda":
            with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(x[:2].to(device))
            print(f"dry-run forward ok: {tuple(out.shape)} peak_vram={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB")
        return 0

    ys = manifest.iloc[train_rows][Y_COLS].to_numpy(dtype=np.float32)
    ms = manifest.iloc[train_rows][M_COLS].to_numpy(dtype=np.float32)
    label_counts = torch.tensor((ys * ms).sum(axis=0), dtype=torch.float32, device=device)

    groups = layer_decay_param_groups(
        model, head, float(cfg.get("lr", 5e-4)), float(cfg.get("weight_decay", 0.05)), float(cfg.get("layer_decay", 0.65))
    )
    optim = torch.optim.AdamW(groups)
    epochs = protocol["epochs"]
    accum = protocol["grad_accum"]
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = epochs * steps_per_epoch
    warmup_steps = (
        int(args.warmup_steps) if args.warmup_steps is not None else protocol["warmup_epochs"] * steps_per_epoch
    )
    protocol["warmup_steps"] = warmup_steps
    protocol["warmup_source"] = "cli_override" if args.warmup_steps is not None else "config"
    scheduler = warmup_cosine(
        optim, warmup_steps, total_steps,
        protocol["min_lr"] / max(protocol["base_lr"], 1e-12),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    eps, alpha, clip = float(cfg.get("label_smoothing", 0.05)), float(cfg.get("alpha", 0.25)), protocol["grad_clip"]
    print(f"protocol: {protocol} | optimizer groups: {len(groups)} | lr scales: "
          f"{[round(g['layer_scale'], 4) for g in groups[:3]]}...{round(groups[-1]['layer_scale'], 4)}")

    # ---------------------------------------------------------------- smoke
    if args.smoke:
        model.train()
        it, started, log = iter(train_loader), time.time(), []
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        for step in range(1, args.steps + 1):
            t0 = time.time()
            try:
                x, y, m, w = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y, m, w = next(it)
            x, y, m, w = x.to(device), y.to(device), m.to(device), w.to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = balanced_masked_bce(model(x), y, m, label_counts, sample_weight=w, eps=eps, alpha=alpha)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            finite = bool(torch.isfinite(loss))
            log.append({"step": step, "loss": float(loss.detach()), "finite": finite,
                        "seconds": round(time.time() - t0, 2),
                        "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0})
            print(f"step {step}: loss={float(loss.detach()):.6f} finite={finite} "
                  f"vram={log[-1]['peak_vram_gb']:.2f}GB t={log[-1]['seconds']}s")
        peak = float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0
        payload = {
            "mode": "smoke",
            "steps": args.steps,
            "seconds": round(time.time() - started, 1),
            "protocol": protocol,
            "loss_log": log,
            "all_losses_finite": bool(all(e["finite"] for e in log)),
            "peak_vram_gb": peak,
            "vram_after_load_gb": vram_after_load,
            "amp_requested": amp_name,
            "amp_effective_dtype": amp_probe(device, amp_dtype),
            "trainable_params": n_train_params,
            "n_train": int(len(train_rows)),
            "n_val": int(len(val_rows)),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "manifest_file_sha256": manifest_prov["file_sha256"],
            "manifest_provenance_mode": manifest_prov["provenance_mode"],
            "encoder_load": enc_info,
            **stage1_info,
        }
        out_path = PROJECT_ROOT / "outputs" / "reports" / "stage2_smoke.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: payload[k] for k in ("mode", "all_losses_finite", "peak_vram_gb", "amp_effective_dtype")}, indent=2))
        return 0 if payload["all_losses_finite"] else 1

    # ------------------------------------------------------------ full run
    run_dir = Path(args.run_dir) if args.run_dir else RUN_ROOT / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    history, best_macro, best_epoch, best_metrics = [], float("-inf"), None, None
    step_times, step_log, global_steps, stop = [], [], 0, False
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        ep_start, total, batches, nonfinite = time.time(), 0.0, 0, 0
        optim.zero_grad(set_to_none=True)
        window_t0 = time.time()
        window_loss, window_batches = 0.0, 0
        for i, (x, y, m, w) in enumerate(train_loader):
            if i % accum == 0:
                window_t0 = time.time()
            x, y, m, w = x.to(device, non_blocking=True), y.to(device), m.to(device), w.to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = balanced_masked_bce(model(x), y, m, label_counts, sample_weight=w, eps=eps, alpha=alpha) / accum
            if not torch.isfinite(loss):
                nonfinite += 1
                optim.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            window_loss += float(loss.detach()) * accum
            window_batches += 1
            if (i + 1) % accum == 0:
                step_loss = window_loss / max(window_batches, 1)
                window_loss, window_batches = 0.0, 0
                if clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                global_steps += 1
                step_secs = time.time() - window_t0
                step_times.append(step_secs)
                if args.max_steps:
                    step_log.append({"step": global_steps, "loss": step_loss, "lr": float(optim.param_groups[-1]["lr"]), "seconds": round(step_secs, 3),
                                     "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0})
                if args.max_steps and global_steps >= args.max_steps:
                    stop = True
                    break
            total += float(loss.detach()) * accum
            batches += 1
        train_loss = total / max(batches, 1)
        per_label, macro, used_labels = evaluate(model, val_loader, device, min_pos=min_pos, amp_dtype=amp_dtype)
        peak = float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0
        row = {"epoch": epoch, "train_loss": train_loss, "val_macro_auroc": macro,
               "val_macro_n_labels": len(used_labels), "nonfinite_batches": nonfinite,
               "optimizer_steps": global_steps, "seconds": round(time.time() - ep_start, 1),
               "peak_vram_gb": peak}
        row.update({f"auroc_{k}": v["auroc"] for k, v in per_label.items()})
        row.update({f"n_pos_{k}": v["n_pos"] for k, v in per_label.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(metrics_path, index=False, encoding="utf-8")

        ckpt = {"model": model.state_dict(), "optimizer": optim.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "config": cfg, "seed": seed,
                "labels": list(LABELS), "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha,
                "manifest_file_sha256": manifest_prov["file_sha256"],
                "manifest_provenance_mode": manifest_prov["provenance_mode"],
                "epoch_metrics": row,
                "stage1_checkpoint": stage1_info["stage1_checkpoint"],
                "torch_version": torch.__version__, "encoder_load": enc_info}
        if np.isfinite(macro) and macro > best_macro:
            best_macro, best_epoch, best_metrics = macro, epoch, per_label
            torch.save(ckpt, run_dir / "best.pt")
        torch.save(ckpt, run_dir / "last.pt")
        print(f"epoch {epoch:02d}/{epochs} loss={train_loss:.4f} val_macro={macro:.4f} "
              f"best={best_macro:.4f}@{best_epoch} steps={global_steps}")
        if stop:
            break

    mean_step = float(np.mean(step_times)) if step_times else float("nan")
    med_step = float(np.median(step_times)) if step_times else float("nan")
    est_epoch = med_step * steps_per_epoch
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    reload_model, reload_head, _, _ = build_model(cfg, device, manifest_prov)
    reload_model.load_state_dict(ckpt["model"])
    reload_per_label, reload_macro, reload_labels = evaluate(reload_model, val_loader, device, min_pos=min_pos, amp_dtype=amp_dtype)
    compare = {k: abs(reload_per_label[k]["auroc"] - best_metrics[k]["auroc"]) for k in LABELS
               if np.isfinite(reload_per_label[k]["auroc"]) and np.isfinite(best_metrics[k]["auroc"])}
    max_delta = float(max(compare.values())) if compare else 0.0

    run = {"mode": "pilot" if args.max_steps else "full", "config": cfg, "config_path": str(args.config),
           "seed": seed, "manifest": str(manifest_path), "manifest_sha256": manifest_sha, "labels": list(LABELS),
           "manifest_file_sha256": manifest_prov["file_sha256"],
           "manifest_provenance_mode": manifest_prov["provenance_mode"],
           "macro_min_pos": min_pos, "n_train": int(len(train_rows)), "n_val": int(len(val_rows)),
           "protocol": protocol, "optimizer_steps": global_steps,
           "mean_step_seconds": mean_step, "median_step_seconds": med_step,
           "estimated_epoch_seconds": est_epoch,
           "estimated_full_run_hours": est_epoch * epochs / 3600 if np.isfinite(est_epoch) else None,
           "trainable_params": n_train_params, "vram_after_load_gb": vram_after_load,
           "amp_effective_dtype": amp_probe(device, amp_dtype), "encoder_load": enc_info,
           "best_epoch": best_epoch, "best_val_macro_auroc": best_macro, "best_macro_labels": reload_labels,
           "best_val_per_label": reload_per_label,
           "reload_check": {"max_abs_auroc_delta": max_delta, "reproduced": bool(max_delta <= 1e-6)},
           "device": str(device), "torch_version": torch.__version__, "git_commit": git_commit(),
           "seconds": round(time.time() - started, 1),
           "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else 0.0,
           "run_dir": str(run_dir), **stage1_info}
    if args.max_steps:
        run["step_log"] = step_log
    (run_dir / "run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("mode", "optimizer_steps", "median_step_seconds",
                                          "estimated_epoch_seconds", "peak_vram_gb", "reload_check")}, indent=2))
    return 0 if run["reload_check"]["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
