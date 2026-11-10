"""RETFound ViT-L CFP encoder (plan appendix B)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return candidate
    return None


def find_retfound_dir(explicit: str | Path | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("CFP9_RETFOUND_DIR"):
        candidates.append(Path(os.environ["CFP9_RETFOUND_DIR"]))
    candidates += [
        PROJECT_ROOT / "RETFound-main",
        PROJECT_ROOT / "deliverables" / "code" / "third_party" / "RETFound-main",
        PROJECT_ROOT / "third_party" / "RETFound-main",
    ]
    for candidate in candidates:
        if (candidate / "models_vit.py").is_file():
            return candidate
    return None


def find_encoder_checkpoint(explicit: str | Path | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("CFP9_RETFOUND_CKPT"):
        candidates.append(Path(os.environ["CFP9_RETFOUND_CKPT"]))
    candidates += [
        PROJECT_ROOT / "model" / "FoundmaenatureCFP" / "RETFound_mae_natureCFP.pth",
        PROJECT_ROOT / "deliverables" / "model" / "checkpoints" / "RETFound_mae_natureCFP.pth",
        PROJECT_ROOT / "RETFound-main" / "RETFound_mae_natureCFP.pth",
    ]
    return _first_existing(candidates)


def build_encoder(
    checkpoint: str | Path | None = None,
    drop_path_rate: float = 0.0,
    target: str = "cpu",
    retfound_dir: str | Path | None = None,
    load_weights: bool = True,
):
    resolved_dir = find_retfound_dir(retfound_dir)
    if resolved_dir is None:
        raise SystemExit(
            "RETFound code not found; set CFP9_RETFOUND_DIR or pass retfound_dir "
            "(expected models_vit.py under RETFound-main)"
        )
    if str(resolved_dir) not in sys.path:
        sys.path.insert(0, str(resolved_dir))

    from models_vit import RETFound_mae

    model = RETFound_mae(global_pool=True, num_classes=0, img_size=224, drop_path_rate=drop_path_rate)
    if not load_weights:
        return model.to(target), {
            "retfound_dir": str(resolved_dir),
            "encoder_checkpoint": None,
            "weights_loaded": False,
        }
    ckpt_path = find_encoder_checkpoint(checkpoint)
    if ckpt_path is None:
        raise SystemExit(
            "RETFound encoder checkpoint not found; set CFP9_RETFOUND_CKPT or place "
            "RETFound_mae_natureCFP.pth in model/FoundmaenatureCFP/ or "
            "deliverables/model/checkpoints/"
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)

    # global_pool=True creates `fc_norm` and deletes `norm`; the MAE checkpoint only
    # stores the pretrained `norm.*`.  Map it so frozen features use the pretrained
    # LayerNorm instead of a randomly initialised one.
    remapped = False
    if "fc_norm.weight" not in state and "norm.weight" in state:
        state["fc_norm.weight"] = state["norm.weight"]
        state["fc_norm.bias"] = state["norm.bias"]
        remapped = True

    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(target)
    return model, {
        "retfound_dir": str(resolved_dir),
        "encoder_checkpoint": str(ckpt_path),
        "weights_loaded": True,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "fc_norm_remapped_from_norm": remapped,
    }


def freeze_encoder(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
