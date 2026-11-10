# Checkpoint layout for evaluation reproduction

`predict.py` needs only the primary Stage 2 checkpoint:

```text
stage2_seed42_best.pt
```

Three-seed evaluation needs all six checkpoints in a flat folder:

```text
stage1_v2_seed42_best.pt
stage1_v2_seed43_best.pt
stage1_v2_seed44_best.pt
stage2_seed42_best.pt
stage2_seed43_best.pt
stage2_seed44_best.pt
```

Place them under:

```text
deliverables\model\checkpoints\
```

The repository intentionally does not commit `.pt` files. Download them from the
Baidu package and verify them with `checksums.sha256` / `checkpoint_inventory.csv`.

For a quick single-seed check:

```powershell
& $py src\evaluate_final.py ... --checkpoint-root deliverables\model\checkpoints --seeds 42
```

The headline mean±std requires all six. Stage 2 evaluation needs RETFound source code
(`models_vit.py`) but does not need the separate `RETFound_mae_natureCFP.pth`; every Stage 2
checkpoint already contains the full encoder state.
