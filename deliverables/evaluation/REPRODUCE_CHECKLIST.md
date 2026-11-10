# Evaluation reproduction checklist

1. Put raw data under `dataset\` according to `DATA_LAYOUT.md`.
2. Put all six checkpoints under `deliverables\model\checkpoints\`.
3. Put preprocessed PNGs under `outputs\preprocessed\`.
4. Put `retfound_cfp_224.npy` under `outputs\features\`.
5. Put RETFound code under `RETFound-main\` or set `CFP9_RETFOUND_DIR`.
   Stage 2 evaluation does **not** need the separate RETFound pretrained `.pth`; the full
   Stage 2 checkpoints already contain all weights.
6. Run val first:

```powershell
& $py src\evaluate_final.py `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --preprocessed deliverables\dataset\manifest_v1_preprocessed.public.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --feature-index deliverables\dataset\retfound_cfp_224.public_index.csv `
  --checkpoint-root deliverables\model\checkpoints `
  --report-dir outputs\reports\reproduce `
  --split val --seeds 42,43,44 --bootstrap 300
```

Expected: `VAL SELF-CHECK: PASS`. The script compares against `outputs/runs/*/run.json`
if present, otherwise against `model/val_summary_3seed.csv`. Six of six checkpoints must
match at Δ=0. The verified run is recorded in `release_reproduction_check.json`.

7. Only after val passes, run test / holdout once with `--final`.

Do not overwrite `deliverables\evaluation\final_eval*`.
