# CFP-9 最终评估 — IDRID external holdout

- 生成时间：2026-11-09 22:12:20
- manifest：`outputs\manifest\manifest_v1.csv`，SHA256 `87cd436aad0118119b884b14f7f17b041c699699f7bc0cf84f4fd66960249c8e`
- 行数：103（total 23294），bootstrap 重采样单位：group_id，103 组 × 1000 次
- git commit：`c5a5115a1dc783b0ac6c7c81be2ae7439b90d756`，torch 2.8.0+cu128
- 阈值与温度：仅在 **val** 上拟合，原样应用于本 split；macro 只用 n_pos ≥ 30 的标签

## 1. 结论摘要

- **Stage 2（微调 ViT-L）**：DR AUROC 0.8304 [0.7790, 0.8771]，macro AUROC 0.8304 [0.7790, 0.8771]（3 seed 均值 + group-aware 95% CI）
- **Stage 1 v2（冻结编码器）**：DR AUROC 0.8772 [0.8413, 0.9129]，macro AUROC 0.8772 [0.8413, 0.9129]（3 seed 均值 + group-aware 95% CI）
- **主比较 Stage 2 − Stage 1**：DR AUROC -0.0468 [-0.1077, +0.0133]，macro AUROC -0.0468 [-0.1077, +0.0133]

## 2. Stage 1 v2（cached features + head）

| seed | DR AUROC | macro AUROC | DR AUPR | macro AUPR | DR sens@90%spec | DR ECE (raw→cal) | macro ECE (raw→cal) |
|---|---|---|---|---|---|---|---|
| 42 | 0.8785 | 0.8785 | 0.9459 | 0.9459 | 0.9275 | 0.152→0.152 | 0.152→0.152 |
| 43 | 0.8764 | 0.8764 | 0.9451 | 0.9451 | 0.9130 | 0.163→0.149 | 0.163→0.149 |
| 44 | 0.8768 | 0.8768 | 0.9448 | 0.9448 | 0.9130 | 0.144→0.145 | 0.144→0.145 |
| **mean±std** | 0.8772 ± 0.0011 | 0.8772 ± 0.0011 | 0.9453 ± 0.0006 | 0.9453 ± 0.0006 | 0.9179 ± 0.0084 | | |

## 3. Stage 2（preprocessed image + 完整模型）

| seed | DR AUROC | macro AUROC | DR AUPR | macro AUPR | DR sens@90%spec | DR ECE (raw→cal) | macro ECE (raw→cal) |
|---|---|---|---|---|---|---|---|
| 42 | 0.8363 | 0.8363 | 0.8898 | 0.8898 | 0.8986 | 0.212→0.199 | 0.212→0.199 |
| 43 | 0.8205 | 0.8205 | 0.8860 | 0.8860 | 0.8986 | 0.144→0.138 | 0.144→0.138 |
| 44 | 0.8344 | 0.8344 | 0.9174 | 0.9174 | 0.8406 | 0.189→0.168 | 0.189→0.168 |
| **mean±std** | 0.8304 ± 0.0086 | 0.8304 ± 0.0086 | 0.8977 ± 0.0171 | 0.8977 ± 0.0171 | 0.8792 ± 0.0335 | | |

## 4. Stage 2 − Stage 1（配对 bootstrap，同一重采样）

| 指标 | seed42 Δ [95% CI] (p) | 3-seed 均值 Δ [95% CI] (p) |
|---|---|---|
| DR AUROC | -0.0422 [-0.1526, +0.0548] (p=0.4420) | -0.0468 [-0.1077, +0.0133] (p=0.1340) |
| macro AUROC | -0.0422 [-0.1526, +0.0548] (p=0.4420) | -0.0468 [-0.1077, +0.0133] (p=0.1340) |
| DR AUPR | -0.0561 [-0.1450, +0.0272] (p=0.2480) | -0.0475 [-0.0923, +0.0002] (p=0.0540) |

> seed42 vs seed42 仅作配对参考；主结论用 3-seed 均值。seed 由训练期 val 决定，未在 test 上挑选。

## 5. 逐标签（3-seed 均值）

| label | S1 AUROC | S2 AUROC | Δ AUROC | S2 AUPR | S2 sens@90%spec | 角色 |
|---|---|---|---|---|---|---|
| normal | n/a | n/a | +nan | n/a | n/a | core |
| dr | 0.8772 | 0.8304 | -0.0468 | 0.8977 | 0.8792 | core |
| glaucoma | n/a | n/a | +nan | n/a | n/a | core |
| csc | n/a | n/a | +nan | n/a | n/a | exploratory |
| disc_edema | n/a | n/a | +nan | n/a | n/a | exploratory |
| macular_scar | n/a | n/a | +nan | n/a | n/a | core |
| myopia | n/a | n/a | +nan | n/a | n/a | core |
| retinal_det | n/a | n/a | +nan | n/a | n/a | exploratory |
| retinitis_pig | n/a | n/a | +nan | n/a | n/a | exploratory |

csc 为 exploratory（val n_pos 仅 12），不进入核心 macro 结论。

## 6. 逐数据集 DR AUROC（3-seed 均值）

| dataset | n | S1 DR AUROC | S2 DR AUROC | S2 macro AUROC |
|---|---|---|---|---|
| IDRID | 103 | 0.8772 | 0.8304 | 0.8304 |

## 7. 阈值（val 拟合，原样应用）

| label | thr@spec90 (S1 s42) | val sens | val spec | thr@spec90 (S2 s42) | val sens | val spec |
|---|---|---|---|---|---|---|
| normal | -0.423 | 0.742 | 0.901 | -3.547 | 0.925 | 0.905 |
| dr | 0.122 | 0.740 | 0.900 | -0.742 | 0.901 | 0.900 |
| glaucoma | -0.295 | 0.543 | 0.901 | -3.266 | 0.884 | 0.901 |
| csc | -2.577 | 0.833 | 0.901 | -3.578 | 0.833 | 0.901 |
| disc_edema | -2.456 | 0.812 | 0.902 | -3.594 | 1.000 | 0.939 |
| macular_scar | -1.578 | 0.569 | 0.901 | -3.578 | 0.902 | 0.909 |
| myopia | -2.165 | 0.979 | 0.901 | -3.500 | 0.917 | 0.906 |
| retinal_det | -2.852 | 0.944 | 0.901 | -3.359 | 0.944 | 0.931 |
| retinitis_pig | -2.992 | 1.000 | 0.901 | -3.562 | 1.000 | 0.950 |

## 复现信息

| model | checkpoint | epoch | sha256 |
|---|---|---|---|
| stage1_seed42 | `outputs\runs\stage1_v2\seed_42\best.pt` | 48 | `not hashed` |
| stage1_seed43 | `outputs\runs\stage1_v2\seed_43\best.pt` | 47 | `not hashed` |
| stage1_seed44 | `outputs\runs\stage1_v2\seed_44\best.pt` | 49 | `not hashed` |
| stage2_seed42 | `outputs\runs\stage2\seed_42\best.pt` | 33 | `not hashed` |
| stage2_seed43 | `outputs\runs\stage2\seed_43\best.pt` | 7 | `not hashed` |
| stage2_seed44 | `outputs\runs\stage2\seed_44\best.pt` | 9 | `not hashed` |

- Stage1 config：`configs\stage1_head_v2.yaml`；Stage2 config：`configs\stage2_finetune.yaml`
- features：`outputs\features\retfound_cfp_224.npy`
