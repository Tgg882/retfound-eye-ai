# CFP-9 最终评估 — 内部 test split

- 生成时间：2026-11-09 22:09:06
- manifest：`outputs\manifest\manifest_v1.csv`，SHA256 `87cd436aad0118119b884b14f7f17b041c699699f7bc0cf84f4fd66960249c8e`
- 行数：3474（total 23294），bootstrap 重采样单位：group_id，3379 组 × 1000 次
- git commit：`c5a5115a1dc783b0ac6c7c81be2ae7439b90d756`，torch 2.8.0+cu128
- 阈值与温度：仅在 **val** 上拟合，原样应用于本 split；macro 只用 n_pos ≥ 30 的标签

## 1. 结论摘要

- **Stage 2（微调 ViT-L）**：DR AUROC 0.9531 [0.9489, 0.9568]，macro AUROC 0.9462 [0.9382, 0.9541]（3 seed 均值 + group-aware 95% CI）
- **Stage 1 v2（冻结编码器）**：DR AUROC 0.9191 [0.9142, 0.9242]，macro AUROC 0.9045 [0.8964, 0.9120]（3 seed 均值 + group-aware 95% CI）
- **主比较 Stage 2 − Stage 1**：DR AUROC +0.0339 [+0.0270, +0.0403]，macro AUROC +0.0417 [+0.0310, +0.0534]

## 2. Stage 1 v2（cached features + head）

| seed | DR AUROC | macro AUROC | DR AUPR | macro AUPR | DR sens@90%spec | DR ECE (raw→cal) | macro ECE (raw→cal) |
|---|---|---|---|---|---|---|---|
| 42 | 0.9191 | 0.9054 | 0.9151 | 0.6865 | 0.7593 | 0.027→0.025 | 0.044→0.034 |
| 43 | 0.9193 | 0.9021 | 0.9138 | 0.6835 | 0.7562 | 0.025→0.024 | 0.044→0.032 |
| 44 | 0.9190 | 0.9061 | 0.9131 | 0.6842 | 0.7574 | 0.023→0.023 | 0.043→0.033 |
| **mean±std** | 0.9191 ± 0.0002 | 0.9045 ± 0.0021 | 0.9140 ± 0.0010 | 0.6847 ± 0.0016 | 0.7576 ± 0.0016 | | |

## 3. Stage 2（preprocessed image + 完整模型）

| seed | DR AUROC | macro AUROC | DR AUPR | macro AUPR | DR sens@90%spec | DR ECE (raw→cal) | macro ECE (raw→cal) |
|---|---|---|---|---|---|---|---|
| 42 | 0.9567 | 0.9378 | 0.9585 | 0.8466 | 0.9052 | 0.029→0.023 | 0.035→0.031 |
| 43 | 0.9501 | 0.9466 | 0.9424 | 0.8317 | 0.8604 | 0.020→0.020 | 0.036→0.038 |
| 44 | 0.9523 | 0.9541 | 0.9515 | 0.8295 | 0.8629 | 0.066→0.062 | 0.048→0.042 |
| **mean±std** | 0.9531 ± 0.0034 | 0.9462 ± 0.0081 | 0.9508 ± 0.0081 | 0.8359 ± 0.0093 | 0.8762 ± 0.0252 | | |

## 4. Stage 2 − Stage 1（配对 bootstrap，同一重采样）

| 指标 | seed42 Δ [95% CI] (p) | 3-seed 均值 Δ [95% CI] (p) |
|---|---|---|
| DR AUROC | +0.0376 [+0.0252, +0.0490] (p=0.0000) | +0.0339 [+0.0270, +0.0403] (p=0.0000) |
| macro AUROC | +0.0325 [+0.0091, +0.0551] (p=0.0040) | +0.0417 [+0.0310, +0.0534] (p=0.0000) |
| DR AUPR | +0.0433 [+0.0284, +0.0574] (p=0.0000) | +0.0368 [+0.0282, +0.0453] (p=0.0000) |

> seed42 vs seed42 仅作配对参考；主结论用 3-seed 均值。seed 由训练期 val 决定，未在 test 上挑选。

## 5. 逐标签（3-seed 均值）

| label | S1 AUROC | S2 AUROC | Δ AUROC | S2 AUPR | S2 sens@90%spec | 角色 |
|---|---|---|---|---|---|---|
| normal | 0.9265 | 0.9713 | +0.0447 | 0.8835 | 0.9235 | core |
| dr | 0.9191 | 0.9531 | +0.0339 | 0.9508 | 0.8762 | core |
| glaucoma | 0.8178 | 0.9231 | +0.1054 | 0.8415 | 0.8343 | core |
| csc | 0.9024 | 0.9479 | +0.0455 | 0.5050 | 0.8056 | exploratory |
| disc_edema | 0.9559 | 0.9860 | +0.0300 | 0.8601 | 0.9792 | exploratory |
| macular_scar | 0.8855 | 0.9261 | +0.0406 | 0.7816 | 0.8364 | core |
| myopia | 0.9737 | 0.9574 | -0.0163 | 0.7223 | 0.9116 | core |
| retinal_det | 0.9992 | 0.9999 | +0.0008 | 0.9981 | 1.0000 | exploratory |
| retinitis_pig | 0.9973 | 0.9996 | +0.0023 | 0.9883 | 1.0000 | exploratory |

csc 为 exploratory（val n_pos 仅 12），不进入核心 macro 结论。

## 6. 逐数据集 DR AUROC（3-seed 均值）

| dataset | n | S1 DR AUROC | S2 DR AUROC | S2 macro AUROC |
|---|---|---|---|---|
| APTOS2019 | 548 | 0.9860 | 0.9933 | 0.9933 |
| DDR | 1875 | 0.8767 | 0.9249 | 0.9249 |
| Fundus10 | 654 | 0.9698 | 0.9919 | 0.9556 |
| IDRID | 62 | 0.9333 | 0.9812 | 0.9812 |
| MESSIDOR-2 | 262 | 0.8455 | 0.8252 | 0.8252 |
| PAPILA | 73 | n/a | n/a | n/a |

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

## 8. IDRID external holdout（官方 103 张，单独一节）

- 行数 103，bootstrap 103 组 × 1000 次
- 阈值/温度同样来自 val，未在 holdout 上调整

| seed | S1 DR AUROC | S1 DR AUPR | S2 DR AUROC | S2 DR AUPR | S2 sens@90%spec | S2 ECE (raw→cal) |
|---|---|---|---|---|---|---|
| 42 | 0.8785 | 0.9459 | 0.8363 | 0.8898 | 0.8986 | 0.212→0.199 |
| 43 | 0.8764 | 0.9451 | 0.8205 | 0.8860 | 0.8986 | 0.144→0.138 |
| 44 | 0.8768 | 0.9448 | 0.8344 | 0.9174 | 0.8406 | 0.189→0.168 |
| **stage1 mean±std** | 0.8772 ± 0.0011 | 0.9453 ± 0.0006 | | | | |
| **stage2 mean±std** | 0.8304 ± 0.0086 | 0.8977 ± 0.0171 | | | | |

配对（同一 holdout 重采样）：seed42 Δ = -0.0422 [-0.1526, +0.0548] (p=0.442)；3-seed 均值 Δ = -0.0468 [-0.1077, +0.0133] (p=0.134)。

holdout 仅 DR 有标注（n_pos=69），其余 8 个标签 n_pos=0，故 macro 与 DR 相同；样本量小，CI 较宽。

## 复现信息

| model | checkpoint | epoch | sha256 |
|---|---|---|---|
| stage1_seed42 | `outputs\runs\stage1_v2\seed_42\best.pt` | 48 | `5a3d89f0b8b5c684` |
| stage1_seed43 | `outputs\runs\stage1_v2\seed_43\best.pt` | 47 | `19bbd9b77044cefc` |
| stage1_seed44 | `outputs\runs\stage1_v2\seed_44\best.pt` | 49 | `a0061564eceb2e85` |
| stage2_seed42 | `outputs\runs\stage2\seed_42\best.pt` | 33 | `083a529a90c99723` |
| stage2_seed43 | `outputs\runs\stage2\seed_43\best.pt` | 7 | `ed8a8fdc3f7cb628` |
| stage2_seed44 | `outputs\runs\stage2\seed_44\best.pt` | 9 | `9cbe926c32723aa3` |

- Stage1 config：`configs\stage1_head_v2.yaml`；Stage2 config：`configs\stage2_finetune.yaml`
- features：`outputs\features\retfound_cfp_224.npy`
