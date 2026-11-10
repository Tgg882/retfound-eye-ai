# CFP-9 最终评估 — val（自检）

- 生成时间：2026-11-09 22:03:56
- manifest：`outputs\manifest\manifest_v1.csv`，SHA256 `87cd436aad0118119b884b14f7f17b041c699699f7bc0cf84f4fd66960249c8e`
- 行数：3479（total 23294），bootstrap 重采样单位：group_id，3376 组 × 300 次
- git commit：`02ad51c8b79f1cbc6d08902cdf64b5630d0e4228`，torch 2.8.0+cu128
- 阈值与温度：仅在 **val** 上拟合，原样应用于本 split；macro 只用 n_pos ≥ 30 的标签

## 1. 结论摘要

- **Stage 2（微调 ViT-L）**：DR AUROC 0.9498 [0.9463, 0.9539]，macro AUROC 0.9591 [0.9524, 0.9651]（3 seed 均值 + group-aware 95% CI）
- **Stage 1 v2（冻结编码器）**：DR AUROC 0.9122 [0.9063, 0.9174]，macro AUROC 0.9032 [0.8931, 0.9113]（3 seed 均值 + group-aware 95% CI）
- **主比较 Stage 2 − Stage 1**：DR AUROC +0.0376 [+0.0321, +0.0439]，macro AUROC +0.0559 [+0.0455, +0.0678]

## 2. Stage 1 v2（cached features + head）

| seed | DR AUROC | macro AUROC | DR AUPR | macro AUPR | DR sens@90%spec | DR ECE (raw→cal) | macro ECE (raw→cal) |
|---|---|---|---|---|---|---|---|
| 42 | 0.9128 | 0.9037 | 0.9094 | 0.6848 | 0.7399 | 0.024→0.027 | 0.039→0.029 |
| 43 | 0.9126 | 0.9023 | 0.9097 | 0.6895 | 0.7361 | 0.024→0.024 | 0.036→0.028 |
| 44 | 0.9112 | 0.9036 | 0.9078 | 0.6878 | 0.7424 | 0.024→0.025 | 0.040→0.033 |
| **mean±std** | 0.9122 ± 0.0009 | 0.9032 ± 0.0008 | 0.9090 ± 0.0010 | 0.6874 ± 0.0024 | 0.7395 ± 0.0032 | | |

## 3. Stage 2（preprocessed image + 完整模型）

| seed | DR AUROC | macro AUROC | DR AUPR | macro AUPR | DR sens@90%spec | DR ECE (raw→cal) | macro ECE (raw→cal) |
|---|---|---|---|---|---|---|---|
| 42 | 0.9525 | 0.9607 | 0.9491 | 0.8807 | 0.9011 | 0.031→0.024 | 0.028→0.026 |
| 43 | 0.9473 | 0.9572 | 0.9299 | 0.8602 | 0.8375 | 0.020→0.020 | 0.031→0.035 |
| 44 | 0.9495 | 0.9595 | 0.9488 | 0.8621 | 0.8388 | 0.078→0.073 | 0.047→0.040 |
| **mean±std** | 0.9498 ± 0.0026 | 0.9591 ± 0.0018 | 0.9426 ± 0.0110 | 0.8676 ± 0.0113 | 0.8592 ± 0.0364 | | |

## 4. Stage 2 − Stage 1（配对 bootstrap，同一重采样）

| 指标 | seed42 Δ [95% CI] (p) | 3-seed 均值 Δ [95% CI] (p) |
|---|---|---|
| DR AUROC | +0.0397 [+0.0274, +0.0501] (p=0.0000) | +0.0376 [+0.0321, +0.0439] (p=0.0000) |
| macro AUROC | +0.0569 [+0.0417, +0.0744] (p=0.0000) | +0.0559 [+0.0455, +0.0678] (p=0.0000) |
| DR AUPR | +0.0397 [+0.0220, +0.0543] (p=0.0000) | +0.0336 [+0.0254, +0.0423] (p=0.0000) |

> seed42 vs seed42 仅作配对参考；主结论用 3-seed 均值。seed 由训练期 val 决定，未在 test 上挑选。

## 5. 逐标签（3-seed 均值）

| label | S1 AUROC | S2 AUROC | Δ AUROC | S2 AUPR | S2 sens@90%spec | 角色 |
|---|---|---|---|---|---|---|
| normal | 0.9262 | 0.9708 | +0.0447 | 0.8835 | 0.9278 | core |
| dr | 0.9122 | 0.9498 | +0.0376 | 0.9426 | 0.8592 | core |
| glaucoma | 0.8473 | 0.9356 | +0.0883 | 0.8320 | 0.8748 | core |
| csc | 0.9329 | 0.9433 | +0.0104 | 0.5249 | 0.8333 | exploratory |
| disc_edema | 0.9240 | 0.9991 | +0.0751 | 0.9686 | 1.0000 | exploratory |
| macular_scar | 0.8549 | 0.9666 | +0.1117 | 0.8413 | 0.9150 | core |
| myopia | 0.9756 | 0.9729 | -0.0027 | 0.8388 | 0.9375 | core |
| retinal_det | 0.9442 | 0.9895 | +0.0453 | 0.9356 | 0.9815 | exploratory |
| retinitis_pig | 0.9968 | 0.9994 | +0.0026 | 0.9881 | 1.0000 | exploratory |

csc 为 exploratory（val n_pos 仅 12），不进入核心 macro 结论。

## 6. 逐数据集 DR AUROC（3-seed 均值）

| dataset | n | S1 DR AUROC | S2 DR AUROC | S2 macro AUROC |
|---|---|---|---|---|
| APTOS2019 | 548 | 0.9906 | 0.9945 | 0.9945 |
| DDR | 1878 | 0.8633 | 0.9242 | 0.9242 |
| Fundus10 | 656 | 0.9777 | 0.9887 | 0.9726 |
| IDRID | 62 | 0.8984 | 0.9060 | 0.9060 |
| MESSIDOR-2 | 262 | 0.7751 | 0.8043 | 0.8043 |
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

### val 自检（与训练 run.json 对比）

| model | run.json | recomputed | Δ | ok |
|---|---|---|---|---|
| stage1_seed42 | 0.903747 | 0.903747 | 0.00e+00 | True |
| stage1_seed43 | 0.902335 | 0.902335 | 0.00e+00 | True |
| stage1_seed44 | 0.903605 | 0.903605 | 0.00e+00 | True |
| stage2_seed42 | 0.960693 | 0.960693 | 0.00e+00 | True |
| stage2_seed43 | 0.957215 | 0.957215 | 0.00e+00 | True |
| stage2_seed44 | 0.959503 | 0.959503 | 0.00e+00 | True |
