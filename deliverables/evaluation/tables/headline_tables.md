# Evaluation tables — generated only from frozen final_eval JSON

所有数值由 `scripts/21_deliverable_tables.py` 从 `final_eval.json` / `final_eval_external_holdout.json` / `manifest_v1.csv` 重聚合；没有推理、没有重跑 test。

## 1. 3-seed summary

| scope | metric | mean | std | 95% CI | seed42 | seed43 | seed44 |
|---|---|---:|---:|---|---:|---:|---:|
| internal_test | stage1_dr_auroc | 0.9191 | 0.0002 | [0.9142, 0.9242] | 0.9191 | 0.9193 | 0.9190 |
| internal_test | stage1_macro_auroc | 0.9045 | 0.0021 | [0.8964, 0.9120] | 0.9054 | 0.9021 | 0.9061 |
| internal_test | stage1_dr_aupr | 0.9140 | 0.0010 | [0.9066, 0.9205] | 0.9151 | 0.9138 | 0.9131 |
| internal_test | stage2_dr_auroc | 0.9531 | 0.0034 | [0.9489, 0.9568] | 0.9567 | 0.9501 | 0.9523 |
| internal_test | stage2_macro_auroc | 0.9462 | 0.0081 | [0.9382, 0.9541] | 0.9378 | 0.9466 | 0.9541 |
| internal_test | stage2_dr_aupr | 0.9508 | 0.0081 | [0.9453, 0.9557] | 0.9585 | 0.9424 | 0.9515 |
| idrid_holdout_103 | stage1_dr_auroc | 0.8772 | 0.0011 | [0.8413, 0.9129] | 0.8785 | 0.8764 | 0.8768 |
| idrid_holdout_103 | stage1_macro_auroc | 0.8772 | 0.0011 | [0.8413, 0.9129] | 0.8785 | 0.8764 | 0.8768 |
| idrid_holdout_103 | stage1_dr_aupr | 0.9453 | 0.0006 | [0.9258, 0.9635] | 0.9459 | 0.9451 | 0.9448 |
| idrid_holdout_103 | stage2_dr_auroc | 0.8304 | 0.0086 | [0.7790, 0.8771] | 0.8363 | 0.8205 | 0.8344 |
| idrid_holdout_103 | stage2_macro_auroc | 0.8304 | 0.0086 | [0.7790, 0.8771] | 0.8363 | 0.8205 | 0.8344 |
| idrid_holdout_103 | stage2_dr_aupr | 0.8977 | 0.0171 | [0.8563, 0.9400] | 0.8898 | 0.8860 | 0.9174 |

## 2. Paired Stage 2 − Stage 1

| scope | metric | seed42 Δ [95% CI] | 3-seed Δ [95% CI] | p (3-seed) |
|---|---|---|---|---:|
| internal_test | dr_auroc | +0.0376 [+0.0252, +0.0490] | +0.0339 [+0.0270, +0.0403] | 0.0000 |
| internal_test | macro_auroc | +0.0325 [+0.0091, +0.0551] | +0.0417 [+0.0310, +0.0534] | 0.0000 |
| internal_test | dr_aupr | +0.0433 [+0.0284, +0.0574] | +0.0368 [+0.0282, +0.0453] | 0.0000 |
| idrid_holdout_103 | dr_auroc | -0.0422 [-0.1526, +0.0548] | -0.0468 [-0.1077, +0.0133] | 0.1340 |
| idrid_holdout_103 | macro_auroc | -0.0422 [-0.1526, +0.0548] | -0.0468 [-0.1077, +0.0133] | 0.1340 |
| idrid_holdout_103 | dr_aupr | -0.0561 [-0.1450, +0.0272] | -0.0475 [-0.0923, +0.0002] | 0.0540 |

## 3. Per-label (internal test, 3-seed mean)

| label | S1 AUROC | S2 AUROC | Δ | S2 AUPR | S2 sens@90%spec | role | core macro |
|---|---:|---:|---:|---:|---:|---|---|
| csc | 0.9024 | 0.9479 | +0.0455 | 0.5050 | 0.8056 | exploratory | False |
| disc_edema | 0.9559 | 0.9860 | +0.0300 | 0.8601 | 0.9792 | exploratory | False |
| dr | 0.9191 | 0.9531 | +0.0339 | 0.9508 | 0.8762 | core | True |
| glaucoma | 0.8178 | 0.9231 | +0.1054 | 0.8415 | 0.8343 | core | True |
| macular_scar | 0.8855 | 0.9261 | +0.0406 | 0.7816 | 0.8364 | core | True |
| myopia | 0.9737 | 0.9574 | -0.0163 | 0.7223 | 0.9116 | core | True |
| normal | 0.9265 | 0.9713 | +0.0447 | 0.8835 | 0.9235 | core | True |
| retinal_det | 0.9992 | 0.9999 | +0.0008 | 0.9981 | 1.0000 | exploratory | False |
| retinitis_pig | 0.9973 | 0.9996 | +0.0023 | 0.9883 | 1.0000 | exploratory | False |

## 4. Per-dataset (3-seed mean)

| scope | dataset | n | S1 DR AUROC | S2 DR AUROC | Δ DR | S2 macro |
|---|---|---:|---:|---:|---:|---:|
| internal_test | APTOS2019 | 548 | 0.9860 | 0.9933 | 0.0074 | 0.9933 |
| internal_test | DDR | 1875 | 0.8767 | 0.9249 | 0.0482 | 0.9249 |
| internal_test | Fundus10 | 654 | 0.9698 | 0.9919 | 0.0222 | 0.9556 |
| internal_test | IDRID | 62 | 0.9333 | 0.9812 | 0.0478 | 0.9812 |
| internal_test | MESSIDOR-2 | 262 | 0.8455 | 0.8252 | -0.0203 | 0.8252 |
| internal_test | PAPILA | 73 | n/a | n/a | n/a | n/a |
| idrid_holdout_103 | IDRID | 103 | 0.8772 | 0.8304 | -0.0468 | 0.8304 |

注：PAPILA 没有 DR 标签，DR AUROC 为 n/a；IDRID holdout 只有 DR 有标注，因此 macro=DR。

## 5. 角色与阈值

- csc：exploratory（全数据 80 阳性，val 12），不进入核心 macro 结论。
- disc_edema / retinal_det / retinitis_pig：在 test 上 n_pos < 30，只作逐标签报告，不进入 macro。
- 阈值：val specificity=0.90 的 raw-logit 阈值；温度：val 拟合，仅用于概率校准。