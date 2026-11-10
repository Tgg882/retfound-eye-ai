# Evaluation — CFP-9 v1

本目录是 **一次性冻结评估**（2026-11-09）的交付副本。2026-11-10 的交付工作只做审计、
复制、重聚合表格和画图；**没有重跑 test，没有改模型，没有改阈值/seed**。

## 1. 文件

| 文件 | 内容 |
|---|---|
| `final_eval.{md,json,csv}` | 内部 test（3,474 张，排除 IDRID official holdout），6 个 checkpoint |
| `final_eval_external_holdout.{md,json,csv}` | IDRID official test holdout（103 张） |
| `final_eval_val.{md,json,csv}` | val 自检（6/6 与 `run.json` 完全一致，Δ=0） |
| `provenance_audit.json` | manifest / 6 checkpoint / git commit / 报告内容审计 |
| `release_reproduction_check.json` | 发布版相对路径 + public sidecar 的 val 6/6 复现检查（未重跑 test） |
| `tables/stage_summary_3seed.csv` | 3-seed mean ± std、group-aware bootstrap CI |
| `tables/paired_stage2_minus_stage1.csv` | 配对 bootstrap 差值 |
| `tables/per_label_3seed.csv` | 逐标签 AUROC/AUPR/ECE/sens/spec |
| `tables/per_dataset_3seed.csv` | 逐数据集 DR/macro AUROC |
| `tables/headline_tables.md` | 上述表格的可读汇总 |
| `figures/` | ROC、PR、calibration、per-dataset heatmap、bootstrap forest |

## 2. 两套评估口径

- **内部 test**：manifest v1 的 `split=test` 且 `is_external_holdout=0`，共 3,474 张。
- **IDRID official holdout**：`is_external_holdout=1`，共 103 张；只用于最终一次性评估。

阈值和温度只在 **val** 上拟合，原样应用到 test 与 holdout。macro AUROC 只使用该 split
上 `n_pos >= 30` 的标签；本 test 上进入 macro 的是 normal / dr / glaucoma / macular_scar /
myopia。csc 始终为 exploratory。

## 3. Headline（3-seed mean ± std）

| 指标 | Stage 1 v2 | Stage 2 |
|---|---:|---:|
| 内部 test DR AUROC | 0.9191 ± 0.0002 | 0.9531 ± 0.0034 |
| 内部 test macro AUROC | 0.9045 ± 0.0021 | 0.9462 ± 0.0081 |
| 内部 test DR AUPR | 0.9140 ± 0.0010 | 0.9508 ± 0.0081 |
| IDRID holdout DR AUROC | 0.8772 ± 0.0011 | 0.8304 ± 0.0086 |

配对 bootstrap（Stage 2 − Stage 1）：

- 内部 test DR AUROC：`+0.0339 [+0.0270, +0.0403]`
- 内部 test macro AUROC：`+0.0417 [+0.0310, +0.0534]`
- IDRID holdout DR AUROC：`-0.0468 [-0.1077, +0.0133]`，p=0.134

**IDRID holdout 方向不利，但 95% CI 跨 0，不能宣称显著退化。**

## 4. 图表的重要说明

`final_eval.json` 从未持久化 test 的逐图分数（一次性规则禁止重跑 test），因此：

- `roc_val_dr.png`、`pr_val_dr.png`、`calibration_val_dr.png` 是 **val split** 的曲线；
- `per_dataset_auroc_heatmap.png` 和 `bootstrap_ci_dr.png` 直接来自冻结的
  `final_eval*.json`，属于 test / holdout 结果。

不得为了画 test ROC/PR 而重跑 test。

## 5. IDRID 表述（必须遵守）

IDRID 的 train（288）和 val（62）已经参与训练与选模；**只有官方 test 103 张是 holdout**。
因此正确表述是：

> IDRID official test split (103 images) held out from training.

不能写成 "unseen external dataset" 或 "完全未见过的新数据集外部验证"。

## 6. 校准与限制

- IDRID holdout 上 ECE 为 0.138–0.199（约 0.14–0.20），说明跨数据集域偏移下概率校准失效。
- Stage 2 在内部 test 提升明显，但在 IDRID holdout 与 MESSIDOR-2 上方向不利；
  不能把内部 test 提升外推为普遍外部泛化提升。
- csc 全数据仅 80 阳性（val 12），始终标记 exploratory，不进入核心 macro 结论。
- 数据集不是多病共存标注：manifest 中没有多阳性图，macro 是各标签单任务 AUROC 的平均。

## 7. 禁止事项

test 与 IDRID holdout 的结果 **不得再用于**：调参、调阈值、选 seed、选 checkpoint、
挑模型版本或反复评估后择优。后续若要改动，只能新建版本并说明原因，不能覆盖本目录结果。
