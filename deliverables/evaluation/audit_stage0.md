# Stage 0 — 状态审计（只审计，不重跑 test）

审计日期：2026-11-10 ｜ 结论：**通过，无需 final_eval_v2**

## 1. final_eval 是否存在

存在，且为一次性 test 评估结果：

| 文件 | 时间 | 大小 |
|---|---|---:|
| `outputs/reports/final_eval.json` | 2026-11-09 22:09:06 | 67,028 B |
| `outputs/reports/final_eval.csv` | 2026-11-09 22:09:06 | 10,015 B |
| `outputs/reports/final_eval.md` | 2026-11-09 22:13:27 | 6,375 B |

本次未运行 `src/evaluate_final.py --split test`，未重新推理，未覆盖任何既有评估文件。

## 2. 校验

| 项 | 结果 |
|---|---|
| manifest SHA256（磁盘 vs JSON） | 一致：`87cd436aad0118119b884b14f7f17b041c699699f7bc0cf84f4fd66960249c8e` |
| 6 个 checkpoint SHA256（磁盘 vs JSON） | 6/6 一致 |
| git commit `c5a5115a1dc783b0ac6c7c81be2ae7439b90d756` | 对象存在（评估代码提交） |
| 结果文件提交 | `e07dade` |
| val 自检 | 6/6 与 `run.json` 精确一致，max Δ = 0 |
| test 行数 | 3,474 |
| bootstrap | 3,379 组 × 1,000 次，按 group_id 重采样 |

checkpoint 明细见 `deliverables/model/checkpoint_inventory.csv`，机器可读审计见
`deliverables/evaluation/provenance_audit.json`。

## 3. 报告内容检查

| 要求 | 状态 |
|---|---|
| Stage 1 v2（3 seeds） | 有 |
| Stage 2（3 seeds） | 有 |
| 3-seed 汇总 | 有 |
| per-dataset | 有 |
| per-label | 有 |
| group-aware bootstrap 95% CI | 有 |
| IDRID official holdout | 有（103 张） |
| ECE / calibration | 有 |
| Stage 2 − Stage 1 配对 bootstrap | 有 |

## 4. 发现并补充的报告缺口

`outputs/reports/final_eval_external_holdout.md` 的“复现信息”表把 6 个 checkpoint 写成
`not hashed`，原因是 holdout payload 生成时没有持久化 `checkpoint_sha256`。

处理：**不修改原始结果文件**；用 `final_eval.json` 的 6 个 SHA256 对磁盘重新计算并验证，
补充 `checkpoint_inventory.csv` 与 `provenance_audit.json`。这不影响任何 AUROC/CI/ECE 数值。

## 5. 是否发现评估 bug

未发现需要重跑的评估 bug。因此不创建 `final_eval_v2.*`。

## 6. 冻结口径

- 内部 test：3-seed mean ± std 为 headline；seed42 仅作预先指定的配对参考。
- IDRID official test holdout：103 张；IDRID train/val 已参与训练，不是 unseen dataset。
- csc：exploratory，不进入核心 macro。
- 后续禁止用 test / holdout 调参、调阈值、选 seed 或选 checkpoint。
