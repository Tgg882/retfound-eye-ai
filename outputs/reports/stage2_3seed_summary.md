# Stage 2 三 seed 复现汇总（val only）

日期：2026-11-09
配置：`configs/stage2_finetune.yaml`（50 ep / warmup 10 / bs 8 x accum 2 / lr 5e-4 / layer_decay 0.65 /
drop_path 0.2 / bf16 / grad_checkpointing，eval_bs 16）
初始化：`outputs/runs/stage1_v2/seed_42/best.pt`（head strict 加载，manifest_v1 SHA256 校验通过）
数据：`outputs/manifest/manifest_v1.csv`，SHA256 `87cd436aad0118119b884b14f7f17b041c699699f7bc0cf84f4fd66960249c8e`

macro 只使用 val 中 n_pos >= 30 的 5 个标签：normal / dr / glaucoma / macular_scar / myopia。
csc（val n_pos=12）保持 exploratory，不进入核心结论。test 与 IDRID external holdout 全程未使用。

## 逐 seed 结果（best epoch，按 val macro 选择）

| seed | best epoch | best val macro | DR AUROC | final-epoch macro | 训练时长 | epoch 均值 | peak VRAM | nonfinite | reload | max dAUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 33 | 0.9607 | 0.9525 | 0.9507 | 7.47 h | 533.8 s | 5.30 GB | 0 | 0.0 | 复现 |
| 43 | 7 | 0.9572 | 0.9473 | 0.9479 | 7.21 h | 514.8 s | 5.30 GB | 0 | 0.0 | 复现 |
| 44 | 9 | 0.9595 | 0.9495 | 0.9495 | 7.28 h | 519.6 s | 5.30 GB | 0 | 0.0 | 复现 |

## 聚合（mean +- std，n=3 seeds）

| 指标 | mean | std | min | max |
|---|---|---|---|---|
| val macro AUROC | 0.9591 | 0.0018 | 0.9572 | 0.9607 |
| DR AUROC | 0.9498 | 0.0026 | 0.9473 | 0.9525 |

两者 std 都远小于 0.01，三 seed 结论一致，不存在被单个 seed 主导的情况。

## 逐标签 AUROC（各自 best epoch）

| label | mean +- std | seed42 | seed43 | seed44 | 计入 macro |
|---|---|---|---|---|---|
| normal | 0.9708 +- 0.0059 | 0.9777 | 0.9676 | 0.9673 | 是 |
| dr | 0.9498 +- 0.0026 | 0.9525 | 0.9473 | 0.9495 | 是 |
| glaucoma | 0.9356 +- 0.0025 | 0.9380 | 0.9359 | 0.9330 | 是 |
| macular_scar | 0.9666 +- 0.0125 | 0.9566 | 0.9626 | 0.9806 | 是 |
| myopia | 0.9729 +- 0.0058 | 0.9787 | 0.9727 | 0.9672 | 是 |
| csc | 0.9433 +- 0.0287 | 0.9105 | 0.9556 | 0.9638 | 否（exploratory, n_pos=12） |
| disc_edema | 0.9991 +- 0.0006 | 0.9997 | 0.9990 | 0.9986 | 否（n_pos=16） |
| retinal_det | 0.9895 +- 0.0143 | 0.9730 | 0.9990 | 0.9964 | 否（n_pos=18） |
| retinitis_pig | 0.9994 +- 0.0007 | 0.9998 | 0.9985 | 0.9998 | 否（n_pos=20） |

## best epoch 分布

33 / 7 / 9。seed 43、44 的 val 峰值落在 warmup 阶段（epoch 10 之前），seed 42 落在 lr 衰减后段。
但三者 final-epoch macro 都在 0.9479-0.9507，与各自 best 只差 0.005-0.010，说明是平台期的小幅波动而非后期退化。

## 异常判定

- 无 NaN / 无 OOM / nonfinite_batches 全 0（3 seed x 50 ep x 1004 step）
- peak VRAM 恒定 5.30 GB（< 7.5 GB 门槛），bf16 AMP 全程生效
- 每个 seed 的 best.pt 重载后 val macro 完全复现（max |dAUROC| = 0.0）
- 唯一不稳定项：csc（0.9105-0.9638，std 0.0287），但属 exploratory、n_pos=12，不参与选模结论
- 结论：**无异常 seed**；seed 43/44 的早峰需要在论文中如实说明（val 平台期自 epoch ~7-9 起即已形成）

## 相对 Stage 1 v2 baseline

Stage 1 v2 seed 42：val macro 0.9037 / DR 0.9128。
Stage 2 三 seed：val macro 0.9591 +- 0.0018 / DR 0.9498 +- 0.0026。
macro 提升 +0.055，DR 提升 +0.037，三个 seed 全部超过 baseline。

## 产物

- `outputs/runs/stage2/seed_{42,43,44}/`：best.pt / last.pt / metrics.csv / run.json / train.log
- `outputs/runs/stage2/_aborted/seed_44_ep7/`：2026-11-08 中止的 7-epoch 半成品（已归档，不参与汇总）
- `outputs/reports/stage2_3seed_summary.csv`、`stage2_3seed_summary.py`（本报告的生成脚本）
