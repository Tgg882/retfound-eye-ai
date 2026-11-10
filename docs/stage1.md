# Stage 1 — 冻结编码器，训练 9 标签分类头

数据版本：`outputs/manifest/manifest_v1.csv`（23,294 行；唯一允许训练的 manifest）。
特征：`outputs/features/retfound_cfp_224.npy`（23,299 × 1024，与 manifest 之间通过 `feat_row` 对齐）。

## 训练闭环

`src/train_head.py`

- 外层 epoch 循环，每 epoch 在 val 上评估。
- 产物：`outputs/runs/stage1*/seed_<seed>/{best.pt,last.pt,metrics.csv,run.json}`。
- checkpoint 保存 head / optimizer / scheduler / epoch / config / `manifest_sha256`。
- best 按 val macro AUROC 选择，macro 只统计 val `n_pos >= macro_min_pos`（默认 30）的标签：
  `normal, dr, glaucoma, macular_scar, myopia`；`csc` 为 exploratory，不进核心 macro。
- `test` 与 `external_holdout` 不参与训练、不参与选模（代码内断言）。
- `--smoke` 只写 `outputs/reports/stage1_smoke.json`，不写 run 目录；full run 不覆盖该文件。

修复记录（本轮）：

1. `evaluate()` 原先用 manifest 行号直接索引特征数组。manifest v1 比特征文件少 5 行（G3 leakage guard），
   行号整体错位，导致 val AUROC 恒为随机。现统一经 `feat_row` 映射，并加回归测试
   `tests/test_head_alignment.py`。
2. `sqrt_balanced_sampler` 改用 `torch.Generator().manual_seed(seed)`。
3. `sample_weight` 改为逐样本加权（`tests/test_loss_and_sampler.py` 覆盖）。

## 3-seed 结果（val，mean ± std，seed 42/43/44）

| protocol | macro (≥30 pos) | DR AUROC | DR 门禁 ≥0.90 |
|---|---:|---:|---|
| v1: 30 epoch, hidden 512（原定协议） | 0.8724 ± 0.0022 | 0.8873 ± 0.0012 | ❌ |
| v2: 50 epoch, hidden 1024（val 调参后确认） | 0.9032 ± 0.0008 | 0.9122 ± 0.0009 | ✅ |

v2 逐标签（mean ± std）：

| label | v1 | v2 |
|---|---:|---:|
| normal | 0.9121 ± 0.0010 | 0.9262 ± 0.0007 |
| dr | 0.8873 ± 0.0012 | 0.9122 ± 0.0009 |
| glaucoma | 0.8136 ± 0.0031 | 0.8473 ± 0.0025 |
| csc (exploratory) | 0.9059 ± 0.0054 | 0.9329 ± 0.0059 |
| disc_edema | 0.8804 ± 0.0171 | 0.9240 ± 0.0015 |
| macular_scar | 0.7867 ± 0.0098 | 0.8549 ± 0.0062 |
| myopia | 0.9624 ± 0.0028 | 0.9756 ± 0.0012 |
| retinal_det | 0.9350 ± 0.0045 | 0.9442 ± 0.0006 |
| retinitis_pig | 0.9943 ± 0.0003 | 0.9968 ± 0.0003 |

DR 未被门禁卡住的池化值主要由 DDR（≈0.847）与 MESSIDOR-2（≈0.736）拉低；
APTOS2019（≈0.985）与 Fundus10（≈0.965）已远高于阈值。

其他验收：无 NaN / 无 nonfinite batch；mask 生效（打乱 mask=0 槽位后 loss 完全不变）；
稀有标签输出概率非恒定；`best.pt` reload 后 AUROC 完全复现（max |Δ| = 0.0）。

## 复现命令

```powershell
# pilot
& .venv\Scripts\python.exe src\train_head.py --config configs\stage1_head.yaml --epochs 2 --seed 42 --run-dir outputs\runs\stage1_pilot\seed_42
# 原定协议 3 seeds
& .venv\Scripts\python.exe src\train_head.py --config configs\stage1_head.yaml --seed 42
# 修订协议 3 seeds
& .venv\Scripts\python.exe src\train_head.py --config configs\stage1_head_v2.yaml --seed 42 --run-dir outputs\runs\stage1_v2\seed_42
```

汇总：`outputs/reports/stage1_3seed_summary.{json,csv}`
