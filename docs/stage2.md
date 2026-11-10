# Stage 2 — 解冻 ViT-L 全量微调（脚本就绪，未训练）

状态：**只完成初始化链路、8 GB 安全 smoke 与短 pilot，未启动任何正式训练。**

## 1. 初始化链路（硬约束）

`configs/stage2_finetune.yaml`

```yaml
stage1_checkpoint: outputs/runs/stage1_v2/seed_42/best.pt
hidden: 1024          # 与 Stage 1 v2 一致
```

`src/train_finetune.py::load_stage1_head` 在读取任何影像之前完成 4 项校验，任一失败即退出：

| 校验 | 失败信息 |
|---|---|
| `stage1_checkpoint` 存在 | `stage1_checkpoint not found` |
| `ckpt["labels"] == 9 标签契约` | `label order mismatch` |
| `ckpt["manifest_sha256"] == manifest_v1` 的 SHA256 | `manifest sha256 mismatch` |
| `head.load_state_dict(..., strict=True)` | `size mismatch for net.1.weight ...` |

四条守卫都用伪造 checkpoint 实测过（缺字段 / v1 的 hidden 512 / 打乱标签 / 错误 SHA），全部按预期拦截。

运行产物中写入：`stage1_checkpoint`、`stage1_checkpoint_sha256`、`stage1_manifest_sha256`、`stage1_epoch`、
`stage1_val_macro_auroc`、`head_loaded=true`、`head_load_mode="strict=True"`、`labels_verified`、
`manifest_sha256_verified`（`run.json` 与 `stage2_smoke.json` 均有）。

## 2. 输入管线必须与特征缓存一致（本轮关键修复）

R2-3 缓存（`scripts/04_dedup.py`）用的是 `Resize(256) + CenterCrop(224)`、`model(x)` 前向、再 L2 归一化。
Stage 2 必须完全复刻，否则 Stage 1 的头部看到的是另一个特征分布：

| 差异 | 与缓存的特征余弦 | 1 步后 val macro |
|---|---|---|
| 整图 `Resize(224)`（无裁剪） | — | 0.8825 |
| `CenterCrop` 但走 `forward_features` | 0.9426 | 0.8790 |
| **`CenterCrop` + `forward_head(.., pre_logits=True)` + L2** | **1.000000** | **0.9036**（Stage 1 参考 0.9037） |

根因：该 RETFound 类的 `forward = forward_features + forward_head`，而 **两者都会调用 `fc_norm`**；
缓存是走完整 `forward` 生成的。Stage 2 现用 `forward_head(forward_features(x), pre_logits=True)` + L2 归一化，
与缓存逐位一致（cos = 1.000000，min = 1.000000）。

## 3. 与计划 9.2 的对齐

| 计划参数 | 计划值 | 当前 config | 状态 |
|---|---|---|---|
| input_size | 224 | 224 | ✅ |
| batch_size | 16 | 8 × accum 2 = **16 有效** | ✅（拆分为显存约束） |
| epochs | 50 | 50 | ✅ |
| warmup_epochs | 10 | 10 | ✅ |
| lr | 5e-4 | 5e-4 | ✅ |
| min_lr | 1e-6 | 1e-6 | ✅ |
| layer_decay | 0.65 | 0.65（每层 lr = 5e-4 · 0.65^(L−layer)，52 个 param group） | ✅ |
| drop_path | 0.2 | 0.2 | ✅ |
| weight_decay | 0.05 | 0.05 | ✅ |
| label_smoothing / alpha | 0.05 / 0.25 | 0.05 / 0.25 | ✅ |
| grad_clip | 1.0 | 1.0 | ✅ |
| AMP | on | bf16（实测 autocast 生效为 bfloat16） | ✅（bf16 而非 fp16） |
| model selection | best val macro AUROC | 同（`n_pos>=30` 的标签） | ✅ |

**偏差记录**：唯一偏差是 micro-batch 8 × 梯度累积 2（有效 batch 仍为 16），原因是 8 GB 显存；
AMP 用 bf16 而非计划未指定的 fp16。二者都不改变优化目标或有效 batch。

另：`grad_checkpointing: true` 原先对这个 RETFound 类是**空操作**（该类重写了 `forward_features`，
不走 timm 的 checkpoint 分支）。现已在 `enable_true_grad_checkpointing()` 里重写 block 循环，
`torch.utils.checkpoint` 真正生效。

## 4. 8 GB 安全 smoke 与吞吐

| 配置 | peak VRAM | step 时间 | 预计 epoch | 50 epoch |
|---|---:|---:|---:|---:|
| batch 4 × 4，无真 checkpointing | 5.79 GB | 3.14 s | 52 min | ~44 h |
| batch 4 × 4，真 checkpointing | 5.23 GB | 0.51 s | 8.5 min | 7.1 h |
| **batch 8 × 2，真 checkpointing（定稿）** | **5.30 GB** | **0.40 s** | **6.8 min** | **5.6 h** |

- 数据加载不是瓶颈：稳态 2.5 ms/批（4 workers）、10.9 ms/批（0 workers）。
- smoke 含 forward + backward + optimizer step，3 步 loss 均有限，`amp_effective_dtype = bfloat16`。
- 峰值 5.30 GB < 7.5 GB 阈值，未触发降级到 batch 2 × 8。

## 5. 短 pilot 结论

| pilot | steps | 初始 → 结束 val macro | 结论 |
|---|---:|---|---|
| lr 5e-4，warmup 压到 10 步 | 50 | 0.9037 → 0.8349 | 满 LR 立刻训练会掉点 |
| lr 1e-4，warmup 20 步 | 100 | 0.9037 → 0.8938 | 稳定但仍未回升 |
| lr 5e-4，warmup 5 步 | 20 | 0.9037 → 0.8699 | 同上 |

- `best.pt` reload 复现：全部 pilot 均为 `max |ΔAUROC| = 0.0`。
- 显存稳定（全程 5.2–5.3 GB 峰值），无 NaN / 无 nonfinite。
- **50–100 步内看不到 loss 下降是预期现象**：loss 是逐标签掩码均值，micro-batch 8 时稀有标签样本极少，
  方差很大；而协议要求 10 个 epoch 的 warmup（约 10,040 步），前 50 步 LR 只有峰值的 0.5%。
  因此短 pilot 只能验证「显存 + 吞吐 + checkpoint 可复现」，无法验证学习性。

## 6. 结论与建议

- 8 GB 本机**可跑**：按定稿配置约 **5.6 小时 / 50 epoch**，落在计划对 12–16 GB 机器的 16–24 小时估计内，
  无需换 24 GB 机器，也暂不需要 LoRA / 部分解冻。
- 正式开跑前建议加一道 **1500 步 warmup 对齐门禁**（warmup 300 步，约 10 分钟）：
  确认 LR 升到峰值后 val macro 不继续下滑，再投入 5.6 小时。
- test 与 IDRID external holdout 至今未参与任何训练或选型。

## 7. 命令

```powershell
# 初始化链路自检
& .venv\Scripts\python.exe src\train_finetune.py --config configs\stage2_finetune.yaml --dry-run
# 8 GB 安全 smoke
& .venv\Scripts\python.exe src\train_finetune.py --config configs\stage2_finetune.yaml --smoke --steps 3
# 短 pilot（独立目录，不影响正式 run）
& .venv\Scripts\python.exe src\train_finetune.py --config configs\stage2_finetune.yaml --max-steps 20 --warmup-steps 5 --run-dir outputs\runs\stage2_pilot\b8_ckpt20
# 正式训练（未执行）
& .venv\Scripts\python.exe src\train_finetune.py --config configs\stage2_finetune.yaml --seed 42
```

## 8. 1500 步 warmup 门禁（seed 42，已通过）

命令（`--warmup-steps 300` 仅为门禁压缩 warmup；正式 run 仍用 config 的 10 epoch）：

```powershell
& .venv\Scripts\python.exe src\train_finetune.py `
  --config configs\stage2_finetune.yaml --max-steps 1500 --warmup-steps 300 `
  --run-dir outputs\runs\stage2_gate\seed_42
```

| 指标 | 值 |
|---|---:|
| optimizer_steps | 1500 |
| median_step_seconds | 0.4057 |
| estimated_epoch_seconds | 407.3（+ 验证 163 s @ eval_bs4） |
| peak_vram_gb | 5.299 |
| all_losses_finite | True（nonfinite_batches = 0） |
| amp_effective_dtype | bfloat16 |
| val macro 初始（Stage 1 v2） | 0.9037 |
| **val macro 结束** | **0.9371（+0.0334）** |
| reload max｜ΔAUROC｜ | 0.0 |
| stage1_checkpoint | `outputs/runs/stage1_v2/seed_42/best.pt`（sha 5a3d89f0…） |
| head_loaded / manifest_sha256_verified | True / True |

逐标签：normal 0.9257、**dr 0.9313**、glaucoma 0.9263、csc* 0.9004、disc_edema 0.9801、
macular_scar 0.9318、myopia 0.9704、retinal_det 0.9598、retinitis_pig 0.9985（*exploratory）。
loss：前 50 步均值 0.1727 → 后 50 步 0.0996。

**判定：PASS**（macro ≥0.895 ✓，DR ≥0.90 ✓，显存 <7.5 GB ✓，reload 误差 0 ✓，无 NaN ✓）。

验证开销实测：`eval_batch_size=4` 时单次 val = 163 s；改到 8 / 16 分别为 41.3 s / 35.1 s
（峰值仅 1.76 / 1.79 GB，AUROC 不变）。若正式 run 采用 eval_bs=16，50 epoch 由约 7.9 h 降到约 6.1 h。
