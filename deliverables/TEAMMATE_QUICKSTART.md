# CFP-9 队友快速开始（推理 / 评测复现 / 训练）

本文只描述**发布版路径**：GitHub 拉代码，百度网盘放大文件。

## 1. 目录放置

```text
<repo>\
├── src\ scripts\ configs\ tests\ data_contract\
├── dataset\                         # 百度：6 个原始 CFP 数据集
├── deliverables\
│   ├── dataset\                     # GitHub：相对路径 manifest + sidecar
│   ├── model\checkpoints\           # 百度：6 个 .pt
│   └── evaluation\                  # GitHub：冻结评测结果
├── outputs\
│   ├── preprocessed\                # 百度：256x256 PNG
│   └── features\retfound_cfp_224.npy # 百度：Stage 1 缓存特征
└── RETFound-main\                   # GitHub/第三方快照：models_vit.py
```

原始数据必须是：

```text
dataset\
├── 01_MESSIDOR2\
├── 02_APTOS2019\
├── 03_IDRID\
├── _raw\Fundus10\images\original\
├── 05_PAPILA\
└── DDR-dataset\DR_grading\
```

Fundus10 使用 `_raw\Fundus10\images\original`，不要用 `04_Fundus10`。
默认数据根就是 `<repo>\dataset\`；数据在别处时设置 `$env:CFP9_DATA_ROOT`。

## 2. 环境

```powershell
Set-Location <repo>
py -3.10 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r deliverables\code\requirements-lock.txt
& $py -m pytest -q
```

预期：`27 passed, 1 skipped`。

## 3. 单图推理（只做主模型）

主模型是 `stage2_seed42_best.pt`，完整 checkpoint 已含 ViT-L，不再需要单独的
RETFound 编码器权重。

```powershell
& $py deliverables\model\predict.py `
  --image C:\path\to\fundus.jpg `
  --ckpt deliverables\model\checkpoints\stage2_seed42_best.pt
```

输出 9 个标签的 logit、sigmoid 概率、温度校准概率、val 阈值和
`positive_labels`。`positive_labels` 是研究输出，不是临床诊断。

## 4. 复现 val 评测（先做这个）

把 6 个 checkpoint 放入 `deliverables\model\checkpoints\`：

```text
stage1_v2_seed42_best.pt
stage1_v2_seed43_best.pt
stage1_v2_seed44_best.pt
stage2_seed42_best.pt
stage2_seed43_best.pt
stage2_seed44_best.pt
```

运行：

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

验收：输出 `VAL SELF-CHECK: PASS`；6/6 的 macro AUROC 与
`deliverables\model\val_summary_3seed.csv` 精确一致（Δ=0）。

public manifest 的文件 SHA 是 `434b19…`，sidecar
`manifest_v1.public.json` 会把它追溯到冻结 manifest 的 `87cd43…`；不要删除 sidecar。

## 5. 一次性 test / IDRID holdout

只有在 val 通过后才运行；每个 split 只运行一次，并输出到新目录：

```powershell
& $py src\evaluate_final.py `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --preprocessed deliverables\dataset\manifest_v1_preprocessed.public.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --feature-index deliverables\dataset\retfound_cfp_224.public_index.csv `
  --checkpoint-root deliverables\model\checkpoints `
  --report-dir outputs\reports\reproduce `
  --split test --final --bootstrap 1000 --hash-checkpoints

& $py src\evaluate_final.py `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --preprocessed deliverables\dataset\manifest_v1_preprocessed.public.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --feature-index deliverables\dataset\retfound_cfp_224.public_index.csv `
  --checkpoint-root deliverables\model\checkpoints `
  --report-dir outputs\reports\reproduce `
  --split external_holdout --final --bootstrap 1000 --hash-checkpoints
```

把 `outputs\reports\reproduce\final_eval.json` 和
`final_eval_external_holdout.json` 与 `deliverables\evaluation\` 中的冻结 JSON 对照。
**不要**覆盖冻结文件，也不要在看到 test 结果后改 seed、阈值或 checkpoint。

## 6. 训练代码

训练代码已包含在 GitHub 仓库；Stage 2 需要 RETFound 预训练编码器权重
（CC BY-NC 4.0，需另行从官方发布获取，不随 GitHub 分发）。

Stage 1（冻结 RETFound，只训 9 标签 head，使用百度网盘的 `.npy` 特征）：

```powershell
& $py src\train_head.py --config configs\stage1_head_v2.yaml `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --feature-index deliverables\dataset\retfound_cfp_224.public_index.csv `
  --smoke --steps 2
```

Stage 2（从 Stage 1 head 初始化，再全量微调 ViT-L）：

```powershell
$env:CFP9_RETFOUND_CKPT = "D:\weights\RETFound_mae_natureCFP.pth"
& $py src\train_finetune.py --config configs\stage2_finetune.yaml `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --preprocessed deliverables\dataset\manifest_v1_preprocessed.public.csv `
  --stage1-checkpoint outputs\runs\stage1_v2\seed_42\best.pt `
  --smoke --steps 2
```

正式 Stage 2 在 8 GB 显存上约 7.5 小时 / 50 epoch。完整参数和验收见
`deliverables\code\reproduce.md`。
