# Code — CFP-9 v1

这是最终训练/评估代码快照，包含 `src/`、`scripts/`、`configs/`、`tests/`、
`data_contract/` 和 RETFound 源码（`third_party/RETFound-main`，不含权重）。

## 1. 环境

- Python **3.10.19**
- PyTorch **2.8.0+cu128** / torchvision **0.23.0+cu128**
- timm **0.9.2**
- 其他依赖见 `requirements-lock.txt`
- 训练机器：RTX 5060 Laptop 8 GB（compute capability 12.0），bf16
- 完整安装：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

## 2. 路径约定

本目录是可移植源码快照。推荐把它当作工作根目录 `$ROOT`（把 `src/`、`configs/` 等
放在 `$ROOT` 下），然后：

```powershell
$ROOT = (Get-Location).Path
$env:CFP9_DATA_ROOT = "D:\datasets\cfp9"   # 原始只读数据根目录，按实际修改
```

发布版默认把原始数据放在 `$ROOT\dataset\`，代码会通过 `src/paths.py` 自动解析
`manifest_v1.public.csv` 中的相对路径。数据放在别处时设置：

```powershell
$env:CFP9_DATA_ROOT = "D:\datasets\cfp9"
$env:CFP9_PROJECT_ROOT = (Get-Location).Path   # 仅当代码不在项目根时设置
$env:CFP9_RETFOUND_DIR = "D:\code\RETFound-main"
$env:CFP9_RETFOUND_CKPT = "D:\weights\RETFound_mae_natureCFP.pth"
```

`$ROOT/outputs/` 下需要存在 `preprocessed/`、`features/` 和 `runs/`（或百度下载的等价目录）。
原始数据布局见 `../dataset/DATA_LAYOUT.md`。

关键文件：

```
deliverables/dataset/manifest_v1.public.csv
deliverables/dataset/manifest_v1.public.json   # public -> frozen provenance
deliverables/dataset/manifest_v1_preprocessed.public.csv
deliverables/dataset/retfound_cfp_224.public_index.csv
outputs/features/retfound_cfp_224.npy
outputs/preprocessed/<dataset>/*.png
deliverables/model/checkpoints/*.pt
```

RETFound 源码：`third_party/RETFound-main`。Stage 2 完整 checkpoint 的推理和评测
只需 RETFound 代码，不需要额外加载预训练编码器权重。Stage 1 单图推理或从零复现
Stage 2 训练时，才需要单独提供 `RETFound_mae_natureCFP.pth`（可用
`CFP9_RETFOUND_CKPT` / `--encoder-ckpt` 指定）。

## 3. 训练

Stage 1（冻结编码器，只训练 head）：

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py src\train_head.py --config configs\stage1_head_v2.yaml `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --feature-index deliverables\dataset\retfound_cfp_224.public_index.csv `
  --smoke --steps 2
& $py src\train_head.py --config configs\stage1_head_v2.yaml `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --feature-index deliverables\dataset\retfound_cfp_224.public_index.csv `
  --seed 42 --run-dir outputs\runs\stage1_v2\seed_42
```

Stage 2（从 Stage 1 v2 seed42 初始化，解冻 ViT-L）：

```powershell
& $py src\train_finetune.py --config configs\stage2_finetune.yaml `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --preprocessed deliverables\dataset\manifest_v1_preprocessed.public.csv `
  --stage1-checkpoint outputs\runs\stage1_v2\seed_42\best.pt `
  --smoke --steps 2
& $py src\train_finetune.py --config configs\stage2_finetune.yaml `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --preprocessed deliverables\dataset\manifest_v1_preprocessed.public.csv `
  --stage1-checkpoint outputs\runs\stage1_v2\seed_42\best.pt `
  --seed 42 --run-dir outputs\runs\stage2\seed_42
```

`configs/stage2_finetune.yaml` 必须保持：

```yaml
stage1_checkpoint: outputs/runs/stage1_v2/seed_42/best.pt
hidden: 1024
epochs: 50
warmup_epochs: 10
batch_size: 8
grad_accum: 2
eval_batch_size: 16
lr: 5.0e-4
layer_decay: 0.65
drop_path: 0.2
amp: bf16
grad_checkpointing: true
```

Stage 2 的 head 加载是 `strict=True`，并校验 checkpoint 的 labels 与 manifest SHA256。
`manifest_v1.public.json` 将 public 文件哈希可审计地映射到冻结 manifest 哈希，不要删除。

## 4. 最终评估

发布版使用相对路径 metadata：

```powershell
# 1) val 自检（不消耗 test）
& $py src\evaluate_final.py `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --preprocessed deliverables\dataset\manifest_v1_preprocessed.public.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --feature-index deliverables\dataset\retfound_cfp_224.public_index.csv `
  --checkpoint-root deliverables\model\checkpoints `
  --report-dir outputs\reports\reproduce `
  --split val --seeds 42,43,44 --bootstrap 300

# 2) test 只运行一次（已有 final_eval.json 时不要加 --force）
& $py src\evaluate_final.py --split test --final --bootstrap 1000 --hash-checkpoints

# 3) IDRID official holdout 只运行一次
& $py src\evaluate_final.py --split external_holdout --final --bootstrap 1000 --hash-checkpoints
```

若 `outputs/reports/final_eval.json` 已存在，脚本应拒绝覆盖；禁止用 `--force` 重跑
已经冻结的 test。

## 5. 单张图片推理

使用交付的 `../model/predict.py`：

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py ..\model\predict.py --image C:\path\to\fundus.jpg `
  --ckpt ..\model\checkpoints\stage2_seed42_best.pt

# Stage 1 baseline（需要 RETFound 编码器权重）
& $py ..\model\predict.py --image C:\path\to\fundus.jpg `
  --ckpt ..\model\checkpoints\stage1_v2_seed42_best.pt --stage 1 `
  --encoder-ckpt D:\weights\RETFound_mae_natureCFP.pth
```

输入可以是原始 CFP；`predict.py` 会执行与 `scripts/02_preprocess.py` 相同的视野裁剪，
再 Resize(256) + CenterCrop(224)。若输入已经是 256×256 预处理 PNG，加 `--preprocessed`。

## 6. 测试

```powershell
& $py -m pytest -q
```

当前代码验证结果：`27 passed, 1 skipped`。

## 7. 禁止事项

- 不得用 test / IDRID holdout 调参、调阈值、选 seed 或选 checkpoint；
- 不得覆盖 `outputs/runs/stage1_v2`、`outputs/runs/stage2`、`outputs/reports/final_eval*`；
- 不得把 checkpoint、features、preprocessed 图片直接提交 git；使用 DVC 指针或路径 + SHA256。
