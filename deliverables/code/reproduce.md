# Reproduce — CFP-9 v1

目标：从干净环境得到与交付一致的 manifest v1、Stage 1 v2、Stage 2、val 自检和一次性
test / IDRID holdout 评估。**test 只能运行一次。**

## 0. 目录假设

```powershell
$ROOT = "C:\work\cfp9"          # 工作根目录（含 src/ configs/ outputs/）
$env:CFP9_DATA_ROOT = "D:\datasets\cfp9"
$env:CFP9_PROJECT_ROOT = $ROOT
$py = "$ROOT\.venv\Scripts\python.exe"
Set-Location $ROOT
```

原始数据结构（只读）：

```
$CFP9_DATA_ROOT/
├── DDR-dataset/DR_grading/{train,valid,test}.txt + image folders
├── _raw/Fundus10/images/original/<class>/*.jpg
├── 01_MESSIDOR2/
├── 02_APTOS2019/
├── 03_IDRID/
└── 05_PAPILA/
```

## 0A. 发布版目录（推荐）

如果是队友从 GitHub + 百度网盘拿到的发布版，推荐直接使用项目根目录作为工作根：

```
C:\work\cfp9\
├── src\ scripts\ configs\ tests\ data_contract\
├── dataset\                         # 原始 CFP 数据，gitignored
│   ├── 01_MESSIDOR2\
│   ├── 02_APTOS2019\
│   ├── 03_IDRID\
│   ├── _raw\Fundus10\images\original\
│   ├── 05_PAPILA\
│   └── DDR-dataset\DR_grading\
├── deliverables\
│   ├── dataset\                    # 相对路径 manifest / prep / feature index
│   ├── evaluation\                 # 冻结评测结果
│   └── model\checkpoints\          # 6 个 checkpoint（百度网盘）
├── outputs\
│   ├── preprocessed\                # 256x256 PNG（百度网盘）
│   └── features\retfound_cfp_224.npy
└── RETFound-main\                   # 或 deliverables\code\third_party\RETFound-main
```

默认 `CFP9_DATA_ROOT` 是仓库根目录下的 `dataset\`，所以路径解析不需要改代码。
如果数据放在别处：

```powershell
$env:CFP9_DATA_ROOT = "D:\datasets\cfp9"
```

注意：`dataset\` 只放原始图；`deliverables\dataset\` 只放 metadata，不要把 26 GB 原图放进去。

## 1. 环境

```powershell
py -3.10 -m venv .venv
& $py -m pip install --upgrade pip
& $py -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements-lock.txt
& $py -m pip check
& $py -c "import torch,numpy;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_capability(),numpy.__version__)"
```

验收：torch `2.8.0+cu128`，CUDA available=True，capability `(12, 0)`，numpy `1.26.4`。

## 2. 恢复冻结数据（推荐）

```powershell
dvc pull outputs/manifest/manifest_v1.csv.dvc
dvc pull outputs/manifest/manifest_v0_clean.csv.dvc
dvc pull outputs/manifest/manifest_v0_clean_preprocessed.csv.dvc
dvc pull outputs/manifest/dedup_groups_v1.csv.dvc
dvc pull outputs/manifest/conflict_groups_v1.csv.dvc
dvc pull outputs/manifest/splits/v1.dvc
```

验收：

- `manifest_v1.csv` 23,294 行；
- SHA256 = `87cd436aad0118119b884b14f7f17b041c699699f7bc0cf84f4fd66960249c8e`；
- `test_dropped_v1.csv` 5 行；IDRID official test 103 行。

## 3. 从原始数据重建（可选，耗时）

```powershell
& $py scripts\00_rebuild_fundus10.py
& $py scripts\01_convert_ddr.py
& $py scripts\02_preprocess.py
& $py scripts\03_quality_assess.py
& $py scripts\04_dedup.py
& $py scripts\10_clean_conflicts.py
& $py scripts\05_build_manifest.py --tag v1 --from-manifest outputs\manifest\manifest_v0_clean.csv `
  --group-map outputs\manifest\dedup_groups_v1.csv `
  --quality outputs\reports\quality.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --centered-features outputs\features\retfound_cfp_224_centered.npy `
  --feature-index outputs\features\retfound_cfp_224_index.csv
```

验收：23,294 行、conflict groups=0、无监督行=0、IDRID holdout=103、test leakage guard
删除 5 张。预处理失败必须写 `outputs/reports/preprocess_failures.csv`，不能静默丢弃。

## 4. 测试

```powershell
& $py -m pytest -q
```

验收：`27 passed, 1 skipped`（或不少于冻结版本的通过数）。

## 5. Stage 1 v2

发布版训练统一使用相对路径 manifest、预处理映射和 feature index：

```powershell
$PUB_MANIFEST = "deliverables\dataset\manifest_v1.public.csv"
$PUB_PREP = "deliverables\dataset\manifest_v1_preprocessed.public.csv"
$PUB_INDEX = "deliverables\dataset\retfound_cfp_224.public_index.csv"

& $py src\train_head.py --config configs\stage1_head_v2.yaml `
  --manifest $PUB_MANIFEST --features outputs\features\retfound_cfp_224.npy --feature-index $PUB_INDEX `
  --smoke --steps 2
& $py src\train_head.py --config configs\stage1_head_v2.yaml `
  --manifest $PUB_MANIFEST --features outputs\features\retfound_cfp_224.npy --feature-index $PUB_INDEX `
  --seed 42 --run-dir outputs\runs\stage1_v2\seed_42
& $py src\train_head.py --config configs\stage1_head_v2.yaml `
  --manifest $PUB_MANIFEST --features outputs\features\retfound_cfp_224.npy --feature-index $PUB_INDEX `
  --seed 43 --run-dir outputs\runs\stage1_v2\seed_43
& $py src\train_head.py --config configs\stage1_head_v2.yaml `
  --manifest $PUB_MANIFEST --features outputs\features\retfound_cfp_224.npy --feature-index $PUB_INDEX `
  --seed 44 --run-dir outputs\runs\stage1_v2\seed_44
```

每个 run 目录应包含 `best.pt`、`last.pt`、`metrics.csv`、`run.json`。best 只按 val macro
选择，macro 只用 n_pos≥30 的标签。

冻结参考（val）：

| seed | best epoch | macro AUROC | DR AUROC |
|---|---:|---:|---:|
| 42 | 48 | 0.903747 | 0.912822 |
| 43 | 47 | 0.902335 | 0.912559 |
| 44 | 49 | 0.903605 | 0.911190 |

## 6. Stage 2

确认 `configs/stage2_finetune.yaml` 的 `stage1_checkpoint` 指向
`outputs/runs/stage1_v2/seed_42/best.pt`，然后：

```powershell
& $py src\train_finetune.py --config configs\stage2_finetune.yaml `
  --manifest $PUB_MANIFEST --preprocessed $PUB_PREP `
  --stage1-checkpoint outputs\runs\stage1_v2\seed_42\best.pt `
  --smoke --steps 2
& $py src\train_finetune.py --config configs\stage2_finetune.yaml `
  --manifest $PUB_MANIFEST --preprocessed $PUB_PREP `
  --stage1-checkpoint outputs\runs\stage1_v2\seed_42\best.pt `
  --seed 42 --run-dir outputs\runs\stage2\seed_42
& $py src\train_finetune.py --config configs\stage2_finetune.yaml `
  --manifest $PUB_MANIFEST --preprocessed $PUB_PREP `
  --stage1-checkpoint outputs\runs\stage1_v2\seed_42\best.pt `
  --seed 43 --run-dir outputs\runs\stage2\seed_43
& $py src\train_finetune.py --config configs\stage2_finetune.yaml `
  --manifest $PUB_MANIFEST --preprocessed $PUB_PREP `
  --stage1-checkpoint outputs\runs\stage1_v2\seed_42\best.pt `
  --seed 44 --run-dir outputs\runs\stage2\seed_44
```

`manifest_v1.public.json` 是 public manifest 的 provenance sidecar。Stage 2 会把
`manifest_v1.public.csv` 的文件 SHA（`434b19…`）追溯到冻结 manifest 的语义 SHA
（`87cd43…`），再与 Stage 1 checkpoint 比较；不是关闭校验，也不是接受任意 manifest。

验收：NaN=0、OOM=0、peak VRAM < 7.5 GB、head strict load、manifest SHA256 校验通过、
非有限 batch=0。冻结参考（val）：

| seed | best epoch | macro AUROC | DR AUROC |
|---|---:|---:|---:|
| 42 | 33 | 0.960693 | 0.952548 |
| 43 | 7 | 0.957215 | 0.947269 |
| 44 | 9 | 0.959503 | 0.949522 |

## 7. 评估（先 val，后一次性 test）

### 7.1 发布版：直接用相对路径 metadata

```powershell
# 先做 val 自检；不写冻结目录
& $py src\evaluate_final.py `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --preprocessed deliverables\dataset\manifest_v1_preprocessed.public.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --feature-index deliverables\dataset\retfound_cfp_224.public_index.csv `
  --checkpoint-root deliverables\model\checkpoints `
  --report-dir outputs\reports\reproduce `
  --split val --seeds 42,43,44 --bootstrap 300

# test：只运行一次；不要覆盖交付的 final_eval.*
& $py src\evaluate_final.py `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --preprocessed deliverables\dataset\manifest_v1_preprocessed.public.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --feature-index deliverables\dataset\retfound_cfp_224.public_index.csv `
  --checkpoint-root deliverables\model\checkpoints `
  --report-dir outputs\reports\reproduce `
  --split test --final --bootstrap 1000 --hash-checkpoints

# IDRID official holdout
& $py src\evaluate_final.py `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --preprocessed deliverables\dataset\manifest_v1_preprocessed.public.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --feature-index deliverables\dataset\retfound_cfp_224.public_index.csv `
  --checkpoint-root deliverables\model\checkpoints `
  --report-dir outputs\reports\reproduce `
  --split external_holdout --final --bootstrap 1000 --hash-checkpoints
```

`deliverables\model\checkpoints` 需要 6 个文件：

```text
stage1_v2_seed42_best.pt
stage1_v2_seed43_best.pt
stage1_v2_seed44_best.pt
stage2_seed42_best.pt
stage2_seed43_best.pt
stage2_seed44_best.pt
```

只做单 seed 快速检查时用 `--seeds 42`；headline 的 3-seed mean±std 需要全部 6 个。

### 7.2 本地冻结版

```powershell
# val 自检：必须与 run.json 对齐
& $py src\evaluate_final.py --split val --bootstrap 300

# test 只运行一次
& $py src\evaluate_final.py --split test --final --bootstrap 1000 --hash-checkpoints

# IDRID official holdout 只运行一次
& $py src\evaluate_final.py --split external_holdout --final --bootstrap 1000 --hash-checkpoints
```

验收：

- val 6/6 checkpoint 与 `run.json` 或随包 `val_summary_3seed.csv` 精确一致；
- test 输出 `final_eval.{md,json,csv}`，3,474 行；
- holdout 输出 103 行；
- manifest SHA256、6 checkpoint SHA256、git commit 写入 JSON；
- 若 `final_eval.json` 已存在，不要加 `--force`；只做审计。

冻结 headline（3-seed mean ± std）：

| 指标 | Stage 1 v2 | Stage 2 |
|---|---:|---:|
| 内部 test DR AUROC | 0.9191 ± 0.0002 | 0.9531 ± 0.0034 |
| 内部 test macro AUROC | 0.9045 ± 0.0021 | 0.9462 ± 0.0081 |
| IDRID holdout DR AUROC | 0.8772 ± 0.0011 | 0.8304 ± 0.0086 |

## 8. 单图推理

```powershell
& $py ..\model\predict.py --image C:\path\to\fundus.jpg `
  --ckpt ..\model\checkpoints\stage2_seed42_best.pt
```

验收：输出 9 个 logit、原始概率、校准概率、val 阈值和 `positive_labels`。

## 9. 禁止事项

- test / IDRID holdout 只评估一次，不得回头调参、调阈值、选 seed、选 checkpoint；
- 不得覆盖 `outputs/runs/stage1_v2`、`outputs/runs/stage2`、`outputs/reports/final_eval*`；
- IDRID train/val 已参与训练，不能写成 unseen external dataset；
- csc 始终 exploratory，不进入核心 macro。
