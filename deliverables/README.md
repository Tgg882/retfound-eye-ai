# CFP-9 Final Deliverables

CFP-only、9 标签多标签眼底照分类研究模型，编码器为 **RETFound ViT-L (CFP)**。
训练数据冻结于 `manifest_v1`；Stage 1 冻结编码器训练 head，Stage 2 从 Stage 1 v2
seed42 初始化后全量微调 ViT-L。

**研究用途，不是医疗器械，不能用于临床诊断或筛查。**

---

## 1. 四个目录

| 目录 | 内容 |
|---|---|
| `dataset/` | data card、manifest 版本/SHA256/DVC 指针、分布表、数据图 |
| `code/` | 可移植源码、configs、tests、依赖锁、复现说明 |
| `model/` | model card、主模型与 baseline checkpoint、标签映射、预处理、阈值、校准、单图推理 |
| `evaluation/` | 一次性 test / IDRID official holdout / val 自检的冻结结果、表格、图、审计 |

队友直接看：`TEAMMATE_QUICKSTART.md`（环境、推理、评测复现、训练命令）。

建议先读：

1. `dataset/data_card.md`
2. `model/model_card.md`
3. `evaluation/README.md`
4. `code/reproduce.md`

## 2. 主模型

| 角色 | 文件 | seed | val macro AUROC |
|---|---|---:|---:|
| **主模型** | `model/checkpoints/stage2_seed42_best.pt` | 42 | 0.960693 |
| baseline | `model/checkpoints/stage1_v2_seed42_best.pt` | 42 | 0.903747 |
| 稳定性模型 | Stage 2 seed43/44（百度网盘） | 43/44 | 0.957215 / 0.959503 |

seed42 是**预先指定的主 seed**，不是用 test 结果挑选的。全量 6 个 checkpoint 的路径与
SHA256 见 `model/checkpoint_inventory.csv`；交付内两个 checkpoint 的校验和见
`model/checksums.sha256`。复现 3-seed 评测需要 6 个 checkpoint；文件命名和下载位置见
`model/CHECKPOINTS_README.md`。

### 加载

```python
import torch
ckpt = torch.load("model/checkpoints/stage2_seed42_best.pt", map_location="cpu", weights_only=False)
print(ckpt["labels"])            # 9 标签顺序
print(ckpt["manifest_sha256"])   # 训练数据版本
state = ckpt["model"]            # 完整 Stage 2 state_dict
```

完整模型构造方式见 `model/predict.py`：RETFound ViT-L + 1024→1024→9 head，
池化特征 L2 normalize。Stage 1 checkpoint 只有 `ckpt["head"]`，需要额外加载
RETFound 编码器权重。

## 3. 单张图片推理

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py deliverables\model\predict.py --image C:\path\to\fundus.jpg `
  --ckpt deliverables\model\checkpoints\stage2_seed42_best.pt
```

输出 9 个标签的 logit、原始 sigmoid 概率、val 温度校准概率、val 阈值和 `positive_labels`。
`predict.py` 对原始图片执行与训练一致的视野裁剪、Resize(256)、CenterCrop(224) 和
ImageNet normalization；若输入已是 256×256 预处理 PNG，加 `--preprocessed`。

Stage 1 baseline：

```powershell
& $py deliverables\model\predict.py --image C:\path\to\fundus.jpg `
  --ckpt deliverables\model\checkpoints\stage1_v2_seed42_best.pt --stage 1 `
  --encoder-ckpt D:\weights\RETFound_mae_natureCFP.pth `
  --retfound-code RETFound-main
```

`predict.py` 已用 val 中的 APTOS 样例实测：原图与预处理 PNG 输出一致；Stage 1 编码器
加载成功。

## 3A. GitHub + 百度网盘发布版路径

推荐直接把本仓库根目录作为工作根：

```text
<repo>\
├── src\ scripts\ configs\ tests\ data_contract\
├── dataset\                         # 原始 CFP 数据（百度网盘，gitignored）
├── deliverables\
│   ├── dataset\                     # 相对路径 manifest / prep / feature index
│   ├── model\checkpoints\           # 6 个 .pt（百度网盘）
│   └── evaluation\                  # 冻结评测结果
├── outputs\
│   ├── preprocessed\                # 256x256 PNG（百度网盘）
│   └── features\                    # retfound_cfp_224.npy（百度网盘）
└── RETFound-main\                   # RETFound 代码（可从 GitHub/第三方快照获得）
```

原始数据目录必须是：

```text
dataset\
├── 01_MESSIDOR2\
├── 02_APTOS2019\
├── 03_IDRID\
├── _raw\Fundus10\images\original\
├── 05_PAPILA\
└── DDR-dataset\DR_grading\
```

注意：Fundus10 是 `_raw\Fundus10\images\original`，**不是** `04_Fundus10`。
默认数据根就是 `<repo>\dataset\`，不需要改代码；若数据在别处，设置
`$env:CFP9_DATA_ROOT`。

`deliverables\dataset\manifest_v1.public.csv` 是相对路径副本，旁边的
`manifest_v1.public.json` 是 provenance sidecar，用于把 public 文件 SHA
（`434b19…`）追溯到冻结 manifest 语义 SHA（`87cd43…`）。**不要删除 sidecar。**

## 4. 队友快速复现

完整命令与验收标准见 `code/reproduce.md`。以下命令在**仓库根目录**执行。

```powershell
$py = ".\.venv\Scripts\python.exe"

# 1) 环境
& $py -m pip install -r deliverables\code\requirements-lock.txt
& $py -m pytest -q
# 预期：27 passed, 1 skipped

# 2) 单图推理（主模型）
& $py deliverables\model\predict.py `
  --image C:\path\to\fundus.jpg `
  --ckpt deliverables\model\checkpoints\stage2_seed42_best.pt

# 3) val 自检（不消耗 test；6 个 checkpoint 需从百度网盘放入 checkpoints\）
& $py src\evaluate_final.py `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --preprocessed deliverables\dataset\manifest_v1_preprocessed.public.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --feature-index deliverables\dataset\retfound_cfp_224.public_index.csv `
  --checkpoint-root deliverables\model\checkpoints `
  --report-dir outputs\reports\reproduce `
  --split val --seeds 42,43,44 --bootstrap 300
# 预期：6/6 val macro 与 val_summary_3seed.csv 精确一致（Δ=0）

# 4) 训练代码（Stage 1 冻结编码器，只训 head）
& $py src\train_head.py --config configs\stage1_head_v2.yaml `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --features outputs\features\retfound_cfp_224.npy `
  --feature-index deliverables\dataset\retfound_cfp_224.public_index.csv `
  --smoke --steps 2

# 5) Stage 2（从 Stage 1 head 初始化，全量微调 ViT-L）
& $py src\train_finetune.py --config configs\stage2_finetune.yaml `
  --manifest deliverables\dataset\manifest_v1.public.csv `
  --preprocessed deliverables\dataset\manifest_v1_preprocessed.public.csv `
  --stage1-checkpoint outputs\runs\stage1_v2\seed_42\best.pt `
  --smoke --steps 2
```

复现 test / IDRID holdout 时，请输出到新的 `--report-dir` 并只运行一次；不要覆盖
`deliverables\evaluation\final_eval*`。

## 5. 数据来源与许可证

| 数据集 | 来源 | 许可证/条款 |
|---|---|---|
| DDR | 南京医科大学附属医院 | CC BY-NC-SA 4.0 |
| Fundus10 | Mendeley Data | CC BY 4.0 |
| APTOS2019 | Kaggle Blindness Detection | Kaggle competition rules（研究用途） |
| MESSIDOR-2 | ADCIS | 研究用途授权 |
| IDRID | IEEE DataPort | 研究用途授权 |
| PAPILA | figshare | 开放研究许可 |

RETFound 代码与权重遵循其原始仓库/权重许可证；本交付不再分发编码器权重。
每个数据集的原始条款以官方发布页为准。

## 6. 评估结论

内部 test（3,474 张，3-seed mean ± std）：

| 指标 | Stage 1 v2 | Stage 2 |
|---|---:|---:|
| DR AUROC | 0.9191 ± 0.0002 | 0.9531 ± 0.0034 |
| macro AUROC | 0.9045 ± 0.0021 | 0.9462 ± 0.0081 |

IDRID official test holdout（103 张，仅 DR 有标注）：

| 指标 | Stage 1 v2 | Stage 2 |
|---|---:|---:|
| DR AUROC | 0.8772 ± 0.0011 | 0.8304 ± 0.0086 |

Stage 2 − Stage 1 在 IDRID holdout 上为 `-0.0468 [-0.1077, +0.0133]`，**95% CI 跨 0，
不能宣称显著退化**；ECE 约 0.138–0.199（≈0.14–0.20），提示明显域偏移与校准失效。

**IDRID train/val 已参与训练；只有官方 test 103 张是 holdout。这不是严格的
unseen-dataset external validation。**

## 7. 已知限制

1. 内部 test 提升不等于外部泛化提升；IDRID holdout 与 MESSIDOR-2 上 Stage 2 方向不利。
2. IDRID holdout 的 ECE 约 0.14–0.20，跨域概率校准失效，阈值决策不可直接临床使用。
3. csc 全数据仅 80 阳性（val 12），始终 exploratory，不进入核心 macro。
4. 数据中没有多阳性图，模型学不到跨标签共病关系；macro 是各标签单任务 AUROC 的平均。
5. 各数据集相机、人群、分级口径差异大；DDR 缺患者 ID，分组依赖 SHA1/pHash/embedding。
6. APTOS/MESSIDOR/IDRID 的 DR 分级被二值化；模型不预测具体 DR grade。
7. 研究用途，非医疗器械；不能用于筛查、治疗或监管决策。

## 8. 禁止再用于调参的结果

以下结果已经冻结，**禁止**用于调参、调阈值、选 seed、选 checkpoint 或反复评估后择优：

- `evaluation/final_eval.*`（内部 test）
- `evaluation/final_eval_external_holdout.*`（IDRID official test holdout）

阈值和温度只在 val 上拟合；test / holdout 只允许一次性报告。若未来必须修正评估代码，
应新建 `final_eval_v2.*` 等版本并说明原因，不能覆盖本目录。

