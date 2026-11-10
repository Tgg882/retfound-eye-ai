# CFP-9 · RETFound 眼底多疾病多标签分类（发布版）

基于 **RETFound ViT-L/16（CFP 预训练权重）** 两阶段迁移的 **9 标签彩色眼底照（CFP）** 分类：
训练代码、数据契约、冻结评测结果、推理脚本与模型分发说明。

`Python 3.10` · `PyTorch 2.8.0 + CUDA 12.8` · `ViT-L/16 · 304.4M 参数` · `6 个公开 CFP 数据集 · 23,294 张` · `9 标签 masked BCE`

> ⚠️ **研究用途。** 本项目不是医疗器械，不能用于临床诊断、筛查或任何医疗决策。

---

## 1. 这是什么

| 项目 | 说明 |
|---|---|
| 任务 | 单张彩色眼底照 → 9 个疾病标签的**独立概率**（多标签，不是 softmax 单选） |
| 骨干 | RETFound `vit_large_patch16`（CFP 版），输入 224×224，输出 1024-d 特征 |
| 头部 | `LayerNorm(1024) → Linear(1024→1024) → GELU → Dropout(0.3) → Linear(1024→9)` |
| 损失 | masked BCE + label smoothing 0.05 + 按标签频次 `count^-0.25` 重加权；未见标注的标签用 mask 屏蔽 |
| 影像模态 | **仅 CFP**（彩色眼底照）。不含 OCT、不含超广角 |
| 主模型 | `deliverables/model/checkpoints/stage2_seed42_best.pt`（3.4 GB，**完整模型**） |
| 数据规模 | 6 个公开数据集，23,294 张有监督图像；train 16,238 / val 3,479 / test 3,577 |

**两阶段训练：**

1. **Stage 1 —— 只训头。** 冻结 RETFound 编码器，用预先缓存的 1024-d 特征训练 9 标签 MLP。
   这一步不碰 ViT，普通 GPU 甚至 CPU 都能跑，几分钟一轮。
2. **Stage 2 —— 头 + ViT-L 全量微调。** 以 Stage 1 的 head **严格初始化**
   （`strict=True` + manifest SHA256 双重校验，标签顺序不一致直接报错退出），
   再解冻 ViT-L 微调 50 epoch：lr 5e-4、layer decay 0.65、warmup 10 epoch、
   drop_path 0.2、bf16、梯度检查点、micro-batch 8 × 梯度累积 2（有效 batch 16）。

---

## 2. 结果速览

**内部 test（3,474 张 / 3,379 组，group-aware bootstrap，3 个 seed）：**

| 指标 | Stage 1 v2（只训头） | Stage 2（全量微调） |
|---|---:|---:|
| DR AUROC | 0.9191 ± 0.0002 | **0.9531 ± 0.0034** |
| macro AUROC（5 个达标标签） | 0.9045 ± 0.0021 | **0.9462 ± 0.0081** |
| DR AUPR | 0.9140 ± 0.0010 | **0.9508 ± 0.0081** |
| macro AUPR | 0.6847 ± 0.0016 | **0.8359 ± 0.0093** |
| DR sens @ 90% spec | 0.7576 ± 0.0016 | **0.8762 ± 0.0252** |

**Stage 2 − Stage 1（配对 bootstrap，3-seed 均值）：**

- DR AUROC `+0.0339 [+0.0270, +0.0403]`
- macro AUROC `+0.0417 [+0.0310, +0.0534]`

**IDRID 官方 test holdout（103 张，仅 DR 有标注，**训练期全程未打开**）：**

| 指标 | Stage 1 v2 | Stage 2 |
|---|---:|---:|
| DR AUROC | **0.8772 ± 0.0011** | 0.8304 ± 0.0086 |
| DR AUPR | 0.9453 ± 0.0006 | 0.8977 ± 0.0171 |
| ECE | 0.15–0.21 | 0.14–0.21 |

配对差 `-0.0468 [-0.1077, +0.0133]`，**p = 0.134，95% CI 跨 0，不能宣称显著退化**；
但方向不利，且 ECE 0.14–0.21（内部 test 为 0.02–0.07）是明确的域偏移 + 校准失效信号。
详见 [deliverables/evaluation/README.md](deliverables/evaluation/README.md)。

---

## 3. 九个标签

标签顺序是**契约的一部分**（`data_contract/labels.json`），checkpoint 与 manifest 都按此顺序校验，
改动顺序会导致 Stage 2 初始化直接失败。

| 序号 | 标签 | 含义 | val 阳性数 | 计入 macro |
|---:|---|---|---:|---|
| 0 | `normal` | 正常眼底 | 120 | ✅ |
| 1 | `dr` | 糖尿病视网膜病变（主指标） | 1,588 | ✅ |
| 2 | `glaucoma` | 青光眼 | 173 | ✅ |
| 3 | `csc` | 中心性浆液性脉络膜视网膜病变 | 12 | ✗ exploratory |
| 4 | `disc_edema` | 视盘水肿 | 16 | ✗ |
| 5 | `macular_scar` | 黄斑瘢痕 | 51 | ✅ |
| 6 | `myopia` | 近视性眼底改变 | 48 | ✅ |
| 7 | `retinal_det` | 视网膜脱离 | 18 | ✗ |
| 8 | `retinitis_pig` | 视网膜色素变性 | 20 | ✗ |

macro 只统计 val 阳性数 ≥ 30 的 5 个标签；`csc` 全程标为 exploratory，不进核心结论。

---

## 4. 仓库结构

```text
retfound-eye-ai\
├── src\                     # 训练 / 评测 / 推理核心代码
│   ├── train_head.py        #   Stage 1：只训 9 标签头（读缓存特征）
│   ├── train_finetune.py    #   Stage 2：从 Stage 1 初始化，全量微调 ViT-L
│   ├── evaluate_final.py    #   冻结评测：val / test / IDRID holdout，bootstrap、ECE
│   ├── encoder.py           #   RETFound 加载（自动找 models_vit.py 与编码器权重）
│   ├── head.py losses.py samplers.py dataset.py manifest_provenance.py paths.py
├── scripts\                 # 数据管线：重建、格式转换、预处理、质量、去重、manifest
├── configs\                 # stage1_head_v2.yaml / stage2_finetune.yaml（冻结配置）
├── data_contract\           # labels.json、label_mapping.csv、supervision_matrix.csv
├── tests\                   # schema / mask / 标签对齐 / 路径可移植性测试
├── deliverables\
│   ├── TEAMMATE_QUICKSTART.md   # ★ 队友入口：推理 / 评测复现 / 训练
│   ├── BAIDU_RELEASE.md         # ★ 百度网盘大文件放置与校验
│   ├── dataset\                 # data card、分布图、相对路径 manifest + sidecar
│   ├── model\                   # model card、predict.py、阈值/校准、checkpoint 目录
│   ├── evaluation\              # 冻结评测 JSON/CSV/MD + 图表 + provenance 审计
│   └── code\                    # 发布版代码快照 + 依赖锁 + third_party\RETFound-main
└── outputs\                 # 预处理图 / 特征缓存 / 训练 run（gitignored）
```

---

## 5. 环境要求

| 项目 | 要求 / 实测 |
|---|---|
| Python | 3.10（实测 3.10.19） |
| PyTorch | **2.8.0 + cu128**。RTX 50 系是 sm_120，**cu121 及更早的轮子跑不起来** |
| timm | **0.9.2**（1.x 会让 `models_vit.py` 报 `patch_size` 错） |
| 显存（推理） | 单图即可跑；GPU 上模型加载后每张图秒级 |
| 显存（Stage 2 训练） | 8 GB 可跑：实测峰值 **5.30 GB**，50 epoch ≈ **7.5 h** |
| 磁盘 | 6 个 checkpoint ≈ 10.2 GB，预处理图 + 特征 ≈ 2 GB |

依赖完整锁版本见 `deliverables/code/requirements-lock.txt`。

---

## 6. 三种用法

完整步骤见 **[deliverables/TEAMMATE_QUICKSTART.md](deliverables/TEAMMATE_QUICKSTART.md)**。

```powershell
Set-Location <repo>
py -3.10 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r deliverables\code\requirements-lock.txt
& $py -m pytest -q
```

### 6.1 单图推理（只要主模型）

主模型是**完整** Stage 2 checkpoint，已含 ViT-L 全部权重，**不需要**再下载
`RETFound_mae_natureCFP.pth`；但需要仓库内的 RETFound 源码
（`deliverables/code/third_party/RETFound-main/models_vit.py`，已在仓库中）。

```powershell
& $py deliverables\model\predict.py `
  --image C:\path\to\fundus.jpg `
  --ckpt deliverables\model\checkpoints\stage2_seed42_best.pt
```

输出 9 个标签的 `logit`、`prob`、温度校准后 `prob_calibrated`、val 拟合阈值和
`positive_labels`。仅作研究展示。

### 6.2 复现评测

需要 6 个 checkpoint（3 seed × 2 stage）+ 特征缓存。先跑 val 自检，通过后才允许跑 test：

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

预期 `VAL SELF-CHECK: PASS`，6/6 macro AUROC 与 `deliverables/model/val_summary_3seed.csv` **精确一致（Δ=0）**。

> **test 与 IDRID holdout 只允许各跑一次。** 已冻结结果在 `deliverables/evaluation/`，
> 不要用 test 选 seed、调阈值或反复评估后择优。

### 6.3 训练

| 阶段 | 命令 | 额外需要 |
|---|---|---|
| Stage 1 | `src\train_head.py --config configs\stage1_head_v2.yaml` | `outputs\features\retfound_cfp_224.npy`（百度网盘） |
| Stage 2 | `src\train_finetune.py --config configs\stage2_finetune.yaml` | 上述 + 6 数据集预处理图 + `RETFound_mae_natureCFP.pth` |

Stage 2 **从零复现**需要 RETFound 预训练权重（CC BY-NC 4.0，不随仓库分发）：

```powershell
$env:CFP9_RETFOUND_CKPT = "D:\weights\RETFound_mae_natureCFP.pth"
```

---

## 7. 数据与大文件放哪

GitHub 只放代码、manifest、评测结果和小型 metadata；原始数据与模型权重走百度网盘。

| 路径 | 内容 | 来源 | 体积 |
|---|---|---|---|
| `dataset\` | 6 个原始 CFP 数据集 | 百度网盘 | 数十 GB |
| `outputs\preprocessed\` | 256×256 PNG（预处理后） | 百度网盘 | 1.9 GB（23,299 张） |
| `outputs\features\retfound_cfp_224.npy` | Stage 1 特征缓存 | 百度网盘 | 91 MB |
| `deliverables\model\checkpoints\*.pt` | 6 个 checkpoint | 百度网盘 | 10.2 GB |
| `deliverables\dataset\manifest_v1.public.csv` | 相对路径 manifest | **GitHub** | 6.4 MB |

原始数据目录名必须保持（`src/paths.py` 按这些名字把旧绝对路径映射到本地）：

```text
dataset\
├── 01_MESSIDOR2\
├── 02_APTOS2019\
├── 03_IDRID\
├── _raw\Fundus10\images\original\      # 注意不是 04_Fundus10
├── 05_PAPILA\
└── DDR-dataset\DR_grading\
```

数据放在别的盘：

```powershell
$env:CFP9_DATA_ROOT = "D:\cfp9\dataset"
```

`manifest_v1.public.csv` 与 `manifest_v1.public.json` **必须成对保留**：sidecar 把公开 manifest
的文件哈希追溯到冻结 manifest 的 `87cd436a…`，这样坐标改了也不会绕过 Stage 2 的一致性校验。

### `retfound_cfp_224.npy` 是什么

`outputs\features\retfound_cfp_224.npy` 是 **Stage 1 用的 RETFound 特征缓存**：

- 形状 `(23299, 1024)`，`float32`，逐行 L2 归一化（行范数 = 1），约 91 MB；
- 每行 = 一张**预处理后**图像过一遍冻结 RETFound ViT-L 得到的 CLS 特征；
- 行序由同名索引 `retfound_cfp_224_index.csv`（或发布版 `deliverables\dataset\retfound_cfp_224.public_index.csv`）给出；
- 由 `scripts/04_dedup.py` 生成——同一次前向既用于 embedding 去重（泄露守卫），也缓存下来供 Stage 1 使用；
- 有它，Stage 1 训练和 Stage 1 评测**完全不用加载 ViT-L**，几分钟就能跑完；没有它，Stage 1 和
  3-seed 评测复现都做不了。推理主模型（Stage 2）**不需要**这个文件。

---

## 8. 数据来源与许可

| 数据集 | 用途标签 | 许可 / 条款 | 官方入口 |
|---|---|---|---|
| [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection) | dr | Kaggle 竞赛规则（仅研究，不建议公开转载） | Kaggle |
| [MESSIDOR-2](https://www.adcis.net/en/third-party/messidor2/) | dr | ADCIS 研究授权 | ADCIS |
| [IDRiD](https://idrid.grand-challenge.org/) | dr | IEEE DataPort 研究授权 | Grand Challenge |
| [DDR](https://github.com/nkicsl/DDR-dataset) | dr | CC BY-NC-SA 4.0 | GitHub |
| PAPILA | glaucoma | 开放研究许可（以官方 figshare 条款为准） | 见 `deliverables/dataset/data_card.md` |
| Fundus10 | normal / 其余多标签 | CC BY 4.0（课题组整理版） | 见 `deliverables/dataset/data_card.md` |

**原始图像请勿公开分发**给课题组成员以外的人。每个数据集的画像、划分、已知缺陷见
[deliverables/dataset/data_card.md](deliverables/dataset/data_card.md)。

---

## 9. 复现状态

| 检查 | 结果 |
|---|---|
| 全新 clone 上的单元测试 | `12 passed, 16 skipped`（skip = `manifest not built yet`，属正常） |
| 完整管线（manifest 已构建） | `27 passed, 1 skipped` |
| val 自检 6 个 checkpoint | Δ = 0，与 `val_summary_3seed.csv` 精确一致 |
| 训练稳定性 | 3 seed × 50 epoch，nonfinite batch = 0，峰值显存 5.30 GB |
| 可复现性 | 同 seed 重跑 manifest 逐字节一致；checkpoint 重载后 AUROC Δ = 0 |
| test / IDRID holdout | 各只运行一次，结果冻结在 `deliverables/evaluation/` |

---

## 10. 已知限制

1. **IDRID holdout 不是严格的 unseen-dataset 外部验证**：IDRID 的 train/val 已参与训练，
   只有官方 test 的 103 张是 holdout，且仅 DR 有标注（n_pos = 69）。
2. **稀有标签样本极少**：`csc` / `disc_edema` / `retinal_det` / `retinitis_pig` 的 val 阳性只有 12–20 张，
   AUROC 方差大，不作为结论依据。
3. **`positive_labels` 阈值偏噪**：阈值按各标签 val 90% 特异性独立拟合，稀有标签阈值很低，
   一张正常图也可能挂多个阳性。做展示建议用单标签概率。
4. **域偏移**：MESSIDOR-2 上 Stage 2 的 DR AUROC 反而从 0.846 降到 0.825；holdout ECE 0.14–0.21。
5. **只覆盖 CFP**，且标签体系来自 6 个数据集的并集，不等价于临床诊断分类体系。

---

## 11. 引用与许可

- 本项目代码：见 [deliverables/LICENSE](deliverables/LICENSE)
- RETFound 代码与权重：CC BY-NC 4.0，见 [deliverables/THIRD_PARTY_NOTICES.md](deliverables/THIRD_PARTY_NOTICES.md)
  与 `deliverables/code/third_party/RETFound-main/LICENSE`
- 各数据集：遵循各自官方条款（见第 8 节与 data card）

若使用本仓库，请同时引用 RETFound 原文：

> Zhou Y, Chia MA, Wagner SK, et al. *A foundation model for generalizable disease detection
> from retinal images.* Nature, 2023.

---

## 12. 禁止事项

- 不要把 `dataset\`、`outputs\preprocessed\`、`outputs\features\`、`outputs\runs\` 或
  `deliverables\model\checkpoints\*.pt` 提交到 GitHub；
- 不要用 test / IDRID holdout 调参、调阈值、选 seed 或反复评估后择优；
- 不要把本项目输出当作临床诊断结论或对外宣传为医疗器械。