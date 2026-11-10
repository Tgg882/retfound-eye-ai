# Model Card — CFP-9 / RETFound ViT-L

版本：`cfp9-v1` ｜ 冻结日期：2026-11-10 ｜ 主模型：`checkpoints/stage2_seed42_best.pt`

---

## 1. 模型定位

CFP-9 是一个 **9 标签多标签 CFP 分类研究模型**，基于 RETFound CFP 自监督预训练
ViT-L 编码器。它输出 9 个独立 sigmoid 分数；训练时使用标签掩码，未标注标签不参与损失。

**这不是医疗器械，不用于临床诊断。** 数据为多来源公开数据集的合并，标签为数据集级标签，
不是病灶级标注。

## 2. 结构与初始化

| 组件 | 说明 |
|---|---|
| 编码器 | RETFound `RETFound_mae_natureCFP.pth`，ViT-Large，`img_size=224`，`global_pool=True`，`num_classes=0` |
| 池化特征 | `forward_head(forward_features(x), pre_logits=True)`，1024 维，L2 normalize |
| 分类头 | `LayerNorm(1024) -> Linear(1024, 1024) -> GELU -> Dropout(0.3) -> Linear(1024, 9)` |
| Stage 1 | 编码器冻结，只训练分类头；best epoch 48 |
| Stage 2 | 从 Stage 1 v2 seed42 的 head 严格初始化（`strict=True`），解冻 ViT-L 全量微调；best epoch 33 |

Stage 2 超参数（`configs/stage2_finetune.yaml`）：

- epochs 50，warmup_epochs 10，batch_size 8，grad_accum 2（有效 batch 16），eval_batch_size 16
- lr 5e-4，min_lr 1e-6，layer_decay 0.65，weight_decay 0.05，grad_clip 1.0
- drop_path 0.2，dropout 0.3，label_smoothing 0.05，focal alpha 0.25
- AMP bf16，gradient checkpointing，sqrt-balanced sampler，train reject 图不进入优化器

## 3. 标签顺序（强制）

```
0 normal
1 dr
2 glaucoma
3 csc                 (exploratory)
4 disc_edema
5 macular_scar
6 myopia
7 retinal_det
8 retinitis_pig
```

`label_mapping.json` 给出完整来源映射。任何训练、评估、推理都必须使用该顺序；
checkpoint 内部也保存了同一顺序，加载时应校验。

## 4. 输入与预处理

1. RGB 彩色眼底图；读取后按 `scripts/02_preprocess.py` 的 fallback 做视野裁剪：
   灰度阈值 >10 → 最大外轮廓 → bbox + 2% margin；若覆盖率 < 0.30，改为中心正方形裁剪。
2. resize 到 256×256（cubic）。
3. `Resize(256, BICUBIC)` → `CenterCrop(224)` → `ToTensor` → ImageNet Normalize
   （mean 0.485/0.456/0.406，std 0.229/0.224/0.225）。
4. 编码器池化特征 L2 normalize 后进入分类头。

`predict.py` 对原始图片执行同一套裁剪；`--preprocessed` 用于已生成的 256×256 PNG。
完整契约见 `preprocess.yaml`。

## 5. 训练数据

- 唯一数据版本：`outputs/manifest/manifest_v1.csv`（23,294 行，SHA256
  `87cd436aad0118119b884b14f7f17b041c699699f7bc0cf84f4fd66960249c8e`）。
- 来源：DDR、Fundus10、APTOS2019、MESSIDOR-2、IDRID、PAPILA，全部为 CFP。
- 排除：OCT、`06_Fundus_CFP` 伪标签、Pterygium、DDR grade 5、Fundus10 的泄漏划分/增强图。
- split：train 16,238 / val 3,479 / test 3,577（其中 IDRID official test 103 为 holdout）。
- **IDRID train/val 已参与训练和选模；只有官方 test 103 张是 holdout。**
- 数据近似“多数据集单标签并集”：manifest 中没有一张图同时带 2 个阳性标签。

## 6. 主模型与稳定性模型

| 角色 | checkpoint | seed | best epoch | val macro AUROC | val DR AUROC |
|---|---|---:|---:|---:|---:|
| **主模型** | `stage2_seed42_best.pt` | 42 | 33 | 0.960693 | 0.952548 |
| 稳定性模型 | Stage 2 seed 43 | 43 | 7 | 0.957215 | 0.947269 |
| 稳定性模型 | Stage 2 seed 44 | 44 | 9 | 0.959503 | 0.949522 |
| baseline | `stage1_v2_seed42_best.pt` | 42 | 48 | 0.903747 | 0.912822 |

**seed 42 是预先指定的主 seed，不是用 test 结果挑出来的。** 3-seed 均值是 headline；
seed42 只作为预先指定的配对参考。

## 7. 阈值与校准

- 阈值：每个标签在 **val** 上取 specificity=0.90 对应的 raw-logit 阈值；decision rule 为
  `positive if raw_logit > threshold`。阈值没有在 test 或 IDRID holdout 上调整。
- 校准：每标签 temperature scaling，`p = sigmoid(logit / T)`，T 仅在 val 上按 NLL 拟合。
  temperature 只影响概率，不改变阈值决策。
- 完整数值：`thresholds.json`、`calibration.json`。

注意：稀有标签的 val 阳性很少（如 csc val n_pos=12），阈值对应的敏感度估计不稳定；
单张图的 `positive_labels` 不能直接解释为临床多病共存诊断。

## 8. 内部 test 结果（一次性评估，禁止再用于调参）

3-seed mean ± std：

| 指标 | Stage 1 v2 | Stage 2 |
|---|---:|---:|
| DR AUROC | 0.9191 ± 0.0002 | 0.9531 ± 0.0034 |
| macro AUROC（test n_pos≥30） | 0.9045 ± 0.0021 | 0.9462 ± 0.0081 |
| DR AUPR | 0.9140 ± 0.0010 | 0.9508 ± 0.0081 |

配对 bootstrap（Stage 2 − Stage 1）：DR AUROC `+0.0339 [+0.0270, +0.0403]`，
macro AUROC `+0.0417 [+0.0310, +0.0534]`。

## 9. IDRID official test holdout（103 张）

| 指标 | Stage 1 v2 | Stage 2 |
|---|---:|---:|
| DR AUROC（3-seed） | 0.8772 ± 0.0011 | 0.8304 ± 0.0086 |
| DR AUPR（3-seed） | 0.9453 ± 0.0006 | 0.8977 ± 0.0171 |

配对差（Stage 2 − Stage 1）：`-0.0468 [-0.1077, +0.0133]`，p=0.134。
**方向不利，但 95% CI 跨 0，不能宣称显著退化。** holdout 只有 DR 有标注（n_pos=69）。

## 10. 已知限制

1. **内部 test 提升不等于外部泛化提升**：Stage 2 在内部 test 明显优于 Stage 1，但在
   IDRID official holdout 上方向相反。IDRID train/val 已参与训练，因此这不是严格
   unseen-dataset external validation。
2. **外部校准较差**：IDRID holdout 上 ECE 约 0.138–0.199（≈0.14–0.20），提示明显域偏移与
   概率校准失效；阈值决策也应谨慎。
3. **csc 为 exploratory**：全数据仅 80 阳性（val 12），低于 100 阳性门限，不进入核心 macro。
4. **标签不是多病共存标注**：训练集中没有多阳性图，模型无法学习跨标签共病关系。
5. **数据域差异**：DDR/APTOS/IDRID/MESSIDOR/Fundus10/PAPILA 的相机、人群、分级口径不同；
   MESSIDOR-2 上 Stage 2 DR AUROC（0.8252）反而低于 Stage 1（0.8455）。
6. **无患者级完整标识**：DDR 缺患者 ID；分组依赖文件名、pHash、embedding 去重。
7. **非临床用途**：不能用于筛查决策、治疗建议或监管用途。

## 11. 适用场景

- 研究性 CFP 多标签表征学习与迁移学习；
- 内部 test 与 official-test holdout 的方法比较；
- 域偏移、校准、去重与数据泄漏控制研究。

不适用于：临床诊断、真实世界筛查部署、跨中心无校准直接使用。
