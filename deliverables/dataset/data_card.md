# Data Card — CFP-9 (manifest v1)

版本：`cfp9-v1` ｜ 生成日期：2026-11-09 ｜ 唯一允许用于训练/评估的 manifest：`outputs/manifest/manifest_v1.csv`

---

## 0. 一句话结论

这是一份 **6 个公开 CFP（彩色眼底照相）数据集合并、去重、清洗后的 9 标签多标签数据集**，
共 23,294 行、22,674 个去重组。**IDRID 的 train/val 图像参与了训练**；只有 IDRID 官方 test 的
103 张被冻结为 holdout。因此 **它不是严格意义上的 unseen-dataset external validation**（见 §9）。

---

## 1. 模态与采集特点

- 模态：全部为 **CFP（color fundus photography，彩色眼底照）**，无 OCT、无眼底荧光造影。
- 采集差异大：不同国家/中心、不同相机（Topcon/Canon/Kowa 等）、不同视场角（约 45°）、
  不同分辨率（最短边 358 px 起），存在明显亮度/对比度/色偏差异。
- 标签来源是各数据集的**数据集级标注**（分类标签或分级），不是逐病灶像素标注。
- 因此本数据集适合"多标签分类 + 域偏移评估"，不适合病灶分割或精确定量。

## 2. 来源、许可证与规模

| dataset | 来源 | 许可证 / 使用条款 | 模态 | 原始数量 | 进入 v0 | 进入 v1 | 提供的标签 |
|---|---|---|---|---:|---:|---:|---|
|---|---|---|---:|---:|---:|---|
| DDR | DDR (Diabetic Retinopathy dataset), 南京医科大学附属医院 | CC BY-NC-SA 4.0 | CFP | 13,673 | 12,522 | 12,519 | dr |
| Fundus10 | Fundus10 (Mendeley, 10 类眼底病) | CC BY 4.0 | CFP | 5,335 | 5,318 | 4,374 | 9 标签中的 7 个 + normal |
| APTOS2019 | APTOS 2019 Blindness Detection (Kaggle) | Kaggle competition rules（仅研究用途） | CFP | 3,662 | 3,662 | 3,654 | dr |
| MESSIDOR-2 | MESSIDOR-2 (ADCIS) | 研究用途授权 | CFP | 1,744 | 1,744 | 1,744 | dr |
| IDRID | IDRID (IEEE DataPort) | 研究用途授权 | CFP | 516 | 516 | 515（含 holdout 103） | dr |
| PAPILA | PAPILA (figshare) | 开放研究许可 | CFP | 488 | 488 | 488 | glaucoma |
| **合计** | | | | **25,418** | **24,250** | **23,294** | |

原始数量说明：

- Fundus10：原始 5,335 张，减去 **17 张 Pterygium** = 5,318（`scripts/00_rebuild_fundus10.py`）。
- DDR：原始 13,673 张，转换 txt 标注并**剔除 grade 5（1,151 张）** 后为 12,522（`scripts/01_convert_ddr.py`）。
- 原始合计 25,418；进入 v0 前共排除 1,168 张（DDR grade 5 1,151 + Pterygium 17）。
- APTOS2019 / IDRID 的减少来自 §6 的冲突组整组删除与 §7 的泄漏门禁。

## 3. 9 个标签定义与映射

标签顺序（**训练、checkpoint、评估、推理必须一致**，定义于 `data_contract/labels.json`）：

```
0 normal, 1 dr, 2 glaucoma, 3 csc, 4 disc_edema,
5 macular_scar, 6 myopia, 7 retinal_det, 8 retinitis_pig
```

| idx | label | 定义 | 来源与映射 |
|---:|---|---|---|
| 0 | normal | 无上述病变的正常眼底 | Fundus10 `Healthy` → 1；其余来源该标签 **masked** |
| 1 | dr | 糖尿病视网膜病变（任意分期） | MESSIDOR-2/APTOS/IDRID `class_0` → 0，`class_1..4` → 1；DDR txt `0` → 0，`1..4` → 1（**grade 5 排除**，不是 0 也不是 1） |
| 2 | glaucoma | 青光眼（视盘/视神经病变） | PAPILA `class_1` → 1，`class_0` → 0，其余来源 masked |
| 3 | csc | 中心性浆液性脉络膜视网膜病变 | Fundus10 `CSC` → 1（**exploratory**，见 §8） |
| 4 | disc_edema | 视盘水肿 | Fundus10 `Disc Edema` → 1 |
| 5 | macular_scar | 黄斑瘢痕 | Fundus10 `Macular Scar` → 1 |
| 6 | myopia | 病理性近视眼底改变 | Fundus10 `Myopia` → 1 |
| 7 | retinal_det | 视网膜脱离 | Fundus10 `Retinal Detachment` → 1 |
| 8 | retinitis_pig | 视网膜色素变性 | Fundus10 `Retinitis Pigmentosa` → 1 |

**mask 语义（关键）**：`m=1` 表示该标签被标注过，`m=0` 表示未标注并强制 `y=0`。
损失与所有指标都只在 `m=1` 的行上计算，因此"未标注"不会被当成负样本。
掩码比例：整体 72.2%（各标签 2.1%–81.2%）。

**重要限制**：本次合并后的数据中 **没有任何一张图同时带 2 个阳性标签**（multi-label 阳性图像 = 0），
9,584 张图为全 0（未标注或阴性）。因此本任务虽为多标签结构，实际近似为"多数据集单标签并集"，
跨标签共病关系无法被学到（见 §11）。

## 4. 划分（v1）

组感知（group-aware）贪心分层 70/15/15，seed=42，分层变量为 19 个标签组合层。

| dataset | train | val | test（含 holdout） | 合计 |
|---|---:|---:|---:|---:|
| DDR | 8,766 | 1,878 | 1,875 | 12,519 |
| Fundus10 | 3,064 | 656 | 654 | 4,374 |
| APTOS2019 | 2,558 | 548 | 548 | 3,654 |
| MESSIDOR-2 | 1,220 | 262 | 262 | 1,744 |
| IDRID | 288 | 62 | 165（含官方 test 103） | 515 |
| PAPILA | 342 | 73 | 73 | 488 |
| **合计** | **16,238** | **3,479** | **3,577** | **23,294** |

- **内部 test = 3,577 − 103 = 3,474 张**（最终评估用的就是这 3,474 张）。
- IDRID 官方 test 103 张固定为 external holdout，**不参与任何重划、训练或选模**。
- 旧划分保留在 `split_legacy` 列与 `outputs/manifest/splits/v0|v1/`；v0 划分文件未被覆盖。

## 5. 排除规则（硬约束）

| 排除对象 | 原因 | 门禁检查 |
|---|---|---|
| `04_Fundus10`（旧目录/旧划分） | 泄漏划分、增强图、`holdout_split_augmented.json` | G1：禁止路径行数 = 0 |
| `06_Fundus_CFP`（伪标签） | 非人工标注、不可靠 | G1：禁止路径行数 = 0 |
| OCT 数据 | 模态不一致（本任务仅 CFP） | 仅 CFP 模态 |
| Pterygium（翼状胬肉） | 不属于 9 标签体系 | 17 张在重建时剔除；G1：Pterygium 行数 = 0 |
| DDR grade 5（增殖期） | 与 1..4 的"有 DR"口径冲突，避免标签噪声 | G1：DDR label 5 行数 = 0 |
| 无法解码/损坏图 | 质量门禁 | `n_decode_failed = 0` |

## 6. 冲突组处理（方案 A）

**定义**：同一 `sha1`（字节完全相同）的多行，若监督信号不一致，即为"冲突组"。

- 处理：**整组删除**（不猜哪一行正确）。可共存的多标签不算冲突。
- 结果：**469 个冲突组，删除 951 行**（24,250 → 23,299）。
  分布：Fundus10 464 组、APTOS2019 4 组、IDRID 1 组。
- **IDRID 1 行例外**：其中一组是 `(IDRID official test, IDRID train)` 的重复。
  holdout 必须保持 103 张，因此**只删除 train 副本 `IDRiD_118.jpg`**，保留 `IDRiD_064_1.jpg`（test）。
  这使 clean 总量为 **23,299 而非 23,298**，IDRID 为 **515 而非 514**。
- 产物：`outputs/manifest/conflict_groups_v1.csv`（469 行）、`outputs/reports/manifest_v0_clean_report.json`。
- 冲突组清除后：**conflict groups = 0，无监督行 = 0**。

## 7. 去重规则（三层）

1. **SHA1 精确去重**：完全相同字节 → 同一 `group_id`。
   v1 中仍有 231 个精确重复组（470 行），但**标签完全一致**（`conflicting_groups = 0`）。
2. **pHash 感知哈希**：使用 **256-bit pHash，Hamming ≤ 6**。
   （64-bit pHash 在眼底图上会崩溃：19,372 张塌缩成一个连通块，故弃用。）
3. **RETFound embedding 去重**：`mean-centred cosine ≥ 0.99` **且** `pHash Hamming ≤ 24`。
   - 只用 raw cosine ≥0.98 会把 22,887 组塌缩到 5,219 组（最大 64）——过聚合；
   - 只用 centred cosine 会链式传播（最大 2,483）——故必须加 pHash 确认；
   - 最终 **22,678 组，548 个多成员组，最大组 11，合并边 209**；再经 §7 泄漏门禁删除 5 张后，`manifest_v1` 实际为 **22,674 组**。

**泄漏门禁（G3）**：test 图若与某训练图 `pHash ≤24` 且 `centred cosine ≥0.99`，则从 test 删除。
共 283 个候选对，**删除 5 张（0.14% < 1%）**，故 23,299 → **23,294**。
test→train 的最小 pHash Hamming = 8（> 6），最大候选 cosine = 0.9948。

## 8. 质量门禁

- 阈值来自 **train split** 的 Laplacian 方差第 1 百分位 = **58.5**。
- 标志：`good` 21,885 / `usable` 1,167 / `reject` 242（**1.04%**）。
- reject 分布：APTOS2019 118、DDR 76、PAPILA 48；原因：blur 240、dark 2。
- 训练时 `drop_train_rejects: true`（train 的 175 张 reject 不进优化器）；
  **val/test 保留 reject 行**用于分层报告，不做静默丢弃。
- 预处理：AutoMorph 在 Windows 不可用，采用计划中的 fallback（裁剪视野 + 缩放 256），
  23,299 张全部成功，**0 失败**（`outputs/reports/preprocess_report.json`）。

## 9. ⚠️ IDRID 的正确表述（禁止写成"完全未见过的新数据集"）

- **IDRID 的 train（288 张）与 val（62 张）参与了训练与选模。**
- **只有 IDRID 官方 test 的 103 张是 holdout**，且从未参与训练或选模。
- 因此本项目的"外部验证"属于 **same-dataset, official-test-split holdout**，
  **不是严格的 unseen-dataset external validation**；IDRID 这一数据源对模型并非"完全未见过"。
- 论文/报告中应写：
  > "IDRID official test split (103 images) held out from training"，
  > 而不是 "external dataset unseen during training"。

## 10. 版本与可追溯性

| 版本 | 行数 | 用途 | 状态 |
|---|---:|---|---|
| `manifest_v0.csv` | 24,250 | 临时基线（含冲突组） | 保留，不覆盖，不用于训练 |
| `manifest_v0_clean.csv` | 23,299 | 冲突清洗后（embedding 去重前） | 保留，作为 v1 的输入基线 |
| `manifest_v1.csv` | **23,294** | **唯一允许训练/评估** | 冻结 |

- `manifest_v1.csv` SHA256：`87cd436aad0118119b884b14f7f17b041c699699f7bc0cf84f4fd66960249c8e`
- DVC 指针：`outputs/manifest/manifest_v1.csv.dvc`（md5 `35fcc9684fc964ee9987f8ec9615dd98`，size 7,621,454）
- 另见 `deliverables/dataset/manifest_versions.md`

## 11. 已知数据限制

1. **无多标签阳性样本**（见 §3）：跨标签关联不可学，macro 实际是"各数据集单标签 AUROC 的平均"。
2. **标签稀疏且极不平衡**：dr 10,631 阳性 vs csc 80、disc_edema 110。
3. **csc 阳性 80 < 100 门限 → exploratory**，不参与核心 macro 结论。
4. **DDR 无患者 ID**（8,250 个文件共用前缀 `007`），分组退化为 文件名 stem + pHash + embedding。
5. **MESSIDOR-2 命名混用**（`YYYYMMDD_<patient>_<eye>_PP` 与 `IMxxxxxx`），已按组重划。
6. **各数据集采集域差异大**（见 §1），跨数据集泛化是本数据的主要风险来源。



