# Manifest 版本、SHA256 与 DVC 指针

本目录只保存**指针与校验和**，原始大文件通过 DVC 管理，不进入 git。

| 文件 | 行数 | SHA256 | DVC 指针 | 用途 |
|---|---|---|---|---|
| `manifest_v0.csv` | 24,250 | `32909ab3e8f96772ec36d082db05673643ce63daa6268b450802ad46975bd3a2` | `outputs/manifest/manifest_v0.csv.dvc` | 临时基线，保留不覆盖，不用于训练 |
| `manifest_v0_clean.csv` | 23,299 | `124df0fb8e9c537ef65014060ab59538c5c1071e2bca4cce43aa9fb85085f94a` | `outputs/manifest/manifest_v0_clean.csv.dvc` | 冲突清洗后（embedding 去重前），v1 的输入 |
| `manifest_v1.csv` | 23,294 | `87cd436aad0118119b884b14f7f17b041c699699f7bc0cf84f4fd66960249c8e` | `outputs/manifest/manifest_v1.csv.dvc` | 唯一允许训练/评估（冻结） |
| `dedup_groups_v1.csv` | - | `527f27344665b9c019bab2ce91b313ca7addd772a171219800ce0b45c4cb2b7f` | `outputs/manifest/dedup_groups_v1.csv.dvc` | embedding 去重 group map |
| `conflict_groups_v1.csv` | 469 组 | `a13f5c5aa7e94d495f5ea17418bf0e85c4c6baab4b3d06615023dd2b1249d887` | `outputs/manifest/conflict_groups_v1.csv.dvc` | 方案 A 冲突组清单 |
| `label_status_v1.csv` | 9 行 | `c365b71c616880ee122d1fd3459ed4a75f2694ab107ad0449e87e275379c1aa1` | （未纳管） | 标签 status（csc exploratory） |

## 特征缓存（Stage 1 输入）

| 文件 | 大小 | SHA256 |
|---|---|---|
| `retfound_cfp_224.npy` | 95,432,832 B | `579cfd0d84d6d81a0294f6f0243daceabfec60dbdfc307cbdbef0d03b40d06d8` |
| `retfound_cfp_224_index.csv` | 2,070,563 B | `2452644065dacf944e73df47741bbe2755f8454ea28fecc09fb70babec01275c` |

## 预处理表

- `manifest_v0_clean_preprocessed.csv`：4,882,398 B，SHA256 `54cf34550e365ed9f28ad7d6b444aac0fb6fb748574b88f6cab398793b08d1dd`

## 还原命令

```bash
dvc pull outputs/manifest/manifest_v1.csv.dvc
dvc pull outputs/manifest/dedup_groups_v1.csv.dvc
dvc pull outputs/manifest/conflict_groups_v1.csv.dvc
```
## 发布版相对路径副本

| 文件 | 用途 | SHA256 |
|---|---|---|
| `manifest_v1.public.csv` | GitHub/百度发布用，`path` 相对 `dataset/` | `434b191da03328f6a571eee9436df74a9faa09c57cd180fd829ad2f58e388003` |
| `manifest_v1.public.json` | provenance sidecar：public SHA → 冻结 manifest SHA；除 `path` 外语义列不变 | `21e844c308f6992f3687cebb61329d693bb30351ae6ce9f8f549c3d4e867df40` |
| `manifest_v1_preprocessed.public.csv` | `path -> outputs/preprocessed/...` 相对映射 | `0d66198ea202ca8d901601e09ae77414fe4e1c5ff261488d2503a4e227ffcf16` |
| `retfound_cfp_224.public_index.csv` | 相对路径特征索引 | `123c871b73feb0cd4cf624c199775633596e3b985c73fd99a218c8503460e672` |

冻结的 `manifest_v1.csv` 保持不变。`manifest_v1.public.json` 记录并校验：public 文件
SHA256、冻结 manifest SHA256、行数、除 `path` 外的语义表哈希、path-key 顺序哈希。
因此 Stage 2 训练用 public 副本时，仍会与 Stage 1 checkpoint 内记录的冻结 SHA 严格比较。
