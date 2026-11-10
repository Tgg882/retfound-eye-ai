# Provenance

交付冻结日期：2026-11-10。交付过程只做审计、复制、重聚合表格与画图；**没有重跑 test，
没有训练，没有改阈值/seed/checkpoint**。

## 数据

| 项 | 值 |
|---|---|
| manifest | `outputs/manifest/manifest_v1.csv` |
| 行数 | 23,294 |
| SHA256 | `87cd436aad0118119b884b14f7f17b041c699699f7bc0cf84f4fd66960249c8e` |
| DVC 指针 | `dataset/dvc_pointers/manifest_v1.csv.dvc` |
| 内部 test | 3,474 |
| IDRID official holdout | 103 |

## 代码与结果提交

| 角色 | commit |
|---|---|
| 评估代码（JSON 内记录） | `c5a5115a1dc783b0ac6c7c81be2ae7439b90d756` |
| 一次性 test + holdout 结果 | `e07dade` |
| Stage 2 三 seed val 汇总 | `02ad51c` |
| 最终交付内容 | 本次 delivery commit（见 `git log`） |

## 6 个 checkpoint

完整路径、epoch、bytes 与 SHA256 见 `model/checkpoint_inventory.csv`。摘要：

| model | best epoch | SHA256 |
|---|---:|---|
| Stage 1 v2 seed42 | 48 | `5a3d89f0b8b5c684b464e4a2b06ceda7e9d43c63076ee760fe7e50e1879ff940` |
| Stage 1 v2 seed43 | 47 | `19bbd9b77044cefcd2fdbfef6a74de4c1f27dffb0ce57c29de01c763fb4110ee` |
| Stage 1 v2 seed44 | 49 | `a0061564eceb2e85bf18aa51b77a00784846d0dbfb5af2deaa6167ba9dfdf5e9` |
| Stage 2 seed42（主模型） | 33 | `083a529a90c99723df0f1c49847894ebcb51cc9f837135061a72de19c8995137` |
| Stage 2 seed43 | 7 | `ed8a8fdc3f7cb628e8efa05b504054b294faedbf52abe5e2a3e1e06bcb5e88d4` |
| Stage 2 seed44 | 9 | `9cbe926c32723aa3d32635ea4ef90b648f0fb8cb1708d3cf8ca6eb919819e20c` |

交付内仅复制主模型与 Stage 1 baseline 两个 checkpoint；其余四个保留路径 + SHA256，
不复制以节约磁盘。

## 评估冻结

| 文件 | 生成时间 | 说明 |
|---|---|---|
| `evaluation/final_eval.json` | 2026-11-09 22:09:06 | test，3,474 行，3,379 组 × 1,000 bootstrap |
| `evaluation/final_eval_external_holdout.json` | 2026-11-09 22:12:20 | IDRID official holdout，103 行 |
| `evaluation/final_eval_val.json` | 2026-11-09 22:03:56 | val 自检，6/6 Δ=0 |

`evaluation/provenance_audit.json` 记录逐项核验结果。已知报告缺口：external holdout 的
md 复现表写 `not hashed`，但同一 6 个 checkpoint 的 SHA256 已由 `final_eval.json` 与磁盘
重新计算验证，原始结果文件未改。

## 禁止

test 与 IDRID holdout 结果已经冻结，禁止用于调参、调阈值、选 seed、选 checkpoint 或
反复评估后择优。
## 发布版路径层

2026-11-10 新增 `manifest_v1.public.csv`、`manifest_v1_preprocessed.public.csv` 和
`retfound_cfp_224.public_index.csv`，仅把路径列改写为相对路径，行序、标签、split、mask 和
样本集合不变。`manifest_v1.public.json` 是可机读的 provenance sidecar，记录并校验
public 文件 SHA（`434b19…`）→ 冻结 manifest SHA（`87cd43…`）、语义列哈希和 path-key 顺序。冻结的 `manifest_v1.csv`、6 个 checkpoint、`final_eval.*` 与上述 test/holdout
结果均未改动；public 副本不替代原 manifest 的 provenance。

相对路径解析由 `src/paths.py` 完成，默认数据根为仓库根目录的 `dataset/`，可用
`CFP9_DATA_ROOT` 覆盖。发布版 val 复核结果见
`evaluation/release_reproduction_check.json`：public metadata 下 6/6 checkpoint 与
训练期 val 精确一致（Δ=0），本次只做 val 自检，没有重跑 test / IDRID holdout。
