# DVC 指针

这些指针从项目的 `outputs/` 目录复制而来，**不能在本目录直接 `dvc pull`**。

在项目根目录执行：

```bash
dvc pull outputs/manifest/manifest_v1.csv.dvc
dvc pull outputs/manifest/dedup_groups_v1.csv.dvc
dvc pull outputs/manifest/conflict_groups_v1.csv.dvc
dvc pull outputs/manifest/splits/v1.dvc
```

| 指针 | 实际路径 |
|---|---|
| `manifest_v1.csv.dvc` | `outputs/manifest/manifest_v1.csv` |
| `dedup_groups_v1.csv.dvc` | `outputs/manifest/dedup_groups_v1.csv` |
| `conflict_groups_v1.csv.dvc` | `outputs/manifest/conflict_groups_v1.csv` |
| `v1.dvc` | `outputs/manifest/splits/v1/` |

manifest SHA256：`87cd436aad0118119b884b14f7f17b041c699699f7bc0cf84f4fd66960249c8e`
