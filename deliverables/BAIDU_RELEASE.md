# CFP-9 百度网盘发布包说明

GitHub 只放代码、manifest、评测结果和小型 metadata；原始数据、预处理图、特征缓存和
checkpoint 放百度网盘。

## 1. 推荐分享目录

```text
cfp9_release\
├── model\
│   ├── stage1_v2_seed42_best.pt
│   ├── stage1_v2_seed43_best.pt
│   ├── stage1_v2_seed44_best.pt
│   ├── stage2_seed42_best.pt
│   ├── stage2_seed43_best.pt
│   ├── stage2_seed44_best.pt
│   └── metadata\                 # 从 deliverables\model\ 复制
│       ├── model_card.md
│       ├── predict.py
│       ├── label_mapping.json
│       ├── preprocess.yaml
│       ├── thresholds.json
│       ├── calibration.json
│       ├── checkpoint_inventory.csv
│       ├── val_summary_3seed.csv
│       └── checksums.sha256
├── preprocessed\
│   └── <APTOS2019|DDR|Fundus10|IDRID|MESSIDOR-2|PAPILA>\*.png
├── features\
│   └── retfound_cfp_224.npy
└── dataset\                      # 六个原始 CFP 数据集，按 DATA_LAYOUT.md 保持目录名
    ├── 01_MESSIDOR2\
    ├── 02_APTOS2019\
    ├── 03_IDRID\
    ├── _raw\Fundus10\images\original\
    ├── 05_PAPILA\
    └── DDR-dataset\DR_grading\
```

可选训练包（只有需要从零复现 Stage 2 训练时才需要）：

```text
RETFound_mae_natureCFP.pth        # CC BY-NC 4.0，保留 LICENSE 和署名
```

## 2. 下载后放置

```text
<repo>\deliverables\model\checkpoints\*.pt
<repo>\outputs\preprocessed\<dataset>\*.png
<repo>\outputs\features\retfound_cfp_224.npy
<repo>\dataset\01_MESSIDOR2\ ...
<repo>\dataset\02_APTOS2019\ ...
...
```

如果原始数据放在其他盘：

```powershell
$env:CFP9_DATA_ROOT = "D:\cfp9\dataset"
```

`deliverables\dataset\manifest_v1.public.csv` 和
`deliverables\dataset\manifest_v1.public.json` 必须一起保留。

## 3. 校验

```powershell
Get-FileHash -Algorithm SHA256 .\stage2_seed42_best.pt
Get-FileHash -Algorithm SHA256 .\stage1_v2_seed42_best.pt
```

与 `deliverables\model\checksums.sha256` 和 `checkpoint_inventory.csv` 对照。完整 6 个
checkpoint 的 SHA256 在 `checkpoint_inventory.csv` 中。

## 4. 数据许可

- Fundus10：CC BY 4.0，通常可分享，需署名；
- DDR：CC BY-NC-SA 4.0，非商业、署名、相同方式共享；
- APTOS2019：Kaggle 竞赛规则，不建议公开转载；
- MESSIDOR-2：ADCIS 研究授权，不建议公开转载；
- IDRID：IEEE DataPort 研究授权，不建议公开转载；
- PAPILA：以官方 figshare 条款为准。

最稳妥的做法是：原始数据只通过受控网盘给团队成员，公开仓库只放 data card、manifest、
官方下载链接和预处理脚本。
