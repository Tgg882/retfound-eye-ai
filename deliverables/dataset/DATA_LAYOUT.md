# CFP-9 release data layout

The default data root is:

```text
<repo>\dataset
```

The release `manifest_v1.public.csv` uses paths relative to this directory. Keep
`manifest_v1.public.json` next to it: the sidecar maps the public file SHA to the frozen
manifest SHA and lets Stage 2 verify that only the `path` column changed.

## Required source layout

```text
dataset\
├── 01_MESSIDOR2\
│   ├── train\class_0..4\
│   ├── val\class_0..4\
│   └── test\class_0..4\
├── 02_APTOS2019\
│   ├── train\class_0..4\
│   ├── val\class_0..4\
│   └── test\class_0..4\
├── 03_IDRID\
│   ├── train\
│   ├── val\
│   └── test\                 # 103 official holdout images are marked in manifest
├── _raw\
│   └── Fundus10\
│       └── images\
│           └── original\
│               └── <9 class folders>\*.jpg
├── 05_PAPILA\
│   ├── train\
│   ├── val\
│   └── test\
└── DDR-dataset\
    └── DR_grading\
        ├── train.txt
        ├── valid.txt
        ├── test.txt
        ├── train\
        ├── valid\
        └── test\
```

**Do not use `04_Fundus10` as the Fundus10 source.** It contains augmented images and
the legacy split; the frozen manifest excludes it. The authoritative source is
`_raw\Fundus10\images\original`.

## Other roots

```powershell
$env:CFP9_DATA_ROOT = "D:\datasets\cfp9"      # raw CFP data
$env:CFP9_RETFOUND_DIR = "D:\code\RETFound-main"
$env:CFP9_RETFOUND_CKPT = "D:\weights\RETFound_mae_natureCFP.pth"
```

`outputs\preprocessed\` and `outputs\features\` are generated/restored artefacts, not raw data.
They go in the Baidu package, not GitHub.
