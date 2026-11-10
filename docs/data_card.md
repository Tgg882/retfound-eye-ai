# Data Card — CFP-9 (manifest v1)

- Version: v1 (2026-11-06)
- Training input: `outputs/manifest/manifest_v1.csv` — **the only manifest allowed for training**
- Superseded: `manifest_v0.csv` (provisional baseline, untouched) and `manifest_v0_clean.csv`
  (conflict-cleaned, pre-embedding) are kept for comparison only.
- Contract: `data_contract/labels.json` (9 labels), `label_mapping.csv`, `supervision_matrix.csv`

## 1. Labels and mask semantics

`normal, dr, glaucoma, csc, disc_edema, macular_scar, myopia, retinal_det, retinitis_pig`

`m=0` = not annotated, forces `y=0`. DR `grade 0` means "no DR", never `normal=1`.

## 2. Pipeline actually run

| Step | Script | Result |
|---|---|---|
| Fundus10 rebuild (originals, Pterygium dropped) | `scripts/00_rebuild_fundus10.py` | 5,318 rows |
| DDR txt conversion (grade 5 dropped) | `scripts/01_convert_ddr.py` | 12,522 rows |
| manifest v0 (SHA1 + 256-bit pHash ≤6 + group split) | `scripts/05_build_manifest.py` | 24,250 rows, untouched baseline |
| Conflict cleaning (decision A) | `scripts/10_clean_conflicts.py` | 469 conflict groups, 951 rows removed → 23,299 |
| Preprocessing (AutoMorph fallback) | `scripts/02_preprocess.py` | 23,299 ok / 0 failed |
| Quality gate (EyeQ fallback metrics) | `scripts/03_quality_assess.py` | 242 rejects (1.04%) |
| Embedding dedup | `scripts/04_dedup.py` | 22,678 groups, max 11 |
| manifest v1 (re-group + re-split + leakage guard) | `scripts/05_build_manifest.py --tag v1` | 23,294 rows |

## 3. Sources used (v1)

| Dataset | Rows | Labels | Supervision |
|---|---:|---|---|
| MESSIDOR-2 | 1,744 | dr | class_0 → 0, class_1..4 → 1 |
| APTOS2019 | 3,654 | dr | same |
| IDRID | 515 | dr | same; official test 103 frozen as external holdout |
| DDR | 12,522 | dr | txt grade 0 → 0, 1..4 → 1, grade 5 dropped |
| Fundus10 | 4,376 | 9 labels | one-hot, other 8 labels 0 |
| PAPILA | 488 | glaucoma | class_1 → 1, class_0 → 0, normal masked |

Excluded: `04_Fundus10`, augmented copies, `holdout_split_augmented.json`, `06_Fundus_CFP`
(pseudo-labels; the processed folder does not exist, only `_raw\Fundus_CFP` parquet), OCTID,
Pterygium, DDR grade 5. `FairVLMed` is absent from this machine (external glaucoma check deferred).

## 4. Splits (v1)

| Dataset | train | val | test |
|---|---:|---:|---:|
| MESSIDOR-2 | 1,220 | 262 | 262 |
| APTOS2019 | 2,558 | 548 | 548 |
| IDRID | 288 | 62 | 165 (incl. 103 holdout) |
| DDR | 8,766 | 1,878 | 1,875 |
| Fundus10 | 3,064 | 656 | 654 |
| PAPILA | 342 | 73 | 73 |
| **Total** | **16,238** | **3,479** | **3,577** |

Policy: group-aware greedy stratified 70/15/15, seed 42, groups = patient key ∪ SHA1 ∪
(pHash ≤24 ∩ centred cosine ≥0.99). Legacy splits kept in `split_legacy`; v0 splits kept in
`outputs/manifest/splits/v0/`.

## 5. Cleaning decisions and evidence

1. **Conflicting labels (decision A).** 469 byte-identical groups carried different supervision
   (Fundus10 464, APTOS 4, IDRID 1). Whole groups deleted → 951 rows.
   Exception: one IDRID group is (`official test`, `train`) duplicates — the holdout is a hard
   constraint, so only the train copy was dropped (`IDRiD_118.jpg`), keeping holdout = 103.
   This makes the clean total 23,299 instead of the projected 23,298, and IDRID 515 instead of 514.
2. **`csc` is exploratory**: 80 positives (< the 100 G5 threshold). Threshold unchanged; status is
   recorded in `outputs/reports/label_status_v1.csv`. Early stopping must not be driven by `csc`.
3. **64-bit pHash collapses** on fundus images (19,372 images in one component). 256-bit pHash with
   Hamming ≤6 is used instead (SHA1 layer unchanged).
4. **RETFound embedding dedup needed a guard.** Raw cosine ≥0.98 merged 22,887 groups into 5,219
   (max 64); mean-centred cosine alone still chained (max 2,483 at 0.995). Final rule:
   `centred cosine ≥ 0.99 AND pHash Hamming ≤ 24`, giving 22,678 groups and max group size 11.
5. **Leakage guard**: 283 test images have a train neighbour with pHash ≤24; 5 of them reach
   centred cosine ≥0.99 and were dropped (`outputs/manifest/test_dropped_v1.csv`), 0.14% < 1%.
   Minimum pHash Hamming from test to train is 8 (> 6).
6. **DDR has no patient id** (8,250 files share the prefix `007`), so group keys fall back to the
   filename stem + pHash + embedding. Documented limitation.
7. **MESSIDOR naming is mixed** (`YYYYMMDD_<patient>_<eye>_PP` and `IMxxxxxx`); the legacy split
   put 8 patients across splits, v1 re-splits by group.

## 6. Quality gate (v1)

Thresholds from the train split (1st percentile of Laplacian variance = 58.5):
`good` 21,885 / `usable` 1,167 / `reject` 242 (1.04%). Train rows with `reject` are excluded at
training time (`drop_train_rejects: true` in `configs/stage1_head.yaml`); val/test keep them for
stratified reporting, per plan §6.2.

## 7. Licenses

MESSIDOR-2 (ADCIS, research), APTOS 2019 (Kaggle rules), IDRID (IEEE DataPort, research),
Fundus10 (Mendeley CC BY 4.0), PAPILA (figshare), DDR (CC BY-NC-SA 4.0).
