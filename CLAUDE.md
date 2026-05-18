# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mtDNA Sanger sequencing quality control pipeline for forensic/degraded DNA samples.
- **Phase 1 (done):** Rule-based QC + ML-ready dataset + XGBoost file-level classifier (recall=0.915, AUC≈0.96)
- **Phase 2 (done):** Windowed-QC pseudo-labeler + 1D U-Net for per-scan-point noise segmentation

**Core problem:** Degraded DNA samples show noise in the *middle* of AB1 traces (not just ends), causing false positives/negatives in variant calling. Pipeline reads Tracy-processed JSON outputs and classifies HV regions as clean (0) or noisy (1).

## Commands

```bash
# Generate labels from metadata
python src/utils/generate_labels.py

# Build feature matrix from labels CSV
python src/utils/features.py labels_v2.csv

# Train file-level XGBoost classifier (GroupKFold, recall-optimized)
python src/train.py --features features.csv --folds 5 --beta 2.0

# Run inference on test_data (trains on full labeled data then predicts)
python src/predict.py --test-dir data/test_data --features features.csv

# Plot model comparison charts
python src/plot.py --features features.csv

# Test windowed-QC pseudo-labeler on one trace
python src/utils/windowed_qc.py data/pipeline_results/BATCH/SAMPLE/TRACE.json

# Visualize segmentation pseudo-labels on one trace
python src/plot_seg.py data/pipeline_results/BATCH/SAMPLE/TRACE.json

# Train 1D U-Net segmentation model (uses pseudo-labels from windowed QC)
python src/train_seg.py --labels labels_v2.csv --epochs 50 --batch 8

# Lint
ruff check src/
```

## Data Layout

```
pipeline_results/
  YYYYMMDD_mtDNA_NN/          # batch folder
    SAMPLE_ID/                # numeric (253719) or coded (LN_25_AA0835)
      WELL_DATE_SAMPLE_REGION_RUN.json   # trace data
      WELL_DATE_SAMPLE_REGION_RUN.abif  # raw trace (not used by ML)
      comparison.json                    # excluded from labeling
      variants.json                      # excluded from labeling

metadata_rerun.tsv             # ground truth: columns Sample.1, Issues, Issues.1, Issues.2
labels_v2.csv                  # generated: (FilePath, Label) — input to training
```

**Four HV regions per sample:** `HV1F`, `HV1R`, `HV2F`, `HV3R`.

## JSON Trace Format

Each trace JSON (output of Tracy aligner) contains:
- `peakA`, `peakC`, `peakG`, `peakT` — raw intensity arrays (4-channel signal)
- `pos` — scan positions
- `basecalls`, `primarySeq`, `secondarySeq` — called bases
- `variants` — called variants vs rCRS reference
- `allele1fraction`, `allele2fraction` — heteroplasmy fractions

## Key Source Files

| File | Purpose |
|------|---------|
| `src/utils/generate_labels.py` | Parses `metadata_rerun.tsv` → `labels_v2.csv` |
| `src/utils/features.py` | 21-feature extractor from Tracy JSON; `extract_features()`, `build_feature_matrix()` |
| `src/utils/windowed_qc.py` | Windowed rule-based QC → per-scan-point pseudo-labels; `compute_noisy_mask()` |
| `src/utils/seg_dataset.py` | PyTorch Dataset for U-Net: loads JSON, builds 20-channel tensor + pseudo-label mask |
| `src/models/segnet.py` | 1D U-Net (`UNet1D`); input (B,20,4096) → output (B,1,4096) noisy prob; ~2.7M params |
| `src/train.py` | File-level XGBoost/LGB/RF classifier with GroupKFold, F_beta threshold tuning, SHAP |
| `src/train_seg.py` | U-Net training with focal+Dice loss, GroupKFold CV, cosine LR schedule |
| `src/predict.py` | Inference on test_data (no labels), outputs predictions.csv |
| `src/plot.py` | 5-model comparison: ROC, PR curves, recall bar charts |
| `src/plot_seg.py` | Per-trace visualization of pseudo-labels and U-Net predictions |
| `src/utils/seq.py` | `load_peaks()`, `normalize_peaks()`, `pad_or_truncate()`, `PeakDataset` (legacy) |

## Architecture

**Labeling pipeline** (`generate_labels.py`):
1. Reads `metadata_rerun.tsv` — columns `Sample.1` (sample ID) and `Issues`/`Issues.1`/`Issues.2` (free-text run notes)
2. Parses Issues text with regex → maps to noisy HV region set per sample
3. Skips samples with contamination notes (`nhiễm mẫu`) or no region mentioned (variant interpretation issues like `16355del?` are NOT noise signals)
4. Walks `pipeline_results/` recursively, extracts `(sample_id, region)` from filename, assigns label 0/1
5. Outputs `labels_v2.csv` (training) and `labels_v2.debug.csv` (with SampleID/Region columns)

**File-level ML pipeline** (`src/train.py`):
- 21 features from `features.py` (peak quality, alignment scores, variant burden, signal shape)
- 5 models: XGBoost (best), LightGBM, RandomForest, GradientBoosting, LogisticRegression
- GroupKFold (n=5) splits by SampleID (regex-extracted from FilePath) to prevent leakage
- `recall_boost=1.5` multiplies `scale_pos_weight` above the class-imbalance ratio
- F_beta threshold tuning on OOF probabilities (default beta=2, min_recall=0.85)
- Best result: XGBoost threshold=0.28 → Recall=0.915, Precision=0.742, F2=0.874

**Segmentation pipeline** (`windowed_qc.py` + `segnet.py` + `train_seg.py`):
- `compute_noisy_mask()`: sliding window (200 pts, 50% overlap) over total signal; flags window if ANY: frac_low_pr>0.40, qual_mean<15, signal_prominence<0.10, drift_ratio>2.5
- AUC of noisy_frac as file-level predictor: 0.785 (sanity check on 2599 traces)
- Pseudo-labels back-projected to scan points via overlap-vote (≥50% covering windows = noisy)
- `UNet1D`: 4-level 1D encoder-decoder, ~2.7M params, base_ch=32; input 20 channels × 4096 pts
- 20 input channels: 4 raw peaks + 8 derivatives (1st + 2nd) + total + max_ch + peak_ratio + bc_mask + bc_qual + rolling_SNR + rolling_baseline + position_index
- Loss: focal (γ=2, α=0.75) × 0.6 + Dice × 0.4 (handles class imbalance in noisy pixels)
- Training: `python src/train_seg.py --labels labels_v2.csv --epochs 50 --batch 8`
- GroupKFold by SampleID; best checkpoint saved per fold at `checkpoints/seg_fold{k}_best.pth`

**Data paths in `generate_labels.py`** currently point to `ROOT/data/pipeline_results` and `ROOT/data/metadata_rerun.tsv` — but actual data lives at repo root (`/mnt/d/mtdna/pipeline_results/`, `/mnt/d/mtdna/metadata_rerun.tsv`). The `data/` subdirectory does not exist. Fix paths if re-running.

## Labeling Rules (critical logic)

- `HV1` alone in Issues → both `HV1F` and `HV1R` noisy
- `HV3F` in Issues → mapped to `HV3R` (only HV3 direction in dataset)
- "4 trình tự failed / fail 4 chiều / fail trình tự" → all 4 regions noisy
- "nhiễm mẫu / nhầm mẫu / nghi ngờ nhiễm" → skip entire sample (mixed signal, unreliable)
- Issues with only variant positions (e.g. `73A?`, `16355del?`) → skip sample

## Environment

- Python 3.13, PyTorch 2.11, NumPy 2.4, Pandas 3.0
- Linter: `ruff` (target: `ruff check src/`)
- Definition of Done requires `ruff clean` + benchmark showing ≥40% false-variant reduction on degraded batch
