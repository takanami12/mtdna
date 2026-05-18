"""
src/predict.py
==============
Run trained XGBoost model on unlabeled test data in data/test_data/.

Usage:
    python src/predict.py
    python src/predict.py --test-dir data/test_data --features features.csv --threshold 0.28
    python src/predict.py --threshold 0.20        # lower threshold = higher recall

Output:
    data/test_data/predictions.csv  — columns:
        file_path, sample_id, region, probability, predicted_label, flagged_noisy

No labels are required for test_data — this is pure inference.
The model is trained on the full labeled dataset (features.csv / labels_v2.csv).
"""

from __future__ import annotations

import argparse
import copy
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import fbeta_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# Keep imports relative so the script works from repo root
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.features import extract_features
from train import ALL_FEATURES, make_models

EXCLUDE_FILES = {"comparison.json", "variants.json"}
TRACE_RE = re.compile(
    r"_(\d{6}|LN_\d+_[A-Z]{2}\d+|MS_\d+_[A-Z]{2}\d+|AB_\d+_[A-Z]{2}\d+)"
    r"_(HV\d[FR])_\d+\.json$",
    re.IGNORECASE,
)


def discover_test_files(test_dir: Path) -> list[dict]:
    """Walk test_dir and return list of {path, sample_id, region} for trace JSONs."""
    records = []
    for p in sorted(test_dir.rglob("*.json")):
        if p.name in EXCLUDE_FILES:
            continue
        m = TRACE_RE.search(p.name)
        if m is None:
            continue
        records.append({
            "path": p,
            "sample_id": m.group(1),
            "region": m.group(2).upper(),
        })
    return records


def extract_test_features(records: list[dict]) -> pd.DataFrame:
    """Extract features for all test files. Drops files that fail to parse."""
    rows = []
    skipped = 0
    for rec in records:
        feat = extract_features(rec["path"])
        if feat is None:
            skipped += 1
            continue
        feat["file_path"] = str(rec["path"])
        feat["sample_id"] = rec["sample_id"]
        feat["region"] = rec["region"]
        rows.append(feat)
    if skipped:
        print(f"  Warning: {skipped} files skipped (missing or unreadable)")
    return pd.DataFrame(rows)


def train_on_full_data(
    features_csv: Path, recall_boost: float = 1.5
) -> xgb.XGBClassifier:
    """Train XGBoost on full labeled dataset."""
    fdf = pd.read_csv(features_csv, index_col=0)
    X = fdf[ALL_FEATURES]
    y = fdf["label"]
    n_noisy, n_clean = int(y.sum()), int((y == 0).sum())
    model = copy.deepcopy(make_models(n_noisy, n_clean, recall_boost)["XGBoost"])
    model.fit(X, y)
    print(f"  Model trained on {len(X)} files (noisy={n_noisy}, clean={n_clean})")
    return model


def predict(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(X_test[ALL_FEATURES])[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict noisy traces on test data")
    parser.add_argument("--test-dir", default="data/test_data",
                        help="Directory containing test batch folders")
    parser.add_argument("--features", default="features.csv",
                        help="Training features CSV (for model training)")
    parser.add_argument("--threshold", type=float, default=0.28,
                        help="Classification threshold (default: 0.28, F2-optimized on OOF)")
    parser.add_argument("--recall-boost", type=float, default=1.5)
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    features_csv = Path(args.features)

    sep = "=" * 65
    print(f"\n{sep}\n  Step 1: Discover test files\n{sep}")
    records = discover_test_files(test_dir)
    print(f"  Found {len(records)} trace JSON files in {test_dir}")
    if not records:
        print("  ERROR: No trace JSON files found. Check --test-dir path.")
        return

    print(f"\n{sep}\n  Step 2: Extract features\n{sep}")
    feat_df = extract_test_features(records)
    print(f"  Features extracted: {len(feat_df)} files × {len(ALL_FEATURES)} features")

    missing = [f for f in ALL_FEATURES if f not in feat_df.columns]
    if missing:
        print(f"  ERROR: Missing features: {missing}")
        return

    print(f"\n{sep}\n  Step 3: Train model on full labeled data\n{sep}")
    model = train_on_full_data(features_csv, args.recall_boost)

    print(f"\n{sep}\n  Step 4: Predict (threshold={args.threshold})\n{sep}")
    proba, pred = predict(model, feat_df, args.threshold)

    feat_df["probability"] = proba
    feat_df["predicted_label"] = pred
    feat_df["flagged_noisy"] = pred.astype(bool)

    # Summary
    n_noisy_pred = int(pred.sum())
    n_total = len(pred)
    print(f"  Total files     : {n_total}")
    print(f"  Flagged noisy   : {n_noisy_pred} ({n_noisy_pred/n_total*100:.1f}%)")
    print(f"  Flagged clean   : {n_total - n_noisy_pred} ({(n_total-n_noisy_pred)/n_total*100:.1f}%)")

    # Per-region breakdown
    print(f"\n  Breakdown by region:")
    print(f"  {'Region':<8} {'Total':>7} {'Noisy':>7} {'%Noisy':>8}")
    for region in ["HV1F", "HV1R", "HV2F", "HV3R"]:
        sub = feat_df[feat_df["region"] == region]
        if len(sub) == 0:
            continue
        n = len(sub)
        nn = int(sub["predicted_label"].sum())
        print(f"  {region:<8} {n:>7} {nn:>7} {nn/n*100:>7.1f}%")

    # Per-sample breakdown
    print(f"\n  Top 10 samples by max noisy probability:")
    sample_max = feat_df.groupby("sample_id")["probability"].max().sort_values(ascending=False)
    for sid, pmax in sample_max.head(10).items():
        sample_files = feat_df[feat_df["sample_id"] == sid]
        noisy_regions = sample_files[sample_files["flagged_noisy"]]["region"].tolist()
        print(f"  {sid:<25} max_prob={pmax:.3f}  noisy_regions={noisy_regions or 'none'}")

    # Save
    out_cols = ["file_path", "sample_id", "region", "probability", "predicted_label", "flagged_noisy"]
    out_path = test_dir / "predictions.csv"
    feat_df[out_cols].to_csv(out_path, index=False)
    print(f"\n  Predictions saved to {out_path}")
    print(f"{sep}\n  Done.\n{sep}")


if __name__ == "__main__":
    main()
