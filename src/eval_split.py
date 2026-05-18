"""
src/eval_split.py
=================
80/20 sample-level train/test split → evaluate all models → show misclassifications.

Usage:
    python src/eval_split.py --features features.csv
    python src/eval_split.py --features features.csv --test-size 0.20 --beta 2.0
    python src/eval_split.py --features features.csv --seed 42

Output:
    eval_split_predictions.csv  — all test files with true label, per-model predictions
    eval_split_errors.csv       — only misclassified rows (any model wrong)
    eval_split_summary.csv      — per-model metrics on test set
"""

from __future__ import annotations

import argparse
import copy
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))

from train import ALL_FEATURES, make_models, tune_threshold


SAMPLE_RE = re.compile(
    r"_(\d{6}|LN_\d+_[A-Z]{2}\d+|MS_\d+_[A-Z]{2}\d+|AB_\d+_[A-Z]{2}\d+)"
    r"_(HV\d[FR])_\d+\.json$",
    re.IGNORECASE,
)


def extract_sid(fp: str) -> str:
    m = SAMPLE_RE.search(Path(fp).name)
    return m.group(1) if m else fp


def extract_region(fp: str) -> str:
    m = SAMPLE_RE.search(Path(fp).name)
    return m.group(2).upper() if m else "?"


def load_data(features_csv: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    fdf = pd.read_csv(features_csv, index_col=0)
    X = fdf[ALL_FEATURES].copy()
    y = fdf["label"].copy()
    groups = pd.Series([extract_sid(fp) for fp in fdf.index], index=fdf.index, name="sample_id")
    return X, y, groups


def split_groups(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.Index, pd.Index]:
    """GroupShuffleSplit — all files of one sample stay in same split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    return X.index[tr_idx], X.index[te_idx]


def metrics_at_threshold(
    y_true: np.ndarray, proba: np.ndarray, threshold: float, beta: float
) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "recall_noisy": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "precision_noisy": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "f1_noisy": f1_score(y_true, pred, pos_label=1, zero_division=0),
        f"f{beta}_noisy": fbeta_score(y_true, pred, beta=beta, pos_label=1, zero_division=0),
        "recall_clean": recall_score(y_true, pred, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="80/20 sample-level split evaluation")
    parser.add_argument("--features", default="features.csv")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--recall-boost", type=float, default=1.5)
    args = parser.parse_args()

    features_csv = Path(args.features)
    beta = args.beta
    sep = "=" * 65

    print(f"\n{sep}\n  Step 1: Load data\n{sep}")
    X, y, groups = load_data(features_csv)
    n_noisy = int(y.sum())
    n_clean = int((y == 0).sum())
    print(f"  Total : {len(X)} files | {n_noisy} noisy ({y.mean()*100:.1f}%) | {groups.nunique()} samples")

    print(f"\n{sep}\n  Step 2: GroupShuffleSplit 80/20 by SampleID\n{sep}")
    train_idx, test_idx = split_groups(X, y, groups, args.test_size, args.seed)

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    g_train = groups.loc[train_idx]

    # Leak check
    train_samples = set(groups.loc[train_idx])
    test_samples = set(groups.loc[test_idx])
    overlap = train_samples & test_samples
    assert not overlap, f"DATA LEAKAGE: {len(overlap)} samples appear in both splits!"

    print(f"  Train : {len(X_train)} files | {int(y_train.sum())} noisy | {g_train.nunique()} samples")
    print(f"  Test  : {len(X_test)} files  | {int(y_test.sum())} noisy | {test_samples.__len__()} samples")
    print(f"  Overlap check: PASSED (0 samples shared)")

    print(f"\n{sep}\n  Step 3: Train all models on train split\n{sep}")
    model_names = ["LogReg", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]
    n_noisy_tr = int(y_train.sum())
    n_clean_tr = int((y_train == 0).sum())
    models_dict = make_models(n_noisy_tr, n_clean_tr, args.recall_boost)

    trained = {}
    for name in model_names:
        m = copy.deepcopy(models_dict[name])
        m.fit(X_train, y_train)
        trained[name] = m
        print(f"  {name} trained.")

    print(f"\n{sep}\n  Step 4: Predict on test split + tune threshold on train OOF\n{sep}")

    # Build result dataframe
    result = pd.DataFrame({
        "file_path": X_test.index,
        "sample_id": groups.loc[test_idx].values,
        "region": [extract_region(fp) for fp in X_test.index],
        "true_label": y_test.values,
    })

    summary_rows = []
    for name in model_names:
        m = trained[name]
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(X_test)[:, 1]
        else:
            proba = m.decision_function(X_test)

        # Tune threshold on train set (self-predict — optimistic but avoids test leakage)
        proba_train = (
            m.predict_proba(X_train)[:, 1]
            if hasattr(m, "predict_proba")
            else m.decision_function(X_train)
        )
        thresh, _ = tune_threshold(y_train.values, proba_train, beta=beta)

        pred = (proba >= thresh).astype(int)
        result[f"prob_{name}"] = proba.round(4)
        result[f"pred_{name}"] = pred

        met = metrics_at_threshold(y_test.values, proba, thresh, beta)
        summary_rows.append({"model": name, **met})

        print(
            f"  {name:<20} thresh={thresh:.2f}  "
            f"recall={met['recall_noisy']:.3f}  prec={met['precision_noisy']:.3f}  "
            f"f{beta}={met[f'f{beta}_noisy']:.3f}  AUC={met['roc_auc']:.3f}"
        )

    # Error analysis
    result["n_models_correct"] = sum(
        (result[f"pred_{n}"] == result["true_label"]).astype(int)
        for n in model_names
    )
    result["n_models_wrong"] = len(model_names) - result["n_models_correct"]
    result["any_wrong"] = result["n_models_wrong"] > 0
    result["all_wrong"] = result["n_models_wrong"] == len(model_names)

    print(f"\n{sep}\n  Step 5: Error summary\n{sep}")
    print(f"  Test files total        : {len(result)}")
    print(f"  Files with any error    : {result['any_wrong'].sum()} ({result['any_wrong'].mean()*100:.1f}%)")
    print(f"  Files with all wrong    : {result['all_wrong'].sum()}")

    print(f"\n  Error breakdown by region:")
    for region in ["HV1F", "HV1R", "HV2F", "HV3R"]:
        sub = result[result["region"] == region]
        if len(sub) == 0:
            continue
        err = sub["any_wrong"].sum()
        print(f"  {region:<6} {len(sub):>4} total, {err:>3} errors ({err/len(sub)*100:.1f}%)")

    print(f"\n  Per-model false negatives (noisy predicted as clean):")
    for name in model_names:
        fn = ((result["true_label"] == 1) & (result[f"pred_{name}"] == 0)).sum()
        fp = ((result["true_label"] == 0) & (result[f"pred_{name}"] == 1)).sum()
        print(f"  {name:<20} FN={fn:>3}  FP={fp:>3}")

    # Save outputs
    out_dir = features_csv.parent
    all_path = out_dir / "eval_split_predictions.csv"
    err_path = out_dir / "eval_split_errors.csv"
    sum_path = out_dir / "eval_split_summary.csv"

    result.to_csv(all_path, index=False)

    errors = result[result["any_wrong"]].copy()
    errors = errors.sort_values("n_models_wrong", ascending=False)
    errors.to_csv(err_path, index=False)

    summary_df = pd.DataFrame(summary_rows).set_index("model")
    summary_df.to_csv(sum_path)

    print(f"\n  Saved:")
    print(f"    {all_path}  ({len(result)} rows)")
    print(f"    {err_path}  ({len(errors)} error rows)")
    print(f"    {sum_path}")
    print(f"\n{sep}\n  Done.\n{sep}")


if __name__ == "__main__":
    main()
