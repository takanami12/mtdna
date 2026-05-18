"""
src/train.py
============
Train and evaluate ML models for noisy-trace classification.

Usage:
    python src/train.py [--features features.csv] [--root /mnt/d/mtdna]
    python src/train.py --beta 2.0                  # F2-optimized threshold (default)
    python src/train.py --min-recall 0.95           # hard floor on recall
    python src/train.py --no-shap --no-ablation     # faster run

Recall-maximization strategy
-----------------------------
In forensic mtDNA analysis, missing a noisy trace (false negative) is more
dangerous than over-flagging a clean trace (false positive). Therefore:

  1. scale_pos_weight is boosted above the class-imbalance ratio so the model
     assigns higher raw probabilities to the noisy class.
  2. Classification threshold is tuned on out-of-fold (OOF) probabilities using
     F_beta score (beta=2 → recall weighted 2x more than precision).
  3. If --min-recall is set, the threshold is the lowest value that achieves
     that recall floor, regardless of F_beta.

OOF threshold tuning is leak-free: each sample's probability comes from a model
that never saw that sample during training (GroupKFold grouped by SampleID).

Data leakage prevention
-----------------------
Each sample produces up to 4 JSON files (HV1F/R, HV2F, HV3R). Files from the
same sample must stay in the same CV fold.
=> GroupKFold(n_splits=5) grouped by SampleID.
=> Anti-leakage assertion runs each fold.
=> Threshold search runs on OOF probabilities, not on test-fold probabilities.

Biological rationale for each feature group
--------------------------------------------
A. Peak quality (qual_mean, qual_p10, pct_qual_lt*, pr_*, frac_low_pr)
   Primary/secondary peak ratio at each basecall position directly measures
   how much competing-channel signal leaks in — the physical signature of
   mid-trace noise in degraded DNA (dye blobs, baseline drift, polyC stutter).

B. Alignment scores (align1/2/3score, align_diff_12, align_ratio_12)
   Tracy aligns trace to rCRS under three models: primary allele, secondary
   allele, and heterozygous. In noisy traces, the secondary-allele model (align2)
   fits poorly because noise creates random spurious secondary signals rather
   than a real alternate allele. align2score is the single strongest predictor.

C. Variant burden (n_variants, n_fail_variants, v_qual_min, hetindel)
   HV1/HV2/HV3 regions have limited true polymorphisms vs rCRS. Excess called
   variants are noise artifacts; a noisy trace inflates n_variants.

D. Signal shape (baseline_drift, dyeblob_ratio, n_bases, trace_len)
   Baseline drift captures systematic signal instability. Dye-blob ratio flags
   the known 60-140bp artifact region common in degraded/short-fragment DNA.

Models compared
---------------
  LogisticRegression : linear baseline (requires StandardScaler)
  RandomForest       : non-linear, robust, feature importance
  GradientBoosting   : sklearn GBM, strong on small tabular
  XGBoostClassifier  : best-in-class tabular, handles imbalance (chosen final)
  LGBMClassifier     : fast, accurate alternative to XGBoost

Metrics (reported at tuned threshold unless noted)
----------------------------------------------------
  ROC-AUC            : threshold-independent ranking quality
  PR-AUC             : precision-recall area, suitable for imbalanced classes
  Recall (noisy)     : primary target metric — fraction of noisy traces caught
  Precision (noisy)  : fraction of flagged traces that are truly noisy
  F_beta (noisy)     : F score with configurable beta (default beta=2)
  F1-macro           : balanced F1 for both classes
"""

from __future__ import annotations

import argparse
import copy
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import shap
except ImportError:
    shap = None
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Feature groups ─────────────────────────────────────────────────────────────
FEATURE_GROUPS = {
    "A_peak_quality": [
        "qual_mean", "qual_p10", "pct_qual_lt20", "pct_qual_lt10",
        "pr_p10", "pr_p25", "pr_mean", "frac_low_pr",
    ],
    "B_alignment": [
        "align1score", "align2score", "align3score",
        "align_diff_12", "align_ratio_12",
    ],
    "C_variants": [
        "n_variants", "n_fail_variants", "v_qual_min", "hetindel",
    ],
    "D_signal_shape": [
        "baseline_drift", "dyeblob_ratio", "n_bases", "trace_len",
    ],
    "E_simplex_entropy": [
        "entropy_mean", "entropy_p90", "entropy_frac_high",
    ],
}
ALL_FEATURES = [f for feats in FEATURE_GROUPS.values() for f in feats]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(
    features_csv: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load feature matrix, labels, and sample groups.

    Returns (X, y, groups) where groups is a Series of SampleID strings.
    """
    fdf = pd.read_csv(features_csv, index_col=0)
    X = fdf[ALL_FEATURES].copy()
    y = fdf["label"].copy()

    def _sid(fp: str) -> str:
        m = re.search(
            r"_(\d{6}|LN_\d+_[A-Z]{2}\d+|MS_\d+_[A-Z]{2}\d+|AB_\d+_[A-Z]{2}\d+)"
            r"_(HV\d[FR])_\d+\.json$",
            Path(fp).name,
            re.IGNORECASE,
        )
        return m.group(1) if m else fp

    groups = pd.Series([_sid(fp) for fp in fdf.index], index=fdf.index, name="sample_id")
    print(f"Dataset: {len(X)} files | {int(y.sum())} noisy ({y.mean()*100:.1f}%) | {groups.nunique()} samples")
    return X, y, groups


# ── Model factory ─────────────────────────────────────────────────────────────

def make_models(n_noisy: int, n_clean: int, recall_boost: float = 1.5) -> dict:
    """Instantiate all models.

    recall_boost multiplies scale_pos_weight above the imbalance ratio so the
    model is biased toward flagging noisy traces (increases recall at the cost
    of precision). The threshold tuning step compensates for any over-flagging.
    """
    base_pos_weight = n_clean / max(n_noisy, 1)
    pos_weight = base_pos_weight * recall_boost

    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight={0: 1, 1: pos_weight},
                max_iter=1000, C=1.0, solver="lbfgs",
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight={0: 1, 1: pos_weight},
            max_features="sqrt", random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss", random_state=42, verbosity=0,
        ),
        "LightGBM": lgbm.LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, random_state=42, verbose=-1,
        ),
    }


# ── Threshold tuning ──────────────────────────────────────────────────────────

def tune_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    beta: float = 2.0,
    min_recall: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Find optimal classification threshold on OOF probabilities.

    If min_recall is set: lowest threshold achieving >= min_recall.
    Otherwise: threshold maximizing F_beta.

    Returns (threshold, metrics_at_threshold).
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thresh = 0.5
    best_score = -1.0

    for t in thresholds:
        pred = (proba >= t).astype(int)
        r = recall_score(y_true, pred, pos_label=1, zero_division=0)
        p = precision_score(y_true, pred, pos_label=1, zero_division=0)

        if min_recall is not None:
            if r >= min_recall and t < best_thresh:
                best_thresh = t
                best_score = r
        else:
            score = fbeta_score(y_true, pred, beta=beta, pos_label=1, zero_division=0)
            if score > best_score:
                best_score = score
                best_thresh = t

    pred_best = (proba >= best_thresh).astype(int)
    metrics = {
        "threshold": best_thresh,
        "recall_noisy": recall_score(y_true, pred_best, pos_label=1, zero_division=0),
        "precision_noisy": precision_score(y_true, pred_best, pos_label=1, zero_division=0),
        "f1_noisy": f1_score(y_true, pred_best, pos_label=1, zero_division=0),
        f"f{beta}_noisy": fbeta_score(y_true, pred_best, beta=beta, pos_label=1, zero_division=0),
        "recall_clean": recall_score(y_true, pred_best, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }
    return best_thresh, metrics


def pr_curve_table(
    y_true: np.ndarray, proba: np.ndarray, beta: float = 2.0
) -> pd.DataFrame:
    """Precision-recall table at selected thresholds."""
    rows = []
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, pred, pos_label=1, zero_division=0),
            "recall": recall_score(y_true, pred, pos_label=1, zero_division=0),
            "f1": f1_score(y_true, pred, pos_label=1, zero_division=0),
            f"f{beta}": fbeta_score(y_true, pred, beta=beta, pos_label=1, zero_division=0),
        })
    return pd.DataFrame(rows)


# ── CV with OOF collection ────────────────────────────────────────────────────

def cross_validate_oof(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_name: str,
    n_noisy: int,
    n_clean: int,
    recall_boost: float = 1.5,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """GroupKFold CV for one model; returns (oof_proba, oof_y, fold_metrics).

    OOF probabilities are collected without applying any threshold.
    Threshold tuning happens after all folds complete (no leakage).
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof_proba = np.zeros(len(y))
    oof_y = np.zeros(len(y), dtype=int)
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        # Leak check
        overlap = set(groups.iloc[tr_idx]) & set(groups.iloc[te_idx])
        assert not overlap, f"Fold {fold}: {len(overlap)} sample(s) in both splits!"

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        models = make_models(n_noisy, n_clean, recall_boost)
        m = copy.deepcopy(models[model_name])
        m.fit(X_tr, y_tr)

        if hasattr(m, "predict_proba"):
            proba_te = m.predict_proba(X_te)[:, 1]
        else:
            proba_te = m.decision_function(X_te)

        oof_proba[te_idx] = proba_te
        oof_y[te_idx] = y_te.values

        fold_metrics.append({
            "fold": fold,
            "roc_auc": roc_auc_score(y_te, proba_te),
            "pr_auc": average_precision_score(y_te, proba_te),
        })
        print(f"    Fold {fold} | AUC={fold_metrics[-1]['roc_auc']:.3f} PR={fold_metrics[-1]['pr_auc']:.3f}")

    return oof_proba, oof_y, fold_metrics


def cross_validate_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    recall_boost: float = 1.5,
    beta: float = 2.0,
    min_recall: float | None = None,
) -> pd.DataFrame:
    """Run GroupKFold CV for all models; tune threshold per model on OOF probas."""
    n_noisy, n_clean = int(y.sum()), int((y == 0).sum())
    model_names = ["LogReg", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM"]
    records = []

    for name in model_names:
        print(f"\n  [{name}]")
        oof_proba, oof_y, fold_metrics = cross_validate_oof(
            X, y, groups, name, n_noisy, n_clean, recall_boost, n_splits
        )
        thresh, metrics = tune_threshold(oof_y, oof_proba, beta=beta, min_recall=min_recall)
        records.append({
            "model": name,
            "roc_auc_mean": np.mean([f["roc_auc"] for f in fold_metrics]),
            "roc_auc_std": np.std([f["roc_auc"] for f in fold_metrics]),
            "pr_auc_mean": np.mean([f["pr_auc"] for f in fold_metrics]),
            "threshold": thresh,
            **{k: v for k, v in metrics.items() if k != "threshold"},
        })

    return pd.DataFrame(records).set_index("model")


# ── Final model ───────────────────────────────────────────────────────────────

def train_final_model(
    X: pd.DataFrame, y: pd.Series, recall_boost: float = 1.5
) -> xgb.XGBClassifier:
    """Train XGBoost on full dataset with recall-boosted class weight."""
    n_noisy, n_clean = int(y.sum()), int((y == 0).sum())
    model = copy.deepcopy(make_models(n_noisy, n_clean, recall_boost)["XGBoost"])
    model.fit(X, y)
    return model


# ── Feature importance ────────────────────────────────────────────────────────

def feature_importance_report(model: xgb.XGBClassifier) -> pd.Series:
    imp = model.get_booster().get_score(importance_type="gain")
    return pd.Series(imp).sort_values(ascending=False)


def shap_analysis(
    model: xgb.XGBClassifier, X: pd.DataFrame, n_samples: int = 500
) -> pd.Series:
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X.iloc[idx])
    return pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns).sort_values(ascending=False)


# ── Ablation ──────────────────────────────────────────────────────────────────

def ablation_study(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series,
    recall_boost: float = 1.5, n_splits: int = 5,
) -> pd.DataFrame:
    n_noisy, n_clean = int(y.sum()), int((y == 0).sum())
    gkf = GroupKFold(n_splits=n_splits)

    def _run(X_sub: pd.DataFrame) -> tuple[float, float]:
        aucs = []
        for tr_idx, te_idx in gkf.split(X_sub, y, groups):
            m = copy.deepcopy(make_models(n_noisy, n_clean, recall_boost)["XGBoost"])
            m.fit(X_sub.iloc[tr_idx], y.iloc[tr_idx])
            proba = m.predict_proba(X_sub.iloc[te_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[te_idx], proba))
        return float(np.mean(aucs)), float(np.std(aucs))

    results = []
    base_mean, base_std = _run(X)
    results.append({"removed_group": "none (baseline)", "roc_auc_mean": base_mean, "roc_auc_std": base_std})

    for group_name, feats in FEATURE_GROUPS.items():
        remaining = [f for f in ALL_FEATURES if f not in feats]
        mean, std = _run(X[remaining])
        results.append({"removed_group": group_name, "roc_auc_mean": mean, "roc_auc_std": std})

    df = pd.DataFrame(results)
    df["auc_drop"] = base_mean - df["roc_auc_mean"]
    return df.sort_values("auc_drop", ascending=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train noisy-trace classifier (recall-focused)")
    parser.add_argument("--features", default="features.csv")
    parser.add_argument("--root", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--beta", type=float, default=2.0,
                        help="F_beta for threshold tuning. beta>1 favors recall (default: 2.0)")
    parser.add_argument("--min-recall", type=float, default=None,
                        help="Hard floor on recall. Overrides --beta if set.")
    parser.add_argument("--recall-boost", type=float, default=1.5,
                        help="Multiplier on scale_pos_weight (default: 1.5)")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()

    features_path = Path(args.features)
    beta = args.beta
    min_recall = args.min_recall

    # 1. Load data
    sep = "=" * 65
    print(f"\n{sep}\n  Step 1: Load data\n{sep}")
    X, y, groups = load_data(features_path)

    if min_recall:
        print(f"  Mode: min-recall >= {min_recall}")
    else:
        print(f"  Mode: maximize F{beta} (recall weighted {beta}x over precision)")
    print(f"  recall_boost = {args.recall_boost}x  (scale_pos_weight boosted above imbalance ratio)")

    # 2. Cross-validation
    print(f"\n{sep}\n  Step 2: {args.folds}-fold GroupKFold CV — all models\n{sep}")
    cv_summary = cross_validate_all_models(
        X, y, groups,
        n_splits=args.folds,
        recall_boost=args.recall_boost,
        beta=beta,
        min_recall=min_recall,
    )

    print(f"\n{'-'*65}")
    print("  CV Summary (threshold tuned on OOF probabilities)")
    fbeta_key = f"f{beta}_noisy"
    print(f"  {'Model':<20} {'ROC-AUC':>10} {'PR-AUC':>8} {'Threshold':>10} {'Recall':>8} {'Precision':>10} {'F'+str(beta):>8}")
    print(f"  {'-'*80}")
    for model_name, row in cv_summary.sort_values("roc_auc_mean", ascending=False).iterrows():
        print(
            f"  {model_name:<20} "
            f"{row['roc_auc_mean']:>6.3f}±{row['roc_auc_std']:.3f}  "
            f"{row['pr_auc_mean']:>6.3f}  "
            f"{row['threshold']:>10.2f}  "
            f"{row['recall_noisy']:>7.3f}  "
            f"{row['precision_noisy']:>9.3f}  "
            f"{row.get(fbeta_key, float('nan')):>7.3f}"
        )

    cv_summary.to_csv(features_path.parent / "cv_results.csv")
    print(f"\n  Saved to cv_results.csv")

    # 3. Best threshold from XGBoost OOF
    xgb_row = cv_summary.loc["XGBoost"]
    best_threshold = xgb_row["threshold"]
    print(f"\n  >> XGBoost OOF threshold = {best_threshold:.2f} (F{beta}-optimized)")
    print(f"     Recall(noisy)={xgb_row['recall_noisy']:.3f}  Precision(noisy)={xgb_row['precision_noisy']:.3f}")
    print(f"     Recall(clean)={xgb_row['recall_clean']:.3f}")

    # 4. PR curve table for XGBoost OOF
    print(f"\n{sep}\n  Step 3: Precision-Recall curve (XGBoost OOF)\n{sep}")
    # Re-run XGBoost OOF to get probabilities for PR table
    n_noisy, n_clean = int(y.sum()), int((y == 0).sum())
    oof_proba, oof_y, _ = cross_validate_oof(
        X, y, groups, "XGBoost", n_noisy, n_clean, args.recall_boost, args.folds
    )
    pr_table = pr_curve_table(oof_y, oof_proba, beta=beta)
    print(f"\n  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>6} {'F'+str(beta):>8}")
    print(f"  {'-'*45}")
    for _, r in pr_table.iterrows():
        fbeta_col = f"f{beta}"
        marker = " <-- chosen" if abs(r["threshold"] - best_threshold) < 0.06 else ""
        print(
            f"  {r['threshold']:>10.2f} {r['precision']:>10.3f} {r['recall']:>8.3f} "
            f"{r['f1']:>6.3f} {r[fbeta_col]:>8.3f}{marker}"
        )

    # 5. Final model + feature importance
    print(f"\n{sep}\n  Step 4: Final XGBoost (full dataset)\n{sep}")
    final_model = train_final_model(X, y, recall_boost=args.recall_boost)

    imp = feature_importance_report(final_model)
    print(f"\n  Feature importance (gain):")
    for feat, val in imp.items():
        bar = "█" * int(val / imp.iloc[0] * 30)
        print(f"  {feat:<25} {val:>9.1f}  {bar}")

    # 6. SHAP
    if not args.no_shap and shap is not None:
        print(f"\n{sep}\n  Step 5: SHAP attribution (n=500)\n{sep}")
        shap_imp = shap_analysis(final_model, X)
        for feat, val in shap_imp.items():
            bar = "█" * int(val / shap_imp.iloc[0] * 30)
            print(f"  {feat:<25} {val:>8.4f}  {bar}")

    # 7. Ablation
    if not args.no_ablation:
        print(f"\n{sep}\n  Step 6: Ablation (XGBoost, feature group removal)\n{sep}")
        abl = ablation_study(X, y, groups, args.recall_boost, args.folds)
        for _, row in abl.iterrows():
            drop_str = f"Δ={row['auc_drop']:+.4f}" if row["removed_group"] != "none (baseline)" else "(baseline)"
            print(f"  {row['removed_group']:<35} {row['roc_auc_mean']:.3f}±{row['roc_auc_std']:.3f}  {drop_str}")

    print(f"\n{sep}\n  Done. Use threshold={best_threshold:.2f} in production.\n{sep}")


if __name__ == "__main__":
    main()
