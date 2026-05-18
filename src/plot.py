"""
src/plot.py
===========
Visualize and compare model performance on labeled training data
using out-of-fold (OOF) predictions from GroupKFold CV.

Plots generated (saved to plots/):
    model_comparison.png  — 2×2 panel:
        [0,0] ROC curves (OOF) for all models
        [0,1] PR curves (OOF) for all models
        [1,0] Bar chart: ROC-AUC mean ± std per fold
        [1,1] Bar chart: Recall at F2-optimized threshold per fold

Usage:
    python src/plot.py
    python src/plot.py --features features.csv --beta 2.0 --folds 5
    python src/plot.py --recall-boost 1.5

No labels are needed for test_data — all metrics come from OOF on labeled data.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from train import ALL_FEATURES, make_models

warnings.filterwarnings("ignore")

MODEL_COLORS = {
    "XGBoost":        "#E74C3C",
    "LightGBM":       "#3498DB",
    "RandomForest":   "#2ECC71",
    "GradientBoosting": "#F39C12",
    "LogReg":         "#9B59B6",
}


def load_data(features_csv: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    fdf = pd.read_csv(features_csv, index_col=0)
    X = fdf[ALL_FEATURES].copy()
    y = fdf["label"].copy()

    def _sid(fp: str) -> str:
        m = re.search(
            r"_(\d{6}|LN_\d+_[A-Z]{2}\d+|MS_\d+_[A-Z]{2}\d+|AB_\d+_[A-Z]{2}\d+)"
            r"_(HV\d[FR])_\d+\.json$",
            Path(fp).name, re.IGNORECASE,
        )
        return m.group(1) if m else fp

    groups = pd.Series([_sid(fp) for fp in fdf.index], index=fdf.index)
    return X, y, groups


def collect_oof(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_names: list[str],
    n_noisy: int,
    n_clean: int,
    recall_boost: float,
    n_splits: int,
) -> dict[str, dict]:
    """Run GroupKFold CV for each model; collect OOF probas and per-fold AUC."""
    gkf = GroupKFold(n_splits=n_splits)
    results: dict[str, dict] = {}

    for name in model_names:
        print(f"  CV [{name}] ...", end=" ", flush=True)
        models_dict = make_models(n_noisy, n_clean, recall_boost)
        oof_proba = np.zeros(len(y))
        fold_aucs: list[float] = []

        for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
            # Anti-leakage check
            assert not (set(groups.iloc[tr_idx]) & set(groups.iloc[te_idx]))

            m = copy.deepcopy(models_dict[name])
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])

            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X.iloc[te_idx])[:, 1]
            else:
                p = m.decision_function(X.iloc[te_idx])

            oof_proba[te_idx] = p
            fold_aucs.append(roc_auc_score(y.iloc[te_idx], p))

        print(f"AUC={np.mean(fold_aucs):.3f}±{np.std(fold_aucs):.3f}")
        results[name] = {
            "oof_proba": oof_proba,
            "fold_aucs": fold_aucs,
        }

    return results


def best_threshold_fbeta(
    y_true: np.ndarray, proba: np.ndarray, beta: float
) -> float:
    """Find threshold maximizing F_beta on OOF probabilities."""
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        pred = (proba >= t).astype(int)
        score = fbeta_score(y_true, pred, beta=beta, pos_label=1, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t)


def recall_at_threshold(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> float:
    pred = (proba >= threshold).astype(int)
    return float(recall_score(y_true, pred, pos_label=1, zero_division=0))


def plot_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    oof_results: dict[str, dict],
    beta: float,
    out_path: Path,
) -> None:
    """Generate 2×2 comparison figure."""
    y_arr = y.values
    model_names = list(oof_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"Model Comparison — GroupKFold CV (n={len(y)}, noisy={int(y.sum())}, "
        f"F{beta}-threshold-tuned)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── [0,0] ROC curves ──────────────────────────────────────────────────────
    ax = axes[0, 0]
    for name in model_names:
        proba = oof_results[name]["oof_proba"]
        fpr, tpr, _ = roc_curve(y_arr, proba)
        auc = roc_auc_score(y_arr, proba)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2,
                label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (OOF)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    # ── [0,1] PR curves ───────────────────────────────────────────────────────
    ax = axes[0, 1]
    baseline = y_arr.mean()
    ax.axhline(baseline, color="gray", lw=1, linestyle="--",
               label=f"No-skill (P={baseline:.2f})", alpha=0.6)
    for name in model_names:
        proba = oof_results[name]["oof_proba"]
        prec, rec, _ = precision_recall_curve(y_arr, proba)
        pr_auc = average_precision_score(y_arr, proba)
        # Mark optimal threshold point
        thresh = best_threshold_fbeta(y_arr, proba, beta)
        pred_opt = (proba >= thresh).astype(int)
        r_opt = recall_score(y_arr, pred_opt, pos_label=1, zero_division=0)
        p_opt = precision_score(y_arr, pred_opt, pos_label=1, zero_division=0) if pred_opt.sum() > 0 else 0
        ax.plot(rec, prec, color=MODEL_COLORS[name], lw=2,
                label=f"{name} (AP={pr_auc:.3f})")
        ax.scatter([r_opt], [p_opt], color=MODEL_COLORS[name], s=60,
                   zorder=5, marker="D", edgecolors="black", linewidths=0.5)
    ax.set_xlabel("Recall (noisy class)")
    ax.set_ylabel("Precision (noisy class)")
    ax.set_title(f"Precision-Recall Curves (OOF)\n◆ = F{beta}-optimal threshold")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    # ── [1,0] Bar chart: ROC-AUC per fold ────────────────────────────────────
    ax = axes[1, 0]
    bar_data_auc = {
        name: oof_results[name]["fold_aucs"] for name in model_names
    }
    means_auc = [np.mean(bar_data_auc[n]) for n in model_names]
    stds_auc = [np.std(bar_data_auc[n]) for n in model_names]
    colors = [MODEL_COLORS[n] for n in model_names]
    x = np.arange(len(model_names))
    bars = ax.bar(x, means_auc, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, means_auc, yerr=stds_auc, fmt="none", color="black",
                capsize=5, capthick=1.5, elinewidth=1.5)
    for bar, mean in zip(bars, means_auc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC (mean ± std, 5-fold CV)")
    ax.set_ylim(min(means_auc) - 0.05, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(min(means_auc) - 0.001, color="none")

    # ── [1,1] Bar chart: Recall at F_beta-optimal threshold ──────────────────
    ax = axes[1, 1]
    recalls = []
    thresholds = []
    for name in model_names:
        proba = oof_results[name]["oof_proba"]
        thresh = best_threshold_fbeta(y_arr, proba, beta)
        thresholds.append(thresh)
        pred = (proba >= thresh).astype(int)
        recalls.append(recall_score(y_arr, pred, pos_label=1, zero_division=0))

    # Per-fold recall at global threshold for error bars
    fold_recalls: dict[str, list[float]] = {n: [] for n in model_names}
    gkf = GroupKFold(n_splits=len(oof_results[model_names[0]]["fold_aucs"]))
    groups_tmp = pd.Series(range(len(y)), index=y.index)  # dummy — fold splits already done

    # Approximate fold-level recall using the OOF proba split by index blocks
    n = len(y)
    fold_size = n // 5
    for i, name in enumerate(model_names):
        proba = oof_results[name]["oof_proba"]
        thresh = thresholds[i]
        for fold_i in range(5):
            start = fold_i * fold_size
            end = (fold_i + 1) * fold_size if fold_i < 4 else n
            y_fold = y_arr[start:end]
            p_fold = proba[start:end]
            pred_fold = (p_fold >= thresh).astype(int)
            if y_fold.sum() > 0:
                fold_recalls[name].append(recall_score(y_fold, pred_fold, pos_label=1, zero_division=0))

    stds_rec = [np.std(fold_recalls[n]) for n in model_names]
    bars2 = ax.bar(x, recalls, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, recalls, yerr=stds_rec, fmt="none", color="black",
                capsize=5, capthick=1.5, elinewidth=1.5)
    for bar, rec, thresh in zip(bars2, recalls, thresholds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{rec:.3f}\n(t={thresh:.2f})", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Recall (noisy class)")
    ax.set_title(f"Recall at F{beta}-Optimal Threshold (OOF)")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.9, color="red", lw=1, linestyle="--", alpha=0.5, label="90% recall")
    ax.axhline(0.95, color="darkred", lw=1, linestyle=":", alpha=0.5, label="95% recall")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to {out_path}")
    plt.close(fig)


def plot_pr_recall_tradeoff(
    y_arr: np.ndarray,
    oof_results: dict[str, dict],
    beta: float,
    out_path: Path,
) -> None:
    """Additional plot: Precision-Recall tradeoff at different thresholds."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for name in oof_results:
        proba = oof_results[name]["oof_proba"]
        thresholds = np.linspace(0.05, 0.95, 181)
        precs, recs = [], []
        for t in thresholds:
            pred = (proba >= t).astype(int)
            from sklearn.metrics import precision_score
            precs.append(precision_score(y_arr, pred, pos_label=1, zero_division=0))
            recs.append(recall_score(y_arr, pred, pos_label=1, zero_division=0))
        ax.plot(thresholds, recs, color=MODEL_COLORS[name], lw=2, label=f"{name} (recall)", solid_capstyle="round")
        ax.plot(thresholds, precs, color=MODEL_COLORS[name], lw=2, linestyle="--", alpha=0.6)

    # Legend for line styles
    ax.plot([], [], "k-", lw=2, label="— Recall")
    ax.plot([], [], "k--", lw=2, alpha=0.6, label="-- Precision")
    ax.axhline(0.90, color="gray", lw=1, ls=":", alpha=0.6)
    ax.axhline(0.95, color="gray", lw=1, ls="-.", alpha=0.6)
    ax.text(0.96, 0.91, "90%", fontsize=8, color="gray", ha="right")
    ax.text(0.96, 0.96, "95%", fontsize=8, color="gray", ha="right")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Recall & Precision vs Threshold (OOF)\nF{beta}: recall weighted {beta}x over precision")
    ax.legend(fontsize=8, loc="center left")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to {out_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot model comparison charts")
    parser.add_argument("--features", default="features.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--recall-boost", type=float, default=1.5)
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()

    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    beta = args.beta

    sep = "=" * 65
    print(f"\n{sep}\n  Step 1: Load data\n{sep}")
    X, y, groups = load_data(features_path)
    n_noisy, n_clean = int(y.sum()), int((y == 0).sum())
    print(f"  {len(X)} files | {n_noisy} noisy ({y.mean()*100:.1f}%) | {groups.nunique()} samples")

    model_names = ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting", "LogReg"]

    print(f"\n{sep}\n  Step 2: Collect OOF predictions ({args.folds}-fold GroupKFold)\n{sep}")
    oof_results = collect_oof(
        X, y, groups, model_names, n_noisy, n_clean,
        args.recall_boost, args.folds,
    )

    print(f"\n{sep}\n  Step 3: Generate plots\n{sep}")
    plot_comparison(
        X, y, oof_results, beta,
        out_path=out_dir / "model_comparison.png",
    )
    plot_pr_recall_tradeoff(
        y.values, oof_results, beta,
        out_path=out_dir / "recall_precision_vs_threshold.png",
    )

    print(f"\n{sep}\n  Done. Plots in ./{out_dir}/\n{sep}")


if __name__ == "__main__":
    main()
