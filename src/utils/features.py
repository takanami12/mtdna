"""
src/utils/features.py
=====================
Extract interpretable features from Tracy-processed JSON trace files.

Features fall into four groups:
  A. Peak quality     — signal clarity at basecall positions
  B. Alignment        — align2/align3 scores (secondary allele fit)
  C. Variant burden   — number and quality of called variants
  D. Signal shape     — baseline drift, dye-blob region

Correlation with noisy label (full dataset, N=2599):
  align2score      -0.53  *** strongest
  align3score      -0.49
  frac_low_pr      +0.45
  pr_mean          -0.44
  n_variants       +0.36
  qual_mean        -0.36
  align_diff_12    +0.32

Usage:
  from src.utils.features import extract_features, build_feature_matrix
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Threshold for "ambiguous" peak: primary/secondary < this value
_LOW_PR_THRESH = 3.0
# Scan points considered dye-blob region (empirically ~60-140 bp → first ~800 pts)
_DYEBLOB_SCAN_PTS = 800
# Window size for baseline drift computation
_DRIFT_WINDOW = 200


def extract_features(json_path: str | Path) -> dict[str, float] | None:
    """Extract all features from one Tracy JSON file.

    Returns None if the file is missing or unreadable.

    Feature groups
    --------------
    A. Peak quality (at basecall positions)
        qual_mean         : mean Phred-like quality across all basecalls
        qual_p10          : 10th percentile quality
        pct_qual_lt20     : fraction of bases with qual < 20
        pct_qual_lt10     : fraction of bases with qual < 10
        pr_p10            : 10th percentile of primary/secondary peak ratio
        pr_p25            : 25th percentile of peak ratio
        pr_mean           : mean peak ratio
        frac_low_pr       : fraction of basecalls with peak ratio < 3

    B. Alignment scores
        align1score       : raw align1 score (primary allele vs ref)
        align2score       : raw align2 score (secondary allele vs ref) — best single predictor
        align3score       : raw align3 score (heterozygous model)
        align_diff_12     : align1 - align2 (large gap → noisy)
        align_ratio_12    : align1 / align2 (same, ratio form)

    C. Variant burden
        n_variants        : total called variants
        n_fail_variants   : variants with filter != PASS
        v_qual_min        : minimum variant quality score
        hetindel          : binary — het indel detected (0/1)

    D. Signal shape
        baseline_drift    : std of windowed total-signal mean (drift proxy)
        dyeblob_ratio     : mean signal in first 800 scan pts / global mean
        n_bases           : number of basecalls (sequence length)
        trace_len         : raw scan-point length
    """
    path = Path(json_path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None

    # ── Raw signal ───────────────────────────────────────────────────────────
    peaks = np.array([d[k] for k in ("peakA", "peakC", "peakG", "peakT")], dtype=np.float32)
    total_signal = peaks.sum(axis=0)
    L = peaks.shape[1]

    # ── A. Peak quality ───────────────────────────────────────────────────────
    qual = np.array(d.get("basecallQual", []), dtype=np.float32)
    bp = np.array(d.get("basecallPos", []), dtype=int)

    qual_mean = float(qual.mean()) if len(qual) else 0.0
    qual_p10 = float(np.percentile(qual, 10)) if len(qual) else 0.0
    pct_qual_lt20 = float((qual < 20).mean()) if len(qual) else 0.0
    pct_qual_lt10 = float((qual < 10).mean()) if len(qual) else 0.0

    if len(bp) and len(bp) > 0 and bp.max() < L:
        peak_at_call = peaks[:, bp]                         # (4, n_bases)
        sorted_pk = np.sort(peak_at_call, axis=0)[::-1]    # descending along channel axis
        denom = sorted_pk[1].astype(np.float64)
        denom[denom < 1.0] = 1.0
        peak_ratio = sorted_pk[0].astype(np.float64) / denom

        pr_p10 = float(np.percentile(peak_ratio, 10))
        pr_p25 = float(np.percentile(peak_ratio, 25))
        pr_mean = float(peak_ratio.mean())
        frac_low_pr = float((peak_ratio < _LOW_PR_THRESH).mean())
    else:
        pr_p10 = pr_p25 = pr_mean = frac_low_pr = 0.0

    # ── B. Alignment scores ────────────────────────────────────────────────────
    a1 = float(d.get("align1score") or 0)
    a2 = float(d.get("align2score") or 0)
    a3 = float(d.get("align3score") or 0)
    align_diff_12 = a1 - a2
    align_ratio_12 = a1 / max(a2, 1.0)

    # ── C. Variant burden ──────────────────────────────────────────────────────
    variants = d.get("variants", {})
    vrows = variants.get("rows", []) if isinstance(variants, dict) else []
    vcols = variants.get("columns", []) if isinstance(variants, dict) else []

    n_variants = len(vrows)

    filter_idx = vcols.index("filter") if "filter" in vcols else -1
    n_fail = (
        sum(1 for r in vrows if r[filter_idx] != "PASS")
        if filter_idx >= 0
        else 0
    )

    qual_idx = vcols.index("qual") if "qual" in vcols else -1
    if vrows and qual_idx >= 0:
        vq = [r[qual_idx] for r in vrows if isinstance(r[qual_idx], (int, float))]
        v_qual_min = float(min(vq)) if vq else 60.0
    else:
        v_qual_min = 60.0

    hetindel = int(bool(d.get("hetindel", 0)))

    # ── D. Signal shape ────────────────────────────────────────────────────────
    wins = [
        total_signal[i : i + _DRIFT_WINDOW]
        for i in range(0, L - _DRIFT_WINDOW, _DRIFT_WINDOW // 2)
    ]
    baseline_drift = float(np.std([w.mean() for w in wins])) if wins else 0.0

    dyeblob_pts = min(_DYEBLOB_SCAN_PTS, L)
    global_mean = float(total_signal.mean()) or 1.0
    dyeblob_ratio = float(total_signal[:dyeblob_pts].mean()) / global_mean

    return {
        # A
        "qual_mean": qual_mean,
        "qual_p10": qual_p10,
        "pct_qual_lt20": pct_qual_lt20,
        "pct_qual_lt10": pct_qual_lt10,
        "pr_p10": pr_p10,
        "pr_p25": pr_p25,
        "pr_mean": pr_mean,
        "frac_low_pr": frac_low_pr,
        # B
        "align1score": a1,
        "align2score": a2,
        "align3score": a3,
        "align_diff_12": align_diff_12,
        "align_ratio_12": align_ratio_12,
        # C
        "n_variants": float(n_variants),
        "n_fail_variants": float(n_fail),
        "v_qual_min": v_qual_min,
        "hetindel": float(hetindel),
        # D
        "baseline_drift": baseline_drift,
        "dyeblob_ratio": dyeblob_ratio,
        "n_bases": float(len(qual)),
        "trace_len": float(L),
    }


def build_feature_matrix(
    csv_path: str | Path,
    root: str | Path | None = None,
    *,
    drop_missing: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and label series y from a labels CSV.

    Parameters
    ----------
    csv_path : path to labels CSV with columns FilePath, Label
    root     : base directory for resolving relative FilePaths;
               defaults to directory containing the CSV
    drop_missing : if True, skip files that cannot be found/parsed

    Returns
    -------
    X : DataFrame of shape (n_samples, n_features)
    y : Series of integer labels (0=clean, 1=noisy)
    """
    csv_path = Path(csv_path)
    root = Path(root) if root else csv_path.resolve().parent

    df = pd.read_csv(csv_path)
    records: list[dict] = []
    labels: list[int] = []

    for _, row in df.iterrows():
        fp = Path(row["FilePath"])
        full = fp if fp.is_absolute() else root / fp
        feat = extract_features(full)
        if feat is None:
            if drop_missing:
                continue
            feat = {k: float("nan") for k in extract_features.__doc__.split() if False}
        feat["file_path"] = str(row["FilePath"])
        records.append(feat)
        labels.append(int(row["Label"]))

    X = pd.DataFrame(records).set_index("file_path")
    y = pd.Series(labels, index=X.index, name="label")
    return X, y


if __name__ == "__main__":
    import sys

    csv = sys.argv[1] if len(sys.argv) > 1 else "labels_v2.csv"
    root = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Building feature matrix from {csv} ...")
    X, y = build_feature_matrix(csv, root)
    print(f"Shape: {X.shape}  |  noisy={y.sum()}  clean={(y==0).sum()}")

    corr = X.corrwith(y).sort_values(key=abs, ascending=False)
    print("\nFeature correlation with label:")
    print(corr.to_string())

    out = X.copy()
    out["label"] = y
    out_path = Path(csv).with_name("features.csv")
    out.to_csv(out_path)
    print(f"\nSaved to {out_path}")
