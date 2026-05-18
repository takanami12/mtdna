"""
src/utils/windowed_qc.py
========================
Rule-based windowed QC → per-scan-point pseudo-labels for segmentation.

A window is "noisy" if ANY condition holds:
  - frac_low_pr  > THRESH_LOW_PR  (>40% basecalls have poor peak ratio)
  - qual_mean    < THRESH_QUAL    (mean Phred < 15)
  - snr          < THRESH_SNR     (signal-to-noise < 2.5)
  - drift_ratio  > THRESH_DRIFT   (local drift / global drift > 2.5x)

Window labels are back-projected to scan points via overlap-average voting.

Usage:
    from src.utils.windowed_qc import compute_noisy_mask
    result = compute_noisy_mask("path/to/trace.json")
    mask = result.mask          # np.ndarray shape (L,), dtype uint8
    df   = result.windows       # per-window metrics DataFrame

    python src/utils/windowed_qc.py path/to/trace.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# Window geometry
DEFAULT_WIN_PTS  = 200   # ~60-70 bp at typical trace speed
DEFAULT_STEP_PTS = 100   # 50% overlap

# Peak ratio threshold for "ambiguous" basecall
_LOW_PR_THRESH = 3.0

# Noisy-window thresholds
THRESH_LOW_PR  = 0.40   # >40% of basecalls have peak_ratio < 3
THRESH_QUAL    = 15.0   # mean Phred quality
# Signal prominence: std(window) / global_mean.
# For clean Sanger: peaks cause high std → high prominence (0.3-0.8).
# For flat/blank signal: std≈0 → low prominence (<0.10) → noisy/empty.
THRESH_PROM    = 0.10   # flag window if signal is essentially flat
THRESH_DRIFT   = 2.5    # local drift / global drift


class MaskResult(NamedTuple):
    mask:            np.ndarray    # (L,) uint8, per scan point
    windows:         pd.DataFrame  # per-window metrics + noisy flag
    noisy_fraction:  float         # fraction of scan points flagged
    n_noisy_windows: int


def compute_noisy_mask(
    json_path: str | Path,
    win_pts:       int   = DEFAULT_WIN_PTS,
    step_pts:      int   = DEFAULT_STEP_PTS,
    low_pr_thresh: float = _LOW_PR_THRESH,
    thresh_low_pr: float = THRESH_LOW_PR,
    thresh_qual:   float = THRESH_QUAL,
    thresh_prom:   float = THRESH_PROM,
    thresh_drift:  float = THRESH_DRIFT,
) -> MaskResult | None:
    """
    Compute per-scan-point noisy mask from a Tracy JSON file.
    Returns None if file missing or unreadable.
    """
    path = Path(json_path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None

    peaks = np.array(
        [d[k] for k in ("peakA", "peakC", "peakG", "peakT")], dtype=np.float32
    )
    total = peaks.sum(axis=0)
    L = peaks.shape[1]

    bp = np.array(d.get("basecallPos",  []), dtype=int)
    ql = np.array(d.get("basecallQual", []), dtype=np.float32)

    # Peak ratio at each basecall position
    if len(bp) > 0 and bp.max() < L:
        pk_at = peaks[:, bp]
        srt   = np.sort(pk_at, axis=0)[::-1]
        denom = np.where(srt[1] < 1.0, 1.0, srt[1]).astype(np.float64)
        pr    = (srt[0].astype(np.float64) / denom).astype(np.float32)
    else:
        bp = np.array([], dtype=int)
        ql = np.array([], dtype=np.float32)
        pr = np.array([], dtype=np.float32)

    # Global reference values
    global_mean = float(total.mean()) or 1.0
    glob_means  = [
        total[i : i + win_pts].mean()
        for i in range(0, L - win_pts, step_pts)
    ]
    global_drift = max(float(np.std(glob_means)), 1.0) if glob_means else 1.0

    rows: list[dict] = []
    for start in range(0, L - win_pts + 1, step_pts):
        end = start + win_pts

        in_win = (bp >= start) & (bp < end)
        pr_win = pr[in_win]
        ql_win = ql[in_win]

        frac_low = float((pr_win < low_pr_thresh).mean()) if len(pr_win) > 0 else 0.5
        q_mean   = float(ql_win.mean())                   if len(ql_win)  > 0 else 5.0

        sig_win = total[start:end]

        # Signal prominence: std(window) / global_mean.
        # High → sharp Sanger peaks (clean). Low → flat/missing signal (noisy).
        prominence = float(sig_win.std()) / global_mean

        # Local drift: std of sub-window means within this window
        sub_w     = max(win_pts // 4, 1)
        sub_step  = max(sub_w // 2, 1)
        sub_means = [
            sig_win[i : i + sub_w].mean()
            for i in range(0, win_pts - sub_w + 1, sub_step)
        ]
        local_drift = float(np.std(sub_means)) if len(sub_means) > 1 else 0.0
        drift_ratio = local_drift / global_drift

        noisy = int(
            (frac_low   > thresh_low_pr) or
            (q_mean     < thresh_qual)   or
            (prominence < thresh_prom)   or
            (drift_ratio > thresh_drift)
        )

        rows.append({
            "start":       start,
            "end":         end,
            "frac_low_pr": round(frac_low,    4),
            "qual_mean":   round(q_mean,      3),
            "prominence":  round(prominence,  4),
            "drift_ratio": round(drift_ratio, 4),
            "noisy":       noisy,
        })

    if not rows:
        return MaskResult(np.zeros(L, dtype=np.uint8), pd.DataFrame(), 0.0, 0)

    win_df = pd.DataFrame(rows)

    # Back-project window labels → per scan-point via overlap voting
    accum = np.zeros(L, dtype=np.float32)
    count = np.zeros(L, dtype=np.float32)
    for row in win_df.itertuples(index=False):
        accum[row.start : row.end] += row.noisy
        count[row.start : row.end] += 1.0
    count = np.where(count == 0, 1.0, count)
    smooth = accum / count
    mask = (smooth >= 0.5).astype(np.uint8)

    return MaskResult(
        mask            = mask,
        windows         = win_df,
        noisy_fraction  = float(mask.mean()),
        n_noisy_windows = int(win_df["noisy"].sum()),
    )


def batch_compute_masks(
    json_paths: list[str | Path],
    **kwargs,
) -> pd.DataFrame:
    """Summary DataFrame for a list of trace files."""
    records = []
    for p in json_paths:
        r = compute_noisy_mask(p, **kwargs)
        records.append({
            "file_path":       str(p),
            "noisy_fraction":  r.noisy_fraction  if r else None,
            "n_noisy_windows": r.n_noisy_windows if r else None,
            "n_total_windows": len(r.windows)    if r else None,
            "trace_len":       len(r.mask)        if r else None,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/utils/windowed_qc.py <trace.json>")
        sys.exit(1)

    result = compute_noisy_mask(sys.argv[1])
    if result is None:
        print("ERROR: cannot parse file")
        sys.exit(1)

    print(f"Trace length   : {len(result.mask):,} scan points")
    print(f"Noisy fraction : {result.noisy_fraction:.3f}")
    print(f"Noisy windows  : {result.n_noisy_windows} / {len(result.windows)}")
    print("\nWindow metrics:")
    print(result.windows.to_string(index=False))
