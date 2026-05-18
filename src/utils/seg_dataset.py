"""
src/utils/seg_dataset.py
========================
PyTorch Dataset for 1D U-Net segmentation training.

Each sample returns:
  X : FloatTensor (N_CHANNELS, TARGET_LEN)
  y : FloatTensor (1, TARGET_LEN)   — windowed-QC pseudo-label mask

Input channels (20 total):
   0-3  : raw peaks A/C/G/T  (normalized per-channel to [0,1])
   4-7  : 1st derivative of peaks
   8-11 : 2nd derivative of peaks
   12   : total signal (normalized)
   13   : max channel (normalized)
   14   : peak ratio at basecall positions (0 elsewhere)
   15   : basecall position mask (binary)
   16   : basecall quality (normalized to [0,1] via /60)
   17   : rolling SNR (window=50 pts, normalized)
   18   : rolling baseline (window=200 pts, normalized)
   19   : position index (0 → 1 left to right)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.windowed_qc import compute_noisy_mask
from models.segnet import N_CHANNELS, TARGET_LEN


def _load_and_build_channels(
    json_path: str | Path,
    target_len: int = TARGET_LEN,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load Tracy JSON, build (N_CHANNELS, target_len) feature array and
    (1, target_len) pseudo-label mask.

    Returns None on failure.
    """
    path = Path(json_path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None

    mask_result = compute_noisy_mask(path)
    if mask_result is None:
        return None

    # ── Raw peaks ─────────────────────────────────────────────────────────────
    peaks = np.array(
        [d[k] for k in ("peakA", "peakC", "peakG", "peakT")], dtype=np.float32
    )
    L_orig = peaks.shape[1]

    # Resample everything to target_len
    if L_orig != target_len:
        from scipy.signal import resample
        peaks_r = resample(peaks, target_len, axis=1).astype(np.float32)
        mask_r  = resample(mask_result.mask.astype(np.float32), target_len)
        mask_r  = (mask_r >= 0.5).astype(np.float32)
    else:
        peaks_r = peaks.copy()
        mask_r  = mask_result.mask.astype(np.float32)

    # Per-channel normalization
    for i in range(4):
        mx = peaks_r[i].max()
        if mx > 0:
            peaks_r[i] /= mx

    # ── Derivatives ──────────────────────────────────────────────────────────
    d1 = np.gradient(peaks_r, axis=1).astype(np.float32)
    d2 = np.gradient(d1,      axis=1).astype(np.float32)

    # ── Aggregate channels ────────────────────────────────────────────────────
    total   = peaks_r.sum(axis=0)
    max_ch  = peaks_r.max(axis=0)
    if total.max() > 0:
        total = total / total.max()
    if max_ch.max() > 0:
        max_ch = max_ch / max_ch.max()

    # ── Basecall-position features ────────────────────────────────────────────
    bp_orig = np.array(d.get("basecallPos",  []), dtype=int)
    ql_orig = np.array(d.get("basecallQual", []), dtype=np.float32)

    scale = target_len / L_orig if L_orig > 0 else 1.0
    bp    = np.clip((bp_orig * scale).astype(int), 0, target_len - 1)

    bc_mask = np.zeros(target_len, dtype=np.float32)
    bc_qual = np.zeros(target_len, dtype=np.float32)
    pr_arr  = np.zeros(target_len, dtype=np.float32)

    if len(bp) > 0:
        bc_mask[bp] = 1.0
        if len(ql_orig) == len(bp):
            bc_qual[bp] = np.clip(ql_orig / 60.0, 0.0, 1.0)

        pk_at = peaks_r[:, bp]
        srt   = np.sort(pk_at, axis=0)[::-1]
        denom = np.where(srt[1] < 0.01, 0.01, srt[1]).astype(np.float64)
        ratio = np.clip(srt[0].astype(np.float64) / denom, 0, 10).astype(np.float32) / 10.0
        pr_arr[bp] = ratio

    # ── Rolling statistics ────────────────────────────────────────────────────
    from scipy.ndimage import uniform_filter1d
    sig_1d = peaks_r.sum(axis=0)   # unnormalized total for rolling stats

    roll_mean = uniform_filter1d(sig_1d, size=50).astype(np.float32)
    roll_std  = np.sqrt(
        uniform_filter1d((sig_1d - roll_mean) ** 2, size=50) + 1e-6
    ).astype(np.float32)
    roll_snr  = roll_mean / roll_std
    if roll_snr.max() > 0:
        roll_snr = roll_snr / roll_snr.max()

    roll_base = uniform_filter1d(sig_1d, size=200).astype(np.float32)
    if roll_base.max() > 0:
        roll_base = roll_base / roll_base.max()

    pos_idx = np.linspace(0.0, 1.0, target_len, dtype=np.float32)

    # ── Stack channels ────────────────────────────────────────────────────────
    X = np.stack([
        *peaks_r,           # 0-3
        *d1,                # 4-7
        *d2,                # 8-11
        total,              # 12
        max_ch,             # 13
        pr_arr,             # 14
        bc_mask,            # 15
        bc_qual,            # 16
        roll_snr,           # 17
        roll_base,          # 18
        pos_idx,            # 19
    ], axis=0).astype(np.float32)   # (20, target_len)

    assert X.shape == (N_CHANNELS, target_len), f"Channel shape mismatch: {X.shape}"

    y = mask_r.reshape(1, target_len)   # (1, target_len)
    return X, y


class TraceSegDataset(Dataset):
    """
    Dataset of Tracy JSON traces with windowed-QC pseudo-labels.

    Parameters
    ----------
    json_paths  : list of paths to Tracy JSON files
    target_len  : fixed length to resample traces to (default 4096)
    augment     : if True, apply random flip augmentation
    """

    def __init__(
        self,
        json_paths: list[str | Path],
        target_len: int  = TARGET_LEN,
        augment:    bool = False,
    ):
        self.target_len = target_len
        self.augment    = augment
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        skipped = 0
        for p in json_paths:
            result = _load_and_build_channels(p, target_len)
            if result is None:
                skipped += 1
                continue
            self.samples.append(result)

        if skipped:
            print(f"  Warning: {skipped}/{len(json_paths)} files skipped")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X, y = self.samples[idx]
        if self.augment and np.random.rand() < 0.5:
            X = X[:, ::-1].copy()
            y = y[:, ::-1].copy()
        return torch.from_numpy(X), torch.from_numpy(y)


def build_dataset_from_labels_csv(
    csv_path: str | Path,
    root: str | Path | None = None,
    **kwargs,
) -> TraceSegDataset:
    """
    Build dataset from a labels CSV (FilePath, Label columns).
    Labels are ignored — windowed QC provides pseudo-labels.
    """
    import pandas as pd
    csv_path = Path(csv_path)
    root     = Path(root) if root else csv_path.parent
    df = pd.read_csv(csv_path)
    paths = [
        (root / row["FilePath"]) if not Path(row["FilePath"]).is_absolute()
        else Path(row["FilePath"])
        for _, row in df.iterrows()
    ]
    return TraceSegDataset(paths, **kwargs)
