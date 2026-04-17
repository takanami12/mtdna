import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PEAK_FIELDS = ["peakA", "peakC", "peakG", "peakT"]


def load_peaks(json_path: str | Path) -> np.ndarray:
    """Return peak signals as float32 array of shape (4, L)."""
    with open(json_path) as f:
        data = json.load(f)
    return np.array([data[field] for field in PEAK_FIELDS], dtype=np.float32)


def normalize_peaks(peaks: np.ndarray) -> np.ndarray:
    """Scale each channel independently to [0, 1]."""
    maxv = peaks.max(axis=1, keepdims=True)
    maxv[maxv == 0] = 1.0
    return peaks / maxv


def pad_or_truncate(peaks: np.ndarray, max_len: int) -> np.ndarray:
    """Truncate or zero-pad along the length axis to reach max_len."""
    L = peaks.shape[1]
    if L >= max_len:
        return peaks[:, :max_len]
    pad = np.zeros((4, max_len - L), dtype=np.float32)
    return np.concatenate([peaks, pad], axis=1)


def preprocess(json_path: str | Path, max_len: int = 2000) -> np.ndarray:
    """Full preprocessing pipeline: load → normalize → pad/truncate."""
    peaks = load_peaks(json_path)
    peaks = normalize_peaks(peaks)
    peaks = pad_or_truncate(peaks, max_len)
    return peaks  # shape: (4, max_len)


class PeakDataset(Dataset):
    """Dataset that reads peak signals from JSON files listed in a CSV.

    CSV must have columns:
        FilePath  — path to the JSON file (absolute or relative to root)
        Label     — 0 (clean) or 1 (noisy)

    Args:
        csv_path: path to the labels CSV file.
        max_len:  fixed signal length after pad/truncate.
        root:     base directory for resolving relative FilePaths.
                  Defaults to the directory containing the CSV file.
    """

    def __init__(self, csv_path: str | Path, max_len: int = 2000,
                 root: str | Path | None = None):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.root = Path(root) if root else Path(csv_path).resolve().parent

        valid_mask = self.df["FilePath"].apply(self._is_valid)
        n_dropped = int((~valid_mask).sum())
        if n_dropped:
            print(f"[PeakDataset] Warning: skipping {n_dropped} missing/empty files "
                  f"({len(self.df) - n_dropped} remain)")
        self.df = self.df[valid_mask].reset_index(drop=True)

    def _resolve(self, filepath: str) -> Path:
        p = Path(filepath)
        if not p.is_absolute():
            return self.root / p
        if p.exists():
            return p
        # Absolute path from a different machine: strip leading components
        # until the remainder resolves under self.root.
        for i in range(1, len(p.parts)):
            candidate = self.root.joinpath(*p.parts[i:])
            if candidate.exists():
                return candidate
        return p  # will raise FileNotFoundError at load time

    def _is_valid(self, filepath: str) -> bool:
        p = self._resolve(filepath)
        return p.exists() and p.stat().st_size > 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        json_path = self._resolve(row["FilePath"])
        x = preprocess(json_path, self.max_len)
        x = torch.from_numpy(x)                        # (4, max_len)
        y = torch.tensor(int(row["Label"]), dtype=torch.long)
        return x, y
