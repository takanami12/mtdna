"""
src/plot_seg.py
===============
Visualize windowed-QC pseudo-labels and (optionally) U-Net predictions
for one trace file.

Usage:
    python src/plot_seg.py path/to/trace.json
    python src/plot_seg.py path/to/trace.json --checkpoint checkpoints/seg_best.pth
    python src/plot_seg.py path/to/trace.json --out plots/seg_example.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))

from utils.windowed_qc import compute_noisy_mask


def _load_raw(json_path: Path) -> dict:
    import json
    with open(json_path) as f:
        return json.load(f)


def plot_trace_segmentation(
    json_path:   str | Path,
    checkpoint:  str | Path | None = None,
    out_path:    str | Path         = "plots/seg_example.png",
    target_len:  int                = 4096,
) -> None:
    json_path = Path(json_path)
    out_path  = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = _load_raw(json_path)

    # Raw signal
    peaks = np.array(
        [d[k] for k in ("peakA", "peakC", "peakG", "peakT")], dtype=np.float32
    )
    total = peaks.sum(axis=0)
    L     = peaks.shape[1]
    x     = np.arange(L)

    bp  = np.array(d.get("basecallPos",  []), dtype=int)
    ql  = np.array(d.get("basecallQual", []), dtype=np.float32)

    # Windowed-QC pseudo-labels
    result = compute_noisy_mask(json_path)
    wq_mask = result.mask if result else np.zeros(L, dtype=np.uint8)

    # Optional: U-Net predictions
    unet_prob = None
    if checkpoint is not None:
        try:
            import torch
            from utils.seg_dataset import _load_and_build_channels
            from models.segnet import UNet1D, N_CHANNELS, TARGET_LEN

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model  = UNet1D(n_channels=N_CHANNELS).to(device)
            model.load_state_dict(
                torch.load(checkpoint, map_location=device)
            )
            model.eval()

            Xy = _load_and_build_channels(json_path, TARGET_LEN)
            if Xy is not None:
                X_t = torch.from_numpy(Xy[0]).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = model(X_t)[0, 0].cpu().numpy()
                # resample back to original L if needed
                if len(prob) != L:
                    from scipy.signal import resample
                    prob = resample(prob, L).astype(np.float32)
                unet_prob = prob
        except Exception as e:
            print(f"  Warning: could not load U-Net checkpoint: {e}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_rows  = 3 if unet_prob is None else 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 4 * n_rows), sharex=True)
    fig.suptitle(json_path.name, fontsize=11, y=1.01)

    chan_colors = {"A": "#2196F3", "C": "#4CAF50", "G": "#9C27B0", "T": "#F44336"}
    labels_ch   = ["A", "C", "G", "T"]

    # Row 0: Raw signal channels
    ax = axes[0]
    for i, (ch, col) in enumerate(zip(labels_ch, chan_colors.values())):
        ax.plot(x, peaks[i], color=col, lw=0.6, alpha=0.8, label=ch)
    ax.set_ylabel("Signal intensity")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.set_title("Raw peak channels (A/C/G/T)")

    # Shade noisy regions on all axes
    def shade_mask(ax_, mask, color, alpha=0.18):
        in_noisy = False
        start_x  = 0
        for i_, v in enumerate(mask):
            if v and not in_noisy:
                start_x  = i_
                in_noisy = True
            elif not v and in_noisy:
                ax_.axvspan(start_x, i_, color=color, alpha=alpha)
                in_noisy = False
        if in_noisy:
            ax_.axvspan(start_x, len(mask), color=color, alpha=alpha)

    shade_mask(axes[0], wq_mask, "red")

    # Row 1: Total signal + basecall quality
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(x, total, color="#607D8B", lw=0.7, label="Total signal")
    if len(bp) > 0 and len(ql) == len(bp):
        ax2.bar(bp, ql, width=3, color="#FF9800", alpha=0.6, label="Qual")
        ax2.set_ylabel("Basecall quality", color="#FF9800", fontsize=8)
    ax.set_ylabel("Total signal")
    ax.set_title("Total signal + basecall quality")
    shade_mask(ax, wq_mask, "red")

    # Row 2: Windowed-QC pseudo-label mask + per-window scores
    ax = axes[2]
    ax.fill_between(x, wq_mask, color="red", alpha=0.4, step="post", label="Noisy mask")
    if result is not None and len(result.windows) > 0:
        wdf = result.windows
        win_x    = (wdf["start"].values + wdf["end"].values) / 2
        ax.plot(win_x, wdf["frac_low_pr"].values, "b.", ms=3, alpha=0.6, label="frac_low_pr")
        ax.plot(win_x, np.clip(wdf["qual_mean"].values / 60, 0, 1), "g.", ms=3, alpha=0.6, label="qual/60")
    ax.set_ylabel("Value (0-1)")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.set_title("Windowed-QC pseudo-label mask")

    # Row 3 (optional): U-Net probability
    if unet_prob is not None:
        ax = axes[3]
        ax.plot(x, unet_prob, color="#E91E63", lw=0.8, label="U-Net prob")
        ax.axhline(0.5, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.fill_between(x, unet_prob, 0, where=unet_prob >= 0.5,
                        color="#E91E63", alpha=0.3, label="Predicted noisy")
        shade_mask(ax, wq_mask, "orange", alpha=0.2)
        ax.set_ylabel("Noisy probability")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("U-Net segmentation prediction")

    axes[-1].set_xlabel("Scan point")

    # Legend for shading
    patch = mpatches.Patch(color="red", alpha=0.3, label="Noisy region (windowed QC)")
    fig.legend(handles=[patch], loc="lower center", fontsize=8, ncol=1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    if result:
        print(f"  Noisy fraction : {result.noisy_fraction:.3f}")
        print(f"  Noisy windows  : {result.n_noisy_windows} / {len(result.windows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to Tracy JSON trace file")
    parser.add_argument("--checkpoint", default=None, help="U-Net checkpoint .pth")
    parser.add_argument("--out", default="plots/seg_example.png")
    parser.add_argument("--target-len", type=int, default=4096)
    args = parser.parse_args()

    plot_trace_segmentation(
        args.json_path,
        checkpoint = args.checkpoint,
        out_path   = args.out,
        target_len = args.target_len,
    )


if __name__ == "__main__":
    main()
