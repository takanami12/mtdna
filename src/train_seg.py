"""
src/train_seg.py
================
Train the 1D U-Net segmentation model on Sanger trace files.

Pseudo-labels come from windowed QC (no manual per-position annotation needed).
GroupKFold by SampleID prevents data leakage between train/val.

Loss: Focal + Dice (handles extreme class imbalance — noisy regions are rare).

Usage:
    python src/train_seg.py
    python src/train_seg.py --labels labels_v2.csv --epochs 60 --batch 8
    python src/train_seg.py --labels labels_v2.csv --no-cv  # train on all data

Output:
    checkpoints/seg_best.pth   — best val-loss checkpoint
    checkpoints/seg_final.pth  — final epoch weights
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))

from utils.seg_dataset import TraceSegDataset, build_dataset_from_labels_csv
from models.segnet import UNet1D, N_CHANNELS, TARGET_LEN  # noqa: E402

SAMPLE_RE = re.compile(
    r"_(\d{6}|LN_\d+_[A-Z]{2}\d+|MS_\d+_[A-Z]{2}\d+|AB_\d+_[A-Z]{2}\d+)_",
    re.IGNORECASE,
)


# ── Losses ────────────────────────────────────────────────────────────────────

def focal_loss(
    pred: torch.Tensor, target: torch.Tensor,
    gamma: float = 2.0, alpha: float = 0.75,
) -> torch.Tensor:
    """Binary focal loss. alpha weights positive (noisy) class."""
    bce  = F.binary_cross_entropy(pred, target, reduction="none")
    pt   = torch.where(target == 1, pred, 1 - pred)
    w    = torch.where(target == 1, alpha * (1 - pt) ** gamma,
                       (1 - alpha) * (1 - pt) ** gamma)
    return (w * bce).mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_f   = pred.reshape(-1)
    target_f = target.reshape(-1)
    inter    = (pred_f * target_f).sum()
    return 1 - (2 * inter + eps) / (pred_f.sum() + target_f.sum() + eps)


def combined_loss(
    pred: torch.Tensor, target: torch.Tensor,
    focal_w: float = 0.6, dice_w: float = 0.4,
) -> torch.Tensor:
    return focal_w * focal_loss(pred, target) + dice_w * dice_loss(pred, target)


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    model: nn.Module, loader: DataLoader, device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    tp = fp = fn = total_loss = 0.0
    n_batches = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_loss += combined_loss(pred, y).item()
        n_batches  += 1
        pred_bin = (pred >= threshold).float()
        tp += (pred_bin * y).sum().item()
        fp += (pred_bin * (1 - y)).sum().item()
        fn += ((1 - pred_bin) * y).sum().item()

    prec   = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1     = 2 * prec * recall / max(prec + recall, 1e-8)
    return {
        "loss":    total_loss / max(n_batches, 1),
        "recall":  recall,
        "precision": prec,
        "f1":      f1,
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = combined_loss(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def train(
    dataset:    TraceSegDataset,
    groups:     np.ndarray,
    out_dir:    Path,
    n_folds:    int   = 5,
    epochs:     int   = 50,
    batch:      int   = 8,
    lr:         float = 1e-3,
    base_ch:    int   = 32,
    do_cv:      bool  = True,
    device_str: str   = "auto",
) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )
    print(f"  Device: {device}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not do_cv:
        # Train on full dataset, no validation
        loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=0)
        model  = UNet1D(n_channels=N_CHANNELS, base_ch=base_ch).to(device)
        opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for ep in range(1, epochs + 1):
            loss = train_one_epoch(model, loader, opt, device)
            sched.step()
            if ep % 10 == 0 or ep == epochs:
                print(f"  Epoch {ep:3d}/{epochs}  loss={loss:.4f}")

        ckpt = out_dir / "seg_final.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"  Saved: {ckpt}")
        return

    # GroupKFold cross-validation
    gkf     = GroupKFold(n_splits=n_folds)
    indices = np.arange(len(dataset))
    dummy_y = np.zeros(len(dataset))

    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(indices, dummy_y, groups), 1):
        print(f"\n  {'='*55}")
        print(f"  Fold {fold}/{n_folds}  |  train={len(tr_idx)}  val={len(va_idx)}")

        tr_loader = DataLoader(
            Subset(dataset, tr_idx), batch_size=batch, shuffle=True,  num_workers=0
        )
        va_loader = DataLoader(
            Subset(dataset, va_idx), batch_size=batch, shuffle=False, num_workers=0
        )

        model = UNet1D(n_channels=N_CHANNELS, base_ch=base_ch).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_loss = float("inf")
        best_path = out_dir / f"seg_fold{fold}_best.pth"

        for ep in range(1, epochs + 1):
            tr_loss = train_one_epoch(model, tr_loader, opt, device)
            sched.step()

            if ep % 10 == 0 or ep == epochs:
                vm = compute_metrics(model, va_loader, device)
                flag = ""
                if vm["loss"] < best_loss:
                    best_loss = vm["loss"]
                    torch.save(model.state_dict(), best_path)
                    flag = " ✓"
                print(
                    f"  ep {ep:3d}  tr_loss={tr_loss:.4f}  "
                    f"val_loss={vm['loss']:.4f}  "
                    f"recall={vm['recall']:.3f}  "
                    f"prec={vm['precision']:.3f}"
                    f"{flag}"
                )

        # Reload best and evaluate
        model.load_state_dict(torch.load(best_path, map_location=device))
        vm = compute_metrics(model, va_loader, device)
        print(
            f"\n  Fold {fold} best → "
            f"recall={vm['recall']:.3f}  prec={vm['precision']:.3f}  f1={vm['f1']:.3f}"
        )
        fold_metrics.append(vm)

    print(f"\n  {'='*55}")
    print("  Cross-validation summary:")
    for k in ["loss", "recall", "precision", "f1"]:
        vals = [m[k] for m in fold_metrics]
        print(f"    {k:<12}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels",   default="labels_v2.csv")
    parser.add_argument("--root",     default=None,
                        help="Base dir for resolving FilePaths in labels CSV")
    parser.add_argument("--out-dir",  default="checkpoints")
    parser.add_argument("--folds",    type=int,   default=5)
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--batch",    type=int,   default=8)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--base-ch",  type=int,   default=32)
    parser.add_argument("--no-cv",    action="store_true",
                        help="Train on full data without cross-validation")
    parser.add_argument("--device",   default="auto")
    args = parser.parse_args()

    sep = "=" * 65
    print(f"\n{sep}\n  Step 1: Build dataset\n{sep}")
    dataset = build_dataset_from_labels_csv(
        args.labels,
        root=args.root,
        augment=not args.no_cv,
    )
    print(f"  Loaded {len(dataset)} traces (target_len={TARGET_LEN})")

    if len(dataset) == 0:
        print("  ERROR: no traces loaded. Check --labels and --root paths.")
        return

    # Extract SampleID groups for GroupKFold
    labels_df = pd.read_csv(args.labels)
    root = Path(args.root) if args.root else Path(args.labels).parent

    # Match groups to successfully-loaded samples
    groups_all: list[str] = []
    loaded_idx = 0
    for _, row in labels_df.iterrows():
        fp = Path(row["FilePath"])
        full = fp if fp.is_absolute() else root / fp
        if not full.exists():
            continue
        m = SAMPLE_RE.search(str(fp))
        groups_all.append(m.group(1) if m else str(full))
        loaded_idx += 1
        if loaded_idx == len(dataset):
            break

    groups = np.array(groups_all)
    if len(groups) != len(dataset):
        print(f"  WARNING: group count ({len(groups)}) ≠ dataset size ({len(dataset)}). "
              "Using sequential IDs.")
        groups = np.arange(len(dataset)).astype(str)

    n_noisy = sum(1 for X, y in dataset if y.mean() > 0.05)
    print(f"  Traces with >5% noisy pts: {n_noisy} / {len(dataset)}")

    print(f"\n{sep}\n  Step 2: Train\n{sep}")
    train(
        dataset    = dataset,
        groups     = groups,
        out_dir    = Path(args.out_dir),
        n_folds    = args.folds,
        epochs     = args.epochs,
        batch      = args.batch,
        lr         = args.lr,
        base_ch    = args.base_ch,
        do_cv      = not args.no_cv,
        device_str = args.device,
    )


if __name__ == "__main__":
    main()
