"""Train two 1D-CNN models for mtDNA Sanger noise detection.

Model HV1   : trained on HV1F + HV1R files
Model HV2_3 : trained on HV2F + HV3R files

Usage:
    python train.py [--csv labels.csv] [--epochs 50] [--batch-size 32]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.model import CNN1D
from src.utils.seq import PeakDataset

# ── helpers ──────────────────────────────────────────────────────────────────

def get_hv_region(filepath: str) -> str | None:
    m = re.search(r"HV\d[FR]", Path(filepath).name)
    return m.group() if m else None


def build_loaders(
    csv_path: str,
    regions: list[str],
    max_len: int,
    batch_size: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 4,
    data_root: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """Return (train_loader, val_loader, test_loader, class_weights)."""
    df = pd.read_csv(csv_path)
    region_set = set(regions)
    mask = df["FilePath"].apply(lambda p: get_hv_region(p) in region_set).values
    valid_idx = np.where(mask)[0]
    labels = df["Label"].values[valid_idx]

    print(f"  Regions {regions}: {len(valid_idx)} files  "
          f"(label 0: {(labels==0).sum()}, label 1: {(labels==1).sum()})")

    # stratified split
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        valid_idx, labels,
        test_size=val_ratio + test_ratio,
        stratify=labels,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=y_temp,
        random_state=seed,
    )

    dataset = PeakDataset(csv_path, max_len=max_len, root=data_root)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True, **loader_kw)
    val_loader   = DataLoader(Subset(dataset, val_idx),   shuffle=False, **loader_kw)
    test_loader  = DataLoader(Subset(dataset, test_idx),  shuffle=False, **loader_kw)

    # inverse-frequency weights to handle class imbalance
    counts = np.bincount(y_train, minlength=2).astype(float)
    class_weights = torch.tensor(len(y_train) / (2 * counts), dtype=torch.float32)

    return train_loader, val_loader, test_loader, class_weights


# ── train / eval loops ───────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc=f"  Epoch {epoch:03d} [train]", leave=False, unit="batch")
    for x, y in bar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "  [val]",
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for x, y in bar:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")
    return total_loss / total, correct / total


# ── full training pipeline for one model ─────────────────────────────────────

def train_model(
    model_name: str,
    regions: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    print(f"\n{'='*60}")
    print(f"  Model : {model_name}")
    print(f"  Regions : {regions}")
    print("=" * 60)

    train_loader, val_loader, test_loader, class_weights = build_loaders(
        args.csv, regions, args.max_len, args.batch_size,
        num_workers=args.num_workers, data_root=args.data_root,
    )

    model = CNN1D(in_channels=4, num_classes=2, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"{model_name}_best.pth"

    best_val_loss = float("inf")
    patience_counter = 0

    epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"  {model_name}", unit="epoch")
    for epoch in epoch_bar:
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device, f"  Epoch {epoch:03d} [val] ")
        scheduler.step(vl_loss)

        improved = vl_loss < best_val_loss
        if improved:
            best_val_loss = vl_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        mark = "✓" if improved else " "
        epoch_bar.write(
            f"{mark} Epoch {epoch:03d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {vl_loss:.4f} acc {vl_acc:.4f}"
        )

        if patience_counter >= args.patience:
            epoch_bar.write(f"  Early stopping at epoch {epoch}.")
            break

    # final evaluation on held-out test set
    model.load_state_dict(torch.load(save_path, map_location=device))
    ts_loss, ts_acc = evaluate(model, test_loader, criterion, device, "  [test] ")
    print(f"\n  [{model_name}] Test — loss: {ts_loss:.4f}  acc: {ts_acc:.4f}")
    print(f"  Best checkpoint: {save_path}")
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train mtDNA noise-detection CNNs")
    p.add_argument("--csv",         default="labels.csv",   help="Path to labels CSV")
    p.add_argument("--max-len",     type=int, default=2000,  help="Fixed signal length (pad/truncate)")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--dropout",     type=float, default=0.5)
    p.add_argument("--patience",    type=int, default=10,    help="Early-stopping patience (epochs)")
    p.add_argument("--output-dir",  default="checkpoints",   help="Directory for saved .pth files")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-root",   default=None,
                   help="Root directory for resolving relative FilePaths in CSV. "
                        "Defaults to the directory containing the CSV file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_model("model_HV1",   ["HV1F", "HV1R"], args, device)
    train_model("model_HV2_3", ["HV2F", "HV3R"], args, device)


if __name__ == "__main__":
    main()
