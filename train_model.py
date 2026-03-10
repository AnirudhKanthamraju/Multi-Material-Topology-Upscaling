"""
train_model.py — Train a named FNO2d model for topology upscaling.

Dataset:
  All images in the input/output folders are loaded and split 80/20 (train/test).
  Validation uses the test-split inputs mapped against the 100x50 ground-truth folder.

Model naming:
  --model_name  gives this run a unique name. All outputs go to models/<model_name>/
  Architecture hyperparameters (modes1, modes2, width) are saved in config.json
  alongside the weights so the model can always be exactly reconstructed later.

Outputs per named model:
  models/<model_name>/
  ├── config.json
  ├── <model_name>_final.pth
  ├── training_curves.png
  └── weights/
      ├── <model_name>_epoch_0020.pth
      └── ...

Usage:
  python train_model.py \\
      --model_name   fno_base \\
      --data_in_dir  ./CNT_40x20/CNT_40x20 \\
      --data_out_dir ./CNT_80X40/CNT_80X40 \\
      --data_val_dir ./CNT_100x50/CNT_100x50 \\
      --modes1 12 --modes2 12 --width 32 \\
      --batch_size 8 --epochs 60 --save_every 20 \\
      --learning_rate 1e-3
"""

import os
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.fno import FNO2d


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _sorted_images(directory: str):
    """Return image filenames sorted numerically by stem."""
    exts = {".jpg", ".jpeg", ".png"}
    files = [
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts
    ]
    files.sort(key=lambda f: int(os.path.splitext(f)[0]))
    return files


class TopologyDataset(Dataset):
    """Pairs (low-res, target-res) topology images from two directories."""

    def __init__(self, in_dir: str, out_dir: str, indices: list):
        """
        Args:
            in_dir, out_dir : directories with numerically-named images.
            indices         : which sorted-file positions to include.
        """
        in_all  = _sorted_images(in_dir)
        out_all = _sorted_images(out_dir)

        if len(in_all) == 0:
            raise FileNotFoundError(f"No images in {in_dir}")

        self.in_paths  = [os.path.join(in_dir,  in_all[i])  for i in indices]
        self.out_paths = [os.path.join(out_dir, out_all[i]) for i in indices]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.in_paths)

    def __getitem__(self, idx):
        x = Image.open(self.in_paths[idx]).convert("RGB")
        y = Image.open(self.out_paths[idx]).convert("RGB")
        return (
            self.to_tensor(x),   # (3, H_in,  W_in)
            self.to_tensor(y),   # (3, H_out, W_out)
        )


def split_indices(n_total: int, train_ratio: float = 0.8):
    """Return (train_indices, test_indices) for an 80/20 split."""
    n_train = int(n_total * train_ratio)
    all_idx  = list(range(n_total))
    return all_idx[:n_train], all_idx[n_train:]


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(modes1: int, modes2: int, width: int, device: str) -> FNO2d:
    return FNO2d(modes1=modes1, modes2=modes2, width=width).to(device)


def save_config(model_dir: str, model_name: str, args):
    """Persist architecture hyperparameters to config.json."""
    cfg = {
        "model_name" : model_name,
        "modes1"     : args.modes1,
        "modes2"     : args.modes2,
        "width"      : args.width,
    }
    path = os.path.join(model_dir, "config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Config saved to {path}")
    return cfg


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def save_training_curves(train_losses, test_losses, val_losses, model_dir: str, model_name: str):
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Train Loss (40x20 -> 80x40)", color="#2c7bb6", linewidth=2)
    ax.plot(epochs, test_losses,  label="Test Loss  (40x20 -> 80x40)", color="#fdae61", linewidth=2, linestyle="-.")
    ax.plot(epochs, val_losses,   label="Val Loss   (40x20 -> 100x50)", color="#d7191c", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("MSE Loss", fontsize=13)
    ax.set_title(f"FNO Losses -- {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    path = os.path.join(model_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Training curves saved to {path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---- Discover total available samples ----
    all_in_files = _sorted_images(args.data_in_dir)
    n_total      = len(all_in_files)
    if n_total == 0:
        raise FileNotFoundError(f"No images found in {args.data_in_dir}")

    train_idx, test_idx = split_indices(n_total, train_ratio=0.8)
    print(f"[INFO] Dataset: {n_total} total — {len(train_idx)} train / {len(test_idx)} test (80/20)")

    # ---- Datasets ----
    train_dataset = TopologyDataset(args.data_in_dir, args.data_out_dir, train_idx)
    test_dataset  = TopologyDataset(args.data_in_dir, args.data_out_dir, test_idx)
    val_dataset   = TopologyDataset(args.data_in_dir, args.data_val_dir, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[INFO] Train loader: {len(train_loader)} batches | Test/Val loader: {len(val_loader)} batches")

    # ---- Model dir ----
    model_dir     = os.path.join("models", args.model_name)
    weights_dir   = os.path.join(model_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    save_config(model_dir, args.model_name, args)

    # ---- Build model (optionally load) ----
    model = build_model(args.modes1, args.modes2, args.width, device)

    load_path = args.load_model or os.path.join(model_dir, f"{args.model_name}_final.pth")
    if args.load_model and os.path.isfile(load_path):
        print(f"[INFO] Loading weights from {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
    else:
        print("[INFO] Initialising model with random weights.")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    train_losses, test_losses, val_losses = [], [], []

    print(f"\n[INFO] Training '{args.model_name}' for {args.epochs} epochs "
          f"(checkpoints every {args.save_every} epochs)\n")

    for epoch in range(1, args.epochs + 1):

        # Training
        model.train()
        running = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch, output_size=(y_batch.shape[-2], y_batch.shape[-1]))
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * x_batch.size(0)
        train_loss = running / len(train_dataset)

        # Testing & Validation
        model.eval()
        test_running = 0.0
        val_running = 0.0
        with torch.no_grad():
            for x_batch, y_test in test_loader:
                x_batch, y_test = x_batch.to(device), y_test.to(device)
                test_preds = model(x_batch, output_size=(y_test.shape[-2], y_test.shape[-1]))
                test_running += criterion(test_preds, y_test).item() * x_batch.size(0)

            for x_batch, y_val in val_loader:
                x_batch, y_val = x_batch.to(device), y_val.to(device)
                val_preds  = model(x_batch, output_size=(y_val.shape[-2], y_val.shape[-1]))
                val_running += criterion(val_preds, y_val).item() * x_batch.size(0)
        test_loss = test_running / len(test_dataset)
        val_loss = val_running / len(val_dataset)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch:4d}/{args.epochs}]  Train: {train_loss:.6f}  |  Test: {test_loss:.6f}  |  Val: {val_loss:.6f}")

        if epoch % args.save_every == 0:
            ckpt = os.path.join(weights_dir, f"{args.model_name}_epoch_{epoch:04d}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  -> Checkpoint: {ckpt}")

    final_path = os.path.join(model_dir, f"{args.model_name}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n[INFO] Final weights saved to {final_path}")

    save_training_curves(train_losses, test_losses, val_losses, model_dir, args.model_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a named FNO2d for topology upscaling.")

    # Identifiers
    parser.add_argument("--model_name",        type=str,   required=True,
                        help="Unique model name. Outputs saved to models/<model_name>/")

    # Data
    parser.add_argument("--data_in_dir",       type=str,   required=True,
                        help="Low-res input dir (e.g. CNT_40x20/CNT_40x20).")
    parser.add_argument("--data_out_dir",      type=str,   required=True,
                        help="Training target dir (e.g. CNT_80X40/CNT_80X40).")
    parser.add_argument("--data_val_dir",      type=str,   required=True,
                        help="Validation ground-truth dir (e.g. CNT_100x50/CNT_100x50).")

    # Architecture
    parser.add_argument("--modes1",            type=int,   default=12,
                        help="Fourier modes along height (default 12).")
    parser.add_argument("--modes2",            type=int,   default=12,
                        help="Fourier modes along width  (default 12).")
    parser.add_argument("--width",             type=int,   default=32,
                        help="Hidden channel width (default 32).")

    # Training
    parser.add_argument("--batch_size",        type=int,   default=8)
    parser.add_argument("--epochs",            type=int,   default=100)
    parser.add_argument("--save_every",        type=int,   default=20)
    parser.add_argument("--learning_rate",     type=float, default=1e-3)
    parser.add_argument("--load_model",        type=str,   default=None,
                        help="Optional: path to .pth to resume training.")

    args = parser.parse_args()
    train(args)
