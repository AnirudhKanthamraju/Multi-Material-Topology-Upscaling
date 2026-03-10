"""
eval_model.py — Evaluate or run inference with a named FNO2d model.

The script reads models/<model_name>/config.json to reconstruct the exact
architecture, then loads models/<model_name>/<model_name>_final.pth (or a
specific checkpoint via --checkpoint).

Two modes:
  1. Batch evaluation  — Run all test-split images through the model and report
                         aggregate MSE & MAE against a ground-truth directory.
  2. Single-image inference — Upscale one image to a target resolution and
                              save the result as a PNG.

Usage examples:
  # Batch evaluation:
  python eval_model.py \\
      --model_name      fno_base \\
      --data_in_dir     ./CNT_40x20/CNT_40x20 \\
      --data_target_dir ./CNT_100x50/CNT_100x50

  # Single-image inference:
  python eval_model.py \\
      --model_name  fno_base \\
      --input_image ./CNT_40x20/CNT_40x20/125.jpg \\
      --target_res  100 50

  # Batch using a specific checkpoint instead of final weights:
  python eval_model.py \\
      --model_name fno_base \\
      --checkpoint models/fno_base/weights/fno_base_epoch_0060.pth \\
      --data_in_dir     ./CNT_40x20/CNT_40x20 \\
      --data_target_dir ./CNT_100x50/CNT_100x50
"""

import os
import json
import argparse

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

from models.fno import FNO2d


# ---------------------------------------------------------------------------
# Config / model loading
# ---------------------------------------------------------------------------

def load_config(model_name: str) -> dict:
    """Read config.json from the named model directory."""
    cfg_path = os.path.join("models", model_name, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"No config found at {cfg_path}. "
            f"Has '{model_name}' been trained yet?"
        )
    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"[INFO] Config loaded: modes1={cfg['modes1']}, modes2={cfg['modes2']}, width={cfg['width']}")
    return cfg


def load_model(model_name: str, checkpoint: str = None, device: str = "cpu") -> FNO2d:
    """
    Reconstruct a named model from its config and load its weights.

    Args:
        model_name : name of the model (folder under models/).
        checkpoint : optional explicit .pth path; defaults to <model_name>_final.pth.
        device     : 'cuda' or 'cpu'.
    Returns:
        Loaded, eval-mode FNO2d.
    """
    cfg   = load_config(model_name)
    model = FNO2d(
        modes1=cfg["modes1"],
        modes2=cfg["modes2"],
        width =cfg["width"],
    ).to(device)

    weights_path = checkpoint or os.path.join("models", model_name, f"{model_name}_final.pth")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. "
            f"Train '{model_name}' first, or pass --checkpoint."
        )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f"[INFO] Weights loaded from {weights_path}")
    return model


# ---------------------------------------------------------------------------
# Loss comparison
# ---------------------------------------------------------------------------

def compare_topologies(preds: torch.Tensor, targets: torch.Tensor):
    """
    Compute pixel-wise MSE and MAE between predicted and ground-truth topologies.

    Args:
        preds   : (N, H, W) predicted topology tensor.
        targets : (N, H, W) ground-truth topology tensor at the same resolution.
    Returns:
        Tuple (mse, mae) as Python floats.
    """
    mse = nn.MSELoss()(preds, targets).item()
    mae = nn.L1Loss()(preds, targets).item()
    print("\n" + "=" * 52)
    print("  Pixel-wise Evaluation Metrics")
    print("=" * 52)
    print(f"  Mean Squared Error (MSE) : {mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print("=" * 52 + "\n")
    return mse, mae


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _sorted_images(directory: str):
    exts = {".jpg", ".jpeg", ".png"}
    files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in exts]
    files.sort(key=lambda f: int(os.path.splitext(f)[0]))
    return files


def _to_tensor(path: str, device) -> torch.Tensor:
    """Load an RGB image as a (3, H, W) float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return transforms.ToTensor()(img).to(device)


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def batch_evaluate(args, model, device):
    """
    Evaluate the model on the test split (last 20% by sorted filename).
    Compares predictions at target resolution against data_target_dir.
    """
    in_files     = _sorted_images(args.data_in_dir)
    target_files = _sorted_images(args.data_target_dir)

    n_total  = len(in_files)
    n_train  = int(n_total * 0.8)
    test_idx = list(range(n_train, n_total))

    print(f"[INFO] Total dataset: {n_total} images -- evaluating on test split: {len(test_idx)} samples")

    all_preds, all_targets = [], []

    with torch.no_grad():
        for i in test_idx:
            if i >= len(target_files):
                print(f"[WARN] No matching target for index {i}, skipping.")
                continue

            x = _to_tensor(os.path.join(args.data_in_dir,     in_files[i]),     device).unsqueeze(0) # (1, 3, H_in, W_in)
            t = _to_tensor(os.path.join(args.data_target_dir, target_files[i]), device).unsqueeze(0) # (1, 3, H_tgt, W_tgt)

            pred = model(x, output_size=(t.shape[-2], t.shape[-1]))
            all_preds.append(pred)
            all_targets.append(t)

    compare_topologies(
        torch.cat(all_preds,   dim=0),
        torch.cat(all_targets, dim=0),
    )


def single_image_infer(args, model, device):
    """Upscale a single input image to the requested target resolution."""
    nelx, nely = args.target_res   # user convention: width first, height second
    print(f"[INFO] Upscaling {args.input_image} -> {nelx} wide x {nely} tall ...")

    x = _to_tensor(args.input_image, device).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        pred = model(x, output_size=(nely, nelx))   # PyTorch: (H=nely, W=nelx)

    out_arr = (pred.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    out_img  = Image.fromarray(out_arr, "RGB")
    stem     = os.path.splitext(os.path.basename(args.input_image))[0]
    save_path = f"{stem}_{args.model_name}_upscaled_{nelx}x{nely}.png"
    out_img.save(save_path)
    print(f"[INFO] Saved upscaled topology to: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate or run inference with a named FNO2d model.")

    # Model identity
    parser.add_argument("--model_name",       type=str, required=True,
                        help="Name of the trained model (folder under models/).")
    parser.add_argument("--checkpoint",       type=str, default=None,
                        help="Optional specific .pth checkpoint. Defaults to <model_name>_final.pth.")

    # Batch evaluation
    parser.add_argument("--data_in_dir",      type=str, default=None,
                        help="Low-res input directory -- batch mode.")
    parser.add_argument("--data_target_dir",  type=str, default=None,
                        help="Ground-truth high-res directory -- batch mode.")

    # Single-image inference
    parser.add_argument("--input_image",      type=str, default=None,
                        help="Path to a single input image -- inference mode.")
    parser.add_argument("--target_res",       type=int, nargs=2, default=[100, 50],
                        metavar=("NELX", "NELY"),
                        help="Target resolution: nelx (width) then nely (height). "
                             "E.g. --target_res 100 50  -> 100 wide x 50 tall.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = load_model(args.model_name, args.checkpoint, device)

    if args.input_image:
        single_image_infer(args, model, device)
    elif args.data_in_dir and args.data_target_dir:
        batch_evaluate(args, model, device)
    else:
        print("[ERROR] Provide --input_image for inference, "
              "or --data_in_dir + --data_target_dir for batch evaluation.")
