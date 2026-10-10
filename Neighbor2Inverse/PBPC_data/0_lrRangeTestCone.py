"""
LR range test (Smith, 2017) for cone-beam Neighbor2Inverse.

Reproduces the algorithm of Lightning 2.5.0's `Tuner.lr_find` (exponential mode),
which the Neighbor2Inverse paper used to choose its initial learning rate. Lightning's
own finder cannot be used here: the cone path uses manual optimization, so Lightning
never steps the finder's scheduler, and one Lightning "batch" is a whole scan.

What is reproduced from lightning/pytorch/tuner/lr_finder.py (v2.5.0):
  - LR schedule (_ExponentialLR): step 0 uses min_lr; step k >= 1 uses
    min_lr * (max_lr / min_lr) ** ((k + 1) / num_training).
  - Loss smoothing (_LRCallback): beta = 0.98, bias-corrected with global_step = k + 1.
  - Early stop: after a step k >= 1 whose smoothed loss exceeds
    early_stop_threshold * best smoothed loss.
  - Suggestion (_LRFinder.suggestion): steepest negative gradient of the smoothed loss,
    skipping the first 10 and last 1 points.

Note on Lightning's smoothing: global_step is already k + 1 when the loss is recorded,
so the bias correction uses (1 - beta ** (k + 2)) instead of (1 - beta ** (k + 1)).
The first smoothed value is therefore about half the true loss, which makes the
early stop fire at roughly 2x (not 4x) the initial loss, and a flat loss appears to
rise by ~7% between steps 10 and 50. This script reproduces that behaviour for the
official suggestion, and additionally records a correctly bias-corrected curve
(loss_smoothed_corrected) with its own suggestion as a cross-check.

The training data path is the same as training_step_cone: shuffled training scans,
CONE_NUM_REALIZATIONS fresh Neighbor masks per scan, CONE_MINI_BATCH_SIZE slices per
optimizer step, per-scan normalization, the optimizer from configure_optimizers, and
bf16 autocast when the YAML uses precision "bf16-mixed".

Run from PBPC_data/:
    python 0_lrRangeTestCone.py --gpu 0
Then set lightning_params.lr in trainparamsCone.yml from the suggestion (after
checking the plot).
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trainparams", type=str, default="./trainparamsCone.yml")
parser.add_argument("--gpu", type=int, default=None, help="GPU used by both torch and ASTRA")
parser.add_argument("--min_lr", type=float, default=1e-8)
parser.add_argument("--max_lr", type=float, default=1.0)
parser.add_argument("--num_training", type=int, default=100)
parser.add_argument("--early_stop_threshold", type=float, default=4.0, help="<= 0 disables early stopping")
parser.add_argument("--seed", type=int, default=42, help="same seed as 1_trainNeighbor2Inverse.py")
parser.add_argument("--out_dir", type=str, default="./lr_range_test/")
args = parser.parse_args()

# Must be set before torch/ASTRA initialise CUDA, so both use the same GPU.
if args.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import sys
sys.path.append("../")
import contextlib
import numpy as np
import pandas as pd
import h5py
import torch
import yaml
import lightning as pl
from torch.utils.data import DataLoader

from modelLightning import Neighbor2InverseSlice, pad_to_divisible, unpad_from_divisible
from network import UNet
from dataset import ConeBeamProjDataset

BETA = 0.98  # Lightning _LRCallback default
SKIP_BEGIN, SKIP_END = 10, 1  # Lightning suggestion() defaults


def lightning_exponential_lr(k, min_lr, max_lr, num_training):
    """LR used at step k by Lightning's _ExponentialLR (including its step-0 behaviour)."""
    if k == 0:
        return min_lr
    return min_lr * (max_lr / min_lr) ** ((k + 1) / num_training)


def lightning_suggestion(lrs, smoothed_losses):
    """Lightning _LRFinder.suggestion(skip_begin=10, skip_end=1). Returns (lr, index) or (None, None)."""
    losses = np.asarray(smoothed_losses[SKIP_BEGIN:-SKIP_END], dtype=np.float64)
    losses = losses[np.isfinite(losses)]
    if len(losses) < 2:
        return None, None
    # np.gradient matches torch.gradient here: unit spacing, central differences, one-sided edges.
    min_grad = int(np.argmin(np.gradient(losses)))
    idx = min_grad + SKIP_BEGIN
    return lrs[idx], idx


def autocast_context(precision, device):
    precision = str(precision)
    if precision == "bf16-mixed":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    if precision in ("32", "32-true"):
        return contextlib.nullcontext()
    raise NotImplementedError(f"LR range test does not handle precision '{precision}'")


def build_model(trainparams):
    """Same construction as 1_trainNeighbor2Inverse.py (cone path)."""
    base_network = UNet(**trainparams["base_network"]["params"])
    return Neighbor2InverseSlice(
        network=base_network,
        **trainparams["lightning_params"],
        optimizer_algo=trainparams["optimizer_algo"],
        scheduler_algo=trainparams["scheduler_algo"],
        optimizer_params=trainparams["optimizer_params"],
        scheduler_params=trainparams["scheduler_params"],
        n_slicesPR=trainparams.get("dataset", {}).get("n_slicesPR"),
    )


def save_plot(df, suggestion, suggestion_corr, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["lr"], df["loss_raw"], alpha=0.3, label="raw loss")
    ax.plot(df["lr"], df["loss_smoothed"], label="smoothed loss (Lightning)")
    ax.plot(df["lr"], df["loss_smoothed_corrected"], ls=":", label="smoothed loss (bias-corrected)")
    if suggestion is not None:
        ax.axvline(suggestion, color="r", ls="--", label=f"Lightning suggestion = {suggestion:.2e}")
    if suggestion_corr is not None:
        ax.axvline(suggestion_corr, color="k", ls=":", label=f"corrected suggestion = {suggestion_corr:.2e}")
    # Skip the first 10 points (as the suggestion does) when setting the y-range,
    # so the smoothing warm-up doesn't flatten the interesting region.
    tail = df.iloc[SKIP_BEGIN:]
    finite = tail[["loss_smoothed", "loss_smoothed_corrected"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite):
        lo, hi = finite.min().min(), finite.max().max()
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("training loss (MSE)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    with open(args.trainparams, "r") as f:
        trainparams = yaml.safe_load(f)
    if not trainparams["lightning_params"].get("coneBeam", False):
        raise ValueError("This LR range test is for the cone-beam path (coneBeam: True)")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    pl.seed_everything(args.seed)
    model = build_model(trainparams).to(device)
    model.train()

    opt_cfg = model.configure_optimizers()
    optimizer = opt_cfg["optimizer"] if isinstance(opt_cfg, dict) else opt_cfg
    precision = trainparams.get("trainer_params", {}).get("precision", "32")

    loader_params = dict(trainparams["train_loader"])
    loader_params["persistent_workers"] = False  # loader is re-iterated below
    loader = DataLoader(ConeBeamProjDataset(**trainparams["dataset_train"]), **loader_params)

    n = args.num_training
    lrs, raw_losses, smoothed_losses, corrected_losses, scans = [], [], [], [], []
    avg_loss, best_loss = 0.0, 0.0
    k = 0
    stop = False

    while not stop:
        for batch in loader:
            path_proj, scan_id, det_off, src_off = model._unpack_cone_batch(batch)

            with autocast_context(precision, device):
                slice_groups = model.cone_training_slice_groups()

                with h5py.File(path_proj, "r") as f:
                    data = f["images"]

                    for realization_idx, slice_group in enumerate(slice_groups, start=1):
                        print(f"[{k}/{n}] {scan_id}: realization {realization_idx}/{len(slice_groups)}")
                        reco_sub1, reco_sub2 = model.make_cone_pair(data, det_off, src_off)

                        for start in range(0, slice_group.numel(), model.CONE_MINI_BATCH_SIZE):
                            idx = slice_group[start:start + model.CONE_MINI_BATCH_SIZE]

                            lr = lightning_exponential_lr(k, args.min_lr, args.max_lr, n)
                            for group in optimizer.param_groups:
                                group["lr"] = lr

                            batch_sub1 = model.normalize_cone(reco_sub1[idx], scan_id).unsqueeze(1)
                            batch_sub2 = model.normalize_cone(reco_sub2[idx], scan_id).unsqueeze(1)
                            noisy_inpt, pad = pad_to_divisible(batch_sub1, 32)

                            optimizer.zero_grad()
                            noisy_output = unpad_from_divisible(model(noisy_inpt), pad)
                            loss = model.loss(noisy_output, batch_sub2)
                            loss.backward()
                            optimizer.step()

                            # Lightning _LRCallback.on_train_batch_end, with global_step = k + 1
                            current_loss = loss.item()
                            current_step = k + 1
                            avg_loss = BETA * avg_loss + (1 - BETA) * current_loss
                            smoothed = avg_loss / (1 - BETA ** (current_step + 1))
                            corrected = avg_loss / (1 - BETA ** current_step)  # standard EMA bias correction

                            if (
                                args.early_stop_threshold > 0
                                and current_step > 1
                                and smoothed > args.early_stop_threshold * best_loss
                            ):
                                stop = True
                            if smoothed < best_loss or current_step == 1:
                                best_loss = smoothed

                            lrs.append(lr)
                            raw_losses.append(current_loss)
                            smoothed_losses.append(smoothed)
                            corrected_losses.append(corrected)
                            scans.append(scan_id)
                            k += 1

                            if stop or k >= n:
                                break
                        del reco_sub1, reco_sub2
                        torch.cuda.empty_cache()
                        if stop or k >= n:
                            break
            if stop or k >= n:
                stop = True
                break

    if k < n:
        print(f"Stopped early after {k} steps: smoothed loss exceeded "
              f"{args.early_stop_threshold}x the best value.")

    suggestion, sugg_idx = lightning_suggestion(lrs, smoothed_losses)
    suggestion_corr, sugg_idx_corr = lightning_suggestion(lrs, corrected_losses)

    df = pd.DataFrame({
        "step": np.arange(len(lrs)),
        "lr": lrs,
        "loss_raw": raw_losses,
        "loss_smoothed": smoothed_losses,
        "loss_smoothed_corrected": corrected_losses,
        "scan_id": scans,
    })
    csv_path = os.path.join(args.out_dir, "lr_range_test.csv")
    png_path = os.path.join(args.out_dir, "lr_range_test.png")
    df.to_csv(csv_path, index=False)
    save_plot(df, suggestion, suggestion_corr, png_path)

    if suggestion is None:
        print("No suggestion: too few points (need more than "
              f"{SKIP_BEGIN + SKIP_END + 1} steps).")
    else:
        print(f"Suggested learning rate (Lightning): {suggestion:.3e} (step {sugg_idx})")
    if suggestion_corr is not None:
        print(f"Cross-check, bias-corrected smoothing: {suggestion_corr:.3e} (step {sugg_idx_corr})")
    print(f"Saved {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
