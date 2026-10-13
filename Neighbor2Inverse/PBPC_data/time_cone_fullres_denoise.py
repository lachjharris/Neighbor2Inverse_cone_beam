"""
Cost of the f(y) step of L_reg for the cone path: denoise every slice of a saved
full-resolution reconstruction with a trained U-Net, no gradients.

Same preprocessing as SavePredictionCallbackCone / training: per-scan normalisation
(cone_stats.csv), reflect-pad to a multiple of 32, forward, unpad; bf16 autocast as in
training (precision "bf16-mixed"). Slices go host -> GPU -> host, as they would in training.

For each batch size: wall time for all slices, peak GPU memory allocated by torch, and
the maximum difference from the first batch size's output (should be ~bf16 rounding).
Also prints input vs output std inside the FOV as a check that the checkpoint loaded.
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True, help="Lightning checkpoint (.ckpt)")
parser.add_argument("--trainparams", type=str, default="./trainparamsCone.yml")
parser.add_argument("--csv", type=str, default="./valCone.csv")
parser.add_argument("--scan_id", type=str, default="Mouse_10")
parser.add_argument("--stats_path", type=str, default="./cone_stats.csv")
parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8])
parser.add_argument("--gpu", type=int, default=None)
args = parser.parse_args()

if args.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import sys
sys.path.append("../")
import time
import numpy as np
import pandas as pd
import torch
import yaml

from network import UNet
from modelLightning import pad_to_divisible, unpad_from_divisible

torch.set_float32_matmul_precision("medium")  # as 1_trainNeighbor2Inverse.py


def main():
    device = torch.device("cuda")

    with open(args.trainparams, "r") as f:
        trainparams = yaml.safe_load(f)

    net = UNet(**trainparams["base_network"]["params"])
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = {k[len("network."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("network.")}
    net.load_state_dict(state, strict=True)  # fails loudly on any missing or unexpected key
    net = net.to(device).eval()
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch')}, {len(state)} tensors)")

    row = pd.read_csv(args.csv).set_index("scan_id").loc[args.scan_id]
    stats = pd.read_csv(args.stats_path).set_index("scan_id").loc[args.scan_id]
    mean, std = float(stats["mean"]), float(stats["std"])

    V = np.load(row["path_reco_fullres"])
    n_slices = V.shape[0]
    print(f"{args.scan_id}: volume {V.shape} {V.dtype}, stats mean {mean:.5f} std {std:.5f}")

    torch.zeros(1, device=device)  # CUDA context
    reference = None

    for bs in args.batch_sizes:
        out = np.empty_like(V)

        def run_batch(start):
            stop = min(start + bs, n_slices)
            x = torch.from_numpy(V[start:stop]).to(device)[:, None]
            x = (x - mean) / std
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                x_pad, pad = pad_to_divisible(x, 32)
                y = unpad_from_divisible(net(x_pad), pad)
            out[start:stop] = (y.float() * std + mean)[:, 0].cpu().numpy()

        run_batch(0)  # warm-up (cuDNN autotuning, allocator)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        for start in range(0, n_slices, bs):
            run_batch(start)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        peak = torch.cuda.max_memory_allocated() / 1024**3
        msg = f"batch {bs}: {dt:.1f} s for {n_slices} slices ({dt / n_slices * 1e3:.0f} ms/slice), peak GPU allocated {peak:.2f} GiB"
        if reference is None:
            reference = out
        else:
            msg += f", max |diff| vs batch {args.batch_sizes[0]}: {np.abs(out - reference).max():.3g}"
        print(msg)

    fov = V[n_slices // 2] != 0  # FDK zeroes outside the reconstruction disk
    z = slice(64, n_slices - 64)
    vin, vout = V[z][:, fov], reference[z][:, fov]
    print(f"inside FOV, slices {z.start}:{z.stop}: input std {vin.std():.5f}, output std {vout.std():.5f}, "
          f"residual std {(vin - vout).std():.5f}, mean in/out {vin.mean():.5f}/{vout.mean():.5f}")


if __name__ == "__main__":
    main()
