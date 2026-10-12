"""
Offline analysis of the diagnostic snapshots saved around the epoch-11 Mouse_5 loss spike
(lightning_params.snapshot_steps; see Neighbor2InverseSlice._save_snapshot).

Each snapshot holds the network weights from BEFORE optimizer update N, plus the normalized
input/target batch that update N used. This script:

  A. Inventory: what each snapshot contains (scan, epoch, batch, logged loss, file contents).
  B. Reproduction check: each snapshot's weights on its own batch, in bf16 autocast as in
     training, should reproduce the loss logged on the "update ... step=N" line.
  C. Cross table: every saved weight set x the key batches (Mouse_9 before the spike,
     Mouse_5 at onset/end of the spike, Mouse_5 in epoch 12), in bf16 AND fp32.
     Reports res_mean (= mean of output - input, the uniform shift) and the loss.
  D. Layer trace for three contrasts, using forward hooks on every conv / transposed conv:
       1. same weights (553), Mouse_9 input vs Mouse_5 input  -> why Mouse_5 and not Mouse_9
       2. same Mouse_5 input (553), weights 552 vs 553         -> did the last Mouse_9 update tip it?
       3. same Mouse_5 input (576), weights 576 vs 577         -> what switched back at the recovery?
     Per layer: mean, fraction of positive pre-activations (the LeakyReLU "switch" state) and
     max |value|; then the layers and channels that change most.

Run from PBPC_data/ (a GPU is strongly recommended; on CPU this is slow):
    python analyze_snapshots.py --snap_dir model_weights/Cone/N2I_cone_projSub_LNei/lightning_logs/version_9/snapshots --gpu 0 > snapshot_analysis.txt
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--snap_dir", type=str, required=True)
parser.add_argument("--trainparams", type=str, default="./trainparamsCone.yml")
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--trace_precision", choices=["bf16", "fp32"], default="bf16",
                    help="precision for the layer trace (training used bf16 autocast)")
parser.add_argument("--n_layers", type=int, default=3, help="most-changed layers to detail per contrast")
parser.add_argument("--n_channels", type=int, default=5, help="most-changed channels to list per detailed layer")
args = parser.parse_args()

if args.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import sys
sys.path.append("../")
import contextlib
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from network import UNet
from modelLightning import pad_to_divisible, unpad_from_divisible

KEY_BATCHES = (552, 553, 576, 577)  # Mouse_9 before spike, Mouse_5 onset, Mouse_5 end, Mouse_5 e12


# ---------------------------------------------------------------------------- helpers
def nbytes(obj):
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nbytes(v) for v in obj)
    return 0


def label(s):
    return f"{s['scan_id']} e{s['epoch']} r{s['realization']}b{s['batch']}"


def precision_ctx(precision, device):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def forward(net, x, precision, device):
    """Same forward path as training: pad to a multiple of 32, network, unpad."""
    with torch.no_grad(), precision_ctx(precision, device):
        xp, pad = pad_to_divisible(x.to(device), 32)
        out = unpad_from_divisible(net(xp), pad)
    return out.float()


def metrics(out, s, device):
    x = s["input"].to(device).float()
    t = s["target"].to(device).float()
    return {
        "loss": float(F.mse_loss(out, t)),
        "err_mean": float((out - t).mean()),
        "res_mean": float((out - x).mean()),
        "res_std": float((out - x).std(correction=0)),
    }


def layer_stats(net, x, precision, device):
    """Pre-activation statistics of every conv / transposed conv, in execution order.
    Computed inside the hook because the following LeakyReLU is in-place."""
    stats = []
    hooks = []
    for name, m in net.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            def hook(mod, inp, out, name=name):
                o = out.detach().float()
                stats.append({
                    "name": name,
                    "mean": float(o.mean()),
                    "std": float(o.std(correction=0)),
                    "maxabs": float(o.abs().max()),
                    "frac_pos": float((o > 0).float().mean()),
                    "chan_mean": o.mean(dim=(0, 2, 3)).cpu(),
                    "chan_frac_pos": (o > 0).float().mean(dim=(0, 2, 3)).cpu(),
                })
            hooks.append(m.register_forward_hook(hook))
    try:
        forward(net, x, precision, device)
    finally:
        for h in hooks:
            h.remove()
    return stats


def print_contrast(title, desc_a, desc_b, sa, sb, ma, mb):
    print(f"\n{'=' * 100}\n{title}")
    print(f"  A = {desc_a}:  res_mean={ma['res_mean']:+.4f}  loss={ma['loss']:.5f}")
    print(f"  B = {desc_b}:  res_mean={mb['res_mean']:+.4f}  loss={mb['loss']:.5f}")
    print(f"\n  {'layer':<16}{'mean A':>10}{'mean B':>10}{'d(mean)/std':>13}"
          f"{'frac+ A':>10}{'frac+ B':>10}{'max|x| A':>11}{'max|x| B':>11}")
    changes = []
    for a, b in zip(sa, sb):
        assert a["name"] == b["name"]
        rel = (b["mean"] - a["mean"]) / (a["std"] + 1e-8)
        changes.append((abs(rel), a, b))
        print(f"  {a['name']:<16}{a['mean']:>10.4f}{b['mean']:>10.4f}{rel:>13.3f}"
              f"{a['frac_pos']:>10.3f}{b['frac_pos']:>10.3f}{a['maxabs']:>11.3f}{b['maxabs']:>11.3f}")

    print(f"\n  Layers with the largest change in mean (relative to A's std), with the channels that change most:")
    for _, a, b in sorted(changes, key=lambda c: -c[0])[:args.n_layers]:
        d = b["chan_mean"] - a["chan_mean"]
        top = torch.argsort(d.abs(), descending=True)[:args.n_channels]
        print(f"    {a['name']} ({len(d)} channels):")
        for c in top.tolist():
            print(f"      ch {c:>3}: mean {float(a['chan_mean'][c]):+.4f} -> {float(b['chan_mean'][c]):+.4f}   "
                  f"frac+ {float(a['chan_frac_pos'][c]):.3f} -> {float(b['chan_frac_pos'][c]):.3f}")


# ---------------------------------------------------------------------------- main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    t0 = time.time()

    with open(args.trainparams, "r") as f:
        net_params = yaml.safe_load(f)["base_network"]["params"]

    paths = sorted(glob.glob(os.path.join(args.snap_dir, "snapshot_step*.pt")))
    if not paths:
        raise FileNotFoundError(f"No snapshot_step*.pt files in {args.snap_dir}")
    snaps = {}
    for p in paths:
        s = torch.load(p, map_location="cpu", weights_only=False)
        snaps[int(s["update_step"])] = s
    steps = sorted(snaps)

    net = UNet(**net_params).to(device)
    net.eval()  # no BatchNorm/Dropout in this UNet, so train/eval mode give identical outputs

    def load(step):
        net.load_state_dict(snaps[step]["network_state_dict"])

    # ---- A. inventory
    print(f"\nA. INVENTORY ({len(steps)} snapshots)")
    for st in steps:
        s = snaps[st]
        parts = {k: nbytes(v) / 1e6 for k, v in s.items() if nbytes(v) > 0}
        parts_str = ", ".join(f"{k} {v:.1f} MB" for k, v in parts.items())
        print(f"  step {st}: {label(s)}  logged loss={s['loss']:.6f}  input {tuple(s['input'].shape)}")
        print(f"            {parts_str}")

    # ---- B. reproduction check
    print("\nB. REPRODUCTION CHECK (weights before update N on update N's batch, bf16 as in training)")
    for st in steps:
        load(st)
        m = metrics(forward(net, snaps[st]["input"], "bf16", device), snaps[st], device)
        print(f"  step {st}: recomputed loss={m['loss']:.6f}  logged={snaps[st]['loss']:.6f}  "
              f"diff={m['loss'] - snaps[st]['loss']:+.2e}  res_mean={m['res_mean']:+.4f}")

    # ---- C. cross table
    batches = [b for b in KEY_BATCHES if b in snaps]
    for precision in ("bf16", "fp32"):
        print(f"\nC. CROSS TABLE ({precision}): rows = weights (before update N), "
              f"columns = batch; cells = res_mean / loss")
        print("  " + " " * 14 + "".join(f"{label(snaps[b]):>24}" for b in batches))
        for st in steps:
            load(st)
            cells = []
            for b in batches:
                m = metrics(forward(net, snaps[b]["input"], precision, device), snaps[b], device)
                cells.append(f"{m['res_mean']:+.3f} / {m['loss']:.4f}")
            print(f"  weights {st:<6}" + "".join(f"{c:>24}" for c in cells))

    # ---- D. layer traces
    contrasts = [
        ("D1. SAME WEIGHTS (before 553), Mouse_9 input vs Mouse_5 input", (553, 552), (553, 553)),
        ("D2. SAME MOUSE_5 INPUT (553), weights before 552 vs before 553", (552, 553), (553, 553)),
        ("D3. SAME MOUSE_5 e11 INPUT (576), weights before 576 vs before 577 (recovery)", (576, 576), (577, 576)),
    ]
    p = args.trace_precision
    print(f"\nD. LAYER TRACES ({p}); pre-activation statistics; frac+ = fraction of values > 0")
    for title, (wa, ba), (wb, bb) in contrasts:
        if not all(k in snaps for k in (wa, ba, wb, bb)):
            print(f"\n{title}: skipped (snapshot missing)")
            continue
        load(wa)
        sa = layer_stats(net, snaps[ba]["input"], p, device)
        ma = metrics(forward(net, snaps[ba]["input"], p, device), snaps[ba], device)
        load(wb)
        sb = layer_stats(net, snaps[bb]["input"], p, device)
        mb = metrics(forward(net, snaps[bb]["input"], p, device), snaps[bb], device)
        print_contrast(
            title,
            f"weights {wa} on {label(snaps[ba])}",
            f"weights {wb} on {label(snaps[bb])}",
            sa, sb, ma, mb,
        )

    print(f"\nDone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
