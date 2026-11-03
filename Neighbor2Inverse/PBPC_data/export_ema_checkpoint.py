"""
Write a copy of cone checkpoints with state_dict replaced by one of their EMA weights (ema_decays),
so every existing tool (infer_cone_fullres.py, visualize_cone_correction.py) runs EMA models
unchanged, with strict=True loading.

Each copy goes to a sibling folder named after the EMA tag, with the same file name:
    .../version_0/checkpoints/periodic_epoch=49_....ckpt
 -> .../version_0/checkpoints_ema0p996/periodic_epoch=49_....ckpt
so globs such as checkpoints/periodic_epoch=49[-_]*.ckpt never pick up EMA copies, and the same glob
works in the EMA folder. The copy keeps "epoch" and "global_step" and records its source under
"ema_export"; optimizer/scheduler states and the other EMA copies are dropped (inference does not use
them). Existing files are not overwritten.

Usage (from PBPC_data/):
    python export_ema_checkpoint.py --ema ema0p996 --ckpts model_weights/Cone/<run>/lightning_logs/version_0/checkpoints/periodic_epoch=*.ckpt
"""
import argparse
import os

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpts", nargs="+", required=True, help="checkpoints saved with ema_decays set")
    p.add_argument("--ema", required=True, help="EMA tag, e.g. ema0p996 (decay 0.996) or ema0p99")
    args = p.parse_args()

    for path in args.ckpts:
        ckpt = torch.load(path, map_location="cpu")
        weights = (ckpt.get("ema") or {}).get("weights", {})
        if args.ema not in weights:
            raise KeyError(f"{path}: no EMA '{args.ema}' (available: {sorted(weights) or 'none'})")
        ema_sd = weights[args.ema]["state_dict"]
        if set(ema_sd) != set(ckpt["state_dict"]):
            raise ValueError(f"{path}: EMA parameter names differ from state_dict")

        src_dir, name = os.path.split(os.path.abspath(path))
        out_dir = os.path.join(os.path.dirname(src_dir), f"{os.path.basename(src_dir)}_{args.ema}")
        out = os.path.join(out_dir, name)
        if os.path.exists(out):
            print(f"exists, skipped: {out}")
            continue
        os.makedirs(out_dir, exist_ok=True)
        torch.save({
            "state_dict": dict(ema_sd),
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
            "ema_export": {"source": os.path.abspath(path), "tag": args.ema,
                           "decay": weights[args.ema]["decay"], "updates": ckpt["ema"]["steps"]},
        }, out)
        print(f"{args.ema} (decay {weights[args.ema]['decay']:g}, {ckpt['ema']['steps']} updates, "
              f"epoch {ckpt.get('epoch')}): {out}")


if __name__ == "__main__":
    main()
