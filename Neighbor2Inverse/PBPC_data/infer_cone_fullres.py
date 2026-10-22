"""
Full-resolution inference of one or more cone checkpoints on one scan, for side-by-side comparison.

Input: the scan's full-res reconstruction from 0_calculateStatsCone.py --save_dir (training
geometry: 2024 frames, rows [:512], full-res Paganin), listed as path_reco_fullres in --csv, with
the scan's own stats in cone_stats.csv (as in training: each scan normalised by its own mean/std).

Each checkpoint is loaded into a model built exactly as 1_trainNeighbor2Inverse.py builds it, and
the volume is denoised with the model's own denoise_fullres_cone (the f(y) step of L_reg: one
slice per forward pass, normalise with the scan's stats, reflect-pad to 32, unpad, un-normalise).

Outputs in --out_dir:
  <label>_denoised.npy          denoised volume per checkpoint (unless --no_volumes)
  z<k>_compare.tif              per preview slice: input, each output, each residual (input - output),
                                ImageJ slice labels; float32, physical units (as the reconstructions)
Prints per checkpoint: output / residual statistics inside the FOV (central slices).

--half_res {binned, subsample, both}: instead of the full-res volume, denoise reconstructions at the
TRAINING voxel size ([256, 645, 645], training geometry, normalised with the scan's stats):
  binned     all photons: projections binned 2x2 -> Paganin at the half-res pixel -> FDK
  subsample  one Neighbor subsample g1 (fixed seed) via the model's make_cone_pair, i.e. exactly a
             training input (~1/4 of the photons per voxel: the noise level the network trained on)
  both       both inputs, side by side in the preview stacks
Cached in --out_dir. Preview slices are still given as full-res indices and mapped to z // 2.
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpts", type=str, nargs="+", required=True, help="checkpoint paths")
parser.add_argument("--labels", type=str, nargs="+", required=True, help="one short label per checkpoint")
parser.add_argument("--csv", type=str, default="./testCone.csv")
parser.add_argument("--scan_id", type=str, default=None, help="default: first row of --csv")
parser.add_argument("--trainparams", type=str, default="./trainparamsCone.yml")
parser.add_argument("--slices", type=int, nargs="+", default=[100, 170, 340, 420], help="full-res preview slices")
parser.add_argument("--out_dir", type=str, default=None, help="default: ./inference_<scan_id>")
parser.add_argument("--no_volumes", action="store_true", help="do not save the denoised volumes")
parser.add_argument("--bf16", action="store_true", help="run the network under bf16 autocast (as in training)")
parser.add_argument("--half_res", nargs="?", const="binned", default=None, choices=["binned", "subsample", "both"],
                    help="denoise reconstruction(s) at the training voxel size (see module docstring)")
parser.add_argument("--subsample_seed", type=int, default=0, help="Neighbor mask seed for --half_res subsample")
args = parser.parse_args()

if len(args.ckpts) != len(args.labels):
    parser.error("--ckpts and --labels must have the same length")

import sys
sys.path.append("../")
import contextlib
import h5py
import numpy as np
import pandas as pd
import tifffile
import torch
import yaml

from network import UNet
from modelLightning import Neighbor2InverseSlice
from astra_cone_array import _create_circular_mask

torch.set_float32_matmul_precision("medium")


def build_model(tp, ckpt_path):
    model = Neighbor2InverseSlice(
        network=UNet(**tp["base_network"]["params"]),
        **tp["lightning_params"],
        optimizer_algo=tp["optimizer_algo"], scheduler_algo=tp["scheduler_algo"],
        optimizer_params=tp["optimizer_params"], scheduler_params=tp["scheduler_params"],
        n_slicesPR=tp.get("dataset", {}).get("n_slicesPR"),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to("cuda").eval(), ckpt.get("epoch")


def halfres_volume(model, row, out_dir, chunk=64):
    """Training-voxel-size reconstruction from all photons (see module docstring); cached as .npy."""
    path = os.path.join(out_dir, f"{row['scan_id']}_reco_halfres_binned.npy")
    if os.path.exists(path):
        print(f"loading cached {path}")
        return np.load(path)
    n, rows = model.CONE_N_PROJECTIONS, model.CONE_DETECTOR_ROWS
    with h5py.File(row["path_proj"], "r") as f:
        data = f["images"]
        if data.shape[0] != n or data.shape[1] < rows:
            raise ValueError(f"{row['path_proj']}: shape {data.shape}, expected {n} frames and >= {rows} rows")
        cols = 2 * (data.shape[2] // 2)
        binned = np.empty((n, rows // 2, cols // 2), dtype=np.float32)
        for s0 in range(0, n, chunk):
            s1 = min(s0 + chunk, n)
            block = data[s0:s1, :rows, :cols].astype(np.float32)
            binned[s0:s1] = block.reshape(s1 - s0, rows // 2, 2, cols // 2, 2).mean(axis=(2, 4))
    print(f"binned projections {binned.shape}: mean {binned.mean():.5f}")
    thickness = model._paganin_thickness_m(binned)          # half-res Paganin, metres
    del binned
    vol = model.reconstruct_cone(torch.from_numpy(thickness),
                                 detector_offset_m=float(row["detector_offset_m"]),
                                 source_offset_m=float(row["source_offset_m"])).cpu().numpy().astype(np.float32)
    del thickness
    np.save(path, vol)
    print(f"half-res reconstruction {vol.shape} saved to {path}")
    return vol


def subsample_volume(model, row, out_dir, seed):
    """Half-res reconstruction of one Neighbor subsample g1, exactly as a training input; cached as .npy."""
    path = os.path.join(out_dir, f"{row['scan_id']}_reco_halfres_sub1_seed{seed}.npy")
    if os.path.exists(path):
        print(f"loading cached {path}")
        return np.load(path)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad(), h5py.File(row["path_proj"], "r") as f:
        reco1, reco2 = model.make_cone_pair(f["images"], float(row["detector_offset_m"]),
                                            float(row["source_offset_m"]), generator=gen,
                                            tag=f"{row['scan_id']} sub")
    vol = reco1.float().cpu().numpy()
    del reco1, reco2
    torch.cuda.empty_cache()
    np.save(path, vol)
    print(f"subsample reconstruction {vol.shape} saved to {path}")
    return vol


def highpass_std(v, disk):
    """Std of (v - 3x3 box mean) inside the FOV: a rough per-voxel noise level."""
    pad = np.pad(v, ((0, 0), (1, 1), (1, 1)), mode="edge")
    box = sum(pad[:, i:i + v.shape[1], j:j + v.shape[2]] for i in range(3) for j in range(3)) / 9.0
    return float((v - box)[:, disk].std())


def main():
    with open(args.trainparams, "r") as f:
        tp = yaml.safe_load(f)

    df = pd.read_csv(args.csv)
    if args.scan_id is None:
        row = df.iloc[0]
    else:
        match = df[df["scan_id"].astype(str) == args.scan_id]
        if len(match) != 1:
            raise KeyError(f"scan_id {args.scan_id!r} not found exactly once in {args.csv}")
        row = match.iloc[0]
    scan_id = str(row["scan_id"])
    out_dir = args.out_dir or (f"./inference_{scan_id}_halfres" if args.half_res else f"./inference_{scan_id}")
    os.makedirs(out_dir, exist_ok=True)

    bad = [z for z in args.slices if not 0 <= z < 512]
    if bad:
        raise ValueError(f"preview slices out of range: {bad}")
    if args.half_res:
        args.slices = [z // 2 for z in args.slices]
    kinds = (["binned", "subsample"] if args.half_res == "both" else [args.half_res]) if args.half_res else ["fullres"]

    inputs = None
    previews = {z: [] for z in args.slices}  # z -> [(frame, label, role)]
    disk, z_central = None, None

    for ckpt_path, label in zip(args.ckpts, args.labels):
        model, epoch = build_model(tp, ckpt_path)
        mean, std = model.cone_scan_stats(scan_id)
        if inputs is None:
            inputs = {}
            for k in kinds:
                if k == "fullres":
                    inputs[k] = model.load_fullres_cone(row["path_reco_fullres"])
                elif k == "binned":
                    inputs[k] = halfres_volume(model, row, out_dir)
                else:
                    inputs[k] = subsample_volume(model, row, out_dir, args.subsample_seed)
            first = inputs[kinds[0]]
            margin = 2 * model.CONE_SLICE_MARGIN if kinds == ["fullres"] else model.CONE_SLICE_MARGIN
            disk = _create_circular_mask((first.shape[1], first.shape[2]))
            z_central = slice(margin, first.shape[0] - margin)
            for k, vol in inputs.items():
                print(f"{scan_id} input '{k}' {vol.shape}: inside FOV std {vol[z_central][:, disk].std():.5f}, "
                      f"high-pass (noise-like) std {highpass_std(vol[z_central], disk):.5f}; "
                      f"scan stats mean {mean:.5f} std {std:.5f}")
                for z in args.slices:
                    previews[z].append((vol[z], f"input {k} z{z}", "input"))

        for k, vol in inputs.items():
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else contextlib.nullcontext()
            with ctx:
                denoised = model.denoise_fullres_cone(vol, scan_id)

            vin = vol[z_central][:, disk]
            vout = denoised[z_central][:, disk]
            res = vin - vout
            print(f"[{label} | {k}] {ckpt_path} (epoch {epoch}): inside FOV: output std {vout.std():.5f}, "
                  f"high-pass std {highpass_std(denoised[z_central], disk):.5f}; residual mean {res.mean():+.5f} "
                  f"std {res.std():.5f} ({res.std() / std:.4f} scan std)")
            del vin, vout, res

            tag = label if kinds == ["fullres"] else f"{label} ({k})"
            for z in args.slices:
                previews[z].append((denoised[z], f"{tag} z{z}", "output"))
                previews[z].append((vol[z] - denoised[z], f"{tag} residual z{z}", "residual"))

            if not args.no_volumes:
                name = f"{label}_denoised.npy" if kinds == ["fullres"] else f"{label}_{k}_denoised.npy"
                np.save(os.path.join(out_dir, name), denoised)
                print(f"saved {os.path.join(out_dir, name)}")
            del denoised
        del model
        torch.cuda.empty_cache()

    for z, items in previews.items():
        ordered = [it for role in ("input", "output", "residual") for it in items if it[2] == role]
        path = os.path.join(out_dir, f"z{z}_compare.tif")
        tifffile.imwrite(path, np.stack([f for f, _, _ in ordered]).astype(np.float32),
                         imagej=True, metadata={"Labels": [l for _, l, _ in ordered]})
        print(f"saved {path}")


if __name__ == "__main__":
    main()
