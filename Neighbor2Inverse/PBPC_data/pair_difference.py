"""
Look at the Neighbor pair difference g1 - g2 of any scans, built exactly as in training
(make_cone_pair: Neighbor mask shared across angles -> subsample -> Paganin -> cone FDK), no network.

g1 - g2 = (signal difference between neighbouring pixels) + (noise difference). For one mask the
noise term dominates on noisy data; averaging g1 - g2 over masks cancels the signal term too (each
mask picks neighbour pairs at random in both directions). The mean SQUARED difference over K masks
keeps it: noise adds a roughly even floor (~2 x noise variance), the signal gap adds on top where
neighbouring pixels really differ. On high-dose scans the noise floor is low, so even one g1 - g2 is
mostly signal difference.

The same mask seeds are used for every scan, so a mouse's noisy and HD pairs get identical masks.

Per scan, in --out_dir: <scan_id>_pair_difference.tif, an ImageJ hyperstack (axes ZCYX):
    Z = the --slices (half-res slice indices)
    C = g1, g2, g1 - g2 (first mask), mean (g1 - g2)^2 over --n_masks masks
values normalised with the scan's stats (as in training), float32. Display ranges preset:
g1/g2 mean +- 3 std, g1 - g2 +- 3 std, mean squared difference 0 .. 99th percentile.
Prints per scan: mean (g1 - g2)^2 inside the FOV over all central slices, for each mask.

Usage (from PBPC_data/, on a GPU node). Frame count, arc and stats override the config, so one
config serves both datasets:
    python pair_difference.py --csvs trainCone_noisy.csv valCone_noisy.csv testCone_noisy.csv \\
        --stats_path ./cone_stats_noisy_n1950.csv --n_projections 1950 --total_rotation_deg -346.922
    python pair_difference.py --csvs hdCone.csv \\
        --stats_path ./cone_stats_hd.csv --n_projections 2024 --total_rotation_deg -360.094
"""
import os
import argparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--csvs", nargs="+", required=True, help="scan CSVs (path_proj, scan_id, offsets, ...)")
parser.add_argument("--trainparams", default="./trainparamsCone_noisy.yml", help="config the model is built from")
parser.add_argument("--stats_path", default=None, help="per-scan stats CSV (default: df_stats_path of the config)")
parser.add_argument("--n_projections", type=int, default=None, help="frames per scan (default: config)")
parser.add_argument("--total_rotation_deg", type=float, default=None, help="first-to-last arc (default: config)")
parser.add_argument("--slices", type=int, nargs="+", default=[60, 100, 140, 170, 200],
                    help="half-res slice indices (central range 32..223; 128 is the chip gap)")
parser.add_argument("--n_masks", type=int, default=6, help="masks for the mean squared difference")
parser.add_argument("--mask_seed", type=int, default=1000, help="seed of the first mask (mask k: seed + k)")
parser.add_argument("--out_dir", default="./pair_difference")
args = parser.parse_args()

import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import tifffile


def save_scan(scan_id, g1, g2, diff, msd, fov, slices, out_dir):
    """
    g1, g2, diff, msd: [n_slices, H, W] float arrays for the chosen slices; fov: bool, same shape
    (non-zero pixels of the raw reconstruction; normalisation moves the zero corners off zero).
    Writes the ZCYX hyperstack.
    """
    stack = np.stack([g1, g2, diff, msd], axis=1).astype(np.float32)  # Z, C, Y, X
    v = g1[fov]
    d = diff[fov]
    ranges = (float(v.mean() - 3 * v.std()), float(v.mean() + 3 * v.std()),
              float(v.mean() - 3 * v.std()), float(v.mean() + 3 * v.std()),
              float(-3 * d.std()), float(3 * d.std()),
              0.0, float(np.percentile(msd[fov], 99)))
    names = ["g1", "g2", "g1 - g2 (mask 0)", f"mean (g1 - g2)^2 over masks"]
    labels = [f"z{z} | {scan_id} | {n}" for z in slices for n in names]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{scan_id}_pair_difference.tif")
    tifffile.imwrite(path, stack, imagej=True,
                     metadata={"axes": "ZCYX", "Labels": labels, "mode": "grayscale", "Ranges": ranges})
    print(f"saved {path}: Z = slices {list(slices)}, C = {names}")
    return path


def main():
    import h5py
    import torch
    import yaml
    from network import UNet
    from modelLightning import Neighbor2InverseSlice

    torch.set_float32_matmul_precision("medium")
    with open(args.trainparams, "r") as f:
        tp = yaml.safe_load(f)
    lp = tp["lightning_params"]
    if args.stats_path:
        lp["df_stats_path"] = args.stats_path
    if args.n_projections:
        lp["cone_n_projections"] = args.n_projections
    if args.total_rotation_deg is not None:
        lp["coneBeam_params"]["total_rotation_deg"] = args.total_rotation_deg
    print(f"stats {lp['df_stats_path']}, frames {lp['cone_n_projections']}, "
          f"arc {lp['coneBeam_params']['total_rotation_deg']} deg, masks {args.n_masks} (seeds {args.mask_seed}..)")

    # Model as 1_trainNeighbor2Inverse.py builds it; the network is not used (pairs only).
    model = Neighbor2InverseSlice(
        network=UNet(**tp["base_network"]["params"]), **lp,
        optimizer_algo=tp["optimizer_algo"], scheduler_algo=tp["scheduler_algo"],
        optimizer_params=tp["optimizer_params"], scheduler_params=tp["scheduler_params"],
        n_slicesPR=tp.get("dataset", {}).get("n_slicesPR"),
    ).to("cuda")

    df = pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True).drop_duplicates("scan_id")
    for _, row in df.iterrows():
        scan_id = str(row["scan_id"])
        det, src = float(row["detector_offset_m"]), float(row["source_offset_m"])
        smp = float(row.get("sample_offset_m", 0.0) or 0.0)
        sel = torch.tensor(args.slices, device=model.device)
        msd_sum, first = None, None
        per_mask = []
        for k in range(args.n_masks):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(args.mask_seed + k)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                with h5py.File(row["path_proj"], "r") as f:
                    reco1, reco2 = model.make_cone_pair(f["images"], det, src, generator=gen,
                                                        tag=f"{scan_id} pairdiff m{k}", sample_offset_m=smp)
                central = model.cone_slice_indices(reco1.shape[0])
                fov = reco1[central] != 0  # raw zeros outside the FOV (normalisation shifts them)
                g1c = model.normalize_cone(reco1[central], scan_id).float()
                g2c = model.normalize_cone(reco2[central], scan_id).float()
                dc = g1c - g2c
                per_mask.append(float((dc[fov] ** 2).mean()))
                g1s = model.normalize_cone(reco1[sel], scan_id).float()
                g2s = model.normalize_cone(reco2[sel], scan_id).float()
                d2 = ((g1s - g2s) ** 2).cpu().numpy()
            msd_sum = d2 if msd_sum is None else msd_sum + d2
            if first is None:
                first = (g1s.cpu().numpy(), g2s.cpu().numpy(), (g1s - g2s).cpu().numpy())
                fov_sel = (reco1[sel] != 0).cpu().numpy()
            del reco1, reco2, g1c, g2c, dc
            torch.cuda.empty_cache()
        print(f"{scan_id}: mean (g1 - g2)^2 in FOV, central slices, per mask: "
              + ", ".join(f"{v:.5f}" for v in per_mask) + f"; average {np.mean(per_mask):.5f}")
        save_scan(scan_id, *first, msd_sum / args.n_masks, fov_sel, args.slices, args.out_dir)


if __name__ == "__main__":
    main()
