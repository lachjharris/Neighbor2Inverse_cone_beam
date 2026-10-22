"""
Look at the cone L_reg correction exp_diff on a validation scan, exactly as validation
computes it: model built as in 1_trainNeighbor2Inverse.py, checkpoint weights, the fixed
validation mask (val_mask_seed + val scan index), make_cone_pair + cone_reg_correction,
bf16 autocast as Lightning's "bf16-mixed".

Computes the correction in both modes of the model's cone_reg_correction:
  "upstream": subsample the forward-projected thickness directly (Neighbor2Inverse Eq. 7)
  "mirrored": forward projection -> TIE (inverse Paganin) -> subsample -> Paganin -> FDK,
              the same operator order as the training pair (used for training with
              cone_reg_mode: 'mirrored')

Prints:
  - L_Nei and L_reg recomputed on the central slices (should match the checkpoint's
    val_loss1 and val_loss - val_loss1 for that epoch)
  - size of exp_diff, and where its large values are: radius in the FOV disk, slice
    (horizontal chip gap at half-res slice 128), association with edges of f(g1),
    correlation with the pair's own noise difference g1 - g2
Saves (float32 TIFF, ImageJ slice labels) in --out_dir:
  exp_diff_central.tif          all central slices of exp_diff ("upstream", normalised units)
  exp_diff_mirrored_central.tif the same for the "mirrored" correction
  slice<k>_panels.tif           g1, f(g1), g1 - f(g1), g2, f(g1) - g2, exp_diff, exp_diff mirrored
  slice<k>_fy_fullres.tif       noisy full-res slice 2k, f(y) at 2k and 2k+1 (normalised)
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--trainparams", type=str, default="./trainparamsCone.yml")
parser.add_argument("--val_index", type=int, default=0, help="row of dataset_val CSV (= validation batch_idx)")
parser.add_argument("--slices", type=int, nargs="+", default=[100, 128, 170], help="half-res slice indices to save as panels")
parser.add_argument("--large", type=float, default=1.0, help="|exp_diff| threshold for the location statistics")
parser.add_argument("--out_dir", type=str, default="./correction_check")
args = parser.parse_args()

import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import h5py
import tifffile
import torch
import yaml

from network import UNet
from modelLightning import Neighbor2InverseSlice, pad_to_divisible, unpad_from_divisible
from astra_cone_array import _create_circular_mask

torch.set_float32_matmul_precision("medium")
CHIP_GAP_HALF_SLICE = 128  # horizontal chip gap: full-res row 256


def save_stack(path, frames, labels):
    tifffile.imwrite(path, np.stack([np.asarray(f, dtype=np.float32) for f in frames]),
                     imagej=True, metadata={"Labels": labels})
    print("saved", path)


def fit(e, d):
    """Least-squares slope of d on e, correlation, and rms of e (1D arrays)."""
    e = e.astype(np.float64); d = d.astype(np.float64)
    ec, dc = e - e.mean(), d - d.mean()
    return float((ec * dc).sum() / (ec * ec).sum()), float(np.corrcoef(e, d)[0, 1]), float(np.sqrt(np.mean(e**2)))



def main():
    with open(args.trainparams, "r") as f:
        tp = yaml.safe_load(f)

    # Model exactly as 1_trainNeighbor2Inverse.py builds it
    model = Neighbor2InverseSlice(
        network=UNet(**tp["base_network"]["params"]),
        **tp["lightning_params"],
        optimizer_algo=tp["optimizer_algo"], scheduler_algo=tp["scheduler_algo"],
        optimizer_params=tp["optimizer_params"], scheduler_params=tp["scheduler_params"],
        n_slicesPR=tp.get("dataset", {}).get("n_slicesPR"),
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to("cuda")
    print(f"loaded {args.ckpt} (epoch {ckpt.get('epoch')})")

    row = pd.read_csv(tp["dataset_val"]["df_path"]).iloc[args.val_index]
    scan_id = str(row["scan_id"])
    det, src = float(row["detector_offset_m"]), float(row["source_offset_m"])
    smp = float(row.get("sample_offset_m", 0.0))
    mean, std = model.cone_scan_stats(scan_id)

    captured = {}
    denoise = model.denoise_fullres_cone
    def denoise_and_keep(volume, sid):
        out = denoise(volume, sid)
        captured["fy"] = out
        return out
    model.denoise_fullres_cone = denoise_and_keep

    gen = torch.Generator(device="cpu")
    gen.manual_seed(model.val_mask_seed + args.val_index)  # as validation_step_cone
    volume = model.load_fullres_cone(row["path_reco_fullres"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        with h5py.File(row["path_proj"], "r") as f:
            reco1, reco2, masks = model.make_cone_pair(
                f["images"], det, src, generator=gen, tag=f"{scan_id} viz", return_masks=True,
                sample_offset_m=smp)
        model.cone_reg_mode = "upstream"
        exp_diff = model.cone_reg_correction(volume, scan_id, masks, det, src, tag=f"{scan_id} upstream",
                                             sample_offset_m=smp)
        model.cone_reg_mode = "mirrored"
        exp_diff_m = model.cone_reg_correction(volume, scan_id, masks, det, src, tag=f"{scan_id} mirrored",
                                               sample_offset_m=smp)

        idx = model.cone_slice_indices(reco1.shape[0])
        g1 = model.normalize_cone(reco1[idx], scan_id)
        g2 = model.normalize_cone(reco2[idx], scan_id)
        e = exp_diff[idx].float()
        out = torch.empty_like(g1)
        for s in range(0, g1.shape[0], model.val_batch_size):
            x, pad = pad_to_divisible(g1[s:s + model.val_batch_size].unsqueeze(1), 32)
            out[s:s + model.val_batch_size] = unpad_from_divisible(model(x), pad)[:, 0].float()

    e2 = exp_diff_m[idx].float().cpu().numpy()
    g1, g2, fg1, e = (t.float().cpu().numpy() for t in (g1, g2, out, e))
    central = idx.cpu().numpy()
    del reco1, reco2, exp_diff, exp_diff_m, out
    torch.cuda.empty_cache()

    # ----- reproduce validation -----
    diff = fg1 - g2
    print(f"\nrecomputed on {len(central)} central slices: L_Nei={np.mean(diff**2):.6f} "
          f"L_reg={np.mean((diff - e)**2):.6f} with the upstream correction "
          f"(compare val_loss1 and val_loss - val_loss1 of an upstream-mode run)")

    # ----- current vs mirrored correction: does it predict the pair difference? -----
    n_ = g1.shape[-1]
    disk_ = _create_circular_mask((n_, n_))
    d_pair = (g1 - g2)[:, disk_].ravel()
    d_net = diff[:, disk_].ravel()
    print("\ncorrection      rms(FOV)  slope(g1-g2 ~ e)  corr   slope(f(g1)-g2 ~ e)  corr   L_reg")
    print("(a correction equal to the true signal gap gives slopes ~1 and L_reg < L_Nei)")
    for name, ee in (("upstream", e), ("mirrored", e2)):
        eflat = ee[:, disk_].ravel()
        s1, c1, rms = fit(eflat, d_pair)
        s2, c2, _ = fit(eflat, d_net)
        print(f"{name}        {rms:8.5f}  {s1:16.3f}  {c1:6.3f}  {s2:18.3f}  {c2:6.3f}  {np.mean((diff - ee)**2):.6f}")
    print(f"no correction (L_Nei)                                                      {np.mean(diff**2):.6f}")

    # ----- size and location of exp_diff -----
    n = g1.shape[-1]
    disk = _create_circular_mask((n, n))
    yy, xx = np.mgrid[:n, :n]
    radius = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2) / (n / 2)
    ein = e[:, disk]
    a = np.abs(ein)
    print(f"\nexp_diff inside FOV: rms={np.sqrt(np.mean(ein**2)):.5f}; |e| percentiles "
          + ", ".join(f"p{p}={np.percentile(a, p):.4f}" for p in (50, 90, 99, 99.9)) + f", max={a.max():.4f}")

    large = np.abs(e) > args.large
    large &= disk[None]
    n_large = int(large.sum())
    print(f"|e| > {args.large}: {n_large} voxels ({n_large / ein.size:.2e} of FOV)")

    if n_large:
        edges = [0, 0.5, 0.8, 0.9, 0.95, 1.0]
        print("by radius (r/R)   share of FOV   share of large   enrichment")
        r_fov = np.broadcast_to(radius[disk], ein.shape).ravel()
        r_large = np.broadcast_to(radius, e.shape)[large]
        for lo, hi in zip(edges[:-1], edges[1:]):
            f_fov = np.mean((r_fov >= lo) & (r_fov < hi))
            f_lrg = np.mean((r_large >= lo) & (r_large < hi))
            print(f"  {lo:4.2f}-{hi:4.2f}       {f_fov:10.3f}   {f_lrg:14.3f}   {f_lrg / max(f_fov, 1e-12):10.2f}")
        per_slice = large.sum(axis=(1, 2))
        top = np.argsort(per_slice)[::-1][:5]
        gap = np.where(central == CHIP_GAP_HALF_SLICE)[0]
        print("slices with most large values: " + ", ".join(f"{central[i]} ({per_slice[i]})" for i in top)
              + f"; median per slice {int(np.median(per_slice))}"
              + (f"; chip-gap slice {CHIP_GAP_HALF_SLICE}: {per_slice[gap[0]]}" if len(gap) else ""))

    # association with edges of f(g1): in-plane gradient magnitude
    gy, gx = np.gradient(fg1, axis=(1, 2))
    gmag = np.hypot(gy, gx)[:, disk]
    hi_g = gmag > np.percentile(gmag, 95)
    lo_g = gmag < np.percentile(gmag, 50)
    print(f"\nmean |e| at strongest 5% edges of f(g1): {a[hi_g].mean():.5f}; "
          f"at weakest 50%: {a[lo_g].mean():.5f}; ratio {a[hi_g].mean() / a[lo_g].mean():.2f}")
    print(f"corr(|e|, edge strength) = {np.corrcoef(a.ravel(), gmag.ravel())[0, 1]:.3f}")
    nd = (g1 - g2)[:, disk]
    print(f"corr(e, g1 - g2) = {np.corrcoef(ein.ravel(), nd.ravel())[0, 1]:.3f}  "
          f"(share of the pair's own noise difference carried by the correction)")

    # ----- images -----
    os.makedirs(args.out_dir, exist_ok=True)
    save_stack(os.path.join(args.out_dir, "exp_diff_central.tif"), list(e),
               [f"half-res slice {k}" for k in central])
    save_stack(os.path.join(args.out_dir, "exp_diff_mirrored_central.tif"), list(e2),
               [f"half-res slice {k} (mirrored)" for k in central])
    fy = captured["fy"]
    for k in args.slices:
        if k not in central:
            print(f"slice {k} not in the central range {central[0]}..{central[-1]}, skipped")
            continue
        i = int(np.where(central == k)[0][0])
        save_stack(os.path.join(args.out_dir, f"slice{k}_panels.tif"),
                   [g1[i], fg1[i], g1[i] - fg1[i], g2[i], fg1[i] - g2[i], e[i], e2[i]],
                   ["g1 (norm)", "f(g1)", "g1 - f(g1)", "g2 (norm)", "f(g1) - g2", "exp_diff", "exp_diff mirrored"])
        save_stack(os.path.join(args.out_dir, f"slice{k}_fy_fullres.tif"),
                   [(volume[2 * k] - mean) / std, (fy[2 * k] - mean) / std, (fy[2 * k + 1] - mean) / std],
                   [f"noisy full-res {2 * k}", f"f(y) full-res {2 * k}", f"f(y) full-res {2 * k + 1}"])


if __name__ == "__main__":
    main()
