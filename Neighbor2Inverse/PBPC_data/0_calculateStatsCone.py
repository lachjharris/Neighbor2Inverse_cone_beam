"""
Per-scan normalization statistics for cone-beam Neighbor2Inverse.

Cone-beam counterpart of 0_calculateStats.py: for every scan listed in the
given CSVs, reconstruct the FULL-RESOLUTION volume with this repo's own
pipeline (Paganin -> cone-beam FDK) and store whole-volume mean/std/min/max
in a stats CSV keyed by scan_id. As upstream, these full-resolution stats are
used to normalize both the Neighbor-subsampled reconstructions during training
and full-resolution reconstructions at inference.

Geometry and phase-retrieval settings are read from the training YAML, which
stores values for the Neighbor-subsampled projections. Full-resolution values
are derived by dividing pixel sizes by NEIGHBOR_FACTOR.
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trainparams", type=str, default="./trainparamsCone.yml")
parser.add_argument("--csvs", type=str, nargs="+", default=["./trainCone.csv", "./valCone.csv"])
parser.add_argument("--stats_path", type=str, default="./cone_stats.csv")
parser.add_argument("--gpu", type=int, default=None, help="GPU used by both torch and ASTRA")
parser.add_argument("--chunk_angles", type=int, default=64, help="projections phase-retrieved per GPU chunk")
parser.add_argument("--save_dir", type=str, default=None, help="optionally save each full-res volume as <scan_id>_reco_fullres.npy")
args = parser.parse_args()

# Must be set before torch/ASTRA initialise CUDA, so both use the same GPU.
if args.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import h5py
import torch
import yaml

from modelLightning import compute_paganin_batch
from astra_cone_array import astra_cone_from_array

NEIGHBOR_FACTOR = 2  # Neighbor subsampling halves rows and columns


def paganin_constants(PR_params):
    """Same definitions as Neighbor2InverseSlice.__init__."""
    wavelength = (12.398424 * 10**(-7)) / PR_params["energy"]  # mm
    mu = 4 * np.pi * PR_params["beta"] / wavelength            # 1/mm
    sigma = PR_params["delta"] / mu * PR_params["z"]           # mm^2
    return mu, sigma


def phase_retrieve_fullres(path_proj, mu, sigma, pixel_size_mm, batchsizePR, chunk_angles, device):
    """
    Paganin-retrieve every full-resolution projection, chunked over angles.
    Phase retrieval acts on each projection independently, so chunking gives
    the same result as processing the whole stack at once.
    Returns projected thickness in metres, shape [angle, row, col].
    """
    with h5py.File(path_proj, "r") as f:
        data = f["images"]
        n_angles, n_rows, n_cols = data.shape
        thickness_m = np.empty((n_angles, n_rows, n_cols), dtype=np.float32)

        for start in range(0, n_angles, chunk_angles):
            stop = min(start + chunk_angles, n_angles)
            proj = torch.from_numpy(data[start:stop].astype(np.float32)).to(device)
            with torch.no_grad():
                phase_mm = compute_paganin_batch(
                    proj, mu=mu, sigma=sigma,
                    pixel_size=pixel_size_mm, batch_size=batchsizePR,
                )[:, 0]
            thickness_m[start:stop] = phase_mm.cpu().numpy() * 1e-3
            del proj, phase_mm
            torch.cuda.empty_cache()

    return thickness_m


def update_stats_file(stats_path, new_rows):
    """Replace rows with the same scan_id, keep the rest (no duplicate rows)."""
    df_new = pd.DataFrame(new_rows)
    if os.path.exists(stats_path):
        df_old = pd.read_csv(stats_path)
        df_old = df_old[~df_old["scan_id"].astype(str).isin(df_new["scan_id"])]
        df_new = pd.concat((df_old, df_new), ignore_index=True)
    df_new.to_csv(stats_path, index=False)
    return df_new


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    with open(args.trainparams, "r") as f:
        lp = yaml.safe_load(f)["lightning_params"]

    PR_params = lp["PR_params"]
    mu, sigma = paganin_constants(PR_params)
    pixel_size_full_mm = PR_params["pixel_size"] / NEIGHBOR_FACTOR

    cone_params = dict(lp["coneBeam_params"])
    cone_params["true_pixel_size_m"] = cone_params["true_pixel_size_m"] / NEIGHBOR_FACTOR

    print(f"full-res Paganin pixel: {pixel_size_full_mm} mm, "
          f"full-res detector pixel: {cone_params['true_pixel_size_m']} m")

    df_scans = pd.concat([pd.read_csv(p) for p in args.csvs], ignore_index=True)
    if df_scans["scan_id"].duplicated().any():
        raise ValueError("Duplicate scan_id across the given CSVs")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for _, row in df_scans.iterrows():
        scan_id = str(row["scan_id"])
        print(f"\n=== {scan_id} ===")

        thickness_m = phase_retrieve_fullres(
            row["path_proj"], mu, sigma, pixel_size_full_mm,
            PR_params["batchsizePR"], args.chunk_angles, device,
        )
        print("phase-retrieved:", thickness_m.shape)

        reco = astra_cone_from_array(
            thickness_m,
            **cone_params,
            horizontal_detector_offset_m=float(row["detector_offset_m"]),
            horizontal_source_offset_m=float(row["source_offset_m"]),
        )
        del thickness_m
        print("reconstructed:", reco.shape)

        if args.save_dir:
            np.save(os.path.join(args.save_dir, f"{scan_id}_reco_fullres.npy"), reco)

        # Whole volume, as in upstream 0_calculateStats.py. float64 accumulation for accuracy.
        stats = {
            "scan_id": scan_id,
            "mean": float(reco.mean(dtype=np.float64)),
            "std": float(reco.std(dtype=np.float64)),
            "min": float(reco.min()),
            "max": float(reco.max()),
            "n_slices": reco.shape[0],
            "shape_y": reco.shape[1],
            "shape_x": reco.shape[2],
        }
        print(stats)
        del reco

        # Write after every scan so a crash doesn't lose finished scans.
        update_stats_file(args.stats_path, [stats])

    print(f"\nSaved stats to {args.stats_path}")


if __name__ == "__main__":
    main()
