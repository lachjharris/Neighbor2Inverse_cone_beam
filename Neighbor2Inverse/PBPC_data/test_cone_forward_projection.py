"""
Round-trip check of astra_cone_forward_from_volume (cone-beam forward projection)
on a saved full-resolution reconstruction, before it is used in L_reg.

1. Forward project the full-res volume V with the full-res geometry (55 um pixels;
   CONE_N_PROJECTIONS frames, CONE_DETECTOR_ROWS rows, as training and 0_calculateStatsCone.py):
   time, peak GPU memory, fraction of projection values <= 0.
2. Full-res round trip: FDK(FP(V)) vs V. Checks geometry and scale (slope ~1).
3. Half-res scale check: subsample FP(V) with two horizontally neighbouring pixels of
   every 2x2 cell (fixed, not random), FDK each at half resolution (training geometry),
   compare against V binned 2x2x2. Also prints C1 - C2 normalised by the scan std,
   a rough preview of the size of L_reg's correction term (horizontal neighbours only).

Comparisons use voxels inside the reconstruction disk and the central slices
(CONE_SLICE_MARGIN = 32 half-res = 64 full-res slices excluded at each end).
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trainparams", type=str, default="./trainparamsCone.yml")
parser.add_argument("--csv", type=str, default="./valCone.csv")
parser.add_argument("--scan_id", type=str, default="Mouse_10")
parser.add_argument("--stats_path", type=str, default="./cone_stats.csv")
parser.add_argument("--gpu", type=int, default=None, help="GPU used by both torch and ASTRA")
args = parser.parse_args()

# Must be set before torch/ASTRA initialise CUDA, so both use the same GPU.
if args.gpu is not None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import sys
sys.path.append("../")
import time
import resource
import threading
import numpy as np
import pandas as pd
import h5py
import torch
import yaml

from astra_cone_array import astra_cone_from_array, astra_cone_forward_from_volume, _create_circular_mask
from modelLightning import Neighbor2InverseSlice

NEIGHBOR_FACTOR = 2
N_PROJECTIONS = Neighbor2InverseSlice.CONE_N_PROJECTIONS  # same frames and rows as training
DETECTOR_ROWS = Neighbor2InverseSlice.CONE_DETECTOR_ROWS  # and 0_calculateStatsCone.py
SLICE_MARGIN_HALF = 32  # Neighbor2InverseSlice.CONE_SLICE_MARGIN
CHUNK = 16              # slices per chunk when accumulating statistics
FP_KEYS = ("total_rotation_deg", "source_origin_m", "origin_det_m", "true_pixel_size_m", "rotation_offset_deg")


def host_peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # ru_maxrss is in kB on Linux


class GpuPeak:
    """Polls device-wide used GPU memory (includes ASTRA's allocations)."""

    def __enter__(self):
        free, total = torch.cuda.mem_get_info()
        self.base = total - free
        self.peak = self.base
        self.stop = False
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()
        return self

    def _poll(self):
        while not self.stop:
            free, total = torch.cuda.mem_get_info()
            self.peak = max(self.peak, total - free)
            time.sleep(0.02)

    def __exit__(self, *exc):
        self.stop = True
        self.thread.join()

    def report(self):
        return f"GPU peak used {self.peak / 1024**3:.2f} GiB (before: {self.base / 1024**3:.2f} GiB)"


def compare(ref, test, z_range, disk):
    """Agreement of test with ref over slices z_range, voxels inside disk (float64 accumulation)."""
    n = sa = sb = saa = sbb = sab = sdd = 0.0
    for z0 in range(z_range[0], z_range[1], CHUNK):
        z1 = min(z0 + CHUNK, z_range[1])
        a = ref[z0:z1][:, disk].astype(np.float64)
        b = test[z0:z1][:, disk].astype(np.float64)
        n += a.size
        sa += a.sum(); sb += b.sum()
        saa += (a * a).sum(); sbb += (b * b).sum(); sab += (a * b).sum()
        sdd += ((b - a) ** 2).sum()
    ma, mb = sa / n, sb / n
    va, vb = saa / n - ma**2, sbb / n - mb**2
    cov = sab / n - ma * mb
    return (
        f"mean ref={ma:.6g} test={mb:.6g} ratio={mb / ma:.5f} | "
        f"slope(test~ref)={cov / va:.5f} corr={cov / np.sqrt(va * vb):.5f} | "
        f"rms(test-ref)/std(ref)={np.sqrt(sdd / n) / np.sqrt(va):.4f}"
    )


def main():
    with open(args.trainparams, "r") as f:
        cone_params = dict(yaml.safe_load(f)["lightning_params"]["coneBeam_params"])

    row = pd.read_csv(args.csv).set_index("scan_id").loc[args.scan_id]
    offsets = dict(
        horizontal_detector_offset_m=float(row["detector_offset_m"]),
        horizontal_source_offset_m=float(row["source_offset_m"]),
    )
    stats = pd.read_csv(args.stats_path).set_index("scan_id").loc[args.scan_id]
    scan_std = float(stats["std"])

    with h5py.File(row["path_proj"], "r") as f:
        h5_shape = f["images"].shape
    print(f"{args.scan_id}: projections in HDF5 {h5_shape}")
    if h5_shape[0] != N_PROJECTIONS or h5_shape[1] < DETECTOR_ROWS:
        raise ValueError(f"expected {N_PROJECTIONS} frames and at least {DETECTOR_ROWS} rows")
    n_angles = N_PROJECTIONS

    # Full-res geometry, as in 0_calculateStatsCone.py
    full_params = dict(cone_params)
    full_params["true_pixel_size_m"] = cone_params["true_pixel_size_m"] / NEIGHBOR_FACTOR
    half_params = dict(cone_params)  # training geometry (subsampled projections)

    torch.cuda.init()
    torch.zeros(1, device="cuda")  # create the CUDA context before measuring

    # ----- 1. forward projection -----
    V = np.load(row["path_reco_fullres"])
    H, W = V.shape[0], V.shape[2]
    print(f"full-res volume {V.shape} {V.dtype}, host peak so far {host_peak_gb():.1f} GiB")
    if H != DETECTOR_ROWS or H % 2 or W % 2:
        raise ValueError(f"full-res volume {V.shape}: expected {DETECTOR_ROWS} slices and even sizes "
                         "(regenerate with 0_calculateStatsCone.py --save_dir)")

    with GpuPeak() as gp:
        t0 = time.perf_counter()
        P = astra_cone_forward_from_volume(
            V, num_projections=n_angles, detector_rows=H, detector_cols=W,
            **{k: full_params[k] for k in FP_KEYS if k in full_params}, **offsets,
        )
        t_fp = time.perf_counter() - t0
    print(f"[1] FP: {P.shape}, {t_fp:.1f} s, {gp.report()}, host peak {host_peak_gb():.1f} GiB")
    print(f"    FP values (m): min={P.min():.4g} max={P.max():.4g} mean={P.mean():.4g}, "
          f"fraction <= 0: {(P <= 0).mean():.4f} (FDK clips these to eps)")

    disk_full = _create_circular_mask((W, W))
    z_full = (NEIGHBOR_FACTOR * SLICE_MARGIN_HALF, H - NEIGHBOR_FACTOR * SLICE_MARGIN_HALF)

    # ----- 2. full-res round trip -----
    with GpuPeak() as gp:
        t0 = time.perf_counter()
        V2 = astra_cone_from_array(P, **full_params, **offsets)
        t_fdk = time.perf_counter() - t0
    print(f"[2] full-res FDK(FP(V)): {V2.shape}, {t_fdk:.1f} s, {gp.report()}, host peak {host_peak_gb():.1f} GiB")
    print(f"    slices {z_full[0]}:{z_full[1]}, inside disk: {compare(V, V2, z_full, disk_full)}")
    del V2

    # ----- 3. half-res scale check -----
    g1 = np.ascontiguousarray(P[:, 0::2, 0::2])  # top-left of each 2x2 cell
    g2 = np.ascontiguousarray(P[:, 0::2, 1::2])  # top-right (horizontal neighbour)
    del P
    t0 = time.perf_counter()
    C1 = astra_cone_from_array(g1, **half_params, **offsets)
    del g1
    C2 = astra_cone_from_array(g2, **half_params, **offsets)
    del g2
    t_half = time.perf_counter() - t0
    print(f"[3] half-res FDK pair: {C1.shape}, {t_half:.1f} s for both, host peak {host_peak_gb():.1f} GiB")

    Hh, Wh = C1.shape[0], C1.shape[2]
    Vb = V[: 2 * Hh, : 2 * Wh, : 2 * Wh].reshape(Hh, 2, Wh, 2, Wh, 2).mean(axis=(1, 3, 5), dtype=np.float64).astype(np.float32)
    del V
    disk_half = _create_circular_mask((Wh, Wh))
    z_half = (SLICE_MARGIN_HALF, Hh - SLICE_MARGIN_HALF)
    print(f"    C1 vs V binned 2x2x2: {compare(Vb, C1, z_half, disk_half)}")
    print(f"    C2 vs V binned 2x2x2: {compare(Vb, C2, z_half, disk_half)}")

    d = (C1[z_half[0]:z_half[1]][:, disk_half].astype(np.float64)
         - C2[z_half[0]:z_half[1]][:, disk_half].astype(np.float64)) / scan_std
    print(f"    (C1 - C2) / scan std: mean={d.mean():+.5f} mean_square={np.mean(d**2):.6f} "
          f"max_abs={np.abs(d).max():.4f}")


if __name__ == "__main__":
    main()
