"""
Combine a run's per-epoch prediction TIFFs (written by SavePredictionCallbackCone) into one
ImageJ hyperstack, so the prediction and residual can be scrolled through the epochs in Fiji.

Input, per run: <run_dir>/lightning_logs/version_*/predictions/{start|epoch}_<N>_<scan>_slice<k>_progress.tif
(float32 stacks: input, prediction, residual).
Output: <out_dir>/<run name>_<scan>_slice<k>_epochs.tif, axes TCYX:
    T = start (untrained network, prediction = input), epoch 0, 1, ..., in order
    C = 2 channels: prediction, residual (input - prediction); physical units, unscaled
Opens in grayscale mode with one slider for the epoch and one for the channel. Each plane is
labelled (e.g. "epoch 12 prediction"), and the display ranges are preset to the PNGs' windows:
prediction = input FOV mean +- 3 std, residual = +- 3 x the input's neighbour-difference noise
estimate (same rule as the callback, so the same window for every epoch and run on that slice).

Combined mode (--combine): one hyperstack for several runs on the same slice, axes TZCYX with
    T = epoch, Z = run (in the order given, named by --names), C = prediction, residual,
so one epoch can be held while flipping between runs (Z), or one run scrolled through training (T).
Every run must have the same input slice and the same epochs. --crop y0 y1 x0 x1 keeps only that
pixel region (rows y0..y1-1, cols x0..x1-1) to keep the file small; the display ranges are computed
from the cropped input.

EMA images (runs with ema_decays) are named ..._slice<k>_<tag>_progress.tif (tag e.g. ema0p996). The
per-run mode writes one stack per group (raw and each EMA). In --combine mode, --groups picks the group
per run: raw (default for every run) or an EMA tag; the same run may be given twice, e.g. raw vs EMA:
    python make_epoch_stacks.py --combine --runs <run> <run> --groups raw ema0p996 --names r1 r1_ema \
        --out_name s43_r1_raw_vs_ema

Usage (from PBPC_data/):
    python make_epoch_stacks.py --runs model_weights/Cone/<run1> model_weights/Cone/<run2> ...
    python make_epoch_stacks.py --combine --runs <run1> <run2> --names LNei r1 --out_name s43_LNei_vs_r1 \
        [--crop y0 y1 x0 x1]
"""
import argparse
import glob
import os
import re

import numpy as np
import tifffile

PATTERN = re.compile(r"^(start|epoch)_(\d+)_(.+)_progress\.tif$")
EMA_GROUP = re.compile(r"^(.+)_(ema[0-9p]+)$")  # <scan>_slice<k>_ema0p996 -> (<scan>_slice<k>, ema0p996)


def split_group(scan_slice):
    """<scan>_slice<k>[_<ema tag>] -> (<scan>_slice<k>, 'raw' or the EMA tag)."""
    m = EMA_GROUP.match(scan_slice)
    return (m.group(1), m.group(2)) if m else (scan_slice, "raw")


def select_group(pred_dir, groups, group):
    """The one scan/slice of a run in the requested group ('raw' or an EMA tag)."""
    found = {split_group(k)[0]: v for k, v in groups.items() if split_group(k)[1] == group}
    if len(found) != 1:
        raise ValueError(f"{pred_dir}: expected one scan/slice in group '{group}', found {sorted(found)} "
                         f"(all groups: {sorted(groups)})")
    return next(iter(found.items()))


def noise_estimate(img, fov):
    """Same as SavePredictionCallbackCone.noise_estimate: std of horizontal neighbour differences / sqrt(2)."""
    both = fov[:, 1:] & fov[:, :-1]
    d = (img[:, 1:] - img[:, :-1])[both]
    return float(d.std() / np.sqrt(2))


def collect(run_dir):
    """{scan_slice: [(order key, label, path), ...]} for the newest version folder of a run."""
    versions = sorted(glob.glob(os.path.join(run_dir, "lightning_logs", "version_*")),
                      key=lambda p: int(p.rsplit("_", 1)[-1]))
    if not versions:
        raise FileNotFoundError(f"no lightning_logs/version_* in {run_dir}")
    pred_dir = os.path.join(versions[-1], "predictions")
    groups = {}
    for path in glob.glob(os.path.join(pred_dir, "*_progress.tif")):
        m = PATTERN.match(os.path.basename(path))
        if not m:
            continue
        kind, number, scan_slice = m.group(1), int(m.group(2)), m.group(3)
        key = (-1, 0) if kind == "start" else (number, 1)
        label = "start" if kind == "start" else f"epoch {number}"
        groups.setdefault(scan_slice, []).append((key, label, path))
    if not groups:
        raise FileNotFoundError(f"no *_progress.tif files in {pred_dir}")
    return pred_dir, {k: sorted(v) for k, v in groups.items()}


def build(run_dir, out_dir):
    pred_dir, groups = collect(run_dir)
    run_name = os.path.basename(os.path.normpath(run_dir))
    for scan_slice, items in groups.items():
        first = tifffile.imread(items[0][2])
        if first.ndim != 3 or first.shape[0] != 3:
            raise ValueError(f"{items[0][2]}: expected 3 slices (input, prediction, residual), got {first.shape}")
        inpt = first[0]
        fov = inpt != 0
        vals = inpt[fov]
        p_lo, p_hi = float(vals.mean() - 3 * vals.std()), float(vals.mean() + 3 * vals.std())
        r = 3 * noise_estimate(inpt, fov)

        stack = np.empty((len(items), 2) + inpt.shape, dtype=np.float32)
        labels = []
        for t, (_, label, path) in enumerate(items):
            a = tifffile.imread(path)
            if not np.array_equal(a[0], inpt):
                raise ValueError(f"{path}: input slice differs from {items[0][2]}")
            stack[t, 0], stack[t, 1] = a[1], a[2]
            labels += [f"{label} prediction", f"{label} residual"]

        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"{run_name}_{scan_slice}_epochs.tif")
        tifffile.imwrite(out, stack, imagej=True,
                         metadata={"axes": "TCYX", "Labels": labels, "mode": "grayscale",
                                   "Ranges": (p_lo, p_hi, -r, r)})
        print(f"{run_name}: {len(items)} time points ({items[0][1]} .. {items[-1][1]}) from {pred_dir}")
        print(f"    wrote {out}  shape {stack.shape}; display prediction {p_lo:.4g}..{p_hi:.4g}, residual +-{r:.4g}")


def build_combined(run_dirs, names, out_dir, out_name, crop=None, group_names=None):
    """One TZCYX hyperstack: T = epoch, Z = run, C = prediction / residual (see the module docstring)."""
    if len(names) != len(run_dirs):
        raise ValueError("--names needs one name per run")
    group_names = group_names or ["raw"] * len(run_dirs)
    if len(group_names) != len(run_dirs):
        raise ValueError("--groups needs one entry per run (raw or an EMA tag such as ema0p996)")
    per_run = []
    for run_dir, group in zip(run_dirs, group_names):
        pred_dir, groups = collect(run_dir)
        scan_slice, items = select_group(pred_dir, groups, group)
        per_run.append((pred_dir, scan_slice, items))
    scan_slice = per_run[0][1]
    times = [label for _, label, _ in per_run[0][2]]
    for (pred_dir, ss, items), name in zip(per_run, names):
        if ss != scan_slice:
            raise ValueError(f"{name}: slice {ss} differs from {scan_slice}")
        if [label for _, label, _ in items] != times:
            raise ValueError(f"{name}: epochs differ from {names[0]} ({len(items)} vs {len(times)} time points)")

    ys = slice(crop[0], crop[1]) if crop else slice(None)
    xs = slice(crop[2], crop[3]) if crop else slice(None)
    inpt = tifffile.imread(per_run[0][2][0][2])[0]
    inpt_c = inpt[ys, xs]
    fov = inpt_c != 0
    vals = inpt_c[fov]
    p_lo, p_hi = float(vals.mean() - 3 * vals.std()), float(vals.mean() + 3 * vals.std())
    r = 3 * noise_estimate(inpt_c, fov)

    stack = np.empty((len(times), len(names), 2) + inpt_c.shape, dtype=np.float32)
    labels = []
    for t, label in enumerate(times):
        for zi, ((_, _, items), name) in enumerate(zip(per_run, names)):
            a = tifffile.imread(items[t][2])
            if not np.array_equal(a[0], inpt):
                raise ValueError(f"{items[t][2]}: input slice differs from {per_run[0][2][0][2]}")
            stack[t, zi, 0], stack[t, zi, 1] = a[1][ys, xs], a[2][ys, xs]
            labels += [f"{label} | {name} | prediction", f"{label} | {name} | residual"]

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{out_name}_{scan_slice}_epochs.tif")
    tifffile.imwrite(out, stack, imagej=True,
                     metadata={"axes": "TZCYX", "Labels": labels, "mode": "grayscale",
                               "Ranges": (p_lo, p_hi, -r, r)})
    print(f"wrote {out}: T = {len(times)} time points ({times[0]} .. {times[-1]}), Z = runs {names}, "
          f"C = prediction, residual; {stack.nbytes / 1e9:.2f} GB; "
          f"display prediction {p_lo:.4g}..{p_hi:.4g}, residual +-{r:.4g}"
          + (f"; crop rows {crop[0]}..{crop[1] - 1}, cols {crop[2]}..{crop[3] - 1}" if crop else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run folders, e.g. model_weights/Cone/<name>")
    p.add_argument("--out_dir", default="./epoch_stacks")
    p.add_argument("--combine", action="store_true", help="one hyperstack for all --runs (Z = run)")
    p.add_argument("--names", nargs="+", help="--combine: one short name per run")
    p.add_argument("--out_name", help="--combine: output file name prefix")
    p.add_argument("--crop", nargs=4, type=int, metavar=("Y0", "Y1", "X0", "X1"),
                   help="--combine: keep rows Y0..Y1-1 and cols X0..X1-1")
    p.add_argument("--groups", nargs="+",
                   help="--combine: per run, raw (default) or an EMA tag such as ema0p996")
    args = p.parse_args()
    if args.combine:
        if not args.names or not args.out_name:
            p.error("--combine needs --names and --out_name")
        build_combined(args.runs, args.names, args.out_dir, args.out_name, args.crop, args.groups)
    else:
        if args.groups:
            p.error("--groups is for --combine (the per-run mode writes every group)")
        for run_dir in args.runs:
            build(run_dir, args.out_dir)


if __name__ == "__main__":
    main()
