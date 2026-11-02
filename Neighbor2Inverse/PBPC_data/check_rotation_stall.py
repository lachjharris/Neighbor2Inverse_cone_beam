"""
Check each scan's projections for a rotation stall at the end of the acquisition: the stage
stops turning while frames keep being recorded, so the last frames all show the same view.

Reads only the first --head and last --tail frames of the HDF5 'images' dataset
([frame, row, col]), block-averaged by --bin, so memory and I/O stay small.

Similarity measure: Pearson correlation of the horizontal gradient of each (binned) frame with
the template T = mean of the last --template frames. The gradient removes overall intensity
level and slow drift, so the measure follows edges, which move as the sample rotates.
  - stuck frames all sit on a plateau (limited only by noise);
  - frames taken while rotating fall off the plateau as their angle moves away from T.
Stall onset = first tail frame from which every later frame stays on the plateau (plateau level
from frames --template+1 .. 2*--template from the end, which are not part of T). Frames within
about a degree of the stuck view look identical to it at this noise level, so the onset tends to
come out a few frames early (about 5 in synthetic tests). A stall is reported only if those
plateau frames are flat (no trend towards the template) and the plateau is longer than
2 x --template frames.

Which scenario (compares the stuck view with the first frame in the file):
  1. stopped short: rotation was normal and stopped ~(n_stuck x step) before a full turn ->
     the first frame looks about as different from T as a frame n_stuck frames before onset.
  2. finished early: the stage turned faster than assumed, completed 360 deg at the onset and
     stopped -> T is the 0/360 deg view, so the first frame looks almost like the last moving
     frame (both are adjacent to it in angle).
Frame indices are file indices (0-based); with the _from1.h5 files, original frame = index + 1.

Usage (from PBPC_data/):
    python check_rotation_stall.py --csvs trainCone_noisy.csv valCone_noisy.csv <test csv>
Writes rotation_stall_<scan_id>.png per scan and prints a summary.
"""
import argparse

import h5py
import numpy as np
import pandas as pd


def read_binned(data, idx, bin_factor, chunk=16):
    """Frames data[idx] block-averaged by bin_factor in rows and cols, float32 [n, r, c]."""
    n_rows, n_cols = data.shape[1], data.shape[2]
    r = (n_rows // bin_factor) * bin_factor
    c = (n_cols // bin_factor) * bin_factor
    out = []
    idx = list(idx)
    for start in range(0, len(idx), chunk):
        sel = idx[start:start + chunk]
        block = data[sel[0]:sel[-1] + 1, :r, :c].astype(np.float32)  # sel is contiguous
        block = block.reshape(len(sel), r // bin_factor, bin_factor, c // bin_factor, bin_factor).mean(axis=(2, 4))
        out.append(block)
    return np.concatenate(out)


def gradient_features(frames):
    """Horizontal gradient of each frame, zero-mean and unit-norm, flattened: [n, pixels]."""
    g = np.diff(frames, axis=2).reshape(len(frames), -1).astype(np.float64)
    g -= g.mean(axis=1, keepdims=True)
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
    return g


def check_scan(path_proj, scan_id, head, tail, template, bin_factor):
    with h5py.File(path_proj, "r") as f:
        data = f["images"]
        n = data.shape[0]
        print(f"\n== {scan_id}: images {data.shape} {data.dtype}  ({path_proj})")
        tail = min(tail, n - head)
        head_idx = range(0, head)
        tail_idx = range(n - tail, n)
        head_frames = read_binned(data, head_idx, bin_factor)
        tail_frames = read_binned(data, tail_idx, bin_factor)

    gh = gradient_features(head_frames)
    gt = gradient_features(tail_frames)

    T = gt[-template:].mean(axis=0)
    T -= T.mean()
    T /= np.linalg.norm(T) + 1e-12
    c_tail = gt @ T
    c_head = gh @ T

    # Stuck frames not in the template give the plateau level. They must be flat: at a real stall
    # their correlation with T varies only by noise; without a stall it keeps rising towards the
    # template frames (trend). Noise level of c from frame-to-frame changes (a trend barely affects
    # those): std(diff(c)) / sqrt(2). trend_ratio = std(ref) / noise is ~1 when flat.
    ref = c_tail[-2 * template:-template]
    noise_c = float(np.std(np.diff(ref)) / np.sqrt(2))
    plateau, spread = float(np.median(ref)), float(ref.std())
    trend_ratio = spread / max(noise_c, 1e-12)
    tol = 3 * max(spread, noise_c, 1e-6)
    on = c_tail >= plateau - tol
    onset_local = len(on)  # first index from which every later frame is on the plateau
    for k in range(len(on) - 1, -1, -1):
        if on[k]:
            onset_local = k
        else:
            break
    onset = n - tail + onset_local
    n_stuck = n - onset
    is_stall = trend_ratio < 2.0 and n_stuck > 2 * template
    result = dict(scan_id=scan_id, n_frames=n, onset=onset, n_stuck=n_stuck,
                  plateau=plateau, spread=spread, stall=bool(is_stall), trend_ratio=trend_ratio)
    print(f"plateau corr with stuck view {plateau:.4f}, std {spread:.4f} vs noise {noise_c:.4f} "
          f"(trend ratio {trend_ratio:.2f}; ~1 = flat)")
    if not is_stall:
        print(f"NO CLEAR STALL: the frames just before the template are not a flat plateau "
              f"(trend ratio {trend_ratio:.2f}, plateau {n_stuck} frames); the end of the scan looks like normal rotation.")
    if not is_stall:
        pass
    elif onset_local <= 2:
        print(f"WARNING: every tail frame is on the plateau; the stall may be longer than --tail ({tail}). "
              f"Rerun with a larger --tail.")
    else:
        last_moving = float(c_tail[onset_local - 1])
        back = onset_local - n_stuck  # a frame as far before onset as the stall is long
        c_back = float(c_tail[back]) if back >= 0 else float("nan")
        result.update(last_moving=last_moving, first_frame=float(c_head[0]), n_stuck_back=c_back)
        print(f"stall onset ~ file index {onset} (original frame {onset + 1}): last ~{n_stuck} frames stuck "
              f"(likely a few frames early: frames within ~1 deg of the stuck view can look identical to it)")
        print(f"corr with stuck view: last moving frame (index {onset - 1}) {last_moving:.4f}; "
              f"first frame in file (index 0) {c_head[0]:.4f}; "
              f"frame {n_stuck} before onset (index {onset - n_stuck}) {c_back:.4f}")
        print("  scenario 1 (stopped short) -> first-frame corr close to the 'frames before onset' value")
        print("  scenario 2 (finished early) -> first-frame corr close to the last-moving-frame value")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [3, 1]})
        axes[0].plot(np.arange(n - tail, n), c_tail, ".-", ms=3)
        axes[0].axhspan(plateau - tol, plateau + tol, color="0.85", label="plateau band")
        axes[0].axvline(onset, color="r", ls="--", label=f"onset ~{onset}")
        axes[0].set_xlabel("file frame index")
        axes[0].set_ylabel("corr with stuck view")
        axes[0].set_title(f"{scan_id}: last {tail} frames (stall: {bool(is_stall)})")
        axes[0].legend()
        axes[1].plot(np.arange(head), c_head, ".-", ms=3, color="C1")
        axes[1].axhspan(plateau - tol, plateau + tol, color="0.85")
        axes[1].set_xlabel("file frame index")
        axes[1].set_title(f"first {head} frames")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"rotation_stall_{scan_id}.png", dpi=120)
        plt.close(fig)
        print(f"saved rotation_stall_{scan_id}.png")
    except ImportError:
        pass
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csvs", nargs="+", required=True, help="scan CSVs with path_proj and scan_id columns")
    p.add_argument("--head", type=int, default=40, help="frames read from the start")
    p.add_argument("--tail", type=int, default=300, help="frames read from the end")
    p.add_argument("--template", type=int, default=10, help="last frames averaged into the stuck-view template")
    p.add_argument("--bin", type=int, default=4, help="block-average factor in rows and cols")
    args = p.parse_args()

    df = pd.concat([pd.read_csv(c) for c in args.csvs], ignore_index=True).drop_duplicates("scan_id")
    results = [check_scan(r["path_proj"], str(r["scan_id"]), args.head, args.tail, args.template, args.bin)
               for _, r in df.iterrows()]
    print("\nsummary")
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
