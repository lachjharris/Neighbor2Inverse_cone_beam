"""
Write a trimmed copy of projection HDF5 files: keep only the first --n_keep frames of 'images'.

Made for the noisy mouse scans, whose rotation stage stalled near the end of the acquisition:
check_rotation_stall.py puts the stop at file index ~1953 of the *_from1.h5 files in M3, M4 and M5,
with ~70 frames after it all showing the same view. Keeping frames 0..1949 (n_keep 1950) drops the
stuck frames with a few frames' margin.

Each output keeps the input's layout (dataset 'images', same dtype, one frame per chunk, no
compression), copies the file attributes and replaces 'note' with one describing both trims.
Output name: <input stem>_n<n_keep>.h5 next to the input. Existing outputs are not overwritten.
After writing, the first, middle and last kept frames are compared bit for bit with the input.

Usage (from PBPC_data/):
    python trim_projections.py --n_keep 1950 --paths <file1.h5> <file2.h5> ...
"""
import argparse
import os

import h5py
import numpy as np


def trim(path_in, n_keep, chunk=32):
    stem, ext = os.path.splitext(path_in)
    path_out = f"{stem}_n{n_keep}{ext}"
    if os.path.exists(path_out):
        print(f"SKIP (exists): {path_out}")
        return path_out

    with h5py.File(path_in, "r") as fin:
        src = fin["images"]
        n, rows, cols = src.shape
        if n_keep > n:
            raise ValueError(f"{path_in}: n_keep {n_keep} > {n} frames")
        old_note = str(fin.attrs.get("note", ""))
        tmp = path_out + ".partial"
        with h5py.File(tmp, "w") as fout:
            for k, v in fin.attrs.items():
                fout.attrs[k] = v
            fout.attrs["note"] = (
                f"frames 0..{n_keep - 1} of {os.path.basename(path_in)} "
                f"(dropped its last {n - n_keep} frames: rotation stage stalled, see check_rotation_stall.py)"
                + (f"; that file: {old_note}" if old_note else "")
            )
            dst = fout.create_dataset("images", shape=(n_keep, rows, cols), dtype=src.dtype,
                                      chunks=(1, rows, cols))
            for start in range(0, n_keep, chunk):
                stop = min(start + chunk, n_keep)
                dst[start:stop] = src[start:stop]
            for k in (0, n_keep // 2, n_keep - 1):
                if not np.array_equal(dst[k], src[k]):
                    raise RuntimeError(f"frame {k} differs after copy")
        os.rename(tmp, path_out)

    with h5py.File(path_out, "r") as f:
        print(f"wrote {path_out}: images {f['images'].shape} {f['images'].dtype}, "
              f"chunks {f['images'].chunks}; frames 0, {n_keep // 2}, {n_keep - 1} match the input")
        print(f"    note: {f.attrs['note']}")
    return path_out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paths", nargs="+", required=True, help="input projection HDF5 files")
    p.add_argument("--n_keep", type=int, required=True, help="number of leading frames to keep")
    args = p.parse_args()
    for path in args.paths:
        trim(path, args.n_keep)


if __name__ == "__main__":
    main()
