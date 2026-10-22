"""
Compare cone N2I training runs from their SLURM .out logs.

In an L_reg run, the progress-bar values (train_loss_epoch, val_loss) are totals:
    training:   L_Nei + w * L_reg   (w ramps 0 -> 1.96)
    validation: L_Nei + 1 * L_reg
so they are not comparable with an L_Nei-only run. This script reads the components instead:
    validation: "Cone validation loss for <scan>: <total> (L_Nei=..., L_reg=...)"
                (L_Nei-only runs print just the value, which is L_Nei), assigned to the epoch of the
                following "Epoch N: val_loss = ..." line
    training:   per-update lines "update eN ... loss=... [nei=... reg=... w=...]", averaged per epoch
                (L_Nei-only runs: loss = L_Nei)

Figures:
    cone_val_loss_comparison.png    validation L_Nei of every run (+ L_reg of L_reg runs, dashed)
    cone_train_loss_comparison.png  training L_Nei of every run (+ L_reg, dashed), L_reg weight on a
                                    second axis
"""
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# runs = {
#     "Lnei": Path("slurm/logs/n2i_cone_baseline_61034810.out"),
#     "Lnei + Lreg": Path("slurm/logs/n2i_cone_lreg_60981430.out"),
# }
runs = {
    "Lnei": Path("slurm/logs/n2i_cone_baseline_61034810.out"),
    "Lnei + Lreg": Path("slurm/logs/n2i_cone_lreg_60999940.out"),
    "Lnei + Lreg_r4_50 epoch": Path("slurm/logs/n2i_cone_lreg_61062788.out"),
    "Lnei + Lreg_r4_100 epoch": Path("slurm/logs/n2i_cone_lreg_r4_e100_61088056.out"),
}

RE_VAL = re.compile(r"Cone validation loss for \S+: ([0-9.eE+-]+)(?: \(L_Nei=([0-9.eE+-]+), L_reg=([0-9.eE+-]+)\))?")
RE_EPOCH_END = re.compile(r"Epoch (\d+): val_loss = ([0-9.eE+-]+), learning rate = ([0-9.eE+-]+)")
RE_UPDATE = re.compile(r"update e(\d+) \S+ r\d+ b\d+: step=\d+ loss=([0-9.eE+-]+)"
                       r"(?: nei=([0-9.eE+-]+) reg=([0-9.eE+-]+) w=([0-9.eE+-]+))?")


def parse_log(path):
    """Per-epoch losses. Returns dict with sorted numpy arrays; L_reg entries are NaN for L_Nei-only runs."""
    text = path.read_text(errors="ignore")
    records = re.split(r"[\r\n]+", text)  # progress bars use carriage returns

    val, lr = {}, {}
    train = {}  # epoch -> lists of (nei, reg, w)
    pending = None
    for record in records:
        m = RE_VAL.search(record)
        if m:
            total, nei, reg = m.groups()
            pending = (float(nei if nei is not None else total), float(reg) if reg is not None else np.nan)
        m = RE_EPOCH_END.search(record)
        if m and pending is not None:
            epoch = int(m.group(1))
            val[epoch] = pending  # the last validation of this epoch
            lr[epoch] = float(m.group(3))
            pending = None
        for m in RE_UPDATE.finditer(record):
            epoch, loss, nei, reg, w = m.groups()
            train.setdefault(int(epoch), []).append((
                float(nei if nei is not None else loss),
                float(reg) if reg is not None else np.nan,
                float(w) if w is not None else np.nan,
            ))

    ve = np.array(sorted(val))
    te = np.array(sorted(train))
    t = {e: np.array(train[e]) for e in te}
    return dict(
        val_epochs=ve,
        val_nei=np.array([val[e][0] for e in ve]),
        val_reg=np.array([val[e][1] for e in ve]),
        lr=np.array([lr[e] for e in ve]),
        train_epochs=te,
        train_nei=np.array([t[e][:, 0].mean() for e in te]),
        train_reg=np.array([t[e][:, 1].mean() for e in te]),
        weight=np.array([t[e][:, 2].mean() for e in te]),
        updates=np.array([len(t[e]) for e in te]),
    )


parsed = {}
for name, path in runs.items():
    p = parse_log(path)
    parsed[name] = p
    if len(p["val_epochs"]) == 0:
        print(f"{name}: no validation epochs found in {path}")
        continue
    i = int(np.argmin(p["val_nei"]))
    has_reg = np.isfinite(p["val_reg"]).any()
    print(
        f"{name}: {len(p['val_epochs'])} validation epochs ({p['val_epochs'][0]}-{p['val_epochs'][-1]}), "
        f"{len(p['train_epochs'])} training epochs ({p['updates'].min()}-{p['updates'].max()} updates each)\n"
        f"    best val L_Nei {p['val_nei'][i]:.6f} at epoch {p['val_epochs'][i]}; "
        f"final {p['val_nei'][-1]:.6f}"
        + (f"; final val L_reg {p['val_reg'][-1]:.6f}" if has_reg else "")
        + f"; learning rate changes after epochs "
        + (", ".join(str(e) for e, a, b in zip(p["val_epochs"][:-1], p["lr"][:-1], p["lr"][1:]) if b != a) or "none")
    )

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Validation: L_Nei (solid) for all runs, L_reg (dashed) where present
fig, ax = plt.subplots(figsize=(8, 5))
for k, (name, p) in enumerate(parsed.items()):
    c = colors[k % len(colors)]
    ax.plot(p["val_epochs"], p["val_nei"], color=c, label=f"{name}: L_Nei")
    if np.isfinite(p["val_reg"]).any():
        ax.plot(p["val_epochs"], p["val_reg"], color=c, ls="--", label=f"{name}: L_reg (weight 1)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation loss (Mouse_10)")
ax.set_title("Cone N2I validation: L_Nei comparable across runs")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("cone_val_loss_comparison.png", dpi=200)
plt.close(fig)

# Training: L_Nei (solid), L_reg (dashed), L_reg weight on a second axis
fig, ax = plt.subplots(figsize=(8, 5))
ax2 = None
for k, (name, p) in enumerate(parsed.items()):
    c = colors[k % len(colors)]
    ax.plot(p["train_epochs"], p["train_nei"], color=c, label=f"{name}: L_Nei")
    if np.isfinite(p["train_reg"]).any():
        ax.plot(p["train_epochs"], p["train_reg"], color=c, ls="--", label=f"{name}: L_reg (unweighted)")
        if ax2 is None:
            ax2 = ax.twinx()
            ax2.set_ylabel("L_reg weight w", color="0.4")
            ax2.tick_params(axis="y", colors="0.4")
        ax2.plot(p["train_epochs"], p["weight"], color=c, ls=":", alpha=0.7, label=f"{name}: weight w")
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (mean over updates)")
ax.set_title("Cone N2I training: L_Nei comparable across runs")
handles, labels = ax.get_legend_handles_labels()
if ax2 is not None:
    h2, l2 = ax2.get_legend_handles_labels()
    handles, labels = handles + h2, labels + l2
ax.legend(handles, labels)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("cone_train_loss_comparison.png", dpi=200)
plt.close(fig)

print("Saved:")
print("  cone_val_loss_comparison.png")
print("  cone_train_loss_comparison.png")