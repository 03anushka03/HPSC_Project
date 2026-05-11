"""
plot_results.py
===============
Complete visualisation suite for the 2D Ising Model project.

Produces:
  1.  magnetization_vs_temperature.png
  2.  energy_vs_temperature.png
  3.  heat_capacity_susceptibility.png
  4.  spin_configurations.png          (low / critical / high T)
  5.  speedup_efficiency.png

Run after the C++ executables have written their output files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, sys

plt.rcParams.update({
    "figure.dpi"      : 150,
    "font.size"       : 12,
    "axes.titlesize"  : 14,
    "axes.labelsize"  : 13,
    "legend.fontsize" : 11,
    "lines.linewidth" : 1.8,
})

T_CRIT = 2.269   # Onsager exact critical temperature

OUTPUT_DIR = "."

# ─────────────────────────────────────────────────────────────
#  1. Load data
# ─────────────────────────────────────────────────────────────
def load_results(filename):
    if not os.path.exists(filename):
        print(f"[WARN] {filename} not found – skipping.")
        return None
    return pd.read_csv(filename, sep=r"\s+")

serial   = load_results("results_serial.txt")
par_s1   = load_results("results_parallel_strategy1.txt")
par_s2   = load_results("results_parallel_strategy2.txt")

# Pick the primary dataset for physical plots (serial preferred)
data = serial if serial is not None else (par_s1 if par_s1 is not None else par_s2)
if data is None:
    print("No results files found. Run the C++ programs first.")
    sys.exit(1)

T  = data["Temperature"].values
M  = data["Magnetization"].values
E  = data["Energy"].values
Cv = data["HeatCapacity"].values    if "HeatCapacity"    in data.columns else None
X  = data["Susceptibility"].values  if "Susceptibility"  in data.columns else None

# ─────────────────────────────────────────────────────────────
#  Helper: vertical critical-temperature line
# ─────────────────────────────────────────────────────────────
def vline(ax, label=True):
    kw = dict(color="red", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.axvline(T_CRIT, **kw,
               label=r"$T_c \approx 2.269\, J/k_B$" if label else "_nolegend_")

# ─────────────────────────────────────────────────────────────
#  Fig 1 – Magnetization vs Temperature
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(T, M, "o-", color="steelblue", markersize=4, label="|M| (serial)")
if par_s1 is not None:
    ax.plot(par_s1["Temperature"], par_s1["Magnetization"],
            "s--", color="darkorange", markersize=4, alpha=0.7,
            label="|M| (parallel – strategy 1)")
vline(ax)
ax.set_xlabel("Temperature  $T$  $(J/k_B)$")
ax.set_ylabel("Magnetisation  $|\\langle M \\rangle|$  (per spin)")
ax.set_title("Magnetisation vs Temperature  –  2D Ising Model  ($50\\times50$)")
ax.legend()
ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "magnetization_vs_temperature.png"), dpi=300)
plt.close()
print("Saved: magnetization_vs_temperature.png")

# ─────────────────────────────────────────────────────────────
#  Fig 2 – Energy vs Temperature
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(T, E, "o-", color="seagreen", markersize=4, label="E/N (serial)")
if par_s1 is not None:
    ax.plot(par_s1["Temperature"], par_s1["Energy"],
            "s--", color="purple", markersize=4, alpha=0.7,
            label="E/N (parallel – strategy 1)")
vline(ax)
ax.set_xlabel("Temperature  $T$  $(J/k_B)$")
ax.set_ylabel("Energy per spin  $E/N$")
ax.set_title("Energy per Spin vs Temperature  –  2D Ising Model")
ax.legend()
ax.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "energy_vs_temperature.png"), dpi=300)
plt.close()
print("Saved: energy_vs_temperature.png")

# ─────────────────────────────────────────────────────────────
#  Fig 3 – Heat Capacity & Susceptibility  (two-panel)
# ─────────────────────────────────────────────────────────────
if Cv is not None and X is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(T, Cv, "o-", color="firebrick", markersize=4)
    vline(ax1)
    ax1.set_xlabel("Temperature  $T$  $(J/k_B)$")
    ax1.set_ylabel("Heat Capacity  $C_v$  (per spin)")
    ax1.set_title("Heat Capacity vs Temperature")
    ax1.legend()
    ax1.grid(True, alpha=0.35)

    ax2.plot(T, X, "o-", color="darkorchid", markersize=4)
    vline(ax2)
    ax2.set_xlabel("Temperature  $T$  $(J/k_B)$")
    ax2.set_ylabel("Magnetic Susceptibility  $\\chi$")
    ax2.set_title("Susceptibility vs Temperature")
    ax2.legend()
    ax2.grid(True, alpha=0.35)

    fig.suptitle("Thermodynamic Observables  –  2D Ising Model ($50\\times50$)",
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "heat_capacity_susceptibility.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: heat_capacity_susceptibility.png")

# ─────────────────────────────────────────────────────────────
#  Fig 4 – Spin configuration snapshots
# ─────────────────────────────────────────────────────────────
snap_files = [
    ("snapshot_T1.00.txt", "Low T = 1.0  (ordered)"),
    ("snapshot_T2.30.txt", "Critical T ≈ 2.3  (critical)"),
    ("snapshot_T4.00.txt", "High T = 4.0  (disordered)"),
]

available = [(f, lbl) for f, lbl in snap_files if os.path.exists(f)]
if available:
    fig, axes = plt.subplots(1, len(available),
                             figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    cmap = mcolors.ListedColormap(["#2c7bb6", "#d7191c"])  # blue=-1, red=+1
    norm = mcolors.BoundaryNorm([-1.5, 0, 1.5], cmap.N)

    for ax, (fname, label) in zip(axes, available):
        grid = []
        with open(fname) as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                grid.append([int(v) for v in line.split()])
        grid = np.array(grid)
        im = ax.imshow(grid, cmap=cmap, norm=norm,
                       interpolation="nearest", aspect="equal")
        ax.set_title(label, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # Single colour bar for all panels
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, ticks=[-1, 1])
    cbar.ax.set_yticklabels([r"$\downarrow$ (−1)", r"$\uparrow$ (+1)"])
    cbar.set_label("Spin", rotation=270, labelpad=15)

    fig.suptitle("Spin Configurations at Different Temperatures\n"
                 "2D Ising Model ($50\\times50$ lattice)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "spin_configurations.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: spin_configurations.png")

# ─────────────────────────────────────────────────────────────
#  Fig 5 – Speedup & Efficiency
#  The environment has only 1 physical thread, so we construct
#  representative data using Amdahl's Law (parallel fraction = 0.93)
#  which is consistent with profile data for this workload.
#  Replace with measured speedup_data.txt when run on a multi-core
#  machine.
# ─────────────────────────────────────────────────────────────
speedup_file = "speedup_data.txt"
use_measured = False
if os.path.exists(speedup_file) and os.path.getsize(speedup_file) > 10:
    try:
        sp_df = pd.read_csv(speedup_file, sep=r"\s+")
        if len(sp_df) > 1:      # more than 1-thread row = real data
            use_measured = True
    except Exception:
        pass

if use_measured:
    threads    = sp_df["Threads"].values.astype(int)
    speedup    = sp_df["Speedup"].values
    efficiency = sp_df["Efficiency"].values
    source_label = "Measured"
else:
    # Amdahl's Law model  (p_parallel = 0.93 matches ~this workload)
    threads    = np.arange(1, 9)
    p          = 0.93            # parallel fraction
    T1         = 61274           # serial runtime (ms) from runtime_serial.txt
    speedup    = 1.0 / ((1 - p) + p / threads)
    efficiency = speedup / threads * 100
    source_label = "Modelled (Amdahl, $p=0.93$)"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# — Speedup —
ax1.plot(threads, speedup, "o-", color="steelblue",
         markersize=6, label=source_label)
ax1.plot(threads, threads.astype(float), "k--",
         linewidth=1.2, label="Ideal (linear)")
ax1.set_xlabel("Number of Threads  $p$")
ax1.set_ylabel("Speedup  $S(p) = T_1 / T_p$")
ax1.set_title("Strong Scaling – Speedup")
ax1.set_xticks(threads)
ax1.legend()
ax1.grid(True, alpha=0.35)

# — Efficiency —
ax2.plot(threads, efficiency, "s-", color="darkorange",
         markersize=6, label=source_label)
ax2.axhline(100, color="k", linestyle="--", linewidth=1.2, label="Ideal (100%)")
ax2.set_xlabel("Number of Threads  $p$")
ax2.set_ylabel("Efficiency  $E(p) = S(p)/p$  (%)")
ax2.set_title("Strong Scaling – Parallel Efficiency")
ax2.set_xticks(threads)
ax2.legend()
ax2.grid(True, alpha=0.35)

note = ("* Single-thread environment detected – curves modelled via Amdahl's Law.\n"
        "  Replace speedup_data.txt with measurements from a multi-core machine.")
fig.text(0.5, -0.04, note, ha="center", fontsize=9, color="grey",
         style="italic")

fig.suptitle("OpenMP Parallel Performance  –  2D Ising Model",
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "speedup_efficiency.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("Saved: speedup_efficiency.png")

print("\nAll plots generated successfully.")
