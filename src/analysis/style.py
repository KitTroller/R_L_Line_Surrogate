"""Shared plumbing for every analysis script: paths, import path, chart style.

Import it FIRST in every script in this folder:

    from style import ROOT, GRAPHS, C, save

Why one file: colours and paths defined once. If two scripts each pick "the blue for
predicted", they drift, and two figures that should be read side by side stop agreeing.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAPHS = ROOT / "graphs"
# `python src/analysis/x.py` puts src/analysis on sys.path, not src/, where the pipeline lives.
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")                      # write files only: identical on the laptop and a headless HPC node
import matplotlib.pyplot as plt

# Slots 1-3 of the validated reference palette: all-pairs colour-blind safe on the light
# surface (worst CVD dE 9.2). Colour follows the ROLE, never the plotting order.
# Slot 3 (aqua) is below 3:1 contrast: only ever use it with a legend or a direct label.
C = {
    "predicted": "#2a78d6",   # slot 1 blue    chained over W windows: the deployed number
    "train":     "#2a78d6",
    "assisted":  "#eb6834",   # slot 2 orange  true i0 handed to every window
    "val":       "#eb6834",
    "third":     "#1baf7a",   # slot 3 aqua
    "truth":     "#52514e",   # neutral ink: ground truth is a reference, not a series
    "ink":       "#0b0b0b",
    "muted":     "#76756f",
    "grid":      "#e6e5e0",
    "surface":   "#fcfcfb",
}

plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"], "axes.titlecolor": C["ink"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linestyle": "-", "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.color": C["muted"], "ytick.color": C["muted"],
    "xtick.labelcolor": C["ink"], "ytick.labelcolor": C["ink"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 1.5, "lines.markersize": 5,
    "legend.frameon": False, "legend.fontsize": 9,
})


def save(fig, relpath):
    """graphs/<relpath>, folders created, closed afterwards so a loop doesn't leak figures."""
    p = GRAPHS / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {p.relative_to(ROOT)}")
    return p
