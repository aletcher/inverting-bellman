"""Global colour palette and style constants for all plots.

Sections:
- SHARED  : palette / cmap primitives used everywhere (PALETTE, COLOR_*,
            CMAP, blend helpers).
- PAPER : paper-quality rcParams + figure-size constants
            (set_paper_style / unset_paper_style).
- ARCH    : Reacher architecture-sweep specific cmaps.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns


# ══════════════════════════════════════════════════════════════════════════
# SHARED — palette / cmap primitives used by all experiments
# ══════════════════════════════════════════════════════════════════════════

PALETTE = sns.color_palette("tab10")  # default seaborn (10 colors)

# Named semantic colours.
COLOR_TRUE = PALETTE[0]       # blue — true dynamics / policy
COLOR_WM = PALETTE[1]         # orange — WM dynamics / policy
COLOR_GOAL = PALETTE[2]       # green — goal regions
COLOR_UNSAFE = PALETTE[3]     # red — unsafe / forbidden regions

# Quiver plot colors and alphas (true vs WM dynamics).
COLOR_TRUE_Q = "#1f77b4"      # tab blue
COLOR_WM_Q = "#d62728"        # tab red
ALPHA_TRUE_Q = 0.8
ALPHA_WM_Q = 0.8

# Generic-heatmap colormap (rocket) for value-function plots and dynamics
# heatmaps. The Reacher arch sweep uses its own ARCH_CMAP_* below.
CMAP = sns.color_palette("rocket", as_cmap=True)


def sample_palette(cmap, n, lo=0.4, hi=0.9):
    """Sample *n* evenly-spaced colors from *cmap* in the range [lo, hi]."""
    import numpy as np
    return [cmap(f) for f in np.linspace(lo, hi, n)]


def lighten(color, alpha=0.5):
    """Return the same color with transparency (RGBA)."""
    rgb = mcolors.to_rgb(color)
    return (*rgb, alpha)


def blend_with_white(color, alpha=0.5):
    """Blend *color* toward white to simulate transparency."""
    rgb = mcolors.to_rgb(color)
    return tuple(c * alpha + 1.0 * (1.0 - alpha) for c in rgb)


# ══════════════════════════════════════════════════════════════════════════
# PAPER — paper-quality rcParams + layout constants (shared across plots)
# ══════════════════════════════════════════════════════════════════════════

PAPER_TEXTWIDTH = 6.5    # inches
PAPER_FIG_HEIGHT = 4.0
PAPER_DPI = 300


def set_paper_style():
    """Set seaborn whitegrid + serif/cmr10 rcParams for camera-ready figures.

    Pair with unset_paper_style() after the figure is saved to restore
    matplotlib defaults so subsequent (non-paper) plots aren't affected.
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "axes.formatter.use_mathtext": True,
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": PAPER_DPI,
        "savefig.dpi": PAPER_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        # Black box around the axes (whitegrid defaults to light-grey spines).
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        # Thinner, lighter gridlines so they don't compete with the curves.
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
    })


def unset_paper_style():
    """Restore matplotlib + seaborn defaults."""
    sns.reset_defaults()
    plt.rcdefaults()


# ══════════════════════════════════════════════════════════════════════════
# ARCH — Reacher architecture-sweep specific
# ══════════════════════════════════════════════════════════════════════════

ARCH_CMAP_PQN_RETURN = sns.color_palette("flare_r", as_cmap=True)  # slice plots
ARCH_CMAP_WM_MSE = sns.color_palette("flare_r", as_cmap=True)
ARCH_CMAP_HEATMAP = sns.color_palette("flare_r", as_cmap=True)     # full-grid heatmaps

ARCH_CMAP_LO = 0.0
ARCH_CMAP_HI = 1.0

ARCH_CURVE_MARKER_KW = dict(marker="o", markeredgecolor="white", markeredgewidth=1.2)
