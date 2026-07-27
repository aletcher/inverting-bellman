"""Value-iteration comparison plots.

- plot_two_value_comparison_final  — 2-panel V_a vs V_b (3D + heatmap).
  Used by the vi phase (V*/V_pqn/Q overlays) and the unseen phase
  (V_true/V_wm).
- plot_two_value_comparison_2slice_3d — two orthogonal 3D slices of a
  4D V grid (Reacher). Used by the unseen phase.
"""

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers "3d" projection

from plotting.style import CMAP


def plot_two_value_comparison_final(
    X1,
    X2,
    panels,
    dim1_label,
    dim2_label,
    goal_position=None,
    save_path="value_comparison_final.png",
):
    """paper-quality 2-panel value comparison: 2×2 grid (top: 3D, bottom: heatmaps).

    Designed for V^π_pqn vs V_pqn. Shared colorscale, single colorbar to the
    right of the second heatmap, shared y-axis label (only on leftmost panels),
    clean alignment.

    Args:
        panels: list of 2 (label, V_grid) tuples.
    """
    assert len(panels) == 2, "plot_two_value_comparison_final expects exactly 2 panels"
    grids = [g for _, g in panels]
    vmin = min(float(g.min()) for g in grids)
    vmax = max(float(g.max()) for g in grids)
    levels = jnp.linspace(vmin, vmax, 30)

    gs = matplotlib.gridspec.GridSpec(
        2, 3,
        width_ratios=[1, 1, 0.04],
        height_ratios=[1.4, 1],
        hspace=0.12,
        wspace=0.08,
    )
    fig = plt.figure(figsize=(12, 10))

    # Top row: 3D surfaces. Bottom row: 2D contour heatmaps sharing colorscale.
    for col, (label, V) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        ax.plot_surface(
            X1, X2, V, cmap=CMAP, edgecolor="none",
            alpha=0.9, vmin=vmin, vmax=vmax,
        )
        ax.set_title(label, fontsize=12, pad=10)
        ax.set_xlabel(dim1_label, fontsize=10)
        ax.set_ylabel(dim2_label, fontsize=10)
        if col == 1:
            ax.set_zlabel("Value", fontsize=10)
        else:
            ax.set_zlabel("", fontsize=10)
        ax.set_zlim(vmin, vmax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.view_init()

    cp = None
    for col, (label, V) in enumerate(panels):
        ax = fig.add_subplot(gs[1, col])
        cp = ax.contourf(X1, X2, V, levels=levels, cmap=CMAP)
        cs = ax.contour(
            X1, X2, V, levels=levels, colors="white",
            linewidths=0.4, linestyles="dashed", alpha=0.6,
        )
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.3g")
        ax.set_xlabel(dim1_label, fontsize=10)
        if col == 0:
            ax.set_ylabel(dim2_label, fontsize=10)
        else:
            ax.set_ylabel("")
            ax.tick_params(left=False, labelleft=False)

    # Colorbar in dedicated column, aligned with heatmap row.
    cbar_ax = fig.add_subplot(gs[1, 2])
    fig.colorbar(cp, cax=cbar_ax, format="%.2f")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_two_value_comparison_2slice_3d(
    V_true_4d,
    V_wm_4d,
    axis_grids,
    state_labels,
    slice_dims_top=(0, 1),
    slice_dims_bottom=(2, 3),
    fixed_idx_top=None,
    fixed_idx_bottom=None,
    save_path_top="value_comparison_top.png",
    save_path_bottom="value_comparison_bottom.png",
):
    """3D V_true vs V_wm at two orthogonal slices of a 4D V grid.

    Produces two separate 1x2 plots (one per slice), each with two 3D surfaces:
    V_true (left, V^{true}) and V_wm (right, V) using matplotlib's
    constrained layout + bbox_extra_artists for clean borders.
    """
    assert V_true_4d.ndim == 4 and V_wm_4d.ndim == 4

    def _slice_2d(V, plot_dims, fixed_idx):
        plot_dims = tuple(plot_dims)
        idx = []
        for d in range(4):
            if d in plot_dims:
                idx.append(slice(None))
            else:
                idx.append(fixed_idx[d])
        sl = V[tuple(idx)]
        if plot_dims[0] > plot_dims[1]:
            sl = sl.T
        return sl

    def _midpoint_indices(plot_dims):
        n = V_true_4d.shape[0]
        return {d: n // 2 for d in range(4) if d not in plot_dims}

    if fixed_idx_top is None:
        fixed_idx_top = _midpoint_indices(slice_dims_top)
    if fixed_idx_bottom is None:
        fixed_idx_bottom = _midpoint_indices(slice_dims_bottom)

    def _save_3d_pair(slice_dims, fixed_idx, save_path):
        d0, d1 = slice_dims
        Vt = _slice_2d(V_true_4d, slice_dims, fixed_idx)
        Vw = _slice_2d(V_wm_4d, slice_dims, fixed_idx)
        g0 = np.asarray(axis_grids[d0])
        g1 = np.asarray(axis_grids[d1])
        X0, X1 = np.meshgrid(g0, g1, indexing="ij")
        vmin = float(min(Vt.min(), Vw.min()))
        vmax = float(max(Vt.max(), Vw.max()))

        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5),
            subplot_kw={"projection": "3d"},
            layout="constrained",
        )
        extra = []
        # WM (left) shown as $V$, true (right) shown as $V^{true}$.
        for col, (V2d, z_label) in enumerate([
            (Vw, r"$V$"),
            (Vt, r"$V^{\mathrm{true}}$"),
        ]):
            ax = axes[col]
            ax.plot_surface(
                X0, X1, V2d, cmap=CMAP, edgecolor="none",
                alpha=0.9, vmin=vmin, vmax=vmax,
            )
            ax.set_xlabel(state_labels[d0], fontsize=22, labelpad=10)
            ax.set_ylabel(state_labels[d1], fontsize=22, labelpad=10)
            ax.set_zlim(vmin, vmax)
            ax.set_xlim(float(g0[0]), float(g0[-1]))
            ax.set_ylim(float(g1[0]), float(g1[-1]))
            ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis="both", labelsize=16)
            ax.set_zlabel(z_label, fontsize=20, labelpad=10, rotation=90)
            extra.extend([ax.zaxis.label, ax.xaxis.label, ax.yaxis.label])
        fig.canvas.draw()
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    bbox_extra_artists=extra)
        plt.close(fig)

    _save_3d_pair(slice_dims_top, fixed_idx_top, save_path_top)
    _save_3d_pair(slice_dims_bottom, fixed_idx_bottom, save_path_bottom)
