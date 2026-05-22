"""WM-phase plots: training-loss curve + dynamics quiver overview.

Used only by run.py (loss curve) and eval/world_model.py (quiver).
Paper-figure plot code lives inline in each `scripts/*` file that
produces a figure.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting.style import PALETTE, COLOR_TRUE, COLOR_WM


def plot_loss_curve(losses, save_path="loss.png"):
    """Log-scale Bellman-residual loss curve for WM training.

    Mirrors the TD-loss panel in plot_pqn_training: faint raw values + a
    smoothed overlay with a window proportional to total length.
    """
    losses = np.asarray(losses, dtype=float)
    window = max(1, len(losses) // 100)
    fig, ax = plt.subplots()
    ax.plot(losses, linewidth=0.5, alpha=0.3, color=PALETTE[0])
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, linewidth=1.5, color=PALETTE[0],
                label=f"smoothed (w={window})")
        ax.legend(fontsize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Bellman residual loss")
    ax.set_yscale("log")
    ax.set_title("World model training loss")
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_dynamics_quiver(
    base_states,
    s_true_by_action,
    s_pred_by_action,
    grid_shape,
    G0,
    G1,
    state_labels,
    action_names,
    d0,
    d1,
    stride=20,
    state_ranges=None,
    slice_label=None,
    save_path="dynamics_quiver.png",
):
    """Displacement field: true vs predicted, arrows with real magnitude."""
    n_actions = len(action_names)
    n_h, n_w = grid_shape

    # Build index arrays that always include the first and last grid points.
    idx_h = np.unique(np.append(np.arange(0, n_h, stride), n_h - 1))
    idx_w = np.unique(np.append(np.arange(0, n_w, stride), n_w - 1))

    fig, axes = plt.subplots(1, n_actions, figsize=(6 * n_actions, 5))

    if n_actions == 1:
        axes = [axes]

    # Per-dimension range for normalizing arrow components.
    range_d0 = float(G0.max() - G0.min()) or 1.0
    range_d1 = float(G1.max() - G1.min()) or 1.0

    # Shared scale across all actions and true/WM (preserves relative magnitude).
    # Exclude first column (left wall) where velocity-zeroing creates outlier arrows.
    idx_w_inner = idx_w[1:]
    all_mag = []
    for a in range(n_actions):
        for s_arr in [s_true_by_action[a], s_pred_by_action[a]]:
            dd0 = (s_arr[:, d0] - base_states[:, d0]).reshape(n_h, n_w)[np.ix_(idx_h, idx_w_inner)] / range_d0
            dd1 = (s_arr[:, d1] - base_states[:, d1]).reshape(n_h, n_w)[np.ix_(idx_h, idx_w_inner)] / range_d1
            all_mag.append(np.sqrt(dd0**2 + dd1**2))
    max_mag = float(np.max(np.concatenate([m.flatten() for m in all_mag])))
    scale = max_mag * 12 if max_mag > 0 else 1.0

    for a in range(n_actions):
        s_true = s_true_by_action[a]
        s_pred = s_pred_by_action[a]
        ax = axes[a]

        x = G0[np.ix_(idx_h, idx_w)]
        y = G1[np.ix_(idx_h, idx_w)]

        def _norm_and_clamp(s_arr):
            dd0 = np.array((s_arr[:, d0] - base_states[:, d0]).reshape(n_h, n_w)[np.ix_(idx_h, idx_w)]) / range_d0
            dd1 = np.array((s_arr[:, d1] - base_states[:, d1]).reshape(n_h, n_w)[np.ix_(idx_h, idx_w)]) / range_d1
            mag = np.sqrt(dd0**2 + dd1**2)
            factor = np.where(mag > max_mag, max_mag / np.maximum(mag, 1e-10), 1.0)
            return dd0 * factor, dd1 * factor

        true_d0, true_d1 = _norm_and_clamp(s_true)
        pred_d0, pred_d1 = _norm_and_clamp(s_pred)

        ax.quiver(
            x, y, true_d0, true_d1,
            color=COLOR_TRUE, alpha=1.0, label="true", width=0.003,
            scale=scale, zorder=1,
        )
        ax.quiver(
            x, y, pred_d0, pred_d1,
            color=COLOR_WM, alpha=1.0, label="WM", width=0.003,
            scale=scale, zorder=2,
        )
        ax.set_title(f"action={action_names[a]}", fontsize=12)
        ax.set_xlabel(state_labels[d0], fontsize=10)
        ax.set_ylabel(state_labels[d1], fontsize=10)
        if state_ranges is not None:
            lo0, hi0 = state_ranges[d0]
            lo1, hi1 = state_ranges[d1]
            pad0 = (hi0 - lo0) * 0.03
            pad1 = (hi1 - lo1) * 0.03
            ax.set_xlim(lo0 - pad0, hi0 + pad0)
            ax.set_ylim(lo1 - pad1, hi1 + pad1)
            ax.set_xticks(np.linspace(lo0, hi0, 7))
            ax.set_yticks(np.linspace(-0.06, 0.06, 5))
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
