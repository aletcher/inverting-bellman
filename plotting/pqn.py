"""PQN training-curve plot (eval returns + TD loss)."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plotting.style import PALETTE


def _smooth(arr, window):
    arr = np.array(arr, dtype=float)
    if window <= 1:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def _smooth_nan(arr, window):
    """Moving average that ignores NaN entries (numerator/denominator convolution)."""
    arr = np.array(arr, dtype=float)
    if window <= 1:
        return arr
    mask = ~np.isnan(arr)
    arr_filled = np.where(mask, arr, 0.0)
    kernel = np.ones(window)
    num = np.convolve(arr_filled, kernel, mode="valid")
    den = np.convolve(mask.astype(float), kernel, mode="valid")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def plot_pqn_training(metrics, save_path="pqn_training.png"):
    """Plot PQN training curves: greedy eval (disc/undisc) + per-goal train + TD loss."""
    has_eval = "eval_returns_per_goal" in metrics
    has_eval_undisc = "eval_returns_undiscounted_per_goal" in metrics
    has_train_per_goal = "train/episode_return_mean_per_goal" in metrics
    n_panels = 1 + int(has_eval) + int(has_eval_undisc) + int(has_train_per_goal)
    if n_panels == 4:
        # Flatten to row-major so ax_idx fills top-left → top-right → bottom-left → bottom-right.
        fig, axes_2d = plt.subplots(2, 2, figsize=(16, 8))
        axes = axes_2d.flatten()
    else:
        # Fallback for old checkpoints missing some metrics.
        fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels))
        if n_panels == 1:
            axes = [axes]

    def _plot_eval_panel(ax, key, ylabel, title):
        eval_per_goal = np.array(metrics[key])  # (num_updates, num_goals)
        n_goals = eval_per_goal.shape[1]
        eval_mask = ~np.isnan(eval_per_goal[:, 0])
        eval_steps = np.where(eval_mask)[0]
        if len(eval_steps) > 0:
            eval_data = eval_per_goal[eval_mask]
            for g in range(n_goals):
                ax.plot(eval_steps, eval_data[:, g], marker="o", markersize=3,
                        linewidth=1, alpha=0.7, label=f"goal {g}")
            ax.plot(eval_steps, eval_data.mean(axis=1), linewidth=2,
                    color="k", label="mean", linestyle="--")
            ax.legend(fontsize=7, ncol=min(n_goals + 1, 6))
        ax.set_xlabel("Update step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    ax_idx = 0

    if has_eval:
        _plot_eval_panel(
            axes[ax_idx],
            "eval_returns_per_goal",
            "Discounted return",
            "Greedy eval discounted returns per goal",
        )
        ax_idx += 1

    if has_eval_undisc:
        _plot_eval_panel(
            axes[ax_idx],
            "eval_returns_undiscounted_per_goal",
            "Undiscounted return",
            "Greedy eval undiscounted returns per goal",
        )
        ax_idx += 1

    if has_train_per_goal:
        ax = axes[ax_idx]
        ret = np.array(metrics["train/episode_return_mean_per_goal"], dtype=float)
        n_goals = ret.shape[1]
        window = max(1, ret.shape[0] // 100)
        for g in range(n_goals):
            c = PALETTE[g % len(PALETTE)]
            ax.plot(_smooth_nan(ret[:, g], window), linewidth=1, alpha=0.8,
                    color=c, label=f"goal {g}")
        ax.plot(_smooth_nan(np.nanmean(ret, axis=1), window),
                linewidth=2, color="k", label="mean")
        ax.set_xlabel("Update step")
        ax.set_ylabel("Episode return")
        ax.set_title("Training-rollout episode return per goal")
        ax.legend(fontsize=7, ncol=min(n_goals + 1, 6))
        ax_idx += 1

    td_loss = np.array(metrics["td_loss"], dtype=float)
    window = max(1, len(td_loss) // 100)
    axes[ax_idx].plot(td_loss, linewidth=0.5, alpha=0.3, color=PALETTE[1])
    axes[ax_idx].plot(_smooth(td_loss, window), linewidth=1.5, color=PALETTE[1],
                      label=f"smoothed (w={window})")
    axes[ax_idx].legend(fontsize=8)
    axes[ax_idx].set_xlabel("Update step")
    axes[ax_idx].set_ylabel("TD loss")
    axes[ax_idx].set_title("TD loss")
    axes[ax_idx].set_yscale("log")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
