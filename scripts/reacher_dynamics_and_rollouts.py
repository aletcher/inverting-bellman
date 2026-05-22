"""Reacher dynamics heatmaps + trajectory rollouts (paper Fig. 2).

Renders three plots from a single trained Reacher run, into outputs/figures/:
  - fig2_reacher_dynamics_heatmap_x_theta.png — true vs WM next-x dynamics
    in (θ₁, θ₂) space (3D surface).
  - fig2_reacher_dynamics_heatmap_y_theta.png — same, for next-y.
  - fig2_reacher_trajectory_policy.png        — policy-conditioned
    trajectory rollouts: true vs WM dynamics, per goal.

Reads the WM checkpoint from <run_dir>/<wm_subdir>/ — does NOT retrain or
re-evaluate. Pick any cell from the arch sweep (e.g. d6_w2048_s0) or any
single-seed Reacher run.

Usage:
    uv run python scripts/reacher_dynamics_and_rollouts.py \\
        --run_dir outputs/reacher/sweep_arch_<TS>/d6_w2048_s0 \\
        --wm_subdir wm_track \\
        --config reacher
"""

import argparse
import importlib
import json
import os
import pickle
import sys

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_dynamics(args, ENV_CONFIG, WM_CONFIG, p_params):
    """Construct true and WM dynamics functions in 4D effective state."""
    import gymnax
    from envs.env_dynamics import make_env_dynamics_fn

    basic_env, env_params = gymnax.make("MountainCar-v0")  # placeholder shape
    from envs.reacher import Reacher
    basic_env = Reacher()
    env_params = basic_env.default_params

    from envs.reacher_utils import (
        make_reacher_effective_dynamics_fn, make_reacher_wm_dynamics_fn,
        reacher_effective_to_obs,
    )
    true_dyn_fn = make_reacher_effective_dynamics_fn(basic_env, env_params)
    wm_dyn_fn = make_reacher_wm_dynamics_fn(p_params, WM_CONFIG, ENV_CONFIG["ACTION_DIM"])
    return true_dyn_fn, wm_dyn_fn, reacher_effective_to_obs


def _make_paper_heatmaps(true_dyn_fn, wm_dyn_fn, ENV_CONFIG, out_dir,
                          heatmap_res=40, action_idx=8):
    """Two 3D plots, each 1x2 (WM left, true right):
        (i)  next x in (theta_1, theta_2)
        (ii) next y in (theta_1, theta_2)
    """
    from plotting.style import CMAP

    state_ranges = ENV_CONFIG["STATE_RANGES"]

    t_lo, t_hi = state_ranges[0]
    t1_grid = jnp.linspace(t_lo, t_hi, heatmap_res)
    t2_grid = jnp.linspace(t_lo, t_hi, heatmap_res)
    T1, T2 = jnp.meshgrid(t1_grid, t2_grid, indexing="ij")
    n = heatmap_res * heatmap_res
    base = jnp.stack([
        T1.flatten(), T2.flatten(),
        jnp.zeros(n), jnp.zeros(n),
    ], axis=-1)
    actions = jnp.full(n, action_idx)

    s_true = np.asarray(jax.block_until_ready(true_dyn_fn(base, actions)))
    s_wm = np.asarray(jax.block_until_ready(wm_dyn_fn(base, actions)))

    # Next-state fingertip under true and WM dynamics.
    nx_t = (np.cos(s_true[:, 0]) + np.cos(s_true[:, 1])).reshape(T1.shape)
    ny_t = (np.sin(s_true[:, 0]) + np.sin(s_true[:, 1])).reshape(T1.shape)
    nx_w = (np.cos(s_wm[:, 0]) + np.cos(s_wm[:, 1])).reshape(T1.shape)
    ny_w = (np.sin(s_wm[:, 0]) + np.sin(s_wm[:, 1])).reshape(T1.shape)

    T1_np = np.asarray(T1)
    T2_np = np.asarray(T2)
    theta_ticks = [-2, 0, 2]

    def _save_pair(V_wm, V_true, x_data, y_data, x_lo, x_hi, y_lo, y_hi,
                   xticks, yticks, x_label, y_label,
                   z_left, z_right, save_path):
        vmin = min(float(V_wm.min()), float(V_true.min()))
        vmax = max(float(V_wm.max()), float(V_true.max()))
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5),
            subplot_kw={"projection": "3d"},
            layout="constrained",
        )
        extra = []
        for col, (V, z_label) in enumerate([
            (V_wm, z_left), (V_true, z_right),
        ]):
            ax = axes[col]
            ax.plot_surface(
                x_data, y_data, V, cmap=CMAP, edgecolor="none",
                alpha=0.9, vmin=vmin, vmax=vmax,
            )
            ax.set_xlabel(x_label, fontsize=22, labelpad=10)
            ax.set_ylabel(y_label, fontsize=22, labelpad=10)
            ax.set_zlim(vmin, vmax)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis="both", labelsize=16)
            ax.set_zlabel(z_label, fontsize=20, labelpad=10, rotation=90)
            extra.extend([ax.zaxis.label, ax.xaxis.label, ax.yaxis.label])
        # Draw once so 3D label positions are computed before tight bbox.
        fig.canvas.draw()
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    bbox_extra_artists=extra)
        plt.close(fig)
        print(f"  Saved {save_path}")

    # Next x in (theta_1, theta_2)
    _save_pair(
        nx_w, nx_t, T1_np, T2_np,
        t_lo, t_hi, t_lo, t_hi, theta_ticks, theta_ticks,
        r"$\theta_1$", r"$\theta_2$",
        r"$P_x$", r"$P^{\mathrm{true}}_x$",
        os.path.join(out_dir, "fig2_reacher_dynamics_heatmap_x_theta.png"),
    )
    # Next y in (theta_1, theta_2)
    _save_pair(
        ny_w, ny_t, T1_np, T2_np,
        t_lo, t_hi, t_lo, t_hi, theta_ticks, theta_ticks,
        r"$\theta_1$", r"$\theta_2$",
        r"$P_y$", r"$P^{\mathrm{true}}_y$",
        os.path.join(out_dir, "fig2_reacher_dynamics_heatmap_y_theta.png"),
    )


# ── Trajectory plot (Fig 1 right) ──────────────────────────────────────────


def _break_at_wrap(traj, d0, d1, jump_threshold=3.0):
    """Insert NaN rows between consecutive points where |Δdim| > threshold.

    Prevents matplotlib from drawing a straight line across the figure when a
    trajectory wraps around ±π on an angle dim. Threshold in radians; 3.0
    catches wrap jumps (~2π) but not normal motion.
    """
    traj = np.asarray(traj)
    if traj.shape[0] < 2:
        return traj
    deltas = np.abs(np.diff(traj[:, [d0, d1]], axis=0))
    wraps = np.where(np.any(deltas > jump_threshold, axis=1))[0]
    if len(wraps) == 0:
        return traj
    out = [traj[: wraps[0] + 1]]
    nan_row = np.full((1, traj.shape[1]), np.nan)
    for i, w in enumerate(wraps):
        out.append(nan_row)
        next_end = wraps[i + 1] + 1 if i + 1 < len(wraps) else traj.shape[0]
        out.append(traj[w + 1 : next_end])
    return np.concatenate(out, axis=0)


def _plot_true_vs_wm(ax, true_traj, pred_traj, d0, d1, color):
    """True (dashed) vs WM (solid, on top) trajectory pair, same color."""
    true_line = _break_at_wrap(true_traj, d0, d1)
    pred_line = _break_at_wrap(pred_traj, d0, d1)
    ax.plot(true_line[:, d0], true_line[:, d1],
            color=color, linewidth=4.0, linestyle="--", dashes=(3.0, 2.0),
            zorder=4)
    ax.plot(pred_line[:, d0], pred_line[:, d1],
            color=color, linewidth=4.5, zorder=6)


def _true_vs_wm_legend():
    from matplotlib.lines import Line2D
    return [
        Line2D([0], [0], color="gray", linewidth=4.0, linestyle="--",
               dashes=(3.0, 2.0), label="true"),
        Line2D([0], [0], color="gray", linewidth=4.5, label="WM"),
    ]


def _plot_trajectory_policy(
    per_goal_true,
    per_goal_pred,
    starts,
    goals,
    goal_masks,
    state_labels,
    state_ranges,
    dim_pair,
    n_y=1,
    n_x=1,
    reward_type=None,
    reward_a=None,
    reward_sigma=None,
    save_path="trajectory_policy.png",
):
    """Policy-conditioned trajectories: each panel = one starting state, all
    goals overlaid in different colors. Goal targets marked with lines and a
    shaded region showing the reward threshold (a for sparse, sigma for
    gaussian).

    Layout: rows = n_y (d1 values), cols = n_x (d0 values).
    """
    from matplotlib.lines import Line2D
    from plotting.style import PALETTE

    d0, d1 = dim_pair
    n_goals = len(per_goal_true)
    # Paper view: zoom around the reachable fingertip square.
    xlim = (-1.55, 1.3)
    ylim = (-1.55, 1.3)
    xticks = (-1, 0, 1)
    yticks = (-1, 0, 1)

    n_total = n_y * n_x
    fig, axes = plt.subplots(
        1, n_total, figsize=(5 * n_total, 5), squeeze=False,
    )

    def _truncate_at_goal(traj, goal_arr, mask_arr, half_w):
        """Truncate trajectory at the first step where the goal is reached."""
        if half_w <= 0:
            return traj
        mask_dims = np.where(np.array(mask_arr) > 0)[0]
        if len(mask_dims) == 0:
            return traj
        diffs = traj[:, mask_dims] - np.array(goal_arr)[mask_dims]
        dist = np.sqrt((diffs ** 2).sum(axis=-1))
        in_goal = dist <= half_w
        if not np.any(in_goal):
            return traj
        first_in = int(np.argmax(in_goal))
        return traj[: first_in + 1]

    for idx in range(n_total):
        ax = axes[0, idx]
        ax.grid(True, alpha=0.15)

        for g in range(n_goals):
            color = PALETTE[g % len(PALETTE)]
            true_traj = np.array(per_goal_true[g][idx])
            pred_traj = np.array(per_goal_pred[g][idx])

            if reward_type in ("sparse", "sparse_negative"):
                _half_w = reward_a if reward_a else 0
            elif reward_type == "gaussian":
                _half_w = reward_sigma if reward_sigma else 0
            else:
                _half_w = 0
            true_traj = _truncate_at_goal(true_traj, goals[g], goal_masks[g], _half_w)
            pred_traj = _truncate_at_goal(pred_traj, goals[g], goal_masks[g], _half_w)

            _plot_true_vs_wm(ax, true_traj, pred_traj, d0, d1, color)

            # Goal marker: line/circle + shaded region for reward threshold.
            goal_arr = np.array(goals[g])
            mask_arr = np.array(goal_masks[g])
            half_w = _half_w
            if mask_arr[d0] > 0 and mask_arr[d1] > 0 and half_w > 0:
                circle = plt.Circle((goal_arr[d0], goal_arr[d1]), half_w,
                                    color=color, alpha=0.18, fill=True,
                                    zorder=3)
                ax.add_patch(circle)
                ax.add_patch(plt.Circle(
                    (goal_arr[d0], goal_arr[d1]), half_w,
                    color=color, alpha=0.6, fill=False, linewidth=1.2,
                    zorder=3,
                ))
                ax.plot(goal_arr[d0], goal_arr[d1], "+", color=color,
                        markersize=10, markeredgewidth=2.0, zorder=3)
            elif mask_arr[d0] > 0:
                gv = goal_arr[d0]
                ax.axvline(gv, color=color, linewidth=1.5, alpha=0.5, linestyle=":")
                if half_w > 0:
                    ax.axvspan(gv - half_w, gv + half_w, color=color, alpha=0.08)
                    ax.axvline(gv - half_w, color=color, linewidth=0.8, alpha=0.5)
                    ax.axvline(gv + half_w, color=color, linewidth=0.8, alpha=0.5)
            elif mask_arr[d1] > 0:
                gv = goal_arr[d1]
                ax.axhline(gv, color=color, linewidth=1.5, alpha=0.5, linestyle=":")
                if half_w > 0:
                    ax.axhspan(gv - half_w, gv + half_w, color=color, alpha=0.08)
                    ax.axhline(gv - half_w, color=color, linewidth=0.8, alpha=0.5)
                    ax.axhline(gv + half_w, color=color, linewidth=0.8, alpha=0.5)

        ax.plot(float(starts[idx, d0]), float(starts[idx, d1]),
                "ko", markersize=7, zorder=7)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(list(xticks))
        ax.set_yticks(list(yticks))
        ax.tick_params(axis="both", labelsize=14)
        ax.set_aspect("equal")
        ax.set_xlabel(state_labels[d0], fontsize=16)
        if idx == 0:
            ax.set_ylabel(state_labels[d1], fontsize=16)
        else:
            ax.set_ylabel("")
            ax.tick_params(left=False, labelleft=False)

    handles = []
    if reward_type in ("sparse", "sparse_negative", "gaussian"):
        handles.append(Line2D([0], [0], marker="P", color="black",
                              linestyle="", markersize=17,
                              markerfacecolor="black", markeredgecolor="black",
                              alpha=0.7, label="goal"))
    handles.extend(_true_vs_wm_legend())
    axes[0, 0].legend(handles=handles, fontsize=17, loc="lower left",
                      labelcolor="black", handlelength=1.8, borderpad=0.45,
                      handletextpad=0.55, labelspacing=0.4)

    fig.tight_layout(pad=0.2, w_pad=2.2)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _fp_to_obs(target_fp):
    """Inverse FK: pick a (theta_1, theta_2) pair whose fingertip is near target_fp.

    Solves cos(t1)+cos(t2)=fp_x, sin(t1)+sin(t2)=fp_y via sum-to-product.
    Returns a 6D obs [t1, t2, 0, 0, fp_x_actual, fp_y_actual].
    """
    fp_x, fp_y = target_fp
    r = float(np.sqrt(fp_x ** 2 + fp_y ** 2))
    if r >= 2.0:
        r = 1.99
    A = float(np.arctan2(fp_y, fp_x))
    B = float(np.arccos(r / 2.0))
    t1 = A + B
    t2 = A - B
    # Wrap to [-pi, pi].
    t1 = (t1 + np.pi) % (2 * np.pi) - np.pi
    t2 = (t2 + np.pi) % (2 * np.pi) - np.pi
    fp_x_a = float(np.cos(t1) + np.cos(t2))
    fp_y_a = float(np.sin(t1) + np.sin(t2))
    return np.array([t1, t2, 0.0, 0.0, fp_x_a, fp_y_a])


def _theta_omega_to_obs(t1, t2, w1, w2):
    """Build a 6D obs from explicit (θ₁, θ₂, ω₁, ω₂); fp via FK."""
    fp_x = float(np.cos(t1) + np.cos(t2))
    fp_y = float(np.sin(t1) + np.sin(t2))
    return np.array([float(t1), float(t2), float(w1), float(w2), fp_x, fp_y])


def _make_paper_trajectory_policy(
    p_params, q_params, q_batch_stats, basic_env, env_params,
    ENV_CONFIG, WM_CONFIG, PQN_CONFIG, out_dir,
    target_a_fp=(0.2, 0.0), target_b_fp=(-1.2, -1.0),
    state_a=None, state_b=None,
    goal_indices_override_arg=None,
):
    """Trajectory policy plot for paper: 2 specific starts.

    Each start is built either from an explicit (θ₁, θ₂, ω₁, ω₂) tuple
    (state_a/state_b) if provided, or from an inverse-FK target fingertip
    position with ω=0 (target_a_fp/target_b_fp).
    """
    from envs.env_dynamics import make_env_dynamics_fn
    from envs.reacher_utils import (
        reacher_obs_to_effective, reacher_effective_to_obs,
    )
    from eval.world_model import compute_policy_trajectory_rollouts
    from training.wm import make_world_model

    state_dim = ENV_CONFIG["STATE_DIM"]  # 6 for Reacher obs
    action_dim = ENV_CONFIG["ACTION_DIM"]
    state_labels = list(ENV_CONFIG["STATE_LABELS"])
    # LaTeX x/y instead of fp_x/fp_y for paper plots.
    state_labels[4] = r"$x$"
    state_labels[5] = r"$y$"
    state_ranges = ENV_CONFIG["STATE_RANGES"]
    goals = PQN_CONFIG["GOALS"]
    goal_masks = PQN_CONFIG["REWARD_MASK"]
    goal_indices_override = (
        goal_indices_override_arg
        if goal_indices_override_arg is not None
        else ENV_CONFIG.get("WM_TRAJECTORY_GOAL_INDICES")
    )
    print(f"  goal_indices_override = {goal_indices_override}")

    def _resolve_start(state_tuple, fp_target):
        if state_tuple is not None:
            return _theta_omega_to_obs(*state_tuple), state_tuple, None
        return _fp_to_obs(fp_target), None, fp_target

    start_a_obs, state_a_resolved, fp_a = _resolve_start(state_a, target_a_fp)
    start_b_obs, state_b_resolved, fp_b = _resolve_start(state_b, target_b_fp)
    starts_obs = jnp.array(np.stack([start_a_obs, start_b_obs]))
    src_a = (f"θω={state_a_resolved}" if state_a_resolved is not None
             else f"target_fp={fp_a}")
    src_b = (f"θω={state_b_resolved}" if state_b_resolved is not None
             else f"target_fp={fp_b}")
    print(f"  Start A: {src_a} -> obs {np.round(np.array(starts_obs[0]), 3)}")
    print(f"  Start B: {src_b} -> obs {np.round(np.array(starts_obs[1]), 3)}")

    dynamics_fn = make_env_dynamics_fn(basic_env, env_params, "Reacher")
    # WM uses effective state dim (4D for Reacher), not obs dim (6D).
    wm_dim = WM_CONFIG.get("WM_OUTPUT_DIM", state_dim)
    p_model = make_world_model(WM_CONFIG, wm_dim)

    def sample_fn(_key, _n):
        return starts_obs

    per_goal_true, per_goal_pred, starts_p, n_y_p, n_x_p, goal_idxs = (
        compute_policy_trajectory_rollouts(
            p_params, p_model, q_params, q_batch_stats, PQN_CONFIG,
            state_dim, action_dim, state_ranges, dynamics_fn,
            slice_point=jnp.zeros(state_dim), dim_pair=(4, 5),
            goals=goals, goal_masks=goal_masks,
            n_steps=PQN_CONFIG.get("MAX_STEPS_IN_EPISODE", 100),
            n_x=2, n_y=1, wm_config=WM_CONFIG,
            sample_states_fn=sample_fn,
            state_to_eff_fn=reacher_obs_to_effective,
            eff_to_obs_fn=reacher_effective_to_obs,
            goal_indices_override=goal_indices_override,
        )
    )
    plot_goals = goals[goal_idxs]
    plot_masks = goal_masks[goal_idxs]

    save_path = os.path.join(out_dir, "fig2_reacher_trajectory_policy.png")
    _plot_trajectory_policy(
        per_goal_true, per_goal_pred, starts_p,
        goals=plot_goals, goal_masks=plot_masks,
        state_labels=state_labels, state_ranges=state_ranges,
        dim_pair=(4, 5), n_y=n_y_p, n_x=n_x_p,
        reward_type=PQN_CONFIG.get("REWARD_TYPE"),
        reward_a=PQN_CONFIG.get("REWARD_A"),
        reward_sigma=PQN_CONFIG.get("REWARD_SIGMA"),
        save_path=save_path,
    )
    print(f"  Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--wm_subdir", required=True)
    parser.add_argument("--out_dir", default="outputs/figures",
                        help="Where to write the 3 paper PNGs. Default: "
                             "outputs/figures/ (shared with other figure scripts).")
    parser.add_argument("--config", default="reacher",
                        help="Config module to load.")
    parser.add_argument("--heatmap_res", type=int, default=40)
    parser.add_argument("--heatmap_action", type=int, default=8,
                        help="Action index for the heatmap row (default: (+1,+1) = 8).")
    parser.add_argument("--skip_trajectory", action="store_true",
                        help="Skip trajectory_policy plot (e.g. if PQN checkpoint not available).")
    parser.add_argument("--traj_target_a", type=float, nargs=2, default=(0.2, 0.0),
                        metavar=("FP_X", "FP_Y"),
                        help="Target fingertip pos for first start. "
                             "Ignored if --traj_state_a is set.")
    parser.add_argument("--traj_target_b", type=float, nargs=2, default=(-1.2, -1.0),
                        metavar=("FP_X", "FP_Y"),
                        help="Target fingertip pos for second start. "
                             "Ignored if --traj_state_b is set.")
    parser.add_argument("--traj_state_a", type=float, nargs=4,
                        default=(-1.2, 2.2, -0.8, -0.4),
                        metavar=("T1", "T2", "W1", "W2"),
                        help="Explicit (θ₁, θ₂, ω₁, ω₂) for first start.")
    parser.add_argument("--traj_state_b", type=float, nargs=4,
                        default=(-2.7, -1.8, 0.2, -0.2),
                        metavar=("T1", "T2", "W1", "W2"),
                        help="Explicit (θ₁, θ₂, ω₁, ω₂) for second start.")
    parser.add_argument("--traj_goal_idxs", type=int, nargs="+", default=None,
                        help="Override WM_TRAJECTORY_GOAL_INDICES (which goals to overlay).")
    args = parser.parse_args()

    config_mod = importlib.import_module(f"configs.{args.config}")
    ENV_CONFIG = config_mod.ENV_CONFIG
    ENV_NAME = config_mod.ENV_NAME
    assert ENV_NAME == "Reacher", f"This script is Reacher-specific (got {ENV_NAME})"

    with open(os.path.join(args.run_dir, args.wm_subdir, "wm_config.json")) as f:
        WM_CONFIG = json.load(f)

    wm_ckpt_path = os.path.join(args.run_dir, args.wm_subdir, "wm_checkpoint.pkl")
    if not os.path.exists(wm_ckpt_path):
        # Arch-sweep layout: nested step_*/wm_checkpoint.pkl.
        import glob as _glob
        candidates = sorted(_glob.glob(
            os.path.join(args.run_dir, args.wm_subdir, "step_*", "wm_checkpoint.pkl")
        ))
        if candidates:
            wm_ckpt_path = candidates[-1]
        else:
            raise FileNotFoundError(f"No WM checkpoint at {wm_ckpt_path}")
    with open(wm_ckpt_path, "rb") as f:
        wm_ckpt = pickle.load(f)
    p_params = wm_ckpt["params"]
    print(f"Loaded WM from {wm_ckpt_path}")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    true_dyn_fn, wm_dyn_fn, _ = _build_dynamics(args, ENV_CONFIG, WM_CONFIG, p_params)

    print("\n── Plot 1: paper dynamics heatmaps (theta_1, theta_2) ──")
    _make_paper_heatmaps(true_dyn_fn, wm_dyn_fn, ENV_CONFIG, out_dir,
                         args.heatmap_res, args.heatmap_action)

    if not args.skip_trajectory:
        pqn_ckpt_path = os.path.join(args.run_dir, "pqn_checkpoint.pkl")
        if not os.path.exists(pqn_ckpt_path):
            print(f"\nSkipping trajectory_policy plot: no PQN checkpoint at {pqn_ckpt_path}")
        else:
            with open(pqn_ckpt_path, "rb") as f:
                pqn_ckpt = pickle.load(f)
            q_params = pqn_ckpt["params"]
            q_batch_stats = pqn_ckpt.get("batch_stats", {})
            # Use the run's saved JSON config (live config may have different
            # arch hyperparams or goal definitions).
            saved_pqn_path = os.path.join(args.run_dir, "pqn_config.json")
            with open(saved_pqn_path) as f:
                PQN_CONFIG = json.load(f)
            # JSON stores arrays as nested lists; convert back to jnp.
            for k in ("GOALS", "REWARD_MASK"):
                if k in PQN_CONFIG:
                    PQN_CONFIG[k] = jnp.array(PQN_CONFIG[k])

            from envs.reacher import Reacher
            basic_env = Reacher(
                reward_type=PQN_CONFIG["REWARD_TYPE"],
                sigma=PQN_CONFIG["REWARD_SIGMA"],
                a=PQN_CONFIG["REWARD_A"],
                max_steps_in_episode=PQN_CONFIG["MAX_STEPS_IN_EPISODE"],
                torque_values=PQN_CONFIG.get("REACHER_TORQUE_VALUES"),

            )
            env_params = basic_env.default_params

            print("\n── Plot 2: paper trajectory_policy (2 starts in fp space) ──")
            _make_paper_trajectory_policy(
                p_params, q_params, q_batch_stats, basic_env, env_params,
                ENV_CONFIG, WM_CONFIG, PQN_CONFIG, out_dir,
                target_a_fp=tuple(args.traj_target_a),
                target_b_fp=tuple(args.traj_target_b),
                state_a=(tuple(args.traj_state_a)
                         if args.traj_state_a is not None else None),
                state_b=(tuple(args.traj_state_b)
                         if args.traj_state_b is not None else None),
                goal_indices_override_arg=args.traj_goal_idxs,
            )


if __name__ == "__main__":
    main()
