"""MountainCar quiver: position-trained vs velocity-trained WMs (paper Fig. 5).

Loads two WM checkpoints (one trained on position goals, one on velocity
goals), evaluates each on a (position × velocity) grid for every discrete
action, and writes:
  - outputs/results/fig5_mountaincar_quiver.png      — both WMs overlaid on
    ground-truth displacements, one panel per action.
  - outputs/results/fig5_mountaincar_quiver.txt      — per-(action, dim)
    MSE/NMSE between the two predictions plus pairwise stats vs. ground
    truth (source of the "WMs ~15× closer to each other than to truth"
    number in §5.2 of the paper).

CLI:
    # Auto: trains both mountaincar-position and -velocity sweeps fresh
    # (1 seed each by default), then produces the plot and summary.
    uv run python scripts/mountaincar_wm_quiver.py

    # Reuse existing sweeps (skips training).
    uv run python scripts/mountaincar_wm_quiver.py \\
        --mc_position_sweep outputs/mountaincar-position/seeds_<TS> \\
        --mc_velocity_sweep outputs/mountaincar-velocity/seeds_<TS>

Each WM is loaded from <sweep>/seed_0/wm_<latest>/. Ground-truth dynamics
come from gymnax.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import gymnax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.env_dynamics import make_env_dynamics_fn
from envs.mountaincar import MountainCar
from training.wm import load_wm, make_world_model, apply_wm

ACTION_NAMES_MC = ["left", "none", "right"]
STATE_LABELS_MC = ["position", "velocity"]
STATE_RANGES_MC = [(-1.2, 0.6), (-0.07, 0.07)]
N_HEAT = 200


def _resolve_pkl(path):
    if os.path.isdir(path):
        return os.path.join(path, "wm_checkpoint.pkl")
    return path


def _wm_dir(path):
    if os.path.isdir(path):
        return path
    return os.path.dirname(path)


def _load_wm_with_config(path):
    pkl = _resolve_pkl(path)
    p_params, losses, _ = load_wm(pkl)
    cfg_path = os.path.join(_wm_dir(path), "wm_config.json")
    with open(cfg_path) as f:
        wm_config = json.load(f)
    return p_params, losses, wm_config


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _train_sweep(config_stem, num_seeds):
    """Run PQN+WM training for `num_seeds` seeds; return the sweep dir."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(_REPO_ROOT, "outputs", config_stem, f"seeds_{ts}")
    seeds_csv = ",".join(str(i) for i in range(num_seeds))
    cmd = [
        sys.executable, os.path.join(_REPO_ROOT, "run.py"),
        "--config", f"configs/{config_stem}.py",
        "--phases", "pqn,wm",
        "--seeds", seeds_csv,
        "--sweep_dir", sweep_dir,
    ]
    print(f"\n=== Auto-training {config_stem} "
          f"({num_seeds} seed{'s' if num_seeds > 1 else ''}) ===")
    print(" ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(
            f"run.py failed for {config_stem} (exit {ret.returncode})"
        )
    return sweep_dir


def _seed_wm_dir(sweep_dir, seed_idx):
    """Return <sweep_dir>/seed_<i>/wm_<latest>/ or raise."""
    seed_dir = pathlib.Path(sweep_dir) / f"seed_{seed_idx}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"No seed_{seed_idx}/ under {sweep_dir}")
    wm_subdirs = sorted(
        p for p in seed_dir.iterdir()
        if p.is_dir() and p.name.startswith("wm_")
    )
    if not wm_subdirs:
        raise FileNotFoundError(f"No wm_* subdir under {seed_dir}")
    return str(wm_subdirs[-1])


def _discover_seeds(sweep_dir):
    """Set of seed indices i for which <sweep>/seed_<i>/ exists."""
    return {
        int(p.name.split("_")[1])
        for p in pathlib.Path(sweep_dir).glob("seed_*")
        if p.is_dir()
    }


def _mean_se(values):
    """Return (mean, SE) of a list. SE = 0 for a single value."""
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se


def _predict_grid(p_params, wm_config, state_dim, action_dim, base_states):
    """Return list of (n, state_dim) arrays, one per action, for the given WM."""
    p_model = make_world_model(wm_config, state_dim)
    residual = wm_config.get("RESIDUAL_PREDICTION", True)
    n = base_states.shape[0]
    out = []
    for a in range(action_dim):
        a_oh = jax.nn.one_hot(jnp.full(n, a), action_dim)
        s_pred = apply_wm(p_model, p_params, base_states, a_oh, residual=residual)
        out.append(np.asarray(s_pred))
    return out


def _metrics(pred_a, pred_b):
    """MSE / NMSE between pred_a (prediction) and pred_b (reference).

    NMSE = MSE / Var(pred_b) — unit-free.
    """
    diff = pred_a - pred_b
    var_b = float(np.var(pred_b))
    mse = float(np.mean(diff ** 2))
    return dict(
        mse=mse,
        nmse=(mse / var_b) if var_b > 1e-12 else 0.0,
    )


def _print_and_write_metrics(
    pred_a_per_seed,
    pred_b_per_seed,
    pred_true,
    state_labels,
    action_names,
    out_path,
    label_a,
    label_b,
):
    """Three blocks: A vs B, A vs true, B vs true; averaged across seeds.

    pred_a_per_seed / pred_b_per_seed are lists (one entry per seed) of
    per-action prediction arrays produced by _predict_grid. pred_true is
    seed-independent. Outputs mean ± SE per (action, dim) cell, then a
    block-level Avg of per-seed cell-mean values (also ± SE).
    """
    n_seeds = len(pred_a_per_seed)
    blocks = [
        (f"{label_a}_vs_{label_b}", pred_a_per_seed, pred_b_per_seed),
        (f"{label_a}_vs_true",      pred_a_per_seed, [pred_true] * n_seeds),
        (f"{label_b}_vs_true",      pred_b_per_seed, [pred_true] * n_seeds),
    ]

    def fmt(mean, se):
        return f"{mean:.1e} ± {se:.1e}" if n_seeds > 1 else f"{mean:.1e}"

    lines = [f"Cross-WM metrics (n={n_seeds} seed{'s' if n_seeds != 1 else ''})."]
    summaries = []
    for name, As, Bs in blocks:
        # per-cell across seeds + per-seed block-level averages.
        per_cell = {}  # (a_name, d_name) -> {"mse": [...], "nmse": [...]}
        block_mse_per_seed, block_nmse_per_seed = [], []
        for A, B in zip(As, Bs):
            cell_mses, cell_nmses = [], []
            for a_idx, a_name in enumerate(action_names):
                for d_idx, d_name in enumerate(state_labels):
                    m = _metrics(A[a_idx][:, d_idx], B[a_idx][:, d_idx])
                    per_cell.setdefault((a_name, d_name), {"mse": [], "nmse": []})
                    per_cell[(a_name, d_name)]["mse"].append(m["mse"])
                    per_cell[(a_name, d_name)]["nmse"].append(m["nmse"])
                    cell_mses.append(m["mse"])
                    cell_nmses.append(m["nmse"])
            block_mse_per_seed.append(float(np.mean(cell_mses)))
            block_nmse_per_seed.append(float(np.mean(cell_nmses)))

        # Column widths so mean±SE blocks line up.
        col_w = 22 if n_seeds > 1 else 12
        lines.append(f"\n── {name} ──")
        lines.append(
            f"{'action_dim':<20} {'MSE':>{col_w}} {'NMSE':>{col_w}}"
        )
        for (a_name, d_name), vals in per_cell.items():
            mse_m, mse_s = _mean_se(vals["mse"])
            nmse_m, nmse_s = _mean_se(vals["nmse"])
            lines.append(
                f"{a_name+'_'+d_name:<20} "
                f"{fmt(mse_m, mse_s):>{col_w}} "
                f"{fmt(nmse_m, nmse_s):>{col_w}}"
            )
        block_mse_mean, block_mse_se = _mean_se(block_mse_per_seed)
        block_nmse_mean, block_nmse_se = _mean_se(block_nmse_per_seed)
        lines.append(
            f"Avg {name}: MSE={fmt(block_mse_mean, block_mse_se)} "
            f"NMSE={fmt(block_nmse_mean, block_nmse_se)}"
        )
        summaries.append((name, block_mse_mean, block_nmse_mean))

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nSaved {out_path}")
    return summaries


def _plot_quiver_n_series(
    base,
    series,
    G0, G1,
    grid_shape,
    state_labels,
    action_names,
    d0, d1,
    state_ranges,
    save_path,
    stride=20,
    action_idx=None,
    scale_multiplier=12.0,
    unit_arrows=False,
):
    """Quiver overlay of N displacement fields, one panel per action.

    series : list of (preds_by_action, color, label) or
             (preds_by_action, color, label, alpha) tuples in z-order
             (first element drawn underneath, last on top).
    action_idx : if set, plot only this single action panel; else all actions.
    """
    series = [(s + (0.85,)) if len(s) == 3 else s for s in series]
    if action_idx is not None:
        action_names = [action_names[action_idx]]
        series = [
            ([preds[action_idx]], color, label, alpha)
            for preds, color, label, alpha in series
        ]
    n_actions = len(action_names)
    n_h, n_w = grid_shape
    idx_h = np.unique(np.append(np.arange(0, n_h, stride), n_h - 1))
    idx_w = np.unique(np.append(np.arange(0, n_w, stride), n_w - 1))

    range_d0 = float(G0.max() - G0.min()) or 1.0
    range_d1 = float(G1.max() - G1.min()) or 1.0
    base_np = np.asarray(base)

    # Shared arrow scale across all actions and series.
    idx_w_inner = idx_w[1:]
    all_mag = []
    for a in range(n_actions):
        for preds_by_action, _, _, _ in series:
            s_arr = preds_by_action[a]
            dd0 = (s_arr[:, d0] - base_np[:, d0]).reshape(n_h, n_w)[np.ix_(idx_h, idx_w_inner)] / range_d0
            dd1 = (s_arr[:, d1] - base_np[:, d1]).reshape(n_h, n_w)[np.ix_(idx_h, idx_w_inner)] / range_d1
            all_mag.append(np.sqrt(dd0**2 + dd1**2))
    max_mag = float(np.max(np.concatenate([m.flatten() for m in all_mag]))) or 1.0
    # Unit-arrow mode has magnitude 1, so no max_mag factor.
    scale = scale_multiplier if unit_arrows else max_mag * scale_multiplier

    fig, axes = plt.subplots(1, n_actions, figsize=(7 * n_actions, 4.5), sharey=True)
    if n_actions == 1:
        axes = [axes]

    x = np.asarray(G0)[np.ix_(idx_h, idx_w)]
    y = np.asarray(G1)[np.ix_(idx_h, idx_w)]

    def _norm(s_arr):
        dd0 = (s_arr[:, d0] - base_np[:, d0]).reshape(n_h, n_w)[np.ix_(idx_h, idx_w)] / range_d0
        dd1 = (s_arr[:, d1] - base_np[:, d1]).reshape(n_h, n_w)[np.ix_(idx_h, idx_w)] / range_d1
        if unit_arrows:
            mag = np.sqrt(dd0**2 + dd1**2)
            return dd0 / np.maximum(mag, 1e-10), dd1 / np.maximum(mag, 1e-10)
        mag = np.sqrt(dd0**2 + dd1**2)
        factor = np.where(mag > max_mag, max_mag / np.maximum(mag, 1e-10), 1.0)
        return dd0 * factor, dd1 * factor

    # MountainCar tick locations (d0=position, d1=velocity).
    xticks = [-1.2, -0.6, 0.0, 0.6]
    yticks = [-0.06, -0.03, 0.0, 0.03, 0.06]

    for a, ax in enumerate(axes):
        for z, (preds_by_action, color, label, alpha) in enumerate(series):
            dd0, dd1 = _norm(preds_by_action[a])
            ax.quiver(x, y, dd0, dd1, color=color, alpha=alpha,
                      label=label, width=0.004, headwidth=3, headlength=2.5, headaxislength=2.5,
                      scale=scale, zorder=1 + z)
        ax.set_xlabel(state_labels[d0], fontsize=14)
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelsize=12)
        if a == 0:
            ax.set_ylabel(state_labels[d1], fontsize=14)
            ax.set_yticks(yticks)
            ax.tick_params(axis="y", labelsize=12)
        else:
            ax.set_yticks(yticks)
            ax.tick_params(axis="y", labelleft=False, left=False)
        if state_ranges is not None:
            lo0, hi0 = state_ranges[d0]
            lo1, hi1 = state_ranges[d1]
            pad0 = (hi0 - lo0) * 0.03
            pad1 = (hi1 - lo1) * 0.03
            ax.set_xlim(lo0 - pad0, hi0 + pad0)
            ax.set_ylim(lo1 - pad1, hi1 + pad1)
        ax.legend(fontsize=12, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(f"Saved {save_path}")


# Color palette: true=green underneath (translucent), agents=red/blue on top.
COLOR_TRUE_QUIVER = "#2ca02c"
COLOR_AGENT_1     = "#d62728"
COLOR_AGENT_2     = "#1f77b4"
ALPHA_TRUE        = 0.7
ALPHA_AGENT       = 0.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc_position_sweep", default=None,
                    help="seeds_<TS> dir for mountaincar-position. If omitted, "
                         "a fresh sweep is trained (see --num_seeds).")
    ap.add_argument("--mc_velocity_sweep", default=None,
                    help="seeds_<TS> dir for mountaincar-velocity. If omitted, "
                         "a fresh sweep is trained (see --num_seeds).")
    ap.add_argument("--num_seeds", type=int, default=1,
                    help="Seeds to train per env when auto-training. Ignored "
                         "when --mc_*_sweep is passed. Fig 5 only consumes "
                         "seed_0, so the default of 1 is sufficient.")
    ap.add_argument("--label_a", default="position")
    ap.add_argument("--label_b", default="velocity")
    ap.add_argument("--out_dir", default="outputs/results",
                    help="Where to write the PNG and the per-(action, dim) "
                         "MSE/MAPE metrics file. Default: outputs/results/.")
    ap.add_argument("--n_heat", type=int, default=N_HEAT)
    ap.add_argument("--stride", type=int, default=20,
                    help="Subsample stride for arrow grid (higher = fewer arrows).")
    ap.add_argument("--action_idx", type=int, default=2,
                    help="Plot only this action panel (0=left, 1=none, "
                         "2=right). Default 2 matches paper Fig 5. Pass -1 "
                         "to render all three action panels instead.")
    ap.add_argument("--arrow_scale", type=float, default=5.0,
                    help="Scale multiplier for arrow size (lower = larger arrows).")
    ap.add_argument("--unit_arrows", action="store_true",
                    help="Normalise every arrow to unit length (direction only).")
    args = ap.parse_args()

    if args.mc_position_sweep is None:
        args.mc_position_sweep = _train_sweep("mountaincar-position", args.num_seeds)
    if args.mc_velocity_sweep is None:
        args.mc_velocity_sweep = _train_sweep("mountaincar-velocity", args.num_seeds)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = f"{out_dir}/fig5_mountaincar_quiver.txt"

    # Discover the seeds present in BOTH sweeps. Stats average across these;
    # the plot itself always uses the lowest-index seed (typically seed_0).
    shared_seeds = sorted(
        _discover_seeds(args.mc_position_sweep)
        & _discover_seeds(args.mc_velocity_sweep)
    )
    if not shared_seeds:
        raise FileNotFoundError(
            f"No overlapping seed_i dirs between {args.mc_position_sweep} "
            f"and {args.mc_velocity_sweep}"
        )
    plot_seed = shared_seeds[0]
    print(f"\nCross-WM stats over seeds {shared_seeds} "
          f"(plot uses seed_{plot_seed}).")

    state_dim = 2
    action_dim = 3
    state_labels = STATE_LABELS_MC
    action_names = ACTION_NAMES_MC
    state_ranges = STATE_RANGES_MC
    d0, d1 = 0, 1  # position vs velocity

    g0 = jnp.linspace(*state_ranges[d0], args.n_heat)
    g1 = jnp.linspace(*state_ranges[d1], args.n_heat)
    G0, G1 = jnp.meshgrid(g0, g1)
    base = jnp.stack([G0.flatten(), G1.flatten()], axis=-1)
    grid_shape = (args.n_heat, args.n_heat)

    # Ground-truth dynamics (seed-independent).
    base_env, env_params = gymnax.make("MountainCar-v0")
    env = MountainCar()
    dynamics_fn = make_env_dynamics_fn(env, env_params, "MountainCar")
    pred_true = []
    for a in range(action_dim):
        actions = jnp.full(args.n_heat * args.n_heat, a)
        pred_true.append(np.asarray(dynamics_fn(base, actions)))

    pred_a_per_seed, pred_b_per_seed = [], []
    pred_a_plot = pred_b_plot = None
    for s in shared_seeds:
        wm_a = _seed_wm_dir(args.mc_position_sweep, s)
        wm_b = _seed_wm_dir(args.mc_velocity_sweep, s)
        p_a, _, cfg_a = _load_wm_with_config(wm_a)
        p_b, _, cfg_b = _load_wm_with_config(wm_b)
        if s == plot_seed:
            print(f"  seed {s}: plotting from\n    A={wm_a}\n    B={wm_b}")
            for k in ("DENSE_HIDDEN_SIZE", "DENSE_LAYERS",
                      "ACTIVATION", "RESIDUAL_PREDICTION"):
                if cfg_a.get(k) != cfg_b.get(k):
                    print(f"    ⚠ WM_CONFIG[{k}] differs: "
                          f"A={cfg_a.get(k)} B={cfg_b.get(k)}")
        pred_a_seed = _predict_grid(p_a, cfg_a, state_dim, action_dim, base)
        pred_b_seed = _predict_grid(p_b, cfg_b, state_dim, action_dim, base)
        pred_a_per_seed.append(pred_a_seed)
        pred_b_per_seed.append(pred_b_seed)
        if s == plot_seed:
            pred_a_plot, pred_b_plot = pred_a_seed, pred_b_seed

    _print_and_write_metrics(
        pred_a_per_seed, pred_b_per_seed, pred_true,
        state_labels, action_names,
        out_path=metrics_path,
        label_a=args.label_a, label_b=args.label_b,
    )

    # Three-way overlay including ground truth (Fig. 5) — uses seed_0's preds.
    _plot_quiver_n_series(
        base,
        series=[
            (pred_true,   COLOR_TRUE_QUIVER, "WM (true)",    ALPHA_TRUE),
            (pred_a_plot, COLOR_AGENT_1,     "WM (agent 1)", ALPHA_AGENT),
            (pred_b_plot, COLOR_AGENT_2,     "WM (agent 2)", ALPHA_AGENT),
        ],
        G0=G0, G1=G1, grid_shape=grid_shape,
        state_labels=state_labels, action_names=action_names,
        d0=d0, d1=d1, state_ranges=state_ranges,
        save_path=f"{out_dir}/fig5_mountaincar_quiver.png", stride=args.stride,
        action_idx=(None if args.action_idx < 0 else args.action_idx),
        scale_multiplier=args.arrow_scale,
        unit_arrows=args.unit_arrows,
    )

    print(f"\nDone. PNG: {out_dir}/fig5_mountaincar_quiver.png")
    print(f"      metrics: {metrics_path}")


if __name__ == "__main__":
    main()
