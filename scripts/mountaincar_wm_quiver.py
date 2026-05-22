"""MountainCar quiver: position-trained vs velocity-trained WMs (paper Fig. 5).

Loads two WM checkpoints (one trained on position goals, one on velocity
goals), evaluates each on a (position × velocity) grid for every discrete
action, and writes:
  - outputs/figures/fig5_mountaincar_quiver.png — both WMs overlaid on
    ground-truth displacements, one panel per action.
  - outputs/mountaincar/quiver_compare/results.txt — per-(action, dim)
    MSE/MAPE between the two predictions plus pairwise stats vs. ground
    truth (source of the "WMs ~15× closer to each other than to truth"
    number in §5.2 of the paper).

CLI:
    uv run python scripts/mountaincar_wm_quiver.py \\
        --wm_a outputs/.../wm_<TS_pos> \\
        --wm_b outputs/.../wm_<TS_vel> \\
        --label_a position --label_b velocity

Each WM's wm_config.json is inspected for state_dim / action_dim /
residual / hidden size. Ground-truth dynamics come from gymnax.
"""

import argparse
import json
import os
import sys
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
    diff = pred_a - pred_b
    abs_a = np.abs(pred_a)
    return dict(
        mse=float(np.mean(diff ** 2)),
        mape=float(np.mean(np.where(abs_a > 1e-6, np.abs(diff) / abs_a, 0.0))),
    )


def _print_and_write_metrics(
    pred_a_by_action,
    pred_b_by_action,
    pred_true_by_action,
    state_labels,
    action_names,
    out_path,
    label_a,
    label_b,
):
    """Three blocks: A vs B, A vs true, B vs true."""
    blocks = [
        (f"{label_a}_vs_{label_b}", pred_a_by_action, pred_b_by_action),
        (f"{label_a}_vs_true",      pred_a_by_action, pred_true_by_action),
        (f"{label_b}_vs_true",      pred_b_by_action, pred_true_by_action),
    ]
    lines = []
    summaries = []
    for name, A, B in blocks:
        lines.append(f"\n── {name} ──")
        lines.append(f"{'action_dim':<20} {'MSE':>12} {'MAPE':>12}")
        all_mse, all_mape = [], []
        for a, action_name in enumerate(action_names):
            for d, dim_name in enumerate(state_labels):
                m = _metrics(A[a][:, d], B[a][:, d])
                lines.append(
                    f"{action_name+'_'+dim_name:<20} {m['mse']:>12.4e} "
                    f"{m['mape']:>12.4f}"
                )
                all_mse.append(m["mse"])
                all_mape.append(m["mape"])
        avg = (np.mean(all_mse), np.mean(all_mape))
        lines.append(f"Avg {name}: MSE={avg[0]:.4e} MAPE={avg[1]:.4f}")
        summaries.append((name, *avg))

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
                      label=label, width=0.003, scale=scale, zorder=1 + z)
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
ALPHA_TRUE        = 0.5
ALPHA_AGENT       = 0.85


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_a", required=True, help="WM dir or .pkl")
    ap.add_argument("--wm_b", required=True, help="WM dir or .pkl")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--out_dir", default="outputs/figures",
                    help="Where to write the paper PNG. Default: outputs/figures/.")
    ap.add_argument("--results_dir", default="outputs/mountaincar/quiver_compare",
                    help="Where to write the per-(action, dim) MSE/MAPE table "
                         "(results.txt). Default: outputs/mountaincar/quiver_compare/.")
    ap.add_argument("--n_heat", type=int, default=N_HEAT)
    ap.add_argument("--stride", type=int, default=20,
                    help="Subsample stride for arrow grid (higher = fewer arrows).")
    ap.add_argument("--action_idx", type=int, default=None,
                    help="If set, plot only this action panel (e.g. 2 = right). "
                         "Default: all action panels.")
    ap.add_argument("--arrow_scale", type=float, default=7.0,
                    help="Scale multiplier for arrow size (lower = larger arrows).")
    ap.add_argument("--unit_arrows", action="store_true",
                    help="Normalise every arrow to unit length (direction only).")
    args = ap.parse_args()

    out_dir = args.out_dir
    results_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    p_a, _, cfg_a = _load_wm_with_config(args.wm_a)
    p_b, _, cfg_b = _load_wm_with_config(args.wm_b)
    print(f"\nWM A ({args.label_a}) from {args.wm_a}")
    print(f"WM B ({args.label_b}) from {args.wm_b}")
    for k in ("DENSE_HIDDEN_SIZE", "DENSE_LAYERS", "ACTIVATION", "RESIDUAL_PREDICTION"):
        if cfg_a.get(k) != cfg_b.get(k):
            print(f"  ⚠ WM_CONFIG[{k}] differs: A={cfg_a.get(k)} B={cfg_b.get(k)}")

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

    # Ground-truth dynamics
    base_env, env_params = gymnax.make("MountainCar-v0")
    env = MountainCar()
    dynamics_fn = make_env_dynamics_fn(env, env_params, "MountainCar")
    pred_true = []
    for a in range(action_dim):
        actions = jnp.full(args.n_heat * args.n_heat, a)
        pred_true.append(np.asarray(dynamics_fn(base, actions)))

    pred_a = _predict_grid(p_a, cfg_a, state_dim, action_dim, base)
    pred_b = _predict_grid(p_b, cfg_b, state_dim, action_dim, base)

    _print_and_write_metrics(
        pred_a, pred_b, pred_true, state_labels, action_names,
        out_path=f"{results_dir}/results.txt",
        label_a=args.label_a, label_b=args.label_b,
    )

    # Three-way overlay including ground truth (Fig. 5).
    _plot_quiver_n_series(
        base,
        series=[
            (pred_true, COLOR_TRUE_QUIVER, "WM (true)",   ALPHA_TRUE),
            (pred_a,    COLOR_AGENT_1,     "WM (agent 1)", ALPHA_AGENT),
            (pred_b,    COLOR_AGENT_2,     "WM (agent 2)", ALPHA_AGENT),
        ],
        G0=G0, G1=G1, grid_shape=grid_shape,
        state_labels=state_labels, action_names=action_names,
        d0=d0, d1=d1, state_ranges=state_ranges,
        save_path=f"{out_dir}/fig5_mountaincar_quiver.png", stride=args.stride,
        action_idx=args.action_idx,
        scale_multiplier=args.arrow_scale,
        unit_arrows=args.unit_arrows,
    )

    print(f"\nDone. PNG: {out_dir}/fig5_mountaincar_quiver.png")
    print(f"      results: {results_dir}/results.txt")


if __name__ == "__main__":
    main()
