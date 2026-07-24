"""A2 — Distributional (visitation-weighted) vs uniform Q-error and recovery.

Reviewer 2 asks whether recovery holds under a Q-error bound weighted by the
agent's visitation distribution rather than a uniform bound over all state space.
This reuses eval/track_recovery.py (which emits both *_uniform and *_visit
columns) and plots the contrast:
  (1) Q_NMSE and WM_NMSE, uniform vs visitation-weighted, across checkpoints;
  (2) whether recovery error tracks visitation-weighted Q-error more tightly
      than uniform Q-error (Spearman for each).

Usage:
    uv run python scripts/dist_vs_uniform.py \
        --config configs/mountaincar-position.py \
        --run_dirs outputs/mountaincar-position/seed_0 [--max_ckpts N] [--reuse]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.run_context import build_context
from eval.track_recovery import track_recovery_for_run
from plotting.style import set_paper_style, unset_paper_style, COLOR_TRUE, COLOR_WM


def _load_or_track(run_dir, ctx, max_ckpts, goal_indices, reuse, visitation_from):
    npz = os.path.join(run_dir, "recovery_track", "recovery_tracking.npz")
    if reuse and os.path.exists(npz):
        print(f"Reusing {npz}")
        return dict(np.load(npz))
    return track_recovery_for_run(
        run_dir, ctx["pqn_config"], ctx["wm_config"], ctx["env_config"],
        ctx["basic_env"], ctx["env_params"], ctx["goals"], ctx["goal_masks"],
        ctx["vi_dynamics_fn"], ctx["wm_dynamics_fn"],
        state_to_obs_fn=ctx["state_to_obs_fn"], obs_to_grid_fn=ctx["obs_to_grid_fn"],
        grid_state_dim=ctx["grid_state_dim"], grid_state_ranges=ctx["grid_state_ranges"],
        env_terminated_fn=ctx["env_terminated_fn"], state_to_eff_fn=ctx["state_to_eff_fn"],
        eff_to_obs_fn=ctx["eff_to_obs_fn"], wm_output_dim=ctx["wm_output_dim"],
        wm_sample_fn=ctx["wm_sample_fn"], goal_indices=goal_indices, max_ckpts=max_ckpts,
        visitation_from=visitation_from)


def _spearman(x, y):
    try:
        from scipy.stats import spearmanr
        return spearmanr(x, y)
    except Exception:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--max_ckpts", type=int, default=None)
    ap.add_argument("--goals", type=str, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--wm_num_steps", type=int, default=None)
    ap.add_argument("--visitation_from", choices=["self", "final"], default="self")
    args = ap.parse_args()

    goal_indices = ([int(x) for x in args.goals.split(",")] if args.goals else None)
    ctx = build_context(args.config)
    if args.wm_num_steps is not None:
        ctx["wm_config"]["NUM_STEPS"] = args.wm_num_steps
    env_stem = ctx["env_config"]["ENV_NAME"].lower()
    out_dir = args.out_dir or f"outputs/{env_stem}/dist_vs_uniform"
    os.makedirs(out_dir, exist_ok=True)

    per_seed = [_load_or_track(rd, ctx, args.max_ckpts, goal_indices, args.reuse,
                               args.visitation_from)
                for rd in args.run_dirs]

    def cat(k):
        return np.concatenate([d[k] for d in per_seed])

    q_u, q_v = cat("q_nmse_policy_uniform"), cat("q_nmse_policy_weighted")
    wm_u, wm_v = cat("wm_nmse_uniform"), cat("wm_nmse_visit")

    # Does recovery track on-visitation Q-error more tightly than uniform?
    rho_vv, _ = _spearman(q_v, wm_v)   # visitation Q-error ↔ visitation recovery
    rho_uv, _ = _spearman(q_u, wm_v)   # uniform Q-error ↔ visitation recovery

    set_paper_style()
    # ── Panel plot: uniform vs visitation for Q and WM across checkpoints ──
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for d in per_seed:
        s = d["n_updates"]
        axes[0].plot(s, d["q_nmse_policy_uniform"], color=COLOR_TRUE, marker="o",
                     label="uniform")
        axes[0].plot(s, d["q_nmse_policy_weighted"], color=COLOR_WM, marker="s",
                     linestyle="--", label="visitation-weighted")
        axes[1].plot(s, d["wm_nmse_uniform"], color=COLOR_TRUE, marker="o",
                     label="uniform")
        axes[1].plot(s, d["wm_nmse_visit"], color=COLOR_WM, marker="s",
                     linestyle="--", label="visitation")
    for ax, ttl in zip(axes, [r"$Q_{\mathrm{NMSE}}$", r"$\mathrm{WM}_{\mathrm{NMSE}}$"]):
        ax.set_yscale("log"); ax.set_xlabel("PQN update step"); ax.set_title(ttl)
    axes[0].set_ylabel("NMSE")
    # Visitation breadth on the WM panel: the on-support region grows as the
    # agent trains, so WM_visit can stay low while WM_uniform falls with coverage.
    if all("visit_frac" in d for d in per_seed):
        axb = axes[1].twinx(); axb.grid(False)
        for d in per_seed:
            axb.plot(d["n_updates"], 100.0 * d["visit_frac"], color="0.5",
                     linestyle=":", marker="^", markersize=4)
        axb.set_ylabel("visited cells (%)", color="0.4")
        axb.tick_params(axis="y", labelcolor="0.4")
    # de-duplicate legend labels
    h, l = axes[0].get_legend_handles_labels()
    seen = dict(zip(l, h)); axes[0].legend(seen.values(), seen.keys(), fontsize=8)
    fig.suptitle("Uniform vs visitation-weighted (Q-error and recovery)")
    fig.tight_layout(); fig.savefig(f"{out_dir}/qerr_uniform_vs_visit.png"); plt.close(fig)

    # ── Scatter: recovery vs Q-error, both weightings ──
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(q_u, wm_u, s=34, color=COLOR_TRUE, alpha=0.8, edgecolor="white",
               linewidth=0.6, label=rf"uniform ($\rho_s={rho_uv:.2f}$ vs $WM_{{visit}}$)")
    ax.scatter(q_v, wm_v, s=34, color=COLOR_WM, alpha=0.8, edgecolor="white",
               linewidth=0.6, label=rf"visitation ($\rho_s={rho_vv:.2f}$)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$Q_{\mathrm{NMSE}}$"); ax.set_ylabel(r"$\mathrm{WM}_{\mathrm{NMSE}}$")
    ax.set_title("Recovery tracks visitation-weighted Q-error")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out_dir}/recovery_tracks_visit.png"); plt.close(fig)
    unset_paper_style()

    np.savez(f"{out_dir}/dist.npz", q_uniform=q_u, q_visit=q_v,
             wm_uniform=wm_u, wm_visit=wm_v, rho_visit=rho_vv, rho_uniform=rho_uv)
    print(f"\nmean Q_NMSE  uniform={q_u.mean():.3e}  visit={q_v.mean():.3e}")
    print(f"mean WM_NMSE uniform={wm_u.mean():.3e}  visit={wm_v.mean():.3e}")
    print(f"Spearman(WM_visit, Q_visit)={rho_vv:.3f}  vs  (WM_visit, Q_uniform)={rho_uv:.3f}")
    print(f"Wrote {out_dir}/qerr_uniform_vs_visit.png, recovery_tracks_visit.png, dist.npz")


if __name__ == "__main__":
    main()
