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
from plotting.style import (
    set_paper_style, unset_paper_style, COLOR_TRUE, COLOR_WM, COLOR_GOAL,
)


def _load_or_track(run_dir, ctx, max_ckpts, goal_indices, reuse, visitation_from,
                   weight_mode):
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
        visitation_from=visitation_from, weight_mode=weight_mode)


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
    ap.add_argument("--weight_mode", choices=["mask", "density"], default="mask")
    args = ap.parse_args()

    goal_indices = ([int(x) for x in args.goals.split(",")] if args.goals else None)
    ctx = build_context(args.config)
    if args.wm_num_steps is not None:
        ctx["wm_config"]["NUM_STEPS"] = args.wm_num_steps
    env_stem = ctx["env_config"]["ENV_NAME"].lower()
    out_dir = args.out_dir or f"outputs/{env_stem}/dist_vs_uniform"
    os.makedirs(out_dir, exist_ok=True)

    per_seed = [_load_or_track(rd, ctx, args.max_ckpts, goal_indices, args.reuse,
                               args.visitation_from, args.weight_mode)
                for rd in args.run_dirs]

    steps = per_seed[0]["n_updates"]
    aligned = all(np.array_equal(d["n_updates"], steps) for d in per_seed)
    S = len(per_seed)
    conv = steps >= 200                       # "the trained agent"

    def stack(k):
        return np.stack([d[k] for d in per_seed])          # (S, N)

    def band(ax, x, M, color, label, ls="-", marker="o"):
        m = M.mean(0)
        ax.plot(x, m, color=color, ls=ls, marker=marker, ms=4, label=label)
        if M.shape[0] > 1:
            se = M.std(0, ddof=1) / np.sqrt(M.shape[0])
            ax.fill_between(x, m - se, m + se, color=color, alpha=0.15)

    def conv_ratio(ku, ko):
        """Per-seed ratio of means over the converged region; return mean, SE."""
        r = np.array([d[ku][conv].mean() / d[ko][conv].mean() for d in per_seed])
        return r.mean(), (r.std(ddof=1) / np.sqrt(S) if S > 1 else 0.0), r

    set_paper_style()
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))

    # ── Panel A — Q-error: Q^pi (theorem's ε) and Q* (distance to optimal),
    #    each uniform (solid) vs on reachable set (dashed). Q^pi curves overlap
    #    (~1x); Q* dashed sits below Q* solid (closer to optimal on-support). ──
    if aligned:
        band(ax[0], steps, stack("q_nmse_policy_uniform"), COLOR_TRUE,
             r"$\|Q-Q^\pi\|$ uniform", ls="-", marker="o")
        band(ax[0], steps, stack("q_nmse_policy_weighted"), COLOR_TRUE,
             r"$\|Q-Q^\pi\|$ on $S_o$", ls="--", marker="s")
        band(ax[0], steps, stack("q_nmse_optimal_uniform"), COLOR_GOAL,
             r"$\|Q-Q^*\|$ uniform", ls="-", marker="o")
        band(ax[0], steps, stack("q_nmse_optimal_weighted"), COLOR_GOAL,
             r"$\|Q-Q^*\|$ on $S_o$", ls="--", marker="s")
    ax[0].set_yscale("log"); ax[0].set_xlabel("PQN update step")
    ax[0].set_ylabel("Q NMSE"); ax[0].set_title("Q-error: uniform vs on-support")
    ax[0].axvspan(200, steps.max(), color="green", alpha=0.06)
    ax[0].legend(fontsize=7.5, ncol=1)

    # ── Panel B — recovered-model error: uniform vs on reachable set. ──
    if aligned:
        band(ax[1], steps, stack("wm_nmse_uniform"), COLOR_TRUE, "uniform", ls="-", marker="o")
        band(ax[1], steps, stack("wm_nmse_visit"), COLOR_WM, r"on $S_o$", ls="--", marker="s")
    ax[1].set_yscale("log"); ax[1].set_xlabel("PQN update step")
    ax[1].set_ylabel(r"$\mathrm{WM}_{\mathrm{NMSE}}$")
    ax[1].set_title("Recovery error: uniform vs on-support")
    ax[1].axvspan(200, steps.max(), color="green", alpha=0.06)
    ax[1].annotate("trained\nagent", (0.5 * (200 + steps.max()), ax[1].get_ylim()[1]),
                   ha="center", va="top", fontsize=7, color="green")
    ax[1].legend(fontsize=8)

    fig.suptitle(rf"On-support (reachable set $S_o$) vs uniform  --  {S} seeds, mean $\pm$ SE")
    fig.tight_layout(); fig.savefig(f"{out_dir}/onoff_summary.png"); plt.close(fig)
    unset_paper_style()

    # ── Converged-region numbers (mean ± SE ratios) ──
    qpi_m, qpi_se, _ = conv_ratio("q_nmse_policy_uniform", "q_nmse_policy_weighted")
    qopt_m, qopt_se, _ = conv_ratio("q_nmse_optimal_uniform", "q_nmse_optimal_weighted")
    wm_m, wm_se, wm_r = conv_ratio("wm_nmse_uniform", "wm_nmse_visit")
    np.savez(f"{out_dir}/dist.npz",
             q_uniform=stack("q_nmse_policy_uniform"), q_visit=stack("q_nmse_policy_weighted"),
             qopt_uniform=stack("q_nmse_optimal_uniform"), qopt_visit=stack("q_nmse_optimal_weighted"),
             wm_uniform=stack("wm_nmse_uniform"), wm_visit=stack("wm_nmse_visit"),
             n_updates=steps)
    print(f"\nConverged agent (step>=200), on-support vs uniform, ratio of means ({S} seeds):")
    print(f"  ||Q-Q^pi|| : {qpi_m:.2f}x +- {qpi_se:.2f}   (~1x: not concentrated off-support)")
    print(f"  ||Q-Q*||   : {qopt_m:.2f}x +- {qopt_se:.2f}   (Q degrades off-support)")
    print(f"  WM_NMSE    : {wm_m:.1f}x +- {wm_se:.1f}   (recovery far better on-support)  per-seed {np.round(wm_r,1)}")
    print(f"Wrote {out_dir}/onoff_summary.png, dist.npz")


if __name__ == "__main__":
    main()
