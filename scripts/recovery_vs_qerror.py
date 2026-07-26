"""A1 — World-model recovery quality vs Q-function estimation error.

For each PQN checkpoint of one or more runs, compute Q_NMSE (estimation error)
and WM_NMSE (recovery quality) via eval/track_recovery.py, then plot recovery
against error. Answers reviewer 1 ("How does recovery quality scale with
Q-function estimation error?").

Usage:
    uv run python scripts/recovery_vs_qerror.py \
        --config configs/mountaincar-position.py \
        --run_dirs outputs/mountaincar-position/seed_0 \
        [--max_ckpts N] [--goals 0,1,2,3] [--out_dir DIR] [--reuse]
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
from eval.aggregate_arch_sweep import plot_mean_se
from plotting.style import set_paper_style, unset_paper_style, COLOR_TRUE, COLOR_WM


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--max_ckpts", type=int, default=None)
    ap.add_argument("--goals", type=str, default=None, help="comma list of goal indices")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--reuse", action="store_true", help="reuse existing npz if present")
    ap.add_argument("--wm_num_steps", type=int, default=None,
                    help="override WM_CONFIG NUM_STEPS (e.g. small value for a smoke test)")
    ap.add_argument("--wm_batch_size", type=int, default=None,
                    help="override WM_CONFIG BATCH_SIZE (smaller = cheaper/step on CPU)")
    ap.add_argument("--visitation_from", choices=["self", "final"], default="self",
                    help="weight each checkpoint by its OWN policy visitation (self) "
                         "or by the converged agent's visitation (final)")
    ap.add_argument("--weight_mode", choices=["mask", "density"], default="mask",
                    help="on-support weighting: reachable-set mask (default, "
                         "theory-faithful, low-noise) or raw visitation density")
    args = ap.parse_args()

    goal_indices = ([int(x) for x in args.goals.split(",")] if args.goals else None)
    ctx = build_context(args.config)
    if args.wm_num_steps is not None:
        ctx["wm_config"]["NUM_STEPS"] = args.wm_num_steps
    if args.wm_batch_size is not None:
        ctx["wm_config"]["BATCH_SIZE"] = args.wm_batch_size
    env_stem = ctx["env_config"]["ENV_NAME"].lower()
    out_dir = args.out_dir or f"outputs/{env_stem}/recovery_vs_qerror"
    os.makedirs(out_dir, exist_ok=True)

    per_seed = [_load_or_track(rd, ctx, args.max_ckpts, goal_indices, args.reuse,
                               args.visitation_from, args.weight_mode)
                for rd in args.run_dirs]

    # Pool points for the scatter + Spearman.
    q_all = np.concatenate([d["q_nmse_policy_uniform"] for d in per_seed])
    wm_all = np.concatenate([d["wm_nmse_uniform"] for d in per_seed])
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(q_all, wm_all)
    except Exception:
        rho, pval = float("nan"), float("nan")

    set_paper_style()
    # ── scatter: WM_NMSE vs Q_NMSE (log-log) ──
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for i, d in enumerate(per_seed):
        ax.scatter(d["q_nmse_policy_uniform"], d["wm_nmse_uniform"],
                   s=36, alpha=0.85, edgecolor="white", linewidth=0.6,
                   label=(f"seed {i}" if len(per_seed) > 1 else None))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$Q_{\mathrm{NMSE}}$ (estimation error)")
    ax.set_ylabel(r"$\mathrm{WM}_{\mathrm{NMSE}}$ (recovery error)")
    ax.set_title(rf"Recovery vs Q-error  ($\rho_s={rho:.2f}$)")
    if len(per_seed) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out_dir}/scatter.png"); plt.close(fig)

    # ── curves vs training step (mean±SE where seeds align) ──
    steps = per_seed[0]["n_updates"]
    aligned = all(np.array_equal(d["n_updates"], steps) for d in per_seed)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    if aligned:
        q_stack = np.stack([d["q_nmse_policy_uniform"] for d in per_seed])
        wm_stack = np.stack([d["wm_nmse_uniform"] for d in per_seed])
        plot_mean_se(ax, steps, q_stack, COLOR_TRUE, r"$Q_{\mathrm{NMSE}}$", marker="o")
        plot_mean_se(ax, steps, wm_stack, COLOR_WM, r"$\mathrm{WM}_{\mathrm{NMSE}}$", marker="s")
    else:
        for d in per_seed:
            ax.plot(d["n_updates"], d["q_nmse_policy_uniform"], color=COLOR_TRUE, marker="o")
            ax.plot(d["n_updates"], d["wm_nmse_uniform"], color=COLOR_WM, marker="s")
    ax.set_yscale("log"); ax.set_xlabel("PQN update step (checkpoint)")
    ax.set_ylabel("NMSE"); ax.set_title("Q-error and recovery error vs training")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{out_dir}/curves_vs_step.png"); plt.close(fig)
    unset_paper_style()

    np.savez(f"{out_dir}/metrics.npz",
             q_nmse=q_all, wm_nmse=wm_all, spearman_rho=rho, spearman_p=pval)
    print(f"\nSpearman rho(WM_NMSE, Q_NMSE) = {rho:.3f} (p={pval:.2g}), n={q_all.size}")
    print(f"Wrote {out_dir}/scatter.png, curves_vs_step.png, metrics.npz")


if __name__ == "__main__":
    main()
