"""Per-cell unseen-goal evaluator for the architecture sweep.

For each (D, W, seed) cell:
  1. Ensure a final-PQN-checkpoint world model exists (train if missing via
     track_wm.track_wm_for_run with final_only=True, save_wm_ckpts=True,
     subdir='wm_track_final').
  2. Load it and build a 4D-effective-state dynamics wrapper.
  3. Run evaluate_unseen_goals on the full UNSEEN_GOALS list.
  4. Save a flat unseen_summary.npz per cell for the aggregator.

This is the "headline" planning-quality metric: how good is the policy you get
by doing exact VI on the learned WM, when targeting goals never seen during
training?
"""

import os
import pickle

import numpy as np


def run_unseen_for_cell(
    run_dir,
    pqn_config,
    wm_config,
    env_config,
    cfg,
    use_wandb=False,
    force=False,
    only_labels=None,
    wm_subdir="wm_track_final",
    unseen_subdir="unseen_goals_arch",
):
    """Run VI-on-WM unseen-goal evaluation for a single sweep cell.

    Args:
        run_dir: cell directory (must contain pqn_checkpoint.pkl + checkpoints/).
        pqn_config, wm_config, env_config: live configs (after JSON merge).
        cfg: full config dict (for REACHER_TORQUE_VALUES env-build kwargs).
        use_wandb: log per-goal metrics to the active wandb run.
        force: re-run even if unseen_summary.npz already exists.
        only_labels: optional list of goal labels to include (subset of
            env_config["UNSEEN_GOALS"]). None ⇒ all configured goals.
        wm_subdir: per-cell directory holding (or to receive) the saved WM
            checkpoint for this run. Use sampler-tagged names like
            'wm_track_final_reset' / 'wm_track_final_policy' to keep multiple
            sampler variants alongside each other without colliding.
        unseen_subdir: per-cell directory for the per-goal results +
            unseen_summary.npz. Should match the wm_subdir's sampler tag.

    Returns:
        Path to the saved unseen_summary.npz.
    """
    from envs.reacher import Reacher
    from envs.reacher_utils import (
        reacher_obs_to_effective, reacher_effective_to_obs,
        make_reacher_effective_dynamics_fn, make_reacher_wm_dynamics_fn,
    )
    from eval.unseen_goals import evaluate_unseen_goals
    from eval.track_wm import track_wm_for_run

    env_name = env_config["ENV_NAME"]
    if env_name != "Reacher":
        raise NotImplementedError(
            f"unseen_goals_arch_sweep currently only supports "
            f"ENV_NAME=Reacher (got {env_name!r})."
        )

    out_dir = os.path.join(run_dir, unseen_subdir)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "unseen_summary.npz")

    # Snapshot env_config inside the unseen subdir — captures VI_GRID_RES,
    # UNSEEN_GOALS, UNSEEN_NUM_EVAL_STARTS, etc. as-of-this-run.
    from configs.utils import save_config as _save_config
    _save_config(env_config, os.path.join(out_dir, "env_config.json"))

    if os.path.exists(summary_path) and not force:
        print(f"  Unseen cell {run_dir}: {unseen_subdir}/unseen_summary.npz "
              f"exists, skipping.")
        return summary_path

    # Cells transferred from the other cluster (without step_*.pkl) cannot be
    # WM-trained here. Skip cleanly so the launcher's `set -e` doesn't abort.
    import glob as _glob
    if not _glob.glob(os.path.join(run_dir, "checkpoints", "step_*.pkl")):
        # Allow reuse if a wm_<subdir>/ checkpoint already exists (e.g. rsync'd).
        wm_dir_exists_with_ckpt = bool(
            _glob.glob(os.path.join(run_dir, wm_subdir, "step_*",
                                    "wm_checkpoint.pkl"))
        )
        if not wm_dir_exists_with_ckpt:
            print(f"  Unseen cell {run_dir}: no step_*.pkl checkpoints AND no "
                  f"{wm_subdir}/.../wm_checkpoint.pkl — skipping (this is "
                  f"normal for cells transferred without checkpoints).")
            return None

    unseen_goals = env_config.get("UNSEEN_GOALS")
    if not unseen_goals:
        raise ValueError(
            f"No UNSEEN_GOALS in env_config for {env_name}; nothing to do."
        )
    if only_labels is not None:
        unseen_goals = [g for g in unseen_goals if g["label"] in only_labels]
        if not unseen_goals:
            raise ValueError(
                f"only_labels={only_labels!r} matched none of the configured "
                f"UNSEEN_GOALS labels {[g['label'] for g in env_config['UNSEEN_GOALS']]}"
            )

    # 1. Build env and goals (matches scripts/architecture_sweep._build_env_and_goals).
    basic_env = Reacher(
        reward_type=pqn_config["REWARD_TYPE"],
        sigma=pqn_config["REWARD_SIGMA"],
        a=pqn_config["REWARD_A"],
        max_steps_in_episode=pqn_config["MAX_STEPS_IN_EPISODE"],
        torque_values=cfg["REACHER_TORQUE_VALUES"],

    )
    env_params = basic_env.default_params

    # 2. Ensure final-checkpoint WM exists in {wm_subdir}/. Skip the ~7-15s
    # training step if a saved WM checkpoint is already there.
    wm_dir = os.path.join(run_dir, wm_subdir)
    existing_wm_ckpts = []
    if os.path.isdir(wm_dir):
        for d in sorted(os.listdir(wm_dir)):
            if d.startswith("step_") and os.path.isdir(os.path.join(wm_dir, d)):
                ckpt_path = os.path.join(wm_dir, d, "wm_checkpoint.pkl")
                if os.path.exists(ckpt_path):
                    existing_wm_ckpts.append(ckpt_path)

    if existing_wm_ckpts:
        wm_ckpt_path = existing_wm_ckpts[-1]
        print(f"  Reusing existing WM checkpoint: {wm_ckpt_path}")
    else:
        track_wm_for_run(
            run_dir,
            pqn_config,
            wm_config,
            env_config,
            basic_env,
            env_params,
            goals=pqn_config["GOALS"],
            goal_masks=pqn_config["REWARD_MASK"],
            env_terminated_fn=None,
            state_to_eff_fn=reacher_obs_to_effective,
            eff_to_obs_fn=reacher_effective_to_obs,
            wm_output_dim=wm_config.get("WM_OUTPUT_DIM"),
            save_wm_ckpts=True,
            use_wandb=False,         # don't pollute the main wandb arch run
            subdir=wm_subdir,
            final_only=True,
        )
        # Locate the just-written checkpoint.
        step_subdirs = sorted(
            d for d in os.listdir(wm_dir) if d.startswith("step_") and
            os.path.isdir(os.path.join(wm_dir, d))
        )
        if not step_subdirs:
            raise FileNotFoundError(
                f"No saved WM checkpoint under {wm_dir}/step_*/ — "
                f"track_wm_for_run with save_wm_ckpts=True should have created one."
            )
        wm_ckpt_path = os.path.join(wm_dir, step_subdirs[-1], "wm_checkpoint.pkl")
    with open(wm_ckpt_path, "rb") as f:
        wm_data = pickle.load(f)
    p_params = wm_data["params"]
    print(f"  Loaded WM from {wm_ckpt_path}")

    # 4. Build dynamics + lift functions in 4D effective space.
    true_dyn_fn = make_reacher_effective_dynamics_fn(basic_env, env_params)
    wm_dyn_fn = make_reacher_wm_dynamics_fn(
        p_params, wm_config, env_config["ACTION_DIM"],
    )

    # 5. Build eval starts via env.reset so each rollout sees a different
    # initial pose (reproducible via fixed PRNGKey).
    import jax
    n_eval_starts = int(env_config.get("UNSEEN_NUM_EVAL_STARTS", 16))
    reset_keys = jax.random.split(jax.random.PRNGKey(0), n_eval_starts)
    eval_starts_obs = jax.vmap(
        lambda k: basic_env.reset(k, env_params)[0]
    )(reset_keys)
    eval_starts_obs = jax.block_until_ready(eval_starts_obs)
    print(f"  Sampled {n_eval_starts} eval starts via basic_env.reset (PRNGKey(0))")

    # 6. Run evaluate_unseen_goals on the 4D grid. Sweep-level oracle cache
    # holds cell-independent true-policy results; first cell fills it, others
    # skip ~5 s/goal. Keyed by VI_GRID_RES (auto-invalidates on grid change);
    # STATE_RANGES / unseen-goal spec changes need manual rm -rf.
    sweep_dir = os.path.dirname(run_dir)
    oracle_cache_dir = os.path.join(
        sweep_dir, "oracle", f"grid{env_config['VI_GRID_RES']}",
    )

    print(f"\n── Unseen-goal VI for {len(unseen_goals)} goals "
          f"(grid_dim=4, grid_res={env_config['VI_GRID_RES']}, "
          f"oracle={oracle_cache_dir}) ──")
    metrics = evaluate_unseen_goals(
        p_params, wm_config, pqn_config, env_config,
        dynamics_fn=true_dyn_fn,
        env_terminated_fn=None,  # Reacher has no env-level termination
        unseen_goals=unseen_goals,
        out_dir=out_dir,
        wm_dynamics_fn=wm_dyn_fn,
        grid_state_dim=env_config["VI_STATE_DIM"],
        grid_state_ranges=env_config["VI_STATE_RANGES"],
        state_to_obs_fn=reacher_effective_to_obs,
        obs_to_grid_fn=reacher_obs_to_effective,
        eval_starts=eval_starts_obs,
        oracle_cache_dir=oracle_cache_dir,
        # Aggregator only consumes the cell-level unseen_summary.npz; the
        # default (skip_per_goal_artifacts=True) already drops per-goal
        # results.txt / V_*.npy / value_comparison.png + summary.txt.
    )

    # 7. Persist a flat per-cell summary for the aggregator.
    labels = [m["label"] for m in metrics]
    arr = lambda key: np.array([m[key] for m in metrics], dtype=np.float64)
    summary = {
        "labels": np.array(labels),
        "return_true_undisc_mean": arr("return_true_mean"),
        "return_true_undisc_std":  arr("return_true_std"),
        "return_wm_undisc_mean":   arr("return_wm_mean"),
        "return_wm_undisc_std":    arr("return_wm_std"),
        "return_true_disc_mean":   arr("disc_return_true_mean"),
        "return_true_disc_std":    arr("disc_return_true_std"),
        "return_wm_disc_mean":     arr("disc_return_wm_mean"),
        "return_wm_disc_std":      arr("disc_return_wm_std"),
    }
    np.savez(summary_path, **summary)
    print(f"  Saved unseen summary to {summary_path}")

    if use_wandb:
        import wandb as _wandb
        for m in metrics:
            label = m["label"]
            _wandb.log({
                f"unseen/{label}/return_wm_disc_mean":   m["disc_return_wm_mean"],
                f"unseen/{label}/return_true_disc_mean": m["disc_return_true_mean"],
            })
        # Aggregate over all unseen goals — the headline single number.
        _wandb.log({
            "unseen/mean_return_wm_disc":   float(summary["return_wm_disc_mean"].mean()),
            "unseen/mean_return_true_disc": float(summary["return_true_disc_mean"].mean()),
            "unseen/mean_regret_disc":
                float((summary["return_true_disc_mean"]
                       - summary["return_wm_disc_mean"]).mean()),
        })

    return summary_path
