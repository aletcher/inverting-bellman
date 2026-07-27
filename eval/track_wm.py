"""Per-checkpoint world-model tracking.

For each PQN intermediate checkpoint step_*.pkl produced by training/pqn.py,
train a fresh world model via training.wm.train_world_model and record the
final dynamics MSE. Used by scripts/architecture_sweep.py for the "WM accuracy
vs PQN training step" curves (paper §G.2 fig. 4).
"""

import glob
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np


def track_wm_for_run(
    run_dir,
    pqn_config,
    wm_config,
    env_config,
    basic_env,
    env_params,
    goals,
    goal_masks,
    env_terminated_fn=None,
    state_to_eff_fn=None,
    eff_to_obs_fn=None,
    wm_output_dim=None,
    save_wm_ckpts=False,
    use_wandb=False,
    wandb_step_metric_name="wm_track/pqn_n_updates",
    eval_batch_size=4096,
    subdir="wm_track",
    final_only=False,
):
    """Train a fresh WM on every step_*.pkl checkpoint in run_dir.

    Args:
        run_dir: cell directory containing checkpoints/step_*.pkl.
        pqn_config, wm_config, env_config: live configs (not the saved JSON).
        basic_env, env_params: built env for WM dynamics target + reset sampler.
        goals: ContinuousGoal pytree (target_state, reward_mask).
        goal_masks: REWARD_MASK array (passed directly to train_world_model).
        env_terminated_fn: None for envs without termination (Reacher).
        state_to_eff_fn, eff_to_obs_fn, wm_output_dim: effective-state lift
            functions (Reacher -> 4D effective + FK lift to 6D obs).
        save_wm_ckpts: if True, pickle WM params per checkpoint.
        use_wandb: if True, log per-checkpoint metrics into the active wandb run.
        wandb_step_metric_name: custom step axis for wm_track/* metrics.
        eval_batch_size: # (s, a) pairs used to compute final dyn MSE per ckpt.
        subdir: per-cell output subdirectory name. Override (e.g.
            "wm_track_policy") to keep multiple WM-tracking variants alongside
            each other without overwriting.

    Returns:
        dict with arrays {n_updates, env_steps, final_dyn_mse, wm_loss_final}.
        Also np.savez'd to f"{run_dir}/{subdir}/tracking.npz".
    """
    from training.wm import train_world_model
    from envs.env_dynamics import make_env_dynamics_fn

    env_name = env_config["ENV_NAME"]
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    out_dir = os.path.join(run_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Persist wm_config alongside the output for reproducible WM loading later.
    from configs.utils import save_config as _save_config
    _save_config(wm_config, os.path.join(out_dir, "wm_config.json"))

    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "step_*.pkl")))
    if not ckpt_files:
        raise FileNotFoundError(
            f"No PQN checkpoints found at {checkpoint_dir}/step_*.pkl"
        )
    if final_only:
        ckpt_files = ckpt_files[-1:]

    print(f"\n── WM tracking ({len(ckpt_files)} checkpoint{'s' if len(ckpt_files) > 1 else ''} from {checkpoint_dir}) ──")

    # True dynamics function for eval target (same across checkpoints).
    eval_dynamics_fn = make_env_dynamics_fn(basic_env, env_params, env_name)

    # WM state sampler (same logic as run.py). Reacher with SAMPLE_FROM_RESET=True
    # is the headline path.
    from envs.samplers import make_reacher_uniform_sampler, make_env_reset_sampler
    if wm_config.get("SAMPLE_FROM_RESET", False):
        wm_sample_fn = make_env_reset_sampler(basic_env, env_params)
    elif env_name == "Reacher":
        wm_sample_fn = make_reacher_uniform_sampler(pqn_config["STATE_RANGES"])
    else:
        # MountainCar: train_world_model falls through to its built-in uniform
        # sampler over STATE_RANGES when sample_states_fn is None.
        wm_sample_fn = None

    # wandb custom step metric for the wm_track run.
    if use_wandb:
        import wandb as _wandb
        _wandb.define_metric(wandb_step_metric_name)
        _wandb.define_metric("wm_track/*", step_metric=wandb_step_metric_name)

    n_updates_list = []
    env_steps_list = []
    final_dyn_list = []
    wm_loss_final_list = []

    num_envs = pqn_config["NUM_ENVS"]
    num_steps_pqn = pqn_config["NUM_STEPS"]

    # One-shot end-of-training dynamics MSE: |WM(s, a) - true(s, a)|^2 averaged
    # over eval_batch_size random (s, a) pairs. Replaces per-step tracking.
    from training.wm import make_world_model, apply_wm

    @jax.jit
    def _eval_dyn_mse(p_params, batch_s, batch_a):
        out_dim = wm_output_dim if wm_output_dim is not None else pqn_config["STATE_DIM"]
        p_model = make_world_model(wm_config, out_dim)
        residual = wm_config.get("RESIDUAL_PREDICTION", True)
        angle_dims = wm_config.get("ANGLE_DIMS")
        wm_input_dims = wm_config.get("WM_INPUT_DIMS")
        a_oh = jax.nn.one_hot(batch_a, pqn_config["ACTION_DIM"])
        s_pred = apply_wm(
            p_model, p_params, batch_s, a_oh, residual=residual,
            state_to_eff_fn=state_to_eff_fn if eff_to_obs_fn is not None else None,
            angle_dims=angle_dims, eff_to_obs_fn=eff_to_obs_fn,
            wm_input_dims=wm_input_dims,
        )
        s_true = eval_dynamics_fn(batch_s, batch_a)
        return jnp.mean((s_pred - s_true) ** 2)

    for ckpt_idx, ckpt_path in enumerate(ckpt_files):
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        q_params = ckpt["params"]
        q_batch_stats = ckpt["batch_stats"]
        n_updates = int(ckpt["n_updates"])
        env_steps = n_updates * num_envs * num_steps_pqn

        per_ckpt_dir = os.path.join(out_dir, f"step_{n_updates:06d}")
        if save_wm_ckpts:
            os.makedirs(per_ckpt_dir, exist_ok=True)

        t0 = time.time()
        p_params, step_losses = train_world_model(
            q_params,
            q_batch_stats,
            wm_config,
            pqn_config,
            goals,
            goal_masks,
            env_terminated_fn=env_terminated_fn,
            sample_states_fn=wm_sample_fn,
            use_wandb=False,  # we log to the wm_track wandb run ourselves
            state_to_eff_fn=state_to_eff_fn,
            eff_to_obs_fn=eff_to_obs_fn,
            wm_output_dim=wm_output_dim,
        )
        dt = time.time() - t0

        step_losses = jax.block_until_ready(step_losses)
        loss_arr = np.asarray(step_losses)
        final_loss = float(loss_arr[-1])

        # One-shot post-training dynamics MSE on a fresh batch.
        rng_eval = jax.random.PRNGKey(
            int(wm_config.get("SEED", 0)) + ckpt_idx + 1
        )
        if wm_sample_fn is not None:
            rng_s, rng_a = jax.random.split(rng_eval)
            eval_states = wm_sample_fn(rng_s, eval_batch_size)
        else:
            from training.wm import sample_states
            rng_s, rng_a = jax.random.split(rng_eval)
            eval_states = sample_states(
                rng_s, pqn_config["STATE_RANGES"],
                pqn_config["STATE_DIM"], eval_batch_size,
            )
        eval_actions = jax.random.randint(
            rng_a, (eval_batch_size,), 0, pqn_config["ACTION_DIM"],
        )
        final_dyn = float(jax.block_until_ready(
            _eval_dyn_mse(p_params, eval_states, eval_actions)
        ))

        n_updates_list.append(n_updates)
        env_steps_list.append(env_steps)
        final_dyn_list.append(final_dyn)
        wm_loss_final_list.append(final_loss)

        print(
            f"  [{ckpt_idx + 1}/{len(ckpt_files)}] step={n_updates:6d} "
            f"env_steps={env_steps:>9d}  dyn_mse={final_dyn:.6f}  "
            f"loss={final_loss:.6f}  ({dt:.1f}s)"
        )

        if save_wm_ckpts:
            with open(os.path.join(per_ckpt_dir, "wm_checkpoint.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "params": jax.tree.map(np.array, p_params),
                        "step_losses": loss_arr,
                        "final_dyn_mse": final_dyn,
                        "n_updates": n_updates,
                    },
                    f,
                )

        if use_wandb:
            import wandb as _wandb
            _wandb.log({
                wandb_step_metric_name: n_updates,
                "wm_track/dyn_mse_final": final_dyn,
                "wm_track/wm_loss_final": final_loss,
                "wm_track/env_steps": env_steps,
            })

    result = {
        "n_updates": np.array(n_updates_list, dtype=np.int64),
        "env_steps": np.array(env_steps_list, dtype=np.int64),
        "final_dyn_mse": np.array(final_dyn_list, dtype=np.float64),
        "wm_loss_final": np.array(wm_loss_final_list, dtype=np.float64),
    }
    npz_path = os.path.join(out_dir, "tracking.npz")
    np.savez(npz_path, **result)
    print(f"Saved WM tracking to {npz_path}")
    return result
