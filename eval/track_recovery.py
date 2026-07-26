"""Per-checkpoint recovery-vs-Q-error tracking (rebuttal experiments A1 + A2).

For each PQN intermediate checkpoint step_*.pkl (a different Q-quality level of
the same run), compute:
  - Q_NMSE of PQN's Q-values vs Q^pi (policy) and Q* (optimal), both UNIFORMLY
    over the VI grid and WEIGHTED by the agent's visitation distribution;
  - WM_NMSE of the world model recovered from that Q, scored on a UNIFORM state
    set and on a VISITATION-sampled state set.

A1 (recovery vs Q-error scaling) uses the *_uniform columns; A2 (distributional
vs uniform) contrasts *_uniform with *_visit. The visitation distribution is the
"state-action distribution used to train the agent" that reviewer 2 asks about.

Goal-independent precompute (dynamics, Q*), the jitted Bellman backup, and the
visitation weights are computed once and reused across checkpoints. Reuses:
  eval.value_iteration (precompute_dynamics, precompute_rewards_dones,
    bellman_iteration, q_nmse_by_weighting),
  eval.world_model.dynamics_nmse, eval.visitation.grid_visitation,
  training.wm.train_world_model.
"""

import glob
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from eval.value_iteration import (
    precompute_dynamics, precompute_rewards_dones, bellman_iteration,
    q_nmse_by_weighting,
)
from eval.world_model import dynamics_nmse
from eval.visitation import grid_visitation


def _weights_at_states(states_grid, w, state_ranges):
    """Look up the visitation weight of each state at its nearest grid cell.

    states_grid: (n, state_dim) in grid space. Returns (n,) per-state weights,
    used to compute the visitation-WEIGHTED WM_NMSE on the same uniform eval set
    as the uniform metric (shared scale — see eval/world_model.dynamics_nmse)."""
    R = w.shape[0]
    state_dim = w.ndim
    lin = jnp.zeros(states_grid.shape[0], dtype=jnp.int32)
    for d in range(state_dim):
        lo, hi = state_ranges[d]
        b = jnp.clip(jnp.round((states_grid[:, d] - lo) / (hi - lo) * (R - 1)),
                     0, R - 1).astype(jnp.int32)
        lin = lin * R + b
    return w.reshape(-1)[lin]


def _to_weights(w, weight_mode, mass=0.99):
    """Convert a visitation density grid `w` into the weighting used for the
    on-support metrics.

    - "mask" (default): the reachable SET S_o — a binary indicator of the cells
      covering the top `mass` fraction of visitation (dropping the rare-excursion
      tail), weighted uniformly. This matches the theory (S_o is a set, not a
      density) and is far less noisy than density-weighting, which peaks on a few
      goal-boundary cells where Q is hardest to estimate.
    - "density": the raw visitation density (previous behaviour).
    """
    if weight_mode == "density":
        return w
    if weight_mode != "mask":
        raise ValueError(f"weight_mode must be 'mask' or 'density', got {weight_mode!r}")
    flat = w.reshape(-1)
    order = jnp.argsort(flat)[::-1]
    csum = jnp.cumsum(flat[order]) / (jnp.sum(flat) + 1e-12)
    keep_sorted = csum <= mass
    # always keep at least the top cell
    keep_sorted = keep_sorted.at[0].set(True)
    mask = jnp.zeros_like(flat).at[order].set(keep_sorted.astype(flat.dtype))
    return mask.reshape(w.shape)


def track_recovery_for_run(
    run_dir, pqn_config, wm_config, env_config,
    basic_env, env_params, goals, goal_masks,
    vi_dynamics_fn, wm_dynamics_fn,
    state_to_obs_fn=None, obs_to_grid_fn=None,
    grid_state_dim=None, grid_state_ranges=None,
    env_terminated_fn=None, state_to_eff_fn=None, eff_to_obs_fn=None,
    wm_output_dim=None, wm_sample_fn=None,
    goal_indices=None, max_ckpts=None, n_visit_starts=512,
    n_wm_eval=None, subdir="recovery_track", visitation_from="self",
    weight_mode="mask",
):
    """Track Q_NMSE + WM_NMSE (uniform & visitation-weighted) per checkpoint.

    vi_dynamics_fn: (s,a)->s' in GRID space (effective for Reacher).
    wm_dynamics_fn: (s,a)->s' in OBS space, the true-dynamics target for WM_NMSE.
    state_to_obs_fn / obs_to_grid_fn: grid<->obs lifts (None for 2D envs).
    Returns dict of per-checkpoint arrays; also saved to
    f"{run_dir}/{subdir}/recovery_tracking.npz".
    """
    from training.wm import train_world_model, sample_states

    gamma = pqn_config["GAMMA"]
    action_dim = pqn_config["ACTION_DIM"]
    obs_state_dim = pqn_config["STATE_DIM"]
    state_dim = grid_state_dim if grid_state_dim is not None else obs_state_dim
    state_ranges = (grid_state_ranges if grid_state_ranges is not None
                    else env_config.get("VI_STATE_RANGES", env_config["STATE_RANGES"]))
    state_ranges = [tuple(r) for r in state_ranges]
    vi_grid_res = env_config["VI_GRID_RES"]
    vi_max_iter = env_config["VI_MAX_ITER"]
    conv = env_config.get("VI_CONVERGENCE_THRESHOLD", 0.0)
    reward_type = pqn_config["REWARD_TYPE"]
    sigma = pqn_config["REWARD_SIGMA"]
    a_thr = pqn_config["REWARD_A"]
    terminate = pqn_config["TERMINATE_ON_GOAL"]
    if goal_indices is None:
        goal_indices = list(range(len(goals)))
    if n_wm_eval is None:
        n_wm_eval = int(wm_config["EVAL_HEATMAP_RES"]) ** 2

    out_dir = os.path.join(run_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    ckpt_files = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "step_*.pkl")))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints at {run_dir}/checkpoints/step_*.pkl")
    # Drop the final checkpoint (duplicate of pqn_checkpoint.pkl).
    if len(ckpt_files) > 1:
        ckpt_files = ckpt_files[:-1]
    if max_ckpts is not None:
        ckpt_files = ckpt_files[:max_ckpts]

    def _load(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return d["params"], d["batch_stats"], int(d["n_updates"])

    # ── Goal-independent precompute (once) ──
    print(f"\n── Recovery tracking: {len(ckpt_files)} checkpoints from {run_dir} ──",
          flush=True)
    dyn_fn = jax.jit(precompute_dynamics, static_argnums=(0, 1, 2, 3))
    axis_grids, all_next_states = dyn_fn(
        vi_dynamics_fn, vi_grid_res,
        tuple(tuple(r) for r in state_ranges), action_dim)
    all_next_states = jax.block_until_ready(all_next_states)
    precompute_fn = jax.jit(precompute_rewards_dones, static_argnums=(3, 6, 7, 8, 9))
    bellman_fn = jax.jit(bellman_iteration, static_argnums=(5, 6, 7))

    # Per-goal rewards/dones + optimal Q* (checkpoint-independent).
    goal_rd, goal_qstar = {}, {}
    for gi in goal_indices:
        goal_j = jnp.asarray(goals[gi]); mask_j = jnp.asarray(goal_masks[gi])
        rew, dn = precompute_fn(all_next_states, goal_j, mask_j, reward_type,
                                sigma, a_thr, state_dim, terminate,
                                env_terminated_fn, state_to_obs_fn)
        rew = jax.block_until_ready(rew)
        _, q_star, _ = bellman_fn(all_next_states, rew, dn, axis_grids,
                                  gamma, vi_max_iter, action_dim, conv, None)
        goal_rd[gi] = (goal_j, mask_j, rew, dn)
        goal_qstar[gi] = jax.block_until_ready(q_star)

    # ── Visitation distribution. `visitation_from`:
    #   "self"  → each checkpoint weighted by ITS OWN greedy-policy visitation
    #             (the on-support region grows as the agent trains);
    #   "final" → all checkpoints weighted by the converged agent's visitation.
    # Roll out the greedy policy on true dynamics from env.reset() starts (the
    # starts are policy-independent, computed once). ──
    reset_keys = jax.random.split(jax.random.PRNGKey(0), n_visit_starts)
    start_obs = jax.block_until_ready(jax.vmap(
        lambda k: basic_env.reset(k, env_params)[0])(reset_keys))
    start_states = obs_to_grid_fn(start_obs) if obs_to_grid_fn is not None else start_obs
    goals_j = jnp.asarray(goals)[jnp.asarray(goal_indices)]
    masks_j = jnp.asarray(goal_masks)[jnp.asarray(goal_indices)]

    def _visit(qp, qbs):
        w = grid_visitation(
            qp, qbs, pqn_config, vi_dynamics_fn, goals_j, masks_j,
            start_states, state_dim, action_dim, state_ranges, vi_grid_res,
            max_steps=pqn_config["MAX_STEPS_IN_EPISODE"], a_threshold=a_thr,
            terminate_on_goal=terminate, state_to_obs_fn=state_to_obs_fn,
            obs_state_dim=obs_state_dim, env_terminated_fn=env_terminated_fn)
        return jax.block_until_ready(w)

    w_fixed = None
    if visitation_from == "final":
        rq, rbs, _ = _load(ckpt_files[-1])
        w_fixed = _visit(rq, rbs)

    # ── WM eval state set (fixed across checkpoints). WM_NMSE uniform vs
    # visitation share this set and a fixed scale; only the MSE numerator is
    # re-weighted (dynamics_nmse holds the variance denominator uniform). ──
    rng = jax.random.PRNGKey(int(wm_config.get("SEED", 0)) + 12345)
    r_u, _ = jax.random.split(rng)
    if wm_sample_fn is not None:
        eval_states_uniform = wm_sample_fn(r_u, n_wm_eval)          # obs space
    else:
        eval_states_uniform = sample_states(
            r_u, pqn_config["STATE_RANGES"], obs_state_dim, n_wm_eval)
    eval_states_grid = (obs_to_grid_fn(eval_states_uniform)
                        if obs_to_grid_fn is not None else eval_states_uniform)

    def _breadth(w):
        occ = float((w > 0).mean())                       # fraction of cells visited
        wf = w.reshape(-1)
        ent = -float(jnp.sum(jnp.where(wf > 0, wf * jnp.log(wf), 0.0)))
        return occ, ent

    rows = []
    for ci, path in enumerate(ckpt_files):
        t0 = time.time()
        q_params, q_bs, n_updates = _load(path)

        # Per-checkpoint (or fixed) visitation. `w` is the raw density (used for
        # the breadth report); `w_eff` is the weighting for the on-support metrics
        # (reachable-set mask by default — see _to_weights).
        w = w_fixed if w_fixed is not None else _visit(q_params, q_bs)
        visit_frac, visit_entropy = _breadth(w)
        w_eff = _to_weights(w, weight_mode)
        eval_weights = _weights_at_states(eval_states_grid, w_eff, state_ranges)

        # Q_NMSE averaged over goals (uniform + visitation-weighted + optimal).
        acc = {}
        for gi in goal_indices:
            goal_j, mask_j, rew, dn = goal_rd[gi]
            d = q_nmse_by_weighting(
                q_params, q_bs, goal_j, mask_j, axis_grids,
                all_next_states, rew, dn, pqn_config, action_dim, state_dim,
                bellman_fn, gamma, vi_max_iter, conv,
                Q_star=goal_qstar[gi], weights=w_eff,
                state_to_obs_fn=state_to_obs_fn, obs_state_dim=obs_state_dim)
            for k, v in d.items():
                acc.setdefault(k, []).append(v)
        qrow = {k: float(np.mean(v)) for k, v in acc.items()}

        # WM: train fresh from this Q, then score uniform + visitation.
        p_params, losses = train_world_model(
            q_params, q_bs, wm_config, pqn_config, goals, goal_masks,
            env_terminated_fn=env_terminated_fn, sample_states_fn=wm_sample_fn,
            use_wandb=False, state_to_eff_fn=state_to_eff_fn,
            eff_to_obs_fn=eff_to_obs_fn, wm_output_dim=wm_output_dim)
        wm_u, _, _ = dynamics_nmse(
            p_params, wm_config, env_config, wm_dynamics_fn, eval_states_uniform,
            action_dim, obs_state_dim, weights=None, state_to_eff_fn=state_to_eff_fn,
            eff_to_obs_fn=eff_to_obs_fn, wm_output_dim=wm_output_dim)
        wm_v, _, _ = dynamics_nmse(
            p_params, wm_config, env_config, wm_dynamics_fn, eval_states_uniform,
            action_dim, obs_state_dim, weights=eval_weights, state_to_eff_fn=state_to_eff_fn,
            eff_to_obs_fn=eff_to_obs_fn, wm_output_dim=wm_output_dim)

        row = {"n_updates": n_updates,
               "wm_nmse_uniform": float(wm_u), "wm_nmse_visit": float(wm_v),
               "visit_frac": visit_frac, "visit_entropy": visit_entropy,
               **qrow}
        rows.append(row)
        print(f"  [{ci+1}/{len(ckpt_files)}] step={n_updates:6d}  "
              f"visit={visit_frac*100:4.1f}%  "
              f"Q_NMSE u={qrow['q_nmse_policy_uniform']:.3e}/"
              f"v={qrow.get('q_nmse_policy_weighted', float('nan')):.3e}  "
              f"WM_NMSE u={wm_u:.3e}/v={wm_v:.3e}  ({time.time()-t0:.1f}s)",
              flush=True)

    keys = sorted({k for r in rows for k in r})
    result = {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}
    npz_path = os.path.join(out_dir, "recovery_tracking.npz")
    np.savez(npz_path, **result)
    print(f"Saved recovery tracking to {npz_path}", flush=True)
    return result
