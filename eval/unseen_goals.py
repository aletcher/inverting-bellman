"""Unseen-goal evaluation for continuous environments (MountainCar, Reacher).

Runs VI on true dynamics and on the WM-derived dynamics for goals not seen
during PQN/WM training, then compares V*, policy agreement, and rollout
returns. Produces the Table 1 unseen-goal numbers in the paper.
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from eval.value_iteration import (
    precompute_dynamics,
    precompute_rewards_dones,
    bellman_iteration,
    interpolate_V,
)
from eval.world_model import make_wm_dynamics_fn
from envs.goals import compute_reward, goal_achieved
from plotting.vi import (
    plot_two_value_comparison_final,
    plot_two_value_comparison_2slice_3d,
)


# ── Custom reward/done for multi-goal (closest-of-two) ──────────────────────


def _precompute_rewards_dones_multi(
    all_next_states, goals_list, mask, reward_type, sigma, a_threshold,
    state_dim, env_terminated_fn,
):
    """Rewards/dones for a closest-of-N-goals task.

    Reward = max over goals of the per-goal reward. Done = any goal achieved.
    """
    grid_shape = all_next_states.shape[1:-1]

    def _per_action(ns):
        flat = ns.reshape(-1, state_dim)
        r = jnp.zeros(flat.shape[0])
        d = jnp.zeros(flat.shape[0], dtype=bool)
        for g in goals_list:
            goal = jnp.array(g)
            r = jnp.maximum(r, compute_reward(flat, goal, mask, reward_type, sigma, a_threshold))
            d = jnp.logical_or(d, goal_achieved(flat, goal, mask, a_threshold))
        return r.reshape(grid_shape), d.astype(float).reshape(grid_shape)

    all_r, all_d = jax.vmap(lambda ns: _per_action(ns))(all_next_states)

    # Env termination (if any).
    if env_terminated_fn is not None:
        all_dones_env = jax.vmap(
            lambda ns: env_terminated_fn(ns.reshape(-1, state_dim))
            .astype(float).reshape(grid_shape)
        )(all_next_states)
        all_d = jnp.maximum(all_d, all_dones_env)

    return all_r, all_d


# ── Forbidden-region env_terminated_fn builder ──────────────────────────────


def make_forbidden_terminated_fn(forbidden_spec, state_labels, base_env_terminated_fn=None):
    """Build an env_terminated_fn that terminates on forbidden state regions.

    forbidden_spec: a single dict or a list of dicts, each with
        "dim" (label), "condition" ("lt" or "gt"), "threshold" (float).
    Example: {"dim": "velocity", "condition": "lt", "threshold": -0.05}
    Example list: [{"dim": "velocity", "condition": "lt", "threshold": -0.04},
                   {"dim": "velocity", "condition": "gt", "threshold": 0.04}]
    """
    if isinstance(forbidden_spec, dict):
        forbidden_spec = [forbidden_spec]

    specs = []
    for spec in forbidden_spec:
        specs.append((
            state_labels.index(spec["dim"]),
            spec["condition"],
            spec["threshold"],
        ))

    def terminated_fn(obs):
        forbidden = jnp.zeros(obs.shape[:-1], dtype=bool)
        for dim_idx, cond, threshold in specs:
            val = obs[..., dim_idx]
            if cond == "lt":
                forbidden = jnp.logical_or(forbidden, val < threshold)
            else:
                forbidden = jnp.logical_or(forbidden, val > threshold)
        if base_env_terminated_fn is not None:
            return jnp.logical_or(forbidden, base_env_terminated_fn(obs))
        return forbidden

    return terminated_fn


# ── Policy rollout on true dynamics ─────────────────────────────────────────


def _rollout_policy_on_true_dynamics(
    policy_Q, axis_grids, dynamics_fn, start_states,
    goal, mask, reward_type, sigma, a_threshold,
    terminate_on_goal, env_terminated_fn,
    gamma=0.99, max_steps=200,
    state_to_obs_fn=None,
):
    """Roll out a grid-VI policy on true dynamics.

    Returns (undiscounted_return, discounted_return) per starting state.

    If state_to_obs_fn is set, s and s_next live in grid space (e.g. 4D
    effective for Reacher) but reward/done/env_terminated lookups happen in obs
    space — the lift is applied just before those calls.
    """
    action_dim = policy_Q.shape[0]
    n_starts = start_states.shape[0]
    _lift = state_to_obs_fn if state_to_obs_fn is not None else (lambda s: s)

    def _step(carry, t):
        s, done, total_r, total_r_disc = carry
        q_vals = jnp.stack([
            interpolate_V(policy_Q[a], s, axis_grids)
            for a in range(action_dim)
        ], axis=-1)
        action = jnp.argmax(q_vals, axis=-1)
        s_next = dynamics_fn(s, action)
        s_next_obs = _lift(s_next)
        # Env termination (forbidden zones) BEFORE reward: entering an unsafe
        # region yields zero reward immediately.
        new_done = done
        if env_terminated_fn is not None:
            new_done = jnp.logical_or(new_done, env_terminated_fn(s_next_obs))
        r = compute_reward(s_next_obs, goal, mask, reward_type, sigma, a_threshold)
        r = jnp.where(new_done, 0.0, r)
        # Goal achievement checked after reward (reaching goal gives reward, then terminates).
        achieved = goal_achieved(s_next_obs, goal, mask, a_threshold) if terminate_on_goal else jnp.zeros(n_starts, dtype=bool)
        new_done = jnp.logical_or(new_done, achieved)
        total_r = total_r + r
        total_r_disc = total_r_disc + (gamma ** t) * r
        return (s_next, new_done, total_r, total_r_disc), None

    init = (start_states, jnp.zeros(n_starts, dtype=bool),
            jnp.zeros(n_starts), jnp.zeros(n_starts))
    (_, _, ret_undisc, ret_disc), _ = jax.lax.scan(
        _step, init, jnp.arange(max_steps))
    return ret_undisc, ret_disc


def _rollout_policy_on_true_dynamics_multi(
    policy_Q, axis_grids, dynamics_fn, start_states,
    goals_list, mask, reward_type, sigma, a_threshold,
    env_terminated_fn, gamma=0.99, max_steps=200,
):
    """Like _rollout_policy_on_true_dynamics but for closest-of-N-goals."""
    action_dim = policy_Q.shape[0]
    n_starts = start_states.shape[0]
    mask_arr = jnp.array(mask)

    def _step(carry, t):
        s, done, total_r, total_r_disc = carry
        q_vals = jnp.stack([
            interpolate_V(policy_Q[a], s, axis_grids)
            for a in range(action_dim)
        ], axis=-1)
        action = jnp.argmax(q_vals, axis=-1)
        s_next = dynamics_fn(s, action)
        # Env termination (forbidden zones) BEFORE reward.
        new_done = done
        if env_terminated_fn is not None:
            new_done = jnp.logical_or(new_done, env_terminated_fn(s_next))
        r = jnp.zeros(n_starts)
        achieved = jnp.zeros(n_starts, dtype=bool)
        for g in goals_list:
            goal = jnp.array(g)
            r = jnp.maximum(r, compute_reward(s_next, goal, mask_arr, reward_type, sigma, a_threshold))
            achieved = jnp.logical_or(achieved, goal_achieved(s_next, goal, mask_arr, a_threshold))
        r = jnp.where(new_done, 0.0, r)
        new_done = jnp.logical_or(new_done, achieved)
        total_r = total_r + r
        total_r_disc = total_r_disc + (gamma ** t) * r
        return (s_next, new_done, total_r, total_r_disc), None

    init = (start_states, jnp.zeros(n_starts, dtype=bool),
            jnp.zeros(n_starts), jnp.zeros(n_starts))
    (_, _, ret_undisc, ret_disc), _ = jax.lax.scan(
        _step, init, jnp.arange(max_steps))
    return ret_undisc, ret_disc


# ── Per-goal evaluation ─────────────────────────────────────────────────────


def evaluate_single_unseen_goal(
    unseen_goal_spec, dynamics_fn, wm_dynamics_fn, env_config, pqn_config,
    env_terminated_fn, out_dir,
    grid_state_dim=None,
    grid_state_ranges=None,
    state_to_obs_fn=None,
    obs_to_grid_fn=None,
    eval_starts=None,
    oracle_cache_dir=None,
    skip_per_goal_artifacts=True,
):
    """Evaluate one unseen goal: VI on true vs WM dynamics, compare policies + returns.

    The default behavior (all kwargs None) operates entirely in obs space and
    matches the original MountainCar codepath. To run VI on a smaller
    effective grid (e.g. 4D for Reacher) while keeping rewards/dones in
    obs space, supply:

      grid_state_dim    — dim of the VI grid (e.g. 4 for Reacher effective)
      grid_state_ranges — list of (lo, hi) tuples of length grid_state_dim
      state_to_obs_fn   — grid → obs lift (e.g. reacher_effective_to_obs)
      obs_to_grid_fn    — obs → grid projection (e.g. reacher_obs_to_effective);
                          used to project eval_starts from obs into grid coords
      eval_starts       — (n_starts, obs_dim) array of evaluation starts in obs
                          space; replaces the MountainCar-specific default.
      oracle_cache_dir  — sweep-shared dir memoizing cell-independent true-policy
                          results (V*, Q*, rewards/dones grids, true-side rollout
                          returns) per goal. Cache hit skips the ~5 s/goal
                          true-side VI + rollout. Caller invalidates manually
                          (rm -rf) if STATE_RANGES / eval_starts / specs change.
      skip_per_goal_artifacts
                       — if True (default), skip per-goal results.txt, V_true.npy,
                         V_wm.npy, value_comparison.png, trajectory_comparison.png
                         and summary.txt. Pass False from deep-dive callers
                         (run.py); arch sweep keeps default since the
                         aggregator only reads unseen_summary.npz.
    """
    obs_state_dim = env_config["STATE_DIM"]
    state_dim = grid_state_dim if grid_state_dim is not None else obs_state_dim
    action_dim = env_config["ACTION_DIM"]
    state_labels = env_config["STATE_LABELS"]
    if grid_state_ranges is not None:
        state_ranges = grid_state_ranges
    else:
        state_ranges = env_config.get("VI_STATE_RANGES", env_config["STATE_RANGES"])
    vi_grid_res = env_config["VI_GRID_RES"]
    vi_max_iter = env_config["VI_MAX_ITER"]
    convergence_threshold = env_config.get("VI_CONVERGENCE_THRESHOLD", 0.0)
    gamma = pqn_config["GAMMA"]
    max_steps = pqn_config.get("MAX_STEPS_IN_EPISODE", 200)

    label = unseen_goal_spec["label"]
    reward_type = unseen_goal_spec.get("reward_type", "sparse")
    sigma = unseen_goal_spec.get("sigma", 0.1)
    a_threshold = unseen_goal_spec.get("a", 0.1)
    is_multi = "goals" in unseen_goal_spec
    terminate_on_goal = unseen_goal_spec.get("terminate_on_goal", True)

    # Handle forbidden region.
    forbidden = unseen_goal_spec.get("forbidden")
    goal_env_terminated_fn = env_terminated_fn
    if forbidden:
        goal_env_terminated_fn = make_forbidden_terminated_fn(
            forbidden, state_labels, env_terminated_fn,
        )

    goal_dir = f"{out_dir}/{label.replace(' ', '_').replace('/', '_')}"
    os.makedirs(goal_dir, exist_ok=True)
    print(f"\n── Unseen goal: {label} ──")
    t_goal_start = time.time()

    # Oracle cache (sweep-level, cell-independent true-side artifacts). True-
    # policy results depend only on env/goal/grid/eval_starts — none of which
    # change across (D, W, seed) cells. Cache once per goal; subsequent cells
    # skip ~5 s/goal of redundant true-side VI + rollout.
    oracle_goal_dir = (
        os.path.join(oracle_cache_dir,
                     label.replace(' ', '_').replace('/', '_'))
        if oracle_cache_dir else None
    )
    _CACHE_FILES = [
        "all_rewards_true.npy", "all_dones_true.npy",
        "V_true.npy", "Q_true.npy", "pi_true.npy",
        "ret_true_undisc.npy", "ret_true_disc.npy",
    ]
    oracle_loaded = (
        oracle_goal_dir is not None
        and all(os.path.exists(os.path.join(oracle_goal_dir, f))
                for f in _CACHE_FILES)
    )

    # 1. Precompute dynamics grids. Always needed: ns_wm changes per cell;
    #    ns_true is needed for V^π_wm even when the true-side is cached.
    t0 = time.time()
    dyn_fn_jit = jax.jit(precompute_dynamics, static_argnums=(0, 1, 2, 3))
    sr_tuple = tuple(tuple(r) for r in state_ranges)
    axis_grids, ns_true = dyn_fn_jit(dynamics_fn, vi_grid_res, sr_tuple, action_dim)
    axis_grids, ns_wm = dyn_fn_jit(wm_dynamics_fn, vi_grid_res, sr_tuple, action_dim)
    ns_true = jax.block_until_ready(ns_true)
    ns_wm = jax.block_until_ready(ns_wm)

    # Forbidden-region zero-out for rewards (multi-action grid only).
    if forbidden:
        forbidden_fn = make_forbidden_terminated_fn(forbidden, state_labels)
        def _forbidden_mask(all_next_states):
            def _per_action(ns):
                flat = ns.reshape(-1, state_dim)
                if state_to_obs_fn is not None:
                    flat = state_to_obs_fn(flat)
                return (forbidden_fn(flat).astype(float)
                        .reshape(all_next_states.shape[1:-1]))
            return jax.vmap(_per_action)(all_next_states)

    bellman_fn = jax.jit(bellman_iteration, static_argnums=(5, 6, 7))

    # 2. Rewards/dones for WM side (always). True side: only if oracle missed.
    t_rd = time.time()
    if is_multi:
        goals_list = unseen_goal_spec["goals"]
        mask = jnp.array(unseen_goal_spec["mask"])
        if state_to_obs_fn is not None:
            raise NotImplementedError(
                "Multi-goal unseen specs with grid_state_dim != obs_state_dim "
                "are not yet supported (would need to lift inside "
                "_precompute_rewards_dones_multi)."
            )
        all_rewards_wm, all_dones_wm = _precompute_rewards_dones_multi(
            ns_wm, goals_list, mask, reward_type, sigma, a_threshold,
            state_dim, goal_env_terminated_fn,
        )
        if not oracle_loaded:
            all_rewards_true, all_dones_true = _precompute_rewards_dones_multi(
                ns_true, goals_list, mask, reward_type, sigma, a_threshold,
                state_dim, goal_env_terminated_fn,
            )
    else:
        goal = jnp.array(unseen_goal_spec["goal"])
        mask = jnp.array(unseen_goal_spec["mask"])
        precompute_fn = jax.jit(precompute_rewards_dones,
                                static_argnums=(3, 6, 7, 8, 9))
        all_rewards_wm, all_dones_wm = precompute_fn(
            ns_wm, goal, mask, reward_type, sigma, a_threshold,
            state_dim, terminate_on_goal, goal_env_terminated_fn,
            state_to_obs_fn,
        )
        if not oracle_loaded:
            all_rewards_true, all_dones_true = precompute_fn(
                ns_true, goal, mask, reward_type, sigma, a_threshold,
                state_dim, terminate_on_goal, goal_env_terminated_fn,
                state_to_obs_fn,
            )

    if forbidden:
        all_rewards_wm = all_rewards_wm * (1.0 - _forbidden_mask(ns_wm))
        if not oracle_loaded:
            all_rewards_true = all_rewards_true * (1.0 - _forbidden_mask(ns_true))
    all_rewards_wm = jax.block_until_ready(all_rewards_wm)
    if not oracle_loaded:
        all_rewards_true = jax.block_until_ready(all_rewards_true)

    # 3. Bellman optimality on WM (always). On true: only if oracle missed.
    t0 = time.time()
    if oracle_loaded:
        all_rewards_true = jnp.asarray(np.load(
            os.path.join(oracle_goal_dir, "all_rewards_true.npy")))
        all_dones_true = jnp.asarray(np.load(
            os.path.join(oracle_goal_dir, "all_dones_true.npy")))
        V_true = jnp.asarray(np.load(
            os.path.join(oracle_goal_dir, "V_true.npy")))
        Q_true = jnp.asarray(np.load(
            os.path.join(oracle_goal_dir, "Q_true.npy")))
        pi_true = jnp.asarray(np.load(
            os.path.join(oracle_goal_dir, "pi_true.npy")))
        print(f"  Oracle cache hit for {label!r} (true-side loaded from "
              f"{oracle_goal_dir})")
    else:
        V_true, Q_true, _ = bellman_fn(
            ns_true, all_rewards_true, all_dones_true, axis_grids,
            gamma, vi_max_iter, action_dim, convergence_threshold, None,
        )
        V_true = jax.block_until_ready(V_true)
        pi_true = jnp.argmax(Q_true, axis=0).astype(jnp.int32)
    V_wm, Q_wm, _ = bellman_fn(
        ns_wm, all_rewards_wm, all_dones_wm, axis_grids,
        gamma, vi_max_iter, action_dim, convergence_threshold, None,
    )
    V_wm = jax.block_until_ready(V_wm)
    pi_wm = jnp.argmax(Q_wm, axis=0).astype(jnp.int32)
    # 5. Rollout returns on true dynamics. Caller-provided start states (obs
    # space); projected to grid space if the VI grid is smaller (e.g. Reacher
    # 6D obs → 4D effective).
    if eval_starts is None:
        raise ValueError(
            "evaluate_unseen_goals: eval_starts must be provided. "
            "Pass env.reset()-sampled obs from the caller "
            "(see run.py UNSEEN_NUM_EVAL_STARTS)."
        )
    starts_arr = jnp.asarray(eval_starts)
    if obs_to_grid_fn is not None and starts_arr.shape[-1] != state_dim:
        starts_arr = obs_to_grid_fn(starts_arr)
    eval_starts_grid = starts_arr
    n_eval_starts = eval_starts_grid.shape[0]
    positions = eval_starts_grid[:, 0]

    if is_multi:
        goals_list_spec = unseen_goal_spec["goals"]
        mask_spec = unseen_goal_spec["mask"]
        rollout_fn = lambda Q, starts: _rollout_policy_on_true_dynamics_multi(
            Q, axis_grids, dynamics_fn, starts,
            goals_list_spec, mask_spec, reward_type, sigma, a_threshold,
            goal_env_terminated_fn, gamma, max_steps,
        )
    else:
        goal_arr = jnp.array(unseen_goal_spec["goal"])
        mask_arr = jnp.array(unseen_goal_spec["mask"])
        rollout_fn = lambda Q, starts: _rollout_policy_on_true_dynamics(
            Q, axis_grids, dynamics_fn, starts,
            goal_arr, mask_arr, reward_type, sigma, a_threshold,
            terminate_on_goal, goal_env_terminated_fn, gamma, max_steps,
            state_to_obs_fn=state_to_obs_fn,
        )

    if oracle_loaded:
        ret_true_undisc = jnp.asarray(np.load(
            os.path.join(oracle_goal_dir, "ret_true_undisc.npy")))
        ret_true_disc = jnp.asarray(np.load(
            os.path.join(oracle_goal_dir, "ret_true_disc.npy")))
        t_roll_true = 0.0
    else:
        t_roll = time.time()
        ret_true_undisc, ret_true_disc = rollout_fn(Q_true, eval_starts_grid)
        ret_true_undisc = jax.block_until_ready(ret_true_undisc)
        t_roll_true = time.time() - t_roll

    t_roll = time.time()
    ret_wm_undisc, ret_wm_disc = rollout_fn(Q_wm, eval_starts_grid)
    ret_wm_undisc = jax.block_until_ready(ret_wm_undisc)
    t_roll_wm = time.time() - t_roll

    # Save oracle cache on the first cell that misses for this goal.
    if oracle_goal_dir and not oracle_loaded:
        os.makedirs(oracle_goal_dir, exist_ok=True)
        for fname, arr in [
            ("all_rewards_true.npy", all_rewards_true),
            ("all_dones_true.npy",   all_dones_true),
            ("V_true.npy",           V_true),
            ("Q_true.npy",           Q_true),
            ("pi_true.npy",          pi_true),
            ("ret_true_undisc.npy",  ret_true_undisc),
            ("ret_true_disc.npy",    ret_true_disc),
        ]:
            np.save(os.path.join(oracle_goal_dir, fname), np.asarray(arr))
        print(f"  Saved oracle cache to {oracle_goal_dir}/")

    # Headline R*/R_WM (discounted, paper-reported); std-err over starts.
    n_eval = int(ret_true_disc.shape[0])
    se_true = float(ret_true_disc.std() / (n_eval ** 0.5))
    se_wm = float(ret_wm_disc.std() / (n_eval ** 0.5))
    print(
        f"  R*  = {float(ret_true_disc.mean()):.3f} ± {se_true:.3f}    "
        f"R_WM = {float(ret_wm_disc.mean()):.3f} ± {se_wm:.3f}    (n={n_eval})"
    )

    # 6. V^π_wm on TRUE dynamics (Bellman expectation for the WM-derived policy
    # under the real kernel — what that policy actually achieves). Only needed
    # for the 2D value-comparison plot; skip otherwise (~0.5–2.5 s/goal).
    if state_dim == 2:
        t0 = time.time()
        V_pi_wm, _, _ = bellman_fn(
            ns_true, all_rewards_true, all_dones_true, axis_grids,
            gamma, vi_max_iter, action_dim, convergence_threshold, pi_wm,
        )
        V_pi_wm = jax.block_until_ready(V_pi_wm)
    else:
        V_pi_wm = None

    # 7. Save results (skipped during arch sweep: aggregator only consumes
    # the cell-level unseen_summary.npz).
    if not skip_per_goal_artifacts:
        with open(f"{goal_dir}/results.txt", "w") as f:
            f.write(f"Unseen goal: {label}\n\n")
            f.write(f"Eval starts: {n_eval_starts} (sampled from env.reset())\n\n")
            f.write("Undiscounted returns:\n")
            f.write(f"  True policy: {float(ret_true_undisc.mean()):.4f} ± {float(ret_true_undisc.std()):.4f}\n")
            f.write(f"  WM policy:   {float(ret_wm_undisc.mean()):.4f} ± {float(ret_wm_undisc.std()):.4f}\n\n")
            f.write(f"Discounted returns (gamma={gamma}):\n")
            f.write(f"  True policy: {float(ret_true_disc.mean()):.4f} ± {float(ret_true_disc.std()):.4f}\n")
            f.write(f"  WM policy:   {float(ret_wm_disc.mean()):.4f} ± {float(ret_wm_disc.std()):.4f}\n")

    # 8. Plots (skipped during arch sweep — only relevant for single deep-dive
    # cells from run.py).
    if not skip_per_goal_artifacts:
        if state_dim == 2:
            X0, X1 = jnp.meshgrid(axis_grids[0], axis_grids[1], indexing="ij")
            # Value comparison: V* (true optimal) vs V^π_wm (WM policy on true dynamics).
            plot_two_value_comparison_final(
                X0, X1,
                panels=[
                    (r"$V^*$", V_true),
                    (r"$V^{WM}$", V_pi_wm),
                ],
                dim1_label=state_labels[0],
                dim2_label=state_labels[1],
                goal_position=(
                    float(jnp.array(unseen_goal_spec["goal"])[0])
                    if not is_multi and float(mask[0]) > 0.0 else None
                ),
                save_path=f"{goal_dir}/value_comparison.png",
            )
        elif state_dim == 4:
            # Two 3D plots (1x2 each): V_true vs V_wm, one per slice.
            plot_two_value_comparison_2slice_3d(
                V_true_4d=np.asarray(V_true),
                V_wm_4d=np.asarray(V_wm),
                axis_grids=[np.asarray(g) for g in axis_grids],
                state_labels=state_labels[:state_dim],
                slice_dims_top=(0, 1),
                slice_dims_bottom=(2, 3),
                save_path_top=f"{goal_dir}/value_comparison_angle.png",
                save_path_bottom=f"{goal_dir}/value_comparison_omega.png",
            )

    print(f"  → goal '{label}' total: {time.time() - t_goal_start:.2f}s")

    return {
        "label": label,
        "return_true_mean": float(ret_true_undisc.mean()),
        "return_true_std": float(ret_true_undisc.std()),
        "return_wm_mean": float(ret_wm_undisc.mean()),
        "return_wm_std": float(ret_wm_undisc.std()),
        "disc_return_true_mean": float(ret_true_disc.mean()),
        "disc_return_true_std": float(ret_true_disc.std()),
        "disc_return_wm_mean": float(ret_wm_disc.mean()),
        "disc_return_wm_std": float(ret_wm_disc.std()),
    }


# ── Entry point ─────────────────────────────────────────────────────────────


def evaluate_unseen_goals(
    p_params, wm_config, pqn_config, env_config,
    dynamics_fn, env_terminated_fn,
    unseen_goals, out_dir,
    wm_dynamics_fn=None,
    grid_state_dim=None,
    grid_state_ranges=None,
    state_to_obs_fn=None,
    obs_to_grid_fn=None,
    eval_starts=None,
    oracle_cache_dir=None,
    skip_per_goal_artifacts=True,
):
    """Evaluate all unseen goals. Returns list of per-goal metrics dicts.

    wm_dynamics_fn: caller-provided WM dynamics function; if None, the default
    obs-space wrapper from eval/world_model.py is used (MountainCar). For
    Reacher / other effective-state envs, pass an explicit 4D-effective wrapper
    (e.g. envs.reacher_utils.make_reacher_wm_dynamics_fn).

    See evaluate_single_unseen_goal for the meaning of the grid_* / *_to_*_fn /
    eval_starts kwargs. Defaults preserve the original obs-space MountainCar
    behavior.
    """
    if not unseen_goals:
        print("No unseen goals configured — skipping.")
        return []

    obs_state_dim = env_config["STATE_DIM"]
    action_dim = env_config["ACTION_DIM"]
    os.makedirs(out_dir, exist_ok=True)

    if wm_dynamics_fn is None:
        wm_dynamics_fn = make_wm_dynamics_fn(
            p_params, wm_config, action_dim, obs_state_dim,
        )

    all_metrics = []
    for spec in unseen_goals:
        m = evaluate_single_unseen_goal(
            spec, dynamics_fn, wm_dynamics_fn, env_config, pqn_config,
            env_terminated_fn, out_dir,
            grid_state_dim=grid_state_dim,
            grid_state_ranges=grid_state_ranges,
            state_to_obs_fn=state_to_obs_fn,
            obs_to_grid_fn=obs_to_grid_fn,
            eval_starts=eval_starts,
            oracle_cache_dir=oracle_cache_dir,
            skip_per_goal_artifacts=skip_per_goal_artifacts,
        )
        all_metrics.append(m)

    # Summary table: discounted returns R* / R_WM (paper-reported headline).
    print(f"\n── Unseen goals summary (discounted returns) ──")
    for m in all_metrics:
        se_t = m['disc_return_true_std']
        se_w = m['disc_return_wm_std']
        print(
            f"  {m['label']:<32}  R* = {m['disc_return_true_mean']:.3f} ± {se_t:.3f}    "
            f"R_WM = {m['disc_return_wm_mean']:.3f} ± {se_w:.3f}"
        )

    if not skip_per_goal_artifacts:
        with open(f"{out_dir}/summary.txt", "w") as f:
            f.write("Unseen Goals — discounted returns (R*, R_WM), mean ± std over eval starts.\n\n")
            f.write(f"{'Label':<40} {'R*':>20} {'R_WM':>20}\n")
            for m in all_metrics:
                f.write(
                    f"{m['label']:<40} "
                    f"{m['disc_return_true_mean']:>10.4f}±{m['disc_return_true_std']:.4f} "
                    f"{m['disc_return_wm_mean']:>10.4f}±{m['disc_return_wm_std']:.4f}\n"
                )

    return all_metrics
