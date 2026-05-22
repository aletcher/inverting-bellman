"""Policy evaluation for PQN's own greedy policy + PQN comparison.

The headline metric is pqn_vs_true: MSE/MAPE between V_pqn / Q_pqn and
V^π / Q^π (the *true* value function under PQN's greedy policy, via policy
evaluation on a grid). Captures PQN's Bellman self-consistency error —
paper §4.1's claim that PQN's Q-values are "imperfect" yet the extracted WM is
highly accurate.

Works for arbitrary state dimensionality via N-linear interpolation + N-D
Bellman backup. Used at state_dim=2 (MountainCar) and state_dim=4 (Reacher
effective state).
"""

import os
import time
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

from envs.goals import compute_reward, goal_achieved
from training.pqn_utils import make_q_network, make_goal_repr
from plotting.vi import plot_two_value_comparison_final

# ── N-linear interpolation ───────────────────────────────────────────────────


def interpolate_V(V_grid, next_states, axis_grids):
    """N-linear interpolation of V_grid at next_states.

    Args:
        V_grid: array of shape (R, R, ..., R) (state_dim axes).
        next_states: array of shape (..., state_dim).
        axis_grids: tuple of state_dim 1D arrays, each shape (R,).

    Returns array shape next_states.shape[:-1]. The 2^state_dim corner
    enumeration is unrolled at trace time since state_dim is static.
    """
    state_dim = len(axis_grids)
    fracs = []  # list of (i0, frac) per dim
    for d in range(state_dim):
        grid = axis_grids[d]
        n = grid.shape[0]
        f = (next_states[..., d] - grid[0]) / (grid[-1] - grid[0]) * (n - 1)
        f = jnp.clip(f, 0.0, n - 1 - 1e-6)
        i0 = jnp.floor(f).astype(jnp.int32)
        fracs.append((i0, f - i0))

    result = jnp.zeros_like(fracs[0][1])
    for offsets in product([0, 1], repeat=state_dim):  # 2^N corners, static
        idx = tuple(fracs[d][0] + offsets[d] for d in range(state_dim))
        w = jnp.ones_like(fracs[0][1])
        for d in range(state_dim):
            w = w * (fracs[d][1] if offsets[d] == 1 else (1 - fracs[d][1]))
        result = result + V_grid[idx] * w
    return result


# ── Value Iteration ──────────────────────────────────────────────────────────


def precompute_dynamics(dynamics_fn, vi_grid_res, state_ranges, action_dim):
    """Precompute next states for all state-action pairs on the N-D grid.

    Returns (axis_grids, all_next_states) where:
        axis_grids: tuple of state_dim 1D arrays of length vi_grid_res.
        all_next_states: shape (action_dim,) + (vi_grid_res,) * state_dim + (state_dim,).
    """
    state_dim = len(state_ranges)
    axis_grids = tuple(
        jnp.linspace(lo, hi, vi_grid_res) for (lo, hi) in state_ranges
    )
    meshes = jnp.meshgrid(*axis_grids, indexing="ij")
    states_flat = jnp.stack(meshes, axis=-1).reshape(-1, state_dim)
    n_cells = vi_grid_res ** state_dim
    actions = jnp.arange(action_dim)
    grid_shape = (vi_grid_res,) * state_dim

    def _step(a):
        return dynamics_fn(states_flat, jnp.full(n_cells, a)).reshape(
            grid_shape + (state_dim,)
        )

    all_next_states = jax.vmap(_step)(actions)
    return axis_grids, all_next_states


def precompute_rewards_dones(
    all_next_states,
    goal,
    mask,
    reward_type,
    sigma,
    a_threshold,
    rew_scale,
    state_dim,
    terminate_on_goal,
    env_terminated_fn,
    state_to_obs_fn=None,
):
    """Per-action reward and done arrays for a given goal.

    Returns (all_rewards, all_dones), both shape (action_dim,) + grid_shape.
    Goal-dependent — call once per goal and reuse for VI and policy evaluation.
    rew_scale matches the REW_SCALE applied to gc_compute_reward during PQN
    training so VI/PQN values live on the same scale.

    state_to_obs_fn lifts grid-space states (state_dim) to the observation
    space that goal/mask/env_terminated_fn expect. When None the grid *is* the
    observation (2D envs).
    """
    grid_shape = all_next_states.shape[1:-1]  # strip leading action dim and trailing state dim

    def _lift(ns):
        flat = ns.reshape(-1, state_dim)
        return state_to_obs_fn(flat) if state_to_obs_fn is not None else flat

    all_rewards = rew_scale * jax.vmap(
        lambda ns: compute_reward(
            _lift(ns), goal, mask, reward_type, sigma, a_threshold
        ).reshape(grid_shape)
    )(all_next_states)

    # done mask: combine goal termination and env termination
    all_dones = jnp.zeros_like(all_rewards)
    if terminate_on_goal:
        all_dones_goal = jax.vmap(
            lambda ns: goal_achieved(_lift(ns), goal, mask, a_threshold)
            .astype(float)
            .reshape(grid_shape)
        )(all_next_states)
        all_dones = jnp.maximum(all_dones, all_dones_goal)
    if env_terminated_fn is not None:
        all_dones_env = jax.vmap(
            lambda ns: env_terminated_fn(_lift(ns))
            .astype(float)
            .reshape(grid_shape)
        )(all_next_states)
        all_dones = jnp.maximum(all_dones, all_dones_env)

    return all_rewards, all_dones


def bellman_iteration(
    all_next_states,
    all_rewards,
    all_dones,
    axis_grids,
    gamma,
    max_iter,
    action_dim,
    convergence_threshold=0.0,
    policy=None,
):
    """Bellman iteration on precomputed dynamics & per-goal rewards/dones.

    If policy is None: Bellman optimality (V*, Q*).
    If policy is given (shape grid_shape int32 in ij order matching
    all_next_states): Bellman expectation for that fixed policy (V^pi, Q^pi).

    Returns V, Q (per action), and convergence deltas. Works for any state_dim
    via N-linear interpolation.
    """
    actions = jnp.arange(action_dim)
    grid_shape = all_next_states.shape[1:-1]  # strip action dim and trailing state dim

    if policy is None:
        select_v = lambda Q: jnp.max(Q, axis=0)
    else:
        select_v = lambda Q: jnp.take_along_axis(
            Q, policy[None, ...], axis=0
        ).squeeze(0)

    def q_for_action(V, a_idx):
        ns = all_next_states[a_idx]
        r = all_rewards[a_idx]
        done = all_dones[a_idx]
        v_next = interpolate_V(V, ns, axis_grids)
        return r + gamma * (1.0 - done) * v_next

    V0 = jnp.zeros(grid_shape)

    def bellman_backup(carry, _):
        V, converged = carry

        def do_backup(V):
            Q = jax.vmap(lambda a: q_for_action(V, a))(actions)
            V_new = select_v(Q)
            delta = jnp.max(jnp.abs(V_new - V))
            return V_new, delta

        def skip(V):
            return V, 0.0

        V_new, delta = jax.lax.cond(converged, skip, do_backup, V)
        new_converged = converged | (delta < convergence_threshold)
        return (V_new, new_converged), delta

    (V_final, _), deltas = jax.lax.scan(
        bellman_backup, (V0, jnp.bool_(False)), None, length=max_iter
    )

    Q_final = jax.vmap(lambda a: q_for_action(V_final, a))(actions)

    return V_final, Q_final, deltas


# ── PQN evaluation ───────────────────────────────────────────────────────────


def evaluate_pqn_on_grid(
    q_params,
    q_batch_stats,
    goal,
    mask,
    axis_grids,
    pqn_config,
    action_dim,
    state_dim,
    state_to_obs_fn=None,
    obs_state_dim=None,
    chunk_size=1_048_576,
):
    """Evaluate PQN Q-values on the full N-D VI grid (ij order).

    Returns (V, Q_per_action) where:
        V shape: grid_shape (= (R,) * state_dim)
        Q_per_action shape: (action_dim,) + grid_shape

    If state_to_obs_fn is supplied, grid points are lifted to obs space before
    feeding the Q-network. obs_state_dim is the lift output dim; defaults to
    state_dim (identity lift).

    chunk_size: when n_cells > chunk_size, the Q-network forward is split via
    lax.scan. Necessary for large nets (hidden≥1024) at high grids; per-layer
    activation peaks at chunk_size · hidden · 4 bytes (f32) instead of
    n_cells · hidden · 4. Pass None to disable.
    """
    network = make_q_network(pqn_config)

    meshes = jnp.meshgrid(*axis_grids, indexing="ij")
    grid_shape = meshes[0].shape
    grid_pts = jnp.stack(meshes, axis=-1).reshape(-1, state_dim)
    obs = state_to_obs_fn(grid_pts) if state_to_obs_fn is not None else grid_pts
    n = obs.shape[0]
    goal_dim = obs_state_dim if obs_state_dim is not None else state_dim
    goal_repr = make_goal_repr(goal, mask, n, goal_dim)

    def _forward(obs_chunk, goal_chunk):
        return network.apply(
            {"params": q_params, "batch_stats": q_batch_stats},
            obs_chunk,
            goal_chunk,
            train=False,
        )

    if chunk_size is None or n <= chunk_size:
        q = _forward(obs, goal_repr)
    else:
        # goal_repr is a ContinuousGoal pytree (target_state, reward_mask) — split
        # each field along axis 0, then reassemble per chunk inside scan.
        from envs.goals import ContinuousGoal
        n_chunks = (n + chunk_size - 1) // chunk_size
        pad = n_chunks * chunk_size - n
        obs_p = jnp.pad(obs, ((0, pad), (0, 0)))
        target_p = jnp.pad(goal_repr.target_state, ((0, pad), (0, 0)))
        mask_p = jnp.pad(goal_repr.reward_mask, ((0, pad), (0, 0)))
        obs_chunks = obs_p.reshape(n_chunks, chunk_size, obs.shape[1])
        target_chunks = target_p.reshape(n_chunks, chunk_size,
                                         goal_repr.target_state.shape[1])
        mask_chunks = mask_p.reshape(n_chunks, chunk_size,
                                     goal_repr.reward_mask.shape[1])

        def step(_carry, args):
            obs_c, target_c, mask_c = args
            return None, _forward(
                obs_c, ContinuousGoal(target_state=target_c, reward_mask=mask_c)
            )

        _, q_chunks = jax.lax.scan(
            step, None, (obs_chunks, target_chunks, mask_chunks)
        )
        q = q_chunks.reshape(n_chunks * chunk_size, -1)[:n]

    v = q.max(axis=-1).reshape(grid_shape)
    Q_pqn = jnp.stack([q[:, a].reshape(grid_shape) for a in range(action_dim)])

    return v, Q_pqn


def evaluate_pqn_at_states(
    q_params,
    q_batch_stats,
    goal,
    mask,
    states,
    pqn_config,
    action_dim,
    state_to_obs_fn=None,
):
    """Evaluate PQN at a flat batch of grid-space states shape (n, state_dim).

    Returns (V_at_s, Q_at_s) where V_at_s shape (n,) and Q_at_s shape
    (action_dim, n). Lift to obs via state_to_obs_fn for envs whose Q-net takes
    a larger obs (e.g. Reacher: 4D → 6D).
    """
    network = make_q_network(pqn_config)
    obs = state_to_obs_fn(states) if state_to_obs_fn is not None else states
    n = obs.shape[0]
    obs_dim = obs.shape[1]
    goal_repr = make_goal_repr(goal, mask, n, obs_dim)
    q = network.apply(
        {"params": q_params, "batch_stats": q_batch_stats},
        obs, goal_repr, train=False,
    )
    v = q.max(axis=-1)
    Q_at_s = jnp.stack([q[:, a] for a in range(action_dim)])
    return v, Q_at_s


# ── Metrics helpers ──────────────────────────────────────────────────────────


def _metrics(a, b):
    """MSE / MAPE between reference a and prediction b (jax arrays).

    Paper reports V/Q MSE and MAPE (e.g. Reacher §4.1: MSE 4.2e-2, MAPE 0.19).
    """
    diff = a - b
    abs_a = jnp.abs(a)
    return {
        "mse": float(jnp.mean(diff ** 2)),
        "mape": float(jnp.mean(jnp.where(abs_a > 1e-6, jnp.abs(diff) / abs_a, 0.0))),
    }


def _v_q_metrics(V_a, V_b, Q_a, Q_b, action_names):
    """Reference a, prediction b. Returns (V_metrics, {action: Q_metrics})."""
    return (
        _metrics(V_a, V_b),
        {n: _metrics(Q_a[i], Q_b[i]) for i, n in enumerate(action_names)},
    )


def _flatten(label, vm, qm_per_action):
    out = {
        f"{label}_V_mse": vm["mse"],
        f"{label}_V_mape": vm["mape"],
    }
    for aname, m in qm_per_action.items():
        out[f"{label}_Q_{aname}_mse"] = m["mse"]
        out[f"{label}_Q_{aname}_mape"] = m["mape"]
    return out


# ── Per-goal pipeline ────────────────────────────────────────────────────────


def run_for_goal(
    goal_idx,
    goal,
    mask,
    V_pi,
    Q_pi,
    deltas,
    axis_grids,
    V_pqn,
    Q_pqn,
    out_dir,
    action_names,
    state_labels,
    goal_state_labels=None,
    train_support_states=None,
    q_params=None,
    q_batch_stats=None,
    pqn_config=None,
    state_to_obs_fn=None,
):
    """Per-goal artifacts + headline pqn_vs_true metric.

    Compares PQN's Q-values against V^π / Q^π — the *true* value function for
    PQN's own greedy policy, via policy evaluation on the VI grid.
    Captures PQN's Bellman self-consistency error.

    All arrays are in ij order with shape (R,) * state_dim (or
    (action_dim,) + (R,) * state_dim for Q). When goals live in a larger obs
    space (e.g. Reacher 4D grid vs 6D obs goals), pass goal_state_labels of
    length len(goal) for the goal string.
    """
    state_dim = V_pi.ndim
    if goal_state_labels is None:
        goal_state_labels = state_labels
    # Only masked-in dims are meaningful (unmasked dims hold arbitrary zeros).
    # Filter both the printed string and the folder name accordingly.
    goal_masked_pairs = [
        (lbl, float(g))
        for lbl, g, m in zip(goal_state_labels, goal, mask)
        if float(m) > 0
    ]
    if not goal_masked_pairs:
        goal_str_masked = "(no masked dims)"
        goal_label = f"goal{goal_idx}"
    else:
        goal_str_masked = ", ".join(f"{lbl}={g:.2f}" for lbl, g in goal_masked_pairs)
        goal_label = f"goal{goal_idx}_" + "_".join(
            f"{lbl}{g:.2f}" for lbl, g in goal_masked_pairs
        )
    goal_dir = f"{out_dir}/{goal_label}"
    os.makedirs(goal_dir, exist_ok=True)

    print(f"\n{'─' * 50}")
    print(f"Goal {goal_idx}: ({goal_str_masked})  mask={mask}")
    print(f"{'─' * 50}")

    nonzero_deltas = deltas[deltas > 0]
    if len(nonzero_deltas) > 0:
        print(
            f"  Converged at iteration {len(nonzero_deltas)} (final delta: {float(nonzero_deltas[-1]):.2e})"
        )
    else:
        print(f"  Final delta: {float(deltas[-1]):.2e}")

    # Save VI data for downstream tooling. Arrays in ij order, axis_grid_<i>
    # per dim. V_pqn is PQN's own value (Fig 1 input); V_pi is the true value
    # under PQN's greedy policy.
    vi_payload = {
        "V_pqn": np.array(V_pqn),
        "Q_pqn": np.array(Q_pqn),
        "V_pi":  np.array(V_pi),
        "Q_pi":  np.array(Q_pi),
    }
    for d in range(state_dim):
        vi_payload[f"axis_grid_{d}"] = np.array(axis_grids[d])
    np.savez(f"{goal_dir}/vi_data.npz", **vi_payload)

    # pqn_vs_true: V_pqn / Q_pqn vs V^π / Q^π (true values under PQN's own
    # greedy policy). Eval support is the VI grid by default;
    # Q_EVAL_ON_TRAIN_SUPPORT=True switches to the WM's training-support
    # distribution so the headline Q_MSE shares support with WM_MSE.
    if train_support_states is None:
        m_pqn_vs_true = _v_q_metrics(V_pi, V_pqn, Q_pi, Q_pqn, action_names)
        support_tag = "vi_grid"
    else:
        V_pi_s = interpolate_V(V_pi, train_support_states, axis_grids)
        Q_pi_s = jax.vmap(
            lambda Qa: interpolate_V(Qa, train_support_states, axis_grids)
        )(Q_pi)
        V_pqn_s, Q_pqn_s = evaluate_pqn_at_states(
            q_params, q_batch_stats, goal, mask, train_support_states,
            pqn_config, Q_pi.shape[0],
            state_to_obs_fn=state_to_obs_fn,
        )
        m_pqn_vs_true = _v_q_metrics(V_pi_s, V_pqn_s, Q_pi_s, Q_pqn_s, action_names)
        support_tag = f"train_support (n={len(train_support_states)})"
    print(
        f"  V_pqn-V^π [{support_tag}]:  V_mse={m_pqn_vs_true[0]['mse']:.2e} "
        f"V_mape={m_pqn_vs_true[0]['mape']:.3f}"
    )

    with open(f"{goal_dir}/metrics.txt", "w") as f:
        f.write(f"goal_idx: {goal_idx}\n")
        f.write(f"goal: ({goal_str_masked})  mask={list(map(int, mask))}\n")
        f.write(f"eval_support: {support_tag}\n")
        f.write("metric: pqn_vs_true (V_pqn / Q_pqn vs V^π / Q^π under PQN's policy)\n")
        vm, qm_per_action = m_pqn_vs_true
        f.write(f"V_mse: {vm['mse']:.6e}\n")
        f.write(f"V_mape: {vm['mape']:.4f}\n")
        for aname, m in qm_per_action.items():
            f.write(f"Q_{aname}_mse: {m['mse']:.6e}\n")
            f.write(f"Q_{aname}_mape: {m['mape']:.4f}\n")

    # Plots. 2D (MountainCar): single (0,1) slice, full grid.
    # 4D (Reacher): two slices — (w1, w2) = 0 in angle space, (t1, t2) = 0 in velocity space.
    def _idx_closest(grid, val):
        return int(jnp.argmin(jnp.abs(jnp.asarray(grid) - val)))

    # V^π (policy eval under PQN's policy) vs V_pqn comparison plots.
    if state_dim == 2:
        d0, d1 = 0, 1
        X0, X1 = jnp.meshgrid(axis_grids[d0], axis_grids[d1], indexing="ij")
        plot_two_value_comparison_final(
            X0, X1,
            panels=[(r"$V^\pi$", V_pi), (r"$V_{pqn}$", V_pqn)],
            dim1_label=state_labels[d0],
            dim2_label=state_labels[d1],
            save_path=f"{goal_dir}/V_comparison_{state_labels[d0]}_{state_labels[d1]}.png",
        )
    elif state_dim == 4:
        # (w1, w2) = 0 slice plotted in angle space + (t1, t2) = 0 slice in velocity space.
        i_w1_zero = _idx_closest(axis_grids[2], 0.0)
        i_w2_zero = _idx_closest(axis_grids[3], 0.0)
        i_t1_zero = _idx_closest(axis_grids[0], 0.0)
        i_t2_zero = _idx_closest(axis_grids[1], 0.0)
        T1, T2 = jnp.meshgrid(axis_grids[0], axis_grids[1], indexing="ij")
        W1, W2 = jnp.meshgrid(axis_grids[2], axis_grids[3], indexing="ij")
        plot_two_value_comparison_final(
            T1, T2,
            panels=[
                (r"$V^\pi$",   V_pi[:, :, i_w1_zero, i_w2_zero]),
                (r"$V_{pqn}$", V_pqn[:, :, i_w1_zero, i_w2_zero]),
            ],
            dim1_label=state_labels[0],
            dim2_label=state_labels[1],
            save_path=f"{goal_dir}/V_comparison_angle.png",
        )
        plot_two_value_comparison_final(
            W1, W2,
            panels=[
                (r"$V^\pi$",   V_pi[i_t1_zero, i_t2_zero, :, :]),
                (r"$V_{pqn}$", V_pqn[i_t1_zero, i_t2_zero, :, :]),
            ],
            dim1_label=state_labels[2],
            dim2_label=state_labels[3],
            save_path=f"{goal_dir}/V_comparison_w.png",
        )

    return _flatten("pqn_vs_true", *m_pqn_vs_true)


# ── Run all goals ────────────────────────────────────────────────────────────


def run_all_goals(
    q_params,
    q_batch_stats,
    out_dir,
    pqn_config,
    env_config,
    dynamics_fn,
    env_terminated_fn,
    state_to_obs_fn=None,
    grid_state_dim=None,
    grid_state_ranges=None,
    grid_state_labels=None,
    train_support_sampler=None,
    n_train_support_samples=4096,
):
    """Run VI + comparison vs PQN for each goal. Writes per-goal metrics +
    vi_summary.txt with V_mse / V_mape / Q_mse / Q_mape averaged across goals
    — the V/Q numbers the paper inline-quotes (e.g. Reacher Q_MSE = 4.2e-2,
    Q_MAPE = 0.19).

    By default VI runs on observation-space grids (STATE_DIM / STATE_RANGES).
    To run VI on a smaller effective state space (e.g. Reacher 4D rather than
    8D), supply state_to_obs_fn + the grid_state_* overrides.

    Goals and masks stay in observation space (that's how the config stores them).
    """
    goals = pqn_config["GOALS"]
    goal_masks = pqn_config["REWARD_MASK"]
    gamma = pqn_config["GAMMA"]
    reward_type = pqn_config["REWARD_TYPE"]
    sigma = pqn_config["REWARD_SIGMA"]
    a_threshold = pqn_config["REWARD_A"]
    rew_scale = pqn_config.get("REW_SCALE", 1.0)
    obs_state_dim = pqn_config["STATE_DIM"]
    state_dim = grid_state_dim if grid_state_dim is not None else obs_state_dim
    action_dim = pqn_config["ACTION_DIM"]
    terminate_on_goal = pqn_config["TERMINATE_ON_GOAL"]
    action_names = env_config["ACTION_NAMES"]
    vi_grid_res = env_config["VI_GRID_RES"]
    vi_max_iter = env_config["VI_MAX_ITER"]
    # VI grid range can be restricted to the env's valid (non-terminal) region
    # via VI_STATE_RANGES, which is what you want when the global STATE_RANGES
    # extends beyond termination thresholds (else those cells are pure waste).
    if grid_state_ranges is not None:
        state_ranges = grid_state_ranges
    else:
        state_ranges = env_config.get("VI_STATE_RANGES", env_config["STATE_RANGES"])
    goal_state_labels = env_config.get(
        "STATE_LABELS", [f"x{i}" for i in range(obs_state_dim)]
    )
    if grid_state_labels is not None:
        state_labels = grid_state_labels
    else:
        state_labels = goal_state_labels
    convergence_threshold = env_config.get("VI_CONVERGENCE_THRESHOLD", 0.0)

    # Optional train-support sample (same support WM is evaluated on, projected
    # into grid space). Reused across goals; sampled once with fixed RNG.
    train_support_states = None
    if train_support_sampler is not None:
        train_support_states = train_support_sampler(
            jax.random.PRNGKey(0xC0FFEE), n_train_support_samples,
        )
        train_support_states = jax.block_until_ready(train_support_states)
        print(f"  Train-support sample built: {train_support_states.shape}")

    # Precompute dynamics once (goal-independent). static_argnums: dynamics_fn
    # (0), vi_grid_res (1), state_ranges (2 — tuple of tuples), action_dim (3).
    t0 = time.time()
    dyn_fn = jax.jit(precompute_dynamics, static_argnums=(0, 1, 2, 3))
    axis_grids, all_next_states = dyn_fn(
        dynamics_fn,
        vi_grid_res,
        tuple(tuple(r) for r in state_ranges),
        action_dim,
    )
    all_next_states = jax.block_until_ready(all_next_states)
    print(f"  Dynamics precomputed in {time.time() - t0:.1f}s")

    # JIT-compile precompute and bellman once, reuse across goals. bellman is
    # called twice per goal (policy=None for VI, policy=array for PE) — JAX
    # caches a separate compile per pytree shape. static_argnums for
    # precompute_rewards_dones: reward_type(3), state_dim(7), terminate_on_goal(8),
    # env_terminated_fn(9), state_to_obs_fn(10).
    precompute_fn = jax.jit(
        precompute_rewards_dones, static_argnums=(3, 7, 8, 9, 10)
    )
    bellman_fn = jax.jit(
        bellman_iteration, static_argnums=(5, 6, 7)
    )

    all_metrics = []
    for i, (goal, mask) in enumerate(zip(goals, goal_masks)):
        t0 = time.time()
        all_rewards, all_dones = precompute_fn(
            all_next_states, goal, mask, reward_type, sigma, a_threshold,
            rew_scale, state_dim, terminate_on_goal, env_terminated_fn,
            state_to_obs_fn,
        )

        # Evaluate PQN on the N-D grid (ij order — matches the VI grid).
        V_pqn, Q_pqn = evaluate_pqn_on_grid(
            q_params, q_batch_stats, goal, mask, axis_grids,
            pqn_config, action_dim, state_dim,
            state_to_obs_fn=state_to_obs_fn, obs_state_dim=obs_state_dim,
        )
        V_pqn = jax.block_until_ready(V_pqn)

        # Policy evaluation under PQN's own greedy policy: V^π / Q^π on the
        # grid. pqn_vs_true compares V_pqn/Q_pqn against these — PQN's Bellman
        # self-consistency error.
        pi_pqn = jnp.argmax(Q_pqn, axis=0).astype(jnp.int32)
        V_pi, Q_pi, deltas = bellman_fn(
            all_next_states, all_rewards, all_dones, axis_grids,
            gamma, vi_max_iter, action_dim, convergence_threshold,
            pi_pqn,
        )
        V_pi = jax.block_until_ready(V_pi)
        print(f"  Goal {i} PE completed in {time.time() - t0:.1f}s")

        metrics = run_for_goal(
            i, goal, mask, V_pi, Q_pi, deltas, axis_grids,
            V_pqn, Q_pqn, out_dir, action_names, state_labels,
            goal_state_labels=goal_state_labels,
            train_support_states=train_support_states,
            q_params=q_params, q_batch_stats=q_batch_stats,
            pqn_config=pqn_config, state_to_obs_fn=state_to_obs_fn,
        )
        all_metrics.append(metrics)

    # Per-goal table (paper metric: V* / Q* vs V_pqn / Q_pqn). Eval support is
    # whichever was picked above (VI grid or train-support samples).
    n = len(all_metrics)
    avgs = {k: sum(m[k] for m in all_metrics) / n for k in all_metrics[0]}
    avg_q_mse  = sum(avgs[f"pqn_vs_true_Q_{a}_mse"]  for a in action_names) / len(action_names)
    avg_q_mape = sum(avgs[f"pqn_vs_true_Q_{a}_mape"] for a in action_names) / len(action_names)
    avg_v_mse  = avgs["pqn_vs_true_V_mse"]
    avg_v_mape = avgs["pqn_vs_true_V_mape"]

    support_tag = (
        "vi_grid" if train_support_states is None
        else f"train_support (n={n_train_support_samples})"
    )

    with open(f"{out_dir}/vi_summary.txt", "w") as f:
        f.write("Goals (only the masked dims are part of the goal):\n")
        for i, (goal, gmask) in enumerate(zip(goals, goal_masks)):
            masked_parts = [
                f"{lbl}={float(g):.2f}"
                for lbl, g, m in zip(goal_state_labels, goal, gmask)
                if float(m) > 0
            ]
            goal_str = ", ".join(masked_parts) if masked_parts else "(no masked dims)"
            f.write(f"  {i}: ({goal_str})  mask={list(map(int, gmask))}\n")
        f.write(f"\nEval support: {support_tag}\n\n")
        f.write(f"{'goal':>6} {'V_mse':>12} {'V_mape':>10}")
        for aname in action_names:
            f.write(f" {'Q_'+aname+'_mse':>14} {'Q_'+aname+'_mape':>12}")
        f.write("\n")
        for i, m in enumerate(all_metrics):
            f.write(
                f"{i:>6} {m['pqn_vs_true_V_mse']:>12.4e} "
                f"{m['pqn_vs_true_V_mape']:>10.4f}"
            )
            for aname in action_names:
                f.write(
                    f" {m[f'pqn_vs_true_Q_{aname}_mse']:>14.4e} "
                    f"{m[f'pqn_vs_true_Q_{aname}_mape']:>12.4f}"
                )
            f.write("\n")
        f.write(
            f"\nAvg V_mse={avg_v_mse:.4e}  V_mape={avg_v_mape:.4f}\n"
            f"Avg Q_mse={avg_q_mse:.4e}  Q_mape={avg_q_mape:.4f}\n"
        )
    print(f"\nSaved {out_dir}/vi_summary.txt")
    print(
        f"Avg V_mse={avg_v_mse:.2e}  V_mape={avg_v_mape:.3f}  "
        f"Q_mse={avg_q_mse:.2e}  Q_mape={avg_q_mape:.3f}  "
        f"[support: {support_tag}]"
    )

    return all_metrics
