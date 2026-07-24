"""Agent visitation distribution over the VI grid.

Rolls out PQN's greedy policy on the TRUE dynamics from env.reset() starts and
accumulates state occupancy onto the value-iteration grid, yielding a normalised
weight `w(cell)`. This is the "distribution the original agent was trained on"
that reviewer 2 asks about: it lets us compute Q-error / recovery-error either
UNIFORMLY over state space or WEIGHTED by where the agent actually goes.

Grid convention matches eval/value_iteration.py::precompute_dynamics — an
axis-aligned mesh of `vi_grid_res` points per dim over `state_ranges`, ij order.
States live in grid space (== obs for 2D envs; 4D effective for Reacher, lifted
to obs via `state_to_obs_fn` for the Q-network and reward/termination checks).
"""

import jax
import jax.numpy as jnp

from envs.goals import goal_achieved
from training.pqn_utils import make_q_network, make_goal_repr


def grid_visitation(
    q_params, q_batch_stats, pqn_config,
    dynamics_fn, goals, goal_masks, start_states,
    state_dim, action_dim, state_ranges, vi_grid_res,
    max_steps=200, a_threshold=None, terminate_on_goal=True,
    state_to_obs_fn=None, obs_state_dim=None, env_terminated_fn=None,
):
    """Occupancy weights of PQN's greedy policy on the VI grid.

    Args:
        start_states: (n_starts, state_dim) reset states in GRID space.
        goals, goal_masks: (n_goals, obs_state_dim) — rolled out one per goal and
            averaged (the agent is goal-conditioned; visitation is the union over
            training goals, matching the theory's S_o = ∪_g Reach(π_g)).
        state_to_obs_fn: grid→obs lift for the Q-network / reward (None for 2D).

    Returns w: (vi_grid_res,)*state_dim, non-negative, sums to 1.
    """
    if a_threshold is None:
        a_threshold = pqn_config["REWARD_A"]
    if obs_state_dim is None:
        obs_state_dim = pqn_config["STATE_DIM"]
    n_starts = start_states.shape[0]
    R = vi_grid_res
    lift = state_to_obs_fn if state_to_obs_fn is not None else (lambda s: s)
    network = make_q_network(pqn_config)

    los = jnp.array([lo for lo, _ in state_ranges])
    his = jnp.array([hi for _, hi in state_ranges])

    def _rollout_states(goal, mask):
        """Return (T, n, state_dim) visited states and (T, n) validity mask."""
        goal_repr = make_goal_repr(goal, mask, n_starts, obs_state_dim)

        def _step(carry, _t):
            s, done = carry
            s_obs = lift(s)
            q = network.apply(
                {"params": q_params, "batch_stats": q_batch_stats},
                s_obs, goal_repr, train=False,
            )
            action = jnp.argmax(q, axis=-1)
            s_next = dynamics_fn(s, action)
            s_next_obs = lift(s_next)
            # record the pre-step state, valid only while not yet terminated
            rec = (s, jnp.logical_not(done))
            new_done = done
            if env_terminated_fn is not None:
                new_done = jnp.logical_or(new_done, env_terminated_fn(s_next_obs))
            if terminate_on_goal:
                achieved = goal_achieved(s_next_obs, goal, mask, a_threshold)
                new_done = jnp.logical_or(new_done, achieved)
            return (s_next, new_done), rec

        init = (start_states, jnp.zeros(n_starts, dtype=bool))
        _, (states, valid) = jax.lax.scan(_step, init, jnp.arange(max_steps))
        return states, valid

    # Accumulate occupancy over all goals into one linear histogram.
    hist = jnp.zeros(R ** state_dim)
    for gi in range(goals.shape[0]):
        states, valid = _rollout_states(goals[gi], goal_masks[gi])
        flat = states.reshape(-1, state_dim)
        fvalid = valid.reshape(-1).astype(jnp.float32)
        # nearest grid cell per dim → linear index (ij order)
        lin = jnp.zeros(flat.shape[0], dtype=jnp.int32)
        for d in range(state_dim):
            frac = (flat[:, d] - los[d]) / (his[d] - los[d]) * (R - 1)
            b = jnp.clip(jnp.round(frac), 0, R - 1).astype(jnp.int32)
            lin = lin * R + b
        hist = hist.at[lin].add(fvalid)

    total = jnp.sum(hist)
    w = jnp.where(total > 0, hist / total, hist)
    return w.reshape((R,) * state_dim)
