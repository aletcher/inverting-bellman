"""Goal representation, reward/achieved checks, and goal-index utilities.

ContinuousGoal bundles target_state and reward_mask for JAX-compatible
indexing and tree operations. Reward and goal-achieved functions take raw
arrays (not the struct) so they work for all callers (PQN, WM, VI).

GoalIndex + sample_goal are the env-agnostic helpers used by the PQN trainer
to draw and look up goals during rollout.
"""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ContinuousGoal:
    target_state: jnp.ndarray  # shape (num_goals, state_dim)
    reward_mask: jnp.ndarray  # shape (num_goals, state_dim) — per-goal mask for reward dims


@struct.dataclass
class GoalIndex:
    goal_index: jnp.ndarray
    num_goals_completed: jnp.ndarray


def compute_reward(obs, target_state, reward_mask, reward_type, sigma, a):
    """Compute reward; reward_mask selects relevant dims.

    reward_type must be one of: "sparse", "sparse_negative", "gaussian", "l2".
    Works with any batch shape — obs and target_state are broadcast-compatible.
    """
    diff = (obs - target_state) * reward_mask
    dist_sq = (diff**2).sum(axis=-1)

    if reward_type == "sparse":
        return jnp.where(dist_sq < a**2, 1.0, 0.0)
    elif reward_type == "sparse_negative":
        return jnp.where(dist_sq < a**2, 0.0, -1.0)
    elif reward_type == "gaussian":
        return jnp.exp(-dist_sq / (2 * sigma**2))
    else:  # l2
        return -dist_sq


def goal_achieved(obs, target_state, reward_mask, a, invert=False):
    """Binary: is obs within threshold a of goal (masked dims)?

    invert=True flips the predicate (True when obs is OUTSIDE the threshold).
    Works with any batch shape.
    """
    diff = (obs - target_state) * reward_mask
    dist_sq = (diff**2).sum(axis=-1)
    within = dist_sq < a**2
    return ~within if invert else within


def goal_indexes_to_goals(all_goals, goal_indexes):
    return jax.tree.map(lambda x: x[goal_indexes.goal_index], all_goals)


def sample_goal(rng, num_goals):
    rng, _rng = jax.random.split(rng)
    goal_index = jax.random.randint(
        _rng,
        minval=0,
        maxval=num_goals,
        shape=(),
    )
    return GoalIndex(
        goal_index=goal_index,
        num_goals_completed=jnp.asarray(0),
    )
