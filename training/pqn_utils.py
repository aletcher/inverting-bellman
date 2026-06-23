"""PQN-specific helpers: Q-network builder, goal broadcaster, train / save / load."""

import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from training.pqn import QNetwork, make_train
from envs.goals import ContinuousGoal
from configs.utils import serialize_for_pickle


# ── Q-network helpers ───────────────────────────────────────────────────────


def make_q_network(pqn_config):
    """Build QNetwork from PQN config dict."""
    return QNetwork(
        action_dim=pqn_config["ACTION_DIM"],
        num_goals=pqn_config["NUM_GOALS"],
        dense_hidden_size=pqn_config["NETWORK_DENSE_HIDDEN_SIZE"],
        dense_layers=pqn_config["NETWORK_DENSE_LAYERS"],
        norm_type=pqn_config["NORM_TYPE"],
        sigmoid_output=pqn_config["NETWORK_SIGMOID_OUTPUTS"],
        goal_input_dims=tuple(pqn_config["GOAL_INPUT_DIMS"]) if pqn_config.get("GOAL_INPUT_DIMS") else None,
        obs_input_dims=tuple(pqn_config["OBS_INPUT_DIMS"]) if pqn_config.get("OBS_INPUT_DIMS") else None,
    )


def make_goal_repr(goal, mask, n, state_dim):
    """Broadcast a single goal + mask to batch size n for Q-network input."""
    return ContinuousGoal(
        target_state=jnp.broadcast_to(goal[None, :], (n, state_dim)),
        reward_mask=jnp.broadcast_to(mask[None, :], (n, state_dim)),
    )


# ── PQN checkpoints ──────────────────────────────────────────────────────────


def save_pqn(q_params, q_batch_stats, path, config=None, metrics=None):
    """Save PQN params, batch_stats, and optionally config/metrics to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "params": jax.tree.map(np.array, q_params),
        "batch_stats": jax.tree.map(np.array, q_batch_stats),
    }
    if config is not None:
        data["config"] = serialize_for_pickle(config)
    if metrics is not None:
        data["metrics"] = jax.tree.map(np.array, metrics)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[PQN] saved checkpoint to {path}")


def load_pqn(path):
    """Load PQN params, batch_stats, and metrics (if present) from a pickle file.

    Returns (params, batch_stats, metrics_or_None).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[PQN] loaded checkpoint from {path}")
    return data["params"], data["batch_stats"], data.get("metrics")


def train_pqn(config, basic_env, env_params, all_goals, seed=42, checkpoint_dir=None):
    """Train PQN and return (q_params, q_batch_stats, metrics).

    metrics is a dict of arrays with shape (NUM_UPDATES,), including
    'returned_episode_returns', 'td_loss', 'qvals', etc.
    """
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    train_fn = jax.jit(make_train(config, basic_env, env_params, all_goals, checkpoint_dir=checkpoint_dir))
    t0 = time.time()
    out = jax.block_until_ready(train_fn(_rng))
    dt = time.time() - t0
    train_state = out["runner_state"][0]
    metrics = out["metrics"]
    print(f"[PQN] trained in {dt:.1f}s")
    return train_state.params, train_state.batch_stats, metrics


def train_or_load_pqn(config, basic_env, env_params, all_goals, checkpoint_path=None, checkpoint_dir=None):
    """Load PQN from checkpoint if provided, otherwise train.

    Does NOT save — callers save to their own OUT_DIR.
    Returns (q_params, q_batch_stats, metrics_or_None).
    """
    if checkpoint_path:
        return load_pqn(checkpoint_path)
    return train_pqn(
        config, basic_env, env_params, all_goals,
        seed=config.get("SEED", 42),
        checkpoint_dir=checkpoint_dir,
    )
