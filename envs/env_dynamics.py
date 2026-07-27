"""Batched single-step dynamics functions backed by env.step_env.

make_env_dynamics_fn(env, env_params, env_name) returns a vmapped
dynamics_fn(s, a) -> s_next — the true transition kernel that VI runs
on (eval/value_iteration.py), that WM error is measured against
(eval/world_model.py, eval/track_wm.py, eval/unseen_goals.py), and that
true-rollout trajectories use in the paper figure scripts. NOT used in
WM training — the WM loss is the Bellman residual on Q-values, never
a direct regression on s_next.
"""

import jax
import jax.numpy as jnp

from gymnax.environments.classic_control.mountain_car import EnvState as MCEnvState
from gymnax.environments.misc.reacher import EnvState as ReacherEnvState


def mc_obs_to_state(obs):
    """Convert a MountainCar observation vector to gymnax's EnvState."""
    return MCEnvState(
        position=obs[..., 0],
        velocity=obs[..., 1],
        time=0,
    )


def reacher_obs_to_state(obs):
    """Convert a raw-angle Reacher observation to gymnax's EnvState.

    Raw obs layout: [θ₁, θ₂, ω₁, ω₂, fp_x, fp_y]. goal_xy is set to zeros
    (ignored by Reacher which uses its own goal mechanism).
    """
    return ReacherEnvState(
        angles=obs[..., 0:2],
        angle_vels=obs[..., 2:4],
        goal_xy=jnp.zeros(2),
        time=0,
    )


# Registry of obs_to_state functions per environment.
OBS_TO_STATE = {
    "MountainCar": mc_obs_to_state,
    "Reacher": reacher_obs_to_state,
}


def make_env_dynamics_fn(env, env_params, env_name):
    """Return a batched dynamics function (s, a) -> s_next using env.step_env.

    s has shape (..., state_dim), a has shape (...,).
    """
    obs_to_state = OBS_TO_STATE[env_name]

    def _single_step(obs, action):
        state = obs_to_state(obs)
        dummy_key = jax.random.PRNGKey(0)
        next_obs, _, _, _, _ = env.step_env(dummy_key, state, action, env_params)
        return next_obs

    def dynamics_fn(s, a):
        return jax.vmap(_single_step)(s, a)

    return dynamics_fn
