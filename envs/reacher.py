"""Reacher with discrete-action torque grid.

The 6D obs layout matches MuJoCo Playground's Reacher [1]:
    [θ₁, θ₂, ω₁, ω₂, fp_x, fp_y]    (angles wrapped to [-π, π])
Fingertip coords stay in obs for HER-friendly goal-conditioning.

We build on top of `gymnax.environments.misc.reacher.Reacher` [2] so the
repo can stay on a single JAX-RL package for MountainCar + Reacher.
Two changes vs. gymnax implementation:

  - Uses raw angles instead of cos/sin encoding
    (matches MuJoCo Playground).
  - Discrete action space: an N² torque grid (default 3×3, torque
    values [-1, 0, 1] per joint) instead of the continuous box.

Reward is zeroed in `step_env` — the trainer supplies its own
goal-conditioned reward via `goals.continuous_goal.compute_reward`.

[1] Google DeepMind. *MuJoCo Playground*. https://github.com/google-deepmind/mujoco_playground
[2] Lange, R. T. *gymnax: A JAX-based Reinforcement Learning Environment
    Library*, 2022. https://github.com/RobertTLange/gymnax
"""

import jax
import jax.numpy as jnp
from gymnax.environments.misc.reacher import (
    Reacher as _BaseReacher, EnvParams, EnvState as ReacherState,
)
from gymnax.environments import spaces

from envs.reacher_utils import make_action_table


class Reacher(_BaseReacher):
    """Reacher env with raw-angle obs and discrete actions.

    Obs layout (6D):
        dims 0–1: θ₁, θ₂  (joint angles wrapped to [-π, π])
        dims 2–3: ω₁, ω₂  (joint angular velocities)
        dims 4–5: fp_x, fp_y  (fingertip cartesian position)

    Goal conditioning: use ContinuousGoal with
        target_state = [0, 0, 0, 0, goal_x, goal_y]
        reward_mask  = [0, 0, 0, 0, 1,      1     ]
    so reward = Gaussian on fingertip error (obs[4:6] - goal[4:6]).

    HER relabeling: achieved goal = obs[4:6] at episode end.
    """

    def __init__(self, reward_type="gaussian", sigma=0.1, a=0.1,
                 max_steps_in_episode=100, torque_values=None):
        super().__init__(num_joints=2)
        self._reward_type = reward_type
        self._sigma = sigma
        self._a = a
        self._max_steps_in_episode = max_steps_in_episode
        if torque_values is None:
            torque_values = [-1.0, 0.0, 1.0]
        self._action_table = make_action_table([torque_values, torque_values])

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self._max_steps_in_episode)

    def _get_obs_custom(self, state: ReacherState) -> jnp.ndarray:
        """Compute 6D obs: [θ (wrapped to [-π, π]), ω, fp_xy]."""
        angles = (state.angles + jnp.pi) % (2 * jnp.pi) - jnp.pi
        fingertip = jnp.array([jnp.cos(state.angles).sum(),
                               jnp.sin(state.angles).sum()])
        return jnp.concatenate([angles, state.angle_vels, fingertip])

    def step(self, key, state, action, params=None):
        """Override gymnax auto-reset to capture pre-reset obs for PEB."""
        if params is None:
            params = self.default_params
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)
        info["obs_before_reset"] = obs_st
        state = jax.tree.map(
            lambda x, y: jnp.where(done, x, y), state_re, state_st
        )
        obs = jnp.where(done, obs_re, obs_st)
        return obs, state, reward, done, info

    def step_env(self, key, state, action, params):
        """Physics step: maps discrete action → torque, returns 6D obs."""
        torque = self._action_table[action]
        _, new_state, _, done, info = super().step_env(key, state, torque, params)
        obs = jax.lax.stop_gradient(self._get_obs_custom(new_state))
        return obs, new_state, 0.0, done, info

    def reset_env(self, key, params):
        """Reset to random joint configuration; return custom 6D obs."""
        _, state = super().reset_env(key, params)
        obs = self._get_obs_custom(state)
        return obs, state

    def action_space(self, params=None):
        """Discrete action space: N² actions (N = len(torque_values))."""
        return spaces.Discrete(len(self._action_table))

    @property
    def name(self):
        return "Reacher"
