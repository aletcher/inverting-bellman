"""MountainCar with externally-supplied reward (for goal-conditioned PQN).

Wraps gymnax.environments.classic_control.mountain_car.MountainCar [1] —
same physics, just zeroes out the built-in reward so the trainer can
supply its own goal-conditioned reward, and drops the gymnax baseline's
goal-position termination (the trainer's goal-conditioned predicate is
the only termination signal; otherwise episodes end on max_steps).

One behavioural tweak vs. the gymnax baseline: step is overridden so
the pre-reset obs is exposed in info under "obs_before_reset" (needed
for partial-episode bootstrap).

[1] Lange, R. T. *gymnax: A JAX-based Reinforcement Learning Environment
    Library*, 2022. https://github.com/RobertTLange/gymnax
"""

import jax
import jax.numpy as jnp
from gymnax.environments.classic_control.mountain_car import (
    MountainCar as _GymnaxMountainCar,
    EnvParams,
    EnvState,
)


class MountainCar(_GymnaxMountainCar):
    """MountainCar with external reward; only truncation, no env-level termination."""

    def __init__(self, max_steps_in_episode=200):
        super().__init__()
        self._max_steps_in_episode = max_steps_in_episode

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self._max_steps_in_episode)

    def step(self, key, state, action, params=None):
        """Override gymnax auto-reset to capture pre-reset obs for PEB."""
        if params is None:
            params = self.default_params
        key_step, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key_step, state, action, params
        )
        obs_re, state_re = self.reset_env(key_reset, params)

        # Store true next-state obs BEFORE auto-reset
        info["obs_before_reset"] = obs_st

        state = jax.tree.map(
            lambda x, y: jnp.where(done, x, y), state_re, state_st
        )
        obs = jnp.where(done, obs_re, obs_st)
        return obs, state, reward, done, info

    def step_env(self, key, state, action, params):
        # Inline physics (mirrors gymnax MountainCar.step_env).
        # Velocity is zeroed when hitting the left wall (gymnax behaviour).
        velocity = (
            state.velocity
            + (action - 1) * params.force
            - jnp.cos(3 * state.position) * params.gravity
        )
        velocity = jnp.clip(velocity, -params.max_speed, params.max_speed)
        position = state.position + velocity
        position = jnp.clip(position, params.min_position, params.max_position)
        velocity = velocity * (1 - (position == params.min_position) * (velocity < 0))

        new_state = EnvState(position=position, velocity=velocity, time=state.time + 1)
        done = self.is_terminal(new_state, params)
        obs = jax.lax.stop_gradient(self.get_obs(new_state))
        info = {"discount": self.discount(new_state, params)}

        # Reward is computed externally by goal-conditioned training.
        return obs, new_state, 0.0, done, info

    def is_terminal(self, state, params):
        return state.time >= params.max_steps_in_episode

    def reset_env(self, key, params):
        return super().reset_env(key, params)

    @property
    def name(self):
        return "MountainCar"
