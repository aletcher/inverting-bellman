"""State samplers for envs with coupling constraints between obs dims.

Reacher's 6D obs is [θ₁, θ₂, ω₁, ω₂, fp_x, fp_y] where fp_xy is forward
kinematics of the angles — sampling independently per dim would land on
off-manifold states. make_reacher_uniform_sampler samples (θ, ω) freely
and computes fp_xy by FK.

MountainCar's 2D obs has no such constraint, so callers can sample
jax.random.uniform directly over STATE_RANGES — no helper needed.
"""

import jax
import jax.numpy as jnp


def make_reacher_uniform_sampler(state_ranges):
    """Uniform sampler over Reacher's 6D obs respecting fp = FK(θ).

    state_ranges is the env's full STATE_RANGES; the velocity bounds are
    read from index 2 (ω range, shared across both joints).
    """
    vel_range = state_ranges[2]

    def sampler(rng, n):
        rng1, rng2 = jax.random.split(rng)
        angles = jax.random.uniform(rng1, (n, 2), minval=-jnp.pi, maxval=jnp.pi)
        angle_vels = jax.random.uniform(
            rng2, (n, 2), minval=vel_range[0], maxval=vel_range[1],
        )
        fp_x = jnp.cos(angles).sum(axis=-1, keepdims=True)
        fp_y = jnp.sin(angles).sum(axis=-1, keepdims=True)
        return jnp.concatenate([angles, angle_vels, fp_x, fp_y], axis=-1)

    return sampler


def make_env_reset_sampler(basic_env, env_params):
    """Vmapped env.reset() sampler. Returns shape (n, obs_dim) obs samples."""
    def sampler(rng, n):
        rngs = jax.random.split(rng, n)
        return jax.vmap(lambda r: basic_env.reset(r, env_params)[0])(rngs)

    return sampler
