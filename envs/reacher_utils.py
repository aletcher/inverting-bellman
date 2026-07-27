"""Reacher effective-state (4D) lift/project + dynamics-fn wrappers.

The raw Reacher obs is 6D: [θ₁, θ₂, ω₁, ω₂, fp_x, fp_y]. The underlying
dynamics live on 4D: [θ₁, θ₂, ω₁, ω₂]; fp_xy is a deterministic function
of the angles (forward kinematics). Exposing the env's dynamics in this
4D effective basis lets downstream consumers (VI, unseen-goal planning)
work on a tractable 4D grid rather than a 6D grid containing off-manifold
volume in the fingertip dims.

Provides:
  - reacher_obs_to_effective / reacher_effective_to_obs: pure projections.
  - make_action_table: Cartesian product of per-joint torque values.
  - make_reacher_effective_dynamics_fn: 4D-effective true dynamics.
  - make_reacher_wm_dynamics_fn: 4D-effective learned-WM dynamics.
"""

import itertools

import jax.numpy as jnp

from envs.env_dynamics import make_env_dynamics_fn


def make_action_table(grid_values_per_dim):
    """Discretize a multi-dim continuous action space into a flat lookup table.

    grid_values_per_dim: list of per-dim value lists, e.g.
    [[-1., 0., 1.], [-1., 0., 1.]] (Reacher's two joint torques).
    Returns shape (n_actions, n_dims) where rows enumerate the Cartesian
    product. For two 3-value dims: 9 actions, table[0] = [-1, -1],
    table[4] = [0, 0], table[8] = [1, 1].
    """
    return jnp.array(list(itertools.product(*grid_values_per_dim)), dtype=float)


def reacher_obs_to_effective(obs_6d):
    """(..., 6) -> (..., 4): drop fp_xy, keep [θ₁, θ₂, ω₁, ω₂]."""
    return obs_6d[..., 0:4]


def reacher_effective_to_obs(s_4d):
    """(..., 4) -> (..., 6): append fp_xy via forward kinematics."""
    theta = s_4d[..., 0:2]
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    fp_x = cos_t[..., 0:1] + cos_t[..., 1:2]
    fp_y = sin_t[..., 0:1] + sin_t[..., 1:2]
    return jnp.concatenate([s_4d, fp_x, fp_y], axis=-1)


def _chunked_dynamics(forward_fn, chunk_size):
    """Wrap a (states, actions) -> next_states forward_fn with lax.scan chunking.

    When the input batch exceeds chunk_size, splits into sequential chunks
    of static shape chunk_size (padding the trailing partial chunk) so that
    jax.lax.scan can process them. Caps memory at O(chunk_size) instead
    of O(n_cells), regardless of forward_fn's internal allocations.

    Used by both the WM dynamics fn (where peak activation per layer is
    9·chunk·hidden bytes under vmap-over-actions) and the true dynamics fn
    (where the peak is dominated by env-step intermediate buffers at very
    high grid resolutions). Identical chunking for both means VI on true and
    VI on WM hit the same memory ceiling.

    chunk_size=None ⇒ no chunking (single forward pass). Caller must ensure
    the full input fits.
    """
    if chunk_size is None:
        def f(s, a):
            return forward_fn(s, a)
        return f

    import jax

    def f(s, a):
        n = s.shape[0]
        if n <= chunk_size:
            return forward_fn(s, a)
        n_chunks = (n + chunk_size - 1) // chunk_size
        pad = n_chunks * chunk_size - n
        s_p = jnp.pad(s, ((0, pad), (0, 0)))
        a_p = jnp.pad(a, (0, pad))
        s_chunks = s_p.reshape(n_chunks, chunk_size, s.shape[1])
        a_chunks = a_p.reshape(n_chunks, chunk_size)

        def step(_carry, args):
            s_chunk, a_chunk = args
            return None, forward_fn(s_chunk, a_chunk)

        _, out_chunks = jax.lax.scan(step, None, (s_chunks, a_chunks))
        out = out_chunks.reshape(n_chunks * chunk_size, -1)
        return out[:n]

    return f


def make_reacher_effective_dynamics_fn(env, env_params,
                                           chunk_size=1_048_576):
    """Return 4D -> 4D dynamics by lifting to 6D, stepping, projecting back.

    chunk_size: see _chunked_dynamics. Default mirrors the WM dynamics fn
    so VI on the true env can be pushed to grid=80+ without OOM. The peak
    intermediate inside env.step is small for Reacher (no NN forward, just
    angle integration), so the real beneficiary at high grid is the
    vmap-over-9-actions broadcast that happens in precompute_dynamics.
    For the legacy single-action call shape (used by the trajectory rollouts
    (unused now), n is much smaller than chunk_size, so this is a no-op.

    Pass chunk_size=None to disable chunking (matches the pre-chunking
    behavior; safe at grid≤50 on an 80 GB H100).
    """
    f6 = make_env_dynamics_fn(env, env_params, "Reacher")

    def _forward(s_4d, a):
        return reacher_obs_to_effective(f6(reacher_effective_to_obs(s_4d), a))

    return _chunked_dynamics(_forward, chunk_size)


def make_reacher_wm_dynamics_fn(p_params, wm_config, action_dim,
                                    chunk_size=1_048_576):
    """Return 4D -> 4D dynamics through the learned world model.

    Two WM input modes are supported, both producing 4D-effective output:

    - **Legacy 6D-input WM** (WM_INPUT_DIMS unset in wm_config): the WM was
      trained with 6D obs as network input and 4D effective output (residual
      base lifted via state_to_eff_fn=reacher_obs_to_effective). Inference
      lifts the 4D grid query to 6D before forwarding, then takes the 4D
      effective output. This matches reacher-style WMs.

    - **4D-input WM** (WM_INPUT_DIMS=[0,1,2,3]): the WM is a true 4D→4D
      function (network input width = 4 + action_dim). The 4D grid query is
      passed directly with no lift; the residual base is the input itself.

    chunk_size: when the input batch (typically n_cells = N⁴ for grid=N
    inside precompute_dynamics) exceeds this, the forward pass is split
    into sequential chunks via jax.lax.scan. Activations cap at
    chunk_size · WM_DENSE_HIDDEN_SIZE, so memory becomes O(chunk_size)
    instead of O(N⁴).

    Default of 2²⁰ ≈ 1.05 M was chosen empirically to fit at grid=50 on an
    80 GB H100 under vmap-over-9-actions + f32 + 'default' matmul precision
    for the largest WM in the sweep (WM_DENSE_HIDDEN_SIZE=1024,
    WM_DENSE_LAYERS=4): peak per-layer activation under vmap ≈
    9 · chunk · 1024 · 4 ≈ 38 GB, plus ~50 GB for the (n_actions, n_cells, 4)
    next-state grid + Q-table at f32. chunk=2 M OOMs at this WM size.

    For smaller WMs (hidden=256), this default leaves headroom — could push
    to ~4 M, but 1 M keeps a single setting safe across the entire (D, W)
    sweep without per-cell tuning.

    Pass chunk_size=None to disable chunking entirely (safe at grid≤40
    at f32 only for hidden=256; OOMs at any grid≥45 or hidden=1024).

    Used by VI-on-WM for unseen-goal planning evaluation. Reward/done are
    computed in 6D obs space via a separate state_to_obs_fn lift inside
    the VI machinery.
    """
    import jax
    from training.wm import make_world_model, apply_wm

    wm_output_dim = wm_config.get("WM_OUTPUT_DIM", 4)
    if wm_output_dim != 4:
        raise ValueError(
            f"make_reacher_wm_dynamics_fn expects WM_OUTPUT_DIM=4 "
            f"(effective Reacher state), got {wm_output_dim}."
        )
    p_model = make_world_model(wm_config, wm_output_dim)
    residual = wm_config.get("RESIDUAL_PREDICTION", True)
    angle_dims = wm_config.get("ANGLE_DIMS", [0, 1])
    wm_input_dims = wm_config.get("WM_INPUT_DIMS")
    # 4D-input WM: feed the effective query directly. Otherwise lift to 6D obs.
    input_is_effective = (
        wm_input_dims is not None and list(wm_input_dims) == [0, 1, 2, 3]
    )

    def _forward(s_4d_batch, a_batch):
        a_oh = jax.nn.one_hot(a_batch, action_dim)
        if input_is_effective:
            # Network input is 4D effective; residual base = input.
            return apply_wm(
                p_model, p_params, s_4d_batch, a_oh, residual=residual,
                state_to_eff_fn=None,
                angle_dims=angle_dims,
                eff_to_obs_fn=None,
                wm_input_dims=None,  # caller already in effective space
            )
        # Legacy: lift 4D → 6D obs, residual base via obs_to_effective.
        s_6d = reacher_effective_to_obs(s_4d_batch)
        return apply_wm(
            p_model, p_params, s_6d, a_oh, residual=residual,
            state_to_eff_fn=reacher_obs_to_effective,  # 6D → 4D residual base
            angle_dims=angle_dims,
            eff_to_obs_fn=None,                            # output stays 4D effective
        )

    return _chunked_dynamics(_forward, chunk_size)
