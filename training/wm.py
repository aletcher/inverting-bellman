"""World model training: learn P(s,a) -> s' from PQN Q-values."""

import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
from flax.linen.initializers import constant, orthogonal

from envs.goals import compute_reward, goal_achieved, ContinuousGoal
from training.pqn_utils import make_q_network

# ── World model network ──────────────────────────────────────────────────────


class WorldModel(nn.Module):
    activation: str = "tanh"
    hidden_size: int = 64
    num_layers: int = 2
    output_dim: int = 2
    output_init_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        for _ in range(self.num_layers):
            x = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = act_fn(x)
        x = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(self.output_init_scale),
            bias_init=constant(0.0),
        )(x)
        return x


def make_world_model(wm_config, state_dim):
    return WorldModel(
        activation=wm_config["ACTIVATION"],
        hidden_size=wm_config["DENSE_HIDDEN_SIZE"],
        num_layers=wm_config["DENSE_LAYERS"],
        output_dim=state_dim,
        output_init_scale=wm_config.get("OUTPUT_INIT_SCALE", 1.0),
    )


# ── Shared helpers ───────────────────────────────────────────────────────────


def _residual_loss(delta, wm_config):
    """Elementwise residual loss: L1 (default) or L2 (MSE)."""
    return delta ** 2 if wm_config.get("WM_LOSS", "l1") == "mse" else jnp.abs(delta)


def _wrap_angle(theta):
    """Wrap angle to [-π, π]."""
    return (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi


def apply_wm(p_model, p_params, s, a_oh, residual=True, state_to_eff_fn=None,
             angle_dims=None, eff_to_obs_fn=None, wm_input_dims=None):
    """Apply world model: predict next state. With residual=True (default),
    predicts s + delta; with residual=False, predicts s' directly.

    When the WM output dim is smaller than the obs dim (e.g. 4D effective output
    from 6D obs input on Reacher), pass state_to_eff_fn to project s
    before adding the residual: s_pred = state_to_eff_fn(s) + delta.

    When angle_dims is provided, those dims of the final s_pred (in effective
    space) are wrapped to [-π, π]. Needed for envs with raw-angle obs (e.g.
    Reacher) so that rollout chaining and Q-eval stay on the same torus the
    network was trained on.

    When eff_to_obs_fn is provided, the (possibly effective-space) s_pred is
    lifted back to obs space (e.g. 4D → 6D for Reacher, appending fp_xy via
    forward kinematics) before being returned. Use this when the consumer
    expects the full obs (Q-net, true-dynamics comparison, rollout chaining).
    Loss-side callers that already lift manually (e.g. world_model_loss_sampled)
    should leave this kwarg as None.

    When wm_input_dims is provided, s is sliced to those dims along the last
    axis BEFORE concatenation with a_oh. This narrows the network input width —
    e.g. Reacher with WM_INPUT_DIMS=[0,1,2,3] makes the WM a 4D-effective
    network: input = effective state + action, output = next effective state.
    Independent of state_to_eff_fn (which only governs the residual base in
    output space).
    """
    s_in = s if wm_input_dims is None else s[..., jnp.array(tuple(wm_input_dims))]
    p_input = jnp.concatenate([s_in, a_oh], axis=-1)
    s_pred = p_model.apply(p_params, p_input)
    if residual:
        base = state_to_eff_fn(s) if state_to_eff_fn is not None else s
        s_pred = base + s_pred
    if angle_dims is not None and len(angle_dims) > 0:
        idx = jnp.array(tuple(angle_dims))
        s_pred = s_pred.at[..., idx].set(_wrap_angle(s_pred[..., idx]))
    if eff_to_obs_fn is not None:
        s_pred = eff_to_obs_fn(s_pred)
    return s_pred


# ── Helpers ──────────────────────────────────────────────────────────────────


def sample_states(rng, state_ranges, state_dim, n_states):
    """Sample random states uniformly from state_ranges."""
    mins = jnp.array([r[0] for r in state_ranges])
    maxs = jnp.array([r[1] for r in state_ranges])
    return jax.random.uniform(rng, (n_states, state_dim), minval=mins, maxval=maxs)


# ── Training ─────────────────────────────────────────────────────────────────


def world_model_loss_sampled(
    p_params,
    batch_s,
    batch_a,
    batch_goal,
    batch_mask,
    q_params,
    q_batch_stats,
    gamma,
    wm_config,
    pqn_config,
    reward_type,
    sigma,
    a_threshold,
    state_dim,
    action_dim,
    terminate_on_goal,
    env_terminated_fn,
    state_to_eff_fn=None,
    eff_to_obs_fn=None,
    wm_output_dim=None,
):
    """||Q(s,a,g) - M(P(s,a),g)||^p with independently sampled (s, a, g) tuples.

    If eff_to_obs_fn is provided, the WM predicts in the effective state space
    (dim = wm_output_dim, e.g. 4 for Reacher) and is lifted back to obs space
    via eff_to_obs_fn for reward/done.
    """
    out_dim = wm_output_dim if wm_output_dim is not None else state_dim
    p_model = make_world_model(wm_config, out_dim)
    residual = wm_config.get("RESIDUAL_PREDICTION", True)
    angle_dims = wm_config.get("ANGLE_DIMS")
    wm_input_dims = wm_config.get("WM_INPUT_DIMS")
    n = batch_s.shape[0]

    effective_output = eff_to_obs_fn is not None  # WM output is effective state

    a_oh = jax.nn.one_hot(batch_a, action_dim)
    s_pred = apply_wm(
        p_model, p_params, batch_s, a_oh,
        residual=residual,
        state_to_eff_fn=state_to_eff_fn if effective_output else None,
        angle_dims=angle_dims,
        wm_input_dims=wm_input_dims,
    )
    s_pred_obs = eff_to_obs_fn(s_pred) if effective_output else s_pred

    goal_repr = ContinuousGoal(target_state=batch_goal, reward_mask=batch_mask)
    network = make_q_network(pqn_config)
    q_vars = {"params": q_params, "batch_stats": q_batch_stats}

    q_all = network.apply(q_vars, batch_s, goal_repr, train=False)
    target = q_all[jnp.arange(n), batch_a]

    q_pred = network.apply(q_vars, s_pred_obs, goal_repr, train=False)
    v_pred = jnp.max(q_pred, axis=-1)

    r_pred = compute_reward(s_pred_obs, batch_goal, batch_mask, reward_type, sigma, a_threshold)
    done_pred = jnp.zeros(n)
    if terminate_on_goal:
        done_pred = goal_achieved(s_pred_obs, batch_goal, batch_mask, a_threshold).astype(float)
    if env_terminated_fn is not None:
        done_pred = jnp.maximum(done_pred, env_terminated_fn(s_pred_obs).astype(float))
    m_pred = r_pred + gamma * (1 - done_pred) * v_pred

    return jnp.mean(_residual_loss(target - m_pred, wm_config))


def train_world_model(
    q_params,
    q_batch_stats,
    wm_config,
    pqn_config,
    goals,
    goal_masks,
    env_terminated_fn=None,
    sample_states_fn=None,
    use_wandb=False,
    state_to_eff_fn=None,
    eff_to_obs_fn=None,
    wm_output_dim=None,
):
    """Train world model P(s,a) -> s'. Returns (p_params, step_losses).

    Samples (s, a, g) independently each step and minimises the Bellman
    residual ||Q(s,a,g) - M(P(s,a),g)|| where M is the one-step backup
    under the PQN policy.
    """
    gamma = pqn_config["GAMMA"]
    reward_type = pqn_config["REWARD_TYPE"]
    sigma = pqn_config["REWARD_SIGMA"]
    a_threshold = pqn_config["REWARD_A"]
    terminate_on_goal = pqn_config["TERMINATE_ON_GOAL"]
    state_ranges = pqn_config["STATE_RANGES"]
    state_dim = pqn_config["STATE_DIM"]
    action_dim = pqn_config["ACTION_DIM"]
    num_goals = len(goals)

    num_steps = wm_config["NUM_STEPS"]
    lr = wm_config["LR"]
    batch_size = wm_config["BATCH_SIZE"]

    out_dim = wm_output_dim if wm_output_dim is not None else state_dim
    p_model = make_world_model(wm_config, out_dim)
    rng = jax.random.PRNGKey(wm_config.get("SEED", 0))
    rng, init_rng = jax.random.split(rng)
    # WM input width: state_dim + action_dim by default; if WM_INPUT_DIMS is set,
    # the network input is sliced to those obs dims (Reacher 4D-effective input
    # mode: WM_INPUT_DIMS=[0,1,2,3] → input width = 4 + action_dim).
    wm_input_dims = wm_config.get("WM_INPUT_DIMS")
    in_dim = len(wm_input_dims) if wm_input_dims is not None else state_dim
    p_params = p_model.init(init_rng, jnp.zeros((1, in_dim + action_dim)))

    lr_schedule_type = wm_config["LR_SCHEDULE"]
    if lr_schedule_type == "cosine":
        schedule = optax.cosine_decay_schedule(lr, num_steps)
    elif lr_schedule_type == "linear":
        schedule = optax.linear_schedule(lr, 1e-20, num_steps)
    else:  # constant
        schedule = lr

    tx = optax.adam(schedule)
    opt_state = tx.init(p_params)

    print(f"[WM] {num_steps} steps, batch_size={batch_size}, "
          f"loss={wm_config.get('WM_LOSS', 'l1')}, lr_schedule={lr_schedule_type}")

    wandb_log_interval = wm_config.get("WANDB_LOG_INTERVAL", 100)
    if use_wandb:
        import wandb as _wandb

        def _wandb_callback(step, loss):
            # Don't pass an explicit step — PQN already advanced wandb's internal
            # step counter; wm/* uses wm/step as its custom axis (see
            # wandb.define_metric in run.py).
            if int(step) % wandb_log_interval == 0:
                _wandb.log({"wm/loss": float(loss), "wm/step": int(step)})

    def _update_step(carry, step):
        p_params, opt_state, rng = carry

        rng, rng_s, rng_a, rng_g = jax.random.split(rng, 4)
        if sample_states_fn is not None:
            batch_s = sample_states_fn(rng_s, batch_size)
        else:
            batch_s = sample_states(rng_s, state_ranges, state_dim, batch_size)
        batch_a = jax.random.randint(rng_a, (batch_size,), 0, action_dim)
        goal_idxs = jax.random.randint(rng_g, (batch_size,), 0, num_goals)
        batch_goal = goals[goal_idxs]
        batch_mask = goal_masks[goal_idxs]

        loss, grads = jax.value_and_grad(world_model_loss_sampled)(
            p_params,
            batch_s,
            batch_a,
            batch_goal,
            batch_mask,
            q_params,
            q_batch_stats,
            gamma,
            wm_config,
            pqn_config,
            reward_type,
            sigma,
            a_threshold,
            state_dim,
            action_dim,
            terminate_on_goal,
            env_terminated_fn,
            state_to_eff_fn,
            eff_to_obs_fn,
            wm_output_dim,
        )

        updates, opt_state = tx.update(grads, opt_state, p_params)
        p_params = optax.apply_updates(p_params, updates)

        if use_wandb:
            jax.debug.callback(_wandb_callback, step, loss)

        return (p_params, opt_state, rng), loss

    print(f"[WM] JIT-compiling + running {num_steps} steps...")
    t0 = time.time()
    (p_params, _, _), step_losses = jax.jit(
        lambda carry: jax.lax.scan(
            _update_step, carry, jnp.arange(num_steps), length=num_steps,
        )
    )((p_params, opt_state, rng))
    step_losses = jax.block_until_ready(step_losses)
    print(f"[WM] done in {time.time() - t0:.1f}s; "
          f"loss {float(step_losses[0]):.6f} -> {float(step_losses[-1]):.6f}")

    return p_params, step_losses


# ── Checkpoint IO ────────────────────────────────────────────────────────────


def save_wm(p_params, losses, path, config=None):
    """Save world model params, training losses, and optionally config."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "params": jax.tree.map(np.array, p_params),
        "losses": np.array(losses),
    }
    if config is not None:
        data["config"] = config
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[WM] saved checkpoint to {path}")


def load_wm(path):
    """Load world model params, losses, and config (if present).

    Returns (p_params, losses, config_or_None).
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[WM] loaded checkpoint from {path}")
    return data["params"], data.get("losses"), data.get("config")
