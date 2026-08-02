"""Policy-only world model training (pi-learning): learn P(s,a) -> s' from
Boltzmann-rational policies, without access to Q-value magnitudes.

Assumes pi_g = softmax(Q_g / tau) for some Q_g satisfying a Bellman equation
w.r.t. an implicit shared world model. Boltzmann consistency pins Q up to a
per-state constant: Q_g(s,a) = tau * log pi_g(a|s) + V_g(s). Substituting
into the Bellman residual eliminates Q, leaving the world model f_phi and a
per-goal value net V_psi as the only unknowns:

    L(phi, psi) = E_{s,a,g} rho( tau*log pi_g(a|s) + V_psi(s,g)
                                 - [ r_g(s') + gamma*(1-d_g(s'))*V_psi(s',g) ] ),
    s' = f_phi(s, a).

CONSISTENCY selects how log pi is normalised (a per-state constant, absorbed
into what V_psi must represent):
  - "soft": raw tau*log pi; V_psi ~ soft value tau*logsumexp(Q/tau). Exact for
    an entropy-regularised agent; O(gamma*tau*log|A|) bias for a hard-Bellman
    agent such as PQN.
  - "hard": max-normalised, tau*(log pi - max_a' log pi) = Q - max_a' Q for
    any tau; V_psi ~ max_a Q. Well-specified for PQN checkpoints.

The extractor only ever sees log_softmax of the agent's logits — the
per-state Q level is discarded, so information-theoretically the method has
access to pi_g and r_g only.
"""

import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from envs.goals import compute_reward, goal_achieved, ContinuousGoal
from training.pqn import QNetwork
from training.pqn_utils import make_q_network, make_goal_repr
from training.wm import make_world_model, apply_wm, sample_states, _residual_loss


# ── Networks ─────────────────────────────────────────────────────────────────


def make_value_network(piwm_config, pqn_config):
    """Goal-conditioned scalar value net V_psi(s, g).

    Reuses QNetwork with action_dim=1 so obs/goal dim-slicing and LayerNorm
    match the agent's conditioning. The output bound (V_SIGMOID_OUTPUT /
    V_SIGMOID_SCALE) is applied in apply_value, not inside the network, so the
    bound can exceed 1 (soft values overshoot max Q by up to tau*log|A| plus
    accumulated entropy bonuses).
    """
    return QNetwork(
        action_dim=1,
        num_goals=pqn_config["NUM_GOALS"],
        dense_hidden_size=piwm_config["V_DENSE_HIDDEN_SIZE"],
        dense_layers=piwm_config["V_DENSE_LAYERS"],
        norm_type=pqn_config["NORM_TYPE"],
        sigmoid_output=False,
        goal_input_dims=tuple(pqn_config["GOAL_INPUT_DIMS"]) if pqn_config.get("GOAL_INPUT_DIMS") else None,
        obs_input_dims=tuple(pqn_config["OBS_INPUT_DIMS"]) if pqn_config.get("OBS_INPUT_DIMS") else None,
    )


def apply_value(v_model, v_params, obs, goal_repr, piwm_config):
    """V_psi(s, g), bounded in (0, V_SIGMOID_SCALE) when V_SIGMOID_OUTPUT is
    set. Bounding kills the runaway-V degeneracy of the joint (V, WM) fit."""
    v = v_model.apply({"params": v_params}, obs, goal_repr, train=False)[:, 0]
    if piwm_config.get("V_SIGMOID_OUTPUT", False):
        v = piwm_config.get("V_SIGMOID_SCALE", 1.0) * jax.nn.sigmoid(v)
    return v


def normalised_logpi(q_all, tau, consistency):
    """Per-state-normalised log pi from agent logits; the only policy access.

    log_softmax discards the per-state Q level, so downstream code sees the
    Boltzmann policy pi = softmax(q/tau) and nothing more. "hard" additionally
    subtracts the per-state max, making tau*logpi the tau-independent
    advantage Q - max_a' Q.
    """
    logpi = jax.nn.log_softmax(q_all / tau, axis=-1)
    if consistency == "hard":
        logpi = logpi - jnp.max(logpi, axis=-1, keepdims=True)
    elif consistency != "soft":
        raise ValueError(f"Unknown CONSISTENCY: {consistency!r}; expected 'soft' or 'hard'.")
    return logpi


# ── Loss ─────────────────────────────────────────────────────────────────────


def policy_wm_loss_sampled(
    params,
    batch_s,
    batch_a,
    batch_goal,
    batch_mask,
    q_params,
    q_batch_stats,
    gamma,
    piwm_config,
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
    """||tau*log pi(a|s,g) + V(s,g) - M(P(s,a),g)||^p, params = {"wm", "v"}.

    Mirrors world_model_loss_sampled with the Q target replaced by
    tau*log pi + V_psi(s) and the max-Q bootstrap replaced by V_psi(s').
    """
    tau = piwm_config["TAU"]
    consistency = piwm_config.get("CONSISTENCY", "soft")
    out_dim = wm_output_dim if wm_output_dim is not None else state_dim
    p_model = make_world_model(piwm_config, out_dim)
    v_model = make_value_network(piwm_config, pqn_config)
    residual = piwm_config.get("RESIDUAL_PREDICTION", True)
    angle_dims = piwm_config.get("ANGLE_DIMS")
    wm_input_dims = piwm_config.get("WM_INPUT_DIMS")
    n = batch_s.shape[0]

    effective_output = eff_to_obs_fn is not None  # WM output is effective state

    a_oh = jax.nn.one_hot(batch_a, action_dim)
    s_pred = apply_wm(
        p_model, params["wm"], batch_s, a_oh,
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
    logpi_all = normalised_logpi(q_all, tau, consistency)

    v_s = apply_value(v_model, params["v"], batch_s, goal_repr, piwm_config)
    target = tau * logpi_all[jnp.arange(n), batch_a] + v_s

    # Semi-gradient option: freeze V_psi in the bootstrap while keeping the
    # gradient through s_pred_obs into the WM.
    v_boot_params = params["v"]
    if piwm_config.get("V_STOP_GRAD", False):
        v_boot_params = jax.tree.map(jax.lax.stop_gradient, params["v"])
    v_pred = apply_value(v_model, v_boot_params, s_pred_obs, goal_repr, piwm_config)

    r_pred = compute_reward(s_pred_obs, batch_goal, batch_mask, reward_type, sigma, a_threshold)
    done_pred = jnp.zeros(n)
    if terminate_on_goal:
        done_pred = goal_achieved(s_pred_obs, batch_goal, batch_mask, a_threshold).astype(float)
    if env_terminated_fn is not None:
        done_pred = jnp.maximum(done_pred, env_terminated_fn(s_pred_obs).astype(float))
    m_pred = r_pred + gamma * (1 - done_pred) * v_pred

    return jnp.mean(_residual_loss(target - m_pred, piwm_config))


# ── Training ─────────────────────────────────────────────────────────────────


def train_policy_wm(
    q_params,
    q_batch_stats,
    piwm_config,
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
    """Train {world model, value net} jointly from policy log-probs.

    Returns (params, step_losses) with params = {"wm": ..., "v": ...}.
    Samples (s, a, g) independently each step, exactly as train_world_model.
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

    num_steps = piwm_config["NUM_STEPS"]
    lr = piwm_config["LR"]
    batch_size = piwm_config["BATCH_SIZE"]

    out_dim = wm_output_dim if wm_output_dim is not None else state_dim
    p_model = make_world_model(piwm_config, out_dim)
    v_model = make_value_network(piwm_config, pqn_config)
    rng = jax.random.PRNGKey(piwm_config.get("SEED", 0))
    rng, p_rng, v_rng = jax.random.split(rng, 3)

    wm_input_dims = piwm_config.get("WM_INPUT_DIMS")
    in_dim = len(wm_input_dims) if wm_input_dims is not None else state_dim
    p_params = p_model.init(p_rng, jnp.zeros((1, in_dim + action_dim)))
    dummy_goal = make_goal_repr(goals[0], goal_masks[0], 1, state_dim)
    v_params = v_model.init(v_rng, jnp.zeros((1, state_dim)), dummy_goal, train=False)["params"]
    params = {"wm": p_params, "v": v_params}

    def _make_schedule(base_lr):
        schedule_type = piwm_config["LR_SCHEDULE"]
        if schedule_type == "cosine":
            return optax.cosine_decay_schedule(base_lr, num_steps)
        if schedule_type == "linear":
            return optax.linear_schedule(base_lr, 1e-20, num_steps)
        return base_lr  # constant

    v_lr = piwm_config.get("V_LR")
    if v_lr is None:
        tx = optax.adam(_make_schedule(lr))
    else:
        tx = optax.multi_transform(
            {"wm": optax.adam(_make_schedule(lr)), "v": optax.adam(_make_schedule(v_lr))},
            param_labels={
                "wm": jax.tree.map(lambda _: "wm", p_params),
                "v": jax.tree.map(lambda _: "v", v_params),
            },
        )
    opt_state = tx.init(params)

    print(f"[PiWM] {num_steps} steps, batch_size={batch_size}, "
          f"tau={piwm_config['TAU']}, consistency={piwm_config.get('CONSISTENCY', 'soft')}, "
          f"loss={piwm_config.get('WM_LOSS', 'l1')}, lr_schedule={piwm_config['LR_SCHEDULE']}")

    wandb_log_interval = piwm_config.get("WANDB_LOG_INTERVAL", 100)
    if use_wandb:
        import wandb as _wandb

        def _wandb_callback(step, loss):
            if int(step) % wandb_log_interval == 0:
                _wandb.log({"piwm/loss": float(loss), "piwm/step": int(step)})

    def _update_step(carry, step):
        params, opt_state, rng = carry

        rng, rng_s, rng_a, rng_g = jax.random.split(rng, 4)
        if sample_states_fn is not None:
            batch_s = sample_states_fn(rng_s, batch_size)
        else:
            batch_s = sample_states(rng_s, state_ranges, state_dim, batch_size)
        batch_a = jax.random.randint(rng_a, (batch_size,), 0, action_dim)
        goal_idxs = jax.random.randint(rng_g, (batch_size,), 0, num_goals)
        batch_goal = goals[goal_idxs]
        batch_mask = goal_masks[goal_idxs]

        loss, grads = jax.value_and_grad(policy_wm_loss_sampled)(
            params,
            batch_s,
            batch_a,
            batch_goal,
            batch_mask,
            q_params,
            q_batch_stats,
            gamma,
            piwm_config,
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

        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if use_wandb:
            jax.debug.callback(_wandb_callback, step, loss)

        return (params, opt_state, rng), loss

    print(f"[PiWM] JIT-compiling + running {num_steps} steps...")
    t0 = time.time()
    (params, _, _), step_losses = jax.jit(
        lambda carry: jax.lax.scan(
            _update_step, carry, jnp.arange(num_steps), length=num_steps,
        )
    )((params, opt_state, rng))
    step_losses = jax.block_until_ready(step_losses)
    print(f"[PiWM] done in {time.time() - t0:.1f}s; "
          f"loss {float(step_losses[0]):.6f} -> {float(step_losses[-1]):.6f}")

    return params, step_losses


# ── Diagnostics ──────────────────────────────────────────────────────────────


def value_diagnostics(
    params,
    q_params,
    q_batch_stats,
    piwm_config,
    pqn_config,
    goals,
    goal_masks,
    sample_states_fn=None,
    n_eval=4096,
    seed=42,
):
    """Check the nuisance V_psi against its PQN-derived reference, per goal.

    Reference: max_a Q_pqn ("hard") or tau*logsumexp(Q_pqn/tau) ("soft").
    Also scores the reconstructed Q_hat = tau*log pi + V_psi against Q_pqn —
    if V_psi has learned the discarded per-state level, Q_hat recovers Q.
    Returns {goal_idx: {"v_nmse", "q_nmse"}, "avg": {...}}.
    """
    tau = piwm_config["TAU"]
    consistency = piwm_config.get("CONSISTENCY", "soft")
    state_dim = pqn_config["STATE_DIM"]
    state_ranges = pqn_config["STATE_RANGES"]

    rng = jax.random.PRNGKey(seed)
    if sample_states_fn is not None:
        eval_s = sample_states_fn(rng, n_eval)
    else:
        eval_s = sample_states(rng, state_ranges, state_dim, n_eval)

    network = make_q_network(pqn_config)
    q_vars = {"params": q_params, "batch_stats": q_batch_stats}
    v_model = make_value_network(piwm_config, pqn_config)

    metrics = {}
    v_nmses, q_nmses = [], []
    for i in range(len(goals)):
        goal_repr = make_goal_repr(goals[i], goal_masks[i], n_eval, state_dim)
        q_all = network.apply(q_vars, eval_s, goal_repr, train=False)
        if consistency == "hard":
            v_ref = jnp.max(q_all, axis=-1)
        else:
            v_ref = tau * jax.nn.logsumexp(q_all / tau, axis=-1)
        logpi_all = normalised_logpi(q_all, tau, consistency)

        v_psi = np.array(apply_value(v_model, params["v"], eval_s, goal_repr, piwm_config))
        q_hat = tau * logpi_all + v_psi[:, None]

        v_nmse = float(jnp.mean((v_psi - v_ref) ** 2) / jnp.maximum(jnp.var(v_ref), 1e-12))
        q_nmse = float(jnp.mean((q_hat - q_all) ** 2) / jnp.maximum(jnp.var(q_all), 1e-12))
        metrics[i] = {"v_nmse": v_nmse, "q_nmse": q_nmse}
        v_nmses.append(v_nmse)
        q_nmses.append(q_nmse)

    metrics["avg"] = {
        "v_nmse": float(np.mean(v_nmses)),
        "q_nmse": float(np.mean(q_nmses)),
    }
    return metrics


# ── Checkpoint IO ────────────────────────────────────────────────────────────


def save_policy_wm(params, losses, path, config=None):
    """Save {"wm", "v"} params, losses, and config to a pickle file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "params": jax.tree.map(np.array, params),
        "losses": np.array(losses),
    }
    if config is not None:
        from configs.utils import serialize_for_pickle
        data["config"] = serialize_for_pickle(config)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[PiWM] saved checkpoint to {path}")


def load_policy_wm(path):
    """Load (params, losses, config_or_None) from a pickle file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[PiWM] loaded checkpoint from {path}")
    return data["params"], data["losses"], data.get("config")
