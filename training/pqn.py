"""Goal-conditioned PQN trainer for continuous-control envs.

Optionally supports Hindsight Experience Replay (HER) via USE_HER.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState

import wandb

from envs.goals import GoalIndex, goal_indexes_to_goals, sample_goal
from envs.goals import (
    compute_reward as gc_compute_reward,
    goal_achieved as gc_goal_achieved,
    ContinuousGoal,
)
from envs.wrappers import OptimisticResetVecEnvWrapper, BatchEnvWrapper
from flax.linen.initializers import constant, orthogonal


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    reward_e: chex.Array
    reward_gc: chex.Array
    done_ep: chex.Array
    done_ep_or_goal: chex.Array
    done_goal: chex.Array
    next_obs: chex.Array
    next_obs_true: chex.Array  # pre-reset obs for PEB
    q_val: chex.Array
    goal: Any


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


class QNetwork(nn.Module):
    action_dim: int
    num_goals: int

    dense_hidden_size: int = 512
    dense_layers: int = 2

    norm_type: str = "layer_norm"
    sigmoid_output: bool = False
    goal_input_dims: tuple = None  # None = all dims; e.g. (0,) = only first dim of goal
    obs_input_dims: tuple = None  # None = all dims; e.g. (0,1,2,3) = drop obs dims 4+

    @nn.compact
    def __call__(self, obs, goal_repr, train: bool):
        goal_input = goal_repr.target_state
        if self.goal_input_dims is not None:
            goal_input = goal_input[:, jnp.array(self.goal_input_dims)]

        obs_input = obs
        if self.obs_input_dims is not None:
            obs_input = obs_input[:, jnp.array(self.obs_input_dims)]

        embedding = jnp.concatenate([obs_input, goal_input], axis=1)

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        for _ in range(self.dense_layers):
            embedding = nn.Dense(
                self.dense_hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = normalize(embedding)
            embedding = nn.relu(embedding)

        qs = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(embedding)

        if self.sigmoid_output:
            qs = jax.nn.sigmoid(qs)

        return qs


def make_train(config, basic_env, env_params, all_goals, checkpoint_dir=None):
    """Build a JIT-able PQN training fn for a goal-conditioned continuous env.

    Returns a train(rng) -> {runner_state, metrics} callable. The returned fn:
      - rolls out NUM_ENVS parallel ε-greedy trajectories per update
      - optionally applies HER (USE_HER=True): each transition is repeated
        with HER_K resampled goals (strategy in {"random", "future"})
      - runs Q(λ) targets with a single-step or λ-bootstrapped backup
      - saves an intermediate `step_*.pkl` checkpoint to checkpoint_dir
        at each of EVAL_NUMBER eval points (used downstream by the arch
        sweep's WM-tracking phase).

    config is the PQN_CONFIG dict; all_goals is the ContinuousGoal pytree
    built in the caller.
    """
    ALL_GOALS = all_goals
    num_goals = config["NUM_GOALS"]
    use_her = config.get("USE_HER", False)
    her_k = config.get("HER_K", 0) if use_her else 0
    her_strategy = config.get("HER_STRATEGY", "random") if use_her else "random"
    num_her_total = 1 + her_k  # 1 when no HER

    her_info = f", HER_K={her_k} ({her_strategy})" if use_her else ""
    print(f"[PQN] num_goals={num_goals}{her_info}")

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # Decay horizon for ε and LR schedules. Usually = TOTAL_TIMESTEPS; set
    # DECAY_TIMESTEPS smaller to anneal earlier (e.g. for resume).
    config["NUM_UPDATES_DECAY"] = (
        config["DECAY_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["NUM_MINIBATCHES"] = (
        config["NUM_ENVS"]
        * config["NUM_STEPS"]
        * num_her_total
        // config["MINIBATCH_SIZE"]
    )

    print(f"[PQN] num_minibatches={config['NUM_MINIBATCHES']}, "
          f"minibatch_size={config['MINIBATCH_SIZE']}")

    assert (config["NUM_STEPS"] * config["NUM_ENVS"] * num_her_total) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS*num_her_total"

    # Continuous-state goal dispatch (sparse / gaussian rewards on (s, g) pairs).
    def gc_reward(obs, goal_repr):
        return config.get("REW_SCALE", 1.0) * gc_compute_reward(
            obs,
            goal_repr.target_state,
            goal_repr.reward_mask,
            config["REWARD_TYPE"],
            config["REWARD_SIGMA"],
            config["REWARD_A"],
        )

    def gc_achieved(obs, goal_repr):
        return gc_goal_achieved(
            obs,
            goal_repr.target_state,
            goal_repr.reward_mask,
            config["REWARD_A"],
        )

    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            basic_env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = BatchEnvWrapper(basic_env, num_envs=config["NUM_ENVS"])

    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(rng)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosen_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape) < eps,
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),
            greedy_actions,
        )
        return chosen_actions

    def train(rng):
        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        total_grad_steps = (
            config["NUM_UPDATES_DECAY"]
            * config["NUM_MINIBATCHES"]
            * config["NUM_EPOCHS"]
        )
        lr_schedule = config.get("LR_SCHEDULE", "constant")
        lr_end = config.get("LR_END", 1e-4)

        if lr_schedule == "linear":
            lr = optax.linear_schedule(
                init_value=config["LR"], end_value=lr_end,
                transition_steps=total_grad_steps,
            )
        elif lr_schedule == "cosine":
            lr = optax.cosine_decay_schedule(
                init_value=config["LR"], decay_steps=total_grad_steps,
                alpha=lr_end / config["LR"],
            )
        else:
            lr = config["LR"]

        network = QNetwork(
            action_dim=env.action_space(env_params).n,
            dense_hidden_size=config["NETWORK_DENSE_HIDDEN_SIZE"],
            dense_layers=config["NETWORK_DENSE_LAYERS"],
            norm_type=config["NORM_TYPE"],
            num_goals=num_goals,
            sigmoid_output=config.get("NETWORK_SIGMOID_OUTPUTS", False),
            goal_input_dims=tuple(config["GOAL_INPUT_DIMS"]) if config.get("GOAL_INPUT_DIMS") else None,
            obs_input_dims=tuple(config["OBS_INPUT_DIMS"]) if config.get("OBS_INPUT_DIMS") else None,
        )

        rng, _rng = jax.random.split(rng)
        init_x, _ = env.reset(_rng)

        example_goal = jax.tree.map(lambda x: x[0], ALL_GOALS)

        def create_agent(rng):
            rng, _rng = jax.random.split(rng)

            network_variables = network.init(
                _rng,
                init_x,
                jax.tree.map(
                    lambda x: jnp.repeat(
                        x[None, ...], repeats=config["NUM_ENVS"], axis=0
                    ),
                    example_goal,
                ),
                train=False,
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables.get("batch_stats", {}),
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # GREEDY EVAL: one deterministic rollout per goal
        def _eval_greedy(params, batch_stats, rng):
            max_steps = config["MAX_STEPS_IN_EPISODE"]
            terminate_on_goal = config["TERMINATE_ON_GOAL"]
            n_eval = config["EVAL_NUM_ENVS"]

            def _eval_one_episode(goal_repr, rng):
                rng, _rng = jax.random.split(rng)
                obs, env_state = basic_env.reset(_rng, env_params)

                def _step(carry, _):
                    obs, env_state, rng, done = carry
                    rng, _rng = jax.random.split(rng)
                    q_vals = network.apply(
                        {"params": params, "batch_stats": batch_stats},
                        obs[None],
                        jax.tree.map(lambda x: x[None], goal_repr),
                        train=False,
                    )[0]
                    action = jnp.argmax(q_vals)
                    obs_new, env_state_new, _, _, info = basic_env.step(
                        _rng, env_state, action, env_params
                    )
                    obs_true = info.get("obs_before_reset", obs_new)
                    reward = gc_reward(obs_true, goal_repr)
                    reward = jnp.where(done, 0.0, reward)
                    if terminate_on_goal:
                        achieved = gc_achieved(obs_true, goal_repr)
                        done = jnp.logical_or(done, achieved)
                    return (obs_new, env_state_new, rng, done), reward

                rng, _rng = jax.random.split(rng)
                _, rewards = jax.lax.scan(
                    _step,
                    (obs, env_state, _rng, jnp.bool_(False)),
                    None,
                    max_steps,
                )
                # Both discounted (sum_t gamma^t r_t) and undiscounted (sum_t r_t)
                gammas = config["GAMMA"] ** jnp.arange(max_steps)
                return (rewards * gammas).sum(), rewards.sum()

            def _eval_one_goal(goal_idx, rng):
                goal_index = GoalIndex(
                    goal_index=goal_idx,
                    num_goals_completed=jnp.asarray(0),
                )
                goal_repr = goal_indexes_to_goals(ALL_GOALS, goal_index)
                rngs = jax.random.split(rng, n_eval)
                disc, undisc = jax.vmap(_eval_one_episode, in_axes=(None, 0))(
                    goal_repr, rngs
                )
                return disc.mean(), undisc.mean()

            return jax.vmap(_eval_one_goal)(
                jnp.arange(num_goals), jax.random.split(rng, num_goals)
            )

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            (
                train_state,
                expl_state,
                rng,
                goals,
            ) = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, ep_return_acc, ep_length_acc, rng, goal_indexes = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                goal_reprs = goal_indexes_to_goals(ALL_GOALS, goal_indexes)

                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    goal_reprs,
                    train=False,
                )

                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward_e, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
                )

                # Pre-reset obs for PEB (true next state before auto-reset)
                new_obs_true = info.get("obs_before_reset", new_obs)

                # ── Reward & goal termination ──
                reward_gc = jax.vmap(gc_reward)(new_obs_true, goal_reprs)
                if config["TERMINATE_ON_GOAL"]:
                    goals_achieved = jax.vmap(gc_achieved)(new_obs_true, goal_reprs)
                else:
                    goals_achieved = jnp.zeros(config["NUM_ENVS"], dtype=bool)

                reward = reward_gc
                done_ep_or_goal = jnp.logical_or(new_done, goals_achieved)

                # ── Force env reset on goal achievement ──
                # Without this, the env's auto-reset only fires on its physical
                # done (max_steps truncation). With TERMINATE_ON_GOAL=True the
                # goal-achieved bit is otherwise purely a Bellman flag — the cart
                # keeps stepping in post-goal physics drift, biasing the
                # starting-state distribution and corrupting HER's future-state
                # diversity. This block makes the env actually reset on goal.
                if config["TERMINATE_ON_GOAL"]:
                    rng, reset_rng = jax.random.split(rng)
                    obs_re, state_re = env.reset(reset_rng, env_params)

                    def _select(re, st):
                        mask = goals_achieved
                        for _ in range(re.ndim - 1):
                            mask = mask[..., None]
                        return jnp.where(mask, re, st)

                    new_obs = _select(obs_re, new_obs)
                    new_env_state = jax.tree.map(_select, state_re, new_env_state)

                # ── Episode-level accumulator ──
                # Tracks per-env cumulative reward_gc and step count between
                # consecutive done_ep_or_goal events. Snapshots at this step
                # become the per-episode (return, length); accumulators reset
                # for the next episode. Carry persists across rollouts so
                # episodes that span NUM_STEPS boundaries are counted exactly.
                new_ep_return_acc = ep_return_acc + reward_gc
                new_ep_length_acc = ep_length_acc + 1
                ep_return_snapshot = new_ep_return_acc
                ep_length_snapshot = new_ep_length_acc
                new_ep_return_acc = jnp.where(done_ep_or_goal, 0.0, new_ep_return_acc)
                new_ep_length_acc = jnp.where(done_ep_or_goal, 0, new_ep_length_acc)

                transition = Transition(
                    obs=last_obs,
                    action=new_action,
                    reward=reward,
                    reward_e=reward_e,
                    reward_gc=reward_gc,
                    done_ep=new_done,
                    done_goal=goals_achieved,
                    done_ep_or_goal=done_ep_or_goal,
                    next_obs=new_obs,
                    next_obs_true=new_obs_true,
                    q_val=q_vals,
                    goal=goal_indexes,
                )

                # Sample new goals for completed goals / terminated episodes
                rng, _rng = jax.random.split(rng)
                _rngs = jax.random.split(_rng, config["NUM_ENVS"])
                new_goal_indexes = jax.vmap(sample_goal, in_axes=(0, None))(
                    _rngs, num_goals
                )

                new_goals_completed = jax.tree.map(
                    lambda x, y: jax.vmap(jnp.where)(goals_achieved, x, y),
                    goal_indexes.num_goals_completed + 1,
                    goal_indexes.num_goals_completed,
                )
                new_goals_completed = jax.tree.map(
                    lambda x, y: jax.vmap(jnp.where)(new_done, x, y),
                    jnp.zeros_like(new_goals_completed),
                    new_goals_completed,
                )

                goals = jax.tree.map(
                    lambda x, y: jax.vmap(jnp.where)(done_ep_or_goal, x, y),
                    new_goal_indexes,
                    goal_indexes,
                )
                goals = goals.replace(num_goals_completed=new_goals_completed)

                return (new_obs, new_env_state, new_ep_return_acc, new_ep_length_acc, rng, goals), (
                    transition,
                    info,
                    (ep_return_snapshot, ep_length_snapshot),
                )

            rng, _rng = jax.random.split(rng)
            (*expl_state, rng, goals), (transitions, infos, ep_snapshots) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng, goals),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)
            ep_return_snapshots, ep_length_snapshots = ep_snapshots

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"],
                n_updates=train_state.n_updates + 1,
            )

            gamma = config["GAMMA"]

            # ── HER RELABELING (conditional) ──
            if use_her:
                rng, _rng = jax.random.split(rng)
                num_steps = config["NUM_STEPS"]
                num_envs = config["NUM_ENVS"]

                if her_strategy == "random":
                    her_rngs = jax.random.split(
                        _rng,
                        num_steps * num_envs * her_k,
                    )
                    her_goal_indexes = jax.vmap(sample_goal, in_axes=(0, None))(
                        her_rngs, num_goals
                    )
                    her_goal_indexes = jax.tree.map(
                        lambda x: x.reshape(num_steps, num_envs, her_k, *x.shape[1:]),
                        her_goal_indexes,
                    )
                    her_goal_reprs = goal_indexes_to_goals(ALL_GOALS, her_goal_indexes)

                else:  # "future"
                    # For step t, sample t' ~ Uniform{t, ..., NUM_STEPS-1}
                    t_current = jnp.arange(num_steps)[:, None, None]
                    remaining = num_steps - t_current
                    random_offsets = jax.random.randint(
                        _rng,
                        shape=(num_steps, num_envs, her_k),
                        minval=0,
                        maxval=num_steps,
                    )
                    t_future = t_current + (random_offsets % remaining)

                    env_idx = jnp.broadcast_to(
                        jnp.arange(num_envs)[None, :, None],
                        (num_steps, num_envs, her_k),
                    )
                    future_obs = transitions.next_obs_true[t_future, env_idx]

                    from envs.goals import ContinuousGoal

                    default_mask = ALL_GOALS.reward_mask[0]
                    her_goal_reprs = ContinuousGoal(
                        target_state=future_obs,
                        reward_mask=jnp.broadcast_to(
                            default_mask, future_obs.shape
                        ),
                    )

                def _relabel_for_one_goal(transition_slice, her_goal_repr):
                    """Relabel a single transition for a single HER goal."""
                    rkey = transition_slice.next_obs_true
                    new_reward = gc_reward(rkey, her_goal_repr)
                    if config["TERMINATE_ON_GOAL"]:
                        new_done_goal = gc_achieved(rkey, her_goal_repr)
                    else:
                        new_done_goal = jnp.bool_(False)
                    new_done_ep_or_goal = jnp.logical_or(
                        transition_slice.done_ep, new_done_goal
                    )
                    new_q_val = network.apply(
                        {
                            "params": train_state.params,
                            "batch_stats": train_state.batch_stats,
                        },
                        transition_slice.obs[None],
                        jax.tree.map(lambda x: x[None], her_goal_repr),
                        train=False,
                    )[0]
                    return transition_slice.replace(
                        reward=new_reward,
                        reward_gc=new_reward,
                        done_goal=new_done_goal,
                        done_ep_or_goal=new_done_ep_or_goal,
                        q_val=new_q_val,
                        goal=her_goal_repr,
                    )

                # Vectorize over (step, env, her_k)
                _relabel_all = jax.vmap(
                    jax.vmap(
                        jax.vmap(_relabel_for_one_goal, in_axes=(None, 0)),
                        in_axes=(0, 0),
                    ),
                    in_axes=(0, 0),
                )
                her_transitions = _relabel_all(transitions, her_goal_reprs)

                def _concat_her(orig, her):
                    """Concat original (step, env, ...) with HER (step, env, her_k, ...)."""
                    orig_expanded = (
                        orig[..., None] if orig.ndim == 2 else orig[:, :, None, ...]
                    )
                    return jnp.concatenate([orig_expanded, her], axis=2)

                def _concat_her_pytree(orig_pytree, her_pytree):
                    """Concat goal repr pytrees along HER axis (axis=2)."""
                    return jax.tree.map(
                        lambda o, h: _concat_her(o, h),
                        orig_pytree,
                        her_pytree,
                    )

                # Convert original goals (GoalIndex) to reprs, then concat
                orig_goal_reprs = goal_indexes_to_goals(ALL_GOALS, transitions.goal)
                combined_goal_reprs = _concat_her_pytree(
                    orig_goal_reprs, her_transitions.goal
                )

                combined = Transition(
                    obs=transitions.obs,
                    action=transitions.action,
                    reward=_concat_her(transitions.reward, her_transitions.reward),
                    reward_e=transitions.reward_e,
                    reward_gc=_concat_her(
                        transitions.reward_gc, her_transitions.reward_gc
                    ),
                    done_ep=transitions.done_ep,
                    done_goal=_concat_her(
                        transitions.done_goal, her_transitions.done_goal
                    ),
                    done_ep_or_goal=_concat_her(
                        transitions.done_ep_or_goal,
                        her_transitions.done_ep_or_goal,
                    ),
                    next_obs=transitions.next_obs,
                    next_obs_true=transitions.next_obs_true,
                    q_val=_concat_her(transitions.q_val, her_transitions.q_val),
                    goal=combined_goal_reprs,
                )

            # ── NETWORK UPDATE ──
            if use_her:

                def _learn_epoch(carry, _):
                    train_state, rng = carry

                    def _learn_phase(carry, minibatch_indexes):
                        train_state, rng = carry

                        # minibatch_indexes: (MB_SIZE, 3) — [step, env, her_variant]
                        def _her_index(t):
                            return jax.tree.map(
                                lambda x: x[
                                    minibatch_indexes[:, 0],
                                    minibatch_indexes[:, 1],
                                    minibatch_indexes[:, 2],
                                ],
                                t,
                            )

                        def _broadcast_index(t):
                            return jax.tree.map(
                                lambda x: x[
                                    minibatch_indexes[:, 0],
                                    minibatch_indexes[:, 1],
                                ],
                                t,
                            )

                        mb_obs = _broadcast_index(combined.obs)
                        mb_action = _broadcast_index(combined.action)
                        mb_next_obs_true = _broadcast_index(combined.next_obs_true)
                        mb_reward = _her_index(combined.reward)
                        mb_done_goal = _her_index(combined.done_goal)
                        mb_goal_reprs = _her_index(combined.goal)

                        def _loss_fn(params):
                            # Single concatenated forward pass: Q(obs) and
                            # Q(next_obs_true) share batch stats in BatchRenorm.
                            all_obs = jnp.concatenate(
                                [mb_obs, mb_next_obs_true], axis=0
                            )
                            all_goal_reprs = jax.tree.map(
                                lambda x: jnp.concatenate([x, x]),
                                mb_goal_reprs,
                            )
                            all_q_vals, updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.batch_stats,
                                },
                                all_obs,
                                all_goal_reprs,
                                train=True,
                                mutable=["batch_stats"],
                            )
                            q_vals, q_next = jnp.split(all_q_vals, 2)
                            q_next = jax.lax.stop_gradient(q_next)
                            q_next = jnp.max(q_next, axis=-1)
                            target = mb_reward + (1 - mb_done_goal) * gamma * q_next
                            chosen_action_qvals = jnp.take_along_axis(
                                q_vals,
                                jnp.expand_dims(mb_action, axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)
                            loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()
                            return loss, (updates, chosen_action_qvals)

                        (loss, (updates, qvals)), grads = jax.value_and_grad(
                            _loss_fn, has_aux=True
                        )(train_state.params)
                        train_state = train_state.apply_gradients(grads=grads)
                        train_state = train_state.replace(
                            grad_steps=train_state.grad_steps + 1,
                            batch_stats=updates["batch_stats"],
                        )
                        return (train_state, rng), (loss, qvals)

                    # Build minibatch indices: (step, env, her_variant) triples
                    step_indexes = jnp.repeat(
                        jnp.repeat(
                            jnp.arange(config["NUM_STEPS"])[:, None, None],
                            repeats=config["NUM_ENVS"],
                            axis=1,
                        ),
                        repeats=num_her_total,
                        axis=2,
                    )
                    env_indexes = jnp.repeat(
                        jnp.repeat(
                            jnp.arange(config["NUM_ENVS"])[None, :, None],
                            repeats=config["NUM_STEPS"],
                            axis=0,
                        ),
                        repeats=num_her_total,
                        axis=2,
                    )
                    her_indexes = jnp.repeat(
                        jnp.repeat(
                            jnp.arange(num_her_total)[None, None, :],
                            repeats=config["NUM_ENVS"],
                            axis=1,
                        ),
                        repeats=config["NUM_STEPS"],
                        axis=0,
                    )
                    mb_indexes = jnp.concatenate(
                        [
                            step_indexes.flatten()[:, None],
                            env_indexes.flatten()[:, None],
                            her_indexes.flatten()[:, None],
                        ],
                        axis=1,
                    )

                    rng, _rng = jax.random.split(rng)
                    mb_indexes = jax.random.permutation(_rng, mb_indexes)
                    mb_indexes = jnp.reshape(
                        mb_indexes,
                        (config["NUM_MINIBATCHES"], config["MINIBATCH_SIZE"], 3),
                    )

                    rng, _rng = jax.random.split(rng)
                    (train_state, rng), (loss, qvals) = jax.lax.scan(
                        _learn_phase, (train_state, rng), mb_indexes
                    )

                    return (train_state, rng), (loss, qvals)

            else:

                def _learn_epoch(carry, _):
                    train_state, rng = carry

                    def _learn_phase(carry, minibatch_indexes):
                        train_state, rng = carry

                        def _index(t):
                            return jax.tree.map(
                                lambda x: x[
                                    minibatch_indexes[:, 0],
                                    minibatch_indexes[:, 1],
                                ],
                                t,
                            )

                        minibatch = Transition(
                            action=_index(transitions.action),
                            done_ep=_index(transitions.done_ep),
                            done_ep_or_goal=_index(transitions.done_ep_or_goal),
                            done_goal=_index(transitions.done_goal),
                            goal=_index(transitions.goal),
                            next_obs=_index(transitions.next_obs),
                            next_obs_true=_index(transitions.next_obs_true),
                            obs=_index(transitions.obs),
                            q_val=_index(transitions.q_val),
                            reward=_index(transitions.reward),
                            reward_e=_index(transitions.reward_e),
                            reward_gc=_index(transitions.reward_gc),
                        )

                        def _loss_fn(params):
                            mb_goal_reprs = goal_indexes_to_goals(
                                ALL_GOALS, minibatch.goal
                            )
                            # 1-step TD with PEB: concatenated forward pass so
                            # Q(obs) and Q(next_obs_true) share batch stats.
                            all_obs = jnp.concatenate(
                                (minibatch.obs, minibatch.next_obs_true), axis=0
                            )
                            all_goal_reprs = jax.tree.map(
                                lambda x: jnp.concatenate((x, x)),
                                mb_goal_reprs,
                            )
                            all_q_vals, updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.batch_stats,
                                },
                                all_obs,
                                all_goal_reprs,
                                train=True,
                                mutable=["batch_stats"],
                            )
                            q_vals, q_next = jnp.split(all_q_vals, 2)
                            q_next = jax.lax.stop_gradient(q_next)
                            q_next = jnp.max(q_next, axis=-1)
                            target = (
                                minibatch.reward
                                + (1 - minibatch.done_goal) * gamma * q_next
                            )
                            chosen_action_qvals = jnp.take_along_axis(
                                q_vals,
                                jnp.expand_dims(minibatch.action, axis=-1),
                                axis=-1,
                            ).squeeze(axis=-1)
                            loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()
                            return loss, (updates, chosen_action_qvals)

                        (loss, (updates, qvals)), grads = jax.value_and_grad(
                            _loss_fn, has_aux=True
                        )(train_state.params)
                        train_state = train_state.apply_gradients(grads=grads)
                        train_state = train_state.replace(
                            grad_steps=train_state.grad_steps + 1,
                            batch_stats=updates["batch_stats"],
                        )
                        return (train_state, rng), (loss, qvals)

                    # Build minibatch indices: (step, env) pairs
                    step_indexes = jnp.repeat(
                        jnp.arange(config["NUM_STEPS"])[:, None],
                        repeats=config["NUM_ENVS"],
                        axis=1,
                    )
                    env_indexes = jnp.repeat(
                        jnp.arange(config["NUM_ENVS"])[None, :],
                        repeats=config["NUM_STEPS"],
                        axis=0,
                    )
                    mb_indexes = jnp.stack(
                        [step_indexes.flatten(), env_indexes.flatten()], axis=1
                    )

                    rng, _rng = jax.random.split(rng)
                    mb_indexes = jax.random.permutation(_rng, mb_indexes)
                    mb_indexes = jnp.reshape(
                        mb_indexes,
                        (config["NUM_MINIBATCHES"], config["MINIBATCH_SIZE"], 2),
                    )

                    rng, _rng = jax.random.split(rng)
                    (train_state, rng), (loss, qvals) = jax.lax.scan(
                        _learn_phase, (train_state, rng), mb_indexes
                    )

                    return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            eval_number = config.get("EVAL_NUMBER", 20)
            eval_interval = jnp.maximum(config["NUM_UPDATES"] // eval_number, 1)
            should_eval = ((config["NUM_UPDATES"] - train_state.n_updates) % eval_interval) == 0
            rng, _rng = jax.random.split(rng)
            eval_returns_disc, eval_returns_undisc = jax.lax.cond(
                should_eval,
                lambda r: _eval_greedy(train_state.params, train_state.batch_stats, r),
                lambda r: (jnp.full(num_goals, jnp.nan), jnp.full(num_goals, jnp.nan)),
                _rng,
            )

            # Save checkpoint at eval points
            if checkpoint_dir is not None:
                import pickle as _pkl

                def _save_ckpt(params, batch_stats, n_updates):
                    path = f"{checkpoint_dir}/step_{int(n_updates):06d}.pkl"
                    with open(path, "wb") as _f:
                        _pkl.dump({
                            "params": jax.tree.map(np.array, params),
                            "batch_stats": jax.tree.map(np.array, batch_stats),
                            "n_updates": int(n_updates),
                        }, _f)

                jax.lax.cond(
                    should_eval,
                    lambda: jax.debug.callback(
                        _save_ckpt,
                        train_state.params,
                        train_state.batch_stats,
                        train_state.n_updates,
                    ),
                    lambda: None,
                )

            # ── Exploration episode metrics ──
            # Computed from the transition stream so they reflect the right
            # episode boundary (done_ep_or_goal) and the right reward
            # (reward_gc). Snapshots are captured per-step inside _step_env;
            # mask by done_ep_or_goal to keep only end-of-episode values.
            n_ep = jnp.maximum(transitions.done_ep_or_goal.sum(), 1)
            expl_episode_return_mean = (
                (ep_return_snapshots * transitions.done_ep_or_goal).sum() / n_ep
            )
            expl_episode_length_mean = (
                (ep_length_snapshots * transitions.done_ep_or_goal).sum() / n_ep
            )

            # Per-goal version. NaN where a goal had no completed episodes in
            # this rollout, so plotting can skip those points.
            goal_oh = jax.nn.one_hot(transitions.goal.goal_index, num_goals)  # (T, E, G)
            n_ep_per_goal = (transitions.done_ep_or_goal[..., None] * goal_oh).sum(axis=(0, 1))
            return_sums_per_goal = (
                (ep_return_snapshots * transitions.done_ep_or_goal)[..., None] * goal_oh
            ).sum(axis=(0, 1))
            expl_episode_return_mean_per_goal = jnp.where(
                n_ep_per_goal > 0,
                return_sums_per_goal / jnp.maximum(n_ep_per_goal, 1),
                jnp.nan,
            )

            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "td_loss": loss.mean(),
                "qvals": qvals.mean(),
                "eval_returns_per_goal": eval_returns_disc,
                "eval_returns_undiscounted_per_goal": eval_returns_undisc,
                "train/episode_return_mean": expl_episode_return_mean,
                "train/episode_length_mean": expl_episode_length_mean,
                "train/episode_return_mean_per_goal": expl_episode_return_mean_per_goal,
            }

            if config["USE_WANDB"]:
                def callback(metrics):
                    log_dict = {}
                    # High-frequency metrics (WANDB_LOG_INTERVAL)
                    if metrics["update_steps"] % config["WANDB_LOG_INTERVAL"] == 0:
                        log_dict.update({
                            "env_step": metrics["env_step"],
                            "td_loss": metrics["td_loss"],
                            "qvals": metrics["qvals"],
                            "train/episode_return_mean": metrics["train/episode_return_mean"],
                            "train/episode_length_mean": metrics["train/episode_length_mean"],
                        })
                        train_per_goal_ret = metrics["train/episode_return_mean_per_goal"]
                        for g in range(num_goals):
                            log_dict[f"train/episode_return_mean/goal_{g}"] = train_per_goal_ret[g]
                    # Eval-point metrics: logged whenever available (sparse, NaN elsewhere)
                    eval_rets = metrics["eval_returns_per_goal"]
                    if not np.isnan(float(eval_rets[0])):
                        for g in range(num_goals):
                            log_dict[f"eval_returns/goal_{g}"] = eval_rets[g]
                        log_dict["eval_returns/mean"] = jnp.nanmean(eval_rets)
                    eval_rets_undisc = metrics["eval_returns_undiscounted_per_goal"]
                    if not np.isnan(float(eval_rets_undisc[0])):
                        for g in range(num_goals):
                            log_dict[f"eval_returns_undiscounted/goal_{g}"] = eval_rets_undisc[g]
                        log_dict["eval_returns_undiscounted/mean"] = jnp.nanmean(eval_rets_undisc)
                    if log_dict:
                        wandb.log(log_dict, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics)

            runner_state = (
                train_state,
                tuple(expl_state),
                rng,
                goals,
            )

            return runner_state, metrics

        # ── Initialise runner state ──
        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)
        # Persistent episode-level accumulators (carry across rollout boundaries)
        ep_return_acc = jnp.zeros(config["NUM_ENVS"], dtype=obs.dtype)
        ep_length_acc = jnp.zeros(config["NUM_ENVS"], dtype=jnp.int32)
        expl_state = (obs, env_state, ep_return_acc, ep_length_acc)

        rng, _rng = jax.random.split(rng)
        _rngs = jax.random.split(_rng, config["NUM_ENVS"])
        goal_indexes = jax.vmap(sample_goal, in_axes=(0, None))(_rngs, num_goals)

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            expl_state,
            _rng,
            goal_indexes,
        )

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train

