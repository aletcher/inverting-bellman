"""World model evaluation: compare learned dynamics P with true dynamics."""

import os

import jax
import jax.numpy as jnp

from training.wm import make_world_model, apply_wm
from plotting.wm import plot_dynamics_quiver


def predicted_dynamics(p_params, s, a, wm_config, action_dim, state_dim):
    p_model = make_world_model(wm_config, state_dim)
    a_oh = jax.nn.one_hot(a, action_dim)
    return apply_wm(p_model, p_params, s, a_oh,
                    residual=wm_config.get("RESIDUAL_PREDICTION", True),
                    angle_dims=wm_config.get("ANGLE_DIMS"),
                    wm_input_dims=wm_config.get("WM_INPUT_DIMS"))


def make_wm_dynamics_fn(p_params, wm_config, action_dim, state_dim):
    """Build a dynamics_fn(s, a) -> s' backed by the WM."""
    p_model = make_world_model(wm_config, state_dim)
    residual = wm_config.get("RESIDUAL_PREDICTION", True)
    angle_dims = wm_config.get("ANGLE_DIMS")
    wm_input_dims = wm_config.get("WM_INPUT_DIMS")
    def wm_dyn(s, a):
        a_oh = jax.nn.one_hot(a, action_dim)
        return apply_wm(p_model, p_params, s, a_oh, residual=residual,
                        angle_dims=angle_dims, wm_input_dims=wm_input_dims)
    return wm_dyn


def _resolve_slice_point(env_config, state_dim, state_ranges, state_labels):
    """Resolve EVAL_SLICE_POINT (dict, list, or None) to a full (state_dim,) array.

    - None / missing → midpoint of state_ranges.
    - dict (e.g. {"x_dot": 0.0}) → midpoint with named entries overridden.
    - list/array of length state_dim → used as-is.
    """
    midpoints = jnp.array([(lo + hi) / 2 for lo, hi in state_ranges])
    spec = env_config.get("EVAL_SLICE_POINT")
    if spec is None:
        return midpoints
    if isinstance(spec, dict):
        out = midpoints
        for k, v in spec.items():
            if k not in state_labels:
                raise ValueError(
                    f"EVAL_SLICE_POINT key {k!r} not in STATE_LABELS={state_labels}"
                )
            d = state_labels.index(k)
            out = out.at[d].set(float(v))
        return out
    arr = jnp.asarray(spec, dtype=midpoints.dtype)
    if arr.shape != (state_dim,):
        raise ValueError(
            f"EVAL_SLICE_POINT length {arr.shape} != state_dim {state_dim}"
        )
    return arr


def _slice_label(slice_point, state_labels, exclude_dims=()):
    """Format a slice point as a short string, optionally excluding swept dims."""
    parts = []
    for d, lbl in enumerate(state_labels):
        if d in exclude_dims:
            continue
        parts.append(f"{lbl}={float(slice_point[d]):.3g}")
    return ", ".join(parts)


def compare_dynamics(p_params, out_dir, wm_config, env_config, dynamics_fn,
                     goals=None, goal_masks=None, losses=None,
                     plots=True,
                     sample_states_fn=None,
                     state_to_eff_fn=None, eff_to_obs_fn=None,
                     wm_output_dim=None):
    """WM-vs-true dynamics comparison: per-(action, dim) MSE/NMSE + quiver plot.

    Returns the metrics dict; writes results.txt and dynamics_quiver.png.

    Computes WM_MSE on the WM's *training support*:
      - sample_states_fn provided (Reacher SAMPLE_FROM_RESET=True): evaluate on
        samples from it — same env.reset() distribution as training.
      - Otherwise (MountainCar): uniform sample over STATE_RANGES, matching training.

    dynamics_fn: callable (s, a) -> s_next, batched over leading dim.
    env_config: STATE_DIM, ACTION_DIM, STATE_LABELS, STATE_RANGES, ACTION_NAMES,
                EVAL_DIMS_2D (list of (d0,d1) pairs, default [(0,1)]).
                Optional: VI_STATE_RANGES, EVAL_SLICE_POINT.
    state_to_eff_fn, eff_to_obs_fn, wm_output_dim: when the WM was trained with
                a smaller effective-state output (e.g. Reacher 4D [θ, ω] output,
                fp_xy lifted via FK), pass these so eval/rollout chaining lifts
                back to obs space — see world_model_loss_sampled.
    """
    state_dim    = env_config["STATE_DIM"]
    action_dim   = env_config["ACTION_DIM"]
    state_labels = env_config["STATE_LABELS"]
    state_ranges = env_config["STATE_RANGES"]
    action_names = env_config["ACTION_NAMES"]
    eval_dims_2d = env_config.get("EVAL_DIMS_2D")
    if eval_dims_2d is None:
        eval_dims_2d = [(0, 1)]
    # Sweeps restricted to the env's valid (non-terminal) region when set —
    # WM error in the terminal halo isn't meaningful. Falls back to STATE_RANGES
    # if VI_STATE_RANGES is unset (e.g. MountainCar).
    eval_state_ranges = env_config.get("VI_STATE_RANGES", state_ranges)

    slice_point = _resolve_slice_point(env_config, state_dim, state_ranges, state_labels)
    n_heat = wm_config["EVAL_HEATMAP_RES"]
    residual = wm_config.get("RESIDUAL_PREDICTION", True)
    angle_dims = wm_config.get("ANGLE_DIMS")
    angle_dims_set = set(angle_dims) if angle_dims else set()
    wm_input_dims = wm_config.get("WM_INPUT_DIMS")
    out_dim = wm_output_dim if wm_output_dim is not None else state_dim
    p_model = make_world_model(wm_config, out_dim)
    # When WM output is in effective space, lift to obs (e.g. fp via FK) so the
    # reported s_pred matches the true-dynamics output.
    effective_output = eff_to_obs_fn is not None
    _wm_state_to_eff_fn = state_to_eff_fn if effective_output else None

    from training.wm import _wrap_angle

    def _per_dim_err(s_pred_col, s_true_col, d):
        """Absolute error, using angular distance for wrapped-angle dims."""
        diff = s_pred_col - s_true_col
        if d in angle_dims_set:
            diff = _wrap_angle(diff)
        return jnp.abs(diff)

    # Eval batch: WM's training-support sampler, or uniform on STATE_RANGES.
    n_eval = n_heat * n_heat
    eval_rng = jax.random.PRNGKey(42)
    if sample_states_fn is not None:
        # Reacher-style: WM trained on env.reset() obs.
        eval_states = sample_states_fn(eval_rng, n_eval)
    else:
        # MountainCar: WM trained uniformly on STATE_RANGES.
        mins = jnp.array([lo for lo, _ in state_ranges])
        maxs = jnp.array([hi for _, hi in state_ranges])
        eval_states = jax.random.uniform(
            eval_rng, (n_eval, state_dim), minval=mins, maxval=maxs,
        )

    dynamics_metrics = {}
    print(f"\n── Dynamics metrics ({n_eval} states) ──")
    for a in range(action_dim):
        actions = jnp.full(n_eval, a)
        s_true = dynamics_fn(eval_states, actions)
        a_oh = jax.nn.one_hot(actions, action_dim)
        s_pred = apply_wm(p_model, p_params, eval_states, a_oh, residual=residual,
                          state_to_eff_fn=_wm_state_to_eff_fn,
                          angle_dims=angle_dims, eff_to_obs_fn=eff_to_obs_fn,
                          wm_input_dims=wm_input_dims)
        parts = []
        for d in range(state_dim):
            abs_err = _per_dim_err(s_pred[:, d], s_true[:, d], d)
            mse_d = float(jnp.mean(abs_err ** 2))
            var_d = float(jnp.var(s_true[:, d]))
            nmse_d = (mse_d / var_d) if var_d > 1e-12 else 0.0
            dynamics_metrics[f"{action_names[a]}_{state_labels[d]}"] = {
                "mse": mse_d, "nmse": nmse_d,
            }
            parts.append(f"{state_labels[d]}: MSE={mse_d:.1e} NMSE={nmse_d:.1e}")
        print(f"  {action_names[a]:>5s}: {', '.join(parts)}")
    avg_mse = sum(m["mse"] for m in dynamics_metrics.values()) / len(dynamics_metrics)
    avg_nmse = sum(m["nmse"] for m in dynamics_metrics.values()) / len(dynamics_metrics)

    # 2D sweep grid for the dynamics-quiver plot. Per-action true/pred
    # snapshots on a uniform grid in (d0, d1), off-axis dims pinned to
    # slice_point. Skipped when sample_states_fn is set (sweep states would
    # violate coupling constraints, e.g. Reacher fp = FK).
    pair_data = []
    if plots and sample_states_fn is None:
        for d0, d1 in eval_dims_2d:
            pair_label = f"{state_labels[d0]}_{state_labels[d1]}"
            pair_dir = out_dir if len(eval_dims_2d) == 1 else f"{out_dir}/pair_{pair_label}"
            if pair_dir != out_dir:
                os.makedirs(pair_dir, exist_ok=True)

            g0 = jnp.linspace(*eval_state_ranges[d0], n_heat)
            g1 = jnp.linspace(*eval_state_ranges[d1], n_heat)
            G0, G1 = jnp.meshgrid(g0, g1)
            base = jnp.broadcast_to(slice_point, (n_heat * n_heat, state_dim)).copy()
            base = base.at[:, d0].set(G0.flatten())
            base = base.at[:, d1].set(G1.flatten())

            s_true_by_action, s_pred_by_action = [], []
            for a in range(action_dim):
                actions = jnp.full(n_heat * n_heat, a)
                s_true = dynamics_fn(base, actions)
                a_oh = jax.nn.one_hot(actions, action_dim)
                s_pred = apply_wm(p_model, p_params, base, a_oh, residual=residual,
                                  state_to_eff_fn=_wm_state_to_eff_fn,
                                  angle_dims=angle_dims, eff_to_obs_fn=eff_to_obs_fn,
                                  wm_input_dims=wm_input_dims)
                s_true_by_action.append(s_true)
                s_pred_by_action.append(s_pred)

            pair_data.append(dict(
                d0=d0, d1=d1, pair_dir=pair_dir, base=base, G0=G0, G1=G1,
                s_true_by_action=s_true_by_action, s_pred_by_action=s_pred_by_action,
            ))

    print(f"\nAvg: MSE={avg_mse:.1e} NMSE={avg_nmse:.1e}")

    with open(f"{out_dir}/results.txt", "w") as f:
        if goals is not None and goal_masks is not None:
            f.write("Goals (only the dims with reward_mask=1):\n")
            for i, (goal, mask) in enumerate(zip(goals, goal_masks)):
                active = [
                    f"{state_labels[d]}={float(goal[d]):.2f}"
                    for d in range(len(state_labels))
                    if float(mask[d]) > 0.0
                ]
                f.write(f"  {i}: {', '.join(active) if active else '(no masked dims)'}\n")
            f.write("\n")
        f.write(f"Eval batch size: {n_eval}\n")
        f.write("\n")
        if losses is not None:
            f.write(f"WM training loss: {float(losses[0]):.6e} -> {float(losses[-1]):.6e}\n\n")
        f.write(f"{'action_dim':<20} {'MSE':>12} {'NMSE':>10}\n")
        for key, m in dynamics_metrics.items():
            f.write(f"{key:<20} {m['mse']:>12.1e} {m['nmse']:>10.1e}\n")
        f.write(f"\nAvg MSE: {avg_mse:.1e}\n")
        f.write(f"Avg NMSE: {avg_nmse:.1e}\n")
    print(f"Saved {out_dir}/results.txt")

    if not plots:
        return dynamics_metrics

    # 2D quiver plot for each dim pair (sweep-grid path only).
    for pd in pair_data:
        d0, d1, pair_dir = pd["d0"], pd["d1"], pd["pair_dir"]
        base, G0, G1 = pd["base"], pd["G0"], pd["G1"]
        s_true_by_action = pd["s_true_by_action"]
        s_pred_by_action = pd["s_pred_by_action"]

        grid_shape = (n_heat, n_heat)
        # Off-axis label: which dims are pinned and to what value, for plot titles.
        pair_slice_label = _slice_label(slice_point, state_labels, exclude_dims=(d0, d1))
        common = dict(
            base_states=base,
            s_true_by_action=s_true_by_action,
            s_pred_by_action=s_pred_by_action,
            grid_shape=grid_shape,
            G0=G0, G1=G1,
            state_labels=state_labels,
            action_names=action_names,
            d0=d0, d1=d1,
        )

        plot_dynamics_quiver(
            **common, slice_label=pair_slice_label,
            state_ranges=eval_state_ranges,
            save_path=f"{pair_dir}/dynamics_quiver.png",
        )

    return dynamics_metrics


def _build_trajectory_starts(
    slice_point, dim_pair, state_dim, state_ranges,
    n_x=2, n_y=2, trajectory_starts=None, state_labels=None,
    sample_states_fn=None,
):
    """Build n_y × n_x starting states on a slice in the (d0, d1) plane.

    If sample_states_fn is provided, sample n_y * n_x valid states instead of
    building a grid (for envs with coupling constraints between dims).

    If trajectory_starts (dict label→values) is provided and contains keys for
    both swept dims, use those values. Otherwise fall back to linspace over the
    padded valid range. Starts are ordered row-major by d1.

    Returns (starts, n_y, n_x) where starts shape (n_y*n_x, state_dim).
    """
    d0, d1 = dim_pair

    if sample_states_fn is not None:
        n_starts = n_y * n_x
        starts = sample_states_fn(jax.random.PRNGKey(0), n_starts)
        return starts, n_y, n_x

    # Resolve d0 (x) and d1 (y) grid values.
    if trajectory_starts and state_labels:
        lbl0, lbl1 = state_labels[d0], state_labels[d1]
        if lbl0 in trajectory_starts:
            g0 = jnp.array(trajectory_starts[lbl0])
            n_x = len(g0)
        else:
            lo, hi = state_ranges[d0]
            pad = (hi - lo) * 0.1
            g0 = jnp.linspace(lo + pad, hi - pad, n_x)
        if lbl1 in trajectory_starts:
            g1 = jnp.array(trajectory_starts[lbl1])
            n_y = len(g1)
        else:
            lo, hi = state_ranges[d1]
            pad = (hi - lo) * 0.1
            g1 = jnp.linspace(lo + pad, hi - pad, n_y)
    else:
        lo0, hi0 = state_ranges[d0]
        lo1, hi1 = state_ranges[d1]
        g0 = jnp.linspace(lo0 + (hi0 - lo0) * 0.1, hi0 - (hi0 - lo0) * 0.1, n_x)
        g1 = jnp.linspace(lo1 + (hi1 - lo1) * 0.1, hi1 - (hi1 - lo1) * 0.1, n_y)

    G1_grid, G0_grid = jnp.meshgrid(g1, g0, indexing="ij")
    n_starts = n_y * n_x
    starts = jnp.broadcast_to(slice_point, (n_starts, state_dim)).copy()
    starts = starts.at[:, d0].set(G0_grid.flatten())
    starts = starts.at[:, d1].set(G1_grid.flatten())
    return starts, n_y, n_x


def compute_policy_trajectory_rollouts(
    p_params, p_model, q_params, q_batch_stats, pqn_config,
    state_dim, action_dim, state_ranges, dynamics_fn,
    slice_point, dim_pair, goals, goal_masks,
    n_steps=100, n_x=2, n_y=2,
    trajectory_starts=None, state_labels=None, wm_config=None,
    sample_states_fn=None,
    state_to_eff_fn=None, eff_to_obs_fn=None,
    goal_indices_override=None,
):
    """Policy-conditioned trajectory rollouts on a slice (one per goal).

    For each goal, the greedy PQN policy picks actions. True traj uses true
    dynamics with policy(s_true, g); predicted traj uses WM dynamics with
    policy(s_pred, g) — showing realistic compounding error.

    Goal selection:
      - goal_indices_override (list/tuple of ints): use those exact indices.
      - Otherwise: the pair with smallest and largest value on the first masked
        dim (most visually distinct).

    Returns (per_goal_true, per_goal_pred, starts, n_y, n_x, goal_indices).
    """
    from training.pqn_utils import make_q_network, make_goal_repr

    starts, n_y, n_x = _build_trajectory_starts(
        slice_point, dim_pair, state_dim, state_ranges,
        n_x=n_x, n_y=n_y,
        trajectory_starts=trajectory_starts, state_labels=state_labels,
        sample_states_fn=sample_states_fn,
    )
    n_starts = n_y * n_x
    _residual = wm_config.get("RESIDUAL_PREDICTION", True) if wm_config else True
    _angle_dims = wm_config.get("ANGLE_DIMS") if wm_config else None
    _wm_input_dims = wm_config.get("WM_INPUT_DIMS") if wm_config else None
    _state_to_eff_fn = state_to_eff_fn if eff_to_obs_fn is not None else None

    num_goals = goals.shape[0]
    if goal_indices_override is not None:
        # Caller-specified indices (e.g. fp-opposite pair for Reacher).
        goal_indices = jnp.array(sorted({int(i) % num_goals for i in goal_indices_override}))
    elif num_goals <= 2:
        goal_indices = jnp.arange(num_goals)
    else:
        # Find the first dim where any goal_mask is nonzero.
        mask_any = goal_masks.max(axis=0)
        goal_dim = int(jnp.argmax(mask_any))
        goal_vals = goals[:, goal_dim]
        idx_min = int(jnp.argmin(goal_vals))
        idx_max = int(jnp.argmax(goal_vals))
        goal_indices = jnp.array(sorted(set([idx_min, idx_max])))

    network = make_q_network(pqn_config)

    per_goal_true, per_goal_pred = [], []
    for gi in goal_indices:
        goal, mask = goals[int(gi)], goal_masks[int(gi)]
        goal_repr = make_goal_repr(goal, mask, n_starts, state_dim)

        def _true_step(s, _):
            q = network.apply(
                {"params": q_params, "batch_stats": q_batch_stats},
                s, goal_repr, train=False,
            )
            action = jnp.argmax(q, axis=-1)
            s_next = dynamics_fn(s, action)
            return s_next, s_next

        def _pred_step(s, _):
            q = network.apply(
                {"params": q_params, "batch_stats": q_batch_stats},
                s, goal_repr, train=False,
            )
            action = jnp.argmax(q, axis=-1)
            a_oh = jax.nn.one_hot(action, action_dim)
            s_next = apply_wm(p_model, p_params, s, a_oh, residual=_residual,
                              state_to_eff_fn=_state_to_eff_fn,
                              angle_dims=_angle_dims, eff_to_obs_fn=eff_to_obs_fn,
                              wm_input_dims=_wm_input_dims)
            return s_next, s_next

        _, true_traj = jax.lax.scan(_true_step, starts, None, length=n_steps)
        _, pred_traj = jax.lax.scan(_pred_step, starts, None, length=n_steps)
        true_traj = jnp.concatenate([starts[None], true_traj], axis=0)
        pred_traj = jnp.concatenate([starts[None], pred_traj], axis=0)
        goal_true = [true_traj[:, i] for i in range(n_starts)]
        goal_pred = [pred_traj[:, i] for i in range(n_starts)]
        per_goal_true.append(goal_true)
        per_goal_pred.append(goal_pred)

    return per_goal_true, per_goal_pred, starts, n_y, n_x, goal_indices
