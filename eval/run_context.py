"""Build the env + dynamics + lift-fn wiring for the recovery-tracking scripts.

Mirrors run.py's env creation and VI/WM wiring (the MC vs Reacher branches) so
scripts/recovery_vs_qerror.py and scripts/dist_vs_uniform.py don't each
re-derive it. Returns everything eval/track_recovery.py::track_recovery_for_run
needs.
"""

from run import import_config_module


def build_context(config_path):
    """Load a config .py and build the tracking context for its env.

    Returns a dict with: pqn_config, wm_config, env_config, basic_env,
    env_params, goals, goal_masks, vi_dynamics_fn, wm_dynamics_fn,
    state_to_obs_fn, obs_to_grid_fn, grid_state_dim, grid_state_ranges,
    env_terminated_fn, state_to_eff_fn, eff_to_obs_fn, wm_output_dim, wm_sample_fn.
    """
    cfg = import_config_module(config_path)
    PQN_CONFIG = cfg["PQN_CONFIG"]
    WM_CONFIG = cfg["WM_CONFIG"]
    ENV_CONFIG = cfg["ENV_CONFIG"]
    env_name = ENV_CONFIG["ENV_NAME"]
    goals = PQN_CONFIG["GOALS"]
    goal_masks = PQN_CONFIG["REWARD_MASK"]

    from envs.env_dynamics import make_env_dynamics_fn

    ctx = dict(
        pqn_config=PQN_CONFIG, wm_config=WM_CONFIG, env_config=ENV_CONFIG,
        goals=goals, goal_masks=goal_masks,
        env_terminated_fn=None,               # neither env auto-terminates
        state_to_obs_fn=None, obs_to_grid_fn=None,
        grid_state_dim=None, grid_state_ranges=None,
        state_to_eff_fn=None, eff_to_obs_fn=None, wm_output_dim=None,
        wm_sample_fn=None,
    )

    if env_name == "MountainCar":
        from envs.mountaincar import MountainCar
        basic_env = MountainCar(max_steps_in_episode=PQN_CONFIG["MAX_STEPS_IN_EPISODE"])
        env_params = basic_env.default_params
        dyn = make_env_dynamics_fn(basic_env, env_params, env_name)
        ctx.update(vi_dynamics_fn=dyn, wm_dynamics_fn=dyn)   # grid == obs (2D)

    elif env_name == "Reacher":
        from envs.reacher import Reacher
        from envs.reacher_utils import (
            make_reacher_effective_dynamics_fn, reacher_effective_to_obs,
            reacher_obs_to_effective,
        )
        from envs.samplers import (
            make_reacher_uniform_sampler, make_env_reset_sampler,
        )
        basic_env = Reacher(
            reward_type=PQN_CONFIG["REWARD_TYPE"], sigma=PQN_CONFIG["REWARD_SIGMA"],
            a=PQN_CONFIG["REWARD_A"], max_steps_in_episode=PQN_CONFIG["MAX_STEPS_IN_EPISODE"],
            torque_values=cfg["REACHER_TORQUE_VALUES"])
        env_params = basic_env.default_params
        wm_sample_fn = make_reacher_uniform_sampler(PQN_CONFIG["STATE_RANGES"])
        if WM_CONFIG.get("SAMPLE_FROM_RESET", False):
            wm_sample_fn = make_env_reset_sampler(basic_env, env_params)
        ctx.update(
            vi_dynamics_fn=make_reacher_effective_dynamics_fn(basic_env, env_params),
            wm_dynamics_fn=make_env_dynamics_fn(basic_env, env_params, env_name),
            state_to_obs_fn=reacher_effective_to_obs,
            obs_to_grid_fn=reacher_obs_to_effective,
            grid_state_dim=ENV_CONFIG["VI_STATE_DIM"],
            grid_state_ranges=ENV_CONFIG["VI_STATE_RANGES"],
            state_to_eff_fn=reacher_obs_to_effective,
            eff_to_obs_fn=reacher_effective_to_obs,
            wm_output_dim=WM_CONFIG.get("WM_OUTPUT_DIM"),
            wm_sample_fn=wm_sample_fn,
        )
    else:
        raise ValueError(f"Unsupported ENV_NAME {env_name!r}")

    ctx.update(basic_env=basic_env, env_params=env_params)
    return ctx
