"""Policy-only WM extraction (pi-learning) from a trained PQN checkpoint.

Constructs Boltzmann policies pi_g = softmax(Q_pqn/tau), discards the Q
magnitudes (log_softmax), and fits {world model, value net} by minimising
the soft (or hard) Bellman residual on the policy log-probs — see
training/policy_wm.py for the loss.

Writes <checkpoint_dir>/piwm_<TS>/ with: piwm_checkpoint.pkl,
piwm_config.json, piwm_training.png, results.txt (same Avg MSE / Avg NMSE
format as the Q-based wm_<TS>/results.txt, for direct comparison), and
V_psi diagnostics appended to results.txt.

Usage:
    uv run python scripts/policy_wm.py \\
        --pqn_checkpoint outputs/reacher/seeds_20260604_091619/seed_0
    uv run python scripts/policy_wm.py --consistency hard --tau 0.03 \\
        --num_steps 2000 --batch_size 512   # CPU smoke test
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CHECKPOINT = "outputs/reacher/seeds_20260604_091619/seed_0"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=str, default="configs/reacher.py")
    parser.add_argument(
        "--pqn_checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help="PQN run dir (or pqn_checkpoint.pkl path) to extract from.",
    )
    parser.add_argument("--tau", type=float, default=None,
                        help="Boltzmann temperature (default: PIWM_CONFIG['TAU']).")
    parser.add_argument("--consistency", type=str, default=None,
                        choices=["soft", "hard"],
                        help="Which normalisation of log pi (default: config).")
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--v_lr", type=float, default=None)
    parser.add_argument("--v_hidden", type=int, default=None)
    parser.add_argument("--v_layers", type=int, default=None)
    parser.add_argument("--v_stop_grad", action="store_true",
                        help="Semi-gradient bootstrap (freeze V_psi params in target).")
    parser.add_argument("--wm_loss", type=str, default=None, choices=["l1", "mse"])
    parser.add_argument("--seed", type=int, default=None,
                        help="PiWM training seed (default: PIWM_CONFIG['SEED']).")
    parser.add_argument("--num_extraction_goals", type=int, default=None,
                        help="Query the policy at this many fingertip goals tiling the "
                             "reachable disk (sunflower layout) instead of the agent's "
                             "training goals. Denser reward coverage removes the flat-V "
                             "degeneracy of the joint (V, WM) fit.")
    parser.add_argument("--extraction_goal_radius", type=float, default=1.5,
                        help="Max fingertip radius of the extraction-goal disk.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir (default: <checkpoint_dir>/piwm_<TS>).")
    args = parser.parse_args()

    from run import import_config_module

    cfg = import_config_module(args.config)
    PQN_CONFIG = cfg["PQN_CONFIG"]
    ENV_CONFIG = cfg["ENV_CONFIG"]
    if "PIWM_CONFIG" not in cfg:
        raise ValueError(f"{args.config} has no PIWM_CONFIG (only Reacher is wired up).")
    piwm_config = dict(cfg["PIWM_CONFIG"])
    goals = PQN_CONFIG["GOALS"]
    goal_masks = PQN_CONFIG["REWARD_MASK"]
    assert ENV_CONFIG["ENV_NAME"] == "Reacher", "policy_wm.py currently supports Reacher only."

    # Extraction-time goal densification: the policy is queryable at any
    # continuous fingertip target (the Q-net conditions on goal dims [4,5]
    # only), so tile the reachable disk with a sunflower layout. The agent's
    # 4 training goals are irrelevant here — what matters is that the union
    # of reward balls anchors r_g(s') over most of the state space.
    if args.num_extraction_goals is not None:
        import numpy as _np
        import jax.numpy as jnp
        n_g = args.num_extraction_goals
        idx = _np.arange(n_g)
        radii = args.extraction_goal_radius * _np.sqrt((idx + 0.5) / n_g)
        angles = idx * _np.pi * (3.0 - _np.sqrt(5.0))  # golden angle
        fp = _np.stack([radii * _np.cos(angles), radii * _np.sin(angles)], axis=-1)
        state_dim = PQN_CONFIG["STATE_DIM"]
        goals = jnp.zeros((n_g, state_dim)).at[:, 4:6].set(jnp.array(fp))
        goal_masks = jnp.zeros((n_g, state_dim)).at[:, 4:6].set(1.0)
        print(f"[PiWM] extraction goals: {n_g} fingertip targets "
              f"(sunflower, r <= {args.extraction_goal_radius})")

    # CLI overrides.
    _overrides = {
        "TAU": args.tau, "CONSISTENCY": args.consistency,
        "NUM_STEPS": args.num_steps, "BATCH_SIZE": args.batch_size,
        "LR": args.lr, "V_LR": args.v_lr,
        "V_DENSE_HIDDEN_SIZE": args.v_hidden, "V_DENSE_LAYERS": args.v_layers,
        "WM_LOSS": args.wm_loss, "SEED": args.seed,
    }
    for k, v in _overrides.items():
        if v is not None:
            piwm_config[k] = v
    if args.v_stop_grad:
        piwm_config["V_STOP_GRAD"] = True

    # Resolve checkpoint path and run dir.
    ckpt = args.pqn_checkpoint
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, "pqn_checkpoint.pkl")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No PQN checkpoint at {ckpt}")
    pqn_dir = os.path.dirname(ckpt)

    # Pick up training-time network architecture from the saved config
    # (mirrors run.py's checkpoint-loading branch).
    saved_config_path = os.path.join(pqn_dir, "pqn_config.json")
    if os.path.exists(saved_config_path):
        with open(saved_config_path) as f:
            saved_config = json.load(f)
        for k in ("GOAL_INPUT_DIMS", "OBS_INPUT_DIMS", "NETWORK_DENSE_HIDDEN_SIZE",
                  "NETWORK_DENSE_LAYERS", "NORM_TYPE", "NETWORK_SIGMOID_OUTPUTS"):
            if k in saved_config:
                PQN_CONFIG[k] = saved_config[k]
            elif k in PQN_CONFIG and k not in saved_config:
                del PQN_CONFIG[k]

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(pqn_dir, f"piwm_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"PQN checkpoint: {ckpt}")
    print(f"Output dir:     {out_dir}")
    print(f"tau={piwm_config['TAU']}  consistency={piwm_config['CONSISTENCY']}  "
          f"steps={piwm_config['NUM_STEPS']}  batch={piwm_config['BATCH_SIZE']}")
    print("=" * 60)

    # ── Env + samplers + effective-space lifts (run.py Reacher wiring) ──
    from envs.reacher import Reacher
    from envs.samplers import make_reacher_uniform_sampler, make_env_reset_sampler
    from envs.env_dynamics import make_env_dynamics_fn
    from envs.reacher_utils import reacher_obs_to_effective, reacher_effective_to_obs

    basic_env = Reacher(
        reward_type=PQN_CONFIG["REWARD_TYPE"],
        sigma=PQN_CONFIG["REWARD_SIGMA"],
        a=PQN_CONFIG["REWARD_A"],
        max_steps_in_episode=PQN_CONFIG["MAX_STEPS_IN_EPISODE"],
        torque_values=cfg["REACHER_TORQUE_VALUES"],
    )
    env_params = basic_env.default_params
    env_terminated_fn = None

    wm_sample_fn = make_reacher_uniform_sampler(PQN_CONFIG["STATE_RANGES"])
    if piwm_config.get("SAMPLE_FROM_RESET", False):
        wm_sample_fn = make_env_reset_sampler(basic_env, env_params)

    wm_output_dim = piwm_config.get("WM_OUTPUT_DIM")
    state_to_eff_fn = eff_to_obs_fn = None
    if wm_output_dim is not None and wm_output_dim != PQN_CONFIG["STATE_DIM"]:
        state_to_eff_fn = reacher_obs_to_effective
        eff_to_obs_fn = reacher_effective_to_obs

    # ── Load PQN + train ──
    from training.pqn_utils import load_pqn
    from training.policy_wm import train_policy_wm, save_policy_wm, value_diagnostics
    from configs.utils import save_config

    q_params, q_batch_stats, _ = load_pqn(ckpt)
    save_config(piwm_config, os.path.join(out_dir, "piwm_config.json"))

    params, losses = train_policy_wm(
        q_params, q_batch_stats, piwm_config, PQN_CONFIG, goals, goal_masks,
        env_terminated_fn, sample_states_fn=wm_sample_fn,
        state_to_eff_fn=state_to_eff_fn,
        eff_to_obs_fn=eff_to_obs_fn,
        wm_output_dim=wm_output_dim,
    )
    save_policy_wm(params, losses, os.path.join(out_dir, "piwm_checkpoint.pkl"),
                   config=piwm_config)

    from plotting.wm import plot_loss_curve
    plot_loss_curve(losses, save_path=os.path.join(out_dir, "piwm_training.png"))

    # ── WM accuracy: same harness + results.txt format as the Q-based WM ──
    from eval.world_model import compare_dynamics
    compare_dynamics(
        params["wm"], out_dir, piwm_config, ENV_CONFIG,
        make_env_dynamics_fn(basic_env, env_params, "Reacher"),
        goals=goals, goal_masks=goal_masks, losses=losses,
        plots=True, sample_states_fn=wm_sample_fn,
        state_to_eff_fn=state_to_eff_fn,
        eff_to_obs_fn=eff_to_obs_fn,
        wm_output_dim=wm_output_dim,
    )

    # ── V_psi diagnostics: did the nuisance value learn the discarded level? ──
    diag = value_diagnostics(
        params, q_params, q_batch_stats, piwm_config, PQN_CONFIG,
        goals, goal_masks, sample_states_fn=wm_sample_fn,
    )
    ref_name = "max_a Q" if piwm_config["CONSISTENCY"] == "hard" else "tau*logsumexp(Q/tau)"
    print(f"\n── V_psi diagnostics (reference: {ref_name}) ──")
    lines = [f"\nV_psi diagnostics (reference: {ref_name}):\n"]
    for i in range(len(goals)):
        m = diag[i]
        line = f"  goal {i}: V_NMSE={m['v_nmse']:.1e}  Qhat_NMSE={m['q_nmse']:.1e}"
        if len(goals) <= 8 or i < 4:
            print(line)
        lines.append(line + "\n")
    if len(goals) > 8:
        print(f"  ... ({len(goals) - 4} more goals in results.txt)")
    avg = diag["avg"]
    line = f"  avg:    V_NMSE={avg['v_nmse']:.1e}  Qhat_NMSE={avg['q_nmse']:.1e}"
    print(line)
    lines.append(line + "\n")
    with open(os.path.join(out_dir, "results.txt"), "a") as f:
        f.writelines(lines)

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
