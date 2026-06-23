"""Architecture sweep over (depth × width × seed) for Reacher.

Paper §G.2 sweep: D ∈ {1..6} × W ∈ {32, 64, 128, 256, 512, 1024, 2048}
× 10 seeds = 420 cells. Each cell:
  1. Trains PQN for TOTAL_TIMESTEPS steps with
     NETWORK_DENSE_LAYERS=D and NETWORK_DENSE_HIDDEN_SIZE=W;
     saves ~20 intermediate step_*.pkl checkpoints + a final
     pqn_checkpoint.pkl with metrics.
  2. For every PQN checkpoint, trains a fresh world model and records
     dynamics MSE → wm_track/tracking.npz.
After all cells finish, the aggregator runs automatically and writes
aggregate/combined_three_panel.png (paper fig. 4) plus slice plots
and the Table 2 correlation summary.

Each cell logs to wandb (group=arch_sweep_{ts}, job_type ∈
{pqn, wm_track}).

Usage
-----
    uv run python scripts/architecture_sweep.py --config configs/reacher.py

Runs sequentially in one process (~35 GPU-hours on H100).
"""

import jax
import jax.numpy as jnp

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import import_config_module


def _enumerate_cells(depths, widths, seeds):
    """Return ordered list of (D, W, seed) triples.

    Outer dim = (D, W) so all checkpoints of a single cell share a JIT
    compile of train_world_model.
    """
    return [(D, W, S) for D in depths for W in widths for S in seeds]


def _build_env_and_goals(cfg, pqn_config):
    """Build the env + ContinuousGoal pytree exactly as run.py does."""
    env_name = pqn_config["ENV_NAME"]
    from envs.goals import ContinuousGoal

    if env_name == "Reacher":
        from envs.reacher import Reacher
        basic_env = Reacher(
            reward_type=pqn_config["REWARD_TYPE"],
            sigma=pqn_config["REWARD_SIGMA"],
            a=pqn_config["REWARD_A"],
            max_steps_in_episode=pqn_config["MAX_STEPS_IN_EPISODE"],
            torque_values=cfg["REACHER_TORQUE_VALUES"],

        )
    else:
        raise NotImplementedError(
            f"architecture_sweep does not yet handle ENV_NAME={env_name!r}. "
            f"Supported: Reacher. Add a branch in "
            f"_build_env_and_goals to extend it."
        )
    env_params = basic_env.default_params
    all_goals = ContinuousGoal(
        target_state=pqn_config["GOALS"],
        reward_mask=pqn_config["REWARD_MASK"],
    )
    return basic_env, env_params, all_goals


def _wm_lift_for_env(env_name, wm_output_dim, state_dim):
    """Return (state_to_eff_fn, eff_to_obs_fn) if WM_OUTPUT_DIM < STATE_DIM."""
    if wm_output_dim is None or wm_output_dim == state_dim:
        return None, None
    if env_name == "Reacher":
        from envs.reacher_utils import (
            reacher_obs_to_effective, reacher_effective_to_obs,
        )
        return reacher_obs_to_effective, reacher_effective_to_obs
    raise ValueError(
        f"WM_OUTPUT_DIM={wm_output_dim} but no effective<->obs lift "
        f"registered for env_name={env_name!r}"
    )


def _safe_wandb_init(init_timeout=300, retries=2, retry_sleep=10, **kwargs):
    """Robust wandb.init: longer timeout, retry on failure, never crash the cell.

    Default wandb init_timeout is 90s, which trips on slow/flaky network. We
    bump to 300s, retry once after a brief pause, and on terminal failure fall
    back to mode="disabled" so subsequent wandb.log / wandb.finish calls in
    the cell remain valid no-ops instead of raising.
    """
    import time as _time
    import wandb
    settings = wandb.Settings(init_timeout=init_timeout)
    last_err = None
    for attempt in range(retries + 1):
        try:
            return wandb.init(settings=settings, **kwargs)
        except Exception as e:
            last_err = e
            print(f"  [wandb] init attempt {attempt + 1}/{retries + 1} "
                  f"failed: {type(e).__name__}: {e}")
            if attempt < retries:
                _time.sleep(retry_sleep)
    print(f"  [wandb] giving up after {retries + 1} attempts; "
          f"continuing without wandb logging (last error: {last_err}).")
    return wandb.init(mode="disabled", reinit=True)


def _run_pqn_cell(cfg, D, W, seed, run_dir, sweep_group, base_seed=42):
    """Train PQN for one (D, W, seed) cell, saving intermediate checkpoints."""
    final_path = os.path.join(run_dir, "pqn_checkpoint.pkl")

    from training.pqn_utils import train_or_load_pqn, save_pqn
    from configs.utils import save_config
    from plotting.pqn import plot_pqn_training

    PQN_CONFIG = dict(cfg["PQN_CONFIG"])
    PQN_CONFIG["NETWORK_DENSE_LAYERS"] = int(D)
    PQN_CONFIG["NETWORK_DENSE_HIDDEN_SIZE"] = int(W)
    PQN_CONFIG["SEED"] = base_seed + int(seed)

    basic_env, env_params, all_goals = _build_env_and_goals(cfg, PQN_CONFIG)

    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    use_wandb = PQN_CONFIG.get("USE_WANDB", False)
    if use_wandb:
        _safe_wandb_init(
            project=PQN_CONFIG["WANDB_PROJECT"],
            entity=PQN_CONFIG.get("WANDB_ENTITY"),
            group=sweep_group,
            job_type="pqn",
            tags=[f"d{D}", f"w{W}", f"s{seed}",
                  PQN_CONFIG["ENV_NAME"].lower(), "arch_sweep"],
            name=f"d{D}_w{W}_s{seed}_pqn",
            config={**PQN_CONFIG, "_arch_sweep_D": D, "_arch_sweep_W": W,
                    "_arch_sweep_seed": seed},
            reinit=True,
        )

    print(f"\n{'=' * 70}")
    print(f"PQN cell  D={D}  W={W}  seed={seed}  ->  {run_dir}")
    print(f"{'=' * 70}")

    t0 = time.time()
    q_params, q_batch_stats, metrics = train_or_load_pqn(
        PQN_CONFIG, basic_env, env_params, all_goals,
        checkpoint_dir=checkpoint_dir,
    )
    dt = time.time() - t0

    save_pqn(
        q_params, q_batch_stats, final_path,
        config=PQN_CONFIG, metrics=metrics,
    )
    save_config(PQN_CONFIG, os.path.join(run_dir, "pqn_config.json"))
    plot_pqn_training(metrics, save_path=os.path.join(run_dir, "pqn_training.png"))
    print(f"PQN cell d{D}_w{W}_s{seed} done in {dt:.1f}s")

    if use_wandb:
        import wandb
        wandb.summary["wallclock_s"] = dt
        wandb.finish()


def _run_wm_track_cell(cfg, D, W, seed, run_dir, sweep_group):
    """Run per-checkpoint WM training for one cell."""
    subdir = "wm_track"
    from eval.track_wm import track_wm_for_run

    # Use the saved per-cell PQN config so network-arch fields match checkpoints.
    saved_pqn_path = os.path.join(run_dir, "pqn_config.json")
    if not os.path.exists(saved_pqn_path):
        raise FileNotFoundError(
            f"Cannot run wm_track for {run_dir}: missing pqn_config.json. "
            f"Run PQN phase first."
        )
    from configs.utils import load_config as _load_json_cfg
    PQN_CONFIG = dict(cfg["PQN_CONFIG"])
    saved = _load_json_cfg(saved_pqn_path)
    # STATE_RANGES is env-level metadata — let live config override so post-hoc
    # range edits take effect. Mirrors run.py:773.
    saved.pop("STATE_RANGES", None)
    PQN_CONFIG.update(saved)

    WM_CONFIG = dict(cfg["WM_CONFIG"])
    # Vary WM init seed across cells so per-(D, W, seed) WM training is
    # independent; otherwise every cell would init from PRNGKey(0).
    WM_CONFIG["SEED"] = int(seed)
    ENV_CONFIG = dict(cfg["ENV_CONFIG"])

    basic_env, env_params, _ = _build_env_and_goals(cfg, PQN_CONFIG)

    # train_world_model takes plain (num_goals, state_dim) arrays, not the
    # ContinuousGoal pytree the PQN trainer expects.
    goals_arr = PQN_CONFIG["GOALS"]
    goal_masks_arr = PQN_CONFIG["REWARD_MASK"]

    state_to_eff_fn, eff_to_obs_fn = _wm_lift_for_env(
        ENV_CONFIG["ENV_NAME"],
        WM_CONFIG.get("WM_OUTPUT_DIM"),
        PQN_CONFIG["STATE_DIM"],
    )

    eff_sample = "reset" if WM_CONFIG.get("SAMPLE_FROM_RESET") else "uniform"

    use_wandb = PQN_CONFIG.get("USE_WANDB", False)
    if use_wandb:
        _safe_wandb_init(
            project=PQN_CONFIG["WANDB_PROJECT"],
            entity=PQN_CONFIG.get("WANDB_ENTITY"),
            group=sweep_group,
            job_type="wm_track",
            tags=[f"d{D}", f"w{W}", f"s{seed}",
                  PQN_CONFIG["ENV_NAME"].lower(), "arch_sweep",
                  f"sample={eff_sample}", f"subdir={subdir}"],
            name=f"d{D}_w{W}_s{seed}_{subdir}",
            config={**WM_CONFIG, "_arch_sweep_D": D, "_arch_sweep_W": W,
                    "_arch_sweep_seed": seed,
                    "_wm_subdir": subdir, "_wm_sample_mode": eff_sample},
            reinit=True,
        )

    print(f"\n{'=' * 70}")
    print(f"WM-track cell  D={D}  W={W}  seed={seed}  "
          f"sample={eff_sample}  ->  {run_dir}/{subdir}/")
    print(f"{'=' * 70}")

    t0 = time.time()
    track_wm_for_run(
        run_dir,
        PQN_CONFIG,
        WM_CONFIG,
        ENV_CONFIG,
        basic_env,
        env_params,
        goals_arr,
        goal_masks_arr,
        env_terminated_fn=None,
        state_to_eff_fn=state_to_eff_fn,
        eff_to_obs_fn=eff_to_obs_fn,
        wm_output_dim=WM_CONFIG.get("WM_OUTPUT_DIM"),
        save_wm_ckpts=True,
        use_wandb=use_wandb,
        subdir=subdir,
    )
    dt = time.time() - t0
    print(f"WM-track cell d{D}_w{W}_s{seed} done in {dt:.1f}s")

    if use_wandb:
        import wandb
        wandb.summary["wallclock_s"] = dt
        wandb.finish()


def _run_unseen_cell(
    cfg, D, W, seed, run_dir, sweep_group, only_labels=None,
):
    """Run VI-on-WM unseen-goal evaluation for one cell.

    Reuses the existing PQN config saved in pqn_config.json (so the network
    architecture matches the checkpoints). Trains a final-checkpoint WM
    in-place if missing.
    """
    wm_subdir = "wm_track_final"
    unseen_subdir = "unseen_goals_arch"
    unseen_dir = os.path.join(run_dir, unseen_subdir)

    saved_pqn_path = os.path.join(run_dir, "pqn_config.json")
    if not os.path.exists(saved_pqn_path):
        raise FileNotFoundError(
            f"Cannot run unseen for {run_dir}: missing pqn_config.json. "
            f"Run PQN phase first."
        )
    from configs.utils import load_config as _load_json_cfg
    PQN_CONFIG = dict(cfg["PQN_CONFIG"])
    saved = _load_json_cfg(saved_pqn_path)
    # STATE_RANGES is env-level metadata — let the live config override the
    # saved copy so post-hoc edits (e.g. tightening ω-range) take effect.
    # Mirrors run.py:773.
    saved.pop("STATE_RANGES", None)
    PQN_CONFIG.update(saved)
    WM_CONFIG = dict(cfg["WM_CONFIG"])
    ENV_CONFIG = dict(cfg["ENV_CONFIG"])

    from eval.unseen_goals_arch_sweep import run_unseen_for_cell

    use_wandb = PQN_CONFIG.get("USE_WANDB", False)
    if use_wandb:
        _safe_wandb_init(
            project=PQN_CONFIG["WANDB_PROJECT"],
            entity=PQN_CONFIG.get("WANDB_ENTITY"),
            group=sweep_group,
            job_type="unseen",
            tags=[f"d{D}", f"w{W}", f"s{seed}",
                  PQN_CONFIG["ENV_NAME"].lower(), "arch_sweep", "unseen"],
            name=f"d{D}_w{W}_s{seed}_unseen",
            config={"_arch_sweep_D": D, "_arch_sweep_W": W,
                    "_arch_sweep_seed": seed,
                    "unseen_labels": only_labels or "all"},
            reinit=True,
        )

    print(f"\n{'=' * 70}")
    print(f"Unseen cell  D={D}  W={W}  seed={seed}  ->  {run_dir}/{unseen_subdir}/")
    print(f"{'=' * 70}")
    t0 = time.time()
    run_unseen_for_cell(
        run_dir, PQN_CONFIG, WM_CONFIG, ENV_CONFIG, cfg,
        use_wandb=use_wandb,
        only_labels=only_labels,
        wm_subdir=wm_subdir,
        unseen_subdir=unseen_subdir,
    )
    dt = time.time() - t0
    print(f"Unseen cell d{D}_w{W}_s{seed} done in {dt:.1f}s")

    if use_wandb:
        import wandb
        wandb.summary["wallclock_s"] = dt
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Architecture sweep (depth × width × seed) for Reacher."
    )
    parser.add_argument("--config", type=str, default="configs/reacher.py")
    # Paper grid: D ∈ {1..6}, W ∈ {32, 64, 128, 256, 512, 1024, 2048} → 42 archs.
    parser.add_argument("--depths", type=str, default="1,2,3,4,5,6")
    parser.add_argument("--widths", type=str, default="32,64,128,256,512,1024,2048")
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--pivot_depth", type=int, default=4,
                        help="Fixed depth for the by-width plot (passed to aggregator)")
    parser.add_argument("--pivot_width", type=int, default=512,
                        help="Fixed width for the by-depth plot (passed to aggregator)")
    parser.add_argument(
        "--phases", type=str, default="pqn,wm_track",
        help="Comma-separated subset of {pqn, wm_track, unseen}",
    )
    # Style / presentation flags forwarded to aggregate_arch_sweep
    parser.add_argument("--paper_mode", action="store_true",
                        help="Aggregator: drop titles + use paper fonts/sizes/dpi.")
    parser.add_argument("--curve_palette", type=str, default="sequential",
                        choices=["sequential", "tab10"],
                        help="Aggregator: slice-plot curve coloring.")
    parser.add_argument("--cmap_heatmap", type=str, default=None,
                        help="Aggregator: override heatmap colormap.")
    parser.add_argument("--no_colorbar", action="store_true",
                        help="Aggregator: drop colorbars from heatmaps.")
    parser.add_argument("--log_pqn", action="store_true",
                        help="Aggregator: log scale on PQN return slice + heatmap.")
    parser.add_argument("--wm_mse_ymax", type=float, default=None,
                        help="Aggregator: cap WM-MSE slice y-axis + heatmap "
                             "colorscale at this value (e.g. 1e-2). Clips "
                             "bad-fit cells that compress the visible range.")
    parser.add_argument("--base_seed", type=int, default=42)

    args = parser.parse_args()

    cfg = import_config_module(args.config)

    depths = [int(x) for x in args.depths.split(",") if x]
    widths = [int(x) for x in args.widths.split(",") if x]
    seeds = list(range(args.num_seeds))

    phases = set(args.phases.split(","))
    valid_phases = {"pqn", "wm_track", "unseen"}
    unknown = phases - valid_phases
    if unknown:
        raise ValueError(f"Unknown phase(s): {unknown}; valid: {valid_phases}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    sweep_dir = f"outputs/{config_stem}/sweep_arch_{ts}"
    os.makedirs(sweep_dir, exist_ok=True)

    sweep_group = f"arch_sweep_{os.path.basename(sweep_dir)}"

    cells = _enumerate_cells(depths, widths, seeds)

    print(f"Sweep dir   : {sweep_dir}")
    print(f"Sweep group : {sweep_group}")
    print(f"Phases      : {sorted(phases)}")
    print(f"Cells       : {len(cells)}")

    for D, W, S in cells:
        run_dir = os.path.join(sweep_dir, f"d{D}_w{W}_s{S}")
        if "pqn" in phases:
            _run_pqn_cell(cfg, D, W, S, run_dir, sweep_group,
                          base_seed=args.base_seed)
        if "wm_track" in phases:
            _run_wm_track_cell(cfg, D, W, S, run_dir, sweep_group)
        if "unseen" in phases:
            _run_unseen_cell(cfg, D, W, S, run_dir, sweep_group)

    from eval.aggregate_arch_sweep import aggregate_arch_sweep
    aggregate_arch_sweep(
        sweep_dir,
        pivot_depth=args.pivot_depth,
        pivot_width=args.pivot_width,
        paper_mode=args.paper_mode,
        curve_palette=args.curve_palette,
        cmap_heatmap=args.cmap_heatmap,
        show_cbar=not args.no_colorbar,
        log_pqn=args.log_pqn,
        wm_mse_ymax=args.wm_mse_ymax,
    )


if __name__ == "__main__":
    main()
