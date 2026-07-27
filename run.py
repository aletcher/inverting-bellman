"""Unified experiment runner (Reacher / MountainCar).

Phases: pqn, plot_pqn, vi, wm, plot_wm, unseen

Default --num_seeds 1 (cheap dev). Pass --num_seeds 10 to reproduce the
paper-headline numbers (Q_MSE, WM_MSE, per-unseen-goal R*/R_WM, mean ± SE
across seeds), which writes aggregate.json + summary.txt next to the
per-seed dirs.

Usage:
    uv run python run.py --config configs/reacher.py
    uv run python run.py --config configs/mountaincar-position.py
    uv run python run.py --config configs/reacher.py --num_seeds 10
    uv run python run.py --config configs/reacher.py --phases pqn,plot_pqn
    uv run python run.py --config configs/reacher.py --LR 5e-4 --GAMMA 0.95
    uv run python run.py --config configs/reacher.py --phases wm,plot_wm --pqn_checkpoint outputs/...
"""

import jax
import jax.numpy as jnp

import argparse
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ── Config loading ────────────────────────────────────────────────────────────


def import_config_module(path):
    """Import a Python config file and return all uppercase globals as a dict.

    Distinct from configs.utils.load_config (which reads a JSON snapshot of
    such a dict written next to a checkpoint).
    """
    spec = importlib.util.spec_from_file_location("_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {k: getattr(mod, k) for k in dir(mod) if k.isupper() and not k.startswith("_")}


def _coerce(raw_val, existing):
    """Cast raw CLI string to the type of the existing value."""
    if isinstance(existing, bool):
        return raw_val.lower() not in ("0", "false", "no")
    if isinstance(existing, int):
        return int(raw_val)
    if isinstance(existing, float):
        return float(raw_val)
    if isinstance(existing, str):
        return raw_val
    raise ValueError(f"Cannot override value of type {type(existing).__name__} from CLI")


def apply_overrides(cfg, pairs):
    """Apply --KEY VALUE override pairs, keeping flat cfg and nested dicts in sync.

    --LR 5e-4         updates cfg["LR"] and PQN_CONFIG["LR"]
    --WM_LR 5e-4      updates cfg["WM_LR"] and WM_CONFIG["LR"]
    --WM_BATCH_SIZE   updates cfg["WM_BATCH_SIZE"] and WM_CONFIG["BATCH_SIZE"]

    Keys shared between PQN_CONFIG and WM_CONFIG (LR, NUM_STEPS, ...) still
    require the WM_ prefix to target WM_CONFIG; a bare --LR only updates PQN.
    """
    pqn = cfg.get("PQN_CONFIG", {})
    wm  = cfg.get("WM_CONFIG", {})
    for key, raw_val in pairs:
        updated = False
        if key in cfg and not isinstance(cfg[key], dict):
            cfg[key] = _coerce(raw_val, cfg[key])
            updated = True
        if key in pqn:
            pqn[key] = _coerce(raw_val, pqn[key])
            updated = True
        # WM-only keys (no prefix): --SAMPLE_FROM_RESET → WM_CONFIG["SAMPLE_FROM_RESET"]
        if key in wm and key not in pqn:
            wm[key] = _coerce(raw_val, wm[key])
            updated = True
        # Explicit WM_ prefix --WM_LR → WM_CONFIG["LR"] (for shared keys).
        if key.startswith("WM_"):
            sub = key[3:]
            if sub in wm:
                wm[sub] = _coerce(raw_val, wm[sub])
                updated = True
        if not updated:
            raise ValueError(f"Unknown config key: {key}")
    return cfg


def _merge_saved_config(search_dir, fname, target_dict, fixup=None, name="config"):
    """Merge a JSON-snapshot config into target_dict (in-place).

    fixup(dict) is called on the loaded dict before merging — used to
    drop env-metadata keys (PQN) or set defaults for post-hoc-added flags
    (WM). Silent no-op when search_dir is None; warns if the file is
    missing but search_dir is set.
    """
    if search_dir is None:
        return
    from configs.utils import load_config
    path = os.path.join(search_dir, fname)
    if not os.path.exists(path):
        print(f"WARNING: no {fname} found in {search_dir}")
        return
    saved = load_config(path)
    if fixup is not None:
        fixup(saved)
    target_dict.update(saved)
    print(f"Loaded {name} config from {path}")


# ── Multi-seed aggregation ────────────────────────────────────────────────────


_WM_AVG_MSE_RE = re.compile(r"^Avg MSE:\s*([0-9.eE+\-]+)", re.M)
_WM_AVG_NMSE_RE = re.compile(r"^Avg NMSE:\s*([0-9.eE+\-]+)", re.M)
_VI_AVG_RE = re.compile(
    r"^Avg V_mse=([0-9.eE+\-]+)\s+V_nmse=([0-9.eE+\-]+)\s*\n"
    r"Avg Q_mse=([0-9.eE+\-]+)\s+Q_nmse=([0-9.eE+\-]+)",
    re.M,
)


def _parse_wm_results(results_path):
    if not results_path.exists():
        return None
    txt = results_path.read_text()
    out = {}
    for key, rx in (("wm_mse", _WM_AVG_MSE_RE), ("wm_nmse", _WM_AVG_NMSE_RE)):
        m = rx.search(txt)
        if m:
            out[key] = float(m.group(1))
    return out or None


def _parse_vi_results(vi_summary_path):
    """Read vi/vi_summary.txt produced by eval.value_iteration."""
    if not vi_summary_path.exists():
        return None
    txt = vi_summary_path.read_text()
    m = _VI_AVG_RE.search(txt)
    if not m:
        return None
    out = {
        "v_mse":  float(m.group(1)),
        "v_nmse": float(m.group(2)),
        "q_mse":  float(m.group(3)),
        "q_nmse": float(m.group(4)),
    }
    return out


def _parse_unseen_results(results_path):
    """Pull the discounted-return row (R*, R_WM) from a per-seed results.txt.

    Returns mean ± std for each policy, where the std is the per-seed
    spread across the 512 eval starts (i.e. what's printed after `±` in
    the file).
    """
    if not results_path.exists():
        return None
    txt = results_path.read_text()
    out = {"True": None, "WM": None}
    in_disc = False
    for line in txt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Discounted returns"):
            in_disc = True
            continue
        if not in_disc:
            continue
        if stripped == "":
            in_disc = False
            continue
        for key in ("True", "WM"):
            if stripped.startswith(f"{key} policy") and ":" in stripped:
                rhs = stripped.split(":", 1)[1]
                if "±" in rhs:
                    lhs, rhs2 = rhs.split("±", 1)
                    try:
                        m = float(lhs.strip().split()[0])
                        s = float(rhs2.strip().split()[0])
                        out[key] = (m, s)
                    except Exception:
                        pass
        if out["True"] is not None and out["WM"] is not None:
            break
    if out["True"] is None or out["WM"] is None:
        return None
    return {
        "r_star_mean": out["True"][0], "r_star_std": out["True"][1],
        "r_wm_mean":   out["WM"][0],   "r_wm_std":   out["WM"][1],
    }


def _mean_se(values):
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    se = statistics.stdev(values) / (len(values) ** 0.5)
    return mean, se


def _aggregate(seed_dirs):
    """Collect paper-headline metrics across seeds; return JSON-able dict."""
    vi_buf = {"q_mse": [], "q_nmse": []}
    wm_buf = {"wm_mse": [], "wm_nmse": []}
    unseen_buf = {}

    for sd in seed_dirs:
        for vi_dir in sorted(sd.glob("vi*")):
            parsed = _parse_vi_results(vi_dir / "vi_summary.txt")
            if parsed:
                vi_buf["q_mse"].append(parsed["q_mse"])
                vi_buf["q_nmse"].append(parsed["q_nmse"])
                break
        for wm_dir in sorted(sd.glob("wm_*")):
            parsed = _parse_wm_results(wm_dir / "results.txt")
            if parsed:
                wm_buf["wm_mse"].append(parsed.get("wm_mse", float("nan")))
                wm_buf["wm_nmse"].append(parsed.get("wm_nmse", float("nan")))
                break
        wm_dirs_for_unseen = sorted(sd.glob("wm_*"))
        if wm_dirs_for_unseen:
            unseen_root = wm_dirs_for_unseen[-1] / "unseen_goals"
            if unseen_root.exists():
                for goal_dir in sorted(unseen_root.iterdir()):
                    if not goal_dir.is_dir():
                        continue
                    parsed = _parse_unseen_results(goal_dir / "results.txt")
                    if parsed:
                        bucket = unseen_buf.setdefault(
                            goal_dir.name,
                            {"r_star_means": [], "r_star_stds": [],
                             "r_wm_means":   [], "r_wm_stds":   []},
                        )
                        bucket["r_star_means"].append(parsed["r_star_mean"])
                        bucket["r_star_stds"].append(parsed["r_star_std"])
                        bucket["r_wm_means"].append(parsed["r_wm_mean"])
                        bucket["r_wm_stds"].append(parsed["r_wm_std"])

    out = {
        "num_seeds_run": len(seed_dirs),
        "vi": {},
        "wm": {},
        "unseen": {},
    }
    for k, vals in vi_buf.items():
        if vals:
            mean, se = _mean_se(vals)
            out["vi"][k] = {"mean": mean, "se": se, "n": len(vals), "values": vals}
    for k, vals in wm_buf.items():
        if vals:
            mean, se = _mean_se(vals)
            out["wm"][k] = {"mean": mean, "se": se, "n": len(vals), "values": vals}
    for label, bucket in unseen_buf.items():
        entry = {}
        # R*: optimal-policy return is deterministic across seeds, so seed-level
        # SE is always 0. Instead report the per-seed std across the 512 eval
        # starts — same statistical type as Fig 3's R* error bar.
        r_star_means = bucket["r_star_means"]
        r_star_stds = bucket["r_star_stds"]
        if r_star_means:
            entry["r_star"] = {
                "mean": float(statistics.fmean(r_star_means)),
                "se":   float(statistics.fmean(r_star_stds)),
                "n":    len(r_star_means),
                "values": r_star_means,
            }
        # R_WM: pool within-seed (across 512 starts) and between-seed (across
        # seed means) spread via the law of total variance — matches Fig 3.
        r_wm_means = bucket["r_wm_means"]
        r_wm_stds = bucket["r_wm_stds"]
        if r_wm_means:
            within_var = statistics.fmean(s ** 2 for s in r_wm_stds)
            between_var = (
                statistics.variance(r_wm_means) if len(r_wm_means) > 1 else 0.0
            )
            entry["r_wm"] = {
                "mean": float(statistics.fmean(r_wm_means)),
                "se":   float((within_var + between_var) ** 0.5),
                "n":    len(r_wm_means),
                "values": r_wm_means,
            }
        out["unseen"][label] = entry
    return out


def _write_summary_txt(summary, out_path):
    """Human-readable companion to aggregate.json: paper-table layout."""
    lines = [
        f"Paper-table summary — {summary['num_seeds_run']} seeds.",
        "All numbers are mean ± SE across seeds.",
        "",
    ]
    if summary["vi"]:
        q_mse = summary["vi"].get("q_mse")
        q_nmse = summary["vi"].get("q_nmse")
        if q_mse:
            lines.append(f"Q_MSE  = {q_mse['mean']:.1e} ± {q_mse['se']:.1e}")
        if q_nmse:
            lines.append(f"Q_NMSE = {q_nmse['mean']:.1e} ± {q_nmse['se']:.1e}")
    if summary["wm"]:
        wm_mse = summary["wm"].get("wm_mse")
        wm_nmse = summary["wm"].get("wm_nmse")
        if wm_mse:
            lines.append(f"WM_MSE  = {wm_mse['mean']:.1e} ± {wm_mse['se']:.1e}")
        if wm_nmse:
            lines.append(f"WM_NMSE = {wm_nmse['mean']:.1e} ± {wm_nmse['se']:.1e}")
    if summary["unseen"]:
        lines.append("")
        lines.append(f"{'Unseen goal':<40} {'R*':>20} {'R_WM':>20}")
        for label, entry in summary["unseen"].items():
            t = entry.get("r_star")
            w = entry.get("r_wm")
            if t and w:
                lines.append(
                    f"{label:<40} "
                    f"{t['mean']:>10.3f} ± {t['se']:.3f}  "
                    f"{w['mean']:>10.3f} ± {w['se']:.3f}"
                )
    out_path.write_text("\n".join(lines) + "\n")


def _run_multi_seed(args, extras):
    """Loop run.py over N seeds (subprocess-isolated) and aggregate."""
    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(range(args.num_seeds))

    if args.sweep_dir is not None:
        sweep_dir = Path(args.sweep_dir)
    else:
        # Config file stem keeps MountainCar position/velocity runs separate.
        config_stem = Path(args.config).stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_dir = Path(f"outputs/{config_stem}/seeds_{ts}")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep dir : {sweep_dir}")
    print(f"Seeds     : {seeds}")
    print(f"Phases    : {args.phases}")
    print(f"Forwarded : {extras}")

    seed_dirs = []
    if not args.aggregate_only:
        for s in seeds:
            seed_dir = sweep_dir / f"seed_{s}"
            seed_dirs.append(seed_dir)
            cmd = [
                sys.executable, sys.argv[0],
                "--config", args.config,
                "--phases", args.phases,
                "--num_seeds", "1",
                "--run_dir", str(seed_dir),
                "--SEED", str(s),
                "--WM_SEED", str(s),
                *(["--save_pqn_checkpoints"] if args.save_pqn_checkpoints else []),
                *extras,
            ]
            print()
            print(f"=== [seed {s}] {' '.join(cmd)} ===")
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                print(f"  !! seed {s} exited with code {ret.returncode}; continuing")
    else:
        seed_dirs = sorted([p for p in sweep_dir.iterdir()
                            if p.is_dir() and p.name.startswith("seed_")])
        print(f"Found {len(seed_dirs)} existing seed dirs to aggregate")

    summary = _aggregate(seed_dirs)
    has_metrics = bool(summary["vi"] or summary["wm"] or summary["unseen"])
    if not has_metrics:
        print(
            f"\n=== No vi/wm/unseen metrics found under {sweep_dir} — "
            f"skipping aggregate.json + summary.txt ==="
        )
        return
    out_json = sweep_dir / "aggregate.json"
    out_txt = sweep_dir / "summary.txt"
    out_json.write_text(json.dumps(summary, indent=2))
    _write_summary_txt(summary, out_txt)
    print()
    print(f"=== Paper-table summary ({summary['num_seeds_run']} seeds) ===")
    for src, fields in (("vi", ("q_mse", "q_nmse")), ("wm", ("wm_mse", "wm_nmse"))):
        for k in fields:
            v = summary[src].get(k)
            if v:
                print(f"  {k.upper():>8s} = {v['mean']:.1e} ± {v['se']:.1e}  (n={v['n']})")
    for label, entry in summary["unseen"].items():
        t, w = entry.get("r_star"), entry.get("r_wm")
        if t and w:
            print(
                f"  {label:<32}  R* = {t['mean']:.3f} ± {t['se']:.3f}    "
                f"R_WM = {w['mean']:.3f} ± {w['se']:.3f}  (n={w['n']})"
            )
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_txt}")


def main():
    parser = argparse.ArgumentParser(description="Unified experiment runner")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config file, e.g. configs/mountaincar.py",
    )
    parser.add_argument(
        "--phases", type=str, default="pqn,plot_pqn,vi,wm,plot_wm,unseen",
        help="Comma-separated phases: pqn,plot_pqn,vi,wm,plot_wm,unseen",
    )
    parser.add_argument(
        "--num_seeds", type=int, default=1,
        help="Number of seeds to run (default 1, cheap dev). Pass 10 to "
             "reproduce the paper-headline mean ± SE across seeds.",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Override --num_seeds with an explicit comma list, e.g. --seeds 0,3,7.",
    )
    parser.add_argument(
        "--sweep_dir", type=str, default=None,
        help="Multi-seed sweep parent dir. Default: outputs/<env>/seeds_<ts>.",
    )
    parser.add_argument(
        "--aggregate_only", action="store_true",
        help="Multi-seed only: skip running, re-aggregate an existing sweep dir.",
    )
    parser.add_argument("--pqn_checkpoint", type=str, default=None)
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help="Override the output directory. If unset, a fresh timestamped dir is created.",
    )
    parser.add_argument(
        "--save_pqn_checkpoints", action="store_true",
        help="Save intermediate PQN step_*.pkl checkpoints during training "
             "(~1.7 GB per Reacher seed). Off by default; only needed for the "
             "architecture sweep, which has its own checkpoint path.",
    )
    args, extra = parser.parse_known_args()

    # Multi-seed (default) spawns single-seed subprocesses then aggregates.
    if args.seeds is not None or args.num_seeds > 1:
        _run_multi_seed(args, extra)
        return

    # Parse --KEY VALUE override pairs
    override_pairs = []
    it = iter(extra)
    for token in it:
        if token.startswith("--"):
            if "=" in token:
                key, val = token[2:].split("=", 1)
            else:
                key = token[2:]
                try:
                    val = next(it)
                except StopIteration:
                    raise ValueError(f"Missing value for override {token}")
            key = key.upper()
            override_pairs.append((key, val))
        else:
            raise ValueError(f"Unexpected argument: {token}")

    # Allow passing a directory instead of the full .pkl path. Hoisted above
    # config loading so the JSON merge below can use os.path.dirname.
    if args.pqn_checkpoint and os.path.isdir(args.pqn_checkpoint):
        args.pqn_checkpoint = os.path.join(args.pqn_checkpoint, "pqn_checkpoint.pkl")

    cfg = import_config_module(args.config)

    # If --pqn_checkpoint points at a run dir, merge that run's saved
    # pqn_config.json over the Python config defaults BEFORE applying CLI
    # overrides — so precedence is: CLI > saved JSON > Python defaults.
    # This is what makes --phases pqn --pqn_checkpoint <dir> re-train against
    # the saved hyperparameter set.
    _pqn_cfg_search_dir = None
    if args.pqn_checkpoint:
        _pqn_cfg_search_dir = os.path.dirname(args.pqn_checkpoint)
        if os.path.basename(_pqn_cfg_search_dir) == "checkpoints":
            _pqn_cfg_search_dir = os.path.dirname(_pqn_cfg_search_dir)

    # Drop STATE_RANGES (env metadata) from the saved PQN config so the live
    # config wins on post-hoc edits to WM sampler / VI / eval bounds.
    _merge_saved_config(
        _pqn_cfg_search_dir, "pqn_config.json", cfg.get("PQN_CONFIG", {}),
        fixup=lambda d: d.pop("STATE_RANGES", None), name="PQN",
    )

    cfg = apply_overrides(cfg, override_pairs)

    PQN_CONFIG = cfg["PQN_CONFIG"]
    WM_CONFIG  = cfg["WM_CONFIG"]
    ENV_CONFIG = cfg["ENV_CONFIG"]
    goals      = PQN_CONFIG["GOALS"]
    goal_masks = PQN_CONFIG["REWARD_MASK"]
    ENV_NAME   = ENV_CONFIG["ENV_NAME"]

    phases = set(args.phases.split(","))

    from training.pqn_utils import train_or_load_pqn, save_pqn, load_pqn
    from training.wm import save_wm
    from configs.utils import save_config
    from envs.goals import ContinuousGoal

    # ── Create environment ──
    if ENV_NAME == "MountainCar":
        from envs.mountaincar import MountainCar
        basic_env = MountainCar(

            max_steps_in_episode=PQN_CONFIG["MAX_STEPS_IN_EPISODE"],
        )
    elif ENV_NAME == "Reacher":
        from envs.reacher import Reacher
        basic_env = Reacher(
            reward_type=PQN_CONFIG["REWARD_TYPE"],
            sigma=PQN_CONFIG["REWARD_SIGMA"],
            a=PQN_CONFIG["REWARD_A"],
            max_steps_in_episode=PQN_CONFIG["MAX_STEPS_IN_EPISODE"],
            torque_values=cfg["REACHER_TORQUE_VALUES"],

        )
    else:
        raise ValueError(f"Unknown ENV_NAME: {ENV_NAME!r}; expected one of {{MountainCar, Reacher}}.")

    env_params = basic_env.default_params
    all_goals = ContinuousGoal(target_state=goals, reward_mask=goal_masks)

    # Neither env auto-terminates: MountainCar drops gymnax's goal-position
    # termination, Reacher only truncates. Non-None env_terminated_fn comes
    # only from per-unseen-goal forbidden specs (see eval/unseen_goals.py).
    env_terminated_fn = None

    # ── Resolve output directories ──
    # PQN_DIR: parent for PQN checkpoint, config, training plot, VI
    # WM_DIR:  subdirectory of PQN_DIR for WM checkpoint, config, plots
    WM_DIR = None

    if args.pqn_checkpoint and "pqn" not in phases:
        # Checkpoint may be run_dir/pqn_checkpoint.pkl or
        # run_dir/checkpoints/step_*.pkl; strip the trailing /checkpoints.
        PQN_DIR = os.path.dirname(args.pqn_checkpoint)
        if os.path.basename(PQN_DIR) == "checkpoints":
            PQN_DIR = os.path.dirname(PQN_DIR)
    elif args.run_dir is not None:
        # Caller-supplied output dir (used by run_seeds.py to pin per-seed dirs).
        PQN_DIR = args.run_dir
    else:
        # New timestamped dir. Also covers --pqn_checkpoint + --phases pqn:
        # saved JSON sets hyperparams (merged upstream) but the new run writes
        # elsewhere so the source dir stays untouched.
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _her = "her" if PQN_CONFIG["USE_HER"] else "base"
        _rtype = PQN_CONFIG["REWARD_TYPE"]
        _ng = PQN_CONFIG["NUM_GOALS"]
        # Config stem keeps position/velocity MC runs in distinct dirs.
        _config_stem = os.path.splitext(os.path.basename(args.config))[0]
        PQN_DIR = f"outputs/{_config_stem}/{_her}_{_rtype}_L{_ng}_{_ts}"
    os.makedirs(PQN_DIR, exist_ok=True)

    if {"wm", "plot_wm"} & phases:
        _wm_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        WM_DIR = f"{PQN_DIR}/wm_{_wm_ts}"
        os.makedirs(WM_DIR, exist_ok=True)

    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Phases: {', '.join(sorted(phases))}")
    print(f"PQN dir: {PQN_DIR}")
    if WM_DIR:
        print(f"WM dir:  {WM_DIR}")
    print("=" * 60)

    q_params = q_batch_stats = None
    p_params = losses = None

    # ── wandb init (spans PQN through WM; finished after both phases) ──
    _unified_wandb = PQN_CONFIG.get("USE_WANDB") and bool(phases & {"pqn", "wm"})
    if _unified_wandb:
        import wandb
        wandb.init(
            project=PQN_CONFIG["WANDB_PROJECT"],
            entity=PQN_CONFIG["WANDB_ENTITY"],
            config={**PQN_CONFIG, **{f"wm/{k}": v for k, v in WM_CONFIG.items()}},
            name=f"{ENV_NAME}-{int(PQN_CONFIG['TOTAL_TIMESTEPS'] // 1e6)}M",
        )
        # WM logs to wm/* on its own step axis (PQN uses step=update_steps).
        wandb.define_metric("wm/step")
        wandb.define_metric("wm/*", step_metric="wm/step")

    # ── Phase: PQN ──
    pqn_metrics = None
    if "pqn" in phases:
        print("\n── PQN training ──")
        # Intermediate step_*.pkl checkpoints are only needed for the
        # architecture sweep (scripts/architecture_sweep.py runs its own
        # training path). Off by default to save ~1.7 GB per Reacher seed.
        checkpoint_dir = None
        if args.save_pqn_checkpoints:
            checkpoint_dir = f"{PQN_DIR}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
        q_params, q_batch_stats, pqn_metrics = train_or_load_pqn(
            PQN_CONFIG, basic_env, env_params, all_goals, checkpoint_dir=checkpoint_dir,
        )
        save_pqn(q_params, q_batch_stats, f"{PQN_DIR}/pqn_checkpoint.pkl", config=PQN_CONFIG, metrics=pqn_metrics)
        save_config(PQN_CONFIG, f"{PQN_DIR}/pqn_config.json")
    elif args.pqn_checkpoint:
        print(f"\n── Loading PQN from {args.pqn_checkpoint} ──")
        q_params, q_batch_stats, pqn_metrics = load_pqn(args.pqn_checkpoint)
        # Pick up training-time network architecture from the saved config.
        saved_config_path = os.path.join(os.path.dirname(args.pqn_checkpoint), "pqn_config.json")
        if os.path.exists(saved_config_path):
            import json
            with open(saved_config_path) as f:
                saved_config = json.load(f)
            for k in ("GOAL_INPUT_DIMS", "OBS_INPUT_DIMS", "NETWORK_DENSE_HIDDEN_SIZE",
                       "NETWORK_DENSE_LAYERS", "NORM_TYPE",
                       "NETWORK_SIGMOID_OUTPUTS"):
                if k in saved_config:
                    PQN_CONFIG[k] = saved_config[k]
                elif k in PQN_CONFIG and k not in saved_config:
                    del PQN_CONFIG[k]

    # ── Phase: plot_pqn ──
    if "plot_pqn" in phases:
        if pqn_metrics is not None:
            from plotting.pqn import plot_pqn_training
            plot_pqn_training(pqn_metrics, save_path=f"{PQN_DIR}/pqn_training.png")
            print(f"Saved {PQN_DIR}/pqn_training.png")
        else:
            print("\nSkipping PQN plots (no training metrics available)")

    # ── Phase: VI ──
    if "vi" in phases:
        if ENV_CONFIG["VI_GRID_RES"] is None:
            print("\nSkipping VI (VI_GRID_RES not set)")
        else:
            assert q_params is not None, "VI phase requires PQN params (use --phases pqn,vi or --pqn_checkpoint)"
            print("\n── Value iteration ──")
            from eval.value_iteration import run_all_goals
            from envs.env_dynamics import make_env_dynamics_fn
            vi_dir = f"{PQN_DIR}/vi"
            os.makedirs(vi_dir, exist_ok=True)
            # Reacher: VI on the 4D effective state (the 6D obs has redundant
            # fp_xy dims that are FK of the angles — off-manifold elsewhere).
            if ENV_NAME == "Reacher":
                from envs.reacher_utils import (
                    make_reacher_effective_dynamics_fn, reacher_effective_to_obs,
                )
                vi_dynamics_fn = make_reacher_effective_dynamics_fn(basic_env, env_params)
                vi_state_to_obs = reacher_effective_to_obs
                vi_grid_state_dim = ENV_CONFIG["VI_STATE_DIM"]
                vi_grid_state_ranges = ENV_CONFIG["VI_STATE_RANGES"]
                vi_grid_state_labels = ENV_CONFIG["VI_STATE_LABELS"]
            else:
                vi_dynamics_fn = make_env_dynamics_fn(basic_env, env_params, ENV_NAME)
                vi_state_to_obs = None
                vi_grid_state_dim = None
                vi_grid_state_ranges = None
                vi_grid_state_labels = None
            run_all_goals(
                q_params, q_batch_stats, vi_dir, PQN_CONFIG, ENV_CONFIG,
                vi_dynamics_fn,
                env_terminated_fn,
                state_to_obs_fn=vi_state_to_obs,
                grid_state_dim=vi_grid_state_dim,
                grid_state_ranges=vi_grid_state_ranges,
                grid_state_labels=vi_grid_state_labels,
            )

    # ── Env-specific state sampler (Reacher needs fp = FK(θ)). ──
    from envs.samplers import make_reacher_uniform_sampler, make_env_reset_sampler
    env_sample_fn = None
    if ENV_NAME == "Reacher":
        env_sample_fn = make_reacher_uniform_sampler(PQN_CONFIG["STATE_RANGES"])

    # WM sampler precedence: SAMPLE_FROM_RESET > env_sample_fn (uniform).
    wm_sample_fn = env_sample_fn
    if WM_CONFIG.get("SAMPLE_FROM_RESET", False):
        wm_sample_fn = make_env_reset_sampler(basic_env, env_params)

    # WM logs into the unified wandb run created before PQN (above).
    _wm_wandb = PQN_CONFIG.get("USE_WANDB") and "wm" in phases

    # ── Effective-state WM lift functions (visible to wm + plot_wm) ──
    # If WM_OUTPUT_DIM < STATE_DIM the WM predicts in effective space
    # (e.g. Reacher 4D [θ, ω]); env-specific FK appends redundant fp dims
    # for any consumer that expects full obs.
    wm_output_dim = WM_CONFIG.get("WM_OUTPUT_DIM")
    wm_state_to_eff_fn = None
    wm_eff_to_obs_fn = None
    if wm_output_dim is not None and wm_output_dim != PQN_CONFIG["STATE_DIM"]:
        if ENV_NAME == "Reacher":
            from envs.reacher_utils import (
                reacher_obs_to_effective, reacher_effective_to_obs,
            )
            wm_state_to_eff_fn = reacher_obs_to_effective
            wm_eff_to_obs_fn = reacher_effective_to_obs
        else:
            raise ValueError(
                f"WM_OUTPUT_DIM={wm_output_dim} is set but no effective<->obs "
                f"lift functions registered for ENV_NAME={ENV_NAME!r}."
            )

    # ── Phase: WM ──
    if "wm" in phases:
        assert q_params is not None, "WM phase requires PQN params (use --phases pqn,wm or --pqn_checkpoint)"
        print(f"\n── World model training ──")
        save_config(WM_CONFIG, f"{WM_DIR}/wm_config.json")

        from envs.env_dynamics import make_env_dynamics_fn
        wm_dynamics_fn = make_env_dynamics_fn(basic_env, env_params, ENV_NAME)

        from training.wm import train_world_model
        p_params, losses = train_world_model(
            q_params, q_batch_stats, WM_CONFIG, PQN_CONFIG, goals, goal_masks,
            env_terminated_fn, sample_states_fn=wm_sample_fn,
            use_wandb=_wm_wandb,
            state_to_eff_fn=wm_state_to_eff_fn,
            eff_to_obs_fn=wm_eff_to_obs_fn,
            wm_output_dim=wm_output_dim,
        )
        save_wm(p_params, losses, f"{WM_DIR}/wm_checkpoint.pkl", config=WM_CONFIG)

        from plotting.wm import plot_loss_curve
        plot_loss_curve(losses, save_path=f"{WM_DIR}/wm_training.png")

    # Close the unified wandb run after all logging phases (PQN + WM) complete.
    if _unified_wandb:
        import wandb
        wandb.finish()

    # ── Phase: plot_wm ──
    if "plot_wm" in phases and p_params is None:
        print("\nSkipping WM dynamics plots (no WM params — add `wm` to --phases)")
    if "plot_wm" in phases and p_params is not None:
        print("\n── Dynamics comparison plots ──")
        from eval.world_model import compare_dynamics
        from envs.env_dynamics import make_env_dynamics_fn
        compare_dynamics(
            p_params, WM_DIR, WM_CONFIG, ENV_CONFIG,
            make_env_dynamics_fn(basic_env, env_params, ENV_NAME),
            goals=goals, goal_masks=goal_masks, losses=losses,
            plots=True, sample_states_fn=wm_sample_fn,
            state_to_eff_fn=wm_state_to_eff_fn,
            eff_to_obs_fn=wm_eff_to_obs_fn,
            wm_output_dim=wm_output_dim,
        )

    # ── Phase: unseen ──
    if "unseen" in phases:
        if p_params is None:
            print("\nSkipping unseen-goal eval (no WM params — add `wm` to --phases)")
        elif not ENV_CONFIG.get("UNSEEN_GOALS"):
            print("\nNo UNSEEN_GOALS configured in ENV_CONFIG — skipping.")
        else:
            print("\n── Unseen-goal evaluation ──")
            from eval.unseen_goals import evaluate_unseen_goals
            from envs.env_dynamics import make_env_dynamics_fn

            # Sample n_eval_starts fresh env.reset() obs as evaluation starts.
            # Reacher.reset() gives random joint configurations (broad);
            # MountainCar.reset() gives the narrow init region (pos in
            # [-0.6, -0.4], vel=0).
            n_eval_starts = int(
                ENV_CONFIG.get("UNSEEN_NUM_EVAL_STARTS", 512)
            )
            _reset_keys = jax.random.split(
                jax.random.PRNGKey(0), n_eval_starts,
            )
            eval_starts_obs = jax.block_until_ready(jax.vmap(
                lambda k: basic_env.reset(k, env_params)[0]
            )(_reset_keys))
            unseen_kwargs = {"eval_starts": eval_starts_obs}

            # When the WM was trained in an effective state space smaller than
            # the obs (Reacher with WM_OUTPUT_DIM=4), VI on the full obs-space
            # grid is intractable AND the WM Q-net would be built with the
            # wrong output dim. Use effective-state dynamics fns and pass the
            # 4D grid + lift/project fns to evaluate_unseen_goals.
            if (wm_output_dim is not None
                    and wm_output_dim != PQN_CONFIG["STATE_DIM"]):
                if ENV_NAME == "Reacher":
                    from envs.reacher_utils import (
                        make_reacher_effective_dynamics_fn,
                        make_reacher_wm_dynamics_fn,
                        reacher_effective_to_obs,
                        reacher_obs_to_effective,
                    )
                    true_dyn_fn = make_reacher_effective_dynamics_fn(
                        basic_env, env_params,
                    )
                    wm_dyn_fn = make_reacher_wm_dynamics_fn(
                        p_params, WM_CONFIG, PQN_CONFIG["ACTION_DIM"],
                    )
                    unseen_kwargs.update(
                        wm_dynamics_fn=wm_dyn_fn,
                        grid_state_dim=ENV_CONFIG["VI_STATE_DIM"],
                        grid_state_ranges=ENV_CONFIG["VI_STATE_RANGES"],
                        state_to_obs_fn=reacher_effective_to_obs,
                        obs_to_grid_fn=reacher_obs_to_effective,
                    )
                    dynamics_fn_unseen = true_dyn_fn
                else:
                    raise ValueError(
                        f"WM_OUTPUT_DIM={wm_output_dim} but no effective-state "
                        f"unseen-goal wiring registered for ENV_NAME={ENV_NAME!r}"
                    )
            else:
                dynamics_fn_unseen = make_env_dynamics_fn(
                    basic_env, env_params, ENV_NAME,
                )

            evaluate_unseen_goals(
                p_params, WM_CONFIG, PQN_CONFIG, ENV_CONFIG,
                dynamics_fn_unseen,
                env_terminated_fn,
                unseen_goals=ENV_CONFIG["UNSEEN_GOALS"],
                out_dir=f"{WM_DIR}/unseen_goals",
                # Single-cell deep-dive — keep per-goal artifacts (default
                # flips to skip for arch-sweep performance).
                skip_per_goal_artifacts=False,
                **unseen_kwargs,
            )

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
