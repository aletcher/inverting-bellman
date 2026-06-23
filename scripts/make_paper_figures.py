"""Reproduce paper figures end-to-end from a single command.

Trains the three headline sweeps (Reacher, MountainCar-position,
MountainCar-velocity) at the requested seed count, then renders Fig 2,
Fig 3 and Fig 5 from those sweeps and writes consolidated headline
numbers (Q_MSE/NMSE, WM_MSE/NMSE, unseen-goal returns) to
outputs/results/headlines.txt. Each figure script is invoked with
explicit --*_sweep paths, so no figure step retrains.

Fig 4 (the architecture sweep, ~35 GPU-hours on H100) is skipped by
default. Pass --include_arch_sweep to include it.

Usage:
    # Cheap reproduction: 1 seed per env (~minutes).
    uv run python scripts/make_paper_figures.py

    # Paper headline: 10 seeds per env.
    uv run python scripts/make_paper_figures.py --num_seeds 10

    # Regenerate plots + headlines from existing sweeps (no retraining).
    uv run python scripts/make_paper_figures.py \\
        --reacher_sweep      outputs/reacher/seeds_<TS> \\
        --mc_position_sweep  outputs/mountaincar-position/seeds_<TS> \\
        --mc_velocity_sweep  outputs/mountaincar-velocity/seeds_<TS>

    # Include the architecture sweep (Fig 4).
    uv run python scripts/make_paper_figures.py --num_seeds 10 --include_arch_sweep
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd):
    print(f"\n=== {' '.join(cmd)} ===")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise RuntimeError(f"command failed (exit {ret.returncode}): {' '.join(cmd)}")


_CROSS_WM_RE = re.compile(
    r"^Avg (\S[^:]*): .*NMSE=([0-9.eE+\-]+)(?:\s*±\s*([0-9.eE+\-]+))?",
    re.M,
)


def _parse_cross_wm(path):
    """Read a fig5-style metrics file; return list of (name, nmse, nmse_se).

    nmse_se is 0.0 when the file came from a single-seed run (no ± token).
    """
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        txt = f.read()
    out = []
    for m in _CROSS_WM_RE.finditer(txt):
        name = m.group(1).strip()
        nmse = float(m.group(2))
        nmse_se = float(m.group(3)) if m.group(3) else 0.0
        out.append((name, nmse, nmse_se))
    return out


def _write_headlines(envs, out_path):
    """Consolidate per-env aggregate.json into a single human-readable file.

    envs: list of (env_label, sweep_dir, cross_wm_path | None) tuples, in
    display order. cross_wm_path, when provided, points at a metrics file
    written by scripts/mountaincar_wm_quiver.py — its Avg NMSE lines get
    appended under the env block as the "Cross-WM" summary.
    """
    lines = [
        "Paper headlines.",
        "All numbers are mean ± SE across seeds.",
        "",
    ]
    for label, sweep_dir, cross_wm_path in envs:
        agg_path = os.path.join(sweep_dir, "aggregate.json")
        if not os.path.exists(agg_path):
            lines.append(f"{label}")
            lines.append("  (aggregate.json missing — no headline numbers)")
            lines.append("")
            continue
        with open(agg_path) as f:
            agg = json.load(f)
        n_seeds = agg.get("num_seeds_run")
        seed_suffix = (
            f" ({n_seeds} seed{'s' if n_seeds != 1 else ''})"
            if n_seeds is not None else ""
        )
        lines.append(f"{label}{seed_suffix}")
        vi = agg.get("vi") or {}
        wm = agg.get("wm") or {}
        unseen = agg.get("unseen") or {}
        if not (vi or wm or unseen):
            lines.append("  (no metrics aggregated)")
            lines.append("")
            continue
        if vi:
            q_mse = vi.get("q_mse")
            q_nmse = vi.get("q_nmse")
            if q_mse:
                lines.append(f"  Q_MSE   = {q_mse['mean']:.1e} ± {q_mse['se']:.1e}")
            if q_nmse:
                lines.append(f"  Q_NMSE  = {q_nmse['mean']:.1e} ± {q_nmse['se']:.1e}")
        if wm:
            wm_mse = wm.get("wm_mse")
            wm_nmse = wm.get("wm_nmse")
            if wm_mse:
                lines.append(f"  WM_MSE  = {wm_mse['mean']:.1e} ± {wm_mse['se']:.1e}")
            if wm_nmse:
                lines.append(f"  WM_NMSE = {wm_nmse['mean']:.1e} ± {wm_nmse['se']:.1e}")
        if unseen:
            lines.append("")
            lines.append(f"  {'Unseen goal':<20}{'R*':>20}{'R_WM':>20}")
            for goal_label, entry in unseen.items():
                r_star = entry.get("r_star")
                r_wm = entry.get("r_wm")
                if r_star and r_wm:
                    lines.append(
                        f"  {goal_label:<20}"
                        f"{r_star['mean']:>10.3f} ± {r_star['se']:.3f}    "
                        f"{r_wm['mean']:>7.3f} ± {r_wm['se']:.3f}"
                    )
        cross_wm = _parse_cross_wm(cross_wm_path)
        if cross_wm:
            lines.append("")
            lines.append(f"  Cross-WM ({label}):")
            name_w = max(len(n) for n, _, _ in cross_wm)
            any_se = any(se > 0 for _, _, se in cross_wm)
            for name, nmse, nmse_se in cross_wm:
                val = (f"{nmse:.1e} ± {nmse_se:.1e}" if any_se
                       else f"{nmse:.1e}")
                lines.append(f"    {name:<{name_w}} : NMSE = {val}")
        lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"\nWrote {out_path}")


def _train_sweep(config_stem, num_seeds, phases):
    """Run a headline sweep; return the sweep dir."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(_REPO_ROOT, "outputs", config_stem, f"seeds_{ts}")
    seeds_csv = ",".join(str(i) for i in range(num_seeds))
    _run([
        sys.executable, os.path.join(_REPO_ROOT, "run.py"),
        "--config", f"configs/{config_stem}.py",
        "--phases", phases,
        "--seeds", seeds_csv,
        "--sweep_dir", sweep_dir,
    ])
    return sweep_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num_seeds", type=int, default=1,
                    help="Seeds per env when auto-training. Default 1 (cheap); "
                         "10 matches the paper headlines. Ignored for envs "
                         "whose --*_sweep is passed.")
    ap.add_argument("--reacher_sweep", default=None,
                    help="Existing Reacher sweeps_<TS> dir. If passed, skip "
                         "Reacher training and reuse it.")
    ap.add_argument("--mc_position_sweep", default=None,
                    help="Existing mountaincar-position sweeps_<TS> dir. If "
                         "passed, skip MC-position training and reuse it.")
    ap.add_argument("--mc_velocity_sweep", default=None,
                    help="Existing mountaincar-velocity sweeps_<TS> dir. If "
                         "passed, skip MC-velocity training and reuse it.")
    ap.add_argument("--include_arch_sweep", action="store_true",
                    help="Also run scripts/architecture_sweep.py to produce "
                         "Fig 4 (~35 GPU-hours).")
    args = ap.parse_args()

    # Reacher + MC-position feed Fig 2/3 (need WM + unseen) and the
    # consolidated headline table (needs vi + plot_wm to produce the
    # Q_MSE/NMSE + WM_MSE/NMSE numbers in summary.txt).
    headline_phases = "pqn,vi,wm,plot_wm,unseen"
    if all(s is not None for s in
           (args.reacher_sweep, args.mc_position_sweep, args.mc_velocity_sweep)):
        print(f"\n── Reusing existing sweeps (no training) ──")
    else:
        print(f"\n── Headlines (training missing sweeps) ──")
    reacher_sweep = args.reacher_sweep or _train_sweep(
        "reacher", args.num_seeds, headline_phases)
    mc_pos_sweep = args.mc_position_sweep or _train_sweep(
        "mountaincar-position", args.num_seeds, headline_phases)
    # MC-velocity feeds Fig 5 only. run.py skips aggregate.json/summary.txt
    # when no vi/wm/unseen metrics exist, so this sweep produces just the
    # per-seed PQN + WM checkpoints with no top-level summary files.
    mc_vel_sweep = args.mc_velocity_sweep or _train_sweep(
        "mountaincar-velocity", args.num_seeds, "pqn,wm")

    print(f"\n── Fig 2 (Reacher dynamics + rollouts) ──")
    _run([sys.executable,
          os.path.join(_REPO_ROOT, "scripts/reacher_visualisation.py"),
          "--sweep_dir", reacher_sweep])

    print(f"\n── Fig 3 (Unseen-goal bars) ──")
    _run([sys.executable,
          os.path.join(_REPO_ROOT, "scripts/unseen_tasks_bar.py"),
          "--reacher_sweep", reacher_sweep,
          "--mc_sweep", mc_pos_sweep,
          "--num_seeds", str(args.num_seeds)])

    print(f"\n── Fig 5 (MountainCar quiver) ──")
    _run([sys.executable,
          os.path.join(_REPO_ROOT, "scripts/mountaincar_wm_quiver.py"),
          "--mc_position_sweep", mc_pos_sweep,
          "--mc_velocity_sweep", mc_vel_sweep])

    if args.include_arch_sweep:
        print(f"\n── Fig 4 (Reacher arch sweep) ──")
        _run([sys.executable,
              os.path.join(_REPO_ROOT, "scripts/architecture_sweep.py"),
              "--config", "configs/reacher.py",
              "--num_seeds", str(args.num_seeds)])
    else:
        print(f"\n── Fig 4 skipped (pass --include_arch_sweep to include) ──")

    print(f"\n── Consolidated headlines ──")
    fig5_metrics_path = os.path.join(
        _REPO_ROOT, "outputs", "results", "fig5_mountaincar_quiver.txt"
    )
    _write_headlines(
        envs=[
            ("Reacher", reacher_sweep, None),
            ("MountainCar", mc_pos_sweep, fig5_metrics_path),
        ],
        out_path=os.path.join(_REPO_ROOT, "outputs", "results", "headlines.txt"),
    )

    print(f"\nResults written to {os.path.join(_REPO_ROOT, 'outputs', 'results')}/")


if __name__ == "__main__":
    main()
