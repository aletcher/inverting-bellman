"""Bar plot comparing optimal vs WM-derived policy returns on unseen goals.

Reads the 10-seed Reacher and MountainCar sweeps produced by
run.py --config configs/{reacher,mountaincar-position}.py and renders a
side-by-side bar chart of the discounted returns under the optimal (R*) and
WM-derived (R^WM) policies for the three unseen goals in each environment.

Data sources (parsed live):
  outputs/<env>/seeds_<TS>/seed_<i>/wm_*/unseen_goals/<goal>/results.txt
      → per-seed mean ± std of disc return across the 512 eval starts.

For each (env, goal) bar:
  - Height = mean over the 10 seeds of the per-seed mean.
  - Error bar = pooled std across all 10 × 512 = 5120 returns, computed
    via the law of total variance:
        Var(X) = E_seed[Var(X | seed)] + Var_seed(E[X | seed]).
    This matches the "spread of return across (start, seed) draws"
    interpretation, the same statistical type as R*'s own std-over-starts.

Outputs:
    outputs/figures/fig3_unseen_tasks_combined.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ── Reacher / MountainCar goal labels + display names ─────────────────────────
REACHER_GOALS = [
    ("Far_fingertip",   "Fingertip"),
    ("Target_angle",    "Angle"),
    ("Target_velocity", "Velocity"),
]
MC_GOALS = [
    ("Fast_car",      "Fast"),
    ("Gentle_car",    "Gentle"),
    ("Shortest_path", "Path"),
]


# ── Parser for the per-seed results.txt ───────────────────────────────────────
_DISC_LINE = re.compile(
    r"^\s*(True|WM) policy:\s+([0-9.eE+-]+)\s*±\s*([0-9.eE+-]+)\s*$"
)


def _parse_results(path: str) -> tuple[float, float, float, float]:
    """Return (R_star_mean, R_star_std, R_wm_mean, R_wm_std) parsed from the
    "Discounted returns" block of a per-seed results.txt."""
    in_disc = False
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("Discounted returns"):
                in_disc = True
                continue
            if in_disc:
                m = _DISC_LINE.match(line)
                if m:
                    out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
                    if {"True", "WM"} <= out.keys():
                        break
    if {"True", "WM"} != out.keys():
        raise ValueError(f"Did not find both True/WM disc returns in {path}")
    return (*out["True"], *out["WM"])


# ── Sweep aggregation ────────────────────────────────────────────────────────
def _auto_discover_sweep(env_dir: str) -> str:
    """Most recent seeds_<TS> dir under outputs/<env>/. Raises if none."""
    candidates = sorted(glob.glob(os.path.join(env_dir, "seeds_*")))
    if not candidates:
        raise FileNotFoundError(
            f"No seeds_<TS> directory under {env_dir}. "
            f"Did you run: uv run python run.py --config configs/...  ?"
        )
    return candidates[-1]


def _aggregate_env(env_name: str, sweep_dir: str, goals, n_seeds: int = 10):
    """For each (display_goal): collect per-seed (R*, R^WM) (mean, std) over
    the 512 eval starts, return the row used by the bar plot."""
    rows = []
    for goal_dir, display in goals:
        per_seed = []
        for s in range(n_seeds):
            wm_dirs = sorted(glob.glob(
                os.path.join(sweep_dir, f"seed_{s}", "wm_*")
            ))
            if not wm_dirs:
                raise FileNotFoundError(
                    f"No wm_* dir under {sweep_dir}/seed_{s}/")
            # If multiple WM dirs (e.g. user re-trained), take the latest by name.
            results_txt = os.path.join(
                wm_dirs[-1], "unseen_goals", goal_dir, "results.txt"
            )
            per_seed.append(_parse_results(results_txt))
        per_seed = np.asarray(per_seed)   # shape (n_seeds, 4)
        r_star_means = per_seed[:, 0]
        r_star_stds  = per_seed[:, 1]
        r_wm_means   = per_seed[:, 2]
        r_wm_stds    = per_seed[:, 3]

        # R* is cell-independent (oracle) — per-seed values are identical for
        # this single-architecture sweep. Use seed-0 as the reference; the
        # pooled-var formula would give the same answer.
        R_star = float(r_star_means[0])
        R_star_std = float(r_star_stds[0])
        # R^WM pooled std across all (n_seeds × 512) returns.
        R_wm = float(r_wm_means.mean())
        pooled_var = (r_wm_stds ** 2).mean() + r_wm_means.var(ddof=1)
        R_wm_std = float(np.sqrt(pooled_var))
        rows.append((env_name, display, R_star, R_star_std, R_wm, R_wm_std))
    return rows


# ── Style + plot helpers ─────────────────────────────────────────────────────
def _setup_style():
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    plt.rcParams.update({
        "text.usetex": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


LABELS = {"R_star": r"$R^{\star}_{\gamma}$",
          "R_wm":   r"$R^{\mathrm{WM}}_{\gamma}$"}

_FLARE_R = sns.color_palette("flare_r", as_cmap=True)
COLORS = {
    "R_star": mcolors.to_hex(_FLARE_R(0.0)),   # deep purple
    "R_wm":   mcolors.to_hex(_FLARE_R(0.8)),   # pale apricot
}


def _plot_env(ax, df, title, title_x=0.5,
              show_ylabel=True, show_legend=False):
    n = len(df)
    x = np.arange(n)
    w = 0.38

    ax.bar(x - w / 2, df["R_star"], w, yerr=df["R_star_std"],
           color=COLORS["R_star"], label=LABELS["R_star"],
           capsize=2.5, error_kw={"elinewidth": 1.1, "ecolor": "black"},
           edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, df["R_wm"], w, yerr=df["R_wm_std"],
           color=COLORS["R_wm"], label=LABELS["R_wm"],
           capsize=2.5, error_kw={"elinewidth": 1.1, "ecolor": "black"},
           edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(df["goal"].values)
    ax.set_title(title, pad=4, fontweight="normal", x=title_x)
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75])
    ax.tick_params(axis="x", pad=1, length=0)
    ax.tick_params(axis="y", pad=1, length=2)
    ax.grid(axis="y", alpha=0.35, linewidth=0.5)
    ax.grid(axis="x", visible=False)
    if show_ylabel:
        ax.set_ylabel("Discounted return")
    if show_legend:
        ax.legend(loc="upper right", ncol=2, handlelength=1.1,
                  handletextpad=0.4, columnspacing=0.9,
                  borderaxespad=0.2, bbox_to_anchor=(1.0, 1.02))


def plot_combined(data: pd.DataFrame, out_path: str):
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2),
                             gridspec_kw={"wspace": 0.22})
    reacher = data[data["env"] == "Reacher"].reset_index(drop=True)
    mc      = data[data["env"] == "MountainCar"].reset_index(drop=True)
    _plot_env(axes[0], reacher, "Reacher",
              title_x=0.40, show_ylabel=True)
    _plot_env(axes[1], mc, "MountainCar",
              title_x=0.55, show_ylabel=False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["R_star"], label=LABELS["R_star"]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["R_wm"],   label=LABELS["R_wm"]),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.03), frameon=False,
               handlelength=1.2, handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(here)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reacher_sweep", default=None,
        help="seeds_<TS> dir for Reacher. Default: most recent under "
             "outputs/reacher/.")
    p.add_argument(
        "--mc_sweep", default=None,
        help="seeds_<TS> dir for MountainCar-position. Default: most recent "
             "under outputs/mountaincar-position/.")
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument(
        "--out_dir", default="outputs/figures",
        help="Where to write the rendered figure.")
    p.add_argument("--out_stem", default="fig3_unseen_tasks_combined")
    args = p.parse_args()

    reacher_sweep = args.reacher_sweep or _auto_discover_sweep("outputs/reacher")
    mc_sweep      = args.mc_sweep      or _auto_discover_sweep("outputs/mountaincar-position")
    print(f"Reacher sweep : {reacher_sweep}")
    print(f"MC      sweep : {mc_sweep}")

    rows = (
        _aggregate_env("Reacher",     reacher_sweep, REACHER_GOALS, args.n_seeds)
        + _aggregate_env("MountainCar", mc_sweep,      MC_GOALS,      args.n_seeds)
    )
    data = pd.DataFrame(rows, columns=[
        "env", "goal", "R_star", "R_star_std", "R_wm", "R_wm_std",
    ])

    print("\n── Aggregated values ──")
    for r in rows:
        print(f"  {r[0]:>12s}  {r[1]:>10s}: "
              f"R*={r[2]:.3f}±{r[3]:.3f}   R_WM={r[4]:.3f}±{r[5]:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.out_stem}.png")
    plot_combined(data, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
