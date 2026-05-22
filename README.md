# inverting-bellman

Code for the paper **Inverting the Bellman Equation: From Q-Values to World Models**.

We introduce **P-learning**: extracting the implicit world model of a goal-conditioned agent's `Q`-values, effectively inverting the Bellman equation. We train PQN agents on a small set of sparse goals, then fit a world model `P_φ` that makes the agent's `Q`-values self-consistent under its policy. We show that the recovered WM is extremely accurate, and can be used to plan for unseen goals far beyond the training distribution. This repo reproduces the two continuous-control experiments in the paper:

- **Reacher** (§5.1, §E.2) — discrete-action 9-action Reacher trained on 4 cardinal-direction goals.
- **MountainCar** (§5.2) — position- and velocity-conditioned variants; WM extraction + cross-WM comparison.

---

## Install

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync                  # CPU (local)
uv sync --extra cuda     # GPU (CUDA 12 / H100)
```

All commands below should be prefixed with `uv run` (or activate the env: `source .venv/bin/activate`).
Wandb logging is off by default. To enable: `wandb login` once, then pass `--USE_WANDB True` to any `run.py` invocation (or set `USE_WANDB = True` at the top of the relevant config). Per-run metrics + summary tables are written to disk regardless of wandb state, so the paper headlines are reproducible without a wandb account.

---


## Repository layout

```
run.py                    Main entry point; --num_seeds N for multi-seed
configs/                  Per-experiment hyperparameters (Reacher, MountainCar)
envs/                     Reacher + MountainCar envs, utils
training/                 PQN trainer (pqn.py), WM trainer (wm.py)
eval/                     WM-MSE, Q-MSE using value iteration (VI), performance on unseen goals
plotting/                 Per-phase plots: pqn.py (training curves), vi.py, wm.py
scripts/                  Scripts to reproduce Fig. 2, Fig. 3, Fig. 4 and Fig. 5 (including arch. sweep)
```

---

## Reproducing the paper

Every reported result / figure in the paper is produced by a one-line command. `run.py` defaults to a 10-seed run and aggregates the headline metrics into `aggregate.json` (per-seed JSON) + `summary.txt` (human-readable). Pass `--num_seeds 1` for a single-seed dev run.

### §5.1 — Reacher headline (Figure 3 + inline Q / WM numbers)

```bash
uv run python run.py --config configs/reacher.py
```

Default architecture is the paper-headline D=6 / W=2048. Writes per-seed dirs under `outputs/reacher/seeds_<TS>/seed_<i>/` and a sweep-level `summary.txt` + `aggregate.json`.

Expected (10-seed mean ± SE on H100):


|         | value           | source              |
| ------- | --------------- | ------------------- |
| Q_MSE   | 1.9e-2 ± 1.2e-3 | `vi/vi_summary.txt` |
| Q_MAPE  | 0.22 ± 0.01     | `vi/vi_summary.txt` |
| WM_MSE  | 9.1e-5 ± 3.8e-6 | `wm_*/results.txt`  |
| WM_MAPE | 0.07 ± 0.001    | `wm_*/results.txt`  |


Q metrics compare V_pqn / Q_pqn against the *true* value function V^π / Q^π under
PQN's own greedy policy (policy evaluation on the VI grid). Captures PQN's
Bellman self-consistency error — see §5.1 "imperfect Q-values".

Unseen-goal discounted returns (paper Figure 3):


| goal            | R             | R_WM          |
| --------------- | ------------- | ------------- |
| Far fingertip   | 0.712 ± 0.000 | 0.711 ± 0.000 |
| Target angle    | 0.687 ± 0.000 | 0.682 ± 0.001 |
| Target velocity | 0.651 ± 0.000 | 0.651 ± 0.000 |


### §5.2 — MountainCar headline (Figure 3 + inline Q / WM numbers)

```bash
uv run python run.py --config configs/mountaincar-position.py
```

Expected (paper §5.2):


|        | mean   |
| ------ | ------ |
| Q_MSE  | 1.4e-2 |
| WM_MSE | 7.9e-5 |


Unseen-goal discounted returns:


| goal          | R           | R_WM        |
| ------------- | ----------- | ----------- |
| Fast car      | 0.37 ± 0.03 | 0.29 ± 0.02 |
| Gentle car    | 0.62 ± 0.04 | 0.52 ± 0.12 |
| Shortest path | 0.37 ± 0.02 | 0.24 ± 0.11 |

### Figure 2 — Reacher dynamics heatmap + trajectory rollouts

Uses any seed dir from the Reacher headline run above (default config is the paper-headline D=6 / W=2048 architecture).

```bash
uv run python scripts/reacher_dynamics_and_rollouts.py \
    --run_dir outputs/reacher/seeds_<TS>/seed_0 \
    --wm_subdir wm_<TS> \
    --config reacher
# Writes (into outputs/figures/):
#   fig2_reacher_dynamics_heatmap_x_theta.png  — fig. 2 left (next-x)
#   fig2_reacher_dynamics_heatmap_y_theta.png  — fig. 2 left (next-y)
#   fig2_reacher_trajectory_policy.png         — fig. 2 right two panels
```

### Figure 3 — Unseen-goal bar chart (Reacher + MountainCar)

Renders the side-by-side bar chart that compares $R^{\star}_\gamma$ (optimal-policy discounted return) against $R^{\mathrm{WM}}_\gamma$ (WM-derived-policy discounted return) on the three OOD goals per environment. Run **after** the §5.1 and §5.2 headline runs have completed (auto-discovers the most recent `seeds_<TS>/` under each env dir):

```bash
uv run python scripts/unseen_tasks_bar.py
# Writes outputs/figures/fig3_unseen_tasks_combined.png
```

Bars are aggregated across all 10 seeds and 512 eval starts per seed:

- **Bar height (R^WM)** = mean over all (10 seeds × 512 eval starts) = 5120 returns.
- **Error bar (R^WM)** = pooled std across those 5120 returns, computed via the law of total variance ($\mathrm{Var}(X) = E_{\mathrm{seed}}[\mathrm{Var}(X|\mathrm{seed})] + \mathrm{Var}_{\mathrm{seed}}(E[X|\mathrm{seed}])$).
- **Bar height (R★)** = mean over the 512 eval starts; **error bar (R★)** = std over those 512.

To override the auto-discovery, pass explicit sweep paths:

```bash
uv run python scripts/unseen_tasks_bar.py \
    --reacher_sweep outputs/reacher/seeds_<TS_reacher> \
    --mc_sweep      outputs/mountaincar-position/seeds_<TS_mc>
```

### Figure 5 — MountainCar quiver (position- vs velocity-trained WMs)

```bash
# 1. Train velocity-conditioned agent + extract its WM.
uv run python run.py --config configs/mountaincar-velocity.py --num_seeds 1

# 2. Compare against any position-trained WM from the §5.2 run above.
uv run python scripts/mountaincar_wm_quiver.py \
    --wm_a outputs/mountaincar-position/seeds_<TS_pos>/seed_0/wm_<TS> \
    --wm_b outputs/mountaincar-velocity/<her_..._TS_vel>/wm_<TS> \
    --action_idx 2
# Writes:
#   outputs/figures/fig5_mountaincar_quiver.png             — fig. 5
#   outputs/mountaincar/quiver_compare/results.txt          — pairwise MSE/MAPE table
```

### Figure 4 — Reacher architecture sweep

> ⚠️ **Expensive: ~35 GPU-hours on a single H100.** This is the only paper artifact that
> requires the full architecture sweep (42 archs × 10 seeds = 420 PQN runs + WM tracking
> on each). The headline numbers in §5.1 do not depend on this — run only if you want
> to reproduce Figure 4 + Table 2.

```bash
uv run python scripts/architecture_sweep.py --config configs/reacher.py
```

Aggregation runs automatically at the end of the sweep. It writes the paper
figure into `outputs/figures/` and the auxiliary artifacts (per-axis slice
plots + per-run metrics) into the sweep's aggregate subdir:

- `outputs/figures/fig4_arch_sweep_combined_three_panel.png` — **Figure 4**.
- `outputs/reacher/sweep_arch_<TS>/aggregate/pqn_return_*.png`, `wm_dyn_mse_*.png` — per-axis slice plots.
- `outputs/reacher/sweep_arch_<TS>/aggregate/per_run_metrics.npz` + correlation summary — Table 2 numbers.

---

## Per-figure / per-table map


| Paper artifact                        | Reproduction command                                                           | Time (H100) |
| ------------------------------------- | ------------------------------------------------------------------------------ | ----------- |
| Reacher headline       | `run.py --config configs/reacher.py`                                           | ~50 min     |
| MountainCar   | `run.py --config configs/mountaincar-position.py`                              | ~20 min     |
| Fig. 2 (Reacher dyn heatmap + traj)   | `scripts/reacher_dynamics_and_rollouts.py --run_dir <seed_dir>`                | ~2 min      |
| Fig. 3 (rendered bar chart)         | `scripts/unseen_tasks_bar.py` (after both headline runs)                       | ~5 s        |
| Fig. 5 (MountainCar quiver)           | `scripts/mountaincar_wm_quiver.py` (after position + velocity training)        | ~3 min      |
| Table 2 + Fig. 4 (Reacher arch sweep) | `scripts/architecture_sweep.py --config configs/reacher.py` → auto-aggregation | ~35 h       |


---

## Running a subset of phases

A single `run.py` invocation runs the pipeline `pqn → vi → wm → unseen`. To re-run only a subset:

```bash
# Re-evaluate unseen goals on an already-trained seed (uses cached PQN + WM).
uv run python run.py --config configs/reacher.py --num_seeds 1 \
    --phases unseen --run_dir <existing seed dir>

# Re-train the WM (and downstream plots + unseen eval) from a saved PQN checkpoint.
# --pqn_checkpoint accepts either the seed dir or the .pkl path directly.
uv run python run.py --config configs/mountaincar-position.py --num_seeds 1 \
    --phases wm,plot_wm,unseen \
    --pqn_checkpoint outputs/mountaincar/seeds_<TS>/seed_4
```

A fresh `wm_<TS>/` subdir is created inside the seed dir; the old `wm_*/` is left intact.

---

## Picking specific seeds / resuming a partial sweep

`--seeds 0,1,2` overrides `--num_seeds` with an explicit list, and `--sweep_dir` reuses an existing parent so new seeds land alongside the old ones. `--aggregate_only` skips training and re-runs the cross-seed aggregator over every `seed_*` dir it finds.

```bash
# Round 1: seeds 0-4.
uv run python run.py --config configs/mountaincar-position.py \
    --seeds 0,1,2,3,4 \
    --sweep_dir outputs/mountaincar/my_sweep

# Round 2: seeds 5-9 into the same parent dir.
uv run python run.py --config configs/mountaincar-position.py \
    --seeds 5,6,7,8,9 \
    --sweep_dir outputs/mountaincar/my_sweep

# Re-aggregate over all 10 seed_* dirs (only writes summary.txt + aggregate.json;
# does NOT re-train). The in-line aggregation at the end of round 2 only saw
# seeds 5-9, so this final pass is what makes the headline span all 10.
uv run python run.py --config configs/mountaincar-position.py \
    --aggregate_only \
    --sweep_dir outputs/mountaincar/my_sweep
```

The same `--seeds / --sweep_dir / --aggregate_only` pattern fans the work out across machines (run different `--seeds` slices on each, sharing `--sweep_dir`, then aggregate once).

---

## Expected runtimes (H100)

- **PQN training**: < 5 min per Reacher run (1e7 env steps), < 2 min per MountainCar (5e6 env steps).
- **P-learning**: < 10 s for Reacher / MountainCar.
- **10-seed headline runs**: ~50 min Reacher, ~20 min MountainCar (sequential).
- **Arch sweep**: ~35 GPU-hours total for the full 6×7×10 grid; per-cell ~5 min PQN + ~3 min WM tracking.

---

## Citation

```bibtex
@article{letcher2026inverting,
  title   = {Inverting the Bellman Equation: From Q-Values to World Models},
  author  = {Letcher, Alistair and Fellows, Mattie and Goldie, Alexander
             and Richens, Jonathan and Foerster, Jakob and Richardson, Oliver},
  journal = {Preprint},
  year    = {2026},
}
```

Both environments wrap `[gymnax](https://github.com/RobertTLange/gymnax)` implementations. Cite gymnax as:

```bibtex
@misc{gymnax2022,
  author = {Lange, Robert Tjarko},
  title  = {{gymnax}: A {JAX}-based Reinforcement Learning Environment Library},
  url    = {https://github.com/RobertTLange/gymnax},
  year   = {2022},
}
```

License: see [LICENSE](LICENSE).