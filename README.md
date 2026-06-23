# Inverting the Bellman Equation: From Q-Values to World Models

Codebase for the paper [Inverting the Bellman Equation: From Q-Values to World Models](https://arxiv.org/pdf/2606.21173) using a JAX implementation of **P-learning**: extracting the world model contained in an agent's Q-values.

The base script `run.py` trains a PQN agent on a small set of sparse goals in [Reacher](https://gymnasium.farama.org/environments/mujoco/reacher/) and [MountainCar](https://gymnasium.farama.org/environments/classic_control/mountain_car/), then fits a world model `P_φ` by sampling from the agent's Q-values and reward functions. The recovered WM matches the true transition kernel with high fidelity, and can be used to plan for out-of-distribution goals.

---

## Install

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync                  # CPU (local)
uv sync --extra cuda     # GPU (CUDA 12 / H100)
```

All commands below should be prefixed with `uv run`, or the env should be activated beforehand: `source .venv/bin/activate`. Wandb logging is off by default. To enable: `wandb login` once, then pass `--USE_WANDB True` to any `run.py` invocation. Per-run metrics + summary tables are written to disk regardless of wandb state, so the paper results are reproducible without a wandb account.

---

## Repository layout

```
run.py                          Per-env entry point (PQN → VI → WM → unseen goals, multi-seed)
scripts/make_paper_figures.py   Top-level orchestrator: trains all envs + renders Figs 2/3/5
scripts/<fig>.py                Individual figure scripts (re-render a single fig from a sweep)
configs/                        Per-experiment hyperparameters (Reacher, MountainCar)
envs/                           Reacher + MountainCar envs, utils
training/                       PQN trainer (pqn.py), WM trainer (wm.py)
eval/                           Q/WM accuracy via value iteration, unseen-goal eval
plotting/                       Per-phase plotting helpers (pqn.py, vi.py, wm.py)
outputs/                        All training + figure outputs (gitignored)
  ├── <env>/seeds_<TS>/         Per-env multi-seed sweeps (PQN + WM checkpoints, summary.txt)
  └── results/                  Final paper artifacts (Fig 2/3/4/5 + headlines.txt)
```

---

## Reproduce paper figures + headlines

Paper figures and reported numbers are written to `outputs/results/` using the following commands.

```bash
# Cheap reproduction: 1 seed per env (~minutes).
uv run python scripts/make_paper_figures.py

# Paper headlines: 10 seeds per env (~1h on H100).
uv run python scripts/make_paper_figures.py --num_seeds 10

# Include the architecture sweep (Fig 4, ~35 hours on H100).
uv run python scripts/make_paper_figures.py --num_seeds 10 --include_arch_sweep
```

This trains all three agents (Reacher, MountainCar with position-based goals, MountainCar with velocity-based goals), then renders Figs 2/3/5 from those sweeps. Pass `--reacher_sweep <dir>`, `--mc_position_sweep <dir>` or `--mc_velocity_sweep <dir>` to reuse an existing sweep and skip retraining for that environment.

### Expected results (10-seed Reacher)

Comparison of learnt `Q`-values against ground truth `Q^π` (normalised MSE):


|         | value           |
| ------- | --------------- |
| Q_NMSE  | 5.7e-1 ± 1.0e-2 |
| WM_NMSE | 1.2e-4 ± 2.9e-6 |


Results for policies trained using the implicit WM on unseen goals, averaged over 512 environment resets per seed:


| unseen goal     | `R^*`         | `R^WM` |
| --------------- | ------------- | ------------- |
| Far fingertip   | 0.712 ± 0.077 | 0.708 ± 0.083 |
| Target angle    | 0.679 ± 0.072 | 0.673 ± 0.081 |
| Target velocity | 0.652 ± 0.060 | 0.647 ± 0.064 |


### Expected results (10-seed MountainCar-position)


|         | value           |
| ------- | --------------- |
| Q_NMSE  | 1.0e-1 ± 1.1e-2 |
| WM_NMSE | 6.7e-3 ± 1.2e-4 |



| unseen goal   | `R^*`         | `R^WM` |
| ------------- | ------------- | ------------- |
| Fast car      | 0.436 ± 0.019 | 0.357 ± 0.125 |
| Gentle car    | 0.377 ± 0.028 | 0.308 ± 0.132 |
| Shortest path | 0.625 ± 0.045 | 0.591 ± 0.099 |


Comparison of extracted WMs of MountainCar agents trained with position-based goals vs velocity-based goals, confirming the claim that *WMs ~42× are closer to each other than to the truth*:


| comparison           | NMSE            |
| -------------------- | --------------- |
| position vs velocity | 1.7e-4 ± 9.8e-5 |
| position vs true     | 7.2e-3 ± 1.2e-4 |
| velocity vs true     | 7.3e-3 ± 2.4e-5 |


---

## Individual scripts

### Training (`run.py`)

The orchestrator above invokes `run.py` three times. To train just one env, or to run a subset of phases (e.g. only WM extraction from a fixed PQN checkpoint)

```bash
# Full pipeline for one env (PQN → plot_pqn → VI → WM → plot_wm → unseen).
uv run python run.py --config configs/reacher.py # default --num_seeds 1

# Re-train just the WM (and downstream eval) from a saved PQN checkpoint.
uv run python run.py --config configs/mountaincar-position.py --phases wm,plot_wm,unseen \
    --pqn_checkpoint outputs/mountaincar-position/seeds_<TS>/seed_4
```

Each `wm` invocation creates a fresh `wm_<TS>/` subdir inside the seed dir; old `wm_*/` directories are left intact.

### Figures

Each figure has a dedicated script. Re-run cheaply against an existing sweep with `--*_sweep`:

```bash
# Fig 2 — Reacher dynamics heatmap + trajectory rollouts + Q heatmaps.
uv run python scripts/reacher_visualisation.py --sweep_dir outputs/reacher/seeds_<TS>

# Fig 3 — Unseen-goal bar chart (needs both Reacher and MC sweeps).
uv run python scripts/unseen_tasks_bar.py --reacher_sweep outputs/reacher/seeds_<TS> \
    --mc_sweep outputs/mountaincar-position/seeds_<TS>

# Fig 5 — MountainCar quiver (needs both MC sweeps).
uv run python scripts/mountaincar_wm_quiver.py \
    --mc_position_sweep outputs/mountaincar-position/seeds_<TS_pos> \
    --mc_velocity_sweep outputs/mountaincar-velocity/seeds_<TS_vel>

# Fig 4 — Reacher architecture sweep (~35 hours on H100).
uv run python scripts/architecture_sweep.py --config configs/reacher.py

```

### Multi-GPU sweeps

`--seeds 0,1,2` overrides `--num_seeds` with an explicit list, `--sweep_dir` reuses an existing parent so new seeds land alongside the old ones, and `--aggregate_only` skips training and re-aggregates an existing sweep.

```bash
# Machine A: seeds 0-4.
uv run python run.py --config configs/reacher.py \
    --seeds 0,1,2,3,4 --sweep_dir outputs/reacher/my_sweep

# Machine B: seeds 5-9 into the same parent dir.
uv run python run.py --config configs/reacher.py \
    --seeds 5,6,7,8,9 --sweep_dir outputs/reacher/my_sweep

# Re-aggregate over all 10 seed_* dirs (does not re-train).
uv run python run.py --config configs/reacher.py \
    --aggregate_only --sweep_dir outputs/reacher/my_sweep
```

---

## Expected runtimes (single H100)

- **PQN training**: ~3 min per Reacher run (1e7 env steps), ~1 min per MountainCar (5e6 env steps).
- **P-learning**: ~10 s for Reacher and MountainCar.
- **Paper figures (except arch sweep)**: ~70 min.
- **Arch sweep**: ~35 hours total for the full 6×7×10 grid; per-cell ~5 min PQN + ~3 min WM tracking.

---

## Citation

```bibtex
@misc{letcher2026inverting,
      title={Inverting the Bellman Equation: From $Q$-Values to World Models}, 
      author={Alistair Letcher and Mattie Fellows and Alexander D. Goldie and Jonathan Richens and Jakob N. Foerster and Oliver Richardson},
      year={2026},
      eprint={2606.21173},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.21173}, 
}
```

Both environments wrap [gymnax](https://github.com/RobertTLange/gymnax) implementations. Cite gymnax as:

```bibtex
@misc{gymnax2022,
  author = {Lange, Robert Tjarko},
  title  = {{gymnax}: A {JAX}-based Reinforcement Learning Environment Library},
  url    = {https://github.com/RobertTLange/gymnax},
  year   = {2022},
}
```

License: see [LICENSE](LICENSE).