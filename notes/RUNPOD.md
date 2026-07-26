# Running the recovery-vs-Q-error experiments on a RunPod GPU

Branch: `rebuttal-recovery-vs-qerror`. The PQN checkpoints live in `outputs/`
(gitignored), so they do NOT come with the clone — you regenerate them on the
pod (PQN training is ~1 min/seed for MountainCar on GPU).

## 0. Pod
Any CUDA-12-capable GPU pod works (e.g. RunPod "PyTorch 2.x" template, or a plain
CUDA 12 Ubuntu image). `jax[cuda12]` ships its own CUDA libraries via pip wheels,
so you only need a recent NVIDIA driver (RunPod provides one) — no manual CUDA
install. A single 24 GB GPU is plenty for MountainCar.

## 1. Tools
```bash
# uv (manages the Python 3.12 the project pins, and the deps)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env        # or restart the shell
```

## 2. Clone + install
```bash
# private repo → use a GitHub token (or set up an SSH deploy key)
git clone https://<GITHUB_TOKEN>@github.com/aletcher/inverting-bellman.git
cd inverting-bellman
git checkout rebuttal-recovery-vs-qerror
uv sync --extra cuda               # installs Python 3.12 if needed + jax[cuda12]
```

## 3. Verify the GPU is visible to JAX
```bash
uv run python -c "import jax; print(jax.devices())"
# expect: [CudaDevice(id=0)]   (not CpuDevice)
```

## 4. Regenerate PQN checkpoints (the Q-quality axis)
```bash
# 5 seeds, PQN only, saving the 20 intermediate step_*.pkl per seed
uv run python run.py --config configs/mountaincar-position.py \
    --seeds 0,1,2,3,4 --phases pqn --save_pqn_checkpoints
# note the printed "Sweep dir: outputs/mountaincar-position/seeds_<TIMESTAMP>"
export SWEEP=outputs/mountaincar-position/seeds_<TIMESTAMP>
```
(1 seed is enough for a first look — the scaling curve already has ~20 points
from one run. Use `--seeds 0` for a quick pass.)

## 5. Run the experiments (full fidelity — defaults are GPU-appropriate)
```bash
# A1 — recovery vs Q-error (trains a 20000-step WM per checkpoint; ~10 s/ckpt GPU)
uv run python scripts/recovery_vs_qerror.py \
    --config configs/mountaincar-position.py \
    --run_dirs $SWEEP/seed_0 $SWEEP/seed_1 $SWEEP/seed_2 $SWEEP/seed_3 $SWEEP/seed_4
# each checkpoint is weighted by its OWN policy visitation (default);
# add --visitation_from final to weight all by the converged agent instead.

# A2 — distributional vs uniform (reuses A1's per-seed tracking npz)
uv run python scripts/dist_vs_uniform.py \
    --config configs/mountaincar-position.py \
    --run_dirs $SWEEP/seed_0 $SWEEP/seed_1 $SWEEP/seed_2 $SWEEP/seed_3 $SWEEP/seed_4 --reuse
```
On GPU do NOT pass `--wm_num_steps` / `--wm_batch_size` — the defaults (20000
steps, batch 4096) are the paper-grade setting and are cheap on GPU.

Rough GPU budget (5 seeds): PQN ~5 min; recovery ~20 ckpts × 5 seeds × ~10 s WM
≈ 20–30 min. Well under an hour total.

## 6. Outputs
```
$SWEEP/seed_*/recovery_track/recovery_tracking.npz   # per-seed raw metrics
outputs/mountaincar/recovery_vs_qerror/{scatter.png, curves_vs_step.png, metrics.npz}   # A1
outputs/mountaincar/dist_vs_uniform/{qerr_uniform_vs_visit.png, recovery_tracks_visit.png, dist.npz}  # A2
```
Retrieve via the RunPod file browser, `runpodctl send`, `scp`, or commit them to
a results branch. Sanity checks: Spearman ρ(WM_NMSE, Q_NMSE) < 0 (A1);
`WM_NMSE_visit < WM_NMSE_uniform`, and `WM_NMSE_visit` roughly flat while
`WM_NMSE_uniform` falls as `visit_frac` rises (A2 — the broadening-support story).

## Reacher (optional, now feasible on GPU)
Same flow with `--config configs/reacher.py`. Reacher's value iteration runs on a
50⁴ = 6.25M-cell grid (the reason it was infeasible on CPU), so give it a GPU
with ≥24 GB. `build_context` already wires the 4D-effective grid, samplers, and
FK lifts.

## Notes / gotchas
- `run.py --seeds` spawns one subprocess per seed and forwards
  `--save_pqn_checkpoints`; each writes `$SWEEP/seed_S/checkpoints/step_*.pkl`.
- The final `step_*.pkl` duplicates `pqn_checkpoint.pkl`; the tracker drops it, so
  you get exactly the 20 eval-point checkpoints.
- If JAX raises a cuDNN/driver mismatch, `uv pip install -U "jax[cuda12]"` inside
  the venv usually resolves it against the pod's driver.
