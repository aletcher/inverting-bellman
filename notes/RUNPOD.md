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
The repo is private and the remote is SSH. The deploy key lives on the
persistent volume at `/workspace/.secrets/ssh/github_ed25519`; restore it into
the (ephemeral) home dir first — see §2a if this is a migrated pod.
```bash
bash /workspace/.secrets/bootstrap-ssh.sh    # ssh key + git identity
git clone git@github.com:aletcher/inverting-bellman.git
cd inverting-bellman
git checkout rebuttal-recovery-vs-qerror
uv sync --extra cuda               # installs Python 3.12 if needed + jax[cuda12]
```

## 2a. After a pod migration
Migrations wipe `/` but keep `/workspace`, so the clone, the venv and the
`outputs/` survive — only the home dir is gone. One command brings it back:
```bash
bash /workspace/.secrets/bootstrap-ssh.sh
```
This copies the key to `~/.ssh/github_ed25519` (mode 600), writes a
`Host github.com` block pinning it, and restores `~/.gitconfig`. It ends by
running `ssh -T git@github.com`, which should greet you by username.

The copy step is not optional: `/workspace` is a MooseFS/FUSE mount that
ignores `chmod` and reports every file as `0777`, so ssh refuses to read a key
straight off the volume ("permissions are too open"). Only the copy on
overlayfs can hold mode 600. For the same reason the at-rest key is
world-readable in mode terms — it is a repo-scoped deploy key rather than an
account key to limit that. Add a passphrase any time with
`ssh-keygen -p -f /workspace/.secrets/ssh/github_ed25519` (then `ssh-add` it
once per pod).

To register a replacement key: generate with
`ssh-keygen -t ed25519 -f /workspace/.secrets/ssh/github_ed25519`, then add the
`.pub` under repo Settings → Deploy keys, with write access.

## 3. Verify the GPU is visible to JAX
Many RunPod images ship a **system CUDA 13** whose `LD_LIBRARY_PATH` collides with
JAX's bundled CUDA-12 wheels (symptom: `cuSPARSE library was not found`, falls
back to CPU). Drop the system path first:
```bash
unset LD_LIBRARY_PATH                              # do this in every fresh shell
uv run python -c "import jax; print(jax.devices())"
# expect: [CudaDevice(id=0)]   (not CpuDevice)
```
(Optional: `echo 'unset LD_LIBRARY_PATH' >> ~/.bashrc` so new shells inherit it.)

## 4. Regenerate PQN checkpoints (the Q-quality axis)
```bash
# 5 seeds, PQN only, saving the 20 intermediate step_*.pkl per seed
uv run python run.py --config configs/mountaincar-position.py \
    --seeds 0,1,2,3,4 --phases pqn --save_pqn_checkpoints
# grab the sweep dir it just created (latest seeds_* dir)
export SWEEP=$(ls -td outputs/mountaincar-position/seeds_* | head -1)
echo "SWEEP=$SWEEP"
ls $SWEEP/seed_0/checkpoints/        # sanity: 21 step_*.pkl files
```
(1 seed suffices for a first look — one run already gives ~20 scaling points.
Use `--seeds 0` for a quick pass.)

## 5. Run the experiments (full fidelity — defaults are GPU-appropriate)
Do NOT pass `--wm_num_steps` / `--wm_batch_size` on GPU — the defaults (20000
steps, batch 4096) are the paper-grade setting and are cheap on GPU.
```bash
SEEDS="$SWEEP/seed_0 $SWEEP/seed_1 $SWEEP/seed_2 $SWEEP/seed_3 $SWEEP/seed_4"

# A1 — recovery vs Q-error (trains a 20000-step WM per checkpoint; ~10 s/ckpt GPU).
# Each checkpoint is weighted by its OWN policy visitation, on-support = reachable
# set S_o (mask); both are defaults.
uv run python scripts/recovery_vs_qerror.py \
    --config configs/mountaincar-position.py --run_dirs $SEEDS

# A2 — distributional (on-support) vs uniform (reuses A1's per-seed npz).
uv run python scripts/dist_vs_uniform.py \
    --config configs/mountaincar-position.py --run_dirs $SEEDS --reuse

# A3 — controlled off-support Q-perturbation (robustness to off-distribution Q).
#   main: P-learning on the reduced MDP over S_o (the robust regime)
uv run python scripts/q_perturb.py --config configs/mountaincar-position.py \
    --run_dir $SWEEP/seed_0 --wm_sample_region onsupport --scales 0,0.25,0.5,1,2
#   contrast: naive whole-space training (NOT robust — shows the restriction matters)
uv run python scripts/q_perturb.py --config configs/mountaincar-position.py \
    --run_dir $SWEEP/seed_0 --wm_sample_region uniform --scales 0,0.25,0.5,1,2 \
    --out_dir outputs/mountaincar/q_perturb_uniform

# (theory sanity check — CPU, instant, no GPU needed)
uv run python notes/theory_checks.py
```
Rough GPU budget (5 seeds): PQN ~5 min; A1 ~20–30 min; A2 seconds; A3 ~3 min.
Under an hour total.

## 6. Outputs — retrieve these
```
$SWEEP/seed_*/recovery_track/recovery_tracking.npz            # per-seed raw metrics
outputs/mountaincar/recovery_vs_qerror/{scatter.png, curves_vs_step.png, metrics.npz}   # A1
outputs/mountaincar/dist_vs_uniform/{onoff_summary.png, dist.npz}                        # A2
outputs/mountaincar/q_perturb/{q_perturb.png, q_perturb.npz}                             # A3 (on-support)
outputs/mountaincar/q_perturb_uniform/{q_perturb.png, q_perturb.npz}                     # A3 (contrast)
```
Tar + fetch, e.g.:
```bash
tar czf results.tgz outputs/mountaincar $SWEEP/seed_*/recovery_track
# then: runpodctl send results.tgz   (or the RunPod file browser / scp)
```
Sanity checks: A1 Spearman ρ(Q_NMSE, WM_NMSE) > 0 (~0.85); A2 `WM_NMSE` on `S_o`
≪ uniform, Q\* smaller on-support while Q^π ~equal; A3 on-support run: off-support
curve flat, on-support curve rises ~20×.

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
- **`cuSPARSE library was not found` / falls back to CPU** → stale system-CUDA
  `LD_LIBRARY_PATH`; `unset LD_LIBRARY_PATH` (step 3). If it persists, force the
  bundled wheels: `uv pip install --reinstall "jax[cuda12]==0.9.2"`.
- A3 `--run_dir` must point at a single seed dir containing `pqn_checkpoint.pkl`.
- **`git@github.com: Permission denied (publickey)`** after a migration → you
  skipped §2a; run `bash /workspace/.secrets/bootstrap-ssh.sh`.
