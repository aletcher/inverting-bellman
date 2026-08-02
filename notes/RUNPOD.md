# Running on a RunPod GPU

Sections 0–3 (pod, tools, SSH-across-migrations, GPU) are branch-independent —
start there whatever you are running. Section 4 onward is `boltzmann`
(pi-learning); the recovery-vs-Q-error recipe lives on the copy of this file on
`rebuttal-recovery-vs-qerror`, whose scripts do not exist on this branch.

PQN checkpoints live in `outputs/` (gitignored), so they do NOT come with the
clone — you regenerate them on the pod (~1 min/seed for MountainCar on GPU).
They do survive pod migrations, since `/workspace` is persistent.

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
Many RunPod images ship a **system CUDA 13** and bake
`LD_LIBRARY_PATH=/usr/local/cuda/lib64` into the container env (not into any
profile script, so it reaches every process), where it shadows the CUDA-12 libs
in JAX's own wheels. Symptom: `cuSPARSE library was not found`, then a *silent*
fall back to CPU. Drop the system path:
```bash
unset LD_LIBRARY_PATH
uv run python -c "import jax; print(jax.devices())"
# expect: [CudaDevice(id=0)]   (not CpuDevice)
```
`bootstrap-ssh.sh` (§2a) appends that `unset` to `~/.bashrc`, so on a
bootstrapped pod every *interactive* shell already has it.

**Non-interactive shells do not read `~/.bashrc`.** A batch job launched as
`nohup bash something.sh &` still inherits the bad path and trains on CPU, with
nothing but one `cuSPARSE` line in the log to say so. Either `unset` at the top
of the script, or launch with:
```bash
env -u LD_LIBRARY_PATH uv run python run.py ...
```
Cheap guard for a long job — assert the device up front rather than finding out
hours later:
```bash
uv run python -c "import jax,sys; sys.exit(0 if jax.devices()[0].platform=='gpu' else 1)" \
  || { echo "NOT ON GPU"; exit 1; }
```
Harmless: `E... xtile_compiler ... gemm_fusion` lines are Triton autotuning
noise on newer cards (e.g. Blackwell RTX PRO 6000). XLA falls back to another
kernel and the numerics are unaffected.

## 4. PQN checkpoints (the Q you extract from)
`outputs/` is on the persistent volume, so checkpoints from an earlier pod are
still there and usually need no retraining — look before you burn GPU time:
```bash
ls outputs/*/seeds*/seed_*/pqn_checkpoint.pkl
```
To (re)train:
```bash
uv run python run.py --config configs/mountaincar-position.py --phases pqn --seeds 0
export SWEEP=$(ls -td outputs/mountaincar-position/seeds_* | head -1)
```
`--save_pqn_checkpoints` additionally writes the intermediate `step_*.pkl`; it
is off by default and only the architecture sweep needs it (~1.7 GB per Reacher
seed, so leave it off unless you do).

## 5. Pi-learning — policy-only WM extraction (this branch)
`scripts/policy_wm.py` extracts a world model from a PQN checkpoint using only
the policy, under a Boltzmann-rationality assumption at temperature `--tau`:
```bash
uv run python scripts/policy_wm.py \
    --config configs/mountaincar-position.py \
    --pqn_checkpoint $SWEEP/seed_0
```
Defaults come from `PIWM_CONFIG` in the config, and are GPU-appropriate — pass
`--num_steps` / `--batch_size` only to cut a run short for a smoke test.
Results land in `<checkpoint_dir>/piwm_<timestamp>/` unless `--out_dir` is set.

Knobs that change the method rather than the budget: `--consistency {soft,hard}`
(normalisation of log π), `--v_stop_grad` (semi-gradient bootstrap, freezing
V_ψ in the target), and `--wm_loss {l1,mse}`.

The full pipeline (value iteration, the standard WM baseline, unseen-task
eval) runs through `run.py --phases`, which accepts
`pqn,plot_pqn,vi,wm,plot_wm,unseen`.

## 6. Retrieving results
Everything under `outputs/` persists across migrations, so you only need this to
get artefacts off the pod:
```bash
tar czf results.tgz outputs/<env>/...
# then: runpodctl send results.tgz   (or the RunPod file browser / scp)
```

## Reacher (optional, now feasible on GPU)
Same flow with `--config configs/reacher.py`. Reacher's value iteration runs on a
50⁴ = 6.25M-cell grid (the reason it was infeasible on CPU), so give it a GPU
with ≥24 GB. `build_context` already wires the 4D-effective grid, samplers, and
FK lifts.

## Notes / gotchas
- `run.py --seeds` spawns one subprocess per seed and forwards
  `--save_pqn_checkpoints`; each writes `$SWEEP/seed_S/checkpoints/step_*.pkl`.
- The final `step_*.pkl` duplicates `pqn_checkpoint.pkl`.
- **`cuSPARSE library was not found` / falls back to CPU** → stale system-CUDA
  `LD_LIBRARY_PATH`; `unset LD_LIBRARY_PATH` (§3). If it persists, force the
  bundled wheels: `uv pip install --reinstall "jax[cuda12]==0.9.2"`.
- `--pqn_checkpoint` takes a single seed dir containing `pqn_checkpoint.pkl`
  (or the `.pkl` itself), not a sweep dir.
- **`git@github.com: Permission denied (publickey)`** after a migration → you
  skipped §2a; run `bash /workspace/.secrets/bootstrap-ssh.sh`.
