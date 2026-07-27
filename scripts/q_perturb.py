"""A3 — controlled off-support Q-perturbation: is recovery robust to realistic
(off-distribution) Q-error?

Take a converged Q. Inject a deterministic, region-gated perturbation into the
Q-values the WM training sees — either OFF the reachable set S_o (the realistic
"worst off-distribution" case) or ON it (the control) — retrain the WM, and
measure on-support recovery (WM dynamics NMSE on S_o) vs the injected Q-error.

Expected: off-support perturbation leaves on-support recovery ~flat (recovery is
blind to off-support Q-error, because for on-support (s,a) the Bellman target
evaluates Q at on-support successors); on-support perturbation degrades it. This
directly answers reviewer Q2 (distributional vs uniform Q-error bounds).

Usage:
    uv run python scripts/q_perturb.py --config configs/mountaincar-position.py \
        --run_dir <SWEEP>/seed_0 [--scales 0,0.25,0.5,1,2] [--wm_num_steps N]
"""
import argparse
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.run_context import build_context
from eval.visitation import grid_visitation
from eval.track_recovery import _to_weights, _weights_at_states
from eval.world_model import dynamics_nmse
from eval.value_iteration import evaluate_pqn_on_grid
from training.wm import train_world_model
from training.pqn_utils import make_q_network


def make_mask_sampler(mask_grid, state_ranges, R):
    """Sampler drawing states uniformly from the reachable set S_o (mask cells),
    at cell centres + jitter. This is the practical analogue of P-learning on the
    reduced MDP over S_o (Thm B2): WM training then only ever queries Q on S_o."""
    sd = mask_grid.ndim
    flat = (mask_grid.reshape(-1) > 0).astype(jnp.float32)
    p = flat / (jnp.sum(flat) + 1e-12)
    axes = [jnp.linspace(lo, hi, R) for lo, hi in state_ranges]
    steps = jnp.array([float(axes[d][1] - axes[d][0]) for d in range(sd)])

    def sample_fn(key, n):
        k1, k2 = jax.random.split(key)
        cells = jax.random.choice(k1, flat.shape[0], shape=(n,), p=p)
        coords, rem = [], cells
        for d in range(sd - 1, -1, -1):
            coords.insert(0, rem % R); rem = rem // R
        jit = jax.random.uniform(k2, (n, sd), minval=-0.5, maxval=0.5)
        cols = [axes[d][coords[d]] + jit[:, d] * steps[d] for d in range(sd)]
        return jnp.stack(cols, axis=-1)

    return sample_fn


def make_projection(mask_grid, state_ranges, R):
    """Map any state to the centre of the nearest reachable-set ($S_o$) cell.

    Confining every Q-query through this projection makes the WM's Bellman
    bootstrap read Q only on $S_o$ — the practical analogue of the estimator
    $M_{S_o}^{+}Q$ that Theorem B2 analyses. Off-support Q-error is then never
    read, so recovery should be immune to it at any magnitude."""
    from scipy.ndimage import distance_transform_edt
    m = np.asarray(mask_grid) > 0                       # True in S_o
    sd = m.ndim
    # nearest in-S_o cell for every grid cell (edt on the complement)
    _, idx = distance_transform_edt(~m, return_indices=True)   # idx: (sd,)+grid
    axes = [np.linspace(lo, hi, R) for lo, hi in state_ranges]
    centers = np.stack([axes[d][idx[d]] for d in range(sd)], axis=-1)  # grid + (sd,)
    centers_flat = jnp.asarray(centers.reshape(-1, sd), dtype=jnp.float32)
    los = jnp.array([lo for lo, _ in state_ranges])
    his = jnp.array([hi for _, hi in state_ranges])

    def proj(obs):
        lin = jnp.zeros(obs.shape[0], dtype=jnp.int32)
        for d in range(sd):
            bi = jnp.clip(jnp.round((obs[:, d] - los[d]) / (his[d] - los[d]) * (R - 1)),
                          0, R - 1).astype(jnp.int32)
            lin = lin * R + bi
        return centers_flat[lin]

    return proj


def make_perturbed_q_fn(pqn_config, q_vars, mask_grid, state_ranges, R,
                        scale, region, qscale, key, proj=None):
    """Deterministic, region-gated additive Q-perturbation.

    noise(s,a) = scale · qscale · cos(ŝ·W_a + b_a), a fixed smooth field (so the
    Bellman target and bootstrap stay self-consistent), gated to `region`
    ('off' → states NOT in S_o; 'on' → states in S_o) via the S_o indicator.
    """
    network = make_q_network(pqn_config)
    sd = len(state_ranges)
    A = pqn_config["ACTION_DIM"]
    los = jnp.array([r[0] for r in state_ranges])
    his = jnp.array([r[1] for r in state_ranges])
    kW, kb = jax.random.split(key)
    W = jax.random.normal(kW, (sd, A)) * 3.0           # per-action random frequency
    b = jax.random.uniform(kb, (A,), minval=0.0, maxval=2 * jnp.pi)
    flat_ind = (mask_grid.reshape(-1) > 0).astype(jnp.float32)   # 1 in S_o

    def _onsupport(obs):
        lin = jnp.zeros(obs.shape[0], dtype=jnp.int32)
        for d in range(sd):
            frac = (obs[:, d] - los[d]) / (his[d] - los[d])
            bi = jnp.clip(jnp.round(frac * (R - 1)), 0, R - 1).astype(jnp.int32)
            lin = lin * R + bi
        return flat_ind[lin]

    def q_fn(obs, goal_repr):
        o = proj(obs) if proj is not None else obs      # confine Q-query to S_o
        q = network.apply(q_vars, o, goal_repr, train=False)
        obs_n = (o - los) / (his - los)
        noise = jnp.cos(obs_n @ W + b)                  # [n, A]
        ind = _onsupport(o)
        gate = ind if region == "on" else (1.0 - ind)
        return q + scale * qscale * noise * gate[:, None]

    return q_fn, (W, b)


def injected_qnmse(pqn_config, q_vars, W, b, mask_grid, state_ranges, R,
                   scale, qscale, region, goals):
    """NMSE of the injected perturbation over the target region (the x-axis)."""
    sd = len(state_ranges)
    los = np.array([r[0] for r in state_ranges]); his = np.array([r[1] for r in state_ranges])
    axes = [np.linspace(los[d], his[d], R) for d in range(sd)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, sd)
    obs_n = (mesh - los) / (his - los)
    noise = np.cos(obs_n @ np.asarray(W) + np.asarray(b))       # [R^sd, A]
    ind = (np.asarray(mask_grid).reshape(-1) > 0)
    sel = ind if region == "on" else ~ind
    pert = scale * qscale * noise[sel]
    inj_mse = float(np.mean(pert ** 2))
    return inj_mse / (qscale ** 2 + 1e-12)      # normalised by Var(Q) ≈ qscale^2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True, help="run dir with pqn_checkpoint.pkl")
    ap.add_argument("--scales", default="0,0.25,0.5,1,2")
    ap.add_argument("--regions", default="off,on")
    ap.add_argument("--wm_sample_region", choices=["onsupport", "uniform"],
                    default="onsupport",
                    help="train the WM on S_o (reduced MDP, Thm B2's regime; "
                         "default) or uniformly over the whole box (naive)")
    ap.add_argument("--confine", action="store_true",
                    help="project every Q-query onto S_o so the bootstrap never "
                         "reads off-support Q (the M_{S_o}^+ estimator of Thm B2)")
    ap.add_argument("--wm_num_steps", type=int, default=None)
    ap.add_argument("--wm_batch_size", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    ctx = build_context(args.config)
    P, E = ctx["pqn_config"], ctx["env_config"]
    if args.wm_num_steps is not None:
        ctx["wm_config"]["NUM_STEPS"] = args.wm_num_steps
    if args.wm_batch_size is not None:
        ctx["wm_config"]["BATCH_SIZE"] = args.wm_batch_size
    scales = [float(x) for x in args.scales.split(",")]
    regions = args.regions.split(",")
    out_dir = args.out_dir or f"outputs/{E['ENV_NAME'].lower()}/q_perturb"
    os.makedirs(out_dir, exist_ok=True)

    adim, sdim = P["ACTION_DIM"], P["STATE_DIM"]
    R = E["VI_GRID_RES"]
    state_ranges = [tuple(r) for r in E.get("VI_STATE_RANGES", E["STATE_RANGES"])]

    # Converged Q.
    ck = os.path.join(args.run_dir, "pqn_checkpoint.pkl")
    d = pickle.load(open(ck, "rb"))
    q_params, q_bs = d["params"], d["batch_stats"]
    q_vars = {"params": q_params, "batch_stats": q_bs}

    # S_o from the converged greedy policy (union over goals) → binary mask grid.
    goals, masks = ctx["goals"], ctx["goal_masks"]
    reset_keys = jax.random.split(jax.random.PRNGKey(0), 512)
    start_obs = jax.block_until_ready(jax.vmap(
        lambda k: ctx["basic_env"].reset(k, ctx["env_params"])[0])(reset_keys))
    w = grid_visitation(q_params, q_bs, P, ctx["vi_dynamics_fn"],
                        jnp.asarray(goals), jnp.asarray(masks), start_obs,
                        sdim, adim, state_ranges, R,
                        max_steps=P["MAX_STEPS_IN_EPISODE"], a_threshold=P["REWARD_A"],
                        terminate_on_goal=P["TERMINATE_ON_GOAL"])
    mask_grid = _to_weights(jax.block_until_ready(w), "mask")
    print(f"S_o covers {float((mask_grid>0).mean())*100:.1f}% of cells", flush=True)

    # WM training distribution: on S_o (reduced MDP, Thm B2) or uniform (naive).
    if args.wm_sample_region == "onsupport":
        wm_sample_fn = make_mask_sampler(mask_grid, state_ranges, R)
    else:
        wm_sample_fn = ctx["wm_sample_fn"]
    print(f"WM trained on: {args.wm_sample_region}", flush=True)

    # qscale = representative std of clean Q over the grid (per-goal, averaged).
    qstds = []
    for gi in range(len(goals)):
        _, Q = evaluate_pqn_on_grid(q_params, q_bs, jnp.asarray(goals[gi]),
                                    jnp.asarray(masks[gi]),
                                    tuple(jnp.linspace(lo, hi, R) for lo, hi in state_ranges),
                                    P, adim, sdim)
        qstds.append(float(jnp.std(Q)))
    qscale = float(np.mean(qstds))
    print(f"qscale (std of clean Q) = {qscale:.4f}", flush=True)

    # WM eval set (uniform) + on-support weights (S_o mask), fixed across runs.
    from training.wm import sample_states
    eval_states = sample_states(jax.random.PRNGKey(777), P["STATE_RANGES"], sdim,
                                int(ctx["wm_config"]["EVAL_HEATMAP_RES"]) ** 2)
    eval_w = _weights_at_states(eval_states, mask_grid, state_ranges)

    def train_and_score(q_value_fn):
        p_params, _ = train_world_model(
            q_params, q_bs, ctx["wm_config"], P, goals, masks,
            env_terminated_fn=None, sample_states_fn=wm_sample_fn,
            use_wandb=False, q_value_fn=q_value_fn)
        on, _, _ = dynamics_nmse(p_params, ctx["wm_config"], E, ctx["wm_dynamics_fn"],
                                 eval_states, adim, sdim, weights=eval_w)
        return float(on)

    proj = make_projection(mask_grid, state_ranges, R) if args.confine else None
    if args.confine:
        print("Q-queries CONFINED to S_o (projection) — bootstrap never reads "
              "off-support Q", flush=True)

    rows = []
    base_key = jax.random.PRNGKey(3)
    for region in regions:
        for scale in scales:
            if scale == 0.0 and proj is None:
                qfn, WB = None, (None, None)
                inj = 0.0
            else:
                qfn, WB = make_perturbed_q_fn(P, q_vars, mask_grid, state_ranges, R,
                                              scale, region, qscale, base_key, proj=proj)
                inj = (injected_qnmse(P, q_vars, WB[0], WB[1], mask_grid, state_ranges,
                                      R, scale, qscale, region, goals) if scale > 0 else 0.0)
            on = train_and_score(qfn)
            rows.append({"region": region, "scale": scale, "inj_qnmse": inj,
                         "wm_nmse_onsupport": on})
            print(f"  region={region:3s} scale={scale:<4} inj_QNMSE={inj:.3e}  "
                  f"on-support WM_NMSE={on:.3e}", flush=True)

    # ── Plot: on-support recovery vs injected Q-error, off vs on support ──
    from plotting.style import set_paper_style, unset_paper_style, COLOR_TRUE, COLOR_UNSAFE
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    colors = {"off": COLOR_TRUE, "on": COLOR_UNSAFE}
    labels = {"off": "perturb OFF-support Q", "on": "perturb ON-support Q (control)"}
    for region in regions:
        rr = [r for r in rows if r["region"] == region]
        rr.sort(key=lambda r: r["inj_qnmse"])
        x = [r["inj_qnmse"] for r in rr]; y = [r["wm_nmse_onsupport"] for r in rr]
        ax.plot(x, y, "o-", color=colors.get(region, None), label=labels.get(region, region))
    ax.set_xlabel(r"injected Q-error in perturbed region ($Q_{\mathrm{NMSE}}$)")
    ax.set_ylabel(r"on-support recovery  $\mathrm{WM}_{\mathrm{NMSE}}$")
    ax.set_yscale("log")
    _regime = r"reduced MDP on $S_o$" if args.wm_sample_region == "onsupport" else "whole state space"
    _conf = r", Q confined to $S_o$" if args.confine else ""
    ax.set_title(f"On-support recovery vs injected Q-error\n(P-learning on the {_regime}{_conf})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out_dir}/q_perturb.png"); plt.close(fig)
    unset_paper_style()

    keys = ["region", "scale", "inj_qnmse", "wm_nmse_onsupport"]
    np.savez(f"{out_dir}/q_perturb.npz",
             **{k: np.array([r[k] for r in rows], dtype=object if k == "region" else float)
                for k in keys})
    print(f"\nWrote {out_dir}/q_perturb.png, q_perturb.npz")


if __name__ == "__main__":
    main()
