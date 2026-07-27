"""Margin curves delta(eps) of the LEARNED Bellman profile map in MountainCar.

For each trained PQN checkpoint (4 sparse position goals), evaluate
    m_g(s) = r_g(s) + gamma * D_g(s) * max_a Q_hat(s, a, g)
on a state grid, and estimate the separation-margin curve
    delta(eps) = min over state pairs with ||x - y|| >= eps (normalized coords)
                 of max_l |m_{g_l}(x) - m_{g_l}(y)|,
for nested goal subsets L = 1..4 of the SAME 4-goal-trained agent
(no retraining: this measures the information content of L value probes).

delta(eps) is the continuum analogue of the column separation gap(M): any two
world models with Bellman residual <= tau agree to within eps wherever
delta(eps) > 2*tau. Pairs are found via k-NN in profile space (l-inf metric),
so reported delta values are estimates (upper bounds are exact for discovered
pairs; increasing K checks stability).

Also reported: the same curves excluding "dynamically degenerate" states
(identical successor under all 3 actions — the left-wall funnel and clip
saturation), where exact values provably coincide for every goal and any
learned separation is approximation noise.

Run from the repo root:  .venv/bin/python scripts/margin_curve_mountaincar.py
Outputs: JSON + markdown to ../pq-neurips/rebuttal/.
"""

import json
import os
import pickle
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from training.pqn_utils import make_q_network, make_goal_repr

SEEDS_DIR = os.path.join(REPO, "outputs/mountaincar-position/seeds_20260604_100625")
OUT_DIR = os.path.join(os.path.dirname(REPO), "pq-neurips/rebuttal")

N_FULL = 1001          # reference grid (matches mountaincar_diagnostic.py)
STRIDE = 4             # subsample stride -> 251 x 251 = 63,001 states
X_MIN, X_MAX = -1.2, 0.6
V_MIN, V_MAX = -0.07, 0.07
EPS_LIST = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
K_NN = 200             # profile-space neighbours per state
NESTED = [3, 1, 2, 0]  # config GOALS = [-1.2, -0.6, 0.0, 0.6]; nest 0.6, -0.6, 0.0, -1.2
DILATE = 2             # degenerate-mask dilation (subgrid cells)
CHUNK = 16384


def state_grid():
    xs = np.linspace(X_MIN, X_MAX, N_FULL)[::STRIDE]
    vs = np.linspace(V_MIN, V_MAX, N_FULL)[::STRIDE]
    X, V = np.meshgrid(xs, vs, indexing="ij")
    return xs.size, np.stack([X.ravel(), V.ravel()], axis=1)


def degenerate_mask(n_side, S):
    """States whose true successor is identical under all 3 actions (gymnax dynamics)."""
    x, v = S[:, 0], S[:, 1]
    nxt = []
    for a in (0, 1, 2):
        v2 = np.clip(v + (a - 1) * 0.001 - 0.0025 * np.cos(3 * x), V_MIN, V_MAX)
        x2 = np.clip(x + v2, X_MIN, X_MAX)
        v2 = v2 * (1 - ((x2 == X_MIN) & (v2 < 0)))
        nxt.append(np.stack([x2, v2], axis=1))
    same = np.all(nxt[0] == nxt[1], axis=1) & np.all(nxt[1] == nxt[2], axis=1)
    return binary_dilation(same.reshape(n_side, n_side), iterations=DILATE).ravel()


def learned_m_fields(seed_dir, S):
    """m_g(s) = r_g(s) + gamma * D_g(s) * max_a Q_hat(s,a,g), one field per goal."""
    with open(os.path.join(seed_dir, "pqn_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(seed_dir, "pqn_checkpoint.pkl"), "rb") as f:
        ckpt = pickle.load(f)
    net = make_q_network(cfg)
    variables = {"params": ckpt["params"], "batch_stats": ckpt["batch_stats"]}
    gamma, a_thr = cfg["GAMMA"], cfg["REWARD_A"]
    goals = np.asarray(cfg["GOALS"], dtype=np.float32)
    masks = np.asarray(cfg["REWARD_MASK"], dtype=np.float32)
    assert cfg["REWARD_TYPE"] == "sparse" and cfg["TERMINATE_ON_GOAL"]

    @jax.jit
    def v_batch(obs, goal, mask):
        gr = make_goal_repr(goal, mask, obs.shape[0], obs.shape[1])
        q = net.apply(variables, obs, gr, train=False)
        return jnp.max(q, axis=1)

    obs = jnp.asarray(S, dtype=jnp.float32)
    fields = []
    for gl, mk in zip(goals, masks):
        vhat = np.concatenate([
            np.asarray(v_batch(obs[i:i + CHUNK], jnp.asarray(gl), jnp.asarray(mk)))
            for i in range(0, S.shape[0], CHUNK)
        ])
        dist_sq = (((S - gl) * mk) ** 2).sum(axis=1)
        r = (dist_sq < a_thr ** 2).astype(np.float64)      # sparse reward
        m = r + gamma * (1.0 - r) * vhat                   # D_g = 1 - r_g
        fields.append(m)
    return fields


def margin_curves(fields, S_norm, degen):
    """delta(eps) per nested L, all-pairs and excluding degenerate endpoints."""
    out = {}
    for L in range(1, 5):
        P = np.stack([fields[i] for i in NESTED[:L]], axis=1)
        tree = cKDTree(P)
        dist, nbr = tree.query(P, k=K_NN + 1, p=np.inf)
        dist, nbr = dist[:, 1:], nbr[:, 1:]
        i = np.repeat(np.arange(P.shape[0]), K_NN)
        j, pd = nbr.ravel(), dist.ravel()
        sd = np.linalg.norm(S_norm[i] - S_norm[j], axis=1)
        ok = ~(degen[i] | degen[j])
        out[L] = {
            "all": [float(pd[sd >= e].min()) if (sd >= e).any() else None for e in EPS_LIST],
            "excl_degen": [float(pd[(sd >= e) & ok].min()) if ((sd >= e) & ok).any() else None
                           for e in EPS_LIST],
        }
        # context: local profile roughness = median profile distance to state-NN
        stree = cKDTree(S_norm)
        _, snn = stree.query(S_norm, k=2)
        out[L]["local_roughness"] = float(np.median(
            np.abs(P[snn[:, 1]] - P).max(axis=1)))
    return out


def main():
    n_side, S = state_grid()
    S_norm = np.stack([(S[:, 0] - X_MIN) / (X_MAX - X_MIN),
                       (S[:, 1] - V_MIN) / (V_MAX - V_MIN)], axis=1)
    degen = degenerate_mask(n_side, S)
    print(f"grid {n_side}^2 = {S.shape[0]} states; degenerate (dilated): {degen.mean():.2%}")

    seed_dirs = sorted(d for d in os.listdir(SEEDS_DIR) if d.startswith("seed_"))
    results = {}
    for sd in seed_dirs:
        t0 = time.time()
        fields = learned_m_fields(os.path.join(SEEDS_DIR, sd), S)
        results[sd] = margin_curves(fields, S_norm, degen)
        print(f"{sd}: done in {time.time() - t0:.1f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {"grid": f"{n_side}x{n_side}", "stride": STRIDE, "eps": EPS_LIST, "k_nn": K_NN,
            "nested_goal_order": [0.6, -0.6, 0.0, -1.2], "seeds": seed_dirs,
            "degenerate_frac": float(degen.mean())}
    with open(os.path.join(OUT_DIR, "margin_curves_mountaincar.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=1)

    # aggregate: median and (min, max) across seeds
    lines = ["# Margin curves of the learned profile map (MountainCar, 10 seeds)", "",
             f"delta(eps) = min over pairs with normalized state distance >= eps of "
             f"max_l |m_l(x) - m_l(y)|; K={K_NN}-NN estimate on a {n_side}^2 grid; "
             f"nested goals {meta['nested_goal_order']}.", ""]
    for variant in ("all", "excl_degen"):
        lines += [f"## {'All pairs' if variant == 'all' else 'Excluding degenerate states'}", "",
                  "| L | " + " | ".join(f"eps={e}" for e in EPS_LIST) + " |",
                  "|---|" + "---|" * len(EPS_LIST)]
        for L in range(1, 5):
            vals = np.array([[results[sd][str(L)][variant][k] if isinstance(results[sd], dict)
                              and str(L) in results[sd] else results[sd][L][variant][k]
                              for sd in seed_dirs] for k in range(len(EPS_LIST))])
            cells = [f"{np.median(v):.1e} [{v.min():.0e},{v.max():.0e}]" for v in vals]
            lines.append(f"| {L} | " + " | ".join(cells) + " |")
        lines.append("")
    rough = np.array([[results[sd][L]["local_roughness"] for sd in seed_dirs]
                      for L in range(1, 5)])
    lines += ["Local profile roughness (median profile distance between state-adjacent "
              "grid points; the resolution floor of the learned fields), median across seeds: "
              + ", ".join(f"L={L}: {np.median(rough[L-1]):.1e}" for L in range(1, 5)), ""]
    with open(os.path.join(OUT_DIR, "margin_curves_mountaincar.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
