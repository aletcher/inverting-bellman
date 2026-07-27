"""Certificate loop: does the margin curve delta(eps) explain the WM accuracy?

Certificate (pointwise, two-line proof): for candidate successor functions
f1, f2 with per-(s,a) Bellman residuals r_i(s,a) := max_l |m_hat_l(f_i(s,a)) -
Q_hat(s,a,g_l)|, the triangle inequality gives
    max_l |m_hat_l(f1(s,a)) - m_hat_l(f2(s,a))| <= r_1(s,a) + r_2(s,a),
while if ||f1(s,a) - f2(s,a)|| >= eps then, by definition of the margin
delta(eps) = min_{||x-y||>=eps} max_l |m_hat_l(x) - m_hat_l(y)|, the same
quantity is >= delta(eps). Hence
    delta(eps) > r_1(s,a) + r_2(s,a)  =>  ||f1(s,a) - f2(s,a)|| < eps.
We apply this with f1 = extracted WM, f2 = true dynamics, and check the
predicted bound eps*(r_1 + r_2) against the actual successor error, per (s,a),
across 10 seeds. delta(eps) is the K-NN grid estimate (may overestimate the
continuum margin), so the validity fraction is a genuine test.

Run from repo root:  .venv/bin/python scripts/certificate_loop_mountaincar.py
Outputs: JSON + markdown to ../pq-neurips/rebuttal/.
"""

import glob
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
from scipy.spatial import cKDTree

from training.pqn_utils import make_q_network, make_goal_repr
from training.wm import make_world_model, apply_wm
from scripts.margin_curve_mountaincar import (
    state_grid, degenerate_mask, X_MIN, X_MAX, V_MIN, V_MAX, CHUNK)

SEEDS_DIR = os.path.join(REPO, "outputs/mountaincar-position/seeds_20260604_100625")
OUT_DIR = os.path.join(os.path.dirname(REPO), "pq-neurips/rebuttal")
EPS_GRID = np.concatenate([np.arange(0.01, 0.1, 0.01), np.arange(0.1, 0.52, 0.02)])
K_NN = 200
NORM = np.array([X_MAX - X_MIN, V_MAX - V_MIN])


def true_dynamics(S, a):
    x, v = S[:, 0], S[:, 1]
    v2 = np.clip(v + (a - 1) * 0.001 - 0.0025 * np.cos(3 * x), V_MIN, V_MAX)
    x2 = np.clip(x + v2, X_MIN, X_MAX)
    v2 = v2 * (1 - ((x2 == X_MIN) & (v2 < 0)))
    return np.stack([x2, v2], axis=1)


class Agent:
    """Learned Q-network + reward/termination helpers for one seed."""

    def __init__(self, seed_dir):
        with open(os.path.join(seed_dir, "pqn_config.json")) as f:
            self.cfg = json.load(f)
        with open(os.path.join(seed_dir, "pqn_checkpoint.pkl"), "rb") as f:
            ckpt = pickle.load(f)
        self.net = make_q_network(self.cfg)
        self.vars = {"params": ckpt["params"], "batch_stats": ckpt["batch_stats"]}
        self.gamma = self.cfg["GAMMA"]
        self.a_thr = self.cfg["REWARD_A"]
        self.goals = np.asarray(self.cfg["GOALS"], dtype=np.float32)
        self.masks = np.asarray(self.cfg["REWARD_MASK"], dtype=np.float32)

        @jax.jit
        def _q(obs, goal, mask):
            gr = make_goal_repr(goal, mask, obs.shape[0], obs.shape[1])
            return self.net.apply(self.vars, obs, gr, train=False)  # (n, |A|)
        self._q = _q

    def q_per_goal(self, X):
        """Q_hat(x, ., g_l) for all goals: (n, |A|, L)."""
        obs = jnp.asarray(X, dtype=jnp.float32)
        out = []
        for gl, mk in zip(self.goals, self.masks):
            q = np.concatenate([np.asarray(self._q(obs[i:i + CHUNK], jnp.asarray(gl),
                                                   jnp.asarray(mk)))
                                for i in range(0, X.shape[0], CHUNK)])
            out.append(q)
        return np.stack(out, axis=2)

    def m_profile(self, X):
        """m_hat_l(x) = r_l + gamma * (1 - r_l) * max_a Q_hat: (n, L)."""
        q = self.q_per_goal(X)          # (n, |A|, L)
        vhat = q.max(axis=1)            # (n, L)
        dist_sq = np.stack([(((X - gl) * mk) ** 2).sum(axis=1)
                            for gl, mk in zip(self.goals, self.masks)], axis=1)
        r = (dist_sq < self.a_thr ** 2).astype(np.float64)
        return r + self.gamma * (1.0 - r) * vhat


def load_wm(seed_dir, state_dim=2):
    wm_dir = sorted(glob.glob(os.path.join(seed_dir, "wm_*")))[-1]
    with open(os.path.join(wm_dir, "wm_checkpoint.pkl"), "rb") as f:
        ck = pickle.load(f)
    wm_cfg = ck["config"] if ck.get("config") else json.load(
        open(os.path.join(wm_dir, "wm_config.json")))
    model = make_world_model(wm_cfg, state_dim)
    residual = wm_cfg.get("RESIDUAL_PREDICTION", True)

    @jax.jit
    def f_wm(S, a_oh):
        return apply_wm(model, ck["params"], S, a_oh, residual=residual)
    return f_wm


def delta_curve(profiles, S_norm):
    """K-NN margin estimate delta(eps) on EPS_GRID (all pairs)."""
    tree = cKDTree(profiles)
    dist, nbr = tree.query(profiles, k=K_NN + 1, p=np.inf)
    dist, nbr = dist[:, 1:], nbr[:, 1:]
    i = np.repeat(np.arange(profiles.shape[0]), K_NN)
    sd = np.linalg.norm(S_norm[i] - S_norm[nbr.ravel()], axis=1)
    pd = dist.ravel()
    return np.array([pd[sd >= e].min() if (sd >= e).any() else np.inf
                     for e in EPS_GRID])


def eps_star(delta, t):
    """Smallest eps in EPS_GRID with delta(eps) > t; inf if none (vectorised)."""
    ok = delta[None, :] > np.asarray(t)[:, None]          # (m, n_eps)
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), -1)
    out = np.where(first >= 0, EPS_GRID[np.maximum(first, 0)], np.inf)
    return out


def main():
    n_side, S = state_grid()
    S_norm = S / NORM - np.array([X_MIN, V_MIN]) / NORM
    seed_dirs = sorted(glob.glob(os.path.join(SEEDS_DIR, "seed_*")))
    rows, per_seed = [], {}
    for sd in seed_dirs:
        t0 = time.time()
        ag = Agent(sd)
        f_wm = load_wm(sd)
        prof_S = ag.m_profile(S)
        delta = delta_curve(prof_S, S_norm)

        q_S = ag.q_per_goal(S)                                    # (n, |A|, L)
        r_wm_all, r_true_all, e_all, mse_all, var_all = [], [], [], [], []
        for a in (0, 1, 2):
            q_sa = q_S[:, a, :]                                   # (n, L)
            F_true = true_dynamics(S, a)
            a_oh = jax.nn.one_hot(jnp.full(S.shape[0], a), 3)
            F_wm = np.asarray(f_wm(jnp.asarray(S, dtype=jnp.float32), a_oh))
            r_wm = np.abs(ag.m_profile(F_wm) - q_sa).max(axis=1)
            r_true = np.abs(ag.m_profile(F_true) - q_sa).max(axis=1)
            e = np.linalg.norm((F_wm - F_true) / NORM, axis=1)
            r_wm_all.append(r_wm); r_true_all.append(r_true); e_all.append(e)
            mse_all.append(((F_wm - F_true) ** 2).mean(axis=1))
            var_all.append(F_true.var(axis=0).mean())
        r_wm = np.concatenate(r_wm_all); r_true = np.concatenate(r_true_all)
        e = np.concatenate(e_all)

        pred = eps_star(delta, r_wm + r_true)
        finite = np.isfinite(pred)
        valid = (e[finite] < pred[finite]).mean() if finite.any() else np.nan
        # dynamics NMSE in raw units, comparable to the paper's metric
        mse = np.mean(np.concatenate(mse_all))
        var = np.mean(var_all)
        stats = {
            "r_wm": np.percentile(r_wm, [50, 90, 99]).tolist() + [float(r_wm.max())],
            "r_true": np.percentile(r_true, [50, 90, 99]).tolist() + [float(r_true.max())],
            "err": np.percentile(e, [50, 90, 99]).tolist() + [float(e.max())],
            "pred_eps": np.percentile(pred[finite], [50, 90, 99]).tolist(),
            "frac_finite_pred": float(finite.mean()),
            "validity": float(valid),
            "nmse": float(mse / var),
            "delta": delta.tolist(),
        }
        per_seed[os.path.basename(sd)] = stats
        rows.append(stats)
        print(f"{os.path.basename(sd)}: med r_wm={stats['r_wm'][0]:.1e} "
              f"med r_true={stats['r_true'][0]:.1e} med err={stats['err'][0]:.1e} "
              f"med pred_eps={stats['pred_eps'][0]:.2f} valid={valid:.3f} "
              f"finite={stats['frac_finite_pred']:.2f} nmse={stats['nmse']:.1e} "
              f"({time.time() - t0:.0f}s)")

    os.makedirs(OUT_DIR, exist_ok=True)
    agg = {k: np.median([r[k] for r in rows], axis=0).tolist()
           for k in ("r_wm", "r_true", "err", "pred_eps")}
    agg["validity"] = float(np.median([r["validity"] for r in rows]))
    agg["frac_finite_pred"] = float(np.median([r["frac_finite_pred"] for r in rows]))
    agg["nmse"] = float(np.median([r["nmse"] for r in rows]))
    with open(os.path.join(OUT_DIR, "certificate_loop_mountaincar.json"), "w") as f:
        json.dump({"meta": {"eps_grid": EPS_GRID.tolist(), "k_nn": K_NN,
                            "grid": f"{n_side}^2", "seeds": len(seed_dirs)},
                   "aggregate_median": agg, "per_seed": per_seed}, f, indent=1)

    md = ["# Certificate loop (MountainCar, 10 seeds, medians across seeds)", "",
          "Pointwise certificate: e(s,a) < eps whenever delta(eps) > r_wm(s,a) + r_true(s,a).",
          "",
          "| quantity | p50 | p90 | p99 | max |",
          "|---|---|---|---|---|",
          "| r_wm (WM Bellman residual) | " + " | ".join(f"{v:.1e}" for v in agg["r_wm"]) + " |",
          "| r_true (Q-hat inconsistency on true dynamics) | " + " | ".join(f"{v:.1e}" for v in agg["r_true"]) + " |",
          "| actual successor error e (normalized) | " + " | ".join(f"{v:.1e}" for v in agg["err"]) + " |",
          "| predicted bound eps* (finite preds) | " + " | ".join(f"{v:.2f}" for v in agg["pred_eps"]) + " | – |",
          "",
          f"Validity of pointwise certificate (finite predictions): {agg['validity']:.1%} "
          f"(finite-prediction fraction {agg['frac_finite_pred']:.1%}); "
          f"WM dynamics NMSE (median): {agg['nmse']:.1e}.", ""]
    with open(os.path.join(OUT_DIR, "certificate_loop_mountaincar.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()


