"""Local (one-step) certificate: per-(s,a) certified ambiguity in MountainCar.

For candidate models restricted to one-step locality (||f'(s,a) - s|| <= R,
as physical dynamics and the residual WM parameterisation satisfy), every
model whose Bellman residual at (s,a) is <= tau'(s,a) must place its successor
inside the CONSISTENT SET
    C(s,a) := { y in B(s, R) : max_l |m_hat_l(y) - Q_hat(s,a,g_l)| <= tau'(s,a) },
so any two such models (the true kernel included, when Q is that accurate)
disagree at (s,a) by at most diam C(s,a). We take tau'(s,a) =
max(r_wm, r_true)(s,a) + slack, where slack absorbs grid resolution
(local field roughness), and report the distribution of diam C(s,a).

Run from repo root: .venv/bin/python scripts/local_certificate_mountaincar.py
Outputs: markdown + JSON to ../pq-neurips/rebuttal/.
"""

import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import jax
import jax.numpy as jnp
import numpy as np

from scripts.certificate_loop_mountaincar import (
    Agent, load_wm, true_dynamics, state_grid, NORM, X_MIN, V_MIN)

SEEDS_DIR = os.path.join(REPO, "outputs/mountaincar-position/seeds_20260604_100625")
OUT_DIR = os.path.join(os.path.dirname(REPO), "pq-neurips/rebuttal")
R_LOC = 0.06        # locality radius (normalized); one-step bound is ~0.046
SLACK = 2e-3        # grid-resolution slack added to tau' (local field roughness)
ONE_STEP = 0.046    # normalized one-step displacement bound (reference)


def main():
    n_side, S = state_grid()
    spacing = 1.0 / (n_side - 1) * (1000 / (n_side - 1) / (1000 / (n_side - 1)))  # noqa
    S_norm = np.stack([(S[:, 0] - X_MIN) / NORM[0], (S[:, 1] - V_MIN) / NORM[1]], axis=1)
    grid_step = S_norm[n_side, 0] - S_norm[0, 0]  # spacing along x (row-major ij)
    rad = int(np.ceil(R_LOC / grid_step))
    offs = [(di, dj) for di in range(-rad, rad + 1) for dj in range(-rad, rad + 1)
            if (di * di + dj * dj) * grid_step ** 2 <= R_LOC ** 2]
    offs = np.array(offs)
    print(f"grid {n_side}^2, spacing {grid_step:.4f}, ball radius {rad} cells, "
          f"{len(offs)} neighbours")

    seed_dirs = sorted(glob.glob(os.path.join(SEEDS_DIR, "seed_*")))
    med_all, p90_all, frac_step_all, med_err_all = [], [], [], []
    for sd in seed_dirs:
        t0 = time.time()
        ag = Agent(sd)
        f_wm = load_wm(sd)
        prof = ag.m_profile(S)                       # (n, L)
        q_S = ag.q_per_goal(S)                       # (n, |A|, L)

        diams, errs = [], []
        ij = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side),
                                  indexing="ij"), axis=-1).reshape(-1, 2)
        for a in (0, 1, 2):
            F_true = true_dynamics(S, a)
            a_oh = jax.nn.one_hot(jnp.full(S.shape[0], a), 3)
            F_wm = np.asarray(f_wm(jnp.asarray(S, dtype=jnp.float32), a_oh))
            r_wm = np.abs(ag.m_profile(F_wm) - q_S[:, a, :]).max(axis=1)
            r_true = np.abs(ag.m_profile(F_true) - q_S[:, a, :]).max(axis=1)
            tau = np.maximum(r_wm, r_true) + SLACK
            errs.append(np.linalg.norm((F_wm - F_true) / NORM, axis=1))

            # subsample states (stride 2 in each dim) for the neighbour pass
            sub_mask = (ij[:, 0] % 2 == 0) & (ij[:, 1] % 2 == 0)
            sub_idx = np.where(sub_mask)[0]
            # neighbour flat indices, boundary-clipped: (n_sub, n_off)
            nb_i = np.clip(ij[sub_idx, 0][:, None] + offs[None, :, 0], 0, n_side - 1)
            nb_j = np.clip(ij[sub_idx, 1][:, None] + offs[None, :, 1], 0, n_side - 1)
            nb = (nb_i * n_side + nb_j)
            d_sub = np.empty(len(sub_idx))
            CH = 2000
            for c0 in range(0, len(sub_idx), CH):
                sl = slice(c0, c0 + CH)
                mism = np.abs(prof[nb[sl]] - q_S[sub_idx[sl], a, :][:, None, :]).max(-1)
                ok = mism <= tau[sub_idx[sl], None]           # (ch, n_off)
                pts = S_norm[nb[sl]]                          # (ch, n_off, 2)
                big = np.where(ok[..., None], pts, np.nan)
                span = np.nanmax(big, axis=1) - np.nanmin(big, axis=1)
                d = np.hypot(span[:, 0], span[:, 1])
                d_sub[sl] = np.where(ok.any(axis=1), d, 0.0)
            diams.append(d_sub)
        diams = np.concatenate(diams); errs = np.concatenate(errs)
        med, p90 = np.percentile(diams, [50, 90])
        frac = float((diams <= ONE_STEP).mean())
        med_all.append(med); p90_all.append(p90); frac_step_all.append(frac)
        med_err_all.append(float(np.median(errs)))
        print(f"{os.path.basename(sd)}: certified ambiguity med={med:.3f} "
              f"p90={p90:.3f} frac<=one-step={frac:.2%} "
              f"(median actual err {np.median(errs):.1e}) ({time.time()-t0:.0f}s)")

    summary = {
        "R_loc": R_LOC, "slack": SLACK, "one_step": ONE_STEP,
        "median_ambiguity": float(np.median(med_all)),
        "p90_ambiguity": float(np.median(p90_all)),
        "frac_leq_one_step": float(np.median(frac_step_all)),
        "median_actual_err": float(np.median(med_err_all)),
        "per_seed_median": med_all,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "local_certificate_mountaincar.json"), "w") as f:
        json.dump(summary, f, indent=1)
    md = ["# Local (one-step) certificate — MountainCar, 10 seeds", "",
          f"Candidates restricted to one-step locality (R = {R_LOC} normalized; "
          f"true one-step bound {ONE_STEP}). Certified ambiguity at (s,a) = "
          f"diameter of the consistent set C(s,a) (threshold max(r_wm, r_true) + {SLACK} grid slack).", "",
          f"- Median certified ambiguity: **{summary['median_ambiguity']:.3f}** "
          f"(vs 0.5 for the global certificate)",
          f"- p90: {summary['p90_ambiguity']:.3f}",
          f"- Fraction of (s,a) certified to within one step (0.046): "
          f"**{summary['frac_leq_one_step']:.1%}**",
          f"- Median actual successor error, for reference: {summary['median_actual_err']:.1e}", ""]
    with open(os.path.join(OUT_DIR, "local_certificate_mountaincar.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
