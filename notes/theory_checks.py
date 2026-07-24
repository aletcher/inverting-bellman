"""Numerical sanity checks for the rebuttal theory (Part B).

B2 (positive): restricting P-learning to the policy-reachable set S_o yields a
   reduced MDP where the paper's approximate-recovery bound holds with the
   ON-SUPPORT error e_o and M restricted to S_o columns:
       ||P_hat(s,a) - P(s,a)||_1 <= ||M_{So}^+||_1 (1 + gamma m) e_o.

B3 (counterexample): exactness of Q on the reachable set is NOT sufficient.
   If Q is inaccurate at an off-support successor, the estimated Bellman matrix
   develops a coincident column and recovery of an on-support transition becomes
   ambiguous: two kernels agree with Q on the whole reachable set yet differ by
   ||.||_1 = Omega(1).

Pure numpy, no repo deps. Run: python notes/theory_checks.py
"""
import numpy as np

rng = np.random.default_rng(0)


def value_and_M(P, R, pi, gamma):
    """Per-goal Bellman matrix M (L x n): M[g,s'] = r(s',g) + gamma V^pi(s',g).

    P: (n,m,n) kernel, R: (n,L) reward r(s,g), pi: (n,L) int action per (state,goal).
    Returns M (L,n) and V (n,L).
    """
    n, m, _ = P.shape
    L = R.shape[1]
    V = np.zeros((n, L))
    for g in range(L):
        Ppi = P[np.arange(n), pi[:, g], :]           # (n,n) under this goal's policy
        V[:, g] = np.linalg.solve(np.eye(n) - gamma * Ppi, R[:, g])
    M = (R + gamma * V).T                              # (L,n)
    return M, V


def Q_of(P, M):
    """Q(s,a,g) = sum_s' P(s,a,s') M(s',g)  -> shape (n,m,L)."""
    return np.einsum("sat,gt->sag", P, M)


# ─────────────────────────────────────────────────────────────────────────────
# B2 — reduced-MDP recovery bound on the policy-reachable set
# ─────────────────────────────────────────────────────────────────────────────
def check_B2():
    # MDP with a forward-closed reachable block {0,1,2} and an off-support tail
    # {3,4}. Start in {0}; the policy stays inside {0,1,2}, so S_o = {0,1,2}.
    n, m, L, gamma = 5, 3, 6, 0.9
    So = [0, 1, 2]
    P = np.zeros((n, m, n))
    for s in range(n):
        for a in range(m):
            if s in So:
                # transitions stay within S_o (forward-closed by construction)
                probs = rng.random(len(So)); probs /= probs.sum()
                P[s, a, So] = probs
            else:
                probs = rng.random(n); probs /= probs.sum()
                P[s, a, :] = probs
    R = rng.standard_normal((n, L))
    pi = rng.integers(0, m, size=(n, L))
    # keep policy inside S_o for S_o states so reachability = S_o
    M, _ = value_and_M(P, R, pi, gamma)
    Q = Q_of(P, M)

    # On-support eps-approximate Q: bounded L1 perturbation over goals, only on
    # S_o (off-support Q may be arbitrary — we never use it here).
    eps_o = 1e-3
    Qhat = Q.copy()
    for s in So:
        for a in range(m):
            d = rng.standard_normal(L); d *= (eps_o / np.sum(np.abs(d)))  # ||d||_1 = eps_o
            Qhat[s, a, :] = Q[s, a, :] + d

    M_So = M[:, So]                                   # (L, |So|)
    M_So_pinv = np.linalg.pinv(M_So)
    max_err, bound = 0.0, 0.0
    for s in So:
        for a in range(m):
            p_hat = M_So_pinv @ Qhat[s, a, :]          # recover on S_o
            p_true = P[s, a, So]
            err = np.sum(np.abs(p_hat - p_true))
            max_err = max(max_err, err)
    pinv_1 = np.max(np.sum(np.abs(M_So_pinv), axis=1))  # ||.||_1 (max abs row sum of columns)
    bound = pinv_1 * (1 + gamma * m) * eps_o
    ok = max_err <= bound + 1e-9
    print(f"[B2] on-support eps_o={eps_o:.1e}  max recovery err={max_err:.3e}  "
          f"bound={bound:.3e}  -> {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# B3 — off-support Q-error breaks recovery of an on-support transition
# ─────────────────────────────────────────────────────────────────────────────
def check_B3():
    # States {0,1,2}; from state 0 (start), the TRUE transition is 0 -> 1.
    # States 1,2 absorb. Reachable set under true dynamics = {0,1}; state 2 is
    # off-support. One action for clarity (recover a single transition).
    n, L, gamma = 3, 4, 0.9
    P = np.zeros((n, 1, n))
    P[0, 0, 1] = 1.0                                   # 0 -> 1 (true)
    P[1, 0, 1] = 1.0                                   # absorbing
    P[2, 0, 2] = 1.0                                   # absorbing (off-support)
    R = rng.standard_normal((n, L))
    pi = np.zeros((n, L), dtype=int)
    M, _ = value_and_M(P, R, pi, gamma)                # true Bellman matrix (L x 3)
    Q = Q_of(P, M)                                     # Q(0,0,:) = M[:,1]

    # With EXACT M, columns M[:,0], M[:,1], M[:,2] are generically independent,
    # so P(0,0) = delta_1 is identifiable. Now corrupt Q ONLY at the off-support
    # state 2 so the ESTIMATED column M_hat[:,2] coincides with M[:,1].
    M_hat = M.copy()
    M_hat[:, 2] = M[:, 1]                              # adversarial off-support error
    offsupport_col_err = np.sum(np.abs(M_hat[:, 2] - M[:, 2]))

    # Reachable-set Q is untouched: Q(0,0,:) and the values on {0,1} are exact.
    q0 = Q[0, 0, :]                                    # = M[:,1]
    # Two kernels consistent with the ESTIMATED M_hat at state 0:
    p_true = np.array([0.0, 1.0, 0.0])                 # 0 -> 1
    p_fake = np.array([0.0, 0.0, 1.0])                 # 0 -> 2
    res_true = np.sum(np.abs(M_hat @ p_true - q0))
    res_fake = np.sum(np.abs(M_hat @ p_fake - q0))
    l1_gap = np.sum(np.abs(p_true - p_fake))
    ambiguous = (res_true < 1e-9) and (res_fake < 1e-9)
    print(f"[B3] reachable-set Q error=0 (exact); off-support column error="
          f"{offsupport_col_err:.2f}")
    print(f"[B3] residual(true P)={res_true:.2e}  residual(fake P)={res_fake:.2e}  "
          f"||P-P'||_1={l1_gap:.1f}  -> {'PASS (non-identifiable)' if ambiguous else 'FAIL'}")
    return ambiguous


if __name__ == "__main__":
    ok2 = check_B2()
    ok3 = check_B3()
    print(f"\nB2 {'PASS' if ok2 else 'FAIL'} | B3 {'PASS' if ok3 else 'FAIL'}")
