# Extended rebuttal notes — recovery quality vs Q-error, uniform vs distributional bounds

Working reference for the two reviewer questions. Full detail here; the concise,
submission-ready response is a separate file, **`notes/rebuttal_response.md`**.

- **R1 (empirical):** How does recovery quality scale with Q-function estimation error?
- **R2 (theoretical):** Is P-learning robust to *realistic* (nonuniform, worst
  off-distribution) Q-error? Do the guarantees survive a **distributional /
  on-support** Q-error bound rather than the **uniform** `‖Q(s,a,·)−Q^π(s,a,·)‖₁ ≤ ε`
  over all `(s,a)` (`results.tex:7`), or is there a counterexample?

Notation: finite goal-conditioned MDP, kernel `P:𝒮×𝒜→Δ(𝒮)`, goals `𝒢`
(`L=|𝒢|`), goal-conditioned policy `π_g`, reward `r(s,g)`, discount `γ`,
`m=|𝒜|`, `n=|𝒮|`. Bellman matrix `M∈ℝ^{L×n}`, `M_{g,s'}=r(s',g)+γV^{π_g}(s',g)`,
so for each `(s,a)`: `M P(s,a)=Q^π(s,a)` (the linear system P-learning inverts,
`P=M⁺Q`).

---

## Part A — Empirical: recovery quality vs Q-estimation error

### A.0 Reframing "on/off-support" (what the reviewer actually means)
The relevant axis is **how Q-error is aggregated**: **uniformly** over state
space vs **weighted by the agent's visitation distribution** `d` (the "state-
action distribution used to train the original agent"). It is NOT the world
model's reset/training region. Under this reading MountainCar suffices: the car
concentrates occupancy in part of the (position, velocity) box, so uniform vs
visitation-weighted Q-error genuinely differ.

### A.1 What was built (branch `rebuttal-recovery-vs-qerror`)
Metric-only refactors (behaviour-preserving):
- `eval/world_model.py::dynamics_nmse(...)` — WM dynamics NMSE on any caller-
  supplied state set, optional per-state `weights`, no disk writes. `compare_dynamics`
  now calls it (headline numbers unchanged).
- `eval/value_iteration.py::_metrics(a,b,weights=None)` — weighted MSE numerator,
  **uniform-scale variance denominator** (see A.3), and
  `q_nmse_by_weighting(...)` returning Q_NMSE vs `Q^π` (policy) and `Q*`
  (optimal), each uniform + visitation-weighted, reusing the VI primitives.
- `eval/visitation.py::grid_visitation(...)` — occupancy `w(cell)` of PQN's
  greedy policy on true dynamics from `env.reset()` starts, unioned over training
  goals (the empirical `S_o = ∪_g Reach(π_g)`).
- `eval/track_recovery.py::track_recovery_for_run(...)` — per checkpoint:
  Q_NMSE (uniform/weighted/optimal) + WM_NMSE (uniform/weighted, WM trained fresh
  from that checkpoint's Q). Goal-independent precompute + `Q*` + visitation are
  computed once and reused across checkpoints.
- `eval/run_context.py::build_context(config_path)` — MC/Reacher env + dynamics +
  lift-fn wiring (mirrors run.py).
- Scripts: `scripts/recovery_vs_qerror.py` (A1), `scripts/dist_vs_uniform.py` (A2).

Q-quality axis is free: the ~20 intermediate `checkpoints/step_*.pkl` per run are
increasing-Q-quality snapshots of one training run (the final one duplicates
`pqn_checkpoint.pkl` and is dropped).

### A.2 How to run (GPU)
```
uv sync --extra cuda
# A1 — scaling (headline); ≥5 seeds recommended
uv run python run.py --config configs/mountaincar-position.py --num_seeds 5 --save_pqn_checkpoints
uv run python scripts/recovery_vs_qerror.py --config configs/mountaincar-position.py \
    --run_dirs outputs/mountaincar-position/seeds_*/seed_*  # -> outputs/mountaincar/recovery_vs_qerror/
# A2 — distributional vs uniform (reuses the tracking npz)
uv run python scripts/dist_vs_uniform.py --config configs/mountaincar-position.py \
    --run_dirs outputs/mountaincar-position/seeds_*/seed_* --reuse
```
On GPU WM training is ~10 s/checkpoint (vs ~15 min CPU). Reacher is the natural
extension (already wired in `build_context`): its 4D-effective grid makes Q_NMSE
the bottleneck (~1–3 min/checkpoint).

### A.3 Metric detail — why the weighted denominator is held fixed
Visitation-weighted NMSE with a *weighted* variance denominator blows up when the
reference is nearly constant on the visited support (weighted `Var(Q^π)→0`; we
observed `9.9e5`). Fix: keep the **uniform** `Var` as a fixed scale and only
re-weight the **MSE numerator**:
`NMSE_visit = E_{s∼w}[(Q−Q̂)²] / Var_uniform(Q^π)`. Uniform and weighted NMSE
then share a denominator and differ only in *where* error is measured — directly
comparable. Same fix in `dynamics_nmse` for WM.

### A.4 CPU smoke results (2 checkpoints, MountainCar, goal 0, tiny WM)
Pipeline validated end-to-end. Uniform Q_NMSE reproduces the standalone
per-checkpoint values exactly (step 5: 1.034, step 20: 0.939). Weighted metrics
are sane and on-thesis:

| step | Q_NMSE unif | Q_NMSE visit | WM_NMSE unif | WM_NMSE visit |
|---|---|---|---|---|
| 5  | 1.034 | 0.821 | 0.0133 | 0.0035 |
| 20 | 0.939 | 0.269 | 0.0224 | 0.0072 |

Both Q-error and recovery-error are **smaller on the visited region** than
uniformly — the empirical signature of A.5. (Absolute WM_NMSE is inflated by the
50-step smoke WM; the full GPU run uses 20 000 steps.)

### A.5 On-support weighting (`--weight_mode`, `--visitation_from`)
Each checkpoint is weighted by **its own** greedy-policy visitation
(`visitation_from="self"`, default). The on-support region is the reachable
**set** `S_o` — a binary mask of the cells covering the top 99% of visitation
mass (`weight_mode="mask"`, default), matching the theory (`S_o` is a set) and
avoiding the high variance of density-weighting, which peaked on the few
goal-boundary cells where `Q` is hardest. (`weight_mode="density"` and
`visitation_from="final"` remain available.)

**A refuted sub-hypothesis (kept as a note).** We initially expected the
reachable set to *broaden* over training (`visit_frac ↑`), which would explain
`WM_NMSE_uniform` falling while `WM_NMSE_visit` stayed flat. The data refute it:
`visit_frac` is flat/slightly-decreasing (ρ=−0.39). This is correct behaviour,
not a bug — MountainCar terminates on goal, so a *better* policy reaches goals
faster and traces a thinner, roughly-constant manifold. Global recovery improves
because the model gets better *everywhere*, not because coverage grows. Do not
use the broadening framing.

### A.6 Observed results (5 seeds, GPU, 20 checkpoints; mean ± SE)
Headline signals are strong and support both reviewer points.

- **R1 — scaling (all checkpoints):** recovery scales with Q-error,
  `Spearman(Q_NMSE, WM_NMSE) = 0.83 ± 0.03` (per-seed; pooled 100 pts
  ρ=0.83, p=8e-27); `WM_NMSE ≪ Q_NMSE` throughout (≈31× at convergence:
  7.0e-3 vs 0.22).
- **R2 — distributional, at the converged agent (step ≥ 200, "the trained
  agent"):** on the reachable set vs uniform, ratio of means over the region —
  `WM_NMSE` **60× ± 7** smaller (1.3e-4 vs 7.0e-3). So for the trained agent the
  on-support bound is the operative one, as Theorem B2 predicts.
- **Recovery is tens-of-× better on `S_o` than uniform at every checkpoint**, even
  early ones where global Q_NMSE is ~1–2 — recovery on the visited region is
  robust even where `Q` is globally poor.

**Q-error: use BOTH metrics, each in its correct role (not Q\* alone).**
The theorem's ε is `‖Q − Q^π‖` (Bellman residual, results.tex:7), NOT distance to
optimal. The two metrics say complementary things and the strongest rebuttal uses
both:
- **`Q*`-error is 2.3× ± 0.2 larger off-support** (0.049 on vs 0.108 uniform) —
  Q *does* degrade off-distribution in the usual sense; this **concedes** the
  reviewer's premise.
- **`Q^π`-error is ≈1.2× ± 0.2** (≈uniform) — the error our bound actually depends
  on is **not** concentrated off-support; the feared off-support blow-up doesn't
  occur for the relevant metric.
- Do **not** present `Q*` alone as "on-support Q-error is smaller" — that swaps
  distance-to-optimal for approximation error and a theory reviewer will catch it.
- **Mid-training (steps ~50–185) the on-support `Q^π`-error rises to ≈/above
  uniform**: during the Q-instability spike the greedy policy chases states where
  its own `Q` is wrong. Real, and not part of the R2 claim (those checkpoints are
  not the trained agent). `WM_NMSE_visit` stays low regardless — a sharper form of
  the paper's WM≫Q result.
- `visit_frac` trend across seeds is `−0.16 ± 0.13` (flat) — broadening firmly
  refuted (A.5).

### A.7 A3 — controlled off-support Q-perturbation (the direct robustness test)
A1/A2 are *descriptive* (they characterise the natural error/recovery landscape).
They do **not** stress-test recovery against the reviewer's exact scenario — and
in fact the natural setting doesn't create it (`‖Q−Q^π‖` is ~uniform, A.6). A3
manufactures it: take a converged `Q`, inject a deterministic, region-gated
perturbation into the `Q` the WM training sees (off `S_o`, or on `S_o` as the
control), retrain, and measure on-support recovery vs the injected Q-error.
(`scripts/q_perturb.py`; hook `q_value_fn` in `training/wm.py`.)

**Full-fidelity result (20000 WM steps): the experiment traces the B2→B3
boundary.** On the reduced MDP over `S_o`, sweeping injected error magnitude:
- **Robust regime (small–moderate off-support error, the reviewer's realistic
  case).** At matched injected `Q_NMSE ≈ 0.1`, off-support perturbation leaves
  on-support recovery essentially intact (`WM_NMSE ≈ 3e-5`) while the *same*
  on-support error degrades it **~700×** (`≈ 2e-2`). Recovery is governed by
  on-support Q-accuracy — the direct evidence for B2.
- **Breakdown regime (large off-support error).** Once off-support `Q_NMSE ≳ 0.5`
  (beyond the natural ~0.2), off-support error *also* breaks recovery
  (`4e-2` at 0.53, up to `0.9` at 2.1): the WM, no longer confined to `S_o`,
  finds a spurious consistent kernel off-support — **B3**. The off-support (blue)
  curve stays ~flat then rises steeply, crossing the on-support (red) curve.
- **Why the earlier smoke (200 steps) looked cleanly flat:** under-trained, the WM
  couldn't exploit the spurious off-support minima. The full run reveals the
  threshold — which is a *feature*: A3 empirically exhibits both the B2 guarantee
  and the B3 limit.
- **Caveat:** the injected perturbation is an adversarial (unstructured cos) field,
  so it likely *overstates* off-support damage vs. a real Q-network's structured
  extrapolation — the true robustness threshold is plausibly higher.
- **Naive whole-space training shows the mirror pattern** (`q_perturb_uniform`):
  robust to on-support error, *sensitive* to off-support error (degrades to ~0.34
  at off-support `Q_NMSE 2`, flat under on-support error). Where you train decides
  which error matters — reinforcing that the on-support restriction is what B2
  buys.

**Why the WM breaks at all (and why it does NOT contradict B2).** Training
on-support constrains the sampled states `s∈S_o`, not the WM's *predicted
successor* `s'=WM(s,a)`. The Bellman loss bootstraps through `max Q(s',·)`, and
`s'` is unconstrained; so off-support Q enters whenever the WM predicts a
successor that leaves `S_o`. The theorem's estimator `M_{S_o}⁺Q` reads Q only on
`S_o` columns *by construction* — the WM is an *unconfined* approximation of it.

**`--confine` (project off-support Q-queries onto `S_o`) confirms this exactly
(full GPU run).** The bootstrap then never reads off-support Q, and off-support
perturbation has **zero** effect: on-support `WM_NMSE ≈ 4.3e-6` is *identical* for
injected off-support error `0 → NMSE 2.1` (a dead-flat line); on-support
perturbation degrades it monotonically (`4e-3 → 4e-2 → 0.16 → 0.22`). The
**realistic-range** unconfined run (injected `NMSE ≤ 0.02`) shows the same
contrast even without confinement: off-support flat (`~1e-5`), on-support rising
(`→ 3.7e-3`, ~100×). This is the definitive B2 confirmation: the estimator the
theorem analyses is exactly immune to off-support Q-error; the unconfined WM
inherits the robustness at realistic error and breaks only when off-support error
grows large enough to lure its successors off `S_o` (B3).

Honest framing for the response: the **confined estimator** (`M_{S_o}⁺Q`, i.e.
`--confine`) is provably and empirically immune to off-support Q-error at any
magnitude; the **unconfined WM** inherits this at realistic error and breaks only
under large adversarial off-support error (the B3 boundary), because it is free
to leave `S_o`.

---

## Part B — Theory: distributional recovery on the policy-reachable set

### B.1 Where the uniform bound is load-bearing
The uniform `ε` is used at exactly two proof steps, both of which quantify over
**all columns of `M` = all next-states × goals**:
- stochastic `P̂=M⁺Q`: `‖E‖₁ ≤ γmε` in `prop:finite-approximate`
  (`appendix-finite.tex:292–296`);
- deterministic column-matching: `‖M_{·,k}−M^π_{·,k}‖₁ ≤ γmε` in
  `prop:finite-deterministic-upper` (`appendix-finite.tex:71–85`).
Recovering `P(s,a)` at even one `(s,a)` needs `M`, whose column `k` is
`V=max_a Q` at successor state `s_k` — accuracy at the *successors*, not the
sampling distribution.

### B.2 Positive result — reduced MDP on the reachable set
The key realization (no new axiom needed): the policy-reachable set is
**forward-invariant by construction**, so restricting to it is a genuine reduced
MDP and the paper's theorem applies verbatim with `ε`, `M` restricted.

**Reachable set + on-policy actions.** `Reach(π_g)` = states with positive
occupancy under `π_g` from `ρ`; `S_o = ⋃_{g∈𝒢} Reach(π_g)`;
`A_o(s) = {π_g(s) : g∈𝒢}` = the actions the policies take at `s` (state-dependent).
The reduced MDP is over the **pairs** `{(s,a) : s∈S_o, a∈A_o(s)}`, which is
forward-invariant: `supp P(s,π_g(s)) ⊆ Reach(π_g) ⊆ S_o`, and every `S_o`
successor has `A_o`-actions available. **`S_o×𝒜` is NOT forward-invariant** — an
off-policy `a∉A_o(s)` can send `P(s,a)` off `S_o`, which is exactly why `A_o` is
needed (maximal valid action set: any `a` with `supp P(s,a)⊆S_o`, which contains
`A_o(s)`). Caveat: this guarantees the *visited* dynamics; recovering `P(s,a)` for
counterfactual actions needs their successors covered too.

**On-support error.** `ε_o = sup{ ‖Q(s,a,·)−Q^π(s,a,·)‖₁ : s∈S_o, a∈A_o(s) }`,
over **all goals**. (Tighter than `S_o×𝒜`: `M_{S_o}`'s columns only need
`V^{π_{g'}}(s_k,g')=Q(s_k,π_{g'}(s_k),g')` — on-support states, on-policy actions.)
No constraint off `S_o`.

**Theorem (reduced-MDP recovery).** Let `M_{S_o}∈ℝ^{L×|S_o|}` be `M` restricted to
`S_o`-columns, full column rank. Then for every `(s,a)` with `s∈S_o, a∈A_o(s)`,
`P̂(s,a)=M_{S_o}⁺Q(s,a)` (a distribution on `S_o`) satisfies
`‖P̂(s,a)−P(s,a)‖₁ ≤ ‖M_{S_o}⁺‖₁ (1+γm) ε_o`.
Deterministic case: column-matching over `S_o` recovers `P(s,a)` exactly whenever
`ε_o < gap(M_{S_o})/(2(1+γm))`.

**Scope.** These are **finite-MDP** results (the paper's approximate bounds are
finite; the continuous results are exact-identifiability only). MountainCar is
deterministic + continuous, so the **deterministic** bound is the apt one; we
verify it numerically on a finite MDP (`theory_checks.py`) and use MountainCar to
illustrate the phenomenon. Continuous approximate bounds are future work.

**Proof sketch.** Since `supp P(s,a)⊆S_o`, `P(s,a)` is a vector on `S_o` and
`M_{S_o}P(s,a)=Q^π(s,a)`. The estimated `M̂_{S_o}=M_{S_o}+E_{S_o}` has column
errors `E_{S_o}[:,k]=γΣ_{a'}π(a'|s_k,·)(Q−Q^π)(s_k,a',·)`; as `s_k∈S_o`,
`‖E_{S_o}‖₁ ≤ γm ε_o`. This is exactly the `‖E‖₁≤γmε` step of
`prop:finite-approximate`, now with every column in `S_o`, so only `ε_o` is
needed. The remaining argument (`M̂_{S_o}⁺M_{S_o}^π=I−M̂_{S_o}⁺E_{S_o}`, full rank
via `‖M_{S_o}⁺‖₁‖E_{S_o}‖₁<1`) carries over with `M→M_{S_o}, ε→ε_o`. ∎

**Goal-union subtlety (the important detail).** Column `s_k` of `M_{S_o}` is
`r(s_k,g)+γV^{π_g}(s_k,g)` for *every* `g∈𝒢` — `M` couples all goals. So the
on-support bound must hold at `(s_k,·,g)` for **all goals**, i.e. over
`S_o×𝒜×𝒢`. A state reached only under goal `g_m` is off-distribution for
`Q(·,·,g_l)`; hence the reduced state space must be the **union** `⋃_g Reach(π_g)`,
and when `Reach(π_g)` is `g`-dependent, `S_o` is strictly larger than any single
`Reach(π_g)`.

**Identifiability bonus.** Full column rank now concerns `M_{S_o}` (`L×|S_o|`), so
it needs only `|𝒢| ≥ |S_o|` (goals spanning `S_o`), not `|𝒢| ≥ |𝒮|` — the
discrete analog of the paper's continuous "coverage" story (`results.tex:5`), and
of the N-local results (`prop:local`).

### B.3 Counterexample — forward-coverage across goals is necessary
Exactness of `Q` on the reachable set is **not** sufficient if an off-support
successor has inaccurate `Q`.

Explicit instance (single action for clarity; `notes/theory_checks.py::check_B3`):
`𝒮={0,1,2}`, start at `0`, true transition `0→1`, states `1,2` absorbing.
Reachable set `={0,1}`; state `2` is off-support. With exact `M`, columns
`M[:,0],M[:,1],M[:,2]` are generically independent, so `P(0)=δ_1` is identifiable.
Corrupt `Q` **only at off-support state 2** so the estimated `M̂[:,2]=M[:,1]`.
Then `M̂` has a coincident column and both
`P=δ_1` and `P'=δ_2` satisfy `M̂ p = Q(0)=M[:,1]` → recovery ambiguous.
Reachable-set `Q` error is 0; off-support column error is `Ω(1)`; `‖P−P'‖₁=2`.
This is the realistic failure mode the reviewer names (nonuniform Q-error, worst
off-distribution), and it delimits B.2: recovery needs coverage of the successors
of visited states, across all goals.

### B.4 Reconciliation with the empirics
On-policy data concentrates on `S_o` and (with HER-style goal relabelling) tends
to cover the reachable states across training goals, so the B.2 conditions
approximately hold and Q-error is smallest on `S_o×𝒢` (A.2/A.5). This is why WM
recovery is orders of magnitude better than the *uniform* Q-error would suggest
(paper's `NMSE_WM=1.2e-4` vs `NMSE_Q=5.7e-1`). **A.7 (A3) demonstrates the
B.2→B.3 boundary causally**: on the reduced MDP over `S_o`, on-support recovery is
~700× more sensitive to on-support than off-support Q-error at realistic
magnitudes (B.2), and off-support error breaks recovery only once it grows large
enough to admit a spurious kernel (B.3).

### B.5 Numerical verification (`notes/theory_checks.py`, both PASS)
- **B2:** on-support `ε_o=1e-3` → max recovery error `1.4e-4 ≤` bound `1.5e-3`.
- **B3:** reachable-set Q exact; off-support column error `34.2`; true and fake
  kernels both residual `0` (non-identifiable); `‖P−P'‖₁=2`.

### B.6 Where it lands in the paper
New appendix subsection near `appendix-finite.tex` (`prop:finite-approximate` /
`prop:local`): Theorem B.2 + Counterexample B.3; one paragraph in `results.tex`
after the ε-approximation definition; scaling + distributional figures into
`experiments.tex` / `appendix-experiments.tex`.

---

## Part C — Reviewer response
The concise, submission-ready reviewer response lives in its own file:
**`notes/rebuttal_response.md`**. Keep it short (numbers, no plots); this extended
file is the working reference behind it.
