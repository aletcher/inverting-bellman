# Reviewer response (concise — the text we actually submit)

Numbers below are mean ± SE over 5 MountainCar seeds. No figures are embedded
here — reference the new figure/appendix by label.

---

We thank the reviewers. Both questions concern how P-learning behaves under
imperfect Q-values; we address them with a new experiment and a strengthened
theorem, and will add both to the paper.

**Recovery quality vs Q-estimation error (new experiment).** We treat the ~20
intermediate training checkpoints of an agent as a sequence of Q-functions of
increasing quality, and measure per checkpoint both the Q-value error (Q_NMSE,
against the true value of the induced policy) and the recovered-model error
(WM_NMSE). Recovery quality improves monotonically with Q-quality — Spearman
ρ(Q_NMSE, WM_NMSE) = 0.84 ± 0.03 (pooled p < 1e-20) — and, consistent with our
main results, the recovered model stays one to two orders of magnitude more
accurate than the Q-function it is extracted from (at convergence, WM_NMSE ≈ 7e-3
vs Q_NMSE ≈ 0.16, ≈22× more accurate). New Fig. [X] (MountainCar).

**Robustness to nonuniform / off-distribution Q-error.** The uniform
ε-approximation enters our proofs only to bound the columns of the Bellman matrix
M — i.e. the values at the *successor* states of a query (s, a) — so it can be
replaced by a distributional bound. The set of states an agent reaches,
S_o = ⋃_g Reach(π_g), is invariant under its own policy, so P-learning restricted
to S_o is a reduced MDP and our approximate-recovery bound applies verbatim with
the on-support error ε_o and M restricted to S_o:

  ‖P̂(s,a) − P(s,a)‖₁ ≤ ‖M_{S_o}⁺‖₁ (1 + γm) ε_o,  for s ∈ S_o,

and identifiability then requires only |G| ≥ |S_o| rather than |G| ≥ |S|. So a
distributional / on-support bound does suffice — with one necessary caveat we now
make explicit: because M couples all goals, ε_o must hold on S_o for *every* goal.
This caveat is tight. We give a 3-state counterexample in which Q is exact on the
entire reachable set for the goal that reaches each state, yet an error at a state
that is reached under one goal but off-distribution for another leaves two
transition kernels indistinguishable from Q, with ‖P − P'‖₁ = Ω(1). New Thm/App.
[Y].

Empirically these conditions hold for the trained agent. Restricted to the states
it actually visits (vs. uniformly over the state space), the trained agent's
optimal-Q error is 3.3× ± 0.4 smaller and its recovered-model error 42× ± 10
smaller — i.e. the on-support regime, not the uniform worst case, is the operative
one. (We also note the recovered model is markedly more accurate than Q even where
the Q-function itself is not especially accurate on-support — a strong form of the
"world model ≫ Q-values" phenomenon.)

We will add the theorem, proof, and counterexample to the appendix, and the
scaling and on-/off-support experiments to the experiments section.
