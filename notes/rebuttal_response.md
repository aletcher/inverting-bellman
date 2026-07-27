# Reviewer response

We thank the reviewers, and address both the empirical question (how recovery scales with Q-error) and the theoretical one (robustness of identifiability to nonuniform, off-distribution Q-error) with a strengthened theorem and new controlled experiments, all of which we will add to the paper.

**Recovery quality vs Q-estimation error.** We ran a new experiment to measure the Q-value error, and recovered WM error, across 20 training checkpoints of a MountainCar agent. As predicted by our theory, WM error decreases with Q-value error (Spearman $\rho = 0.83 \pm 0.03$), and the recovered model is consistently one to two orders of magnitude more accurate than the corresponding Q-function (at convergence, $\mathrm{WM}_{\mathrm{NMSE}} \approx 7\times10^{-3}$ vs $Q_{\mathrm{NMSE}} \approx 0.22$, ${\sim}31\times$).

**Robustness to nonuniform / off-distribution Q-error.** In short, the answer is positive: recovery holds under a distributional (on-support) bound rather than a uniform one. We both prove this and verify it experimentally.

*Theory.* Writing $S_o = \bigcup_g \mathrm{Supp}(\pi_g)$ for the set of states visited by an agent across all goals, the MDP restricted to $S_o \times A$ is a reduced MDP, and our approximate-recovery bound applies verbatim with the on-support error $\varepsilon_o = \sum_{s \in S_o} \| Q(s,a,g) - Q^\pi(s,a,g) \|$ and $M$ restricted to $S_o$:

$$\|\hat{P}(s,a) - P(s,a)\|_1 \;\le\; \|M_{S_o}^{+}\|_1\,(1 + \gamma m)\,\varepsilon_o, \qquad s \in S_o,$$

with identifiability now requiring only $|\mathcal{G}| \ge |S_o|$ rather than $|\mathcal{G}| \ge |\mathcal{S}|$. The one necessary caveat: because $M$ couples all goals, $\varepsilon_o$ must hold on $S_o$ for *every* goal. This is tight — we give a 3-state counterexample where $Q$ is exact on the reachable set of each goal, yet an error at a state reached under one goal but off-distribution for another leaves two kernels indistinguishable from $Q$, with $\|P - P'\|_1 = \Omega(1)$.

*Direct verification.* We inject a controlled error into the Q-values used for recovery (P-learning on $S_o$) and vary its magnitude, placed either off-support or on-support. Recovery is far more sensitive to on-support than off-support error: at a matched injected error of $\mathrm{NMSE} \approx 0.1$, off-support perturbation leaves on-support recovery essentially intact ($\mathrm{WM}_{\mathrm{NMSE}} \approx 3\times10^{-5}$) while the same on-support error degrades it ${\sim}700\times$ (to $\approx 2\times10^{-2}$). Off-support error affects recovery only once it grows large ($\mathrm{NMSE} \gtrsim 0.5$, beyond the agent's natural Q-error), where it can admit a spurious consistent kernel — exactly the boundary of our counterexample. The experiment thus traces both regimes: robustness under the on-support bound, and its breakdown at the counterexample threshold. New Fig. [Z].

*Corroboration (natural setting).* For the trained agent, the recovered model is $60\times \pm 7$ more accurate on its reachable set than uniformly; and while its distance to optimal degrades off-support ($\|Q - Q^{*}\|$ is $2.3\times \pm 0.2$ larger), the Bellman-residual error the bound actually depends on, $\|Q - Q^{\pi}\|$, is comparable on- and off-support ($1.2 \pm 0.2$) — i.e. not concentrated off-distribution.
