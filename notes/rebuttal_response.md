# Reviewer response

We thank the reviewers, and address both the empirical question (how recovery scales with Q-error) and the theoretical one (robustness of identifiability to nonuniform, off-distribution Q-error) with a strengthened theorem and new controlled experiments, all of which we will add to the paper.

**Recovery quality vs Q-estimation error.** We treat an agent's 20 training checkpoints as Q-functions of increasing quality and measure, per checkpoint, both the Q-value error ($Q_{\mathrm{NMSE}}$, against the true value of the induced policy) and the recovered-model error ($\mathrm{WM}_{\mathrm{NMSE}}$). Recovery improves monotonically with Q-quality (Spearman $\rho = 0.84 \pm 0.03$, pooled $p < 10^{-20}$), and the recovered model stays one to two orders of magnitude more accurate than the Q-function it is extracted from (at convergence, $\mathrm{WM}_{\mathrm{NMSE}} \approx 7\times10^{-3}$ vs $Q_{\mathrm{NMSE}} \approx 0.16$; ${\sim}22\times$). New Fig. [X] (MountainCar, 5 seeds).

**Robustness to nonuniform / off-distribution Q-error.** Short answer: yes — recovery holds under a distributional (on-support) bound rather than a uniform one, and we both prove this and verify it directly.

*Theory.* The uniform $\varepsilon$-approximation enters our proofs only to bound the columns of the Bellman matrix $M$ — the values at the *successor* states of a query $(s,a)$ — so it can be replaced by a distributional bound. The set of states an agent reaches, $S_o = \bigcup_g \mathrm{Reach}(\pi_g)$, is invariant under its own policy; hence P-learning restricted to $S_o$ is a reduced MDP, and our approximate-recovery bound applies verbatim with the on-support error $\varepsilon_o$ and $M$ restricted to $S_o$:

$$\|\hat{P}(s,a) - P(s,a)\|_1 \;\le\; \|M_{S_o}^{+}\|_1\,(1 + \gamma m)\,\varepsilon_o, \qquad s \in S_o,$$

with identifiability now requiring only $|\mathcal{G}| \ge |S_o|$ rather than $|\mathcal{G}| \ge |\mathcal{S}|$. The one necessary caveat: because $M$ couples all goals, $\varepsilon_o$ must hold on $S_o$ for *every* goal. This is tight — we give a 3-state counterexample where $Q$ is exact on the reachable set of each goal, yet an error at a state reached under one goal but off-distribution for another leaves two kernels indistinguishable from $Q$, with $\|P - P'\|_1 = \Omega(1)$. New Thm/App. [Y].

*Direct verification.* Injecting a controlled error into the Q-values used for recovery (P-learning on $S_o$), an off-support Q-error as large as $\mathrm{NMSE} \approx 2$ leaves on-support recovery essentially unchanged ($\mathrm{WM}_{\mathrm{NMSE}}$ from $1.0\times10^{-3}$ to $2.0\times10^{-3}$), whereas the same error placed on-support degrades it ${\sim}20\times$ (to $2.1\times10^{-2}$). Recovery therefore depends only on on-support Q-accuracy and is immune to off-distribution Q-error, exactly as the bound predicts. New Fig. [Z].

*Corroboration (natural setting).* For the trained agent, the recovered model is $42\times \pm 10$ more accurate on its reachable set than uniformly; and while its distance to optimal degrades off-support ($\|Q - Q^{*}\|$ is $3.3\times \pm 0.4$ larger), the Bellman-residual error the bound actually depends on, $\|Q - Q^{\pi}\|$, is comparable on- and off-support ($1.0 \pm 0.1$) — i.e. not concentrated off-distribution.
