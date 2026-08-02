# Soft vs. hard consistency in policy-only world-model extraction

## 1. What a Boltzmann policy reveals about Q

If $\pi(a\mid s) = \mathrm{softmax}(Q(s,\cdot)/\tau)_a$, then taking logs,

$$
\log\pi(a\mid s) = \frac{Q(s,a)}{\tau} - \mathrm{logsumexp}\Big(\frac{Q(s,\cdot)}{\tau}\Big)
\quad\Longleftrightarrow\quad
Q(s,a) = \tau\log\pi(a\mid s) + c(s),
$$

with $c(s) = \tau\,\mathrm{logsumexp}(Q(s,\cdot)/\tau)$. This holds for **any** $\tau$. So the policy determines Q *exactly up to a per-state constant*: the level $c(s)$ is lost (log_softmax subtracts it), but every action gap survives, since $Q(s,a)-Q(s,b) = \tau\big(\log\pi(a\mid s)-\log\pi(b\mid s)\big)$. The extraction problem is: recover the world model from the gaps, learning the lost level as a nuisance network $V_\psi(s,g)$.

## 2. The soft variant: exact consistency with a soft-Q agent

Assume Q satisfies the entropy-regularised (soft) Bellman equation

$$
Q(s,a) = r + \gamma(1-d)\,V_{\mathrm{soft}}(s'), \qquad
V_{\mathrm{soft}}(s) = \tau\,\mathrm{logsumexp}\big(Q(s,\cdot)/\tau\big).
$$

The key observation: the per-state constant **is** the soft value, $c(s) = V_{\mathrm{soft}}(s)$, so log_softmax gives exactly

$$
\tau\log\pi(a\mid s) = Q(s,a) - V_{\mathrm{soft}}(s).
$$

Substituting $Q = \tau\log\pi + V$ into the soft Bellman residual eliminates Q entirely:

$$
\underbrace{\tau\log\pi(a\mid s) + V(s)}_{=\,Q(s,a)\text{ when }V=V_{\mathrm{soft}}} - r - \gamma(1-d)\,V(s') = 0
\quad\text{at } V=V_{\mathrm{soft}},\ \text{true dynamics}.
$$

For a genuinely soft agent, policy↔Q consistency is exact by construction — no penalty term, no slack.

## 3. Why PQN misspecifies the soft variant

PQN is *hard* Q-learning: $Q(s,a) \approx r + \gamma(1-d)\max_{a'}Q(s',a')$. Build $\pi = \mathrm{softmax}(Q_{\mathrm{pqn}}/\tau)$ and plug into the soft residual at the true dynamics with $V = V_{\mathrm{soft}}$ (built from $Q_{\mathrm{pqn}}$). The left side collapses to $Q_{\mathrm{pqn}}(s,a)$, and using PQN's own equation:

$$
\text{residual} = \gamma(1-d)\Big(\max_{a'}Q(s',a') - V_{\mathrm{soft}}(s')\Big),
$$

which is not zero. Its magnitude is controlled by the standard logsumexp sandwich: from $\sum_a e^{Q_a/\tau} \ge e^{\max_a Q_a/\tau}$ and $\sum_a e^{Q_a/\tau} \le |A|\,e^{\max_a Q_a/\tau}$,

$$
0 \le V_{\mathrm{soft}}(s) - \max_a Q(s,a) \le \tau\log|A|.
$$

With $|A| = 9$, the soft residual carries an irreducible bias of order $\gamma\,\tau\log 9 \approx 2.2\,\tau$. Since sigmoid puts $Q_{\mathrm{pqn}}\in(0,1)$, this is not academic: at $\tau=0.1$ the ceiling ($\approx 0.22$) is a fifth of the entire Q range; at $\tau=0.01$ it is $\approx 0.022$. The bias shrinks *linearly* in $\tau$ — hence the $\tau$ sweep.

## 4. The hard variant: max-normalised log-probs

Instead normalise $\log\pi$ by its per-state maximum:

$$
A_\tau(s,a) := \tau\Big(\log\pi(a\mid s) - \max_{a'}\log\pi(a'\mid s)\Big)
= Q(s,a) - \max_{a'}Q(s,a'),
$$

**exactly, for every $\tau$** — both logs equal $Q/\tau - \mathrm{logsumexp}(Q/\tau)$, so the logsumexp constants cancel and $A_\tau$ is the advantage relative to the greedy action, $\tau$-independent. The residual

$$
A_\tau(s,a) + V(s) - r - \gamma(1-d)\,V(s')
$$

is then the **hard** Bellman residual with $V \equiv \max_a Q$: at the true dynamics and $V = \max_a Q_{\mathrm{pqn}}$, it equals $Q_{\mathrm{pqn}}(s,a) - r - \gamma(1-d)\max_{a'}Q_{\mathrm{pqn}}(s',a')$ — PQN's own TD error, $\approx 0$ for every action (the hard optimality equation holds action-wise, not just at the argmax). No $\tau$ bias; $\tau$ is purely cosmetic here (it only rescales what the multiplication by $\tau$ already undoes).

## 5. What $V_\psi$ means, and why it's the same code

- **Soft:** $V_\psi(s,g)$ should converge to $\tau\,\mathrm{logsumexp}(Q(s,\cdot,g)/\tau)$.
- **Hard:** $V_\psi(s,g)$ should converge to $\max_a Q(s,a,g)$.

In both cases $V_\psi$ is a free per-$(s,g)$ scalar — it must be learned because the policy destroyed the level. The two variants are literally one code path with a one-line switch: use $\tau\log\pi$ raw, or subtract its per-state max first. The two inputs differ by the per-state constant $\tau\max_{a'}\log\pi(a'\mid s) = \max_a Q(s,a) - V_{\mathrm{soft}}(s)$, and a per-state constant added to the target is absorbed into what $V_\psi$ has to represent. Same loss, same world-model net, same sampling — only the semantics of $V_\psi$ shift.

## 6. How to read the two variants

- **Hard** is the *well-specified ablation* for PQN checkpoints: it isolates "can we jointly learn $V_\psi$ and $f_\phi$ from advantages alone?" with zero modelling bias. Its NMSE should approach the Q-based P-learning baseline, up to $V$-learning error. If it doesn't, the problem is optimisation, not the idea.
- **Soft** is the honest Boltzmann-rationality story — the actual framing of the project — and against a PQN agent its error should floor at the $O(\gamma\tau\log 9)$ bias and degrade gracefully as $\tau$ grows. Comparing the two separates optimisation difficulty (visible in hard) from misspecification bias (the soft−hard gap).
- The roles are not intrinsic: with a genuinely entropy-regularised (soft-Q-learning) agent, the **soft** variant is exact and the **hard** one biased — $\max_a Q$ then under-shoots the true bootstrap value $V_{\mathrm{soft}}$ by up to $\tau\log 9$, with the sign flipped. Which variant is "correct" is a property of the agent's training objective, not of the extraction method.
