# Experimental Evaluation

## Experimental Setup

We evaluate constrained policy optimization methods on the `13Bus` voltage control task in PowerGym. Each algorithm is trained for `1e6` environment steps with the same network width, rollout length, and optimizer settings unless otherwise noted. The reward follows the standard PowerGym formulation and includes power-loss, voltage-deviation, and control-effort terms. The training cost is defined from the voltage-violation signal, so the comparison explicitly measures how quickly each method suppresses unsafe voltage excursions during learning.

Our primary method is **COST-CONE-Pareto**, which replaces the auxiliary safety classifier used in CAF-based variants with a learned **cost critic**, and then combines the reward gradient and the cost-reduction gradient through a Pareto-style cone update. We compare it against two standard baselines:

- **PPO**, which optimizes reward only and has no explicit cost-aware gradient correction.
- **PPO-Lag**, which uses a Lagrangian penalty to trade off reward and constraint violation.

We also report **COST-CONE** as an ablation to isolate the contribution of the Pareto gradient fusion relative to the same cost-critic backbone.

## Algorithm Sketch

The core update of **COST-CONE-Pareto** is summarized below. The method maintains a standard policy network together with a reward critic and a cost critic. At each training iteration, it computes two policy objectives: the clipped PPO reward surrogate and the clipped cost surrogate. The reward gradient and the cost-reduction gradient are then merged through a Pareto projection step before updating the actor.

```text
Algorithm 1: COST-CONE-Pareto

Input:
  policy pi_theta, reward critic V_phi, cost critic C_psi
  rollout horizon H, PPO clip ratio epsilon
  learning rates alpha_pi, alpha_v
  total training iterations K

For iteration = 1, 2, ..., K do:
  1. Collect a rollout batch B = {(s_t, a_t, r_t, c_t, s_{t+1})}_{t=1}^H with pi_theta.
  2. Estimate reward returns and reward advantages:
       A^r_t, R_t <- GAE(r_t, V_phi)
  3. Estimate cost returns and cost advantages:
       A^c_t, C_t <- GAE(c_t, C_psi)
  4. Update critics by regression:
       minimize  L_V = ||V_phi(s_t) - R_t||^2
       minimize  L_C = ||C_psi(s_t) - C_t||^2
  5. Form the clipped reward surrogate:
       J_r(theta) = E[min(r_t(theta) A^r_t, clip(r_t(theta), 1-epsilon, 1+epsilon) A^r_t)]
  6. Form the clipped cost surrogate and convert it to a safety objective:
       J_c(theta) = E[min(r_t(theta) A^c_t, clip(r_t(theta), 1-epsilon, 1+epsilon) A^c_t)]
       J_s(theta) = -J_c(theta)
  7. Compute policy gradients:
       g_r <- grad_theta J_r(theta)
       g_s <- grad_theta J_s(theta)
  8. Resolve reward-safety conflict by Pareto projection:
       g <- ParetoMerge(g_r, g_s)
  9. Update the policy:
       theta <- theta + alpha_pi * g

End for.
```

## Training Curves

Figure 1 compares training return and training cost across algorithms. Several observations are immediate.

1. **COST-CONE-Pareto reaches near-zero cost the fastest.** Its training cost drops sharply and becomes effectively zero within the early stage of training, noticeably earlier than PPO-Lag.
2. **COST-CONE-Pareto preserves high return while enforcing safety.** In contrast to PPO-Lag, whose return plateaus at a substantially lower level, COST-CONE-Pareto recovers a final return close to the best reward-seeking methods while maintaining negligible cost.
3. **PPO alone is not sufficiently safe.** Although PPO eventually attains competitive return, its training cost remains persistently high and exhibits large uncertainty late in training, indicating unstable safety behavior.
4. **The Pareto update is beneficial beyond the plain cost-cone variant.** COST-CONE already reduces cost substantially faster than PPO-Lag, but COST-CONE-Pareto is both more sample efficient and more stable in driving the cost to zero.

Taken together, these results suggest that explicitly modeling the cost signal with a critic and resolving reward-cost conflicts at the gradient level is more effective than scalarized Lagrangian penalization in this task.

![Training comparison](/home/boerduo/kkcode/powergym/data_aggregation/comparison_outputs/algorithm_train_return_cost_comparison.png)

**Figure 1.** Training return and training cost over `1e6` environment steps. Shaded regions denote 95% confidence intervals across seeds.

## Test-Phase Voltage Trajectories

To assess closed-loop behavior after training, we roll out the learned policies on a representative test episode and visualize all bus-voltage trajectories over time. Figure 2 shows that **COST-CONE-Pareto** yields the most desirable safety profile among the compared methods.

First, the trajectories under **PPO** show several buses dipping close to, or below, the lower safety boundary for an extended portion of the episode. This behavior is consistent with the elevated training cost observed in Figure 1. Second, **PPO-Lag** improves safety relative to PPO, but still leaves multiple buses clustered near the lower bound, reflecting a conservative reward-safety trade-off rather than strong safety recovery. Third, **COST-CONE** and **COST-CONE-Pareto** produce visibly tighter voltage bands across time, with fewer severe downward excursions and more coherent trajectories across buses.

Among these methods, **COST-CONE-Pareto** provides the best overall balance: it maintains the voltage trajectories within the admissible region with the smallest sustained deviation while preserving strong task performance. The qualitative behavior in Figure 2 is therefore fully aligned with the quantitative trend in Figure 1.

![Test-phase voltage trajectories](/home/boerduo/kkcode/powergym/data_aggregation/comparison_outputs/algorithm_voltage_trajectory_comparison.png)

**Figure 2.** Test-phase bus-voltage trajectories for a representative seed. The dashed red and orange lines mark the lower and upper safety bounds, respectively.

## Discussion

The empirical evidence indicates that the main difficulty in this benchmark is not merely estimating a scalar penalty, but resolving conflicting policy gradients induced by reward maximization and constraint reduction. PPO ignores this conflict and therefore attains good reward with poor safety. PPO-Lag addresses the conflict only indirectly through a single adaptive multiplier, which can over-regularize reward optimization and still respond slowly to transient violations. In contrast, COST-CONE-Pareto uses a dedicated cost critic to construct a cost-aware policy gradient and then combines it with the reward gradient through Pareto projection. This design leads to:

- faster suppression of unsafe behavior,
- lower residual training cost,
- stronger final return than PPO-Lag, and
- more stable test-time voltage regulation.

Overall, these results support the central claim that **cost-critic-based cone optimization with Pareto gradient composition is an effective and practical strategy for safe voltage control in distribution networks**.

## LaTeX Pseudocode

```latex
\begin{algorithm}[t]
\caption{COST-CONE-Pareto}
\label{alg:cost_cone_pareto}
\begin{algorithmic}[1]
\REQUIRE Policy $\pi_\theta$, reward critic $V_\phi$, cost critic $C_\psi$, rollout horizon $H$, PPO clip ratio $\epsilon$, learning rates $\alpha_\pi,\alpha_v$, total iterations $K$
\FOR{$k = 1$ to $K$}
    \STATE Collect a rollout batch $\mathcal{B}=\{(s_t,a_t,r_t,c_t,s_{t+1})\}_{t=1}^{H}$ using $\pi_\theta$
    \STATE Estimate reward returns and advantages:
    \[
    \hat{R}_t,\hat{A}^{r}_t \leftarrow \mathrm{GAE}(r_t, V_\phi)
    \]
    \STATE Estimate cost returns and advantages:
    \[
    \hat{C}_t,\hat{A}^{c}_t \leftarrow \mathrm{GAE}(c_t, C_\psi)
    \]
    \STATE Update reward critic by minimizing
    \[
    \mathcal{L}_{V} = \mathbb{E}\left[(V_\phi(s_t)-\hat{R}_t)^2\right]
    \]
    \STATE Update cost critic by minimizing
    \[
    \mathcal{L}_{C} = \mathbb{E}\left[(C_\psi(s_t)-\hat{C}_t)^2\right]
    \]
    \STATE Form the clipped reward surrogate
    \[
    J_r(\theta)=\mathbb{E}\left[\min\!\left(\rho_t(\theta)\hat{A}^{r}_t,\,
    \mathrm{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat{A}^{r}_t\right)\right]
    \]
    \STATE Form the clipped cost surrogate
    \[
    J_c(\theta)=\mathbb{E}\left[\min\!\left(\rho_t(\theta)\hat{A}^{c}_t,\,
    \mathrm{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat{A}^{c}_t\right)\right]
    \]
    \STATE Convert the cost objective into a safety-improving objective:
    \[
    J_s(\theta) = -J_c(\theta)
    \]
    \STATE Compute policy gradients
    \[
    g_r \leftarrow \nabla_\theta J_r(\theta), \qquad
    g_s \leftarrow \nabla_\theta J_s(\theta)
    \]
    \STATE Merge gradients with Pareto projection:
    \[
    g \leftarrow \mathrm{ParetoMerge}(g_r,g_s)
    \]
    \STATE Update the actor:
    \[
    \theta \leftarrow \theta + \alpha_\pi g
    \]
\ENDFOR
\end{algorithmic}
\end{algorithm}
```
