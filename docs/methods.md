# Model and numerical conventions

## Random features and the readout

For scalar inputs, define $F_{ik}=\tanh(a_k x_i+b_k)$ and use $y=Fw+\epsilon$, with $\epsilon\sim N(0,\sigma^2 I)$.

For a positive ridge penalty $\lambda$, the conditional readout posterior used in the companion is

$$
m_w=(F^\top F+\lambda I)^{-1}F^\top y,
$$

$$
V_w=\sigma^2(F^\top F+\lambda I)^{-1}.
$$

This corresponds to $w\sim N(0,\sigma^2 I/\lambda)$. This prior convention follows directly by completing the square and fixes the scale convention in the paragraph preceding Equation 7 of the report. The covariance is computed from a Cholesky factor; the implementation does not form a matrix inverse for the mean.

A ridge-only function draw is $F_{\ast}m_w$. A conditional Bayesian draw samples $w\sim N(m_w,V_w)$ before evaluating $F_{\ast}w$. The latter includes readout uncertainty conditional on the chosen hidden-feature draw. Future-observation draws additionally include independent observation noise.

### From the readout posterior to function covariance

For a prediction grid, write $m_{\ast}(A,B)=F_{\ast}m_w$ and $V_{\ast}(A,B)=F_{\ast}V_wF_{\ast}^\top$. For any specified distribution over the hidden features, the corresponding mixture has

$$
\mathrm{Cov}(h_{\ast})=\mathbb E[V_{\ast}(A,B)]+\mathrm{Cov}(m_{\ast}(A,B)).
$$

The first term describes uncertainty in the readout within a fixed feature map; the second describes movement of the conditional mean across feature maps. Substituting $m_w$ for sampled readouts preserves the second term and discards the first. This is a covariance identity; a particular empirical quantile interval need not widen monotonically at every location in a finite simulation.

In the companion, the distribution over $A,B$ is the diagonal Gaussian approximation returned by `diagonal_tagi`. The covariance decomposition remains valid for that mixture, but does not make the approximate hidden distribution an exact posterior. `function_draws` uses the same hidden draws in its two constructions, with one conditional readout draw for each sampled feature map.

The Gaussian readout prior and ridge penalty must agree for this conditional-mean interpretation. More generally, a prior $w\sim N(0,\tau_w^2 I)$ implies $\lambda=\sigma^2/\tau_w^2$. The hierarchical report sets $\tau_w=1$; a directly matched ridge comparison would therefore use $\lambda=\sigma^2$. The companion states and enforces its own matched convention. The report's original execution settings are not reconstructed here.

The [numerical tests](../tests/test_reference.py) check $(F^\top F+\lambda I)m_w=F^\top y$ and $(F^\top F+\lambda I)V_w=\sigma^2 I$. A separate Monte Carlo check compares the variance of a scalar prediction with $\phi_{\ast}^\top V_w\phi_{\ast}$.

## Simplified diagonal TAGI

The implementation follows the one-layer approximation in Section 2 of the report. It is not an implementation of every component of the original TAGI framework.

An initial pair of hidden vectors is sampled from its Gaussian prior. A ridge fit at these initial features supplies a fixed readout $w^{(0)}$. The Gaussian hidden-parameter approximation starts from the specified zero-mean prior.

For each observation, compute

$$
\mu_{z,k}=x\mu_{a,k}+\mu_{b,k},\quad v_{z,k}=x^2v_{a,k}+v_{b,k},
$$

$$
\phi_k=\tanh(\mu_{z,k}),\quad J_k=1-\phi_k^2,
$$

$$
\mu_y=\sum_k w_k^{(0)}\phi_k,\quad s_y=\sigma^2+\sum_k (w_k^{(0)}J_k)^2v_{z,k}.
$$

The parameter-output covariances are

$$
c_{a,k}=w_k^{(0)}J_kxv_{a,k},\quad c_{b,k}=w_k^{(0)}J_kv_{b,k}.
$$

Update both parameter blocks using the same pre-update state:

$$
\mu_{a,k}^{+}=\mu_{a,k}+c_{a,k}(y-\mu_y)/s_y,\quad v_{a,k}^{+}=v_{a,k}-c_{a,k}^2/s_y,
$$

and analogously for $b$. Cross-coordinate covariance is discarded after each observation. The method is order-dependent and uses a fixed readout during this pass. The readout is then refitted separately for every sampled hidden-feature vector.

## Hierarchical model in the report

The hierarchical Stan experiments introduce

$$
z_{a,k},z_{b,k}\sim N(0,1),\quad a_k=\gamma_a z_{a,k},\quad b_k=\gamma_b z_{b,k},
$$

with positive scale hyperpriors for $\gamma_a,\gamma_b$, and an independent Gaussian readout prior. Half-Cauchy, half-$t_3$, and half-normal scale priors are compared. These experiments are documented in the report; the companion code implements the fixed-scale Gaussian approximation and conditional readout calculations.

## Why compare functions?

Permuting hidden units leaves the fitted function unchanged. Since `tanh` is odd, replacing $(a_k,b_k,w_k)$ with $(-a_k,-b_k,-w_k)$ also leaves it unchanged. Consequently, a marginal weight distribution alone does not establish how well a function has been estimated. The report compares chain-specific functions and selected functionals as well as sampler diagnostics.

The model considered here is a static random-feature regression model. No recurrent state or temporal reservoir update is part of these equations.
