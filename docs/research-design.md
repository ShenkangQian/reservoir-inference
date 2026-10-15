# Reading the study as a sequence of inference questions

The [report](../reports/random-feature-posterior-inference.pdf) changes one part of the inferential construction at a time: the ridge penalty, feature scale, treatment of the scales, network width, and finally the readout. The comparisons are useful because each change asks a different question. This note explains what is held fixed, what is varied, and which quantities answer that question.

## From a good curve to an explanation of its shape

The sine example establishes the basic workflow: select a ridge penalty, update a Gaussian approximation for the hidden parameters, sample feature maps, and refit their readouts. The sharper examples then ask whether the feature prior puts enough mass on useful functions.

The derivative bound

$$
|h'(x)|\leq\sum_k |w_k|\,|a_k|
$$

connects the hidden slope scale to function variation. It is an upper bound, not a guarantee that every draw with large slopes will fit a sharp target. Increasing both hidden prior variances also changes the bias distribution. Consequently, the reported improved fits support examining feature scale; they do not, on their own, isolate the effect of the slope prior from that of the bias prior. The distinction matters when turning a visual improvement into a mechanism claim.

The hierarchical model addresses a related question: can the observations inform those scales? Its non-centred construction expresses a hidden weight as a standard-normal coordinate multiplied by a positive scale. In the sine experiment, a small inferred bias scale has a useful interpretation because `tanh(a x)` is odd. It is consistent with the target's symmetry, without requiring a unique set of hidden weights.

The hyperprior comparison asks whether that conclusion depends on one scale distribution. The displayed scale marginals change more than the fitted functions across half-Cauchy, half-$t_3$, and half-normal choices. This is a distinction between parameter sensitivity and predictive sensitivity, rather than a claim that hyperpriors never matter.

## Why individual parameter summaries can be misleading

Both permutations and simultaneous sign changes of a hidden unit preserve the network output. Averaging weights across equivalent sign configurations can cancel coordinates even when every sampled network represents a similar function. In general,

$$
h(x;\mathbb E[\theta])\ne\mathbb E[h(x;\theta)].
$$

Evaluating each sampled network before averaging respects this nonlinearity. It also makes the diagnostic target concrete: a peak, integral, pointwise function value, or error relative to known truth. Agreement of these quantities answers a function-level question; it does not remove the need to inspect how the sampler explores the underlying posterior.

## The paired readout experiment

For each retained Stan state, preserve its hidden parameters and construct two curves:

1. Evaluate the network with the corresponding joint draw of the readout.
2. Recompute a ridge readout using the same observations and hidden-feature matrix, then evaluate that curve on the same grid.

Holding the hidden draw fixed makes this a comparison at the readout stage. It does not compare two independently trained feature distributions. To interpret the substitution exactly as conditional averaging, the ridge penalty must match the Gaussian readout prior: $\lambda=\sigma^2/\tau_w^2$. The [method notes](methods.md) derive the resulting covariance decomposition and distinguish the explicit companion convention from the archived experiment settings.

The report's `K = 10` comparison uses four chains with 250 retained draws each and a 300-point evaluation grid. Figure 38 shows the fitted means and intervals; Figure 39 supplies the numerical summaries. The mean prediction's test RMSE is 0.0534 for ridge and 0.0536 for joint Stan. Mean 90% interval widths are 0.0884 and 0.1358. These measurements answer different questions: accuracy of the estimated mean and variability of the latent function across draws.

## Distinguish the estimands before comparing numbers

The RMSE of the posterior mean is

$$
\sqrt{\frac{1}{n_*}\sum_i(\bar h(x_i)-h_{\mathrm{true}}(x_i))^2}.
$$

The posterior mean of per-draw RMSE instead computes an RMSE for every sampled curve and then averages those errors. Nonlinearity makes these distinct. The report lists the latter as 0.0613 for ridge and 0.0690 for joint Stan; these are compatible with the smaller errors of the fitted means above. The [result notes](experiments.md) keep these definitions separate.

Likewise, uncertainty about $h(x)$ differs from uncertainty about a future response $Y(x)$. Independent observation noise contributes additional variance to the latter. The paired figures show latent-function intervals, so their widths should not be read as future-response intervals or as measured frequentist coverage.

## Connect diagnostics to the computation

The joint Stan function values have a maximum reported R-hat of 1.0070 and minimum bulk/tail ESS of 690/500 across the grid. The global peak, signed integral, and per-draw RMSE also have R-hat below 1.01 and bulk/tail ESS above 800. Reading the worst grid-point diagnostics avoids averaging away a local problem.

The sampler record supplies a complementary view: no reported divergences, E-BFMI between 0.875 and 0.974, and 817 maximum-tree-depth hits among 1,000 retained transitions. Thus the displayed agreement between function summaries coexists with substantial computational effort. Both are part of interpreting this run.

## Evidence and executable scope

The [report](../reports/random-feature-posterior-inference.pdf), [paired metrics](../results/paired-comparison.csv), [function diagnostics](../results/function-diagnostics.csv), and [sampler diagnostics](../results/sampler-diagnostics.csv) preserve the original measurements. The CSVs transcribe the report's displayed precision.

The [NumPy companion](../reference/model.py) separately implements fixed-scale Gaussian hidden updates and conditional Bayesian readouts. Its [tests](../tests/test_reference.py) validate algebraic identities, symmetry, variance contraction, and sampled readout variance. It does not recreate the original Stan chains or supply a completed hierarchical TAGI method. The [reproduction guide](reproduction.md) identifies the outputs produced by the runnable example.
