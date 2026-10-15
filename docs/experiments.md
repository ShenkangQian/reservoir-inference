# Experiments and reported measurements

The source is the 36-page experiment report dated October 12, 2026. Values below are transcribed from Figure 39 and the surrounding discussion. The CSV files preserve the displayed precision.

## Targets

The study uses synthetic regression targets with distinct shape characteristics:

1. `sin(x)`.
2. `x + 5 phi(10 x)`, where `phi` is the standard normal density.
3. `sin(3 pi x / 2) / [1 + 18 x^2 (sign(x) + 1)]`.

Earlier fixed-scale experiments use 20 hidden units and explore the ridge penalty. Increasing the prior variance of the hidden parameters from 0.6 to 3.0 improves the representation of the sharper targets in the reported plots. The derivative identity `d tanh(a x + b) / dx = a [1 - tanh^2(a x + b)]` explains why the slope scale affects local variation.

In the hierarchical sine experiment, the posterior of the bias scale concentrates near zero. This is consistent with fitting an odd target using approximately odd hidden features. Hyperprior sensitivity is visible in scale coordinates, while the displayed fitted functions are similar across the three choices.

## Paired comparison at K = 10

The final comparison uses the same sampled hidden features in two constructions: one keeps the joint Stan draw of the readout, and the other refits a ridge readout. The evaluation grid has 300 points. Four chains retain 250 draws each.

| Metric | Ridge | Joint Stan |
| --- | ---: | ---: |
| Mean pointwise standard deviation | 0.0275 | 0.0415 |
| Mean 90% interval width | 0.0884 | 0.1358 |
| Grid RMSE against truth | 0.0579 | 0.0580 |
| Test RMSE against truth | 0.0534 | 0.0536 |
| Test RMSE against noisy responses | 0.2866 | 0.2867 |

These test errors summarize the fitted mean. They differ from the posterior mean of the *per-draw* RMSE functional, which is 0.0613 for ridge and 0.0690 for joint Stan. Averaging and the nonlinear RMSE operation do not commute.

The primary result is that retaining readout uncertainty has little effect on the fitted mean here but increases average 90% interval width. These function intervals do not include independent future-observation noise and are not a claim of calibrated 90% frequentist coverage.

## Function and sampler diagnostics

| Worst pointwise diagnostic | Ridge | Joint Stan |
| --- | ---: | ---: |
| Maximum R-hat | 1.0159 | 1.0070 |
| Minimum bulk ESS | 488 | 690 |
| Minimum tail ESS | 485 | 500 |
| Maximum MCSE / posterior SD | 0.0466 | 0.0388 |
| Grid points with R-hat at least 1.01 | 9 | 0 |

For the global peak, signed integral, and per-draw test RMSE, all reported R-hat values are below 1.01 and all bulk and tail ESS values exceed 800. These measurements support agreement of the sampled functions across chains in this experiment.

The same run has zero reported divergences and E-BFMI values between 0.875 and 0.974, with 817 of 1,000 transitions reaching the maximum tree depth. Function-level agreement should therefore be read alongside the sampler's efficiency diagnostics; it is not a guarantee of exact posterior exploration. The report motivates hierarchical TAGI as a subsequent direction rather than reporting a completed hierarchical TAGI algorithm.

## Evidence files

- `results/paired-comparison.csv`: function variability and prediction metrics.
- `results/function-diagnostics.csv`: pointwise diagnostics.
- `results/sampler-diagnostics.csv`: per-chain sampler diagnostics.
- `assets/ridge-stan-functions.png`: unmodified embedded image from Figure 38.

No raw Stan chains or original experiment scripts are included in the supplied report. The numerical companion illustrates the documented fixed-scale calculations and does not regenerate the reported Stan tables.
