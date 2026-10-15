# Running the numerical companion

The `reference` module implements the equations for a fixed-scale random-feature model. It is provided separately from the archived experiment report, whose original executable scripts and posterior chains were not part of the source material.

## Environment

Use Python 3.10 or later. Install the two declared dependencies from `requirements.txt` in a virtual environment. The example uses only a CPU and creates all observations locally from a seeded random number generator.

```bash
python -m reference.demo --seed 42 --output outputs/demo
python -m unittest discover -s tests -v
```

`demo.png` compares the ridge-only and conditional-readout constructions. `metrics.json` records the seed, sample sizes, model parameters, and summary values. The plot describes draws under an approximate hidden-parameter distribution; its intervals do not imply validated coverage.

The report has a separate experiment design and seed history. Do not compare the example's summary values directly with its Stan result tables as though they were paired runs.

## Implementation checks

The tests verify the ridge normal equations, conditional covariance, sign/permutation invariance, nonnegative contracting diagonal variances, and the contribution of conditional readout uncertainty. They check numerical identities, not scientific performance claims.

## Data handling

The repository contains synthetic examples and aggregate report measurements. Runtime outputs, caches, virtual environments, credentials, and local environment files are excluded by `.gitignore`. No network or account access is needed to run the example.
