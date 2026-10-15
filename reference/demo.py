"""Generate a small synthetic example; separate from the report experiments."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import diagonal_tagi, features, function_draws, readout_posterior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    settings = dict(seed=args.seed, train_size=160, hidden_units=20, draws=300,
                    noise_variance=0.09, prior_variance=0.6, ridge_penalty=0.01)
    x = rng.uniform(-3.0, 3.0, settings["train_size"])
    y = np.sin(x) + rng.normal(0.0, np.sqrt(settings["noise_variance"]), len(x))
    k = settings["hidden_units"]
    a, b = rng.normal(0.0, np.sqrt(settings["prior_variance"]), (2, k))
    w, _ = readout_posterior(features(x, a, b), y, settings["ridge_penalty"],
                             settings["noise_variance"])
    hidden = diagonal_tagi(x, y, w, settings["noise_variance"],
                           settings["prior_variance"], settings["prior_variance"])
    grid = np.linspace(-3.0, 3.0, 300)
    ridge, conditional = function_draws(
        x, y, grid, hidden, settings["ridge_penalty"], settings["noise_variance"],
        rng, settings["draws"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, layout="constrained")
    summary = {"kind": "synthetic_numerical_companion", "settings": settings, "results": {}}
    for ax, values, label, color in zip(
        axes, (ridge, conditional), ("Ridge readout", "Conditional readout draws"),
        ("#087e8b", "#7251b5")
    ):
        mean = values.mean(axis=0)
        low, high = np.quantile(values, [0.05, 0.95], axis=0)
        ax.scatter(x, y, s=8, color="#999999", alpha=0.25, label="Observations")
        ax.fill_between(grid, low, high, color=color, alpha=0.2, label="90% draw interval")
        ax.plot(grid, mean, color=color, label="Mean")
        ax.plot(grid, np.sin(grid), "k--", linewidth=1, label="Truth")
        ax.set(title=label, xlabel="x", ylabel="h(x)")
        ax.legend(frameon=False, fontsize=8)
        summary["results"][label] = {
            "grid_rmse_truth": float(np.sqrt(np.mean((mean - np.sin(grid))**2))),
            "mean_interval_width": float(np.mean(high - low)),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "demo.png", dpi=180)
    plt.close(fig)
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
