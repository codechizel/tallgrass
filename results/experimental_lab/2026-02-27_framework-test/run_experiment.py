"""Framework smoke test: LogNormal beta prior via the experiment runner.

This is the new-style experiment script — ~25 lines of config instead of ~800
lines of duplicated model-building code. All logic is in production functions.

Usage:
    uv run python results/experiments/2026-02-27_framework-test/run_experiment.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.experiment_runner import ExperimentConfig, run_experiment
from analysis.model_spec import BetaPriorSpec

EXPERIMENT_DIR = Path(__file__).parent

config = ExperimentConfig(
    name="run_01_lognormal",
    description="beta ~ LogNormal(0, 0.5) — framework smoke test",
    beta_prior=BetaPriorSpec("lognormal", {"mu": 0, "sigma": 0.5}),
    n_samples=500,
    n_tune=500,
    n_chains=2,
)

if __name__ == "__main__":
    metrics = run_experiment(config, output_base=EXPERIMENT_DIR)
    print(f"\nDone. Convergence: {metrics['chambers']}")
