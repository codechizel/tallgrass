"""Model specification dataclasses for hierarchical IRT experiments.

Factors the bill discrimination (beta) prior into a frozen dataclass that both
production and experiments consume. Production uses PRODUCTION_BETA as the default;
experiments pass alternative specs to the same model-building functions.

See docs/experiment-framework-deep-dive.md for design rationale.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BetaPriorSpec:
    """Specification for the bill discrimination (beta) prior.

    The distribution field selects the PyMC distribution family.
    The params dict is unpacked as keyword arguments to the distribution constructor.

    Examples:
        BetaPriorSpec("normal", {"mu": 0, "sigma": 1})      # production default
        BetaPriorSpec("lognormal", {"mu": 0, "sigma": 0.5})  # soft positive
        BetaPriorSpec("halfnormal", {"sigma": 1})            # hard zero floor
    """

    distribution: str
    params: dict[str, float]

    def build(self, n_votes: int, dims: str = "vote"):
        """Instantiate the PyMC distribution inside an active model context.

        Must be called inside a ``with pm.Model():`` block.

        Args:
            n_votes: Number of bills/votes (shape parameter).
            dims: PyMC dimension name for the variable.

        Returns:
            PyMC distribution variable named "beta".

        Raises:
            ValueError: If distribution is not one of: normal, lognormal, halfnormal.
        """
        import pymc as pm

        match self.distribution:
            case "normal":
                return pm.Normal("beta", shape=n_votes, dims=dims, **self.params)
            case "lognormal":
                return pm.LogNormal("beta", shape=n_votes, dims=dims, **self.params)
            case "halfnormal":
                return pm.HalfNormal("beta", shape=n_votes, dims=dims, **self.params)
            case _:
                msg = (
                    f"Unknown beta prior distribution: {self.distribution!r}. "
                    f"Supported: normal, lognormal, halfnormal"
                )
                raise ValueError(msg)

    def describe(self) -> str:
        """Human-readable description for logs and reports.

        Returns:
            String like "Normal(mu=0, sigma=1)" or "LogNormal(mu=0, sigma=0.5)".
        """
        name = self.distribution.capitalize()
        if self.distribution == "lognormal":
            name = "LogNormal"
        elif self.distribution == "halfnormal":
            name = "HalfNormal"
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{name}({param_str})"


# Production default: matches the hardcoded pm.Normal("beta", mu=0, sigma=1, ...)
# that has been in build_per_chamber_model() and build_joint_model() since ADR-0017.
PRODUCTION_BETA = BetaPriorSpec("normal", {"mu": 0, "sigma": 1})

# Joint model beta: LogNormal(0, 0.5) eliminates reflection mode multimodality.
# Positive constraint forces each bill to discriminate in the "natural" direction,
# removing 420+ sign-flip axes from the posterior. Experiment (2026-02-27) showed
# R-hat 1.53→1.024 and ESS 7→243 from this change alone. See docs/joint-model-deep-dive.md.
JOINT_BETA = BetaPriorSpec("lognormal", {"mu": 0, "sigma": 0.5})
