"""Frequency-severity model for annual cyber losses at a (hypothetical) bank.

Annual aggregate loss S = X_1 + ... + X_N
    N   ~ Poisson(lam)                      number of material incidents per year
    X_i ~ mixture severity:
            with prob 1 - p_tail : Lognormal(mu, sigma)   "ordinary" incidents
            with prob p_tail     : Pareto(x_min, alpha)   rare, extreme incidents

All parameters are ILLUSTRATIVE ASSUMPTIONS, not calibrated to any real bank.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Params:
    lam: float = 3.0            # expected material incidents per year
    median_loss: float = 400_000.0   # median ordinary incident loss (EUR)
    sigma: float = 1.5          # lognormal shape (higher = fatter body)
    p_tail: float = 0.05        # share of incidents that are "extreme"
    tail_xmin: float = 2_000_000.0   # minimum loss of an extreme incident (EUR)
    tail_alpha: float = 1.7     # Pareto tail index (must be > 1 for a finite mean)

    def validate(self) -> None:
        if self.lam <= 0:
            raise ValueError("lam must be positive")
        if not 0 <= self.p_tail <= 1:
            raise ValueError("p_tail must be in [0, 1]")
        if self.tail_alpha <= 1:
            raise ValueError("tail_alpha must be > 1 (otherwise the mean is infinite)")
        if self.median_loss <= 0 or self.sigma <= 0 or self.tail_xmin <= 0:
            raise ValueError("median_loss, sigma and tail_xmin must be positive")


def sample_severity(n: int, p: Params, rng: np.random.Generator) -> np.ndarray:
    """Draw n individual incident losses from the lognormal/Pareto mixture."""
    losses = rng.lognormal(mean=np.log(p.median_loss), sigma=p.sigma, size=n)
    is_tail = rng.random(n) < p.p_tail
    k = int(is_tail.sum())
    if k:
        # Inverse-CDF sampling for Pareto(x_min, alpha)
        losses[is_tail] = p.tail_xmin * (1 - rng.random(k)) ** (-1 / p.tail_alpha)
    return losses


def simulate_annual_losses(p: Params, n_sims: int = 200_000, seed: int = 42) -> np.ndarray:
    """Monte Carlo: return an array of n_sims simulated annual aggregate losses."""
    p.validate()
    rng = np.random.default_rng(seed)
    counts = rng.poisson(p.lam, size=n_sims)
    total = int(counts.sum())
    severities = sample_severity(total, p, rng)
    year_index = np.repeat(np.arange(n_sims), counts)
    return np.bincount(year_index, weights=severities, minlength=n_sims)


def risk_metrics(losses: np.ndarray, levels=(0.95, 0.99, 0.995, 0.999)) -> dict:
    """Expected loss, VaR and Expected Shortfall (average loss beyond VaR)."""
    out = {"expected_loss": float(losses.mean()), "median": float(np.median(losses))}
    for q in levels:
        var = float(np.quantile(losses, q))
        out[f"VaR_{q:.1%}"] = var
        out[f"ES_{q:.1%}"] = float(losses[losses >= var].mean())
    out["prob_zero_loss"] = float((losses == 0).mean())
    return out


def theoretical_mean(p: Params) -> float:
    """Closed-form E[S] = lam * E[X], used to sanity-check the simulation."""
    ln_mean = p.median_loss * np.exp(p.sigma ** 2 / 2)
    pareto_mean = p.tail_alpha * p.tail_xmin / (p.tail_alpha - 1)
    return p.lam * ((1 - p.p_tail) * ln_mean + p.p_tail * pareto_mean)


def improved_controls(p: Params, freq_reduction: float = 0.30, severity_reduction: float = 0.10) -> Params:
    """Scenario: better controls cut incident frequency and the typical loss size."""
    return replace(p, lam=p.lam * (1 - freq_reduction),
                   median_loss=p.median_loss * (1 - severity_reduction))
