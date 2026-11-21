import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import Params, improved_controls, risk_metrics, simulate_annual_losses, theoretical_mean


def test_simulated_mean_close_to_theory():
    # Use a lighter tail so the sample mean converges quickly
    p = Params(tail_alpha=2.5)
    sim = simulate_annual_losses(p, 400_000, seed=0).mean()
    assert sim == pytest.approx(theoretical_mean(p), rel=0.03)


def test_reproducible_with_seed():
    p = Params()
    assert np.array_equal(simulate_annual_losses(p, 1000, 5), simulate_annual_losses(p, 1000, 5))


def test_var_and_es_ordering():
    m = risk_metrics(simulate_annual_losses(Params(), 100_000, 1))
    assert m["VaR_95.0%"] < m["VaR_99.0%"] < m["VaR_99.9%"]
    assert m["ES_99.0%"] >= m["VaR_99.0%"]


def test_better_controls_reduce_expected_loss():
    p = Params()
    assert theoretical_mean(improved_controls(p)) < theoretical_mean(p)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        simulate_annual_losses(Params(tail_alpha=0.9), 10)
    with pytest.raises(ValueError):
        simulate_annual_losses(Params(lam=-1), 10)
