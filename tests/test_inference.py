"""Regression checks for identification, joint inference, and simulation design."""

import csv

import numpy as np
import pytest

from sim.coverage_simulation import (
    Design,
    estimate_late,
    gaussian_rates,
    ratio_gradient,
    receipt_flag,
    run_scenario,
    run_study,
    sensitivity_bounds,
    windows,
)


@pytest.mark.parametrize("itt", [-2.0, 0.0, 2.0])
def test_gradient_matches_numerical_derivative(itt):
    theta = np.array([itt, 0.4, 0.8, 0.05])

    def ratio(x):
        a, q, s, f = x
        return a * (s - f) / (q - f)

    numerical = []
    for step in np.eye(4) * 1e-6:
        numerical.append((ratio(theta + step) - ratio(theta - step)) / 2e-6)
    np.testing.assert_allclose(ratio_gradient(*theta), numerical, atol=1e-8)


def test_covariance_cancels_compliance_composition_noise():
    flags = np.tile([0.0, 1.0], 100)
    result = estimate_late(2 * flags, np.zeros(200), flags, 1, 0)
    assert result["estimate"] == 2
    assert result["cov_itt_flag"] > 0
    assert result["se"] < 1e-8


def test_independent_validation_uncertainty_increases_variance():
    flags = np.tile([0.0, 1.0], 100)
    y = 2 * flags + np.linspace(-1, 1, len(flags))
    known = estimate_late(y, np.zeros(200), flags, 0.8, 0.05)
    estimated = estimate_late(y, np.zeros(200), flags, 0.8, 0.05, (0.001, 0.0002))
    assert known["estimate"] == estimated["estimate"]
    assert estimated["se"] > known["se"]


@pytest.mark.parametrize("q", [0.0, 0.1, 1.0])
def test_infeasible_compliance_is_reported_not_clipped(q):
    flags = np.r_[np.ones(round(q * 100)), np.zeros(100 - round(q * 100))]
    with pytest.raises(ValueError, match="compliance outside"):
        estimate_late(np.ones(100), np.zeros(100), flags, 0.8, 0.1)


def test_gaussian_classifier_rates_match_independent_simulation():
    design = Design()
    receipt = np.repeat([0.0, 1.0], 30000)
    pre, delivery, _ = windows(receipt, design, np.random.default_rng(615))
    flags = receipt_flag(pre, delivery, design.alpha)
    sensitivity, false_positive = gaussian_rates(design)
    assert abs(flags[:30000].mean() - false_positive) < 0.005
    assert abs(flags[30000:].mean() - sensitivity) < 0.008


def test_serial_correlation_cannot_use_exact_iid_calibration():
    with pytest.raises(ValueError, match="independent daily noise"):
        gaussian_rates(Design(rho=0.6))


@pytest.mark.parametrize("sign", [-1, 1])
def test_partial_identification_contains_imperfect_sensitivity_truth(sign):
    pi, sensitivity, false_positive, tau = 0.5, 0.5, 0.025, sign * 2
    q = pi * sensitivity + (1 - pi) * false_positive
    shares, effects = sensitivity_bounds(pi * tau, q, false_positive, 0.5, 1)
    assert shares[0] <= pi <= shares[1]
    assert effects[0] <= tau <= effects[1]
    assert (q - false_positive) / (1 - false_positive) < pi


def test_bounds_enforce_population_feasibility():
    shares, _ = sensitivity_bounds(1, 0.4, 0.05, 0.1, 0.8)
    assert shares[1] == 1
    with pytest.raises(ValueError, match="incompatible"):
        sensitivity_bounds(1, 0.4, 0.05, 0.1, 0.3)


def test_known_calibration_recovers_late_and_coverage():
    result = run_scenario(Design(), "known", 500, 891)
    assert abs(result["bias"]) < 3
    assert 0.91 < result["coverage"] < 0.99
    assert result["unavailable"] == 0


def test_generated_outputs_account_for_all_replications(tmp_path):
    rows = run_study(10, 903, tmp_path)
    with (tmp_path / "coverage.csv").open() as stream:
        stored = list(csv.DictReader(stream))
    assert len(stored) == len(rows) == 8
    for row in rows:
        assert row["repetitions"] == 10
        assert 0 <= row["coverage"] <= 1 - row["unavailable"] + 1e-12
    assert (tmp_path / "coverage_table.tex").read_text().count("\\\\") == 9
