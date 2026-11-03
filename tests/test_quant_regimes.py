import numpy as np
import pytest

from dmpe.quant_regimes import (
    GaussianHMM, build_regime_observations, fit_garch11, garch_next_variance,
)


def test_garch_parameters_are_stationary_and_forecast_is_positive():
    rng = np.random.default_rng(12)
    returns = rng.normal(0, 0.002, 1200)
    fit = fit_garch11(returns[:900])
    assert fit.omega > 0
    assert fit.alpha >= 0 and fit.beta >= 0
    assert fit.alpha + fit.beta < 0.999
    forecast = garch_next_variance(returns[900:], fit)
    assert np.isfinite(forecast).all()
    assert (forecast > 0).all()


def test_garch_forecast_is_causal():
    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.001, 300)
    fit = fit_garch11(returns[:200])
    first = garch_next_variance(returns[200:], fit)
    changed_future = returns[200:].copy()
    changed_future[20:] *= 20
    second = garch_next_variance(changed_future, fit)
    np.testing.assert_allclose(first[:20], second[:20])


def test_hmm_probabilities_are_filtered_and_normalized():
    rng = np.random.default_rng(8)
    low = rng.normal(0, 0.0005, (500, 2))
    high = rng.normal(0, 0.004, (500, 2))
    train = np.vstack([low, high])
    train[:, 1] = np.log(np.maximum(np.abs(train[:, 0]), 1e-8))
    model = GaussianHMM(n_states=3, max_iter=40).fit(train)
    obs = train[-50:].copy()
    probs = model.filtered_probabilities(obs)
    assert probs.shape == (50, 3)
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)
    altered = obs.copy()
    altered[30:] += 100
    altered_probs = model.filtered_probabilities(altered)
    np.testing.assert_allclose(probs[:30], altered_probs[:30], atol=1e-10)


def test_regime_observations_use_trailing_only_window():
    returns = np.linspace(-0.001, 0.001, 100)
    obs = build_regime_observations(returns, window=20)
    assert np.isnan(obs[:19, 1]).all()
    assert np.isfinite(obs[19:, 1]).all()
    altered = returns.copy()
    altered[80:] *= 100
    altered_obs = build_regime_observations(altered, window=20)
    np.testing.assert_allclose(obs[:80], altered_obs[:80], equal_nan=True)


def test_hmm_rejects_unfitted_filter():
    model = GaussianHMM()
    with pytest.raises(RuntimeError):
        model.filtered_probabilities(np.ones((2, 2)))


def test_kalman_filter_is_causal_after_training_cutoff():
    from dmpe.quant_regimes import KalmanTrendFilter
    rng = np.random.default_rng(44)
    prices = 10 + np.cumsum(rng.normal(0, 0.001, 500))
    model = KalmanTrendFilter().fit(prices[:400])
    a = model.filter(prices[400:])
    changed = prices[400:].copy()
    changed[30:] += 5
    b = model.filter(changed)
    np.testing.assert_allclose(a[:30], b[:30])


def test_bayesian_odds_update_matches_sequential_bayes():
    from dmpe.quant_regimes import bayesian_odds_update
    posterior = bayesian_odds_update(0.5, np.array([2.0, 0.5]))
    assert posterior[0] == pytest.approx(2.0 / 3.0)
    assert posterior[1] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        bayesian_odds_update(0.5, np.array([0.0]))


def test_monte_carlo_probabilities_sum_to_one_and_are_reproducible():
    from dmpe.quant_regimes import monte_carlo_barrier_probabilities
    rng = np.random.default_rng(22)
    history = rng.normal(0, 0.001, 1000)
    a = monte_carlo_barrier_probabilities(history, 0.2, 0.3, 30, n_paths=500, seed=9)
    b = monte_carlo_barrier_probabilities(history, 0.2, 0.3, 30, n_paths=500, seed=9)
    assert a == b
    total = a["target_first_probability"] + a["stop_first_probability"] + a["no_touch_probability"]
    assert total == pytest.approx(1.0)


def test_hmm_first_filtered_probability_uses_initial_state_prior():
    from dmpe.quant_regimes import GaussianHMM
    rng = np.random.default_rng(91)
    train = rng.normal(0, 1, (300, 2))
    model = GaussianHMM(n_states=3, max_iter=8).fit(train)
    first = train[:1]
    z = (first - model.scaler_mean_) / model.scaler_std_
    log_b = model._emission_log_prob(z)[0]
    expected_log = np.log(np.maximum(model.startprob_, 1e-300)) + log_b
    expected = np.exp(expected_log - np.max(expected_log))
    expected /= expected.sum()
    actual = model.filtered_probabilities(first)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_garch_qmle_uses_lagged_variance_and_recovers_persistent_process():
    rng = np.random.default_rng(42)
    omega, alpha, beta = 1e-6, 0.08, 0.90
    variance = omega / (1.0 - alpha - beta)
    returns = np.empty(5000)
    previous = 0.0
    for i in range(len(returns)):
        if i:
            variance = omega + alpha * previous * previous + beta * variance
        returns[i] = rng.normal(0.0, np.sqrt(variance))
        previous = returns[i]
    fit = fit_garch11(returns[:4000])
    assert fit.success
    assert fit.alpha == pytest.approx(alpha, abs=0.04)
    assert fit.beta == pytest.approx(beta, abs=0.06)
    assert fit.alpha + fit.beta < 0.999
