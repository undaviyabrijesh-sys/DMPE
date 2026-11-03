"""Leakage-safe quantitative volatility and regime estimators.

GARCH parameters and HMM emissions/transitions are fitted on a training slice only.
Out-of-sample probabilities are filtered sequentially (never smoothed with future data).
Inputs are decimal 1-minute log returns. These outputs are research features, not signals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class GARCHFit:
    omega: float
    alpha: float
    beta: float
    train_mean: float
    train_variance: float
    success: bool
    objective: float


def _returns_array(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 50 or not np.isfinite(arr).all():
        raise ValueError("At least 50 finite training returns are required.")
    return arr


def _decode_garch(theta: np.ndarray) -> tuple[float, float, float]:
    omega = float(np.exp(np.clip(theta[0], -30.0, 5.0)))
    sigmoid_a = 1.0 / (1.0 + np.exp(-np.clip(theta[1], -30.0, 30.0)))
    alpha = 0.999 * sigmoid_a
    sigmoid_b = 1.0 / (1.0 + np.exp(-np.clip(theta[2], -30.0, 30.0)))
    beta = (0.999 - alpha) * sigmoid_b
    return omega, float(alpha), float(beta)


def fit_garch11(train_returns: pd.Series | np.ndarray) -> GARCHFit:
    """Fit Gaussian-QMLE GARCH(1,1), constraining alpha + beta < 0.999."""
    raw = _returns_array(train_returns)
    mean = float(raw.mean())
    centered = raw - mean
    variance = max(float(np.var(centered)), 1e-12)
    initial = np.array([np.log(variance * 0.05), -2.0, 3.0])

    def objective(theta: np.ndarray) -> float:
        omega, alpha, beta = _decode_garch(theta)
        h = variance
        total = 0.0
        for eps in centered:
            h = max(omega + alpha * eps * eps + beta * h, 1e-12)
            total += np.log(h) + (eps * eps) / h
        return 0.5 * total

    result = minimize(objective, initial, method="L-BFGS-B",
                      bounds=[(-30.0, 5.0), (-12.0, 12.0), (-12.0, 12.0)])
    omega, alpha, beta = _decode_garch(result.x)
    return GARCHFit(omega, alpha, beta, mean, variance,
                    bool(result.success), float(result.fun))


def garch_next_variance(
    returns: pd.Series | np.ndarray, fit: GARCHFit
) -> np.ndarray:
    """Return variance forecasts available after each observed return.

    The first value is initialized from training variance. Every subsequent
    forecast uses only returns observed up to that row.
    """
    arr = np.asarray(returns, dtype=float).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("Returns must be finite; split data at gaps before filtering.")
    out = np.empty(arr.size, dtype=float)
    h = fit.train_variance
    for i, value in enumerate(arr):
        eps = value - fit.train_mean
        h = max(fit.omega + fit.alpha * eps * eps + fit.beta * h, 1e-12)
        out[i] = h
    return out


class GaussianHMM:
    """Small diagonal-Gaussian HMM with train-only scaling and forward filtering."""

    def __init__(self, n_states: int = 3, max_iter: int = 150, tolerance: float = 1e-5,
                 variance_floor: float = 1e-4, sticky: float = 0.90):
        if n_states < 2 or max_iter < 1 or not 0.0 < sticky < 1.0:
            raise ValueError("Invalid HMM configuration.")
        self.n_states = n_states
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.variance_floor = variance_floor
        self.sticky = sticky
        self.fitted = False

    @staticmethod
    def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
        maximum = np.max(values, axis=axis, keepdims=True)
        result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
        return np.squeeze(result, axis=axis)

    def _emission_log_prob(self, x: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - self.means_[None, :, :]
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * self.variances_)[None, :, :]
            + diff * diff / self.variances_[None, :, :], axis=2
        )

    def fit(self, train_observations: np.ndarray) -> "GaussianHMM":
        x = np.asarray(train_observations, dtype=float)
        if x.ndim != 2 or x.shape[0] < max(100, self.n_states * 20) or not np.isfinite(x).all():
            raise ValueError("HMM training data must be a finite 2D array with adequate rows.")
        self.scaler_mean_ = x.mean(axis=0)
        self.scaler_std_ = np.maximum(x.std(axis=0), 1e-8)
        z = (x - self.scaler_mean_) / self.scaler_std_
        # Initialize states by ordered quantiles of the final (volatility) feature.
        order = np.argsort(z[:, -1])
        groups = np.array_split(order, self.n_states)
        self.means_ = np.vstack([z[g].mean(axis=0) for g in groups])
        self.variances_ = np.vstack([
            np.maximum(z[g].var(axis=0), self.variance_floor) for g in groups
        ])
        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
        self.transmat_ = np.full(
            (self.n_states, self.n_states),
            (1.0 - self.sticky) / (self.n_states - 1)
        )
        np.fill_diagonal(self.transmat_, self.sticky)

        previous_ll = -np.inf
        for _ in range(self.max_iter):
            log_b = self._emission_log_prob(z)
            log_a = np.log(np.maximum(self.transmat_, 1e-300))
            log_pi = np.log(np.maximum(self.startprob_, 1e-300))
            n = len(z)
            log_alpha = np.empty((n, self.n_states))
            log_alpha[0] = log_pi + log_b[0]
            for t in range(1, n):
                log_alpha[t] = log_b[t] + self._logsumexp(
                    log_alpha[t - 1][:, None] + log_a, axis=0
                )
            log_beta = np.zeros((n, self.n_states))
            for t in range(n - 2, -1, -1):
                log_beta[t] = self._logsumexp(
                    log_a + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1
                )
            log_likelihood = float(self._logsumexp(log_alpha[-1], axis=0))
            log_gamma = log_alpha + log_beta
            log_gamma -= self._logsumexp(log_gamma, axis=1)[:, None]
            gamma = np.exp(log_gamma)
            xi_sum = np.zeros_like(self.transmat_)
            for t in range(n - 1):
                log_xi = (log_alpha[t][:, None] + log_a + log_b[t + 1][None, :]
                          + log_beta[t + 1][None, :])
                log_xi -= self._logsumexp(log_xi.reshape(-1), axis=0)
                xi_sum += np.exp(log_xi)
            self.startprob_ = np.maximum(gamma[0], 1e-8)
            self.startprob_ /= self.startprob_.sum()
            self.transmat_ = xi_sum + 1e-3
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)
            weights = np.maximum(gamma.sum(axis=0), 1e-8)
            self.means_ = (gamma.T @ z) / weights[:, None]
            diff = z[:, None, :] - self.means_[None, :, :]
            self.variances_ = np.maximum(
                (gamma[:, :, None] * diff * diff).sum(axis=0) / weights[:, None],
                self.variance_floor
            )
            if np.isfinite(previous_ll) and abs(log_likelihood - previous_ll) < self.tolerance:
                break
            previous_ll = log_likelihood

        # Canonicalize state IDs from low to high volatility for stable reporting.
        state_order = np.argsort(self.means_[:, -1])
        self.means_ = self.means_[state_order]
        self.variances_ = self.variances_[state_order]
        self.startprob_ = self.startprob_[state_order]
        self.transmat_ = self.transmat_[state_order][:, state_order]
        self.fitted = True
        return self

    def filtered_probabilities(self, observations: np.ndarray) -> np.ndarray:
        """Forward-only state probabilities; no backward smoothing or future data."""
        if not self.fitted:
            raise RuntimeError("Fit HMM on training data before filtering.")
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.scaler_mean_.size or not np.isfinite(x).all():
            raise ValueError("Filtering observations must be finite and match training features.")
        z = (x - self.scaler_mean_) / self.scaler_std_
        log_b = self._emission_log_prob(z)
        probs = np.empty((len(z), self.n_states), dtype=float)
        prior = self.startprob_.copy()
        for t in range(len(z)):
            prior = prior @ self.transmat_
            log_p = np.log(np.maximum(prior, 1e-300)) + log_b[t]
            log_p -= self._logsumexp(log_p, axis=0)
            posterior = np.exp(log_p)
            probs[t] = posterior
            prior = posterior
        return probs


def build_regime_observations(log_returns: pd.Series | np.ndarray, window: int = 60) -> np.ndarray:
    """Construct past-only HMM inputs: return and trailing realized log-volatility."""
    returns = np.asarray(log_returns, dtype=float).reshape(-1)
    if window < 10 or not np.isfinite(returns).all():
        raise ValueError("Use finite returns and a volatility window of at least 10.")
    trailing_vol = pd.Series(returns).rolling(window, min_periods=window).std(ddof=0).to_numpy()
    safe_vol = np.maximum(trailing_vol, 1e-12)
    return np.column_stack([returns, np.log(safe_vol)])
