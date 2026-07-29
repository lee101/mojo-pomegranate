"""Probability distributions supported by the Mojo-backed models."""

from __future__ import annotations

import math

import numpy as np

from .._lib import addr, f64, i64, lib


class Distribution:
    def __init__(self, inertia=0.0, frozen=False, check_data=True):
        self.inertia = float(inertia)
        self.frozen = bool(frozen)
        self.check_data = bool(check_data)

    def probability(self, X):
        return np.exp(self.log_probability(X))

    def freeze(self):
        self.frozen = True
        return self

    def unfreeze(self):
        self.frozen = False
        return self


class Normal(Distribution):
    def __init__(
        self,
        means=None,
        covs=None,
        covariance_type="full",
        min_cov=None,
        inertia=0.0,
        frozen=False,
        check_data=True,
    ):
        super().__init__(inertia, frozen, check_data)
        self.name = "Normal"
        self.means = None if means is None else f64(means)
        self.covs = None if covs is None else f64(covs)
        self.covariance_type = covariance_type
        self.min_cov = None if min_cov is None else float(min_cov)
        self.d = None if self.means is None else self.means.size
        self._initialized = self.means is not None and self.covs is not None
        if covariance_type not in ("diag", "sphere", "full"):
            raise ValueError("covariance_type must be 'diag', 'sphere', or 'full'")

    def _diag_covs(self) -> np.ndarray:
        if self.covariance_type == "sphere":
            return np.full(self.d, float(np.ravel(self.covs)[0]))
        return self.covs

    def log_probability(self, X):
        X = f64(X)
        if X.ndim != 2:
            raise ValueError("X must have shape (n, d)")
        if not self._initialized or X.shape[1] != self.d:
            raise ValueError("Normal is uninitialized or X has the wrong dimensionality")
        if self.covariance_type == "full":
            delta = X - self.means
            sign, logdet = np.linalg.slogdet(self.covs)
            if sign <= 0:
                raise ValueError("covariance matrix must be positive definite")
            solved = np.linalg.solve(self.covs, delta.T).T
            return -0.5 * (
                self.d * math.log(2 * math.pi)
                + logdet
                + np.einsum("ij,ij->i", delta, solved)
            )
        covs = self._diag_covs()
        means = self.means.reshape(1, -1)
        invvars = np.reciprocal(covs).reshape(1, -1)
        norms = np.array(
            [-0.5 * (self.d * math.log(2 * math.pi) + np.log(covs).sum())]
        )
        priors = np.zeros(1)
        result = np.empty((len(X), 1))
        lib().mp_normal_emissions(
            addr(X), addr(means), addr(invvars), addr(norms), addr(priors),
            addr(result), len(X), self.d, 1,
        )
        return result[:, 0]

    def fit(self, X, sample_weight=None):
        if self.frozen:
            return self
        X = f64(X)
        if X.ndim != 2 or not len(X) or not X.shape[1]:
            raise ValueError("X must have non-zero shape (n, d)")
        weights = np.ones(len(X)) if sample_weight is None else f64(sample_weight).reshape(-1)
        if len(weights) != len(X) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weight must be finite, non-negative, and match X")
        total = weights.sum()
        if total <= 0:
            raise ValueError("sample_weight must have positive total weight")
        means = (X * weights[:, None]).sum(axis=0) / total
        delta = X - means
        if self.covariance_type == "full":
            covs = (delta.T * weights) @ delta / total
            if self.min_cov is not None:
                diagonal = np.diag_indices_from(covs)
                covs[diagonal] = np.maximum(covs[diagonal], self.min_cov)
        elif self.covariance_type == "sphere":
            covs = np.array([(weights[:, None] * delta**2).sum() / (total * X.shape[1])])
            if self.min_cov is not None:
                covs = np.maximum(covs, self.min_cov)
        else:
            covs = (weights[:, None] * delta**2).sum(axis=0) / total
            if self.min_cov is not None:
                covs = np.maximum(covs, self.min_cov)
        if self._initialized and self.inertia:
            means = self.inertia * self.means + (1 - self.inertia) * means
            covs = self.inertia * self.covs + (1 - self.inertia) * covs
        self.means, self.covs = f64(means), f64(covs)
        self.d, self._initialized = X.shape[1], True
        return self

    def sample(self, n):
        rng = np.random.default_rng()
        if self.covariance_type == "full":
            return rng.multivariate_normal(self.means, self.covs, size=n)
        return rng.normal(self.means, np.sqrt(self._diag_covs()), size=(n, self.d))


class Categorical(Distribution):
    def __init__(
        self,
        probs=None,
        n_categories=None,
        pseudocount=0.0,
        inertia=0.0,
        frozen=False,
        check_data=True,
    ):
        super().__init__(inertia, frozen, check_data)
        self.name = "Categorical"
        self.probs = None if probs is None else f64(probs)
        self.n_keys = n_categories
        self.pseudocount = float(pseudocount)
        self._initialized = probs is not None
        self.d = None if probs is None else self.probs.shape[0]
        if self._initialized and self.n_keys is None:
            self.n_keys = self.probs.shape[1]

    def log_probability(self, X):
        X = i64(X)
        if X.ndim != 2 or not self._initialized or X.shape[1] != self.d:
            raise ValueError("Categorical is uninitialized or X has the wrong shape")
        if X.size and (X.min() < 0 or X.max() >= self.probs.shape[1]):
            raise ValueError("categorical values must be contiguous integers starting at zero")
        result = np.empty((len(X), 1))
        packed = np.ascontiguousarray(np.log(self.probs)[None, :, :])
        lib().mp_categorical_emissions(
            addr(X), addr(packed), addr(result), len(X), self.d, 1, self.probs.shape[1]
        )
        return result[:, 0]

    def fit(self, X, sample_weight=None):
        if self.frozen:
            return self
        X = i64(X)
        if X.ndim != 2 or not len(X) or not X.shape[1]:
            raise ValueError("X must have non-zero shape (n, d)")
        weights = np.ones(len(X)) if sample_weight is None else f64(sample_weight).reshape(-1)
        if len(weights) != len(X) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weight must be finite, non-negative, and match X")
        if weights.sum() <= 0:
            raise ValueError("sample_weight must have positive total weight")
        categories = int(self.n_keys or (X.max() + 1))
        counts = np.full((X.shape[1], categories), self.pseudocount)
        for j in range(X.shape[1]):
            counts[j] += np.bincount(X[:, j], weights=weights, minlength=categories)
        probs = counts / counts.sum(axis=1, keepdims=True)
        if self._initialized and self.inertia:
            probs = self.inertia * self.probs + (1 - self.inertia) * probs
        self.probs = f64(probs)
        self.n_keys, self.d, self._initialized = categories, X.shape[1], True
        return self

    def sample(self, n):
        rng = np.random.default_rng()
        result = np.empty((n, self.d), dtype=np.int64)
        for j in range(self.d):
            result[:, j] = rng.choice(self.probs.shape[1], size=n, p=self.probs[j])
        return result


class ConditionalCategorical(Distribution):
    def __init__(
        self,
        probs=None,
        n_categories=None,
        pseudocount=0,
        inertia=0.0,
        frozen=False,
        check_data=True,
    ):
        super().__init__(inertia, frozen, check_data)
        self.name = "ConditionalCategorical"
        self.probs = None if probs is None else [f64(prob) for prob in probs]
        self.n_categories = n_categories
        self.pseudocount = float(pseudocount)
        self._initialized = probs is not None
        self.d = None if probs is None else len(self.probs)
        self.n_parents = None if probs is None else self.probs[0].ndim

    def log_probability(self, X):
        X = i64(X)
        if X.ndim == 2:
            X = X[:, :, None]
        result = np.zeros(X.shape[0])
        for j, probs in enumerate(self.probs):
            indices = tuple(X[:, parent, j] for parent in range(X.shape[1]))
            result += np.log(probs[indices])
        return result


__all__ = ["Distribution", "Normal", "Categorical", "ConditionalCategorical"]
