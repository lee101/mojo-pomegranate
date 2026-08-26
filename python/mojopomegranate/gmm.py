"""General mixture models with Mojo-backed inference and Gaussian EM."""

from __future__ import annotations

import math

import numpy as np

from ._lib import addr, f64, lib
from .distributions import Normal


class GeneralMixtureModel:
    def __init__(
        self,
        distributions,
        priors=None,
        init="random",
        max_iter=1000,
        tol=0.1,
        inertia=0.0,
        frozen=False,
        random_state=None,
        check_data=True,
        verbose=False,
    ):
        self.name = "GeneralMixtureModel"
        self.distributions = list(distributions)
        self.k = len(self.distributions)
        if not self.k:
            raise ValueError("GeneralMixtureModel requires at least one distribution")
        self.priors = f64(priors if priors is not None else np.full(self.k, 1 / self.k))
        if self.priors.shape != (self.k,) or np.any(~np.isfinite(self.priors)):
            raise ValueError("priors must be a finite vector with one value per component")
        if np.any(self.priors < 0) or not np.isclose(self.priors.sum(), 1.0):
            raise ValueError("priors must be non-negative and sum to one")
        self.init = init
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.inertia = float(inertia)
        self.frozen = bool(frozen)
        self.random_state = random_state
        self.check_data = bool(check_data)
        self.verbose = bool(verbose)
        self.d = self.distributions[0].d if self.distributions else None
        self._initialized = all(d._initialized for d in self.distributions)

    def _initialize(self, X):
        X = f64(X)
        rng = np.random.default_rng(self.random_state)
        order = rng.choice(len(X), self.k, replace=False)
        labels = np.argmin(((X[:, None] - X[order][None]) ** 2).sum(axis=2), axis=1)
        for c, distribution in enumerate(self.distributions):
            selected = X[labels == c]
            if not len(selected):
                selected = X[order[c:c + 1]]
            if isinstance(distribution, Normal) and distribution.covariance_type == "full":
                distribution.covariance_type = "diag"
            distribution.fit(selected)
        self.priors = np.bincount(labels, minlength=self.k).astype(float)
        self.priors /= self.priors.sum()
        self.d, self._initialized = X.shape[1], True

    def _emission_matrix(self, X, priors=None):
        X = f64(X)
        if X.ndim != 2 or not len(X) or not X.shape[1]:
            raise ValueError("X must have non-zero shape (n, model dimensionality)")
        if self._initialized and X.shape[1] != self.d:
            raise ValueError("X dimensionality does not match the model")
        if not self._initialized:
            self._initialize(X)
        if all(
            isinstance(d, Normal) and d.covariance_type in ("diag", "sphere")
            for d in self.distributions
        ):
            means = np.ascontiguousarray(np.vstack([d.means for d in self.distributions]))
            variances = np.ascontiguousarray(
                np.vstack([d._diag_covs() for d in self.distributions])
            )
            invvars = np.ascontiguousarray(1.0 / variances)
            norms = np.ascontiguousarray(
                -0.5 * (self.d * math.log(2 * math.pi) + np.log(variances).sum(axis=1))
            )
            log_priors = np.ascontiguousarray(np.log(self.priors))
            result = np.empty((len(X), self.k))
            lib().mp_normal_emissions(
                addr(X), addr(means), addr(invvars), addr(norms), addr(log_priors),
                addr(result), len(X), self.d, self.k,
            )
        else:
            result = np.column_stack(
                [distribution.log_probability(X) for distribution in self.distributions]
            )
            result += np.log(self.priors)
        if priors is not None:
            observation_priors = f64(priors)
            if observation_priors.shape != result.shape:
                raise ValueError("priors must have shape (n, number_of_components)")
            result += np.log(observation_priors)
        return np.ascontiguousarray(result)

    def _posteriors(self, X, priors=None, probability=False, device="cpu"):
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        emissions = self._emission_matrix(X, priors)
        logps = np.empty(len(emissions))
        if device == "gpu" and lib().mp_mixture_gpu(
            addr(emissions), addr(logps), len(emissions), self.k, int(probability)
        ):
            return emissions, logps
        function = (
            lib().mp_mixture_probabilities
            if probability
            else lib().mp_mixture_posteriors
        )
        function(addr(emissions), addr(emissions), addr(logps), len(emissions), self.k)
        return emissions, logps

    def log_probability(self, X, priors=None, device="cpu"):
        return self._posteriors(X, priors, device=device)[1]

    def probability(self, X, priors=None, device="cpu"):
        return np.exp(self.log_probability(X, priors, device=device))

    def predict_log_proba(self, X, priors=None, device="cpu"):
        return self._posteriors(X, priors, device=device)[0]

    def predict_proba(self, X, priors=None, device="cpu"):
        return self._posteriors(X, priors, probability=True, device=device)[0]

    def predict(self, X, priors=None):
        return np.argmax(self._emission_matrix(X, priors), axis=1)

    def fit(self, X, sample_weight=None, priors=None, device="cpu"):
        X = f64(X)
        if sample_weight is not None:
            raise NotImplementedError("weighted GeneralMixtureModel.fit is not covered")
        if not all(
            isinstance(d, Normal) and d.covariance_type in ("diag", "sphere")
            for d in self.distributions
        ):
            raise NotImplementedError("EM fitting is covered for diagonal Normal components")
        if not self._initialized:
            self._initialize(X)
        previous = -np.inf
        counts = np.empty(self.k)
        sums = np.empty((self.k, X.shape[1]))
        sumsq = np.empty_like(sums)
        for iteration in range(self.max_iter):
            log_post, logps = self._posteriors(X, priors, device=device)
            total = float(logps.sum())
            if iteration and total - previous < self.tol:
                break
            lib().mp_weighted_stats(
                addr(X), addr(log_post), addr(counts), addr(sums), addr(sumsq),
                len(X), X.shape[1], self.k,
            )
            new_priors = counts / counts.sum()
            new_means = sums / counts[:, None]
            new_vars = sumsq / counts[:, None] - new_means**2
            for c, distribution in enumerate(self.distributions):
                floor = max(distribution.min_cov or 0.0, np.finfo(float).eps)
                covs = np.maximum(new_vars[c], floor)
                if distribution.covariance_type == "sphere":
                    covs = np.array([covs.mean()])
                distribution.means = (
                    self.inertia * distribution.means
                    + (1 - self.inertia) * new_means[c]
                )
                distribution.covs = (
                    self.inertia * distribution.covs + (1 - self.inertia) * covs
                )
            self.priors = self.inertia * self.priors + (1 - self.inertia) * new_priors
            if self.verbose and iteration:
                print(f"[{iteration}] Improvement: {total - previous}")
            previous = total
        return self

    def sample(self, n):
        rng = np.random.default_rng(self.random_state)
        labels = rng.choice(self.k, size=n, p=self.priors)
        result = np.empty((n, self.d))
        for c, distribution in enumerate(self.distributions):
            mask = labels == c
            if mask.any():
                result[mask] = distribution.sample(int(mask.sum()))
        return result

    def freeze(self):
        self.frozen = True
        for distribution in self.distributions:
            distribution.freeze()
        return self

    def unfreeze(self):
        self.frozen = False
        for distribution in self.distributions:
            distribution.unfreeze()
        return self
