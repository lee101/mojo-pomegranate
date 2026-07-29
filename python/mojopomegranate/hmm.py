"""Dense and sparse-facing hidden Markov models."""

from __future__ import annotations

import math

import numpy as np

from ._lib import addr, f64, i64, lib
from .distributions import Categorical, Normal


class _Silent:
    pass


class DenseHMM:
    def __init__(
        self,
        distributions=None,
        edges=None,
        starts=None,
        ends=None,
        init="random",
        max_iter=1000,
        tol=0.1,
        sample_length=None,
        return_sample_paths=False,
        inertia=0.0,
        frozen=False,
        check_data=True,
        random_state=None,
        verbose=False,
    ):
        self.name = "DenseHMM"
        self.distributions = [] if distributions is None else list(distributions)
        self.start, self.end = _Silent(), _Silent()
        self.edges = None if edges is None else self._log(edges)
        self.starts = None if starts is None else self._log(starts)
        self._explicit_ends = ends is not None
        self.ends = None if ends is None else self._log(ends)
        self.init, self.max_iter, self.tol = init, int(max_iter), float(tol)
        self.sample_length = sample_length
        self.return_sample_paths = return_sample_paths
        self.inertia, self.frozen = float(inertia), bool(frozen)
        self.check_data, self.random_state = bool(check_data), random_state
        self.verbose = bool(verbose)
        self.d = self.distributions[0].d if self.distributions else None
        self._ensure_defaults()

    @staticmethod
    def _log(values):
        with np.errstate(divide="ignore"):
            return np.ascontiguousarray(np.log(f64(values)))

    @property
    def k(self):
        return len(self.distributions)

    @property
    def n_distributions(self):
        return self.k

    def _ensure_defaults(self):
        if not self.k:
            return
        if self.starts is None:
            self.starts = np.full(self.k, -math.log(self.k))
        if self.ends is None:
            self.ends = np.full(self.k, -math.log(self.k))
        if self.edges is None:
            self.edges = np.full((self.k, self.k), -math.log(self.k))

    def add_distribution(self, distribution):
        self.distributions.append(distribution)
        self.d = distribution.d
        return self

    def add_distributions(self, distributions):
        self.distributions.extend(distributions)
        if self.distributions:
            self.d = self.distributions[0].d
        return self

    def add_edge(self, start, end, prob):
        n = self.k
        if start is self.start:
            if self.starts is None or len(self.starts) != n:
                self.starts = np.full(n, -np.inf)
            self.starts[self.distributions.index(end)] = math.log(prob)
        elif end is self.end:
            if self.ends is None or len(self.ends) != n:
                self.ends = np.full(n, -np.inf)
            self.ends[self.distributions.index(start)] = math.log(prob)
            self._explicit_ends = True
        else:
            if self.edges is None or self.edges.shape != (n, n):
                self.edges = np.full((n, n), -np.inf)
            self.edges[self.distributions.index(start), self.distributions.index(end)] = math.log(prob)
        return self

    def _emission_matrix(self, X, priors=None):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("X must have shape (batch, length, d)")
        batch, length, d = X.shape
        if batch < 1 or length < 1 or d < 1:
            raise ValueError("X dimensions must all be non-zero")
        flat_n = batch * length
        if all(isinstance(distribution, Categorical) for distribution in self.distributions):
            Xi = i64(X.reshape(flat_n, d))
            categories = max(distribution.probs.shape[1] for distribution in self.distributions)
            if Xi.size and (Xi.min() < 0 or Xi.max() >= categories):
                raise ValueError("categorical values are outside the model alphabet")
            probs = np.zeros((self.k, d, categories))
            for c, distribution in enumerate(self.distributions):
                probs[c, :, : distribution.probs.shape[1]] = distribution.probs
            with np.errstate(divide="ignore"):
                log_probs = np.ascontiguousarray(np.log(probs))
            emissions = np.empty((flat_n, self.k))
            lib().mp_categorical_emissions(
                addr(Xi), addr(log_probs), addr(emissions), flat_n, d, self.k, categories
            )
        elif all(
            isinstance(distribution, Normal)
            and distribution.covariance_type in ("diag", "sphere")
            for distribution in self.distributions
        ):
            Xf = f64(X.reshape(flat_n, d))
            means = np.ascontiguousarray(np.vstack([dist.means for dist in self.distributions]))
            variances = np.ascontiguousarray(
                np.vstack([dist._diag_covs() for dist in self.distributions])
            )
            invvars = np.ascontiguousarray(1.0 / variances)
            norms = np.ascontiguousarray(
                -0.5 * (d * math.log(2 * math.pi) + np.log(variances).sum(axis=1))
            )
            zero_priors = np.zeros(self.k)
            emissions = np.empty((flat_n, self.k))
            lib().mp_normal_emissions(
                addr(Xf), addr(means), addr(invvars), addr(norms), addr(zero_priors),
                addr(emissions), flat_n, d, self.k,
            )
        else:
            flat = X.reshape(flat_n, d)
            emissions = np.column_stack(
                [distribution.log_probability(flat) for distribution in self.distributions]
            )
        emissions = np.ascontiguousarray(emissions.reshape(batch, length, self.k))
        if priors is not None:
            emissions += np.log(f64(priors))
        return emissions

    def _inputs(self, X, emissions, priors):
        if emissions is None:
            if X is None:
                raise ValueError("Must pass one of X or emissions")
            emissions = self._emission_matrix(X, priors)
        emissions = f64(emissions)
        if emissions.ndim != 3 or emissions.shape[2] != self.k:
            raise ValueError("emissions must have shape (batch, length, k)")
        if not emissions.shape[0] or not emissions.shape[1] or not self.k:
            raise ValueError("emissions dimensions must all be non-zero")
        self._ensure_defaults()
        if (
            self.edges.shape != (self.k, self.k)
            or self.starts.shape != (self.k,)
            or self.ends.shape != (self.k,)
        ):
            raise ValueError("HMM edges, starts, and ends do not match its state count")
        return emissions

    def forward(self, X=None, emissions=None, priors=None):
        emissions = self._inputs(X, emissions, priors)
        result = np.empty_like(emissions)
        batch, length, k = emissions.shape
        lib().mp_hmm_forward(
            addr(emissions), addr(self.edges), addr(self.starts), addr(result),
            batch, length, k,
        )
        return result

    def backward(self, X=None, emissions=None, priors=None):
        emissions = self._inputs(X, emissions, priors)
        result = np.empty_like(emissions)
        batch, length, k = emissions.shape
        lib().mp_hmm_backward(
            addr(emissions), addr(self.edges), addr(self.ends), addr(result),
            batch, length, k,
        )
        return result

    def forward_backward(self, X=None, emissions=None, priors=None):
        emissions = self._inputs(X, emissions, priors)
        batch, length, k = emissions.shape
        alpha = np.empty_like(emissions)
        beta = np.empty_like(emissions)
        post = np.empty_like(emissions)
        transitions = np.empty((batch, k, k))
        logps = np.empty(batch)
        lib().mp_hmm_finish(
            addr(emissions), addr(self.edges), addr(self.starts), addr(self.ends),
            addr(alpha), addr(beta), addr(post), addr(transitions), addr(logps),
            batch, length, k,
        )
        return transitions, post, np.exp(post[:, 0]), np.exp(post[:, -1]), logps

    def log_probability(self, X, priors=None):
        emissions = self._emission_matrix(X, priors)
        alpha = self.forward(emissions=emissions)
        last = alpha[:, -1] + self.ends
        largest = last.max(axis=1)
        return largest + np.log(np.exp(last - largest[:, None]).sum(axis=1))

    def probability(self, X, priors=None):
        return np.exp(self.log_probability(X, priors))

    def predict_log_proba(self, X, priors=None):
        return self.forward_backward(X, priors=priors)[1]

    def predict_proba(self, X, priors=None):
        return np.exp(self.predict_log_proba(X, priors))

    def predict(self, X, priors=None):
        return np.argmax(self.predict_log_proba(X, priors), axis=-1)

    def viterbi(self, X=None, emissions=None, priors=None):
        emissions = self._inputs(X, emissions, priors)
        batch, length, k = emissions.shape
        scores = np.empty_like(emissions)
        traceback = np.empty(emissions.shape, dtype=np.int64)
        paths = np.empty((batch, length), dtype=np.int64)
        lib().mp_hmm_viterbi(
            addr(emissions), addr(self.edges), addr(self.starts), addr(self.ends),
            addr(scores), addr(traceback), addr(paths), batch, length, k,
        )
        return paths

    def fit(self, X, sample_weight=None, priors=None):
        if self.frozen:
            return self
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("fit currently requires equal-length sequences")
        if sample_weight is not None:
            weights = f64(sample_weight).reshape(-1)
        else:
            weights = np.ones(len(X))
        if len(weights) != len(X) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weight must be finite, non-negative, and match X")
        if weights.sum() <= 0:
            raise ValueError("sample_weight must have positive total weight")
        previous = -np.inf
        for iteration in range(self.max_iter):
            emissions = self._emission_matrix(X, priors)
            transitions, log_post, starts, ends, logps = self.forward_backward(
                emissions=emissions
            )
            total = float(np.dot(weights, logps))
            if iteration and total - previous < self.tol:
                break
            weighted_post = log_post + np.log(weights)[:, None, None]
            flat_post = np.ascontiguousarray(weighted_post.reshape(-1, self.k))
            flat_x = X.reshape(-1, X.shape[-1])
            if all(isinstance(dist, Normal) for dist in self.distributions):
                flat_x = f64(flat_x)
                counts = np.empty(self.k)
                sums = np.empty((self.k, X.shape[-1]))
                sumsq = np.empty_like(sums)
                lib().mp_weighted_stats(
                    addr(flat_x), addr(flat_post), addr(counts), addr(sums), addr(sumsq),
                    len(flat_x), X.shape[-1], self.k,
                )
                means = sums / counts[:, None]
                variances = sumsq / counts[:, None] - means**2
                for c, dist in enumerate(self.distributions):
                    floor = max(dist.min_cov or 0.0, np.finfo(float).eps)
                    dist.means = self.inertia * dist.means + (1 - self.inertia) * means[c]
                    dist.covs = self.inertia * dist.covs + (1 - self.inertia) * np.maximum(
                        variances[c], floor
                    )
            elif all(isinstance(dist, Categorical) for dist in self.distributions):
                flat_x = i64(flat_x)
                categories = max(dist.probs.shape[1] for dist in self.distributions)
                counts = np.empty((self.k, X.shape[-1], categories))
                lib().mp_weighted_categorical_stats(
                    addr(flat_x), addr(flat_post), addr(counts), len(flat_x),
                    X.shape[-1], self.k, categories,
                )
                for c, dist in enumerate(self.distributions):
                    adjusted = counts[c] + dist.pseudocount
                    update = adjusted / adjusted.sum(axis=1, keepdims=True)
                    dist.probs = self.inertia * dist.probs + (1 - self.inertia) * update
            transition_counts = (transitions * weights[:, None, None]).sum(axis=0)
            end_counts = (ends * weights[:, None]).sum(axis=0)
            start_counts = (starts * weights[:, None]).sum(axis=0)
            denom = transition_counts.sum(axis=1) + end_counts
            with np.errstate(divide="ignore"):
                new_edges = np.log(transition_counts / denom[:, None])
                new_ends = np.log(end_counts / denom)
                new_starts = np.log(start_counts / start_counts.sum())
            self.edges = self.inertia * self.edges + (1 - self.inertia) * new_edges
            self.ends = self.inertia * self.ends + (1 - self.inertia) * new_ends
            self.starts = self.inertia * self.starts + (1 - self.inertia) * new_starts
            if self.verbose and iteration:
                print(f"[{iteration}] Improvement: {total - previous}")
            previous = total
        return self

    def sample(self, n):
        if self.sample_length is None and not self._explicit_ends:
            raise ValueError("Must specify sample_length or explicit end probabilities")
        rng = np.random.default_rng(self.random_state)
        starts = np.exp(self.starts)
        edges = np.column_stack([np.exp(self.edges), np.exp(self.ends)])
        samples, paths = [], []
        for _ in range(int(n)):
            state = int(rng.choice(self.k, p=starts / starts.sum()))
            values, states = [], []
            limit = self.sample_length or 1_000_000
            for _ in range(limit):
                values.append(self.distributions[state].sample(1)[0])
                states.append(state)
                if len(values) == limit:
                    break
                probabilities = edges[state]
                state = int(rng.choice(self.k + 1, p=probabilities / probabilities.sum()))
                if state == self.k:
                    break
            samples.append(np.asarray(values))
            paths.append(states)
        return (samples, paths) if self.return_sample_paths else samples

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


class SparseHMM(DenseHMM):
    def __init__(
        self,
        distributions=None,
        edges=None,
        starts=None,
        ends=None,
        init="random",
        max_iter=1000,
        tol=0.1,
        sample_length=None,
        return_sample_paths=False,
        inertia=0.0,
        frozen=False,
        check_data=True,
        random_state=None,
        verbose=False,
    ):
        dense_edges = None
        if edges is not None and distributions is not None:
            dense_edges = np.zeros((len(distributions), len(distributions)))
            for source, destination, probability in edges:
                dense_edges[distributions.index(source), distributions.index(destination)] = probability
        super().__init__(
            distributions=distributions,
            edges=dense_edges,
            starts=starts,
            ends=ends,
            init=init,
            max_iter=max_iter,
            tol=tol,
            sample_length=sample_length,
            return_sample_paths=return_sample_paths,
            inertia=inertia,
            frozen=frozen,
            check_data=check_data,
            random_state=random_state,
            verbose=verbose,
        )
        self.name = "SparseHMM"
