"""Categorical Bayesian networks with Mojo likelihood evaluation."""

from __future__ import annotations

import itertools

import numpy as np

from ._lib import addr, f64, i64, lib
from .distributions import Categorical, ConditionalCategorical


class BayesianNetwork:
    def __init__(
        self,
        distributions=None,
        edges=None,
        structure=None,
        algorithm=None,
        include_parents=None,
        exclude_parents=None,
        max_parents=None,
        pseudocount=0.0,
        max_iter=20,
        tol=1e-6,
        inertia=0.0,
        frozen=False,
        check_data=True,
        verbose=False,
    ):
        self.name = "BayesianNetwork"
        self.distributions = [] if distributions is None else list(distributions)
        self.edges = [] if edges is None else list(edges)
        self.structure = None if structure is None else tuple(tuple(p) for p in structure)
        self.algorithm = algorithm
        self.include_parents, self.exclude_parents = include_parents, exclude_parents
        self.max_parents, self.pseudocount = max_parents, float(pseudocount)
        self.max_iter, self.tol = int(max_iter), float(tol)
        self.inertia, self.frozen = float(inertia), bool(frozen)
        self.check_data, self.verbose = bool(check_data), bool(verbose)
        self.d = len(self.distributions)
        self._refresh_structure()

    def _refresh_structure(self):
        if self.structure is not None:
            self._parents = list(self.structure)
            return
        mapping = {id(dist): i for i, dist in enumerate(self.distributions)}
        parents = [[] for _ in self.distributions]
        for parent, child in self.edges:
            parents[mapping[id(child)]].append(mapping[id(parent)])
        self._parents = [tuple(values) for values in parents]

    def add_distribution(self, distribution):
        if not isinstance(distribution, (Categorical, ConditionalCategorical)):
            raise ValueError("Must be Categorical or ConditionalCategorical")
        self.distributions.append(distribution)
        self.d = len(self.distributions)
        self._refresh_structure()
        return self

    def add_distributions(self, distributions):
        for distribution in distributions:
            self.add_distribution(distribution)
        return self

    def add_edge(self, parent, child):
        if parent is child:
            raise ValueError("Cannot have self-loops")
        self.edges.append((parent, child))
        self._refresh_structure()
        return self

    def add_edges(self, edges):
        for edge in edges:
            self.add_edge(*edge)
        return self

    def _packed(self):
        categories = []
        cpts = []
        offsets = [0]
        parent_values = []
        parent_offsets = [0]
        for node, distribution in enumerate(self.distributions):
            if isinstance(distribution, Categorical):
                table = distribution.probs[0]
                categories.append(table.shape[-1])
            else:
                table = distribution.probs[0]
                categories.append(table.shape[-1])
            cpts.extend(np.ravel(table))
            offsets.append(len(cpts))
            parent_values.extend(self._parents[node])
            parent_offsets.append(len(parent_values))
        with np.errstate(divide="ignore"):
            log_cpts = np.ascontiguousarray(np.log(cpts), dtype=np.float64)
        return (
            log_cpts,
            i64(offsets),
            i64(parent_values),
            i64(parent_offsets),
            i64(categories),
        )

    def log_probability(self, X):
        X = i64(X)
        packed = self._packed()
        if not self.d or X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError("X must have shape (n, number_of_nodes)")
        categories = packed[-1]
        if X.size and any(
            np.any((X[:, node] < 0) | (X[:, node] >= category_count))
            for node, category_count in enumerate(categories)
        ):
            raise ValueError("categorical value is outside its node alphabet")
        result = np.empty(len(X))
        lib().mp_bn_log_probability(
            addr(X), *(addr(value) for value in packed), addr(result), len(X), self.d
        )
        return result

    def probability(self, X):
        return np.exp(self.log_probability(X))

    def fit(self, X, sample_weight=None):
        if self.frozen:
            return self
        if self.algorithm is not None:
            raise NotImplementedError("Bayesian-network structure learning is not covered")
        X = i64(X)
        if X.ndim != 2 or not len(X) or not X.shape[1]:
            raise ValueError("X must have non-zero shape (n, number_of_nodes)")
        weights = np.ones(len(X)) if sample_weight is None else f64(sample_weight).reshape(-1)
        if len(weights) != len(X) or np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weight must be finite, non-negative, and match X")
        if weights.sum() <= 0:
            raise ValueError("sample_weight must have positive total weight")
        if self.structure is not None and not self.distributions:
            categories = [int(X[:, j].max() + 1) for j in range(X.shape[1])]
            distributions = []
            for node, parents in enumerate(self.structure):
                shape = tuple(categories[p] for p in parents) + (categories[node],)
                if parents:
                    distributions.append(
                        ConditionalCategorical(
                            [np.full(shape, 1 / categories[node])],
                            pseudocount=self.pseudocount,
                        )
                    )
                else:
                    distributions.append(
                        Categorical(
                            [np.full(categories[node], 1 / categories[node])],
                            pseudocount=self.pseudocount,
                        )
                    )
            self.distributions = distributions
            self.d = len(distributions)
        for node, distribution in enumerate(self.distributions):
            parents = self._parents[node]
            if isinstance(distribution, Categorical):
                distribution.fit(X[:, node:node + 1], weights)
                continue
            shape = distribution.probs[0].shape
            counts = np.full(shape, distribution.pseudocount)
            indices = tuple(X[:, parent] for parent in parents) + (X[:, node],)
            np.add.at(counts, indices, weights)
            probs = counts / counts.sum(axis=-1, keepdims=True)
            distribution.probs = [f64(probs)]
        if self.structure is not None and not self.edges:
            for node, parents in enumerate(self._parents):
                for parent in parents:
                    self.edges.append((self.distributions[parent], self.distributions[node]))
        return self

    def _evidence(self, X):
        if hasattr(X, "_masked_data") and hasattr(X, "_masked_mask"):
            data = np.asarray(X._masked_data)
            mask = np.asarray(X._masked_mask)
            return i64(data), np.asarray(mask, dtype=bool)
        data = np.asarray(X)
        mask = np.isfinite(data) & (data >= 0)
        return i64(np.where(mask, data, 0)), mask

    def predict_proba(self, X):
        data, mask = self._evidence(X)
        categories = self._packed()[-1]
        assignments = i64(list(itertools.product(*(range(int(c)) for c in categories))))
        logp = self.log_probability(assignments)
        marginals = [np.zeros((len(data), int(c))) for c in categories]
        for row in range(len(data)):
            compatible = np.all(
                (assignments == data[row]) | ~mask[row],
                axis=1,
            )
            selected = logp[compatible]
            maximum = selected.max()
            weights = np.exp(selected - maximum)
            weights /= weights.sum()
            valid_assignments = assignments[compatible]
            for node, category_count in enumerate(categories):
                marginals[node][row] = np.bincount(
                    valid_assignments[:, node],
                    weights=weights,
                    minlength=int(category_count),
                )
        return marginals

    def predict_log_proba(self, X):
        with np.errstate(divide="ignore"):
            return [np.log(values) for values in self.predict_proba(X)]

    def predict(self, X):
        data, mask = self._evidence(X)
        result = data.copy()
        for node, marginal in enumerate(self.predict_proba(X)):
            result[~mask[:, node], node] = marginal[~mask[:, node]].argmax(axis=1)
        return result

    def sample(self, n):
        rng = np.random.default_rng()
        result = np.empty((int(n), self.d), dtype=np.int64)
        remaining = set(range(self.d))
        order = []
        while remaining:
            ready = [node for node in remaining if all(p in order for p in self._parents[node])]
            if not ready:
                raise ValueError("sampling requires an acyclic network")
            for node in ready:
                order.append(node)
                remaining.remove(node)
        for row in range(int(n)):
            for node in order:
                distribution = self.distributions[node]
                if isinstance(distribution, Categorical):
                    probabilities = distribution.probs[0]
                else:
                    index = tuple(result[row, parent] for parent in self._parents[node])
                    probabilities = distribution.probs[0][index]
                result[row, node] = rng.choice(len(probabilities), p=probabilities)
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
