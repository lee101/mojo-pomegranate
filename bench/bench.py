"""Benchmarks against pomegranate on identical CPU inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojopomegranate.bayesian_network import BayesianNetwork  # noqa: E402
from mojopomegranate.distributions import (  # noqa: E402
    Categorical,
    ConditionalCategorical,
    Normal,
)
from mojopomegranate.gmm import GeneralMixtureModel  # noqa: E402
from mojopomegranate.hmm import DenseHMM  # noqa: E402
from pomegranate.bayesian_network import BayesianNetwork as UpBayesianNetwork  # noqa: E402
from pomegranate.distributions import Categorical as UpCategorical  # noqa: E402
from pomegranate.distributions import ConditionalCategorical as UpConditional  # noqa: E402
from pomegranate.distributions import Normal as UpNormal  # noqa: E402
from pomegranate.gmm import GeneralMixtureModel as UpGMM  # noqa: E402
from pomegranate.hmm import DenseHMM as UpDenseHMM  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def processor():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


CASES = []


def case(name):
    def register(function):
        CASES.append((name, function))
        return function

    return register


@case("Normal.log_probability (1M x 8)")
def normal_logp():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1_000_000, 8))
    means = rng.normal(size=8)
    covs = rng.uniform(0.5, 2, size=8)
    ours = Normal(means, covs, covariance_type="diag")
    upstream = UpNormal(means, covs, covariance_type="diag")
    return lambda: ours.log_probability(X), lambda: upstream.log_probability(X)


@case("GMM.predict_proba k=4 (300k x 8)")
def mixture_predict():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300_000, 8))
    means = rng.normal(size=(4, 8))
    covs = rng.uniform(0.5, 2, size=(4, 8))
    ours = GeneralMixtureModel(
        [Normal(means[i], covs[i], covariance_type="diag") for i in range(4)]
    )
    upstream = UpGMM(
        [UpNormal(means[i], covs[i], covariance_type="diag") for i in range(4)]
    )
    return lambda: ours.predict_proba(X), lambda: upstream.predict_proba(X)


@case("GMM.fit k=3, 8 EM steps (100k x 6)")
def mixture_fit():
    rng = np.random.default_rng(2)
    X = np.vstack(
        [rng.normal(i * 2, 1, size=(100_000 // 3, 6)) for i in range(3)]
    )
    means = np.array([[-1.0] * 6, [1.0] * 6, [3.0] * 6])
    covs = np.ones((3, 6))

    def ours():
        GeneralMixtureModel(
            [Normal(means[i], covs[i], covariance_type="diag") for i in range(3)],
            max_iter=8,
            tol=0,
        ).fit(X)

    def upstream():
        UpGMM(
            [UpNormal(means[i], covs[i], covariance_type="diag") for i in range(3)],
            max_iter=8,
            tol=0,
        ).fit(X)

    return ours, upstream


@case("DenseHMM.forward k=6 (1k x 300)")
def hmm_forward():
    rng = np.random.default_rng(3)
    X = rng.integers(0, 4, size=(1_000, 300, 1))
    probabilities = rng.dirichlet(np.ones(4), size=6)
    edges = rng.dirichlet(np.ones(7), size=6)
    starts = rng.dirichlet(np.ones(6))
    ours = DenseHMM(
        [Categorical([probabilities[i]]) for i in range(6)],
        edges=edges[:, :6],
        starts=starts,
        ends=edges[:, 6],
    )
    upstream = UpDenseHMM(
        [UpCategorical([probabilities[i].tolist()]) for i in range(6)],
        edges=edges[:, :6].tolist(),
        starts=starts.tolist(),
        ends=edges[:, 6].tolist(),
    )
    return lambda: ours.forward(X), lambda: upstream.forward(X)


@case("DenseHMM.predict_proba k=4 (500 x 300)")
def hmm_posterior():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 300, 3))
    means = rng.normal(size=(4, 3))
    covs = rng.uniform(0.5, 2, size=(4, 3))
    edges = rng.dirichlet(np.ones(5), size=4)
    starts = rng.dirichlet(np.ones(4))
    ours = DenseHMM(
        [Normal(means[i], covs[i], covariance_type="diag") for i in range(4)],
        edges=edges[:, :4],
        starts=starts,
        ends=edges[:, 4],
    )
    upstream = UpDenseHMM(
        [
            UpNormal(
                means[i].tolist(), covs[i].tolist(), covariance_type="diag"
            )
            for i in range(4)
        ],
        edges=edges[:, :4].tolist(),
        starts=starts.tolist(),
        ends=edges[:, 4].tolist(),
    )
    return lambda: ours.predict_proba(X), lambda: upstream.predict_proba(X)


@case("BayesianNetwork.log_probability (50k x 8)")
def bayesian_logp():
    rng = np.random.default_rng(5)
    X = rng.integers(0, 2, size=(50_000, 8))
    root_prob = rng.dirichlet(np.ones(2))
    cpts = [rng.dirichlet(np.ones(2), size=2) for _ in range(7)]
    ours_distributions = [Categorical([root_prob])] + [
        ConditionalCategorical([table]) for table in cpts
    ]
    upstream_distributions = [UpCategorical([root_prob.tolist()])] + [
        UpConditional([table.tolist()]) for table in cpts
    ]
    ours_edges = [
        (ours_distributions[(i - 1) // 2], ours_distributions[i]) for i in range(1, 8)
    ]
    upstream_edges = [
        (upstream_distributions[(i - 1) // 2], upstream_distributions[i])
        for i in range(1, 8)
    ]
    ours = BayesianNetwork(ours_distributions, ours_edges)
    upstream = UpBayesianNetwork(upstream_distributions, upstream_edges)
    return lambda: ours.log_probability(X), lambda: upstream.log_probability(X)


def main():
    print(f"Machine: {processor()}; {platform.system()} {platform.release()}")
    print()
    print("| case | mojo-pomegranate | pomegranate | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, build in CASES:
        ours, upstream = build()
        ours()
        upstream()
        ours_time = timeit(ours)
        upstream_time = timeit(upstream)
        ratio = upstream_time / ours_time
        result = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {ours_time * 1e3:.1f} ms | "
            f"{upstream_time * 1e3:.1f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
