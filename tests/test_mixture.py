import numpy as np
import pytest
import mojopomegranate.gmm as gmm_module

from pomegranate.distributions import Categorical as UpCategorical
from pomegranate.distributions import Normal as UpNormal
from pomegranate.gmm import GeneralMixtureModel as UpGMM

from mojopomegranate.distributions import Categorical, Normal
from mojopomegranate.gmm import GeneralMixtureModel
from mojopomegranate._lib import addr, lib


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def gaussian_models(max_iter=1000, tol=0.1):
    ours = GeneralMixtureModel(
        [
            Normal([-1.0, 0.0], [0.5, 1.0], covariance_type="diag"),
            Normal([2.0, 1.0], [1.0, 0.7], covariance_type="diag"),
        ],
        priors=[0.4, 0.6],
        max_iter=max_iter,
        tol=tol,
    )
    upstream = UpGMM(
        [
            UpNormal([-1.0, 0.0], [0.5, 1.0], covariance_type="diag"),
            UpNormal([2.0, 1.0], [1.0, 0.7], covariance_type="diag"),
        ],
        priors=[0.4, 0.6],
        max_iter=max_iter,
        tol=tol,
    )
    return ours, upstream


def test_gaussian_mixture_inference_parity():
    X = np.random.default_rng(0).normal(size=(400, 2))
    ours, upstream = gaussian_models()
    assert np.allclose(
        ours.log_probability(X), numpy(upstream.log_probability(X)), atol=2e-6
    )
    assert np.allclose(
        ours.predict_log_proba(X), numpy(upstream.predict_log_proba(X)), atol=2e-6
    )
    assert np.allclose(
        ours.predict_proba(X), numpy(upstream.predict_proba(X)), atol=2e-6
    )
    assert np.array_equal(ours.predict(X), numpy(upstream.predict(X)))
    assert np.allclose(ours.probability(X), numpy(upstream.probability(X)), atol=2e-6)


def test_mixture_observation_priors_parity():
    X = np.random.default_rng(1).normal(size=(100, 2))
    priors = np.random.default_rng(2).dirichlet([2, 3], size=len(X))
    ours, upstream = gaussian_models()
    assert np.allclose(
        ours.predict_proba(X, priors),
        numpy(upstream.predict_proba(X, priors)),
        atol=2e-6,
    )


def test_categorical_mixture_inference_parity():
    X = np.random.default_rng(3).integers(0, 3, size=(200, 2))
    ours = GeneralMixtureModel(
        [
            Categorical([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]),
            Categorical([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]),
        ],
        [0.25, 0.75],
    )
    upstream = UpGMM(
        [
            UpCategorical([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]),
            UpCategorical([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]),
        ],
        [0.25, 0.75],
    )
    assert np.allclose(
        ours.predict_proba(X), numpy(upstream.predict_proba(X)), atol=2e-6
    )


def test_gaussian_mixture_em_parity():
    rng = np.random.default_rng(4)
    X = np.vstack(
        [
            rng.normal([-2, 0], [0.5, 0.7], size=(500, 2)),
            rng.normal([2, 1], [0.8, 0.4], size=(600, 2)),
        ]
    )
    ours, upstream = gaussian_models(max_iter=40, tol=1e-6)
    ours.fit(X)
    upstream.fit(X)
    assert np.allclose(ours.priors, numpy(upstream.priors), atol=5e-5)
    for ours_dist, upstream_dist in zip(ours.distributions, upstream.distributions):
        assert np.allclose(ours_dist.means, numpy(upstream_dist.means), atol=5e-5)
        assert np.allclose(ours_dist.covs, numpy(upstream_dist.covs), atol=5e-5)
    assert np.allclose(
        ours.log_probability(X), numpy(upstream.log_probability(X)), atol=5e-4
    )


@pytest.mark.parametrize("n", [7, 10_001])
def test_weighted_stats_simd_tail(n):
    rng = np.random.default_rng(12)
    d, k = 5, 3
    X = rng.normal(size=(n, d))
    log_weights = np.log(rng.dirichlet(np.ones(k), size=n))
    counts = np.empty(k)
    sums = np.empty((k, d))
    sumsq = np.empty((k, d))
    lib().mp_weighted_stats(
        addr(X), addr(log_weights), addr(counts), addr(sums), addr(sumsq), n, d, k
    )
    weights = np.exp(log_weights)
    assert np.allclose(counts, weights.sum(axis=0), rtol=1e-11, atol=2e-9)
    assert np.allclose(sums, weights.T @ X, rtol=1e-11, atol=2e-9)
    assert np.allclose(sumsq, weights.T @ (X * X), rtol=1e-11, atol=2e-9)


@pytest.mark.parametrize("n", [7, 100_001])
@pytest.mark.parametrize(
    "symbol", ["mp_mixture_posteriors", "mp_mixture_probabilities"]
)
def test_mixture_normalization_simd_tail_and_parallel_threshold(n, symbol):
    emissions = np.random.default_rng(13).normal(size=(n, 5))
    result = emissions.copy()
    logps = np.empty(n)
    getattr(lib(), symbol)(addr(emissions), addr(result), addr(logps), n, 5)
    largest = emissions.max(axis=1)
    expected_logps = largest + np.log(
        np.exp(emissions - largest[:, None]).sum(axis=1)
    )
    expected = emissions - expected_logps[:, None]
    if symbol == "mp_mixture_probabilities":
        expected = np.exp(expected)
    assert np.allclose(logps, expected_logps, atol=2e-12)
    assert np.allclose(result, expected, atol=2e-12)


def test_mixture_gpu_or_cpu_fallback_and_device_validation(monkeypatch):
    X = np.random.default_rng(14).normal(size=(4096, 2))
    model, _ = gaussian_models()
    expected = model.predict_proba(X)
    assert np.allclose(model.predict_proba(X, device="gpu"), expected, atol=2e-12)

    real_lib = lib()

    class UnavailableGPU:
        def __getattr__(self, name):
            return getattr(real_lib, name)

        @staticmethod
        def mp_mixture_gpu(*args):
            return 0

    monkeypatch.setattr(gmm_module, "lib", lambda: UnavailableGPU())
    assert np.allclose(model.predict_proba(X, device="gpu"), expected, atol=2e-12)
    with pytest.raises(ValueError, match="device"):
        model.predict_proba(X, device="accelerator")


def test_mixture_sampling_and_boundary_validation():
    model, _ = gaussian_models()
    assert model.sample(11).shape == (11, 2)
    with pytest.raises(ValueError):
        model.predict_proba(np.empty((0, 2)))
    with pytest.raises(ValueError):
        GeneralMixtureModel([])
