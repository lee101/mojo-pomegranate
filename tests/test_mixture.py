import numpy as np
import pytest

from pomegranate.distributions import Categorical as UpCategorical
from pomegranate.distributions import Normal as UpNormal
from pomegranate.gmm import GeneralMixtureModel as UpGMM

from mojopomegranate.distributions import Categorical, Normal
from mojopomegranate.gmm import GeneralMixtureModel


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


def test_mixture_sampling_and_boundary_validation():
    model, _ = gaussian_models()
    assert model.sample(11).shape == (11, 2)
    with pytest.raises(ValueError):
        model.predict_proba(np.empty((0, 2)))
    with pytest.raises(ValueError):
        GeneralMixtureModel([])
