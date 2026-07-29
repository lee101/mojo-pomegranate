import numpy as np
import pytest

from pomegranate.distributions import Categorical as UpCategorical
from pomegranate.distributions import ConditionalCategorical as UpConditional
from pomegranate.distributions import Normal as UpNormal

from mojopomegranate.distributions import Categorical, ConditionalCategorical, Normal


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


@pytest.fixture
def continuous():
    return np.random.default_rng(4).normal(size=(200, 3))


def test_normal_diag_log_probability_parity(continuous):
    ours = Normal([0.5, -1.0, 2.0], [0.7, 1.5, 2.2], covariance_type="diag")
    upstream = UpNormal([0.5, -1.0, 2.0], [0.7, 1.5, 2.2], covariance_type="diag")
    assert np.allclose(
        ours.log_probability(continuous),
        numpy(upstream.log_probability(continuous)),
        atol=2e-6,
    )


def test_normal_diag_simd_tail_parity():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(257, 9))
    means = rng.normal(size=9)
    covs = rng.uniform(0.5, 2.0, size=9)
    ours = Normal(means, covs, covariance_type="diag")
    upstream = UpNormal(means, covs, covariance_type="diag")
    assert np.allclose(
        ours.log_probability(X),
        numpy(upstream.log_probability(X)),
        atol=2e-6,
    )


def test_normal_full_log_probability_parity(continuous):
    cov = [[1.2, 0.2, 0.1], [0.2, 1.7, -0.1], [0.1, -0.1, 0.9]]
    ours = Normal([0.5, -1.0, 2.0], cov, covariance_type="full")
    upstream = UpNormal([0.5, -1.0, 2.0], cov, covariance_type="full")
    assert np.allclose(
        ours.log_probability(continuous),
        numpy(upstream.log_probability(continuous)),
        atol=3e-6,
    )


@pytest.mark.parametrize("covariance_type", ["diag", "full"])
def test_normal_fit_parity(continuous, covariance_type):
    data = continuous.astype(np.float32)
    ours = Normal(covariance_type=covariance_type).fit(data)
    upstream = UpNormal(covariance_type=covariance_type).fit(data)
    assert np.allclose(ours.means, numpy(upstream.means), atol=1e-6)
    assert np.allclose(ours.covs, numpy(upstream.covs), atol=1e-6)


def test_normal_sphere_fit_numpy_mle_reference(continuous):
    ours = Normal(covariance_type="sphere").fit(continuous)
    expected_mean = continuous.mean(axis=0)
    expected_variance = np.mean((continuous - expected_mean) ** 2)
    assert np.allclose(ours.means, expected_mean)
    assert np.allclose(ours.covs, [expected_variance])


def test_categorical_log_probability_parity():
    probs = [[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]]
    X = np.array([[0, 0], [2, 1], [1, 2], [2, 0]])
    ours = Categorical(probs)
    upstream = UpCategorical(probs)
    assert np.allclose(ours.log_probability(X), numpy(upstream.log_probability(X)))


def test_categorical_weighted_fit_parity():
    X = np.array([[0, 1], [0, 2], [1, 1], [2, 0], [2, 0]])
    weights = np.array([1.0, 0.5, 2.0, 3.0, 1.5])
    ours = Categorical(n_categories=3, pseudocount=0.25).fit(X, weights)
    upstream = UpCategorical(n_categories=3, pseudocount=0.25).fit(X, weights)
    assert np.allclose(ours.probs, numpy(upstream.probs), atol=1e-6)


def test_categorical_rejects_out_of_alphabet_values():
    distribution = Categorical([[0.4, 0.6]])
    with pytest.raises(ValueError, match="contiguous integers"):
        distribution.log_probability([[2]])


@pytest.mark.parametrize("value", [[1.5], [np.nan], [np.inf]])
def test_categorical_rejects_non_integer_values(value):
    distribution = Categorical([[0.5, 0.5]])
    with pytest.raises(ValueError, match="finite integer"):
        distribution.log_probability(value)


def test_fit_rejects_invalid_weights(continuous):
    with pytest.raises(ValueError, match="match X"):
        Normal(covariance_type="diag").fit(continuous, np.ones(len(continuous) - 1))
    with pytest.raises(ValueError, match="positive total"):
        Categorical(n_categories=2).fit([[0], [1]], [0, 0])


def test_conditional_categorical_log_probability_parity():
    probs = [np.array([[0.9, 0.1], [0.25, 0.75]])]
    X = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
    ours = ConditionalCategorical(probs)
    upstream = UpConditional(probs)
    assert np.allclose(ours.log_probability(X), numpy(upstream.log_probability(X)))
