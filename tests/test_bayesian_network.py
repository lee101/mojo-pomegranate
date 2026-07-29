import numpy as np

from pomegranate.bayesian_network import BayesianNetwork as UpBayesianNetwork
from pomegranate.distributions import Categorical as UpCategorical
from pomegranate.distributions import ConditionalCategorical as UpConditional

from mojopomegranate.bayesian_network import BayesianNetwork
from mojopomegranate.distributions import Categorical, ConditionalCategorical


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def networks():
    rain = Categorical([[0.8, 0.2]])
    sprinkler = ConditionalCategorical(
        [np.array([[0.6, 0.4], [0.99, 0.01]])]
    )
    wet = ConditionalCategorical(
        [
            np.array(
                [
                    [[0.99, 0.01], [0.1, 0.9]],
                    [[0.2, 0.8], [0.01, 0.99]],
                ]
            )
        ]
    )
    ours = BayesianNetwork(
        [rain, sprinkler, wet],
        [(rain, sprinkler), (rain, wet), (sprinkler, wet)],
    )

    up_rain = UpCategorical([[0.8, 0.2]])
    up_sprinkler = UpConditional(
        [np.array([[0.6, 0.4], [0.99, 0.01]])]
    )
    up_wet = UpConditional(
        [
            np.array(
                [
                    [[0.99, 0.01], [0.1, 0.9]],
                    [[0.2, 0.8], [0.01, 0.99]],
                ]
            )
        ]
    )
    upstream = UpBayesianNetwork(
        [up_rain, up_sprinkler, up_wet],
        [(up_rain, up_sprinkler), (up_rain, up_wet), (up_sprinkler, up_wet)],
    )
    return ours, upstream


def test_bayesian_network_log_probability_parity():
    ours, upstream = networks()
    X = np.array(list(np.ndindex(2, 2, 2)))
    assert np.allclose(
        ours.log_probability(X), numpy(upstream.log_probability(X)), atol=2e-6
    )
    assert np.allclose(ours.probability(X), numpy(upstream.probability(X)), atol=2e-6)
    np.testing.assert_allclose(ours.probability(X).sum(), 1.0)


def test_exact_missing_value_inference():
    model, _ = networks()
    X = np.array([[0, -1, 1], [-1, 1, 1], [1, 0, -1]])
    marginals = model.predict_proba(X)
    assert np.allclose(marginals[0][0], [1, 0])
    assert np.allclose(marginals[1][1], [0, 1])
    assert np.allclose(marginals[2][2], [0.2, 0.8])
    assert all(np.allclose(values.sum(axis=1), 1) for values in marginals)
    completed = model.predict(X)
    assert np.array_equal(completed[:, [0, 1]][[0, 1], [0, 1]], [0, 1])
    assert np.all(completed >= 0)


def test_structure_fit_matches_empirical_cpts():
    rng = np.random.default_rng(7)
    root = rng.binomial(1, 0.3, size=5000)
    child_probability = np.where(root == 0, 0.1, 0.8)
    child = rng.binomial(1, child_probability)
    X = np.column_stack([root, child])
    model = BayesianNetwork(structure=((), (0,)), pseudocount=0.5).fit(X)
    root_prob = model.distributions[0].probs[0, 1]
    child_prob = model.distributions[1].probs[0][:, 1]
    np.testing.assert_allclose(
        root_prob, (root.sum() + 0.5) / (len(root) + 1), atol=1e-12
    )
    expected = np.array(
        [
            ((child[root == value]).sum() + 0.5) / ((root == value).sum() + 1)
            for value in (0, 1)
        ]
    )
    assert np.allclose(child_prob, expected)


def test_fitted_network_log_probability_parity():
    rng = np.random.default_rng(8)
    X = rng.integers(0, 2, size=(1000, 3))
    ours = BayesianNetwork(structure=((), (0,), (0, 1))).fit(X)
    upstream = UpBayesianNetwork(structure=((), (0,), (0, 1))).fit(X)
    assert np.allclose(
        ours.log_probability(X[:100]),
        numpy(upstream.log_probability(X[:100])),
        atol=2e-6,
    )


def test_bayesian_network_ancestral_sample():
    model, _ = networks()
    samples = model.sample(2000)
    assert samples.shape == (2000, 3)
    assert np.all((samples == 0) | (samples == 1))
    np.testing.assert_allclose(samples[:, 0].mean(), 0.2, atol=0.04)
