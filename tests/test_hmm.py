import numpy as np
import pytest

from pomegranate.distributions import Categorical as UpCategorical
from pomegranate.distributions import Normal as UpNormal
from pomegranate.hmm import DenseHMM as UpDenseHMM
from pomegranate.hmm import SparseHMM as UpSparseHMM

from mojopomegranate.distributions import Categorical, Normal
from mojopomegranate.hmm import DenseHMM, SparseHMM


def numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


@pytest.fixture
def categorical_models():
    ours_distributions = [
        Categorical([[0.7, 0.3]]),
        Categorical([[0.2, 0.8]]),
    ]
    upstream_distributions = [
        UpCategorical([[0.7, 0.3]]),
        UpCategorical([[0.2, 0.8]]),
    ]
    kwargs = dict(
        edges=[[0.8, 0.1], [0.2, 0.7]],
        starts=[0.6, 0.4],
        ends=[0.1, 0.1],
    )
    return (
        DenseHMM(ours_distributions, **kwargs),
        UpDenseHMM(upstream_distributions, **kwargs),
    )


@pytest.fixture
def sequences():
    return np.array(
        [
            [[0], [0], [1], [1], [1]],
            [[1], [0], [0], [1], [0]],
            [[1], [1], [1], [0], [0]],
        ]
    )


def test_forward_parity(categorical_models, sequences):
    ours, upstream = categorical_models
    assert np.allclose(ours.forward(sequences), numpy(upstream.forward(sequences)), atol=2e-6)


def test_backward_parity(categorical_models, sequences):
    ours, upstream = categorical_models
    assert np.allclose(
        ours.backward(sequences), numpy(upstream.backward(sequences)), atol=2e-6
    )


def test_forward_backward_parity(categorical_models, sequences):
    ours, upstream = categorical_models
    for ours_value, upstream_value in zip(
        ours.forward_backward(sequences), upstream.forward_backward(sequences)
    ):
        assert np.allclose(ours_value, numpy(upstream_value), atol=3e-6)


@pytest.mark.parametrize("batch", [2, 128])
def test_hmm_parallel_threshold_and_simd_tail(batch):
    rng = np.random.default_rng(11)
    k, length = 5, 32
    transition_rows = rng.dirichlet(np.ones(k + 1), size=k)
    model = DenseHMM(
        [Categorical([[0.5, 0.5]]) for _ in range(k)],
        edges=transition_rows[:, :k],
        starts=rng.dirichlet(np.ones(k)),
        ends=transition_rows[:, k],
    )
    emissions = rng.normal(size=(batch, length, k))
    actual = model.forward(emissions=emissions)
    expected = np.empty_like(actual)
    expected[:, 0] = model.starts + emissions[:, 0]
    for b in range(batch):
        for t in range(1, length):
            candidates = expected[b, t - 1, :, None] + model.edges
            largest = candidates.max(axis=0)
            expected[b, t] = (
                emissions[b, t]
                + largest
                + np.log(np.exp(candidates - largest).sum(axis=0))
            )
    assert np.allclose(actual, expected, atol=2e-12)
    beta = np.empty_like(emissions)
    beta[:, -1] = model.ends
    for b in range(batch):
        for t in range(length - 2, -1, -1):
            candidates = (
                model.edges + emissions[b, t + 1][None] + beta[b, t + 1][None]
            )
            largest = candidates.max(axis=1)
            beta[b, t] = largest + np.log(
                np.exp(candidates - largest[:, None]).sum(axis=1)
            )
    assert np.allclose(model.backward(emissions=emissions), beta, atol=5e-9)
    last = expected[:, -1] + model.ends
    largest = last.max(axis=1)
    logps = largest + np.log(np.exp(last - largest[:, None]).sum(axis=1))
    _, post, _, _, actual_logps = model.forward_backward(emissions=emissions)
    assert np.allclose(actual_logps, logps, atol=2e-12)
    assert np.allclose(post, expected + beta - logps[:, None, None], atol=5e-9)


def test_inference_api_parity(categorical_models, sequences):
    ours, upstream = categorical_models
    assert np.allclose(
        ours.log_probability(sequences),
        numpy(upstream.log_probability(sequences)),
        atol=2e-6,
    )
    assert np.allclose(
        ours.predict_log_proba(sequences),
        numpy(upstream.predict_log_proba(sequences)),
        atol=2e-6,
    )
    assert np.allclose(
        ours.predict_proba(sequences),
        numpy(upstream.predict_proba(sequences)),
        atol=2e-6,
    )
    assert np.array_equal(ours.predict(sequences), numpy(upstream.predict(sequences)))


def test_hmm_observation_priors_parity(categorical_models, sequences):
    ours, upstream = categorical_models
    rng = np.random.default_rng(2)
    priors = rng.dirichlet([2, 2], size=sequences.shape[:2])
    assert np.allclose(
        ours.predict_proba(sequences, priors),
        numpy(upstream.predict_proba(sequences, priors)),
        atol=3e-6,
    )


def test_normal_hmm_parity():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(8, 20, 2))
    ours = DenseHMM(
        [
            Normal([-1.0, 0.0], [0.5, 1.0], covariance_type="diag"),
            Normal([1.0, 2.0], [1.2, 0.7], covariance_type="diag"),
        ],
        edges=[[0.75, 0.2], [0.1, 0.85]],
        starts=[0.4, 0.6],
        ends=[0.05, 0.05],
    )
    upstream = UpDenseHMM(
        [
            UpNormal([-1.0, 0.0], [0.5, 1.0], covariance_type="diag"),
            UpNormal([1.0, 2.0], [1.2, 0.7], covariance_type="diag"),
        ],
        edges=[[0.75, 0.2], [0.1, 0.85]],
        starts=[0.4, 0.6],
        ends=[0.05, 0.05],
    )
    assert np.allclose(
        ours.log_probability(X), numpy(upstream.log_probability(X)), atol=2e-5
    )
    assert np.allclose(
        ours.predict_proba(X), numpy(upstream.predict_proba(X)), atol=4e-6
    )


def test_sparse_hmm_parity(sequences):
    ours_dist = [Categorical([[0.7, 0.3]]), Categorical([[0.2, 0.8]])]
    up_dist = [UpCategorical([[0.7, 0.3]]), UpCategorical([[0.2, 0.8]])]
    ours_edges = [
        (ours_dist[0], ours_dist[0], 0.8),
        (ours_dist[0], ours_dist[1], 0.1),
        (ours_dist[1], ours_dist[0], 0.2),
        (ours_dist[1], ours_dist[1], 0.7),
    ]
    upstream_edges = [
        (up_dist[0], up_dist[0], 0.8),
        (up_dist[0], up_dist[1], 0.1),
        (up_dist[1], up_dist[0], 0.2),
        (up_dist[1], up_dist[1], 0.7),
    ]
    ours = SparseHMM(
        ours_dist, ours_edges, starts=[0.6, 0.4], ends=[0.1, 0.1]
    )
    upstream = UpSparseHMM(
        up_dist, upstream_edges, starts=[0.6, 0.4], ends=[0.1, 0.1]
    )
    assert np.allclose(
        ours.predict_proba(sequences),
        numpy(upstream.predict_proba(sequences)),
        atol=3e-6,
    )
    dense = DenseHMM(
        ours_dist,
        edges=[[0.8, 0.1], [0.2, 0.7]],
        starts=[0.6, 0.4],
        ends=[0.1, 0.1],
    )
    assert np.allclose(ours.forward(sequences), dense.forward(sequences))
    assert np.allclose(ours.backward(sequences), dense.backward(sequences))
    assert np.allclose(ours.log_probability(sequences), dense.log_probability(sequences))
    assert np.array_equal(ours.viterbi(sequences), dense.viterbi(sequences))


def test_baum_welch_one_iteration_parity(sequences):
    ours = DenseHMM(
        [Categorical([[0.7, 0.3]]), Categorical([[0.2, 0.8]])],
        edges=[[0.8, 0.1], [0.2, 0.7]],
        starts=[0.6, 0.4],
        ends=[0.1, 0.1],
        max_iter=1,
        tol=0,
    )
    upstream = UpDenseHMM(
        [UpCategorical([[0.7, 0.3]]), UpCategorical([[0.2, 0.8]])],
        edges=[[0.8, 0.1], [0.2, 0.7]],
        starts=[0.6, 0.4],
        ends=[0.1, 0.1],
        max_iter=1,
        tol=0,
    )
    ours.fit(sequences)
    upstream.fit(sequences)
    assert np.allclose(ours.edges, numpy(upstream.edges), atol=3e-6)
    assert np.allclose(ours.starts, numpy(upstream.starts), atol=3e-6)
    assert np.allclose(ours.ends, numpy(upstream.ends), atol=3e-6)
    for ours_dist, up_dist in zip(ours.distributions, upstream.distributions):
        assert np.allclose(ours_dist.probs, numpy(up_dist.probs), atol=3e-6)


def test_viterbi_published_dynamic_programming_reference(categorical_models, sequences):
    model, _ = categorical_models
    emissions = model._emission_matrix(sequences)
    paths = model.viterbi(emissions=emissions)
    reference = np.empty_like(paths)
    for batch in range(len(sequences)):
        scores = model.starts + emissions[batch, 0]
        traceback = np.empty((sequences.shape[1], model.k), dtype=int)
        for t in range(1, sequences.shape[1]):
            candidates = scores[:, None] + model.edges
            traceback[t] = candidates.argmax(axis=0)
            scores = candidates.max(axis=0) + emissions[batch, t]
        state = int(np.argmax(scores + model.ends))
        reference[batch, -1] = state
        for t in range(sequences.shape[1] - 1, 0, -1):
            state = traceback[t, state]
            reference[batch, t - 1] = state
    assert np.array_equal(paths, reference)


def test_programmatic_model_construction(sequences):
    distributions = [Categorical([[0.7, 0.3]]), Categorical([[0.2, 0.8]])]
    model = DenseHMM()
    model.add_distributions(distributions)
    model.add_edge(model.start, distributions[0], 0.6)
    model.add_edge(model.start, distributions[1], 0.4)
    model.add_edge(distributions[0], distributions[0], 0.8)
    model.add_edge(distributions[0], distributions[1], 0.1)
    model.add_edge(distributions[0], model.end, 0.1)
    model.add_edge(distributions[1], distributions[0], 0.2)
    model.add_edge(distributions[1], distributions[1], 0.7)
    model.add_edge(distributions[1], model.end, 0.1)
    direct = DenseHMM(
        distributions,
        edges=[[0.8, 0.1], [0.2, 0.7]],
        starts=[0.6, 0.4],
        ends=[0.1, 0.1],
    )
    assert np.allclose(model.log_probability(sequences), direct.log_probability(sequences))


def test_hmm_sampling_shapes():
    model = DenseHMM(
        [Categorical([[0.8, 0.2]]), Categorical([[0.3, 0.7]])],
        edges=[[0.7, 0.2], [0.2, 0.7]],
        starts=[0.5, 0.5],
        ends=[0.1, 0.1],
        sample_length=6,
        random_state=9,
        return_sample_paths=True,
    )
    samples, paths = model.sample(4)
    assert len(samples) == len(paths) == 4
    assert all(1 <= len(sample) <= 6 and sample.shape[1] == 1 for sample in samples)
    assert all(len(sample) == len(path) for sample, path in zip(samples, paths))
