# mojo-pomegranate

`mojo-pomegranate` is a standalone Mojo port of the compute-heavy parts of
[pomegranate](https://github.com/jmschrei/pomegranate): mixture-model EM,
hidden Markov dynamic programming, and categorical Bayesian-network
likelihoods. Its Python classes keep pomegranate's current names and method
signatures for the covered subset, while NumPy buffers call a single compiled
Mojo shared library through `ctypes`.

This is a focused port, not a reimplementation of every pomegranate model.
Parity tests run against the real `pomegranate 1.0.0` package in the pixi
environment. The test suite compares numerical results, posterior
probabilities, fitted parameters, and one full Baum-Welch update.

## Covered API

| Area | Covered |
| --- | --- |
| Distributions | `Normal` likelihood and fit (`diag`, `sphere`, `full`); `Categorical` likelihood and weighted fit; `ConditionalCategorical.log_probability` |
| Mixtures | `GeneralMixtureModel`: probability, log probability, posterior, prediction, sampling, and EM for diagonal/spherical Normal components |
| HMMs | `DenseHMM`: Normal and Categorical emissions, forward, backward, forward-backward, Viterbi, posterior prediction, sampling, and equal-length Baum-Welch fitting |
| Sparse HMM surface | `SparseHMM` accepts pomegranate edge triples and has the same inference API; the current kernel densifies the graph |
| Bayes nets | `BayesianNetwork`: explicit categorical DAGs, fixed-structure parameter fitting, joint probability, exact missing-value marginals, prediction, and ancestral sampling |

The following are not covered: arbitrary nested distributions, Torch
autograd/GPU tensors, variable-length HMM training, genuinely sparse HMM
kernels, full-covariance mixture EM, Bayesian-network structure learning,
loopy belief propagation, and incremental `summarize`/`from_summaries`
training. Bayesian missing-value inference enumerates the joint state space,
so it is exact but intended for small discrete networks. Missing values are
written as `-1` or `NaN` instead of requiring a Torch `MaskedTensor`.

Methods return NumPy arrays rather than Torch tensors. Common NumPy call sites
can change only their imports, but code depending on `torch.Tensor`, autograd,
or device methods must adapt. Continuous inputs are converted to C-contiguous
`float64`; categorical inputs must contain exact finite integers and are
converted to C-contiguous `int64`.

## Install

```bash
pixi install
pixi run build
pixi run test
```

The pinned pixi environment supplies Mojo, NumPy, pytest, and upstream
pomegranate. `pixi run build` writes `dist/libmojo-pomegranate.so`.

## Usage

This example is also representative of the HMM parity tests:

```python
import numpy as np

from mojopomegranate.distributions import Categorical
from mojopomegranate.hmm import DenseHMM

background = Categorical([[0.7, 0.3]])
signal = Categorical([[0.2, 0.8]])

model = DenseHMM(
    [background, signal],
    edges=[[0.8, 0.1], [0.2, 0.7]],
    starts=[0.6, 0.4],
    ends=[0.1, 0.1],
)

X = np.array([[[0], [0], [1], [1], [1]]])
print(model.log_probability(X))
print(model.predict_proba(X))
print(model.viterbi(X))
```

Run it inside the environment with `pixi run python example.py`.

The upstream module layout is mirrored:

```python
from mojopomegranate.gmm import GeneralMixtureModel
from mojopomegranate.hmm import DenseHMM, SparseHMM
from mojopomegranate.bayesian_network import BayesianNetwork
from mojopomegranate.distributions import Normal, Categorical
```

## Performance

Measured on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux 6.8.0-136-generic.
Each model is warmed first and the table reports the best of three runs from
`pixi run bench`. That task holds `/tmp/mojo-bench.lock`, so concurrent factory
jobs do not overlap these measurements.

| case | mojo-pomegranate | pomegranate | result |
| --- | ---: | ---: | ---: |
| Normal.log_probability (1M x 8) | 14.5 ms | 63.6 ms | 4.39x faster |
| GMM.predict_proba k=4 (300k x 8) | 28.6 ms | 45.5 ms | 1.59x faster |
| GMM.predict_proba GPU k=4 (300k x 8) | 18.5 ms | 43.5 ms | 2.35x faster |
| GMM.fit k=3, 8 EM steps (100k x 6) | 115.9 ms | 126.5 ms | 1.09x faster |
| DenseHMM.forward k=6 (1k x 300) | 28.7 ms | 37.9 ms | 1.32x faster |
| DenseHMM.predict_proba k=4 (500 x 300) | 53.7 ms | 75.0 ms | 1.40x faster |
| BayesianNetwork.log_probability (50k x 8) | 4.5 ms | 8127.8 ms | 1817.60x faster |

Large independent HMM sequences and mixture-normalization rows use thresholded
CPU parallelism; small calls stay serial to avoid launch overhead. HMM forward,
contiguous HMM reductions, mixture normalization, and weighted Normal statistics
use the host SIMD width with scalar remainder loops. Statistics keep the first
SIMD accumulator block in registers across rows. Mixture posterior output is
updated in place, and `predict_proba` writes probabilities directly instead of
allocating a second array for a NumPy exponentiation.

GMM normalization has an optional GPU path through
`predict_proba(X, device="gpu")`; the same device argument is accepted by the
other normalized GMM inference methods and `fit`. CPU remains the default. The
GPU wrapper checks that at least 4000 MiB is free, caps each call below 2 GiB,
and silently runs the CPU kernel if a context or allocation is unavailable.
The GPU benchmark above used about 12 MiB on an RTX 5090 and includes host-device
copies and synchronization. `pixi run bench` omits this row and reports the skip
when less than 4000 MiB is free. HMM recurrence and the streaming emission and
statistics kernels remain CPU-only because transfer and recurrence costs do not
justify GPU offload.

The Bayesian-network result is large because pomegranate 1.0.0 indexes each
conditional distribution in a Python loop over rows, while Mojo evaluates the
packed CPTs in one native call. The benchmark's upstream HMM and Bayes-network
constructors use the documented Python-list form and therefore pomegranate's
default float32 dtype; Mojo computes in float64 throughout.

## How it works

`src/kernels.mojo` is one compilation unit. It exports C-ABI entry points for
Normal and Categorical emission matrices, mixture normalization and weighted
statistics, HMM forward/backward/Viterbi and transition expectations, and
packed Bayesian-network likelihoods.

Python owns every input, result, and scratch allocation. Arrays are
C-contiguous and row-major:

```text
NumPy model API
    |
    | ctypes: buffer address + dimensions
    v
dist/libmojo-pomegranate.so
    |
    +-- mixture emissions and sufficient statistics
    +-- HMM log-space dynamic programming
    +-- packed categorical CPT traversal
```

Buffers cross the ABI as integer addresses and are reconstructed as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside Mojo. No Mojo allocation
crosses the FFI boundary, and one Python method call normally becomes one or
two native calls over the full batch.

## License

MIT
