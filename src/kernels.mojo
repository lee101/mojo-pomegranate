"""Compute kernels exported to the NumPy/ctypes API.

All arrays are caller-owned, contiguous, row-major buffers. The ABI receives
addresses as Int because an exported function with an inferred pointer origin
would be parametric.
"""

from std.algorithm import sync_parallelize
from std.math import exp, log
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime PARALLEL_MIN_WORK = 32768


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def logadd(a: Float64, b: Float64) -> Float64:
    if a < b:
        return b + log(1.0 + exp(a - b))
    return a + log(1.0 + exp(b - a))


@export("mp_normal_emissions")
def mp_normal_emissions(
    x_addr: Int,
    means_addr: Int,
    invvars_addr: Int,
    log_norms_addr: Int,
    log_priors_addr: Int,
    dst_addr: Int,
    n: Int,
    d: Int,
    k: Int,
) abi("C"):
    var x = fp(x_addr)
    var means = fp(means_addr)
    var invvars = fp(invvars_addr)
    var log_norms = fp(log_norms_addr)
    var log_priors = fp(log_priors_addr)
    var dst = fp(dst_addr)

    @always_inline
    @parameter
    def compute_row(r: Int):
        for c in range(k):
            var acc = SIMD[DType.float64, W](0.0)
            var j = 0
            while j + W <= d:
                var diff = x.load[width=W](r * d + j) - means.load[width=W](c * d + j)
                acc += diff * diff * invvars.load[width=W](c * d + j)
                j += W
            var total = acc.reduce_add()
            while j < d:
                var delta = x[r * d + j] - means[c * d + j]
                total += delta * delta * invvars[c * d + j]
                j += 1
            dst[r * k + c] = log_priors[c] + log_norms[c] - 0.5 * total

    for r in range(n):
        compute_row(r)


@export("mp_categorical_emissions")
def mp_categorical_emissions(
    x_addr: Int,
    log_probs_addr: Int,
    dst_addr: Int,
    n: Int,
    d: Int,
    k: Int,
    categories: Int,
) abi("C"):
    var x = ip(x_addr)
    var probs = fp(log_probs_addr)
    var dst = fp(dst_addr)
    for r in range(n):
        for c in range(k):
            var score = 0.0
            for j in range(d):
                var value = Int(x[r * d + j])
                score += probs[(c * d + j) * categories + value]
            dst[r * k + c] = score


@export("mp_mixture_posteriors")
def mp_mixture_posteriors(
    emissions_addr: Int,
    log_post_addr: Int,
    logps_addr: Int,
    n: Int,
    k: Int,
) abi("C"):
    var emissions = fp(emissions_addr)
    var post = fp(log_post_addr)
    var logps = fp(logps_addr)
    for r in range(n):
        var largest = emissions[r * k]
        for c in range(1, k):
            if emissions[r * k + c] > largest:
                largest = emissions[r * k + c]
        var total = 0.0
        var c = 0
        if k >= W:
            var acc = SIMD[DType.float64, W](0.0)
            while c + W <= k:
                acc += exp(emissions.load[width=W](r * k + c) - largest)
                c += W
            total = acc.reduce_add()
        while c < k:
            total += exp(emissions[r * k + c] - largest)
            c += 1
        var logp = largest + log(total)
        logps[r] = logp
        c = 0
        while c + W <= k:
            post.store(r * k + c, emissions.load[width=W](r * k + c) - logp)
            c += W
        while c < k:
            post[r * k + c] = emissions[r * k + c] - logp
            c += 1


@export("mp_mixture_probabilities")
def mp_mixture_probabilities(
    emissions_addr: Int,
    probabilities_addr: Int,
    logps_addr: Int,
    n: Int,
    k: Int,
) abi("C"):
    var emissions = fp(emissions_addr)
    var probabilities = fp(probabilities_addr)
    var logps = fp(logps_addr)
    for r in range(n):
        var largest = emissions[r * k]
        for c in range(1, k):
            if emissions[r * k + c] > largest:
                largest = emissions[r * k + c]
        var total = 0.0
        var c = 0
        if k >= W:
            var acc = SIMD[DType.float64, W](0.0)
            while c + W <= k:
                acc += exp(emissions.load[width=W](r * k + c) - largest)
                c += W
            total = acc.reduce_add()
        while c < k:
            total += exp(emissions[r * k + c] - largest)
            c += 1
        var logp = largest + log(total)
        logps[r] = logp
        c = 0
        while c + W <= k:
            probabilities.store(
                r * k + c,
                exp(emissions.load[width=W](r * k + c) - logp),
            )
            c += W
        while c < k:
            probabilities[r * k + c] = exp(emissions[r * k + c] - logp)
            c += 1


@export("mp_weighted_stats")
def mp_weighted_stats(
    x_addr: Int,
    log_weights_addr: Int,
    counts_addr: Int,
    sums_addr: Int,
    sumsq_addr: Int,
    n: Int,
    d: Int,
    k: Int,
) abi("C"):
    var x = fp(x_addr)
    var weights = fp(log_weights_addr)
    var counts = fp(counts_addr)
    var sums = fp(sums_addr)
    var sumsq = fp(sumsq_addr)
    for c in range(k):
        counts[c] = 0.0
    for i in range(k * d):
        sums[i] = 0.0
        sumsq[i] = 0.0
    for r in range(n):
        for c in range(k):
            var weight = exp(weights[r * k + c])
            counts[c] += weight
            for j in range(d):
                var value = x[r * d + j]
                sums[c * d + j] += weight * value
                sumsq[c * d + j] += weight * value * value


@export("mp_weighted_categorical_stats")
def mp_weighted_categorical_stats(
    x_addr: Int,
    log_weights_addr: Int,
    counts_addr: Int,
    n: Int,
    d: Int,
    k: Int,
    categories: Int,
) abi("C"):
    var x = ip(x_addr)
    var weights = fp(log_weights_addr)
    var counts = fp(counts_addr)
    for i in range(k * d * categories):
        counts[i] = 0.0
    for r in range(n):
        for c in range(k):
            var weight = exp(weights[r * k + c])
            for j in range(d):
                var value = Int(x[r * d + j])
                counts[(c * d + j) * categories + value] += weight


@export("mp_hmm_forward")
def mp_hmm_forward(
    emissions_addr: Int,
    edges_addr: Int,
    starts_addr: Int,
    alpha_addr: Int,
    batch: Int,
    length: Int,
    k: Int,
) abi("C"):
    var emissions = fp(emissions_addr)
    var edges = fp(edges_addr)
    var starts = fp(starts_addr)
    var alpha = fp(alpha_addr)

    @parameter
    def compute_sequence(b: Int):
        var base = b * length * k
        for c in range(k):
            alpha[base + c] = starts[c] + emissions[base + c]
        for t in range(1, length):
            for dst in range(k):
                var prev = base + (t - 1) * k
                var largest = alpha[prev] + edges[dst]
                for src in range(1, k):
                    var candidate = alpha[prev + src] + edges[src * k + dst]
                    if candidate > largest:
                        largest = candidate
                if largest < -1.0e300:
                    alpha[base + t * k + dst] = largest
                    continue
                var total = 0.0
                for src in range(k):
                    total += exp(alpha[prev + src] + edges[src * k + dst] - largest)
                alpha[base + t * k + dst] = emissions[base + t * k + dst] + largest + log(total)

    if batch * length * k * k >= PARALLEL_MIN_WORK:
        sync_parallelize[compute_sequence](batch)
    else:
        for b in range(batch):
            compute_sequence(b)


@export("mp_hmm_backward")
def mp_hmm_backward(
    emissions_addr: Int,
    edges_addr: Int,
    ends_addr: Int,
    beta_addr: Int,
    batch: Int,
    length: Int,
    k: Int,
) abi("C"):
    var emissions = fp(emissions_addr)
    var edges = fp(edges_addr)
    var ends = fp(ends_addr)
    var beta = fp(beta_addr)

    @parameter
    def compute_sequence(b: Int):
        var base = b * length * k
        for c in range(k):
            beta[base + (length - 1) * k + c] = ends[c]
        for step in range(length - 1):
            var t = length - 2 - step
            for src in range(k):
                var nxt = base + (t + 1) * k
                var largest = edges[src * k] + emissions[nxt] + beta[nxt]
                for dst in range(1, k):
                    var candidate = edges[src * k + dst] + emissions[nxt + dst] + beta[nxt + dst]
                    if candidate > largest:
                        largest = candidate
                if largest < -1.0e300:
                    beta[base + t * k + src] = largest
                    continue
                var acc = SIMD[DType.float64, W](0.0)
                var dst = 0
                while dst + W <= k:
                    acc += exp(
                        edges.load[width=W](src * k + dst) +
                        emissions.load[width=W](nxt + dst) +
                        beta.load[width=W](nxt + dst) - largest
                    )
                    dst += W
                var total = acc.reduce_add()
                while dst < k:
                    total += exp(edges[src * k + dst] + emissions[nxt + dst] + beta[nxt + dst] - largest)
                    dst += 1
                beta[base + t * k + src] = largest + log(total)

    if batch * length * k * k >= PARALLEL_MIN_WORK:
        sync_parallelize[compute_sequence](batch)
    else:
        for b in range(batch):
            compute_sequence(b)


@export("mp_hmm_finish")
def mp_hmm_finish(
    emissions_addr: Int,
    edges_addr: Int,
    starts_addr: Int,
    ends_addr: Int,
    alpha_addr: Int,
    beta_addr: Int,
    log_post_addr: Int,
    transitions_addr: Int,
    logps_addr: Int,
    batch: Int,
    length: Int,
    k: Int,
) abi("C"):
    mp_hmm_forward(emissions_addr, edges_addr, starts_addr, alpha_addr, batch, length, k)
    mp_hmm_backward(emissions_addr, edges_addr, ends_addr, beta_addr, batch, length, k)
    var emissions = fp(emissions_addr)
    var edges = fp(edges_addr)
    var ends = fp(ends_addr)
    var alpha = fp(alpha_addr)
    var beta = fp(beta_addr)
    var post = fp(log_post_addr)
    var transitions = fp(transitions_addr)
    var logps = fp(logps_addr)

    @parameter
    def finish_sequence(b: Int):
        var base = b * length * k
        var last = base + (length - 1) * k
        var largest = alpha[last] + ends[0]
        for c in range(1, k):
            var candidate = alpha[last + c] + ends[c]
            if candidate > largest:
                largest = candidate
        var total = 0.0
        for c in range(k):
            total += exp(alpha[last + c] + ends[c] - largest)
        var logp = largest + log(total)
        logps[b] = logp
        for t in range(length):
            var c = 0
            while c + W <= k:
                post.store(
                    base + t * k + c,
                    alpha.load[width=W](base + t * k + c) +
                    beta.load[width=W](base + t * k + c) - logp,
                )
                c += W
            while c < k:
                post[base + t * k + c] = alpha[base + t * k + c] + beta[base + t * k + c] - logp
                c += 1
        for i in range(k * k):
            transitions[b * k * k + i] = 0.0
        for t in range(length - 1):
            var current = base + t * k
            var nxt = current + k
            for src in range(k):
                var dst = 0
                while dst + W <= k:
                    var index = b * k * k + src * k + dst
                    transitions.store(
                        index,
                        transitions.load[width=W](index) + exp(
                            alpha[current + src] +
                            edges.load[width=W](src * k + dst) +
                            emissions.load[width=W](nxt + dst) +
                            beta.load[width=W](nxt + dst) - logp
                        ),
                    )
                    dst += W
                while dst < k:
                    transitions[b * k * k + src * k + dst] += exp(
                        alpha[current + src] + edges[src * k + dst] +
                        emissions[nxt + dst] + beta[nxt + dst] - logp
                    )
                    dst += 1

    if batch * length * k * k >= PARALLEL_MIN_WORK:
        sync_parallelize[finish_sequence](batch)
    else:
        for b in range(batch):
            finish_sequence(b)


@export("mp_hmm_viterbi")
def mp_hmm_viterbi(
    emissions_addr: Int,
    edges_addr: Int,
    starts_addr: Int,
    ends_addr: Int,
    scores_addr: Int,
    traceback_addr: Int,
    paths_addr: Int,
    batch: Int,
    length: Int,
    k: Int,
) abi("C"):
    var emissions = fp(emissions_addr)
    var edges = fp(edges_addr)
    var starts = fp(starts_addr)
    var ends = fp(ends_addr)
    var scores = fp(scores_addr)
    var traceback = ip(traceback_addr)
    var paths = ip(paths_addr)
    for b in range(batch):
        var base = b * length * k
        for c in range(k):
            scores[base + c] = starts[c] + emissions[base + c]
            traceback[base + c] = Int64(c)
        for t in range(1, length):
            for dst in range(k):
                var best = scores[base + (t - 1) * k] + edges[dst]
                var best_src = 0
                for src in range(1, k):
                    var candidate = scores[base + (t - 1) * k + src] + edges[src * k + dst]
                    if candidate > best:
                        best = candidate
                        best_src = src
                scores[base + t * k + dst] = best + emissions[base + t * k + dst]
                traceback[base + t * k + dst] = Int64(best_src)
        var state = 0
        var best_end = scores[base + (length - 1) * k] + ends[0]
        for c in range(1, k):
            var candidate = scores[base + (length - 1) * k + c] + ends[c]
            if candidate > best_end:
                best_end = candidate
                state = c
        paths[b * length + length - 1] = Int64(state)
        for step in range(length - 1):
            var t = length - 1 - step
            state = Int(traceback[base + t * k + state])
            paths[b * length + t - 1] = Int64(state)


@export("mp_bn_log_probability")
def mp_bn_log_probability(
    x_addr: Int,
    log_cpts_addr: Int,
    cpt_offsets_addr: Int,
    parents_addr: Int,
    parent_offsets_addr: Int,
    categories_addr: Int,
    dst_addr: Int,
    n: Int,
    d: Int,
) abi("C"):
    var x = ip(x_addr)
    var cpts = fp(log_cpts_addr)
    var cpt_offsets = ip(cpt_offsets_addr)
    var parents = ip(parents_addr)
    var parent_offsets = ip(parent_offsets_addr)
    var categories = ip(categories_addr)
    var dst = fp(dst_addr)
    for r in range(n):
        var score = 0.0
        for node in range(d):
            var index = 0
            for pidx in range(Int(parent_offsets[node]), Int(parent_offsets[node + 1])):
                var parent = Int(parents[pidx])
                index = index * Int(categories[parent]) + Int(x[r * d + parent])
            index = index * Int(categories[node]) + Int(x[r * d + node])
            score += cpts[Int(cpt_offsets[node]) + index]
        dst[r] = score
