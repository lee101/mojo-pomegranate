"""Compute kernels exported to the NumPy/ctypes API.

All arrays are caller-owned, contiguous, row-major buffers. The ABI receives
addresses as Int because an exported function with an inferred pointer origin
would be parametric.
"""

from max.algorithm import parallelize
from max.gpu.host import DeviceContext
from max.gpu import global_idx
from std.math import exp, log
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime PARALLEL_HMM_WORK = 200_000
comptime PARALLEL_MIXTURE_WORK = 500_000


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

    def compute_row(r: Int) {imm}:
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

    if n > 1 and n * k >= PARALLEL_MIXTURE_WORK:
        parallelize(compute_row, n, min(n, 16))
    else:
        for r in range(n):
            compute_row(r)


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

    def compute_row(r: Int) {imm}:
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

    if n > 1 and n * k >= PARALLEL_MIXTURE_WORK:
        parallelize(compute_row, n, min(n, 16))
    else:
        for r in range(n):
            compute_row(r)


def mixture_gpu_kernel(
    values: FPtr,
    logps: FPtr,
    n: Int64,
    k: Int64,
    probability: Int64,
):
    var r = Int64(global_idx.x)
    if r >= n:
        return
    var base = Int(r * k)
    var largest = values[base]
    for c in range(1, Int(k)):
        if values[base + c] > largest:
            largest = values[base + c]
    var total = 0.0
    for c in range(Int(k)):
        total += exp(values[base + c] - largest)
    var logp = largest + log(total)
    logps[Int(r)] = logp
    for c in range(Int(k)):
        var value = values[base + c] - logp
        values[base + c] = exp(value) if probability != 0 else value


@export("mp_mixture_gpu")
def mp_mixture_gpu(
    emissions_addr: Int,
    logps_addr: Int,
    n: Int,
    k: Int,
    probability: Int,
) abi("C") -> Int:
    if n <= 0 or k <= 0:
        return 0
    try:
        with DeviceContext() as ctx:
            var memory = ctx.get_memory_info()
            var elements = n * k + n
            var allocation_bytes = UInt(elements) * UInt(8)
            if memory[0] < UInt(4000 * 1024 * 1024):
                return 0
            if allocation_bytes >= UInt(2 * 1024 * 1024 * 1024):
                return 0
            var device_values = ctx.enqueue_create_buffer[DType.float64](n * k)
            var device_logps = ctx.enqueue_create_buffer[DType.float64](n)
            ctx.enqueue_copy(device_values, fp(emissions_addr))
            comptime block_size = 256
            ctx.enqueue_function[mixture_gpu_kernel](
                device_values,
                device_logps,
                Int64(n),
                Int64(k),
                Int64(probability),
                grid_dim=(n + block_size - 1) // block_size,
                block_dim=block_size,
            )
            ctx.enqueue_copy(fp(emissions_addr), device_values)
            ctx.enqueue_copy(fp(logps_addr), device_logps)
            ctx.synchronize()
        return 1
    except:
        return 0


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
    def compute_component(c: Int) {imm}:
        counts[c] = 0.0
        var j = 0
        while j + W <= d:
            sums.store(c * d + j, SIMD[DType.float64, W](0.0))
            sumsq.store(c * d + j, SIMD[DType.float64, W](0.0))
            j += W
        while j < d:
            sums[c * d + j] = 0.0
            sumsq[c * d + j] = 0.0
            j += 1
        var sum_vec = SIMD[DType.float64, W](0.0)
        var sumsq_vec = SIMD[DType.float64, W](0.0)
        for r in range(n):
            var weight = exp(weights[r * k + c])
            counts[c] += weight
            j = 0
            if d >= W:
                var values = x.load[width=W](r * d + j)
                sum_vec += weight * values
                sumsq_vec += weight * values * values
                j += W
            while j + W <= d:
                var values = x.load[width=W](r * d + j)
                sums.store(c * d + j, sums.load[width=W](c * d + j) + weight * values)
                sumsq.store(c * d + j, sumsq.load[width=W](c * d + j) + weight * values * values)
                j += W
            while j < d:
                var value = x[r * d + j]
                sums[c * d + j] += weight * value
                sumsq[c * d + j] += weight * value * value
                j += 1
        if d >= W:
            sums.store(c * d, sum_vec)
            sumsq.store(c * d, sumsq_vec)

    for c in range(k):
        compute_component(c)


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

    def compute_sequence(b: Int) {imm}:
        var base = b * length * k
        for c in range(k):
            alpha[base + c] = starts[c] + emissions[base + c]
        for t in range(1, length):
            var prev = base + (t - 1) * k
            var current = base + t * k
            var dst = 0
            while dst + W <= k:
                var largest_vec = alpha[prev] + edges.load[width=W](dst)
                for src in range(1, k):
                    largest_vec = max(
                        largest_vec,
                        alpha[prev + src] + edges.load[width=W](src * k + dst),
                    )
                var unreachable = largest_vec.lt(-1.0e300)
                var stable_largest = unreachable.select(
                    SIMD[DType.float64, W](0.0), largest_vec
                )
                var total_vec = SIMD[DType.float64, W](0.0)
                for src in range(k):
                    total_vec += exp(
                        alpha[prev + src] +
                        edges.load[width=W](src * k + dst) - stable_largest
                    )
                var result_vec = (
                    emissions.load[width=W](current + dst) +
                    stable_largest + log(total_vec)
                )
                alpha.store(
                    current + dst,
                    unreachable.select(largest_vec, result_vec),
                )
                dst += W
            while dst < k:
                var largest = alpha[prev] + edges[dst]
                for src in range(1, k):
                    var candidate = alpha[prev + src] + edges[src * k + dst]
                    if candidate > largest:
                        largest = candidate
                if largest < -1.0e300:
                    alpha[current + dst] = largest
                    dst += 1
                    continue
                var total = 0.0
                for src in range(k):
                    total += exp(alpha[prev + src] + edges[src * k + dst] - largest)
                alpha[current + dst] = emissions[current + dst] + largest + log(total)
                dst += 1

    if batch > 1 and batch * length * k * k >= PARALLEL_HMM_WORK:
        parallelize(compute_sequence, batch, min(batch, 16))
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

    def compute_sequence(b: Int) {imm}:
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

    if batch > 1 and batch * length * k * k >= PARALLEL_HMM_WORK:
        parallelize(compute_sequence, batch, min(batch, 16))
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

    def finish_sequence(b: Int) {imm}:
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

    if batch > 1 and batch * length * k * k >= PARALLEL_HMM_WORK:
        parallelize(finish_sequence, batch, min(batch, 16))
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
