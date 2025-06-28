from typing import List, Tuple
import numpy as np


####################
# Watermarking Detection and finding the LLM intervals
def get_wm_gumbel_interval(
    pivot_stats: np.ndarray,  # 1D array of pivot statistics
    B=1000,  # number of bootstrap samples
    alpha=0.05,  # significance level
) -> List[Tuple[float, float]]:
    assert pivot_stats.ndim == 1, "Pivot statistic should be a 1D array"
    n = pivot_stats.shape[0]
    block_size = np.ceil(n**0.5)

    Bsamples = np.random.standard_normal((B, n))
    M = np.vstack((Bsamples, pivot_stats - 1))  # (B+1) x n

    block_sums = []
    index = 0
    while index < n:
        if index + block_size <= n:
            block_sums.append(M[:, index : (index + block_size)].sum(axis=1))
            index += block_size  # increase index by block size
        else:
            block_sums.append(M[:, index:].sum(axis=1))  # everything else is last block
            break

    block_sums = np.array(
        block_sums
    ).T  # transpose as above operation will make the samples in columns

    pivot_block_sums = block_sums[-1, :]  # take the last row (n/block_size, )
    Vstats = np.abs(block_sums[:-1, :]).max(axis=1)  # this is (B,)
    th = np.quantile(Vstats, q=100 * (1 - alpha))  # find out (1-alpha) quantile

    Ktilde = []
    left_end = None
    right_end = None
    for i in range(pivot_block_sums.shape[0]):
        if pivot_block_sums[i] > th:
            # current block exceeds the threshold, check if it is continuing the current interval
            right_end = i
            if left_end is not None:
                left_end = i
        else:
            # current block does not exceed the threshold, so switch over to a new interval
            if left_end is not None:
                Ktilde.append((left_end, right_end))  # add existing block
                left_end = None  # reset
                right_end = None
    if left_end is not None:
        # handle the case where the last block is also over threshold
        Ktilde.append((left_end, right_end))

    # Now we have the major blocks detected, for each major blocks, we now refine
    # a useful trick is to store cumulative sums
    # so M[a:b].sum() = Msum[b] - Msum[a-1]
    Msum = M[-1, :].cumsum()
    intervals = []
    for left_end, right_end in Ktilde:
        mid = (left_end + right_end) / 2
        left_index_start = max(0, (left_end - 1) * block_size)
        left_index_end = np.round(min(left_end + 1, mid) * block_size)
        right_index_start = np.round(max(mid, right_end - 1) * block_size)
        right_index_end = min(n, (right_end + 1) * block_size)

        max_sum = -np.inf
        max_interval = None
        for i in range(left_index_start, left_index_end):
            for j in range(right_index_start, right_index_end):
                Mijsum = (Msum[j] - Msum[i - 1]) if i >= 1 else Msum[j]
                if Mijsum > max_sum:
                    max_interval = (i, j)  # track the interval with max sum
                    max_sum = Mijsum
        intervals.append(max_interval)
    return intervals


##################
# X1, .... X100
# S1, S2, ... S10 -> block sums

# S2, S3, S4, S6, S7 => select
# K(tilde) = {S2.start - S4.end, S6.start - S7.end}

# (S2.start - S4.end)
# (20 - 50)
# => Expand
# Left -> 10 - 30  and Right - 40 - 60

# --------
# O(n) => first level
# O(n * # blocks)
