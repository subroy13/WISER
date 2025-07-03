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
            block_sums.append(M[:, int(index) : int(index + block_size)].sum(axis=1))
            index += block_size  # increase index by block size
        else:
            block_sums.append(
                M[:, int(index) :].sum(axis=1)
            )  # everything else is last block
            break

    block_sums = np.array(
        block_sums
    ).T  # transpose as above operation will make the samples in columns

    pivot_block_sums = block_sums[-1, :]  # take the last row (n/block_size, )
    Vstats = np.abs(block_sums[:-1, :]).max(axis=1)  # this is (B,)
    th = np.quantile(Vstats, q=(1 - alpha))  # find out (1-alpha) quantile

    Ktilde = []
    left_end = None
    right_end = None
    for i in range(pivot_block_sums.shape[0]):
        if pivot_block_sums[i] > th:
            # current block exceeds the threshold, check if it is continuing the current interval
            right_end = i
            if left_end is None:
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
        # convert left_end, right_end from block_level index to time level index
        left_index = left_end * block_size
        right_index = (right_end + 1) * block_size - 1  # inclusive
        mid = int((left_index + right_index) / 2)  # middle index

        # now tweak by +/- block_size in both direction, without crossover at mid
        left_index_start = int(max(0, left_index - block_size))
        left_index_end = int(min(left_index + block_size, mid))
        right_index_start = int(max(mid, right_index - block_size))
        right_index_end = int(min(right_index + block_size, n))

        max_sum = -np.inf
        max_interval = None
        for i in range(left_index_start, left_index_end):
            for j in range(right_index_start, right_index_end):
                Mijsum = (Msum[j] - Msum[i - 1]) if i >= 1 else Msum[j]
                Mijsum = Mijsum / max(j - i, 1)  # avoid dividing by 0  (for averaging)
                if Mijsum > max_sum:
                    max_interval = (i, j)  # track the interval with max sum
                    max_sum = Mijsum
        intervals.append(max_interval)
    return intervals
