import numpy as np


####################
# Watermarking Detection and finding the LLM intervals
def get_wm_gumbel_interval(
    pivot_stats: np.ndarray,  # 1D array of pivot statistics
    B=1000,  # number of bootstrap samples
    alpha=0.05,  # significance level
):
    assert pivot_stats.ndim == 1, "Pivot statistic should be a 1D array"
    n = pivot_stats.shape[0]
    block_size = np.ceil(n**0.5)

    Bsamples = np.random.standard_exponential((B, n))
    M = np.vstack((Bsamples, pivot_stats))  # (B+1) x n

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

    # find out (1-alpha) quantile
    th = np.quantile(Vstats, q=100 * (1 - alpha))
    selected_block_indices = np.where(pivot_block_sums > th)[0]  # (k, )

    if selected_block_indices.shape[0] == 0:
        # no intervals detected
        return []


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
