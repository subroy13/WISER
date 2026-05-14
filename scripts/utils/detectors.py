from typing import List, Any, Union, Tuple
import math
import random
import warnings
from time import perf_counter
import numpy as np
from scipy.stats import ks_2samp
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import heapq
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import aligator_cpp.aligator as aligator_cpp  # type: ignore
import seedbs_cpp.seedbs as seedbs_cpp  # type: ignore

##################
# ALIGATOR
# Reference Code: https://github.com/XuandongZhao/llm-watermark-location
# Reference Paper: https://arxiv.org/pdf/2410.03600v2


# Pure python implementation
class Expert:
    def __init__(self, start, end):
        self.start = start  # expert's interval range
        self.end = end
        self.prediction = 0.0
        self.loss = 0.0
        self.count = 0
        self.weight = 0.0

    def __str__(self):
        return f"Expert for cover: {self.start}-{self.end}"


class Aligator:

    def init_experts(self, n: int):
        """
        Initialize expert pool for intervals I_k = [i*2^k, (i+1)*2^k - 1]
        """
        count = 0
        pool: List[List[Expert]] = []
        maxpow2 = int(np.floor(np.log2(n)))
        for k in range(maxpow2 + 1):
            # choices of k
            stop = ((n + 1) >> k) - 1
            if stop < 0:
                break
            elist = []
            for i in range(stop + 1):
                # TO check
                e = Expert(start=i * (2**k), end=(i + 1) * (2**k))
                elist.append(e)
                if k > 4:
                    count += 1  # only count if the interval length is >= 2^4 = 16
            pool.append(elist)
        return pool, count

    def get_awake_set(self, t: int, n: int):
        """
        Compute awake set indices for time t.
        """
        awake_set = []
        maxpow2 = int(np.floor(np.log2(n)))
        for k in range(maxpow2 + 1):
            i = t >> k
            if (((i + 1) << k) - 1 > n) or (k <= 4):
                awake_set.append(-1)
            else:
                awake_set.append(i)
        return awake_set

    def get_forecast(self, awake_set: list[int], pool: List[List[Expert]], pool_size: int, prev_pred: float):
        """
        Compute forecast from awake experts.
        Returns: output, normalizer
        """
        output = 0.0
        normalizer = 0.0
        for k, idx in enumerate(awake_set):
            if idx == -1:
                continue  # skip this expert
            i = idx
            if pool[k][i].weight == 0:
                pool[k][i].weight = 1.0 / pool_size
                pool[k][i].prediction = prev_pred  # isotonic smoothing
            output += pool[k][i].weight * pool[k][i].prediction
            normalizer += pool[k][i].weight
        if normalizer == 0:
            normalizer = 1

        return output / normalizer, normalizer

    def compute_losses(
        self, awake_set: list[int], pool: List[List[Expert]], y: float, B: float, n: int, sigma: float, delta: float
    ):
        """
        Compute losses for awake experts.
        Returns: losses (list)
        """
        norm = 2 * (B + sigma * np.sqrt(np.log(2 * n / delta))) ** 2
        losses = []
        for k, idx in enumerate(awake_set):
            if idx == -1:
                losses.append(-1)
            else:
                i = idx
                loss = (y - pool[k][i].prediction) ** 2 / norm
                losses.append(loss)
        return losses

    def update_weights_and_predictions(
        self, awake_set: list[int], pool: List[List[Expert]], losses: List[float], normalizer: float, y: float
    ):
        """
        Update weights and predictions of awake experts.
        """
        # compute denominator normalizer
        denom = 0.0
        for k, idx in enumerate(awake_set):
            if idx == -1:
                continue
            i = idx
            denom += pool[k][i].weight * np.exp(-losses[k])

        # update weights and predictions
        for k, idx in enumerate(awake_set):
            if idx == -1:
                continue
            i = idx
            pool[k][i].weight *= np.exp(-losses[k]) * normalizer / denom
            pool[k][i].prediction = ((pool[k][i].prediction * pool[k][i].count) + y) / (pool[k][i].count + 1)
            pool[k][i].count += 1
        return pool

    def run_aligator(
        self, n: int, y: Union[List[float], np.ndarray], index: List[int], sigma: float, B: float, delta: float
    ):
        """
        Main driver for ALIGATOR.
        y: list/np.array of true values
        index: list/np.array of indices to process
        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        prev_pred = 0.0

        estimates = np.zeros(n)
        pool, pool_size = self.init_experts(n)
        awake_set = []

        for t in range(n):
            idx = index[t]
            y_curr = y[idx]
            awake_set = self.get_awake_set(idx + 1, n)
            output, normalizer = self.get_forecast(awake_set, pool, pool_size, prev_pred)
            estimates[idx] = output
            losses = self.compute_losses(awake_set, pool, y_curr, B, n, sigma, delta)
            pool = self.update_weights_and_predictions(awake_set, pool, losses, normalizer, y_curr)
            prev_pred = y_curr

        return estimates


class AligatorDetector:

    def __init__(self, vocab_size, alpha=0.05, B=1000):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.prev_pred = 0
        self.B = B

    def detect(self, pivot: np.ndarray, null_distn):
        # calculate threshold empirically
        null_samples = null_distn((self.B,))
        threshold = np.quantile(null_samples, 1 - self.alpha)

        # Start timer
        start_time = perf_counter()

        n = pivot.shape[0]
        y = pivot.copy()
        step = int(n / 30)
        res = []

        # bidirectional circular detection
        for i in range(0, n, step):
            aligator = Aligator()
            alig1 = aligator.run_aligator(n, y, np.arange(0, n), 0, 1, 1e-5)
            aligator2 = Aligator()
            alig2 = aligator2.run_aligator(n, y, np.flip(np.arange(0, n)), 0, 1, 1e-5)
            alig = np.nanmean(np.array([alig1, alig2]), axis=0)
            alig = np.concatenate((alig[n - i :], alig[0 : n - i]))
            res.append(alig)
            y = np.concatenate((y[step:], y[0:step]))

        alig = np.nanmean(np.array(res), axis=0)
        detect_res = np.where(alig > threshold)[0]

        # find sorted intervals
        detect_res = sorted(detect_res.tolist())
        intervals = []
        if len(detect_res) == 0:
            return []
        elif len(detect_res) == 1:
            return [(detect_res[0], detect_res[0])]
        current_start = detect_res[0]
        current_end = detect_res[0]
        for x in detect_res[1:]:
            if current_end + 1 == x:
                current_end += 1  # update current end if next index is detected
            else:
                intervals.append((current_start, current_end))  # got an interval
                current_start = x
                current_end = x

        # check for any leftover intervals
        if current_end > current_start:
            intervals.append((current_start, current_end))

        # end timer
        end_time = perf_counter()

        return intervals, end_time - start_time


# CPP based faster alternative
class AligatorCPPDetector:

    def __init__(self, vocab_size, alpha=0.05, B=1000):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.prev_pred = 0
        self.B = B

    def detect(self, pivot: np.ndarray, null_distn):
        # calculate threshold empirically
        null_samples = null_distn((self.B,))
        threshold = np.quantile(null_samples, 1 - self.alpha)

        # Start time
        start_time = perf_counter()

        n = pivot.shape[0]
        y = pivot.copy()
        step = int(n / 30)
        res = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # This suppresses all warnings within this 'with' block, warnigns from c++

            # bidirectional circular detection
            for i in range(0, n, step):
                alig1 = aligator_cpp.run_aligator(n, y, np.arange(0, n), 0, 1, 1e-5)
                alig2 = aligator_cpp.run_aligator(n, y, np.flip(np.arange(0, n)), 0, 1, 1e-5)
                alig = np.nanmean(np.array([alig1, alig2]), axis=0)
                alig = np.concatenate((alig[n - i :], alig[0 : n - i]))
                res.append(alig)
                y = np.concatenate((y[step:], y[0:step]))

        alig = np.nanmean(np.array(res), axis=0)
        detect_res = np.where(alig > threshold)[0]

        # find sorted intervals
        detect_res = sorted(detect_res.tolist())
        intervals = []
        if len(detect_res) == 0:
            return [], perf_counter() - start_time
        elif len(detect_res) == 1:
            return [(detect_res[0], detect_res[0])], perf_counter() - start_time
        current_start = detect_res[0]
        current_end = detect_res[0]
        for x in detect_res[1:]:
            if current_end + 1 == x:
                current_end += 1  # update current end if next index is detected
            else:
                intervals.append((current_start, current_end))  # got an interval
                current_start = x
                current_end = x

        # check for any leftover intervals
        if current_end > current_start:
            intervals.append((current_start, current_end))

        # End time
        end_time = perf_counter()

        return intervals, end_time - start_time


##############
# Watermark CPD
# Reference Code: https://github.com/doccstat/llm-watermark-cpd
# Reference Paper: https://arxiv.org/pdf/2410.20670


class SeedBSIntervalResult:
    r: int
    s: int
    tau_hat: int
    p_tilde: float

    def __init__(self, r, s, tau_hat, p_tilde):
        self.r = r
        self.s = s
        self.tau_hat = tau_hat
        self.p_tilde = p_tilde
        self.len = s - r

    def __repr__(self):
        return f"{self.r} - {self.s} with length {self.len} ({self.tau_hat})"


class SeedBSNOTDetector:

    def __init__(
        self,
        vocab_size: int,
        B=1000,
        zeta: float = 0.05,
        min_length=50,
        decay=math.sqrt(2),
        significance_permutation_count=99,
        rolling_window_size=20,
        n_jobs=1,
    ):
        self.vocab_size = vocab_size
        self.B = B
        self.min_length = min_length
        self.zeta = zeta
        self.decay = decay
        self.significance_permutation_count = significance_permutation_count
        self.rolling_window_size = rolling_window_size
        self.n_jobs = n_jobs

    def run_seedbs(self, n: int, unique_int=False):
        depth = math.ceil(math.log(n, self.decay))

        boundary_mtx = []
        boundary_mtx.append((1, n))
        for i in range(2, depth + 1):
            int_length = n * (1 / self.decay) ** (i - 1)
            n_int = math.ceil(round(n / int_length, 14)) * 2 - 1
            starts = np.floor(np.linspace(1, n - int_length, int(n_int))).astype(int)
            ends = np.ceil(np.linspace(int_length, n, int(n_int))).astype(int)
            for st, end in zip(starts, ends):
                boundary_mtx.append((st, end))

        if unique_int:
            boundary_mtx = np.unique(boundary_mtx, axis=0)
        return np.array(boundary_mtx)

    def ks_statistic(self, pvalues: np.ndarray):
        result = []
        n = pvalues.size
        for k in range(1, n):
            segment_before = pvalues[:k]
            segment_after = pvalues[k:]
            if len(segment_before) == 0 or len(segment_after) == 0:
                continue
            ks_test_stat = ks_2samp(segment_before, segment_after).statistic
            value = k * (n - k) / (n**1.5) * ks_test_stat
            result.append((k, value))
        if not result:
            return (None, None)
        max_k, max_val = max(result, key=lambda x: x[1])
        return (max_k, max_val)

    def permute_pvalues(self, pvalues: np.ndarray, block_size=1):
        n = pvalues.size
        pvalue_indices = list(range(n - block_size + 1))
        sampled_size = math.ceil(n / block_size)
        sampled_indices = random.sample(pvalue_indices, k=sampled_size)

        permuted_pvalues = []
        for idx in sampled_indices:
            permuted_pvalues.extend(pvalues[idx : idx + block_size])

        # Truncate to original length
        return np.array(permuted_pvalues[:n])

    def segment_significance(self, pvalues: np.ndarray):
        original_ks_statistic = self.ks_statistic(pvalues)
        if original_ks_statistic[1] is None:
            return (None, 1)

        def single_permutation():
            pvalues_permuted = self.permute_pvalues(pvalues, block_size=10)
            ks_statistic_permuted = self.ks_statistic(pvalues_permuted)
            if ks_statistic_permuted[1] is None:
                return 0
            return int(original_ks_statistic[1] <= ks_statistic_permuted[1])

        # Run in parallel
        p_tilde = Parallel(n_jobs=self.n_jobs)(
            delayed(single_permutation)() for _ in range(self.significance_permutation_count)
        )
        p_tilde.append(1)
        return (original_ks_statistic[0], np.mean(p_tilde))

    def detect(self, pivot_stats: np.ndarray, null_distn):
        # use the null distribution, to figure out the p-values
        null_samples = null_distn((self.B,))
        null_sorted = np.sort(null_samples)
        idx = np.searchsorted(null_sorted, pivot_stats, side="right")  # all things on left is smaller
        pvals = (self.B - idx) / self.B

        # Start timer
        start_time = perf_counter()

        seeded_intervals = self.run_seedbs(pvals.size - self.rolling_window_size, unique_int=True)

        # apply segment length cutoff
        segment_length = seeded_intervals[:, 1] - seeded_intervals[:, 0]
        segment_length_cutoff = segment_length >= self.min_length
        seeded_intervals = seeded_intervals[segment_length_cutoff, :]

        # apply significance test
        results = []
        for interval in seeded_intervals:
            ri, si = interval
            tau, pval = self.segment_significance(pvals[ri:si])
            if tau is not None:
                results.append(SeedBSIntervalResult(ri, si, tau + ri - 1, pval))

        # narrowest-over-threshold selection
        selected: List[int] = []
        over_threshold = [res for res in results if res.p_tilde < self.zeta]
        while len(over_threshold) > 0:
            # choose the narrowest interval
            i_min = min(range(len(over_threshold)), key=lambda j: over_threshold[j].s - over_threshold[j].r)
            chosen = over_threshold[i_min]
            selected.append(chosen.tau_hat)

            tau_i = chosen.tau_hat
            over_threshold_new = [res for j, res in enumerate(over_threshold) if not (res.r < tau_i <= res.s)]
            if len(over_threshold_new) == len(over_threshold):
                break  # it is going to infinite loop, so break
            over_threshold = over_threshold_new

        # deduplicate and sort cp
        cps = sorted(list(set(selected)))

        est_intervals = []
        current_index = 0
        is_segment_wm = False
        for cp in cps:
            if is_segment_wm:
                est_intervals.append((current_index, cp))
            is_segment_wm = not is_segment_wm
            current_index = cp
        if is_segment_wm:
            est_intervals.append((current_index, current_index))

        end_time = perf_counter()  # end timer
        est_intervals = [(int(start), int(end)) for start, end in est_intervals]
        return est_intervals, end_time - start_time


class SeedBSNOTDetectorCPP(SeedBSNOTDetector):

    def segment_significance(self, pvalues: np.ndarray):
        n_jobs = max(1, getattr(self, "n_jobs", 1))
        k_obs, p_tilde = seedbs_cpp.segment_significance(
            pvalues.astype(np.float64),
            n_permutations=self.significance_permutation_count,
            block_size=getattr(self, "block_size", 10),
            seed=0,
            n_jobs=n_jobs,
        )

        if k_obs <= 0:
            return (None, 1.0)
        return (k_obs, p_tilde)


###########
# WinMax
# Paper: Kirchenbaucher et al.
class WinMaxDetector:

    def __init__(self, vocab_size, window_interval: int = 5, alpha=0.05, B=1000):
        self.vocab_size = vocab_size
        self.window_interval = window_interval
        self.alpha = alpha
        self.B = B  # number of samples to use to generate p-values

    def detect(self, pivots: np.ndarray, null_distn, agg_fun=None):
        if agg_fun is None:
            agg_fun = np.sum

        max_L = len(pivots) - 2
        min_L = 1

        # calculate null_agg for each L
        null_agg_list = []
        for L in range(min_L, max_L + 1, self.window_interval):
            # calculate the p-values empirically
            null_samples = null_distn((self.B, L))
            null_agg = np.array([agg_fun(null_samples[b, :]) for b in range(self.B)])
            null_agg_list.append(null_agg)

        # Start timer
        start_time = perf_counter()

        min_p_value = float("inf")
        flag_start_idx, flag_end_idx = -1, -1

        # traverse all possible segments
        for i, L in enumerate(range(min_L, max_L + 1, self.window_interval)):

            for start_idx in range(2, len(pivots) - L + 1):
                token_window = pivots[start_idx : (start_idx + L)]
                token_agg = agg_fun(token_window)
                pval = np.sum(null_agg_list[i] > token_agg) / self.B
                if pval < min_p_value:
                    min_p_value = pval
                    flag_start_idx, flag_end_idx = start_idx, start_idx + L

        # end timer
        end_time = perf_counter()

        if min_p_value < self.alpha:
            # there is a watermark
            return [(flag_start_idx, flag_end_idx)], end_time - start_time  # always return the maximum interval
        else:
            return [], end_time - start_time


#########
# Fixed Window Length


class FixedWindowDetector:

    def __init__(self, vocab_size, window_len: int = 40, alpha=0.05, B=1000):
        self.vocab_size = vocab_size
        self.window_len = window_len
        self.alpha = alpha
        self.B = B  # number of samples to use to generate p-values

    def detect(self, pivots: np.ndarray, null_distn, agg_fun=None):
        if agg_fun is None:
            agg_fun = np.sum

        # calculate the p-values empirically
        null_samples = null_distn((self.B, self.window_len))
        null_agg = np.array([agg_fun(null_samples[b, :]) for b in range(self.B)])
        threshold = np.quantile(null_agg, 1 - self.alpha)

        # Start timer
        start_time = perf_counter()

        indices = []
        for start_idx in range(2, len(pivots) - self.window_len + 1):
            token_window = pivots[start_idx : (start_idx + self.window_len)]
            token_agg = agg_fun(token_window)
            if token_agg > threshold:
                indices.append((start_idx, start_idx + self.window_len))

        # end timer
        end_time = perf_counter()

        return indices, end_time - start_time


############
# WaterSeeker
# Paper: https://aclanthology.org/2025.findings-naacl.156.pdf
# Code: https://github.com/THU-BPM/WaterSeeker


class WaterSeekerDetector:

    def __init__(
        self,
        vocab_size: int,
        alpha=0.05,
        B=1000,
        threshold_1=0.5,
        threshold_2=1.5,
        top_k=20,
        min_length=50,
        tolerance=50,
        window_size=50,
    ):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.B = B
        self.threshold_1 = threshold_1
        self.threshold_2 = threshold_2
        self.top_k = top_k
        self.min_length = min_length
        self.tolerance = tolerance
        self.window_size = window_size

    def detect_anomalies(self, token_scores: np.ndarray):
        window_size = self.window_size

        # calculate the moving average of the token scores
        proportions = []
        for i in range(len(token_scores) - window_size + 1):
            window = token_scores[i : (i + window_size)]
            proportion = np.sum(window) / window_size
            proportions.append(proportion)

        # calculate the mean and sd of proportions
        mean_prop = np.mean(proportions)
        sd_prop = np.std(proportions)

        # find top-k proportions
        top_props = sorted(proportions, reverse=True)[: self.top_k]
        top_mean_prop = np.mean(top_props)

        # calculate difference value
        diff_val = max((top_mean_prop - mean_prop) * self.threshold_1, sd_prop * self.threshold_2)
        anomalies = [i for i, p in enumerate(proportions) if p > mean_prop + diff_val]

        # merge adjacent anomalies
        merged_anomalies = []
        current_segment = []

        for i in range(len(anomalies)):
            if not current_segment:
                current_segment = [anomalies[i]]
            else:
                if anomalies[i] - current_segment[-1] <= self.tolerance:
                    current_segment.append(anomalies[i])
                else:
                    merged_anomalies.append(current_segment)
                    current_segment = [anomalies[i]]

        # handle any leftover partition
        if current_segment:
            merged_anomalies.append(current_segment)

        # filter segments that are too short
        valid_segments = []
        for segment in merged_anomalies:
            if self.min_length <= (segment[-1] - segment[0] + window_size - 1):
                valid_segments.append((segment[0], segment[-1] + window_size - 1))

        if valid_segments:
            return valid_segments
        else:
            return None

    def detect(self, pivots: np.ndarray, null_distn, agg_fun=None):
        if agg_fun is None:
            agg_fun = np.sum

        # calculate the p-values empirically
        null_samples = null_distn((self.B, self.window_size))
        null_agg = np.array([agg_fun(null_samples[b, :]) for b in range(self.B)])
        threshold = np.quantile(null_agg, 1 - self.alpha)

        # start timer
        start_time = perf_counter()

        # suspicious segments localization
        indices = self.detect_anomalies(pivots)

        # check if suspicious segments are watermarked
        filtered_indices = []
        if indices is not None:

            for indice in indices:
                found_in_current_indice = False
                max_agg = -float("inf")
                best_index = None

                # local traversal
                for start_idx in range(indice[0], indice[0] + self.window_size):
                    for end_idx in range(indice[-1], indice[-1] - self.window_size, -1):
                        if end_idx - start_idx < self.min_length:
                            break

                        token_window = pivots[start_idx:end_idx]
                        token_agg = agg_fun(token_window)
                        if token_agg > threshold:
                            if token_agg > max_agg:
                                max_agg = token_agg
                                best_index = (start_idx, end_idx)
                            found_in_current_indice = True

                if found_in_current_indice and best_index is not None:
                    filtered_indices.append(best_index)

        # end timer
        end_time = perf_counter()
        return filtered_indices, end_time - start_time


###########
# WISER
# Proposed epidemic based changepoint detector algorithm


class WISERDetector:

    def __init__(self, vocab_size: int, alpha=0.05, B=1000, rho=0.5, C=0.1, gamma=0.1, seed=1234):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.B = B
        self.rho = rho
        self.C = C
        self.gamma = gamma
        self.seed = seed
        self.d = None

    def get_pivot_length(self, pivot_stats: np.ndarray):
        assert pivot_stats.ndim == 1, "Pivot statistic should be a 1D array"
        n = pivot_stats.shape[0]
        return n

    def detect_first_stage(
        self, pivot_stats: np.ndarray, threshold, block_size: int, c: int  # 1D array of pivot statistics
    ):

        # perform the reduceat operation for pivot statistics
        n = pivot_stats.shape[0]
        block_indices = np.arange(0, n, block_size).astype(int)
        pivot_block_sums = np.add.reduceat(pivot_stats, block_indices)  # perform the blocked sum

        # Part 2: Vectorized identification of contiguous blocks over the threshold
        is_over_threshold = pivot_block_sums > threshold

        # Use diff to find where a run of True values starts (0 -> 1) and ends (1 -> 0)
        padded = np.concatenate(([False], is_over_threshold, [False]))
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]  # Get the start and end indices of the blocks
        ends = np.where(diff == -1)[0] - 1

        if starts.size == 0:
            return []

        left_indices = starts * block_size
        right_indices = (ends + 1) * block_size - 1

        # filter intervals whose lengths are small
        lengths = right_indices - left_indices + 1
        is_long_enough = lengths >= (c * block_size)
        filtered_lefts = left_indices[is_long_enough]
        filtered_rights = right_indices[is_long_enough]

        return list(zip(filtered_lefts, filtered_rights))

    def detect_second_stage(
        self, pivot_stats: np.ndarray, major_intervals: List[Tuple[int, int]], block_size: int, mean_under_null: float
    ):
        n = pivot_stats.shape[0]
        M = pivot_stats - mean_under_null  # subtract mu_0 from all

        # type = 1, is the usually parallelized version of CUSUM
        intervals = []

        # a useful trick is to store cumulative sums with a 0 at the beginning
        # This way, the sum of M[i:j+1] is always Vsum[j+1] - Vsum[i]
        Vsum = np.insert(M.cumsum(), 0, 0)

        # common d(tilde) calculation
        Dtilde_sum = 0
        Dtilde_count = 0
        for left_end, right_end in major_intervals:
            # get the wiggling indices
            mid = int((left_end + right_end) / 2)  # middle index
            # now tweak by +/- block_size in both direction, without crossover at mid
            left_index_start = int(max(0, left_end - block_size - self.C * (n ** (0.5 + self.gamma))))
            left_index_end = int(min(left_end + block_size, mid - 1))
            right_index_start = int(max(mid, right_end - block_size))
            right_index_end = int(min(right_end + block_size + self.C * (n ** (0.5 + self.gamma)), n - 1))

            Dtilde_sum += Vsum[right_index_end + 1] - Vsum[left_index_start]
            Dtilde_count += right_index_end - left_index_start
        if Dtilde_count <= 0:
            return []  # no major blocks detected
        d_tilde = Dtilde_sum / Dtilde_count
        self.d = d_tilde

        for left_end, right_end in major_intervals:
            # get the wiggling indices
            mid = int((left_end + right_end) / 2)  # middle index
            # now tweak by +/- block_size in both direction, without crossover at mid
            left_index_start = int(max(0, left_end - block_size - self.C * (n ** (0.5 + self.gamma))))
            left_index_end = int(min(left_end + block_size, mid - 1))
            right_index_start = int(max(mid, right_end - block_size))
            right_index_end = int(min(right_end + block_size + self.C * (n ** (0.5 + self.gamma)), n - 1))

            # Create 1D arrays of all possible left_indices and right_indices values
            i_vals = np.arange(left_index_start, left_index_end + 1)
            j_vals = np.arange(right_index_start, right_index_end + 1)

            # If either search range is empty, skip to the next major interval
            if i_vals.size == 0 or j_vals.size == 0:
                continue

            # block level stuffs that are useful to calculate complementary sums
            Dj = Vsum[right_index_end + 1] - Vsum[left_index_start]
            current_block_size = right_index_end - left_index_start
            # dj = Dj / current_block_size

            # create a vectorized 2D calculation grid for faster search
            i_grid = i_vals[:, np.newaxis]
            j_grid = j_vals[np.newaxis, :]
            lr_sum_grid = Vsum[j_grid + 1] - Vsum[i_grid]  # Calculate sums and sizes for all (i, j) pairs at once
            lr_size_grid = j_grid - i_grid
            lr_c_sum_grid = Dj - lr_sum_grid  # Calculate complementary sums and sizes
            lr_c_size_grid = current_block_size - lr_size_grid

            Mij_grid = (
                lr_c_sum_grid - self.rho * d_tilde * lr_c_size_grid
            )  # calculate Mij statistic for all (i, j) combination

            # find best index
            min_flat_index = np.argmin(Mij_grid)
            min_i_index, min_j_index = np.unravel_index(
                min_flat_index, Mij_grid.shape
            )  # Convert the flat index back to 2D (row, col) coordinates

            # Find the optimal i and j that produced the minimum Mij
            min_i = i_vals[min_i_index]
            min_j = j_vals[min_j_index]

            intervals.append((min_i, min_j))

        return intervals

    def detect(self, pivot_stats: np.ndarray, null_distn, block_size=None, c=2):
        n = self.get_pivot_length(pivot_stats)
        if block_size is None:
            block_size = np.ceil(n**0.5)

        np.random.seed(self.seed)

        Bsamples = null_distn((self.B, n))  # simulate from exact null distn
        block_indices = np.arange(0, n, block_size).astype(int)
        block_sums = np.add.reduceat(Bsamples, block_indices, axis=1)  # perform the blocked sum
        Vstats = np.abs(block_sums).max(axis=1)  # this is (B,)
        th = np.quantile(Vstats, q=(1 - self.alpha))  # find out (1-alpha) quantile
        mean_under_null = np.mean(null_distn((self.B,)))

        # Start timer
        start_time = perf_counter()

        major_intervals = self.detect_first_stage(pivot_stats, th, block_size, c)
        intervals = self.detect_second_stage(pivot_stats, major_intervals, block_size, mean_under_null)

        # end timer
        end_time = perf_counter()

        return intervals, end_time - start_time


#########
# Kadane's Algorithm based detector


class KadaneDetector:
    """
    Uses Kadane's algorithm to solve the optimization problem:

    I(hat) = argmin_{s,t} sum_(k \notin [s, t]) (X_k - mu_0 - rho * d)

    Has the implementation to detect only single watermarked patch
    """

    def __init__(
        self, vocab_size, alpha=0.05, B=1000, rho=0.5, seed=1234, thresholding_procedure: str = "e-BH"
    ) -> None:
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.B = B
        self.rho = rho
        self.seed = seed
        self.thresholding_procedure = thresholding_procedure

    def get_mu_0(self, null_distn):
        np.random.seed(self.seed)
        Bsamples = null_distn((self.B,))  # simulate from exact null distn
        mu_0 = np.mean(Bsamples)
        return mu_0

    def get_pivot_length(self, pivot_stats: np.ndarray):
        assert pivot_stats.ndim == 1, "Pivot statistic should be a 1D array"
        n = pivot_stats.shape[0]
        return n

    def solve_single_maxsum(self, arr, seg_start=0, seg_end=-1):
        """
        Implements Kadane's algorithm to efficiently find
        maximum contiguous subarray sum.
        """
        if len(arr) == 0:
            return 0, 0, 0
        if seg_end < 0:
            seg_end = len(arr) - 1
        if seg_start > seg_end:  # allow single element segments
            return 0, 0, 0

        max_sum = float("-inf")
        current_sum = 0
        start = end = temp_start = seg_start

        for i in range(seg_start, seg_end + 1):
            if current_sum <= 0:
                current_sum = arr[i]
                temp_start = i
            else:
                current_sum += arr[i]

            if current_sum > max_sum:
                max_sum = current_sum
                start = temp_start
                end = i

        return max_sum, start, end

    def find_interval_pvalues(self, pivot_stats: np.ndarray, null_distn, intervals: List[Tuple[int, int]]):
        """
        Uses a bootstrap based method to simulate from null distn
        and calculate the p-values based on the sum
        """
        n = self.get_pivot_length(pivot_stats)
        np.random.seed(self.seed)
        Bsamples = null_distn((self.B, n))  # simulate from exact null distn
        pvalues = []
        for start, end in intervals:
            pivot_sum = np.sum(pivot_stats[start : end + 1])  # float
            null_sum = np.sum(Bsamples[:, start : end + 1], axis=1)  # (B, )
            pval = np.sum(null_sum > pivot_sum) / self.B  # count prop of time null_sum exceeds the observed pivot_sum
            pvalues.append(pval)
        return pvalues

    def filter_intervals(self, intervals, pvalues):
        filtered_intervals = []
        evalues = -np.log(np.clip(pvalues, 1e-10, 1))  # compute e-value from p-value: e-val = -log(p)
        K = len(intervals)

        if self.thresholding_procedure == "e-BH":
            # e-BH procedure: Sort the e-values and reject if ke_k > K/alpha
            sorted_evals_index = np.argsort(evalues)[::-1]
            sorted_evals = evalues[sorted_evals_index]
            sorted_intervals = [intervals[i] for i in sorted_evals_index]
            filtered_intervals = []
            for i, (e_val, interval) in enumerate(zip(sorted_evals, sorted_intervals)):
                if (i * e_val) > K * self.alpha:
                    # null is rejected, so include in the filtered intervals
                    filtered_intervals.append(interval)
        elif self.thresholding_procedure == "e-BY":
            # e-BY procedure
            raise NotImplementedError()
        else:
            raise Exception("Unknown thresholding procedure")

        return filtered_intervals

    def get_true_d(self, pivot_stats: np.ndarray, null_distn, true_intervals: List[Tuple[int, int, str]]):
        wm_pivot_score = 0
        wm_pivot_count = 0
        for start, end, label in true_intervals:
            if label != "unwatermarked":
                wm_pivot_score += np.sum(pivot_stats[start : end + 1])
                wm_pivot_count += end - start + 1
        mu1 = wm_pivot_score / wm_pivot_count
        mu0 = self.get_mu_0(null_distn)
        d_tilde = mu1 - mu0  # pass true d
        return d_tilde


class KadaneGreedyDetector(KadaneDetector):
    """
    Implements a divide and conquer approach
    to find the top-K sum disjoint subarrays
    """

    def solve_top_k_disjoint(self, arr, k):
        """
        Finds top K subarray using a divide and conquer type approach
        For each potential segment boundary, we compute the best subarray
        and use a heap to extract top K non-overlapping ones.
        """
        if (len(arr) == 0) or (k <= 0):
            return 0, []

        n = len(arr)

        # Initial full interval
        max_sum, s, e = self.solve_single_maxsum(arr, 0, n - 1)
        heap = []  # used as heap storage
        heapq.heappush(
            heap, (-max_sum, 0, n - 1, s, e)
        )  # Push tuple: (neg_sum, seg_left, seg_right, best_start, best_end)

        total_sum = 0
        segments = []
        for _ in range(k):
            if not heap:
                break

            neg_sum, seg_l, seg_r, best_s, best_e = heapq.heappop(heap)
            best_value = -neg_sum

            # If the best remaining segment is non-positive, stop
            if best_value <= 0:
                break

            total_sum += best_value
            segments.append((best_value, best_s, best_e))

            # Left region
            if seg_l < best_s:
                left_sum, ls, le = self.solve_single_maxsum(arr, seg_l, best_s - 1)
                heapq.heappush(heap, (-left_sum, seg_l, best_s - 1, ls, le))

            # Right region
            if best_e < seg_r:
                right_sum, rs, re = self.solve_single_maxsum(arr, best_e + 1, seg_r)
                heapq.heappush(heap, (-right_sum, best_e + 1, seg_r, rs, re))

        return total_sum, segments

    def detect(self, pivot_stats: np.ndarray, null_distn, max_k=10, custom_d=None):
        n = self.get_pivot_length(pivot_stats)
        mu_0 = self.get_mu_0(null_distn)

        start_time = perf_counter()
        centered_pivot = pivot_stats - mu_0  # (X_k - mu_0)
        if custom_d is not None:
            d_tilde = custom_d
        else:
            # need to calculate d
            # one way is to run kadane single and estimate
            best_sum, best_start, best_end = self.solve_single_maxsum(centered_pivot, 0, n - 1)
            if best_sum <= 0:
                return [], perf_counter() - start_time
            d_tilde = best_sum / (best_end - best_start + 1)  # create an average of (mu_1 - mu_0)

        pivot_score = centered_pivot - self.rho * d_tilde  # (X_k - mu_0 - rho * tilde(d))

        # apply Kadane's greedy algorithm to find best max_k segments
        best_sum, intervals = self.solve_top_k_disjoint(pivot_score, max_k)
        intervals = [(start, end) for _, start, end in intervals]
        end_time = perf_counter()

        # do the thresholding based on p-values
        pvals = self.find_interval_pvalues(pivot_stats, null_distn, intervals)
        intervals = self.filter_intervals(intervals, pvals)

        return intervals, end_time - start_time


class KadaneDPDetector(KadaneDetector):
    """
    Uses dynamic programming to solve the optimization problem:

    I(hat) = argmin_{s,t} sum_(k \notin [s, t]) (X_k - mu_0 - rho * d)
    """

    def solve_top_k_dp(self, arr, k):
        """
        Finds top K disjoint subarray sums using DP.

        dp[t][i] = best sum of exactly t disjoint subarrays within arr[0..i]

        Recurrence:
            dp[t][i] = max(
                dp[t][i-1],                   # don't end a subarray at i
                best_t_ending_at[i]           # end the t-th subarray at i
            )
        where:
            best_t_ending_at[i] = max over j<=i of (dp[t-1][j-1] + sum(arr[j..i]))
                                = max over j<=i of (dp[t-1][j-1] - prefix[j-1]) + prefix[i]
                                = prefix[i] + max over j<=i of (dp[t-1][j-1] - prefix[j-1])

        Time:  O(K * n)
        Space: O(K * n)
        """
        n = len(arr)
        if n == 0 or k <= 0:
            return 0, []

        # Prefix sums for O(1) range sum queries
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + arr[i]

        # dp[t][i] = best sum of t subarrays in arr[0..i]
        # We store (sum, list of (start,end) segments) for reconstruction
        NEG_INF = float("-inf")
        dp = [[(NEG_INF, [])] * n for _ in range(k + 1)]

        # Base: 0 subarrays, any prefix → sum 0, no segments
        base = [(0, [])] * n

        for t in range(1, k + 1):
            prev = base if t == 1 else dp[t - 1]

            # best_j = max over j<=i of (dp[t-1][j-1] - prefix[j])
            # tracks the best "opening value" if we start a subarray at j
            best_j_val = NEG_INF
            best_j_segs = []
            best_j_start = 0

            for i in range(n):
                # j = i means subarray starts at i
                # opening value = dp[t-1][i-1] - prefix[i]
                prev_sum, prev_segs = prev[i - 1] if i > 0 else (0, [])
                open_val = prev_sum - prefix[i]

                if open_val > best_j_val:
                    best_j_val = open_val
                    best_j_segs = prev_segs
                    best_j_start = i

                # Best sum ending at i using t subarrays
                if best_j_val == NEG_INF:
                    end_here = (NEG_INF, [])
                else:
                    seg_sum = best_j_val + prefix[i + 1]  # = best_j_val + prefix[i+1]
                    end_here = (seg_sum, best_j_segs + [(best_j_start, i)])

                # Either extend previous best, or end a subarray here
                prev_best = dp[t][i - 1] if i > 0 else (NEG_INF, [])
                if end_here[0] >= prev_best[0]:
                    dp[t][i] = end_here
                else:
                    dp[t][i] = prev_best

        # Answer is dp[k][n-1], but also check fewer than k subarrays
        # in case k is larger than the number of positive subarrays
        best_sum, best_segs = NEG_INF, []
        for t in range(1, k + 1):
            s, segs = dp[t][n - 1]
            if s > best_sum:
                best_sum, best_segs = s, segs

        if best_sum <= 0:
            return 0, []

        return best_sum, best_segs

    def detect(self, pivot_stats: np.ndarray, null_distn, max_k=10, custom_d=None, do_thresholding=True):
        n = self.get_pivot_length(pivot_stats)
        mu_0 = self.get_mu_0(null_distn)

        start_time = perf_counter()
        centered_pivot = pivot_stats - mu_0  # (X_k - mu_0)
        if custom_d is not None:
            d_tilde = custom_d
        else:
            # need to calculate d
            # one way is to run kadane single and estimate
            best_sum, best_start, best_end = self.solve_single_maxsum(centered_pivot, 0, n - 1)
            if best_sum <= 0:
                return [], perf_counter() - start_time
            d_tilde = best_sum / (best_end - best_start + 1)  # create an average of (mu_1 - mu_0)

        pivot_score = centered_pivot - self.rho * d_tilde  # (X_k - mu_0 - rho * tilde(d))

        # apply Kadane's greedy algorithm to find best max_k segments
        best_sum, intervals = self.solve_top_k_dp(pivot_score, max_k)
        end_time = perf_counter()

        if do_thresholding:
            # do the thresholding based on p-values
            pvals = self.find_interval_pvalues(pivot_stats, null_distn, intervals)
            intervals = self.filter_intervals(intervals, pvals)

        return intervals, end_time - start_time
