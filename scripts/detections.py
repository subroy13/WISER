# Alligator
from typing import List, Any, Union, Tuple
import numpy as np
import warnings
from time import perf_counter
import aligator_cpp.aligator as aligator_cpp


##################
# ALIGATOR
# Reference Code: https://github.com/XuandongZhao/llm-watermark-location
# Reference Paper: https://arxiv.org/pdf/2410.03600v2

# Pure python implementation
class Expert:
    def __init__(self, start, end):
        self.start = start # expert's interval range
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
                e = Expert(
                    start=i*(2**k),
                    end=(i+1)*(2**k)
                )
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
            i = (t >> k)
            if (((i + 1) << k) - 1 > n) or (k <= 4):
                awake_set.append(-1)
            else:
                awake_set.append(i)
        return awake_set

    def get_forecast(
            self,
            awake_set: list[int], 
            pool: List[List[Expert]], 
            pool_size: int,
            prev_pred: float
        ):
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
            output += (pool[k][i].weight * pool[k][i].prediction)
            normalizer += pool[k][i].weight
        if normalizer == 0:
            normalizer = 1

        return output / normalizer, normalizer

    def compute_losses(
            self,
            awake_set: list[int], 
            pool: List[List[Expert]], 
            y: float,
            B: float, 
            n: int, 
            sigma: float, 
            delta: float
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
            self,
            awake_set: list[int], 
            pool: List[List[Expert]], 
            losses: List[float], 
            normalizer: float, 
            y: float
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
            pool[k][i].weight *= (np.exp(-losses[k]) * normalizer / denom)
            pool[k][i].prediction = ((pool[k][i].prediction * pool[k][i].count) + y) / (pool[k][i].count + 1)
            pool[k][i].count += 1
        return pool

    def run_aligator(
            self,
            n: int, 
            y: Union[List[float], np.ndarray], 
            index: List[int], 
            sigma: float, 
            B: float, 
            delta: float
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

    def __init__(self, vocab_size, alpha = 0.05, B = 1000):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.prev_pred = 0
        self.B = B

    def detect(self, pivot: np.ndarray, null_distn):
        # calculate threshold empirically
        null_samples = null_distn((self.B, ), self.vocab_size)
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
            alig = np.nanmean(np.array([alig1, alig2]), axis = 0)
            alig = np.concatenate((alig[n-i:], alig[0:n-i]))
            res.append(alig)
            y = np.concatenate((y[step:], y[0:step]))

        alig = np.nanmean(np.array(res), axis = 0)
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
                intervals.append((current_start, current_end)) # got an interval
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

    def __init__(self, vocab_size, alpha = 0.05, B = 1000):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.prev_pred = 0
        self.B = B

    def detect(self, pivot: np.ndarray, null_distn):
        # calculate threshold empirically
        null_samples = null_distn((self.B, ), self.vocab_size)
        threshold = np.quantile(null_samples, 1 - self.alpha)

        # Start time
        start_time = perf_counter()

        n = pivot.shape[0]
        y = pivot.copy()
        step = int(n / 30)
        res = []


        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # This suppresses all warnings within this 'with' block, warnigns from c++

            # bidirectional circular detection
            for i in range(0, n, step):
                alig1 = aligator_cpp.run_aligator(n, y, np.arange(0, n), 0, 1, 1e-5)
                alig2 = aligator_cpp.run_aligator(n, y, np.flip(np.arange(0, n)), 0, 1, 1e-5)
                alig = np.nanmean(np.array([alig1, alig2]), axis = 0)
                alig = np.concatenate((alig[n-i:], alig[0:n-i]))
                res.append(alig)
                y = np.concatenate((y[step:], y[0:step]))

        alig = np.nanmean(np.array(res), axis = 0)
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
                intervals.append((current_start, current_end)) # got an interval
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


###########
# WinMax 
# Paper: Kirchenbaucher et al.
class WinMaxDetector:

    def __init__(self, vocab_size, window_interval: int = 5, alpha = 0.05, B = 1000):
        self.vocab_size = vocab_size
        self.window_interval = window_interval
        self.alpha = alpha
        self.B = B   # number of samples to use to generate p-values

    def detect(self, pivots: np.ndarray, null_distn, agg_fun = None):
        if agg_fun is None:
            agg_fun = np.sum
        
        max_L = len(pivots) - 2 
        min_L = 1

        # calculate null_agg for each L
        null_agg_list = []
        for L in range(min_L, max_L + 1, self.window_interval):
            # calculate the p-values empirically
            null_samples = null_distn((self.B, L), self.vocab_size)
            null_agg = np.array([agg_fun(null_samples[b, :]) for b in range(self.B)])
            null_agg_list.append(null_agg)

        # Start timer
        start_time = perf_counter()

        min_p_value = float('inf')
        flag_start_idx, flag_end_idx = -1, -1
        
        # traverse all possible segments
        for i, L in enumerate(range(min_L, max_L + 1, self.window_interval)):
            
            for start_idx in range(2, len(pivots) - L + 1):
                token_window = pivots[start_idx:(start_idx + L)]
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

    def __init__(self, vocab_size, window_len: int = 40, alpha = 0.05, B = 1000):
        self.vocab_size = vocab_size
        self.window_len = window_len
        self.alpha = alpha
        self.B = B   # number of samples to use to generate p-values

    def detect(self, pivots: np.ndarray, null_distn, agg_fun = None):
        if agg_fun is None:
            agg_fun = np.sum
        
        # calculate the p-values empirically
        null_samples = null_distn((self.B, self.window_len), self.vocab_size)
        null_agg = np.array([agg_fun(null_samples[b, :]) for b in range(self.B)])
        threshold = np.quantile(null_agg, 1 - self.alpha)

        # Start timer
        start_time = perf_counter()

        indices = []
        for start_idx in range(2, len(pivots) - self.window_len + 1):
            token_window = pivots[start_idx:(start_idx + self.window_len)]
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

    def __init__(self, 
        vocab_size: int, 
        alpha = 0.05, 
        B = 1000, 
        threshold_1 = 0.5, 
        threshold_2 = 1.5, 
        top_k = 20, 
        min_length = 50,
        tolerance = 50,
        window_size = 50
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
            window = token_scores[i:(i+window_size)]
            proportion = np.sum(window) / window_size
            proportions.append(proportion)

        # calculate the mean and sd of proportions
        mean_prop = np.mean(proportions)
        sd_prop = np.std(proportions)

        # find top-k proportions
        top_props = sorted(proportions, reverse=True)[:self.top_k]
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


    def detect(self, pivots: np.ndarray, null_distn, agg_fun = None):
        if agg_fun is None:
            agg_fun = np.sum

        # calculate the p-values empirically
        null_samples = null_distn((self.B, self.window_size), self.vocab_size)
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
                max_agg = -float('inf')
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



################
# EPIDEMIC
# Proposed Epidemic Detector

class EpidemicDetector:

    def __init__(
        self, 
        vocab_size: int,
        alpha = 0.05, 
        B = 1000, 
        rho = 0.5,
        C = 0.1,
        gamma = 0.1,
        seed = 1234
    ):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.B = B
        self.rho = rho
        self.C = C
        self.gamma = gamma
        self.seed = seed

    def get_pivot_length(self, pivot_stats: np.ndarray):
        assert pivot_stats.ndim == 1, "Pivot statistic should be a 1D array"
        n = pivot_stats.shape[0]
        return n

    def detect_first_stage(
        self,
        pivot_stats: np.ndarray,  # 1D array of pivot statistics
        null_distn,  # distribution under null, returns numpy array in the given shape
        block_size: int,
        c: int
    ):
        n = self.get_pivot_length(pivot_stats)
        np.random.seed(self.seed)

        Bsamples = null_distn((self.B, n), self.vocab_size)  # simulate from exact null distn
        M = np.vstack((Bsamples, pivot_stats))  # (B+1) x n

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
        th = np.quantile(Vstats, q=(1 - self.alpha))  # find out (1-alpha) quantile

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

        # now we convert the major block indices to the minor block indices
        major_intervals = []
        for left_end, right_end in Ktilde:
            # convert left_end, right_end from block_level index to time level index
            left_index = left_end * block_size
            right_index = (right_end + 1) * block_size - 1  # inclusive
            if right_index - left_index + 1 >= c * block_size :
                major_intervals.append((left_index, right_index))

        return major_intervals

    def detect_second_stage(
        self,
        pivot_stats: np.ndarray,
        major_intervals: List[Tuple[int, int]],
        null_distn,
        block_size: int,
        mean_under_null = None
    ):
        n = self.get_pivot_length(pivot_stats)
        if mean_under_null is None:
            np.random.seed(self.seed)
            mean_under_null = np.mean(null_distn((10000, ), self.vocab_size))
        M = pivot_stats - mean_under_null  # subtract mu_0 from all
        M[np.isinf(M)] = M[~np.isinf(M)].max()  # handle infinite values

        # type = 1, is the usually parallelized version of CUSUM
        intervals = []

        # a useful trick is to store cumulative sums, so V[a:b].sum() = Vsum[b] - Vsum[a-1]
        Vsum = M.cumsum()
        for left_end, right_end in major_intervals:
            # get the wiggling indices
            mid = int((left_end + right_end) / 2)  # middle index
            # now tweak by +/- block_size in both direction, without crossover at mid
            left_index_start = int(max(0, left_end - block_size - self.C * (n**(0.5 + self.gamma)) ))
            left_index_end = int(min(left_end + block_size, mid - 1))
            right_index_start = int(max(mid, right_end - block_size))
            right_index_end = int(min(right_end + block_size + self.C * (n**(0.5 + self.gamma)), n - 1))

            # for each choice (s, t) => find the block average from s to t, and compare against block average outside s, t
            min_m = np.inf
            min_interval = None

            Dj = (Vsum[right_index_end] - Vsum[left_index_start - 1]) if left_index_start >= 1 else Vsum[right_index_end] # this is total block sum
            current_block_size = (right_index_end - 1) - left_index_start
            dj = Dj / block_size
            
            for i in range(left_index_start, left_index_end + 1):
                for j in range(right_index_start, right_index_end + 1):
                    LR_sum = (Vsum[j] - Vsum[i-1]) if i >= 1 else Vsum[j]
                    LR_size = max(j - i, 1)
                    LR_c_sum = Dj - LR_sum
                    LR_c_size = max(current_block_size - (j - i), 1)
                    Mij = LR_c_sum - self.rho * dj * LR_c_size  # the adjusted CUSUM statistic
                    if Mij < min_m:
                        min_interval = (i, j)  # track the interval with max sum
                        min_m = Mij
            if min_interval is not None:
                intervals.append(min_interval)
        return intervals
    

    def detect(
        self, 
        pivot_stats: np.ndarray, 
        null_distn,
        block_size = None,
        c = 2
    ):
        if block_size is None:
            n = self.get_pivot_length(pivot_stats)
            block_size = np.ceil(n**0.5)
        
        major_intervals = self.detect_first_stage(pivot_stats, null_distn, block_size, c)
        intervals = self.detect_second_stage(pivot_stats, major_intervals, null_distn, block_size, mean_under_null=None)
        return intervals
    

class EpidemicDetectorV2:

    def __init__(
        self, 
        vocab_size: int,
        alpha = 0.05, 
        B = 1000, 
        rho = 0.5,
        C = 0.1,
        gamma = 0.1,
        seed = 1234
    ):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.B = B
        self.rho = rho
        self.C = C
        self.gamma = gamma
        self.seed = seed

    def get_pivot_length(self, pivot_stats: np.ndarray):
        assert pivot_stats.ndim == 1, "Pivot statistic should be a 1D array"
        n = pivot_stats.shape[0]
        return n

    def detect_first_stage(
        self,
        pivot_stats: np.ndarray,  # 1D array of pivot statistics
        threshold,
        block_size: int,
        c: int
    ):

        # perform the reduceat operation for pivot statistics
        n = pivot_stats.shape[0]
        block_indices = np.arange(0, n, block_size).astype(int)
        pivot_block_sums = np.add.reduceat(pivot_stats, block_indices) # perform the blocked sum

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
        is_long_enough = (lengths >= (c * block_size))
        filtered_lefts = left_indices[is_long_enough]
        filtered_rights = right_indices[is_long_enough]

        return list(zip(filtered_lefts, filtered_rights))

    def detect_second_stage(
        self,
        pivot_stats: np.ndarray,
        major_intervals: List[Tuple[int, int]],
        block_size: int,
        mean_under_null: float
    ):
        n = pivot_stats.shape[0]
        M = pivot_stats - mean_under_null  # subtract mu_0 from all

        # type = 1, is the usually parallelized version of CUSUM
        intervals = []

        # a useful trick is to store cumulative sums with a 0 at the beginning
        # This way, the sum of M[i:j+1] is always Vsum[j+1] - Vsum[i]
        Vsum = np.insert(M.cumsum(), 0, 0)

        for left_end, right_end in major_intervals:
            # get the wiggling indices
            mid = int((left_end + right_end) / 2)  # middle index
            # now tweak by +/- block_size in both direction, without crossover at mid
            left_index_start = int(max(0, left_end - block_size - self.C * (n**(0.5 + self.gamma)) ))
            left_index_end = int(min(left_end + block_size, mid - 1))
            right_index_start = int(max(mid, right_end - block_size))
            right_index_end = int(min(right_end + block_size + self.C * (n**(0.5 + self.gamma)), n - 1))

            # Create 1D arrays of all possible left_indices and right_indices values
            i_vals = np.arange(left_index_start, left_index_end + 1)
            j_vals = np.arange(right_index_start, right_index_end + 1)

            # If either search range is empty, skip to the next major interval
            if i_vals.size == 0 or j_vals.size == 0:
                continue
                
            # block level stuffs that are useful to calculate complementary sums
            Dj = Vsum[right_index_end + 1] - Vsum[left_index_start]
            current_block_size = right_index_end - left_index_start
            dj = Dj / block_size

            # create a vectorized 2D calculation grid for faster search
            i_grid = i_vals[:, np.newaxis]
            j_grid = j_vals[np.newaxis, :]
            lr_sum_grid = Vsum[j_grid + 1] - Vsum[i_grid] # Calculate sums and sizes for all (i, j) pairs at once
            lr_size_grid = j_grid - i_grid
            lr_c_sum_grid = Dj - lr_sum_grid  # Calculate complementary sums and sizes
            lr_c_size_grid = current_block_size - lr_size_grid
            
            Mij_grid = lr_c_sum_grid - self.rho * dj * lr_c_size_grid   # calculate Mij statistic for all (i, j) combination

            # find best index
            min_flat_index = np.argmin(Mij_grid)            
            min_i_index, min_j_index = np.unravel_index(min_flat_index, Mij_grid.shape)  # Convert the flat index back to 2D (row, col) coordinates

            # Find the optimal i and j that produced the minimum Mij
            min_i = i_vals[min_i_index]
            min_j = j_vals[min_j_index]
            
            intervals.append((min_i, min_j))

        return intervals
    

    def detect(
        self, 
        pivot_stats: np.ndarray, 
        null_distn,
        block_size = None,
        c = 2
    ):
        n = self.get_pivot_length(pivot_stats)
        if block_size is None:
            block_size = np.ceil(n**0.5)

        np.random.seed(self.seed)

        Bsamples = null_distn((self.B, n), self.vocab_size)  # simulate from exact null distn
        block_indices = np.arange(0, n, block_size).astype(int)
        block_sums = np.add.reduceat(Bsamples, block_indices, axis=1) # perform the blocked sum
        Vstats = np.abs(block_sums).max(axis=1)  # this is (B,)
        th = np.quantile(Vstats, q=(1 - self.alpha))  # find out (1-alpha) quantile
        mean_under_null = np.mean(null_distn((self.B, ), self.vocab_size))

        # Start timer
        start_time = perf_counter()

        major_intervals = self.detect_first_stage(pivot_stats, th, block_size, c)
        intervals = self.detect_second_stage(pivot_stats, major_intervals, block_size, mean_under_null)

        # end timer
        end_time = perf_counter()

        return intervals, end_time - start_time
    
