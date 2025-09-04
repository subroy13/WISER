# Alligator
from typing import List, Any, Union, Tuple
import numpy as np

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

    def __init__(self, threshold):
        self.threshold = threshold
        self.prev_pred = 0

    def detect(self, pivot: np.ndarray):
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
        detect_res = np.where(alig > self.threshold)[0]

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
        return intervals
    


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
    
