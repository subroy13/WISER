from typing import List, Any, Union
import numpy as np

class Expert:
    def __init__(self):
        self.prediction = 0.0
        self.loss = 0.0
        self.count = 0
        self.weight = 0.0

class Aligator:

    def init_experts(self, n: int):
        """
        Initialize expert pool.
        Returns: pool (list of lists of Expert), pool_size (int)
        """
        count = 0
        pool: List[List[Expert]] = []
        for k in range(int(np.floor(np.log2(n))) + 1):
            stop = ((n + 1) >> k) - 1
            if stop < 1:
                break
            elist = []
            for i in range(1, stop + 1):
                e = Expert()
                elist.append(e)
                if k > 4:
                    count += 1
            pool.append(elist)
        return pool, count

    def get_awake_set(self, t: int, n: int):
        """
        Compute awake set indices for time t.
        """
        # All k values: 0,1,...,floor(log2(t))
        ks = np.arange(int(np.floor(np.log2(t))) + 1)
        i_vals = t >> ks
        cond = (((i_vals + 1) << ks) - 1 > n) | (ks <= 4)
        awake = np.where(cond, -1, i_vals)
        return awake


    def get_forecast(
            self,
            awake_set: np.ndarray, 
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
                continue
            i = idx - 1
            if pool[k][i].weight == 0:
                pool[k][i].weight = 1.0 / pool_size
                pool[k][i].prediction = prev_pred  # isotonic smoothing
            output += (pool[k][i].weight * pool[k][i].prediction)
            normalizer += pool[k][i].weight
        return output / normalizer, normalizer


    def compute_losses(
            self,
            awake_set: np.ndarray, 
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
                i = idx - 1
                loss = (y - pool[k][i].prediction) ** 2 / norm
                losses.append(loss)
        return losses


    def update_weights_and_predictions(
            self,
            awake_set: np.ndarray, 
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
            i = idx - 1
            denom += pool[k][i].weight * np.exp(-losses[k])

        # update weights and predictions
        for k, idx in enumerate(awake_set):
            if idx == -1:
                continue
            i = idx - 1
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
        self.threshold = 0
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
        current_start = 0
        current_end = -1
        for x in detect_res:
            if current_end + 1 == x:
                current_end = x  # update current end if next index is detected
            else:
                intervals.append((current_start, current_end)) # got an interval
                current_start = x

        # check for any leftover intervals
        if current_end > current_start:
            intervals.append((current_start, current_end))

        return intervals
