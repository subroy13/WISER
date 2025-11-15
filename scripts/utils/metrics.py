from typing import Any, Union, List, Tuple
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# IOU is the typical metric that is tracked in segment detection scenarios
def get_iou(intervalsA: List[Tuple[int, int]], intervalsB: List[Tuple[int, int]]):
    coordsA = set([x for start, end in intervalsA for x in range(start, end)])
    coordsB = set([x for start, end in intervalsB for x in range(start, end)])
    num = len(coordsA.intersection(coordsB))
    denom = len(coordsA.union(coordsB))
    return num / denom


# hit count is essentially the numerator for calculating precision and recall
def get_hit_counts(true_intervals: List[Tuple[int, int]], estimated_intervals: List[Tuple[int, int]]):
    # count how many of the true intervals we have nonzero IOU
    hit_count = 0
    est_intervals = estimated_intervals.copy()
    for true_int in true_intervals:
        max_iou = 0
        max_index = None
        for j in range(len(est_intervals)):
            iou = get_iou([true_int], [est_intervals[j]])
            if iou > 0 and iou > max_iou:
                max_iou = iou
                max_index = j

        if max_iou > 0 and max_index is not None:
            hit_count += 1
            est_intervals.pop(max_index)

    return hit_count


def get_rand_index(intervalsA: List[Tuple[int, int]], intervalsB: List[Tuple[int, int]], n: int):
    # convert interval endpoints to sorted changepoints for fast computation
    # reference: https://arxiv.org/pdf/2112.03738
    cpA = sorted([0, n] + [x for interval in intervalsA for x in interval])
    cpB = sorted([0, n] + [x for interval in intervalsB for x in interval])
    num = 0
    for i in range(len(cpA) - 1):
        for j in range(len(cpB) - 1):
            nij = max(0, min(cpA[i + 1], cpB[j + 1]) - max(cpA[i], cpB[j]))
            num += nij * abs(cpA[i + 1] - cpB[j + 1])
    return 1 - 2 * num / (n * (n - 1))


# calculate modified rand index which avoids exchangeability between non-watermarked and watermarked intervals
def get_modified_rand_index(intervalsA: List[Tuple[int, int]], intervalsB: List[Tuple[int, int]], n):
    ri = get_rand_index(intervalsA, intervalsB, n)

    # create mask for intervalsA and intervalsB
    maskA = np.zeros((n,), dtype=np.bool)
    for s, e in intervalsA:
        maskA[s:e] = True

    maskB = np.zeros((n,), dtype=np.bool)
    for s, e in intervalsB:
        maskB[s:e] = True

    # loop through pairs in intervalA
    counter = 0
    for s, e in intervalsA:
        for i in range(s, e):
            for j in range(i + 1, e):
                if (not maskB[i]) and (not maskB[j]):
                    counter += 1
    for s, e in intervalsB:
        for i in range(s, e):
            for j in range(i + 1, e):
                if (not maskA[i]) and (not maskA[j]):
                    counter += 1

    return ri, ri - (2 * counter / (n * (n - 1)))


# metric to find the symmetric differences
def get_symmetric_difference(intA, intB):
    sA, eA = intA
    sB, eB = intB
    return abs(sA - sB) + abs(eA - eB)


# utility function to calculating watermark detection metrics
def get_summarized_results(data, get_interval_func, add_plot=False, verbose=True):
    metrics_list = []
    interval_endpoints = []
    true_intervals = [
        (start, end)
        for (start, end, interval_type) in data["configuration"]["intervals"]
        if interval_type != "unwatermarked"
    ]

    n = 0
    iterator = list(enumerate(data["data"]))
    if verbose:
        iterator = tqdm(iterator, desc=f"Detecting intervals using {get_interval_func.__name__}")
    for sample_index, sample_data in iterator:
        pivots = sample_data["pivots"]
        n = max(n, len(pivots))
        pivots = np.array(pivots)
        pivots[np.isinf(pivots)] = pivots[~np.isinf(pivots)].max()  # replace by maximum for infinite values
        est_intervals, time_taken = get_interval_func(pivots)

        # add detected endpoints to array
        for left_end, right_end in est_intervals:
            interval_endpoints.append(left_end)
            interval_endpoints.append(right_end)

        # calculate metrics
        hit_count = get_hit_counts(true_intervals, est_intervals)
        ri, mod_ri = get_modified_rand_index(true_intervals, est_intervals, n)
        metric_row = {
            "sample_index": sample_index,
            "detected_intervals_count": len(est_intervals),
            "iou": get_iou(est_intervals, true_intervals),
            "recall": hit_count / max(len(true_intervals), 1),  # Recall
            "precision": hit_count / max(len(est_intervals), 1),  # Precision
            "rand_index": ri,
            "modified_rand_index": mod_ri,
            "time": time_taken,
            "khat_exact": len(est_intervals) == len(true_intervals),
            "khat_over": len(est_intervals) > len(true_intervals),
            "khat_under": len(est_intervals) < len(true_intervals),
        }
        metrics_list.append(metric_row)

    metric_df = pd.DataFrame(metrics_list)
    # metric_df['f1'] = 2 * metric_df['precision'] * metric_df['recall'] / (metric_df['precision'] + metric_df['recall'])
    f1 = (
        2
        * metric_df["precision"].mean()
        * metric_df["recall"].mean()
        / (metric_df["precision"].mean() + metric_df["recall"].mean())
    )

    return {
        "model_name": data["configuration"]["model_name"],
        "iou": metric_df["iou"].mean(),
        "iou_lower": metric_df["iou"].quantile(q=0.025),  # get 95% lower, upper CI for IOU
        "iou_upper": metric_df["iou"].quantile(q=0.975),
        "precision": metric_df["precision"].mean(),
        "recall": metric_df["recall"].mean(),
        "f1": f1,
        "rand_index": metric_df["rand_index"].mean(),
        "rand_index_lower": metric_df["rand_index"].quantile(q=0.025),
        "rand_index_upper": metric_df["rand_index"].quantile(q=0.975),
        "modified_rand_index": metric_df["modified_rand_index"].mean(),
        "modified_rand_index_lower": metric_df["modified_rand_index"].quantile(q=0.025),
        "modified_rand_index_upper": metric_df["modified_rand_index"].quantile(q=0.975),
        "time": metric_df["time"].mean(),
        "time_lower": metric_df["time"].quantile(q=0.025),
        "time_upper": metric_df["time"].quantile(q=0.975),
        "khat_over": metric_df["khat_over"].mean(),
        "khat_under": metric_df["khat_under"].mean(),
        "khat_exact": metric_df["khat_exact"].mean(),
    }
