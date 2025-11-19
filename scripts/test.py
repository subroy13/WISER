from typing import Any
import torch
import warnings
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from functools import partial

# import necessary functions as required
from utils.metrics import get_summarized_results, get_iou
from utils.generic import convert_token_func_to_intervals
from utils.llm import generate_llm_tokens, generate_fake_llm_tokens, unwatermarked_token_generation
from watermarking.tokens import (
    gumbel_token_generation,
    inverse_token_generation,
    pf_token_generation,
    redgreen_token_generation,
)
from watermarking.pivots import (
    null_distn_gumbel,
    null_distn_inverse,
    null_distn_pf,
    null_distn_redgreen,
    pivot_statistic_gumbel_func,
    pivot_statistic_inverse_func,
    pivot_statistic_pf_func,
    pivot_statistic_redgreen_func,
)
from watermarking.detections import (
    AligatorCPPDetector,
    WaterSeekerDetector,
    SeedBSNOTDetectorCPP,
    WISERDetector,
)


wm_token_generation = inverse_token_generation
pivot_func = pivot_statistic_inverse_func
null_distn = null_distn_inverse
gap = 100

token_generation_func = {
    "0": unwatermarked_token_generation,
    str(350 - gap): wm_token_generation,
    str(450 - gap): unwatermarked_token_generation,
    "450": wm_token_generation,
    "550": unwatermarked_token_generation,
    str(550 + gap): wm_token_generation,
    str(650 + gap): unwatermarked_token_generation,
}


vocab_size = 7500
ar_coeff = 0.0  # all token distribution is uniform
output_tokens = 1000
seed = 1234
n_repeat = 5

np.random.seed(seed)  # for reproducibility
data_gen_seeds = np.random.randint(low=0, high=1000000000, size=n_repeat)

dataset = []


# Run WISER
def get_epidemic_intervals(x):
    d = WISERDetector(vocab_size)
    return d.detect(x, null_distn=null_distn, block_size=round(np.sqrt(len(x))), c=1)


# looper = tqdm(range(n_repeat), desc=f"Generating data for simulation")
looper = range(n_repeat)
for i in looper:
    dat = generate_fake_llm_tokens(
        token_generation_func,
        pivot_func,
        prompt_tokens=50,
        out_tokens=output_tokens,
        vocab_size=vocab_size,
        ar_coeff=ar_coeff,
        initial_seed=seed,
        data_gen_seed=data_gen_seeds[i],
    )
    dataset.append(dat)
    pivot = np.array(dat["pivots"])
    intervals, _ = get_epidemic_intervals(pivot)
    print(len(intervals))

# pivot_avg = np.zeros_like(dataset[0]["pivots"])
# for i in range(len(dataset)):
#     pivot_avg += np.array(dataset[i]["pivots"])
# plt.plot(pivot_avg / len(dataset))
# plt.show()

# pivot = np.array(dataset[0]["pivots"])
# plt.plot(pivot)
# plt.show()
