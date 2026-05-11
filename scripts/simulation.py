"""
This details the different simulation exercises
performed in the paper.
"""

from typing import Dict, Optional, List
from multiprocessing import Pool, cpu_count
import warnings
import os
import json
import numpy as np
import pandas as pd
from torch import Tensor
from tqdm.auto import tqdm

from utils.llm import generate_fake_llm_tokens, apply_human_edits_simple
from utils.metrics import get_summarized_results
from utils.detectors import WISERDetector
from utils.prf_schemes import prf_factory
from utils.watermarking_schemes import BaseWatermark, UnWatermark, GumbelMaxWatermark
from utils.generic import convert_token_func_to_intervals, generate_watermarking_schemes

# +++++++++++++++++
# CONSTANTS
# +++++++++++++++++
ROOT_DATA_PATH = "../data"
OUTPUT_PATH = "../data/output"
MAX_PROCESSES = 12


# +++++++++++++++++
# FUNCTIONS
# +++++++++++++++++
def simulation_worker(
    vocab_size: int,
    ar_coeff: float,
    watermarked_intervals: List[int],
    watermarking_method: str,
    prf_type: str,
    prompt_tokens: int = 50,
    output_tokens: int = 200,
    sim_name: str = "simulation",
    additional_values_to_keep={},
    kwargs={},
    seed=1234,
    num_repeats: int = 500,
):
    watermarking_scheme = generate_watermarking_schemes(
        watermarked_intervals, watermarking_method, vocab_size, prf_type=prf_type, **kwargs
    )
    intervals, data_gen_type = convert_token_func_to_intervals(watermarking_scheme, output_tokens)

    dataset: list[dict] = []
    np.random.seed(seed)  # for reproducibility
    data_gen_seeds = np.random.randint(low=0, high=1000000000, size=num_repeats)

    # find from the watermarking scheme, which pivot statistic should be used
    wm_method = None
    for k, wm_obj in watermarking_scheme.items():
        if wm_obj.__class__.__name__ != UnWatermark.__name__:
            wm_method = wm_obj
            break

    for i in range(num_repeats):
        response = generate_fake_llm_tokens(
            watermarking_scheme,
            verbose=False,
            prompt_tokens=prompt_tokens,
            out_tokens=output_tokens,
            vocab_size=vocab_size,
            ar_coeff=ar_coeff,
            seed=data_gen_seeds[i],
        )

        # create pivot statistic for watermarking method
        if wm_method is not None:
            response["pivots"] = wm_method.get_pivot_statistic(Tensor(response["gen_tokens"]))

        dataset.append(response)

    # run the detection algorithm
    def get_epidemic_intervals(x):
        # Run WISER
        d = WISERDetector(vocab_size)
        return d.detect(x, null_distn=wm_method.null_distn, block_size=round(np.sqrt(len(x))), c=2)

    detection_result = get_summarized_results(
        {
            "configuration": {
                "model_name": sim_name,
                "intervals": intervals,
            },
            "data": dataset,
        },
        get_epidemic_intervals,
    )
    # track additional config values
    detection_result["vocab_size"] = vocab_size
    detection_result["output_tokens"] = output_tokens
    detection_result["watermark_type"] = data_gen_type
    detection_result["seed"] = seed
    for key in additional_values_to_keep:
        detection_result[key] = additional_values_to_keep[key]

    return detection_result


# -----------------------
# Run simulation setup 1 (single)
# Goal: See effect of time dependence: phi
def perform_simulation_1(seed=1234, save: bool = True):
    # settings
    ar_coeff_list = np.arange(0.1, 1.1, 0.1).tolist()

    # prepare settings for each worker
    settings_list = []
    for ar_coeff in ar_coeff_list:
        settings_list.append(
            (
                1000,  # vocab_size
                ar_coeff,
                [0, 50, 300],  # watermark changes
                "gumbel",  # watermark method
                "counter",  # prf_type
                50,  # prompt_tokens
                500,  # output_tokens
                "simulation1",
                {"ar_coeff": ar_coeff},  # keep this additional value for tracking
                {},  # kwargs
                seed,
                500,  # number of repeats
            )
        )

    with Pool(processes=MAX_PROCESSES) as pool:
        detection_result_list = pool.starmap(simulation_worker, settings_list)

    df = pd.DataFrame(detection_result_list)

    # store the simulation result
    if save:
        df.to_csv("../data/simulations/simulation1.csv", index=False)

    return df


# -----------------------
# Run simulation setup 2A (single)
# Goal: See the effect of varying text sample size, and varying delta
def perform_simulation_2a(seed=1234, save: bool = True):
    # settings
    ar_coeff = 0.0  # NTP are independently distributed spikes
    output_tokens_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    redgreen_delta_list = [1.5, 2.0, 2.5, 3.0, 3.5]

    settings_list = []
    for output_tokens in output_tokens_list:
        for redgreen_delta in redgreen_delta_list:
            settings_list.append(
                (
                    1000,  # vocab_size
                    ar_coeff,
                    [0, int(0.3 * output_tokens), int(0.7 * output_tokens)],
                    "redgreen",  # watermark method
                    "counter",  # prf_type
                    50,  # prompt_tokens
                    500,  # output_tokens
                    "simulation2a",
                    {"delta": redgreen_delta},  # additional values to keep
                    {"delta": redgreen_delta},  # kwargs
                    seed,
                    500,  # number of repeats
                )
            )
    with Pool(processes=MAX_PROCESSES) as pool:
        detection_result_list = pool.starmap(simulation_worker, settings_list)

    df = pd.DataFrame(detection_result_list)

    # store the simulation result
    if save:
        df.to_csv("../data/simulations/simulation2a.csv", index=False)

    return df


# -----------------------
# Run simulation setup 2B (single)
# Goal: See the effect of gumbel strength
def simulation_2b_worker(
    vocab_size: int,
    output_tokens: int,
    d: float,
    watermarked_intervals: List[int],
    sim_name: str = "simulation",
    additional_values_to_keep={},
    kwargs={},
    seed=1234,
    num_repeats: int = 500,
):
    watermarking_scheme = generate_watermarking_schemes(
        watermarked_intervals, "gumbel", vocab_size, prf_type="counter", **kwargs
    )
    intervals, data_gen_type = convert_token_func_to_intervals(watermarking_scheme, output_tokens)
    result_config = {"model_name": sim_name, "intervals": intervals}

    np.random.seed(seed)  # for reproducibility
    data_gen_seeds = np.random.randint(low=0, high=1000000000, size=num_repeats)
    dataset = []

    for i in range(num_repeats):
        np.random.seed(data_gen_seeds[i])

        # generate pivots directly
        pivot = np.random.exponential(scale=1, size=output_tokens)
        for true_interval in intervals:
            left, right, _ = true_interval
            pivot[left:right] = np.random.normal(loc=1 + d, scale=1, size=right - left)
        dataset.append({"pivots": pivot.tolist()})

    # create a gumbel watermarking object
    null_distn = None
    for k, v in watermarking_scheme.items():
        if v.__class__.__name__ == "GumbelMaxWatermark":
            null_distn = v.null_distn
            break

    def get_epidemic_intervals(x):
        # Run WISER
        d = WISERDetector(vocab_size)
        return d.detect(x, null_distn=null_distn, block_size=round(np.sqrt(len(x))), c=2)

    detection_result = get_summarized_results(
        {
            "configuration": result_config,
            "data": dataset,
        },
        get_epidemic_intervals,
    )

    # track additional config values
    detection_result["vocab_size"] = vocab_size
    detection_result["output_tokens"] = output_tokens
    detection_result["seed"] = seed
    for key in additional_values_to_keep:
        detection_result[key] = additional_values_to_keep[key]

    return detection_result


def perform_simulation_2b(seed=1234, save: bool = True):
    # settings
    output_tokens_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    d_list = np.arange(0.5, 5.5, 0.5).tolist()

    # prepare settings list
    settings_list = []
    for output_tokens in output_tokens_list:
        for d in d_list:
            settings_list.append(
                (
                    1000,  # vocab_size
                    output_tokens,
                    d,
                    [0, int(0.3 * output_tokens), int(0.7 * output_tokens)],
                    "simulation2b",
                    {"d": d, "output_tokens": output_tokens},
                    {},  # kwargs
                    seed,
                    500,  # number of repeats
                )
            )

    with Pool(processes=MAX_PROCESSES) as pool:
        detection_result_list = pool.starmap(simulation_2b_worker, settings_list)

    df = pd.DataFrame(detection_result_list)

    # store the simulation result
    if save:
        df.to_csv("../data/simulations/simulation2b.csv", index=False)

    return df


# -----------------------
# Run simulation setup 3
# Goal: See effect of gaps for multiple detection
def perform_simulation_3(seed=1234, save: bool = True):
    # settings
    output_tokens_list = [512, 1024, 2048, 4096, 8192]
    gap_prop_list = np.arange(0.1, 1.1, 0.05).tolist()

    settings_list = []
    for output_tokens in output_tokens_list:
        for gap_prop in gap_prop_list:
            gap = round(max(min(0.3 * output_tokens, output_tokens**gap_prop), 2))

            # each patch is ~ 0.1n length
            watermarked_intervals = [
                0,
                int(0.35 * output_tokens) - gap,
                int(0.45 * output_tokens) - gap,
                int(0.45 * output_tokens),
                int(0.55 * output_tokens),
                int(0.55 * output_tokens) + gap,
                int(0.65 * output_tokens) + gap,
            ]

            settings_list.append(
                (
                    1000,  # vocab_size
                    0.0,  # ar_coeff
                    watermarked_intervals,
                    "gumbel",  # watermark method
                    "counter",  # prf_type
                    50,  # prompt_tokens
                    output_tokens,  # output_tokens
                    "simulation3",
                    {"gap": gap, "gap_prop": gap_prop},  # additional values to keep
                    {},  # kwargs
                    seed,
                    500,  # number of repeats
                )
            )

    with Pool(processes=MAX_PROCESSES) as pool:
        detection_result_list = pool.starmap(simulation_worker, settings_list)

    df = pd.DataFrame(detection_result_list)

    # store the simulation result
    if save:
        df.to_csv("../data/simulations/simulation3.csv", index=False)

    return df


if __name__ == "__main__":
    # df = perform_simulation_3()