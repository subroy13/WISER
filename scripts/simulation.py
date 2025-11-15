from typing import Any
import torch
import warnings
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

# import necessary functions as required
from utils.metrics import get_summarized_results
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

# MAX_PROCESSES = cpu_count() - 2
MAX_PROCESSES = 8


# utility functions that runs each type of detector and gets combined result
def run_detector_methods(data, null_distn, vocab_size):
    outputs = []

    # Run WISER
    def get_epidemic_intervals(x):
        d = WISERDetector(vocab_size)
        return d.detect(x, null_distn=null_distn, block_size=int(np.sqrt(len(x))), c=1)

    res = get_summarized_results(data, get_epidemic_intervals, verbose=True)
    res["method"] = "WISER"
    res["vocab_size"] = vocab_size
    outputs.append(res)

    # Run aligator
    def get_aligator_intervals(x):
        d = AligatorCPPDetector(vocab_size)
        return d.detect(np.array(x), null_distn)

    res = get_summarized_results(data, get_aligator_intervals, verbose=True)
    res["method"] = "Aligator"
    res["vocab_size"] = vocab_size
    outputs.append(res)

    # Run seedbs
    def get_seedbs_intervals(x):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = SeedBSNOTDetectorCPP(vocab_size, n_jobs=1)
            return d.detect(x, null_distn)

    res = get_summarized_results(data, get_seedbs_intervals, verbose=True)
    res["method"] = "SeedBS_NOT"
    res["vocab_size"] = vocab_size
    outputs.append(res)

    # Run waterseeker
    def get_seek_intervals(x):
        d = WaterSeekerDetector(vocab_size)
        return d.detect(x, null_distn)

    res = get_summarized_results(data, get_seek_intervals, verbose=True)
    res["method"] = "Waterseeker"
    res["vocab_size"] = vocab_size
    outputs.append(res)

    return pd.DataFrame(outputs)


# utility function that runs the experiment n_repeat times given a setting
def run_experiment_and_collect_metrics(settings: tuple, seed=1234, n_repeat: int = 500):
    (vocab_size, ar_coeff, output_tokens, token_generation_func, pivot_func, null_distn) = settings
    intervals, data_gen_type = convert_token_func_to_intervals(token_generation_func, output_tokens)
    result_config = {"model_name": "simulation", "intervals": intervals}
    np.random.seed(seed)
    dataset = []
    for _ in tqdm(range(n_repeat), desc=f"Generating data for simulation 1 with ar_coeff = {ar_coeff}"):
        dataset.append(
            generate_fake_llm_tokens(
                token_generation_func,
                pivot_func,
                prompt_tokens=50,
                out_tokens=output_tokens,
                vocab_size=vocab_size,
                ar_coeff=ar_coeff,
                initial_seed=seed,
            )
        )

    # for each dataset, now calculate the watermarked segments
    print(f"Running watermark detections for various methods")
    detection_result = run_detector_methods(
        {
            "configuration": result_config,
            "data": dataset,
        },
        null_distn=null_distn,
        vocab_size=vocab_size,
    )
    detection_result["vocab_size"] = vocab_size
    detection_result["output_tokens"] = output_tokens
    detection_result["ar_coeff"] = ar_coeff
    detection_result["data_gen_type"] = data_gen_type
    return detection_result


########
# Run simulation setup 1 (single)
# Goal: See effect of time dependence: phi
def perform_simulation_1(seed=1234, n_repeat=500):
    # settings
    vocab_size = 1000
    ar_coeff_list = np.arange(0.0, 1.0, 0.1).tolist()
    output_tokens = 500

    token_generation_func = {
        "0": unwatermarked_token_generation,
        "50": pf_token_generation,
        "300": unwatermarked_token_generation,
    }
    pivot_func = pivot_statistic_pf_func
    null_distn = null_distn_pf

    # prepare settings for each worker
    settings_list = [
        ((vocab_size, ar_coeff, output_tokens, token_generation_func, pivot_func, null_distn), seed, n_repeat)
        for ar_coeff in ar_coeff_list
    ]
    with Pool(processes=MAX_PROCESSES) as pool:
        detection_result_list = pool.starmap(run_experiment_and_collect_metrics, settings_list)

    return pd.concat(detection_result_list)


#######
# Run simulation setup 2 (single)
# Goal: See effect of vocab_size
def perform_simulation_2(seed=1234, n_repeat=500):
    # settings
    vocab_size_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    ar_coeff = 1.0  # all token distribution is uniform
    output_tokens = 500

    token_generation_func = {
        "0": unwatermarked_token_generation,
        "50": gumbel_token_generation,
        "300": unwatermarked_token_generation,
    }
    pivot_func = pivot_statistic_gumbel_func
    null_distn = null_distn_gumbel

    # prepare settings for each worker
    settings_list = [
        ((vocab_size, ar_coeff, output_tokens, token_generation_func, pivot_func, null_distn), seed, n_repeat)
        for vocab_size in vocab_size_list
    ]
    with Pool(processes=MAX_PROCESSES) as pool:
        detection_result_list = pool.starmap(run_experiment_and_collect_metrics, settings_list)

    return pd.concat(detection_result_list)


#########
# Run simulation setup 3 (multiple)
# Goal: See effect of gaps for multiple detection
def perform_simulation_3(seed=1234, n_repeat=500):
    # settings
    vocab_size = 1000
    ar_coeff = 1.0  # all token distribution is uniform
    output_tokens = 1000
    gap_list = np.arange(10, 300, 10).tolist()

    # each interval is of length 100
    wm_token_generation = gumbel_token_generation
    null_distn = null_distn_gumbel
    pivot_func = pivot_statistic_gumbel_func
    token_generation_func_list = [
        {
            "0": unwatermarked_token_generation,
            str(350 - gap): wm_token_generation,
            str(450 - gap): unwatermarked_token_generation,
            "450": wm_token_generation,
            "550": unwatermarked_token_generation,
            str(550 + gap): wm_token_generation,
            str(650 + gap): unwatermarked_token_generation,
        }
        for gap in gap_list
    ]

    settings_list = [
        ((vocab_size, ar_coeff, output_tokens, token_generation_func, pivot_func, null_distn), seed, n_repeat)
        for token_generation_func in token_generation_func_list
    ]
    with Pool(processes=MAX_PROCESSES) as pool:
        detection_result_list = pool.starmap(run_experiment_and_collect_metrics, settings_list)

    return pd.concat(detection_result_list)


if __name__ == "__main__":
    df = perform_simulation_2(seed=1234, n_repeat=5000)
    df.to_csv("../data/simulations/simulation2_gumbel.csv", index=False)
