from typing import Any
import torch
import warnings
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from functools import partial
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
MAX_PROCESSES = 12
WATERMARKING_METHODS = {
    "gumbel": (gumbel_token_generation, pivot_statistic_gumbel_func, null_distn_gumbel),
    "inverse": (inverse_token_generation, pivot_statistic_inverse_func, null_distn_inverse),
    "redgreen": (redgreen_token_generation, pivot_statistic_redgreen_func, null_distn_redgreen),
    "pf": (pf_token_generation, pivot_statistic_pf_func, null_distn_pf),
}


# utility function that runs the experiment n_repeat times given a setting
def run_experiment_and_collect_metrics(
    settings: tuple, seed=1234, n_repeat: int = 500, sim_name="simulation", additional_values_to_keep={}
) -> dict:
    (vocab_size, ar_coeff, output_tokens, token_generation_func, pivot_func, null_distn) = settings
    intervals, data_gen_type = convert_token_func_to_intervals(token_generation_func, output_tokens)
    result_config = {"model_name": sim_name, "intervals": intervals}

    # generate data_gen seeds
    np.random.seed(seed)  # for reproducibility
    data_gen_seeds = np.random.randint(low=0, high=1000000000, size=n_repeat)

    dataset = []
    for i in tqdm(range(n_repeat), desc=f"Generating data for {sim_name}"):
        dataset.append(
            generate_fake_llm_tokens(
                token_generation_func,
                pivot_func,
                prompt_tokens=50,
                out_tokens=output_tokens,
                vocab_size=vocab_size,
                ar_coeff=ar_coeff,
                initial_seed=seed,
                data_gen_seed=data_gen_seeds[i],
            )
        )

    # for each dataset, now calculate the watermarked segments

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
        verbose=True,
    )

    # track additional config values
    detection_result["vocab_size"] = vocab_size
    detection_result["output_tokens"] = output_tokens
    detection_result["watermark_type"] = data_gen_type
    detection_result["seed"] = seed
    for key in additional_values_to_keep:
        detection_result[key] = additional_values_to_keep[key]

    return detection_result


######
# Run simulation setup 1 (single)
# Goal: See effect of time dependence: phi
def perform_simulation_1(seed=1234, n_repeat=500, watermarking_method_list=["gumbel"]):
    # settings
    vocab_size = 1000
    ar_coeff_list = np.arange(0.1, 1.1, 0.1).tolist()  # from [0, 0.1, ...., 0.9, 1]
    output_tokens = 500

    token_generation_func_list = []
    pivot_func_list = []
    null_distn_list = []
    for watermarking_method in watermarking_method_list:
        token_fun, pivot_func, null_distn = WATERMARKING_METHODS[watermarking_method]
        token_generation_func_list.append(
            {
                "0": unwatermarked_token_generation,
                "50": token_fun,
                "300": unwatermarked_token_generation,
            }
        )
        pivot_func_list.append(pivot_func)
        null_distn_list.append(null_distn)

    # prepare settings for each worker
    settings_list = []
    for ar_coeff in ar_coeff_list:
        for token_generation_func, pivot_func, null_distn in zip(
            token_generation_func_list, pivot_func_list, null_distn_list
        ):
            settings_list.append(
                (
                    (vocab_size, ar_coeff, output_tokens, token_generation_func, pivot_func, null_distn),
                    seed,
                    n_repeat,
                    "simulation 1",
                    {"ar_coeff": ar_coeff},  # keep this additional value for tracking
                )
            )

    with Pool(processes=MAX_PROCESSES) as pool:
        detection_result_list = pool.starmap(run_experiment_and_collect_metrics, settings_list)

    return pd.DataFrame(detection_result_list)


########
# Run simulation setup 2 (single)
# Goal: See the effect of varying text sample size, and varying delta
def perform_simulation_2(seed=1234, n_repeat=500):
    # settings
    vocab_size = 1000
    ar_coeff = 0.0  # NTP are independently distributed spikes
    output_tokens_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    redgreen_delta_list = [1.5, 2.0, 2.5, 3.0, 3.5]
    redgreen_token_gen_func_list = []
    for i in range(len(redgreen_delta_list)):
        fun = partial(redgreen_token_generation, delta=redgreen_delta_list[i])
        redgreen_token_gen_func_list.append(fun)

    # prepare settings for each configuration
    settings_list = []
    for output_tokens in output_tokens_list:
        for i, wm_token_generation in enumerate(redgreen_token_gen_func_list):
            token_generation_func = {
                "0": unwatermarked_token_generation,
                str(int(0.3 * output_tokens)): wm_token_generation,
                str(int(0.7 * output_tokens)): unwatermarked_token_generation,  # 0.3t - 0.7t
            }
            settings_list.append(
                (
                    (
                        vocab_size,
                        ar_coeff,
                        output_tokens,
                        token_generation_func,
                        pivot_statistic_redgreen_func,
                        null_distn_redgreen,
                    ),
                    seed,
                    n_repeat,
                    "simulation 2",
                    {"delta": redgreen_delta_list[i]},  # keep this additional value for tracking
                )
            )

    with Pool(processes=MAX_PROCESSES) as pool:
        detection_result_list = pool.starmap(run_experiment_and_collect_metrics, settings_list)

    return pd.DataFrame(detection_result_list)


######
# Run simulation setup 2b (single)
# Goal: See the effect of gumbel strength
def perform_sim_2b_single(
    settings: tuple, seed=1234, n_repeat: int = 500, sim_name="simulation2b", additional_values_to_keep={}
):
    (vocab_size, output_tokens, d, true_intervals) = settings
    result_config = {"model_name": sim_name, "intervals": true_intervals}

    np.random.seed(seed)  # for reproducibility
    data_gen_seeds = np.random.randint(low=0, high=1000000000, size=n_repeat)
    dataset = []

    for i in tqdm(range(n_repeat), desc=f"Generating data for {sim_name}"):
        np.random.seed(data_gen_seeds[i])
        pivot = np.random.exponential(scale=1, size=output_tokens)
        for true_interval in true_intervals:
            left, right, _ = true_interval
            pivot[left:right] = np.random.normal(loc=1 + d, scale=1, size=right - left)
        dataset.append({"pivots": pivot.tolist()})

    # for each dataset, now calculate the watermarked segments

    def get_epidemic_intervals(x):
        # Run WISER
        d = WISERDetector(vocab_size)
        return d.detect(x, null_distn=null_distn_gumbel, block_size=round(np.sqrt(len(x))), c=2)

    detection_result = get_summarized_results(
        {
            "configuration": result_config,
            "data": dataset,
        },
        get_epidemic_intervals,
        verbose=True,
    )

    # track additional config values
    detection_result["vocab_size"] = vocab_size
    detection_result["output_tokens"] = output_tokens
    detection_result["seed"] = seed
    for key in additional_values_to_keep:
        detection_result[key] = additional_values_to_keep[key]

    return detection_result


def perform_simulation_2b(seed=1234, n_repeat=500):
    vocab_size = 1000
    output_tokens_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    d_list = np.arange(0.5, 5.5, 0.5).tolist()

    # prepare settings list
    settings_list = []
    for output_tokens in output_tokens_list:
        for d in d_list:
            settings_list.append(
                (
                    (
                        vocab_size,
                        output_tokens,
                        d,
                        [(int(0.3 * output_tokens), int(0.7 * output_tokens), "gumbelsim")],
                    ),
                    seed,
                    n_repeat,
                    "simulation 2B",
                    {"d": d},
                )
            )
    with Pool(processes=MAX_PROCESSES) as pool:
        detection_result_list = pool.starmap(perform_sim_2b_single, settings_list)

    return pd.DataFrame(detection_result_list)


#########
# Run simulation setup 3 (multiple)
# Goal: See effect of gaps for multiple detection
def perform_simulation_3(seed=1234, n_repeat=500):
    # settings
    vocab_size = 1000
    ar_coeff = 0.0  # NTP are independently distributed spikes

    output_tokens_list = [512, 1024, 2048, 4096, 8192]
    gap_prop_list = np.arange(0.1, 1.1, 0.05).tolist()

    wm_token_generation = gumbel_token_generation
    pivot_func = pivot_statistic_gumbel_func
    null_distn = null_distn_gumbel

    # create the settings
    settings_list = []
    for output_tokens in output_tokens_list:
        for gap_prop in gap_prop_list:
            gap = round(max(min(0.3 * output_tokens, output_tokens**gap_prop), 2))

            # each patch is ~ 0.1n length
            token_gen_func = {
                "0": unwatermarked_token_generation,
                str(int(0.35 * output_tokens) - gap): wm_token_generation,
                str(int(0.45 * output_tokens) - gap): unwatermarked_token_generation,
                str(int(0.45 * output_tokens)): wm_token_generation,
                str(int(0.55 * output_tokens)): unwatermarked_token_generation,
                str(int(0.55 * output_tokens) + gap): wm_token_generation,
                str(int(0.65 * output_tokens) + gap): unwatermarked_token_generation,
            }

            settings_list.append(
                (
                    (vocab_size, ar_coeff, output_tokens, token_gen_func, pivot_func, null_distn),
                    seed,
                    n_repeat,
                    "simulation 3",
                    {"gap": gap, "gap_prop": gap_prop},  # keep this additional value for tracking
                )
            )

    with Pool(processes=MAX_PROCESSES) as pool:
        detection_result_list = pool.starmap(run_experiment_and_collect_metrics, settings_list)

    return pd.DataFrame(detection_result_list)


if __name__ == "__main__":
    # df = perform_simulation_1(seed=1234, n_repeat=2000, watermarking_method_list=list(WATERMARKING_METHODS.keys()))
    # df = perform_simulation_2(seed=1234, n_repeat=2000)
    # df = perform_simulation_2b(seed=1234, n_repeat=2000)
    df = perform_simulation_3(seed=1234, n_repeat=2000)
    df.to_csv("../data/simulations/simulation3_all_WISER.csv", index=False)
