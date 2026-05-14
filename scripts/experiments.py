from typing import List
import os
import numpy as np
import pandas as pd

# import the custom functions
from utils.detectors import (
    AligatorCPPDetector,
    WinMaxDetector,
    FixedWindowDetector,
    WaterSeekerDetector,
    SeedBSNOTDetectorCPP,
    WISERDetector,
    KadaneDPDetector,
)
from utils.metrics import get_summarized_results
from utils.generic import read_json
from utils.prf_schemes import prf_factory
from utils.watermarking_schemes import (
    GumbelMaxWatermark,
    InverseWatermark,
    RedGreenWatermark,
    PermuteFlipWatermark,
    BaseWatermark,
)

# define main constants
ROOT_OUTPUT_PATH = "../data/output"
N_CORES = 10
MODEL_LIST = [
    "facebook-opt-125m",
    "google-gemma-3-270m",
    "facebook-opt-1-3b",
    "princeton-nlp-Sheared-LLaMA-1-3B",
    "mistralai-Mistral-7B-v0-1",
    "meta-llama-Meta-Llama-3-8B",
]
WATERMARKING_METHODS_MAPPER = {
    "gumbel": GumbelMaxWatermark,
    "inverse": InverseWatermark,
    "redgreen": RedGreenWatermark,
    "pf": PermuteFlipWatermark,
}


def experiment_comparison(fname_suffix: str, outputfile_name: str, TRUE_K: int):
    # let's get summarized results for different detection algorithm
    all_outputs = []

    for watermark_method in WATERMARKING_METHODS_MAPPER.keys():
        for model_name in MODEL_LIST:
            current_outs = []
            fname = f"data_{model_name}_{fname_suffix}_{watermark_method}.json"
            data = read_json(os.path.join(ROOT_OUTPUT_PATH, fname))
            vocab_size = data["configuration"]["vocab_size"]
            model_name = data["configuration"]["model_name"]
            prf_type: str = data["configuration"]["prf_type"]

            print("=" * 25)
            print(f"\nLoaded file {fname} with model {model_name} and vocab size {vocab_size}\n")

            # load the watermarking method's null distribution
            wm_method: BaseWatermark = WATERMARKING_METHODS_MAPPER[watermark_method](vocab_size, prf_factory(prf_type))
            null_distn = wm_method.null_distn

            # --------------------
            # Run different algorithms
            # Run ORACLE

            rho_choices = np.arange(0.1, 1.0, 0.1).tolist()
            best_res = {}
            for rho in rho_choices:

                def get_oracle_intervals(x):
                    d = KadaneDPDetector(vocab_size, rho=rho)
                    d_tilde = d.get_true_d(x, null_distn, data["configuration"]["intervals"])  # pass the true d
                    return d.detect(x, null_distn, max_k=TRUE_K, do_thresholding=False, custom_d=d_tilde)

                res = get_summarized_results(data, get_oracle_intervals, n_cores=N_CORES)
                if best_res.get("iou") is None:
                    best_res = res
                elif best_res["iou"] < res["iou"]:
                    best_res = res  # update max so far
                else:
                    pass  # do nothing
            best_res["method"] = "Oracle"
            current_outs.append(best_res)

            # --------------------
            # Run WISER
            def get_wiser_intervals(x):
                d = WISERDetector(vocab_size)
                return d.detect(x, null_distn, block_size=20, c=1)  # use (65, 1) or (20, 1)

            res = get_summarized_results(data, get_wiser_intervals, n_cores=N_CORES)
            res["method"] = "WISER"
            current_outs.append(res)

            # --------------------
            # Run Aligator
            def get_aligator_intervals(x):
                d = AligatorCPPDetector(vocab_size)
                return d.detect(x, null_distn)

            res = get_summarized_results(data, get_aligator_intervals, n_cores=N_CORES)
            res["method"] = "Aligator"
            current_outs.append(res)

            # ----------------------
            # Run SeedBS
            def get_seedbs_intervals(x):
                d = SeedBSNOTDetectorCPP(vocab_size)
                return d.detect(x, null_distn)

            res = get_summarized_results(data, get_seedbs_intervals, n_cores=N_CORES)
            res["method"] = "SeedBS"
            current_outs.append(res)

            # -------------------
            # Run Waterseeker
            def get_waterseeker_intervals(x):
                d = WaterSeekerDetector(vocab_size)
                return d.detect(x, null_distn)

            res = get_summarized_results(data, get_waterseeker_intervals, n_cores=N_CORES)
            res["method"] = "WaterSeeker"
            current_outs.append(res)

            # --------------------
            # append the model_name and vocab size also
            for res in current_outs:
                res["model_name"] = model_name
                res["vocab_size"] = vocab_size
                res["watermark_method"] = watermark_method
                res["prf_type"] = prf_type

            all_outputs.extend(current_outs)

    all_outputs = pd.concat(all_outputs)
    all_outputs.to_csv(os.path.join("../data", outfile_name), index=False)


def experiment_kadane(fname_suffix: str, outfile_name: str, TRUE_K: int, k_list: List[int]):
    all_outputs = []

    for watermark_method in WATERMARKING_METHODS_MAPPER.keys():
        for model_name in MODEL_LIST:
            current_outs = []
            fname = f"data_{model_name}_{fname_suffix}_{watermark_method}.json"
            data = read_json(os.path.join(ROOT_OUTPUT_PATH, fname))
            vocab_size = data["configuration"]["vocab_size"]
            model_name = data["configuration"]["model_name"]
            prf_type: str = data["configuration"]["prf_type"]

            print("=" * 25)
            print(f"\nLoaded file {fname} with model {model_name} and vocab size {vocab_size}\n")

            # load the watermarking method's null distribution
            wm_method: BaseWatermark = WATERMARKING_METHODS_MAPPER[watermark_method](vocab_size, prf_factory(prf_type))
            null_distn = wm_method.null_distn

            # Run different Kadane algorithms
            # --------------------
            rho_choices = np.arange(0.1, 1.0, 0.1).tolist()
            best_res = {}
            for rho in rho_choices:

                def get_oracle_intervals(x):
                    d = KadaneDPDetector(vocab_size, rho=rho)
                    d_tilde = d.get_true_d(x, null_distn, data["configuration"]["intervals"])  # pass the true d
                    return d.detect(x, null_distn, max_k=TRUE_K, do_thresholding=False, custom_d=d_tilde)

                res = get_summarized_results(data, get_oracle_intervals, n_cores=N_CORES)
                if best_res.get("iou") is None:
                    best_res = res
                elif best_res["iou"] < res["iou"]:
                    best_res = res  # update max so far
                else:
                    pass  # do nothing
            best_res["method"] = "Oracle"
            current_outs.append(best_res)

            # --------------------
            # Run WISER
            def get_wiser_intervals(x):
                d = WISERDetector(vocab_size)
                return d.detect(x, null_distn, block_size=65, c=1)  # use (65, 1) or (20, 1)

            res = get_summarized_results(data, get_wiser_intervals, n_cores=N_CORES)
            res["method"] = "WISER"
            current_outs.append(res)

            # --------------------
            # Run Kadane's DP with thresholding
            def get_kadane_intervals(x, max_k=2 * TRUE_K):
                d = KadaneDPDetector(vocab_size)
                return d.detect(x, null_distn, max_k=max_k, do_thresholding=True)

            for k in k_list:
                res = get_summarized_results(data, lambda x: get_kadane_intervals(x, max_k=k), n_cores=N_CORES)
                res["method"] = f"Kadane_{k}"
                current_outs.append(res)

            # append the model_name and vocab size also
            for res in current_outs:
                res["model_name"] = model_name
                res["vocab_size"] = vocab_size
                res["watermark_method"] = watermark_method
                res["prf_type"] = prf_type

            all_outputs.extend(current_outs)

    all_outputs = pd.concat(all_outputs)
    all_outputs.to_csv(os.path.join("../data", outfile_name), index=False)


# data_facebook-opt-125m_n500_skipgram_human_edited_med
def experiment_editing():
    all_outputs = []

    for contam_level in ["low", "med", "high"]:
        for model_name in ["facebook-opt-125m", "princeton-nlp-Sheared-LLaMA-1-3B"]:
            current_outs = []

            # read the data
            fname = f"data_{model_name}_n500_skipgram_human_edited_{contam_level}.json"
            data = read_json(os.path.join(ROOT_OUTPUT_PATH, fname))
            vocab_size = data["configuration"]["vocab_size"]
            model_name = data["configuration"]["model_name"]
            prf_type: str = "skipgram"
            watermark_method: str = "gumbel"
            TRUE_K = 2

            print("=" * 25)
            print(f"\nLoaded file {fname} with model {model_name} and vocab size {vocab_size}\n")

            # load the watermarking method's null distribution
            wm_method: BaseWatermark = WATERMARKING_METHODS_MAPPER[watermark_method](vocab_size, prf_factory(prf_type))
            null_distn = wm_method.null_distn

            # --------------------
            # Run different algorithms
            # Run ORACLE

            rho_choices = np.arange(0.1, 1.0, 0.1).tolist()
            best_res = {}
            for rho in rho_choices:

                def get_oracle_intervals(x):
                    d = KadaneDPDetector(vocab_size, rho=rho)
                    d_tilde = d.get_true_d(x, null_distn, data["configuration"]["intervals"])  # pass the true d
                    return d.detect(x, null_distn, max_k=TRUE_K, do_thresholding=False, custom_d=d_tilde)

                res = get_summarized_results(data, get_oracle_intervals, n_cores=N_CORES)
                if best_res.get("iou") is None:
                    best_res = res
                elif best_res["iou"] < res["iou"]:
                    best_res = res  # update max so far
                else:
                    pass  # do nothing
            best_res["method"] = "Oracle"
            current_outs.append(best_res)

            # --------------------
            # Run WISER
            def get_wiser_intervals(x):
                d = WISERDetector(vocab_size)
                return d.detect(x, null_distn, block_size=65, c=1)  # use (65, 1) or (20, 1)

            res = get_summarized_results(data, get_wiser_intervals, n_cores=N_CORES)
            res["method"] = "WISER"
            current_outs.append(res)

            # --------------------
            # Run Aligator
            def get_aligator_intervals(x):
                d = AligatorCPPDetector(vocab_size)
                return d.detect(x, null_distn)

            res = get_summarized_results(data, get_aligator_intervals, n_cores=N_CORES)
            res["method"] = "Aligator"
            current_outs.append(res)

            # ----------------------
            # Run SeedBS
            def get_seedbs_intervals(x):
                d = SeedBSNOTDetectorCPP(vocab_size)
                return d.detect(x, null_distn)

            res = get_summarized_results(data, get_seedbs_intervals, n_cores=N_CORES)
            res["method"] = "SeedBS"
            current_outs.append(res)

            # -------------------
            # Run Waterseeker
            def get_waterseeker_intervals(x):
                d = WaterSeekerDetector(vocab_size)
                return d.detect(x, null_distn)

            res = get_summarized_results(data, get_waterseeker_intervals, n_cores=N_CORES)
            res["method"] = "WaterSeeker"
            current_outs.append(res)

            # --------------------
            # append the model_name and vocab size also
            for res in current_outs:
                res["model_name"] = model_name
                res["vocab_size"] = vocab_size
                res["watermark_method"] = watermark_method
                res["contam_level"] = contam_level
                res["prf_type"] = prf_type

            all_outputs.extend(current_outs)
    df = pd.DataFrame(all_outputs)
    df.to_csv(os.path.join("../data", "human_edit_experiments.csv"), index=False)


if __name__ == "__main__":
    experiment_editing()
