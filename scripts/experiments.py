import os
import numpy as np
import pandas as pd


# import the custom functions
from watermarking.detections import (
    AligatorCPPDetector,
    WinMaxDetector,
    FixedWindowDetector,
    WaterSeekerDetector,
    SeedBSNOTDetectorCPP,
    WISERDetector,
    KadaneDPDetector,
)
from watermarking.pivots import null_distn_gumbel, null_distn_inverse, null_distn_pf, null_distn_redgreen
from utils.generic import read_json
from utils.metrics import get_summarized_results, get_iou, get_rand_index, get_modified_rand_index

# define main constants
ROOT_OUTPUT_PATH = "../data/output"
N_CORES = 10

null_dist_list = {
    "gumbel": null_distn_gumbel,
    "inverse": null_distn_inverse,
    "redgreen": null_distn_redgreen,
    "pf": null_distn_pf,
}
model_list = [
    "facebook-opt-125m",
    "google-gemma-3-270m",
    "facebook-opt-1-3b",
    "princeton-nlp-Sheared-LLaMA-1-3B",
    "mistralai-Mistral-7B-v0-1",
    "meta-llama-Meta-Llama-3-8B",
]


fname_suffix = "n1000"  # MODIFY here to change the suffix / simulation scenario
outfile_name = "n1000_detections.csv"
TRUE_K = 5


# let's get summarized results for different detection algorithm
all_outputs = []
for watermark_method, null_distn in null_dist_list.items():
    outputs = []
    for model_name in model_list:
        current_outs = []
        fname = f"data_{model_name}_{fname_suffix}_{watermark_method}.json"
        data = read_json(os.path.join(ROOT_OUTPUT_PATH, fname))
        vocab_size = data["configuration"]["vocab_size"]
        model_name = data["configuration"]["model_name"]
        print("=" * 25)
        print(f"\nLoaded file {fname} with model {model_name} and vocab size {vocab_size}\n")

        # Run different algorithms

        # --------------------
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
        # Run Kadane's DP with thresholding
        def get_kadane_intervals(x):
            d = KadaneDPDetector(vocab_size)
            return d.detect(x, null_distn, max_k=2 * TRUE_K, do_thresholding=True)

        res = get_summarized_results(data, get_kadane_intervals, n_cores=N_CORES)
        res["method"] = "Kadane"
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

        outputs.extend(current_outs)

    output_df = pd.DataFrame(outputs)
    output_df["watermark_method"] = watermark_method
    all_outputs.append(output_df)

all_outputs = pd.concat(all_outputs)
all_outputs.to_csv(os.path.join("../data", outfile_name), index=False)

# print(all_outputs)
