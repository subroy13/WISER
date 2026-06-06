import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.generic import read_json
from utils.llm import apply_human_edits_simple
from utils.metrics import get_summarized_results
from utils.prf_schemes import prf_factory
from utils.watermarking_schemes import GumbelMaxWatermark
from utils.detectors import (
    WISERDetector,
    KadaneDPDetector,
    WaterSeekerDetector,
)

model_name = "facebook-opt-125m"
contam_level = "superhigh"

ROOT_OUTPUT_PATH = "../data/output"

fname = f"data_{model_name}_n500_skipgram_human_edited_{contam_level}.json"
data = read_json(os.path.join(ROOT_OUTPUT_PATH, fname))
vocab_size = data["configuration"]["vocab_size"]
model_name = data["configuration"]["model_name"]
prf_type: str = "skipgram"
watermark_method: str = "gumbel"
TRUE_K = 2


wm_method = GumbelMaxWatermark(vocab_size, prf_factory(prf_type))
null_distn = wm_method.null_distn


current_outs = []


# rho_choices = np.arange(0.1, 1.0, 0.1).tolist()
# best_res = {}
# for rho in rho_choices:

#     def get_oracle_intervals(x):
#         d = KadaneDPDetector(vocab_size, rho=rho)
#         d_tilde = d.get_true_d(x, null_distn, data["configuration"]["intervals"])  # pass the true d
#         return d.detect(x, null_distn, max_k=TRUE_K, do_thresholding=False, custom_d=d_tilde)

#     res = get_summarized_results(data, get_oracle_intervals, n_cores=1)
#     if best_res.get("iou") is None:
#         best_res = res
#         best_res["rho"] = rho
#     elif best_res["iou"] < res["iou"]:
#         best_res = res  # update max so far
#         best_res["rho"] = rho
#     else:
#         pass  # do nothing

# best_res["method"] = "Oracle"

# print(best_res)


# --------------------
# Run WISER
def get_wiser_intervals(x):
    d = WISERDetector(vocab_size, rho=0.9)
    return d.detect(x, null_distn, block_size=20, c=2)  # use (65, 1) or (20, 1)


res = get_summarized_results(data, get_wiser_intervals, n_cores=1)
print(res)
# res["method"] = "WISER"


# def get_waterseeker_intervals(x):
#     d = WaterSeekerDetector(vocab_size)
#     return d.detect(x, null_distn)


# res = get_summarized_results(data, get_waterseeker_intervals, n_cores=1)
# res["method"] = "WaterSeeker"
# current_outs.append(res)
