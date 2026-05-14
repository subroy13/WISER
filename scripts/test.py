import os
import json
import numpy as np
import pandas as pd

from utils.generic import read_json
from utils.llm import apply_human_edits_simple

# fpath = "data_facebook-opt-125m_n500_skipgram_human_edited_high.json"
# data = read_json(os.path.join("../data/output", fpath))

# for i in range(5):
#     print(len(data["data"][i]["gen_tokens"]), len(data["data"][i]["pivots"]))

gen_tokens = np.random.randint(0, 10000, size=500).tolist()
print(len(gen_tokens))

out = apply_human_edits_simple(gen_tokens, vocab_size=10000, seed=1234, cud_probs=(0.2, 0.2, 0.2))
print(len(out))
