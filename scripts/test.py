import numpy as np
import os
from utils.generic import read_json
from watermarking.pivots import null_distn_gumbel
from watermarking.detections import KadaneDPDetector, KadaneGreeyDetector, WISERDetector
from utils.metrics import get_summarized_results, get_iou, get_rand_index, get_modified_rand_index
import matplotlib.pyplot as plt


# generate diagram
fname = "data_facebook-opt-125m_n500_gumbel.json"
data = read_json(os.path.join("../data/output", fname))

print(data["configuration"])

vocab_size = data["configuration"]["vocab_size"]
model_name = data["configuration"]["model_name"]

x = np.array(data["data"][0]["pivots"])
d2 = KadaneDPDetector(vocab_size)
out = d2.detect(x, null_distn=null_distn_gumbel, max_k=2)
print(out)

# plt.plot(x)
# plt.show()
