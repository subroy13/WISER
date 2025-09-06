import os, json
import numpy as np
from detections import AligatorCPPDetector, AligatorDetector


fname = "data_facebook-opt-125m_n500_gumbel.json"
with open(os.path.join('../data/output', fname), 'r') as f:
    data = json.load(f)
    f.close()

pivots = np.array(data['data'][0]['pivots'])
d1 = AligatorCPPDetector(threshold=3)
print(d1.detect(pivots))

d2 = AligatorDetector(threshold=3)
print(d2.detect(pivots))

