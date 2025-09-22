# WISER: Watermark Identification via Segmenting Epidemic Regions

This repository contains the official implementation for the paper: **WISER: Watermark Identification via Segmenting Epidemic Regions**.

## Abstract

> With the increasing popularity of large language models, concerns over content authenticity have led to the development of myriad watermarking schemes. These schemes can be used to detect a machine-generated text via an appropriate key, while being imperceptible to readers with no such key. The corresponding detection mechanisms usually take the form of statistical hypothesis testing for the existence of watermarks, spurring extensive research in this direction. However, the finer-grained problem of identifying which segments of a mixed-source text are actually watermarked, is much less explored; the existing approaches either lack scalability or theoretical guarantees robust to paraphrase and post-editing. In this work, we introduce an unique perspective to such watermark segmentation problems through the lens of *epidemic change-points*. By highlighting the similarities as well as differences of these two problems, we motivate and propose `WISER`: a novel, computationally efficient, watermark segmentation algorithm. We validate our algorithm by deriving finite sample error-bounds, and establishing its consistency in detecting multiple watermarked segments in a single text. Complementing these theoretical results, our extensive numerical experiments show that `WISER` outperforms state-of-the-art baseline methods, both in terms of computational speed as well as accuracy, on various benchmark datasets embedded with diverse watermarking schemes. Our theoretical and empirical findings establish `WISER` as an effective tool for watermark localization in most settings. It also shows how insights from a classical statistical problem can be developed into a theoretically valid and computationally efficient solution of a modern and pertinent problem.

## Repository Structure

├── data/ # Datasets used for experiments 
├── figures/ # Figures and plots from the paper 
├── scripts/ # Scripts to run experiments and reproduce results 
├── readme.md # This README file 
└── ... # Other source code and utility files

## Installation

1.  Clone the repository:
```bash
git clone https://github.com/your-username/watermarked-llm-detection.git
cd watermarked-llm-detection
```

2.  It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```
3.  Install the required Python packages:
```bash
numpy
pandas
pytorch
datasets
tokenizers
transformers
```

4.  Compile C++ extensions for baseline algorithms. For example, to compile `aligator` on macOS:

```bash
+    c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` aligator.cpp -o aligator`python3-config --extension-suffix`
```

Please see the `scripts` directory for other compilation instructions if available.

## Usage

The core `WISER` algorithm can be used to identify watermarked segments in a given text.

```python
import numpy as np
from detections import WISERDetector
from watermarking_func import null_distn_gumbel

vocab_size = 10000
x = np.random.exponential(500)
d = WISERDetector(vocab_size)
d.detect(x, null_distn=null_distn_gumbel, block_size=65, c = 1) 
```

## Reproducing Results

The scripts to reproduce the experiments and figures from the paper are located in the `scripts/` directory. To run the main experiments, you can look at the `scripts/experiments.ipynb` jupyter notebook. Please refer to the comments within the scripts for more detailed options and configurations.


## Datasets

The datasets used in our experiments are located in the `data/` directory. These include benchmark datasets embedded with various watermarking schemes.

<!-- If you find our work useful in your research, please add a star to  -->


