import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from watermarking_func import (
    unwatermarked_token_generation,
    gumbel_token_generation,
    pivot_statistic_gumbel_func,
    inverse_token_generation,
    inverse_pivot_statistic_func,
)


######################
# CONSTANTS TO ADJUST
model_name = "facebook/opt-125m"

######################

# load the model and tokenizers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
