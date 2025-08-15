import torch
import numpy as np
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_utilities import generate_llm_tokens, get_torch_device, unwatermarked_token_generation
from watermarking_func import (
    gumbel_token_generation, inverse_token_generation,
    pivot_statistic_gumbel_func, pivot_statistic_inverse_func
)

###################
# Constants
model_name = "facebook/opt-125m"
root_data_path = "../data"
output_data_path = "../data/output"
data_configuration = {
    "fname": "data_uwm_n500.json",
    "prompt_tokens": 50,
    "out_tokens": 500,
    "token_generation_func": {
        "0": unwatermarked_token_generation
    },
    "pivot": None
}
##############


# Setup code
# load the model and tokenizers
device = get_torch_device()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

vocab_size = model.get_output_embeddings().weight.shape[0]
print(f"There are {vocab_size} many words in vocabulary")
print(f"The model {model_name} is loaded on device: {device}")


# Utility functions
def get_prompts():
    with open(os.path.join(root_data_path, "prompts.txt"), "r", errors="ignore") as f:
        prompts = f.read().split("\n===\n")
    return prompts


# Run the main simulation loop
prompt_tokens: int = data_configuration.get("prompt_tokens", 0)
out_tokens: int = data_configuration.get("out_tokens", 0)
pivot_func = data_configuration.get("pivot")
pivot_seed = 1234 + prompt_tokens  # this is where the seed for pivot statistic will start from    
token_generation_func_serialized = {
    k: v.__name__ for k, v in data_configuration.get("token_generation_func", {}).items()
}
    
data_list = []
prompts_list = get_prompts()
for prompt in tqdm(prompts_list[:10]):
    response = generate_llm_tokens(
        prompt,
        tokenizer,
        model,
        token_generation_func=data_configuration.get("token_generation_func", {}),
        verbose=False,
        out_tokens=out_tokens,
        prompt_tokens=prompt_tokens
    )
    if pivot_func is not None:
        # calculate pivot function as well
        gen_tokens = response["gen_tokens"]
        response["pivots"] = pivot_func(gen_tokens, seed = pivot_seed)
    data_list.append(response)

# save the JSON
data_outfile = data_configuration.get("fname", "data.json")
with open(os.path.join(output_data_path, data_outfile), "w") as f:
    json.dump({
        "configuration": {
            "token_generation_func": token_generation_func_serialized,
            "model_name": model_name,
            "prompt_tokens": prompt_tokens,
            "out_tokens": out_tokens,
            "vocab_size": vocab_size
        },
        "data": data_list
    }, f)
    f.close()

