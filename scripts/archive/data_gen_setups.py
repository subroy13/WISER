##############
# Import packages

from typing import List, Tuple
import os
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import pandas as pd
import json




############
# List of dataset cases
###########
data_cases = [
    {
        "fname": "data_uwm_n500.json",
        "prompt_tokens": 50,
        "out_tokens": 500,
        "token_generation_func": {
            "0": unwatermarked_token_generation
        },
        "pivot": None
    },
    {
        "fname": "data_gumbel_n500.json",
        "prompt_tokens": 50,
        "out_tokens": 500,
        "token_generation_func": {
            "0": unwatermarked_token_generation,
            "100": gumbel_token_generation,
            "200": unwatermarked_token_generation,
            "400": gumbel_token_generation,
            "450": unwatermarked_token_generation,
        },
        "pivot": pivot_statistic_gumbel_func
    },
    {
        "fname": "data_inverse_n500.json",
        "prompt_tokens": 50,
        "out_tokens": 500,
        "token_generation_func": {
            "0": unwatermarked_token_generation,
            "100": inverse_token_generation,
            "200": unwatermarked_token_generation,
            "400": inverse_token_generation,
            "450": unwatermarked_token_generation,
        },
        "pivot": pivot_statistic_inverse_func
    },
    {
        "fname": "data_gumbel_n1500.json",
        "prompt_tokens": 50,
        "out_tokens": 500,
        "token_generation_func": {
            "0": unwatermarked_token_generation,
            "500": gumbel_token_generation,
            "800": unwatermarked_token_generation,
            "1150": gumbel_token_generation,
            "1300": unwatermarked_token_generation,
        },
        "pivot": pivot_statistic_gumbel_func
    },
    {
        "fname": "data_inverse_n1500.json",
        "prompt_tokens": 50,
        "out_tokens": 1500,
        "token_generation_func": {
            "0": unwatermarked_token_generation,
            "500": inverse_token_generation,
            "800": unwatermarked_token_generation,
            "1150": inverse_token_generation,
            "1300": unwatermarked_token_generation,
        },
        "pivot": pivot_statistic_inverse_func
    }
]


# Loop over dataset cases and call data generation functions
for data_case in data_cases:
    prompt_tokens = data_case.get("prompt_tokens")
    out_tokens = data_case.get("out_tokens")
    pivot_func = data_case.get("pivot")
    pivot_seed = 1234 + prompt_tokens # this is where the seed for pivot statistic will start from    
    token_generation_func_serialized = {
        k: v.__name__ for k, v in data_case.get("token_generation_func", {})
    }
    response_list = []
    for prompt in tqdm(prompt):
        response = generate_llm_tokens(
            prompt,
            tokenizer,
            model,
            token_generation_func=data_case.get("token_generation_func", {}),
            verbose=False,
            out_tokens=out_tokens,
            prompt_tokens=prompt_tokens
        )
        if pivot_func is not None:
            # calculate pivot function as well
            gen_tokens = response["gen_tokens"]
            response["pivots"] = pivot_func(gen_tokens, seed = pivot_seed)
        response_list.append(response)

    # save the JSON
    data_outfile = data_case.get("fname")
    with open(os.path.join(ROOT_DATA_PATH, data_outfile), "w") as f:
        json.dump({
            "configuration": {
                "token_generation_func": token_generation_func_serialized,
                "model_name": model_name,
                "prompt_tokens": prompt_tokens,
                "out_tokens": out_tokens,
                "vocab_size": vocab_size
            },
            "data": response_list
        }, f)
        f.close()

