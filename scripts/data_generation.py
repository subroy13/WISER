import os 
from typing import Any, Union
import json
import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from llm_utilities import get_torch_device, generate_llm_tokens, unwatermarked_token_generation
from watermarking_func import (
    gumbel_token_generation, pivot_statistic_gumbel_func,
    inverse_token_generation, pivot_statistic_inverse_func,
    pf_token_generation, pivot_statistic_pf_func,
    redgreen_token_generation, pivot_statistic_redgreen_func,
    synthid_token_generation, pivot_statistic_synthid_func
)


root_data_path = "../data"
output_data_path = "../data/output"

# Read the list of prompts
def get_prompts():
    with open(os.path.join(root_data_path, "prompts_subset.txt"), "r", errors="ignore") as f:
        prompts = f.read().split("\n===\n")
        f.close()
    return prompts

def normalize_name(name: str):
    name = re.sub(r'[^A-Za-z0-9]', '-', name)
    name = re.sub(r'-+', '-', name)
    return name

def generate_watermarked_data(
    model_name: str,
    token_generation_func: dict,
    pivot_func: Any = None,
    device: Any = None,
    output_filename: Union[str, None] = None,
    prompt_tokens: int = 50,
    output_tokens: int = 200,
    batch_size: int = 8,
    max_token_input_length: int = 256,
    initial_seed: int = 1234
):
    if device is None:
        device = get_torch_device(force_cpu=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device) # type: ignore
    vocab_size = model.get_output_embeddings().weight.shape[0]
    print(f"There are {vocab_size} many words in vocabulary")
    print(f"The model {model_name} is loaded on device: {device}")

    # calculate the intervals
    intervals = []
    last_interval_type = None
    last_interval_index = None
    data_gen_type = "unwatermarked"
    for index in sorted([int(x) for x in token_generation_func.keys()], reverse = False):
        if last_interval_type is not None:
            intervals.append((last_interval_index, index, last_interval_type))
        last_interval_type = token_generation_func[str(index)].__name__.split('_')[0]
        if last_interval_type != "unwatermarked":
            data_gen_type = last_interval_type
        last_interval_index = index
    intervals.append((last_interval_index, output_tokens, last_interval_type))

    data_out_conf = {
        "model_name": model_name,
        "intervals": intervals,
        "prompt_tokens": prompt_tokens,
        "out_tokens": output_tokens,
        "vocab_size": vocab_size,
        "initial_seed": initial_seed,
        "max_token_input_length": max_token_input_length
    }
    if output_filename is None:
        output_filename = f"data_{normalize_name(model_name)}_n{output_tokens}_{data_gen_type}.json"
    
    response_list = []
    pivot_seed = initial_seed + prompt_tokens 
    prompt_list = get_prompts()
    for i in tqdm(range(0, len(prompt_list), batch_size), desc="Processing batches"):
        prompt_batch = prompt_list[i:(i+batch_size)]
        response = generate_llm_tokens(
            prompt_batch,
            tokenizer,
            model,
            token_generation_func=token_generation_func,
            verbose=False,
            out_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            vocab_size=vocab_size,
            max_token_input_length=max_token_input_length,
            batch_size=batch_size
        )
        if pivot_func is not None:
            # calculate pivot function as well
            for j in range(len(response)):
                gen_tokens = response[j]["gen_tokens"]
                response[j]["pivots"] = pivot_func(gen_tokens, seed = pivot_seed, vocab_size = vocab_size)
        response_list.extend(response)

        # save the json file
        with open(os.path.join(output_data_path, output_filename), "w") as f:
            json.dump({"configuration": data_out_conf, "data": response_list}, f)
            f.close()

    # save it at last as well
    with open(os.path.join(output_data_path, output_filename), "w") as f:
        json.dump({"configuration": data_out_conf, "data": response_list}, f)
        f.close()



# invoke the function
if __name__ == "__main__":
    device = get_torch_device()
    # torch.set_num_threads(8) # parallelize with 8 threads max

    output_tokens = 500
    model_name = "facebook/opt-125m"
    token_generation_func = {
        "0": unwatermarked_token_generation,
        "100": gumbel_token_generation,
        "200": unwatermarked_token_generation,
        "325": gumbel_token_generation,
        "400": unwatermarked_token_generation,
    }
    pivot_func = pivot_statistic_gumbel_func
    # token_generation_func = {
    #     "0": unwatermarked_token_generation
    # }
    # pivot_func = None

    generate_watermarked_data(
        model_name,
        token_generation_func,
        pivot_func,
        output_tokens=output_tokens,
        device=device,
        batch_size=4
    )
