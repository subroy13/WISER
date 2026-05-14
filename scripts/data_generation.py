"""
This scripts allows one to generate the required data under different
watermarking schemes and different configurations
"""

from typing import Dict, Optional, List
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.watermarking_schemes import (
    BaseWatermark,
    UnWatermark,
    GumbelMaxWatermark,
    PermuteFlipWatermark,
    RedGreenWatermark,
    InverseWatermark,
)
from utils.llm import generate_llm_tokens, generate_fake_llm_tokens, get_torch_device, apply_human_edits_simple
from utils.generic import convert_token_func_to_intervals, generate_watermarking_schemes

# +++++++++++++++++
# CONSTANTS
# +++++++++++++++++
ROOT_DATA_PATH = "../data"
OUTPUT_PATH = "../data/output"
MODEL_MAPPING = {
    "facebook-opt-125m": "facebook/opt-125m",
    "google-gemma-3-270m": "google/gemma-3-270m",
    "facebook-opt-1-3b": "facebook/opt-1.3b",
    "princeton-nlp-Sheared-LLaMA-1-3B": "princeton-nlp/Sheared-LLaMA-1.3B",
    "mistralai-Mistral-7B-v0-1": "mistralai/Mistral-7B-v0.1",
    "meta-llama-Meta-Llama-3-8B": "meta-llama/Llama-3.1-8B",
}
EXPERIMENT_INTERVALS = {
    "S1": {"intervals": [0, 100, 200, 325, 400], "output_tokens": 500},
    "S2": {"intervals": [0, 100, 200, 350, 500, 700, 900, 1150, 1400, 1700, 2000], "output_tokens": 2500},
    "S3": {"intervals": [0, 290, 350, 380, 440, 470, 530, 560, 620, 650, 710], "output_tokens": 1000},
}


# +++++++++++++++++
# FUNCTIONS
# +++++++++++++++++
def get_prompts():
    with open(os.path.join(ROOT_DATA_PATH, "prompts_subset.txt"), "r", errors="ignore") as f:
        prompts = f.read().split("\n===\n")
        f.close()
    return prompts


# experimental data
def generate_watermarked_data(
    model_name: str,
    watermark_changes_list: List[int],
    watermark_method: str,
    prf_type: str,
    calculate_pivots: bool = True,
    device: Optional[str] = None,
    save_output: bool = True,
    output_filename: Optional[str] = None,
    verbose: bool = False,
    prompt_tokens: int = 50,
    output_tokens: int = 200,
    batch_size: int = 8,
    key: int = 15485863,
    add_human_edits: bool = False,
    cud_probs: tuple = (0.1, 0.1, 0.1),  # Create-update-delete probabilities
):
    """
    Generates text using a specified language model with given watermarking schemes applied
    at specified token intervals, and saves the output to a JSON file.

    Args:
        model_name (str): The Hugging Face model name or path used for generation.
        watermarking_schemes (Dict[str, BaseWatermark]): A dictionary mapping string intervals
            to the watermarking objects to be applied.
        calculate_pivots (bool, optional): Whether to compute pivot statistics for the generated
            tokens. Defaults to True.
        device (Optional[str], optional): The torch device to load the model on (e.g., 'cuda', 'cpu').
            If None, the device is auto-detected. Defaults to None.
        output_filename (Optional[str], optional): The filename for the output JSON. Auto-generated if None.
        prompt_tokens (int, optional): The number of tokens to consume as the prompt. Defaults to 50.
        output_tokens (int, optional): The total number of tokens to generate. Defaults to 200.
        batch_size (int, optional): The batch size used during generation. Defaults to 8.
        key (int, optional): Random seed/key for reproducibility and edit generation. Defaults to 15485863.
        add_human_edits (bool, optional): Whether to apply simulated human edits. Defaults to False.
        cud_probs (tuple, optional): Tuple representing (Create, Update, Delete) probabilities. Defaults to (0.1, 0.1, 0.1).
    """

    torch_device = get_torch_device() if device is None else torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAPPING[model_name])
    tokenizer.pad_token = tokenizer.eos_token  # Use same padding token
    model = AutoModelForCausalLM.from_pretrained(MODEL_MAPPING[model_name]).to(torch_device)  # type: ignore
    vocab_size = model.get_output_embeddings().weight.shape[0]
    print(f"There are {vocab_size} many words in vocabulary")
    print(f"The model {model_name} is loaded on device: {torch_device}")

    # calculate watermarking schemes
    watermarking_schemes = generate_watermarking_schemes(watermark_changes_list, watermark_method, vocab_size, prf_type)
    intervals, data_gen_type = convert_token_func_to_intervals(
        watermarking_schemes, output_tokens
    )  # calculate the intervals

    # input configurations
    data_out_conf = {
        "model_name": model_name,
        "intervals": intervals,
        "prompt_tokens": prompt_tokens,
        "out_tokens": output_tokens,
        "vocab_size": vocab_size,
        "key": key,
    }
    if output_filename is None:
        if add_human_edits:
            output_filename = f"data_{model_name}_n{output_tokens}_{data_gen_type}_human_edited.json"
        else:
            output_filename = f"data_{model_name}_n{output_tokens}_{data_gen_type}.json"

    response_list = []
    prompt_list = get_prompts()
    if verbose:
        loop = tqdm(range(0, len(prompt_list), batch_size), desc="Processing batches")
    else:
        loop = range(0, len(prompt_list), batch_size)

    for i in loop:
        prompt_batch = prompt_list[i : (i + batch_size)]
        response = generate_llm_tokens(
            prompt_batch,
            tokenizer,
            model,
            watermarking_schemes,
            verbose=False,
            out_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            vocab_size=vocab_size,
            batch_size=batch_size,
        )

        # Optional step: apply human edits
        if add_human_edits:
            for j in range(len(response)):
                gen_tokens = response[j]["gen_tokens"]
                response[j]["gen_tokens"] = apply_human_edits_simple(gen_tokens, vocab_size, cud_probs, seed=key + j)

        # find from the watermarking scheme, which pivot statistic should be used
        wm_method = None
        for k, wm_obj in watermarking_schemes.items():
            if wm_obj.__class__.__name__ != UnWatermark.__name__:
                wm_method = wm_obj
                break

        # create pivot statistic for watermarking method
        if calculate_pivots and wm_method is not None:
            for j in range(len(response)):
                x = torch.Tensor(response[j]["gen_tokens"]).to(torch_device).long()
                response[j]["pivots"] = wm_method.get_pivot_statistic(x)

        response_list.extend(response)

        # save the json file
        if save_output:
            with open(os.path.join(OUTPUT_PATH, output_filename), "w") as f:
                json.dump({"configuration": data_out_conf, "data": response_list}, f)
                f.close()

    # save it at last as well
    if save_output:
        with open(os.path.join(OUTPUT_PATH, output_filename), "w") as f:
            json.dump({"configuration": data_out_conf, "data": response_list}, f)
            f.close()

    return response_list


# +++++++++++++++++++
# Experiments data generating functions
# ++++++++++++++++++
if __name__ == "__main__":
    # Modify only these
    model_name = "princeton-nlp-Sheared-LLaMA-1-3B"
    settings = EXPERIMENT_INTERVALS["S1"]
    watermark_method = "gumbel"
    prf_type = "skipgram"

    watermark_changes_list = settings["intervals"]
    output_tokens = settings["output_tokens"]
    output_data = generate_watermarked_data(
        model_name,
        watermark_changes_list,
        watermark_method=watermark_method,
        prf_type=prf_type,
        calculate_pivots=True,
        output_tokens=output_tokens,
        add_human_edits=True,
        cud_probs=(0.05, 0.05, 0.05),
        verbose=True,
        batch_size=4,
        output_filename=f"data_{model_name}_n{output_tokens}_{prf_type}_human_edited_low.json",
        save_output=True,
    )

    rows = []
    for i, elem in enumerate(output_data):
        for j, x in enumerate(elem["pivots"]):
            rows.append({"repeat": i, "token_index": j, "pivot": x})

    df = pd.DataFrame(rows)

    sns.lineplot(data=df, x="token_index", y="pivot", estimator="mean")
    plt.show()
