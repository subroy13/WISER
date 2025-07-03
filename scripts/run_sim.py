import torch
import numpy as np
import pickle
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from watermarking_func import (
    gumbel_token_generation,
    pivot_statistic_gumbel_func,
    inverse_token_generation,
    inverse_pivot_statistic_func,
)
from detection_func import get_wm_gumbel_interval
from llm_utilities import unwatermarked_token_generation, generate_llm_tokens

###################
# Constants
model_name = "facebook/opt-125m"

# Setup code
# load the model and tokenizers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

vocab_size = model.get_output_embeddings().weight.shape[0]
print(f"There are {vocab_size} many words in vocabulary")
print(f"The device is {device}")


# Utility functions
def get_prompts():
    with open("./data/prompts.txt", "r", errors="ignore") as f:
        prompts = f.read().split("\n===\n")
    return prompts


def run_simulation_loop(sim_func, outfile="./response_list.pkl", B=-1):
    prompts = get_prompts()
    if B < 0:
        B = len(prompts)
    else:
        B = min(B, len(prompts))
    response_list = []
    enum_list = list(enumerate(prompts[:B]))
    for i, prompt in tqdm(enum_list):
        response = sim_func(prompt, tokenizer, model)
        response_list.append(response)

        # save the results
        if i % 10 == 0:
            with open(outfile, "wb") as f:
                pickle.dump(response_list, f)

    # save the results finally
    with open(outfile, "wb") as f:
        pickle.dump(response_list, f)


##################
# Define different types of simulations


def sim1_func(prompt, tokenizer, model):
    return generate_llm_tokens(
        prompt,
        tokenizer,
        model,
        token_generation_func={
            "0": unwatermarked_token_generation,
            "50": inverse_token_generation,
            "120": unwatermarked_token_generation,
        },
        statistic_func_list=[pivot_statistic_gumbel_func, inverse_pivot_statistic_func],
        verbose=False,  # no individual progress bar
        out_tokens=150,
    )


def sim2_func(prompt, tokenizer, model):
    token_generation_func = {
        "0": unwatermarked_token_generation,
        "50": gumbel_token_generation,
        "120": unwatermarked_token_generation,
        "170": gumbel_token_generation,
        "180": unwatermarked_token_generation,
    }
    token_generation_func_serialized = {
        k: v.__name__ for k, v in token_generation_func.items()
    }
    response = generate_llm_tokens(
        prompt,
        tokenizer,
        model,
        token_generation_func=token_generation_func,
        statistic_func_list=[pivot_statistic_gumbel_func],
        verbose=False,  # no individual progress bar
        out_tokens=200,
    )
    pivot_stat = [
        x.get("pivot_statistic_gumbel_func") for x in response.get("statistic", [])
    ]
    pivot_stat = np.array(pivot_stat)  # convert to numpy array

    start_time = time.perf_counter()
    intervals = get_wm_gumbel_interval(pivot_stat)
    end_time = time.perf_counter()
    response["detected_intervals"] = intervals
    response["detection_time"] = end_time - start_time

    # print(f"Intervals: {intervals}, Time taken: {end_time - start_time}")
    response["token_generation_func"] = token_generation_func_serialized
    return response


if __name__ == "__main__":
    run_simulation_loop(
        sim_func=sim2_func,
        outfile="./data/output/detection_2.pkl",
        B=100,
    )
