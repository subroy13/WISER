from typing import Any, Union, List, Tuple
import torch
from transformers import PreTrainedTokenizer
from tqdm.auto import tqdm
import numpy as np
import json

# Get pytorch device
def get_torch_device(force_cpu: bool = False):
    if force_cpu:
        device_name = "cpu"
    elif torch.cuda.is_available():
        device_name = "cuda:0"
    elif torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    return torch.device(device_name)


# generate llm text without watermarking
def unwatermarked_token_generation(probs, counter, vocab_size, seed = 1234):
    g = torch.Generator(device = probs.device)
    g.manual_seed(seed + counter)
    gen_tokens = torch.multinomial(probs, 1, generator=g)
    return gen_tokens


# some utility functions for generating llm texts
def generate_llm_tokens(
    prompts: list[str],
    tokenizer,  # usually AutoTokenizer
    model,  # usually AutoModelForCausalLM
    token_generation_func: Any,  # a token generation function, or a dict <start_index>:<token_gen_func>, see below.
    verbose=False,
    prompt_tokens=50,  # take the first 50 tokens of prompt as input
    out_tokens=50,  # output next 50 tokens
    vocab_size=None,
    batch_size = 8,
    max_token_input_length = 256
):
    # It is also possible to provide input to the token_generation_func a dictionary of the following form
    # {
    #     "0": watermark_func_1,
    #     "t1": watermark_func_2,
    #     "t2": watermark_func_3,
    #     ...
    # }
    # It allows to use different watermarking scheme to be added in between
    if vocab_size is None or vocab_size < 0:
        vocab_size = model.get_output_embeddings().weight.shape[0]

    # some preparation
    if isinstance(token_generation_func, dict):
        token_change_times = [int(x) for x in list(token_generation_func.keys())]
        token_change_times = sorted(token_change_times, reverse=True)
    else:
        token_change_times = []

    tokens = tokenizer(
        prompts[:batch_size],
        return_tensors="pt", 
        truncation=True, 
        padding=True,
        max_length=128
    )
    torch_prompt = tokens['input_ids'][:, :prompt_tokens]
    inputs = torch_prompt.to(model.device)
    inputs_to_decode = inputs
    counter_range = tqdm(range(out_tokens)) if verbose else range(out_tokens)

    gen_tokens = []
    past = None
    for counter in counter_range:
        with torch.no_grad():
            if past:
                output = model(inputs[:,-1:], past_key_values = past)  # apply the model
            else:
                output = model(inputs)
        probs = torch.nn.functional.softmax(output.logits[:, -1, :], dim = 1)  # apply softmax over the last dimension
        past = output.past_key_values

        # extract the token generation function
        if len(token_change_times) > 0:
            for key in token_change_times:
                if key <= counter:
                    token_gen_func : Any = token_generation_func[str(key)]
                    break
        else:
            token_gen_func : Any = token_generation_func

        # for each row in batch, run the token generation function
        gen_token_indices = []

        for i in range(batch_size):
            gen_token = token_gen_func( # type: ignore
                probs = probs[i, :].view(-1), # this is passed as a vector (vocab_size, ) 
                counter=counter + prompt_tokens, 
                vocab_size = vocab_size
            ) # calculate the token
            gen_token_indices.append(int(gen_token.item()))

        gen_tokens.append(gen_token_indices) # shape = (out_tokens, batch_size)
        gen_token_indices = torch.tensor(gen_token_indices, dtype = inputs.dtype, device=model.device).view(-1, 1) # shape = (batch_size, 1)
        inputs = torch.concat((inputs, gen_token_indices), dim = 1) # keep first dim as it is, merge across 2nd dim
        inputs_to_decode = torch.concat((inputs_to_decode, gen_token_indices), dim = 1) # this is complete token sequence

        # subset to max size
        if inputs.shape[1] > max_token_input_length:
            inputs = inputs[:, -max_token_input_length:]

    # at the end, produce the decoded text
    out_text_list = tokenizer.batch_decode(inputs_to_decode)
    input_text_list = tokenizer.batch_decode(torch_prompt)
    return [{
        "prompt": input_text_list[i],
        "gen_tokens": np.array(gen_tokens)[:, i].tolist(),
        "output": out_text_list[i]
    } for i in range(batch_size)]



# Some more utility functions for different types of metric calculations
def read_json(fpath: str):
    with open(fpath, "r") as f:
        data = json.load(f)
        f.close()
    return data

# IOU is the typical metric that is tracked in segment detection scenarios
def get_iou(intervalsA: List[Tuple[int, int]], intervalsB: List[Tuple[int, int]]):
  coordsA = set([x for start, end in intervalsA for x in range(start, end)])
  coordsB = set([x for start, end in intervalsB for x in range(start, end)])
  num = len(coordsA.intersection(coordsB))
  denom = len(coordsA.union(coordsB))
  return (num / denom)

# hit count is essentially the numerator for calculating precision and recall
def get_hit_counts(true_intervals: List[Tuple[int, int]], estimated_intervals: List[Tuple[int, int]]):
  # count how many of the true intervals we have nonzero IOU
  hit_count = 0
  est_intervals = estimated_intervals.copy()
  for true_int in true_intervals:
    max_iou = 0
    max_index = None
    for j in range(len(est_intervals)):
      iou = get_iou([true_int], [est_intervals[j]])
      if iou > 0 and iou > max_iou:
        max_iou = iou
        max_index = j
    
    if max_iou > 0 and max_index is not None:
      hit_count += 1
      est_intervals.pop(max_index)

  return hit_count


def get_rand_index(intervalsA: List[Tuple[int, int]], intervalsB: List[Tuple[int, int]], n: int):
  # convert interval endpoints to sorted changepoints for fast computation
  # reference: https://arxiv.org/pdf/2112.03738
  cpA = sorted([0, n] + [x for interval in intervalsA for x in interval])
  cpB = sorted([0, n] + [x for interval in intervalsB for x in interval])
  num = 0
  for i in range(len(cpA) - 1):
    for j in range(len(cpB) - 1):
      nij = max(0, min(cpA[i+1], cpB[j+1]) - max(cpA[i], cpB[j]))
      num += (nij * abs(cpA[i+1] - cpB[j+1]))
  return 1 - 2 * num / (n*(n-1))

# calculate modified rand index which avoids exchangeability between non-watermarked and watermarked intervals
def get_modified_rand_index(intervalsA: List[Tuple[int, int]], intervalsB: List[Tuple[int, int]], n):
    ri = get_rand_index(intervalsA, intervalsB, n)

    # create mask for intervalsA and intervalsB
    maskA = np.zeros((n, ), dtype=np.bool)
    for s, e in intervalsA:
        maskA[s:e] = True

    maskB = np.zeros((n, ), dtype=np.bool)
    for s, e in intervalsB:
        maskB[s:e] = True

    # loop through pairs in intervalA
    counter = 0
    for s, e in intervalsA:
        for i in range(s, e):
            for j in range(i + 1, e):
                if (not maskB[i]) and (not maskB[j]):
                    counter += 1
    for s, e in intervalsB:
        for i in range(s, e):
            for j in range(i + 1, e):
                if (not maskA[i]) and (not maskA[j]):
                    counter += 1

    return ri, ri - (2 * counter / (n*(n-1)))


# metric to find the symmetric differences
def get_symmetric_difference(intA, intB):
  sA, eA = intA
  sB, eB = intB
  return abs(sA - sB) + abs(eA - eB)