from typing import Any, Union, List, Tuple, Dict
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from tqdm.auto import tqdm
import numpy as np

from watermarking_schemes import BaseWatermark
from prf_schemes import prf_factory


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


def generate_spike_logits(vocab_size: int):
    max_index = np.random.randint(low=0, high=vocab_size)
    max_prob = 1 - np.random.uniform(low=1e-3, high=0.5)
    probs = np.ones(vocab_size) * (1 - max_prob) / (vocab_size - 1)
    probs[max_index] = max_prob
    probs = torch.tensor(probs)
    logits = torch.log(probs)
    return logits - logits[0]  # for identifiability, always make first coordinate =


# ------------------------------------
# Different types of LLM Generation Functions
# ------------------------------------


# +++++++++++++++++++++++++++++
# Standard watermarked text generation
def generate_llm_tokens(
    prompts: list[str],
    tokenizer,  # usually AutoTokenizer
    model,  # usually AutoModelForCausalLM
    watermarking_scheme: Union[
        BaseWatermark, Dict[str, BaseWatermark]
    ],  # a token generation function, or a dict <start_index>:<token_gen_func>, see below.
    verbose=False,
    prompt_tokens=50,  # take the first 50 tokens of prompt as input
    out_tokens=50,  # output next 50 tokens
    vocab_size=None,
    batch_size=8,
    max_position_embedding=2047,
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
    watermarking_scheme_dict = (
        watermarking_scheme if isinstance(watermarking_scheme, dict) else {"0": watermarking_scheme}
    )
    token_change_times = [int(x) for x in list(watermarking_scheme_dict.keys())]
    token_change_times = sorted(token_change_times, reverse=True)

    # load the tokenizer and convert prompts to ids
    tokens = tokenizer(prompts[:batch_size], return_tensors="pt", truncation=True, padding=True, max_length=128)
    torch_prompt: torch.Tensor = tokens["input_ids"][:, :prompt_tokens]
    inputs = torch_prompt.to(model.device)
    counter_range = tqdm(range(out_tokens)) if verbose else range(out_tokens)

    past = None
    for counter in counter_range:

        # apply the model with only past key KV-Cache or none
        with torch.no_grad():
            if past:
                output = model(inputs[:, -1:], past_key_values=past)  # apply the model
            else:
                output = model(inputs)
        probs = torch.nn.functional.softmax(output.logits[:, -1, :], dim=1)  # apply softmax over the last dimension
        past = output.past_key_values

        # if the KV cache is too big, reset it
        if past is not None:
            if getattr(past, "get_seq_length") is not None:
                if past.get_seq_length() > max_position_embedding:
                    past = None
            elif past[0][0].shape[1] > max_position_embedding:
                past = None

        # extract the token generation function
        for key in token_change_times:
            if key <= counter:
                wm_scheme = watermarking_scheme_dict[str(key)]
                break
        else:
            # in case it does not break
            wm_scheme = watermarking_scheme_dict["0"]

        # decoding functions does not support batching
        gen_tokens = torch.zeros(batch_size, dtype=inputs.dtype)  # (batch_size, )
        for i in range(batch_size):
            # develop the seed based on current history
            prf_seed = wm_scheme.get_prf_seed(inputs[i, :])
            gen_token = wm_scheme.generate_token(probs[i, :].view(-1), seed=prf_seed)
            gen_tokens[i] = gen_token  # add to generated tokens

        # merge the inputs and generated token for next batch
        inputs = torch.concat((inputs, gen_tokens.view(-1, 1)), dim=1)  # keep first dim as it is, merge across 2nd dim

    # at the end, produce the decoded text
    out_text_list = tokenizer.batch_decode(inputs)
    input_text_list = tokenizer.batch_decode(torch_prompt)

    # extract the generated token indices
    output_tokens: List[List[int]] = inputs[:, prompt_tokens:].cpu().numpy().tolist()

    return [
        {"prompt": input_text_list[i], "gen_tokens": output_tokens[i], "output": out_text_list[i]}
        for i in range(batch_size)
    ]


# +++++++++++++++++++++++++++++
# Fake watermarked text generation
def generate_fake_llm_tokens(
    watermarking_scheme: Union[
        BaseWatermark, Dict[str, BaseWatermark]
    ],  # a token generation function, or a dict <start_index>:<token_gen_func>, see below.
    verbose=False,
    prompt_tokens=50,
    out_tokens=50,  # how many tokens to output
    vocab_size=1000,
    ar_coeff=0.9,
    seed: int = 1234,
):
    # This is a simulation of LLM token generation with / without watermark patches

    # initial preparation
    watermarking_scheme_dict = (
        watermarking_scheme if isinstance(watermarking_scheme, dict) else {"0": watermarking_scheme}
    )
    token_change_times = [int(x) for x in list(watermarking_scheme_dict.keys())]
    token_change_times = sorted(token_change_times, reverse=True)

    counter_range = tqdm(range(out_tokens)) if verbose else range(out_tokens)

    # set a seed for reproducibility
    np.random.seed(seed)
    curr_z = generate_spike_logits(vocab_size)  # generate an initial spiked logits

    g = torch.Generator(device=get_torch_device())
    g.manual_seed(seed)
    inputs = torch.randint(low=0, high=vocab_size, size=(prompt_tokens,), generator=g)

    for counter in counter_range:
        new_z = generate_spike_logits(vocab_size)
        curr_z = curr_z * np.sqrt(ar_coeff) + np.sqrt(1 - ar_coeff) * new_z  # mixture of existing logit & new logit
        probs = F.softmax(curr_z, dim=0)  # convert to probabilities

        # extract the token generation function
        for key in token_change_times:
            if key <= counter:
                wm_scheme = watermarking_scheme_dict[str(key)]
                break
        else:
            wm_scheme = watermarking_scheme_dict["0"]

        # generate the token
        prf_seed = wm_scheme.get_prf_seed(inputs)
        gen_token = wm_scheme.generate_token(probs.view(-1), seed=prf_seed)
        inputs = torch.concat((inputs, gen_token.view(-1)))

    # final output
    output_tokens: List[int] = inputs[prompt_tokens:].cpu().numpy().tolist()

    return {"gen_tokens": output_tokens}


# +++++++++++++++++++++++++++++
# Human editing functions
def apply_human_edits_simple(
    output_tokens: List[int],
    vocab_size: int,
    cud_probs: Tuple[float, float, float] = (0.1, 0.1, 0.1),  # (create, update, delete)
    seed=1234,
):
    # Proceed by (Delete, Sub, Insert) in this order
    g = torch.Generator()
    g.manual_seed(seed)

    tokens = torch.Tensor(output_tokens)

    # deletion process
    if cud_probs[2] > 0:
        idx = torch.rand(size=tokens.shape, generator=g)
        tokens = tokens[idx > cud_probs[2]]  # remove deleted words

    # substitution process
    distribution = lambda x: torch.ones(size=(len(tokens), vocab_size)) / vocab_size
    if cud_probs[1] > 0:
        idx = torch.rand(size=tokens.shape, generator=g) < cud_probs[1]
        new_probs = distribution(tokens)
        samples = torch.multinomial(new_probs, 1, generator=g)
        tokens[idx] = samples[idx]

    # insertion process
    if cud_probs[0] > 0:
        idx = torch.where(torch.rand(size=tokens.shape, generator=g) < cud_probs[0])[0]
        new_probs = distribution(tokens)
        samples = torch.multinomial(new_probs, 1)
        for i in idx.sort(descending=True).values:
            tokens = torch.cat((tokens[:i], samples[i], tokens[i:]))

    return tokens.tolist()
