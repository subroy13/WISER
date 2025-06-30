import torch
from transformers import PreTrainedTokenizer
from tqdm import tqdm


# generate llm text without watermarking
def unwatermarked_token_generation(probs, counter, vocab_size):
    gen_tokens = torch.multinomial(probs, 1)
    return gen_tokens


# Utility function that can be used to generate tokens using LLM
# and also track different pivot statistic as needed
def generate_llm_tokens(
    prompt: str,
    tokenizer: PreTrainedTokenizer,  # usually AutoTokenizer
    model,  # usually AutoModelForCausalLM
    token_generation_func=unwatermarked_token_generation,  # a token generation function
    statistic_func_list=[],
    verbose=False,
    prompt_tokens=50,  # take the first 50 tokens of prompt as input
    out_tokens=50,  # output next 50 tokens
    vocab_size=None,
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
    tokens = tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=2000
    )  # get the token vector
    torch_prompt = tokens[:, :prompt_tokens]  # give the first prompt_tokens as input
    inputs = torch_prompt.to(model.device)
    statistic = []
    counter_range = tqdm(range(out_tokens)) if verbose else range(out_tokens)
    for counter in counter_range:
        with torch.no_grad():
            output = model(inputs)  # apply the model
        probs = torch.nn.functional.softmax(
            output.logits[:, -1, :], dim=-1
        )  # apply softmax over the last dimension

        # extract the token generation function
        if isinstance(token_generation_func, dict):
            token_change_times = [int(x) for x in list(token_generation_func.keys())]
            for key in sorted(token_change_times, reverse=True):
                if key <= counter:
                    token_gen_func = token_generation_func[str(key)]
                    break
        else:
            token_change_times = []  # no change times
            token_gen_func = token_generation_func

        gen_tokens = token_gen_func(
            probs, counter + prompt_tokens, vocab_size
        )  # calculate the token

        # now calculate the statistic functions
        if len(statistic_func_list) > 0:
            stat = {
                func.__name__: func(gen_tokens, counter + prompt_tokens, vocab_size)
                for func in statistic_func_list
            }
            statistic.append(stat)

        gen_tokens = gen_tokens.to(model.device)  # move to the device
        inputs = torch.concat(
            (inputs, gen_tokens), dim=1
        )  # first dim = batch_size, second dim needs to merge

    # at the end, produce the decoded text
    out_text = tokenizer.decode(inputs[0])
    return {
        "prompt": tokenizer.decode(torch_prompt[0]),
        "statistic": statistic,
        "output": out_text,
    }
