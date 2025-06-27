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

vocab_size = model.get_output_embeddings().weight.shape[0]
print(f"There are {vocab_size} many words in vocabulary")
print(f"The device is {device}")


# Main function to generate tokens using LLM
def generate_llm(
    prompt,
    model,
    token_generation_func=unwatermarked_token_generation,
    statistic_func_list=[],
    verbose=False,
    prompt_tokens=50,  # take first 50 tokens as input
    out_tokens=50,  # output next 50 tokens
):

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
            for key in sorted(
                [int(x) for x in list(token_generation_func.keys())], reverse=True
            ):
                if key <= counter:
                    token_gen_func = token_generation_func[str(key)]
                    break
        else:
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


def get_prompts():
    with open("./prompts.txt", "r", errors="ignore") as f:
        prompts = f.read().split("\n===\n")
    return prompts


def run_simulation(llm_generation_func, outfile="./response_list.pkl", B=-1):
    prompts = get_prompts()
    if B < 0:
        B = len(prompts)
    else:
        B = min(B, len(prompts))
    response_list = []
    enum_list = list(enumerate(prompts[:B]))
    for i, prompt in tqdm(enum_list):
        # print(f"Processing prompt {i} out of {len(prompts)}")
        response = llm_generation_func(prompt, model)
        response_list.append(response)

        # save the results
        if i % 10 == 0:
            with open(outfile, "wb") as f:
                pickle.dump(response_list, f)

    # save the results finally
    with open(outfile, "wb") as f:
        pickle.dump(response_list, f)


if __name__ == "__main__":
    # do the simulation here first, for the null case (unwatermarked)
    # run_simulation(
    #     lambda prompt, model: generate_llm(
    #         prompt,
    #         model,
    #         token_generation_func=inverse_token_generation,
    #         statistic_func_list=[
    #             pivot_statistic_gumbel_func,
    #             inverse_pivot_statistic_func,
    #         ],
    #         verbose=False,  # no individual progress bar
    #     )
    # )

    run_simulation(
        lambda prompt, model: generate_llm(
            prompt,
            model,
            token_generation_func={
                "0": unwatermarked_token_generation,
                "50": inverse_token_generation,
                "120": unwatermarked_token_generation,
            },
            statistic_func_list=[
                pivot_statistic_gumbel_func,
                inverse_pivot_statistic_func,
            ],
            verbose=False,  # no individual progress bar
            out_tokens=150,
        ),
        outfile="./output/response_list_uwm_50_inverse_120_uwm.pkl",
        B=50,
    )
