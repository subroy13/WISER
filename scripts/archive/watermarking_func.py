import torch
import numpy as np

###########################
# LLM Token Generation functions with different watermarking scheme
##########################

#############
# GUMBEL Watermarking

# generate llm text with gumbel watermarking
def gumbel_token_generation(probs: torch.Tensor, counter, vocab_size, seed=1234):
    device = probs.device
    g = torch.Generator()
    g.manual_seed(seed + counter)
    unif_noise = torch.rand(vocab_size, generator=g).to(device)
    gumbel_ratio = torch.log(unif_noise) / probs[0]
    return torch.argmax(gumbel_ratio).view(-1, 1)


def pivot_statistic_gumbel_func(gen_tokens, vocab_size, seed=1234):
    # gen_tokens is a numpy array, so convert into torch Tensor for torch operations
    pivot_stat = []
    for counter, gen_token in enumerate(gen_tokens):
        g = torch.Generator()
        g.manual_seed(seed + counter)
        unif_noise = torch.rand(vocab_size, generator=g)
        pivot_stat.append(-torch.log(1 - unif_noise[gen_token]).item())
    return pivot_stat


######################
# Inverse Watermarking

# generate llm text with inverse watermarking
def inverse_token_generation(probs: torch.Tensor, counter, vocab_size, seed=1234):
    g = torch.Generator()
    g.manual_seed(seed + counter)
    unif_noise = torch.rand(1, generator=g)  # (1,)
    pi = torch.randperm(vocab_size, generator=g)  # random permutation (vocab_size, )
    inv_pi = torch.empty_like(pi)
    inv_pi[pi] = torch.arange(vocab_size)

    probs_shuffled = probs[0, inv_pi]  # probs is shape (1, vocab_size)
    cdf = torch.cumsum(probs_shuffled, dim=0)  # (vocab_size,)
    index = torch.searchsorted(
        cdf, unif_noise.item(), right=False
    )  # Find the first index where cdf exceeds unif_noise

    # Return the original vocab index corresponding to the sampled one
    return inv_pi[index].view(-1, 1)


def pivot_statistic_inverse_func(gen_tokens, vocab_size, seed=1234):
    pivot_stat = []
    for counter, gen_token in enumerate(gen_tokens):
        g = torch.Generator()
        g.manual_seed(seed + counter)
        unif_noise = torch.rand(1, generator=g)  # (1,)
        pi = torch.randperm(vocab_size, generator=g)  # random permutation (vocab_size, )
        normalized = pi[gen_token] / (vocab_size - 1) # as pi[gen_token] yields a value between 0 to (vocab_size - 1)
        pivot_stat.append(1 - np.abs((normalized - unif_noise).item()))  # 1 - <..> so that under H0, mean is small
    return pivot_stat
