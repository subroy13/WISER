import torch
import numpy as np
import hashlib

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
    gumbel_ratio = torch.log(unif_noise) / probs
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

def null_distn_gumbel(shape, vocab_size):
    unif_noise = np.random.rand(*shape)
    return -np.log(1 - unif_noise)


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

    probs_shuffled = probs[inv_pi]  # probs is shape (vocab_size, )
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

def null_distn_inverse(shape, vocab_size):
    unif_noise = np.random.rand(*shape)
    pi_wt = np.random.randint(vocab_size)
    normalized = pi_wt / (vocab_size - 1)
    return 1 - np.abs((normalized - unif_noise))


######################
# Red-Green Watermarking - arXiv:2301.10226
def redgreen_token_generation(probs: torch.Tensor, counter, vocab_size, seed=1234, green_list_size = 0.25, delta: float = 2):
    # delta = 2   # from experiments in the paper
    green_list_len = round(vocab_size * green_list_size)
    g = torch.Generator()
    g.manual_seed(seed + counter)
    pi = torch.randperm(vocab_size, generator=g)  # random permutation (vocab_size, )
    logits = torch.log(probs)
    logits[pi[:green_list_len]] += delta
    probs_new = torch.softmax(logits, dim = 0)  # apply softmax on logit scale
    return torch.multinomial(probs_new, 1).view(-1, 1)


def pivot_statistic_redgreen_func(gen_tokens, vocab_size, seed=1234, green_list_size = 0.25):
    # delta = 2   # from experiments in the paper
    green_list_len = round(vocab_size * green_list_size)    
    pivot_stat = []
    for counter, gen_token in enumerate(gen_tokens):
        g = torch.Generator()
        g.manual_seed(seed + counter)
        pi = torch.randperm(vocab_size, generator=g)  # random permutation (vocab_size, )
        normalized = (int(gen_token in pi[:green_list_len]) - green_list_size) / (green_list_size * (1 - green_list_size))**0.5
        pivot_stat.append(normalized)
    return pivot_stat

def null_distn_redgreen(shape, vocab_size, green_list_size = 0.25):
    binom_noise = np.random.binomial(
        n = 1,
        p = green_list_size,
        size = shape
    )
    normalized = (binom_noise - green_list_size) /  (green_list_size * (1 - green_list_size))**0.5
    return normalized



#############################
# Permute-and-Flip Watermarking - arXiv:2402.05864
def pf_token_generation(probs: torch.Tensor, counter, vocab_size, seed=1234, temperature = 1):
    device = probs.device
    logits = torch.log(probs)
    g = torch.Generator()
    g.manual_seed(seed + counter)
    rt = torch.rand(vocab_size, generator=g).to(device)
    biased_logits = (logits / temperature - torch.log(rt))
    return torch.argmax(biased_logits).view(-1, 1)


def pivot_statistic_pf_func(gen_tokens, vocab_size, seed=1234):
    # gen_tokens is a numpy array, so convert into torch Tensor for torch operations
    pivot_stat = []
    for counter, gen_token in enumerate(gen_tokens):
        g = torch.Generator()
        g.manual_seed(seed + counter)
        rt = torch.rand(vocab_size, generator=g)
        pivot_stat.append(-torch.log(rt[gen_token]).item())
    return pivot_stat

def null_distn_pf(shape, vocab_size):
    unif_noise = np.random.rand(*shape)
    return -np.log(unif_noise)