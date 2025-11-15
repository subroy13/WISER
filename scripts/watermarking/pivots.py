import numpy as np
import torch

#############
# GUMBEL Watermarking


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


def pivot_statistic_inverse_func(gen_tokens, vocab_size, seed=1234):
    pivot_stat = []
    for counter, gen_token in enumerate(gen_tokens):
        g = torch.Generator()
        g.manual_seed(seed + counter)
        unif_noise = torch.rand(1, generator=g)  # (1,)
        pi = torch.randperm(vocab_size, generator=g)  # random permutation (vocab_size, )
        normalized = pi[gen_token] / (vocab_size - 1)  # as pi[gen_token] yields a value between 0 to (vocab_size - 1)
        pivot_stat.append(1 - np.abs((normalized - unif_noise).item()))  # 1 - <..> so that under H0, mean is small
    return pivot_stat


def null_distn_inverse(shape, vocab_size):
    unif_noise = np.random.rand(*shape)
    pi_wt = np.random.randint(vocab_size)
    normalized = pi_wt / (vocab_size - 1)
    return 1 - np.abs((normalized - unif_noise))


############
# Red-Green Watermarking


def pivot_statistic_redgreen_func(gen_tokens, vocab_size, seed=1234, green_list_size=0.25):
    # delta = 2   # from experiments in the paper
    green_list_len = round(vocab_size * green_list_size)
    pivot_stat = []
    for counter, gen_token in enumerate(gen_tokens):
        g = torch.Generator()
        g.manual_seed(seed + counter)
        pi = torch.randperm(vocab_size, generator=g)  # random permutation (vocab_size, )
        normalized = (int(gen_token in pi[:green_list_len]) - green_list_size) / (
            green_list_size * (1 - green_list_size)
        ) ** 0.5
        pivot_stat.append(normalized)
    return pivot_stat


def null_distn_redgreen(shape, vocab_size, green_list_size=0.25):
    binom_noise = np.random.binomial(n=1, p=green_list_size, size=shape)
    normalized = (binom_noise - green_list_size) / (green_list_size * (1 - green_list_size)) ** 0.5
    return normalized


############
# Permute and Flip Watermarking


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
