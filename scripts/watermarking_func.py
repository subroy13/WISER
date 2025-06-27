import torch

###########################
# LLM Token Generation functions (with and without watermarking)


# generate llm text without watermarking
def unwatermarked_token_generation(probs, counter, vocab_size):
    gen_tokens = torch.multinomial(probs, 1)
    return gen_tokens


# generate llm text with gumbel watermarking
def gumbel_token_generation(probs, counter, vocab_size, seed=1234):
    g = torch.Generator()
    g.manual_seed(seed + counter)
    unif_noise = torch.rand(vocab_size, generator=g)
    gumbel_ratio = torch.log(unif_noise) / probs[0]
    return torch.argmax(gumbel_ratio).view(-1, 1)


def pivot_statistic_gumbel_func(gen_tokens, counter, vocab_size, seed=1234):
    g = torch.Generator()
    g.manual_seed(seed + counter)
    unif_noise = torch.rand(vocab_size, generator=g)
    return -torch.log(1 - unif_noise[gen_tokens[0, 0]]).item()


# generate llm text with inverse watermarking
def inverse_token_generation(probs, counter, vocab_size, seed=1234):
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


def inverse_pivot_statistic_func(gen_tokens, counter, vocab_size, seed=1234):
    g = torch.Generator()
    g.manual_seed(seed + counter)
    unif_noise = torch.rand(1, generator=g)  # (1,)
    pi = torch.randperm(vocab_size, generator=g)  # random permutation (vocab_size, )
    return torch.abs(pi[gen_tokens[0, 0]] - unif_noise).item()
