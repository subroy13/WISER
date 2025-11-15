import torch

###########################
# LLM Token Generation functions with different watermarking scheme
##########################

# Gumbel Watermarking


# generate llm text with gumbel watermarking
def gumbel_token_generation(probs: torch.Tensor, counter, vocab_size, seed=1234):
    device = probs.device
    g = torch.Generator()
    g.manual_seed(seed + counter)
    unif_noise = torch.rand(vocab_size, generator=g).to(device)
    gumbel_ratio = torch.log(unif_noise) / probs
    return torch.argmax(gumbel_ratio).view(-1, 1)


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
    index = torch.searchsorted(cdf, unif_noise.item(), right=False)  # Find the first index where cdf exceeds unif_noise

    # Return the original vocab index corresponding to the sampled one
    return inv_pi[index].view(-1, 1)


# Red-Green Watermarking - arXiv:2301.10226
def redgreen_token_generation(
    probs: torch.Tensor,
    counter,
    vocab_size,
    seed=1234,
    green_list_size=0.25,
    delta: float = 2,  # from experiments in the paper
):
    green_list_len = round(vocab_size * green_list_size)
    g = torch.Generator()
    g.manual_seed(seed + counter)
    pi = torch.randperm(vocab_size, generator=g)  # random permutation (vocab_size, )
    logits = torch.log(probs)
    logits[pi[:green_list_len]] += delta
    probs_new = torch.softmax(logits, dim=0)  # apply softmax on logit scale
    return torch.multinomial(probs_new, 1).view(-1, 1)


# Permute-and-Flip Watermarking - arXiv:2402.05864
def pf_token_generation(probs: torch.Tensor, counter, vocab_size, seed=1234, temperature=1):
    device = probs.device
    logits = torch.log(probs)
    g = torch.Generator()
    g.manual_seed(seed + counter)
    rt = torch.rand(vocab_size, generator=g).to(device)
    biased_logits = logits / temperature - torch.log(rt)
    return torch.argmax(biased_logits).view(-1, 1)
