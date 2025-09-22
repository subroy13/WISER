import torch
import numpy as np
import hashlib

######################
# Synth-ID Text: Tournament Sampling
# https://github.com/google-deepmind/synthid-text
def _synthid_accumulate_hash(current_hash: torch.Tensor, data: torch.Tensor, multiplier: int = 6364136223846793005, increment: int = 1):
    # f(x, data[T]) = f(f(x, data[:T-1]), data[T])
    # current_hash: (shape, ), 
    # data: (shape, tensor_len)
    for i in range(data.shape[-1]):
        current_hash = torch.add(current_hash, data[..., i])
        current_hash = torch.mul(current_hash, multiplier)
        current_hash = torch.add(current_hash, increment)
    return current_hash

def _synthid_get_gvals(top_k_keys: torch.Tensor, num_apply_hash: int = 12, shift: int = 0):
    # Sample g-values from computed top_k keys
    # top_k_keys - random (k, depth), return Gvalues (k, depth)
    shift = shift or (64 // num_apply_hash)
    for _ in range(num_apply_hash):
        top_k_keys = _synthid_accumulate_hash(top_k_keys, torch.tensor([1], dtype = torch.long)) >> shift
    return (top_k_keys >> 30) % 2


def _synthid_accumulate_hash_np(current_hash: np.ndarray, data: np.ndarray, multiplier: int = 6364136223846793005, increment: int = 1):
    # f(x, data[T]) = f(f(x, data[:T-1]), data[T])
    # current_hash: (shape, ), 
    # data: (shape, tensor_len)
    for i in range(data.shape[-1]):
        current_hash = np.add(current_hash, data[..., i])
        current_hash = np.multiply(current_hash, multiplier)
        current_hash = np.add(current_hash, increment)
    return current_hash


def _synthid_get_gvals_np(top_k_keys: np.ndarray, num_apply_hash: int = 12, shift: int = 0):
    # Sample g-values from computed top_k keys
    # top_k_keys - random (k, depth), return Gvalues (k, depth)
    shift = shift or (64 // num_apply_hash)
    for _ in range(num_apply_hash):
        top_k_keys = _synthid_accumulate_hash_np(top_k_keys, np.array([1], dtype = np.int64)) >> shift
    return (top_k_keys >> 30) % 2


def _run_tournament_sampling(scores: torch.Tensor, g_values: torch.Tensor, num_leaves: int):
    # scores - (vocab_size, ), g_valus: (vocab_size, depth)
    _, depth = g_values.shape
    device = scores.device
    probs = torch.softmax(scores, dim = -1)
    for i in range(depth):
        g_values_at_depth = g_values[:, i]
        g_mass_at_depth = (g_values_at_depth * probs).sum()
        coeff_not_in_g = (1 - g_mass_at_depth) ** (num_leaves - 1)
        coeff_in_g = (1 - (1 - g_mass_at_depth) ** num_leaves) / g_mass_at_depth
        coeffs = torch.where(
            torch.logical_and(g_values_at_depth == 1, probs > 0),
            coeff_in_g,
            coeff_not_in_g,
        )
        probs = probs * coeffs 
    return probs
    

def synthid_token_generation(probs: torch.Tensor, counter, vocab_size, seed=1234, top_k = 8):
    g = torch.Generator(device = probs.device) # set generator seed
    g.manual_seed(seed + counter)
    scores = torch.log(probs)
    top_k_result = torch.topk(scores, k = top_k)
    top_k_scores = top_k_result.values # (k, )
    top_k_indices = top_k_result.indices  # (k, )
    n_depth = int(np.log2(top_k))
    depth_keys = torch.randint(10000, size = (n_depth, ), generator=g, device=probs.device) # (log(top_k), )
    hash_iv = hashlib.sha256(depth_keys.to(torch.long).cpu().numpy().tobytes()).digest()
    torch_max = torch.iinfo(torch.int64).max
    hash_iv = torch.tensor(int.from_bytes(hash_iv, byteorder="big") % torch_max, device=probs.device) # (1,)
    hash_result = torch.vmap(_synthid_accumulate_hash, in_dims=(None, 1), out_dims=1)(hash_iv, top_k_indices[None, :, None]) # (1, k)
    hash_result = torch.vmap(_synthid_accumulate_hash, in_dims=(None, 2), out_dims=2)(hash_result, depth_keys[None, None, :, None]) # (1, k, depth)
    g_vals = _synthid_get_gvals(hash_result[0, :, :]) # (k, depth)
    probs_new = _run_tournament_sampling(top_k_scores, g_vals, num_leaves=3)
    return top_k_indices[torch.multinomial(probs_new, 1)].view(-1, 1)


def pivot_statistic_synthid_func(gen_tokens, vocab_size, seed=1234, top_k = 8):
    pivot_stat = []
    vocab_indices = torch.arange(vocab_size) # (vocab_size, )
    n_depth = int(np.log2(top_k))
    for counter, gen_token in enumerate(gen_tokens):
        g = torch.Generator() # set generator seed
        g.manual_seed(seed + counter)
        depth_keys = torch.randint(10000, size = (n_depth, ), generator=g)
        hash_iv = hashlib.sha256(depth_keys.to(torch.long).cpu().numpy().tobytes()).digest()
        torch_max = torch.iinfo(torch.int64).max
        hash_iv = torch.tensor(int.from_bytes(hash_iv, byteorder="big") % torch_max) # (1,)
        hash_result = torch.vmap(_synthid_accumulate_hash, in_dims=(None, 1), out_dims=1)(hash_iv, vocab_indices[None, :, None]) # (1, vocab_size)
        hash_result = torch.vmap(_synthid_accumulate_hash, in_dims=(None, 2), out_dims=2)(hash_result, depth_keys[None, None, :, None]) # (1, vocab_size, depth)
        g_vals = _synthid_get_gvals(hash_result[0, :, :]) # (vocab_size, depth)
        mean_g_score = g_vals[gen_token].to(torch.float).mean().item()
        pivot_stat.append(mean_g_score)
    return pivot_stat

def null_distn_synthid(shape, vocab_size, top_k = 8):
    n_depth = int(np.log2(top_k))
    torch_max = torch.iinfo(torch.int64).max
    random_keys = np.random.randint(low = 0, high = torch_max, size = (*shape, n_depth) )
    g_vals = _synthid_get_gvals_np(random_keys)
    mean_g_scores = np.mean(g_vals, axis = -1)
    return mean_g_scores