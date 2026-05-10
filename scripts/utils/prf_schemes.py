import torch

"""
Reference: https://github.com/lx10077/TrGoF/blob/main/LLM_codes/alternative_prf_schemes.py
"""

# Generate a global permute table once at startup
rng = torch.Generator(device=torch.device("cpu"))
rng.manual_seed(2971215073)  # fib47 is prime
table_size = 1_000_003
fixed_table = torch.randperm(1_000_003, device=torch.device("cpu"), generator=rng)


# Define a list of PRF schemes
def hashint(integer_tensor: torch.Tensor) -> torch.Tensor:
    """Sane version, in the end we only need a small permutation table."""
    return fixed_table[integer_tensor.cpu() % table_size] + 1


def multiplicative_prf(input_ids: torch.Tensor, salt_key: int) -> int:
    return int(salt_key * input_ids.prod().item())


def additive_prf(input_ids: torch.Tensor, salt_key: int) -> int:
    return int(salt_key * input_ids.sum().item())


def minfunc_prf(input_ids: torch.Tensor, salt_key: int) -> int:
    # not a great idea for non-random input ids as in text
    return int(salt_key * input_ids.min().item())


def simple_skip_prf(input_ids: torch.Tensor, salt_key: int, k=2) -> int:
    # k is the skip distance
    return int(hashint((salt_key * input_ids[::k])).prod().item())


def skipgram_prf(input_ids: torch.Tensor, salt_key: int) -> int:
    # maximum distance skipgram within context
    return int(hashint(salt_key * input_ids[0]).item())


def counter_prf(input_ids: torch.Tensor, salt_key: int) -> int:
    return salt_key + input_ids.shape[0]


prf_pairs = {
    "hashint": hashint,
    "multiplicative": multiplicative_prf,
    "additive": additive_prf,
    "minfunc": minfunc_prf,
    "simple_skip": simple_skip_prf,
    "skipgram": skipgram_prf,
}


def prf_factory(self, prf_type="counter", context_size=5):

    if prf_type == "counter":
        return counter_prf
    else:
        # extract only the last contexts and apply prf function
        prf_fun = prf_pairs[prf_type]
        return lambda input_ids, salt_key: prf_fun(input_ids[-context_size:], salt_key)
