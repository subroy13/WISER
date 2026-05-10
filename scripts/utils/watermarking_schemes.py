from typing import Callable
import numpy as np
import torch


class BaseWatermark:
    """
    Implements a base barebone watermarking scheme
    """

    def __init__(self, vocab_size, prf_func: Callable[[torch.Tensor, int], int], key=None, **kwargs):
        self.vocab_size = vocab_size
        self.prf_func = prf_func
        self.key = 15485863 if key is None else key

    def get_prf_seed(
        self,
        generated_tokens: torch.Tensor,  # (history_len, )
    ):
        return self.prf_func(generated_tokens, self.key)

    def decoding_func(self, probs: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        raise NotImplementedError("Implemented in specialized watermarking scheme")

    def generate_token(self, probs: torch.Tensor, seed: int):
        device = probs.device
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return self.decoding_func(probs, g)

    def pivot_statistic_func(self, gen_token: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        raise NotImplementedError("Implemented in specialized watermarking scheme")

    def get_pivot_statistic(self, gen_tokens: torch.Tensor):  # (n, )
        # gen_tokens is long tensor, containing integer indices
        pivot_state = torch.zeros_like(gen_tokens, dtype=torch.float32)
        for i in range(gen_tokens.shape[0]):
            seed = self.get_prf_seed(gen_tokens[:i])
            g = torch.Generator(device=gen_tokens.device)
            g.manual_seed(seed)
            pivot_state[i] = self.pivot_statistic_func(gen_tokens[i], g)

    def null_distn(self, shape):
        raise NotImplementedError("Implemented in specialized watermarking scheme")


class UnWatermark(BaseWatermark):
    """
    Implements the token generation scheme without watermarking
    """

    def decoding_func(self, probs: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        gen_token = torch.multinomial(probs, 1, generator=g)
        return gen_token


class GumbelMaxWatermark(BaseWatermark):
    """
    Implements Gumbel-max trick watermarking scheme
    """

    def decoding_func(self, probs: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        unif_noise = torch.rand(self.vocab_size, generator=g)
        gumbel_ratio = torch.log(unif_noise) / probs
        return torch.argmax(gumbel_ratio)

    def pivot_statistic_func(self, gen_token: torch.Tensor, g: torch.Generator):
        unif_noise = torch.rand(self.vocab_size, generator=g)  # (1,)
        return -torch.log(1 - unif_noise[gen_token])

    def null_distn(self, shape):
        unif_noise = np.random.rand(*shape)
        return -np.log(1 - unif_noise)


class InverseWatermark(BaseWatermark):
    """
    Implements inverse watermarking scheme
    """

    def decoding_func(self, probs: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        unif_noise = torch.rand(1, generator=g)  # (1,)
        pi = torch.randperm(self.vocab_size, generator=g)  # random permutation (vocab_size, )
        inv_pi = torch.empty_like(pi)
        inv_pi[pi] = torch.arange(self.vocab_size)

        probs_shuffled = probs[inv_pi]
        cdf = torch.cumsum(probs_shuffled, dim=0)  # (vocab_size,)
        index = torch.searchsorted(cdf, unif_noise.item(), right=False)
        index = torch.searchsorted(
            cdf, unif_noise.item(), right=False
        )  # Find the first index where cdf exceeds unif_noise
        index = index.clamp(0, self.vocab_size - 1)

        # Return the original vocab index corresponding to the sampled one
        return inv_pi[index]

    def pivot_statistic_func(self, gen_token: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        unif_noise = torch.rand(1, generator=g)  # (1, )
        pi = torch.randperm(self.vocab_size, generator=g)  # random permutation (vocab_size, )
        normalized = pi[gen_token] / (
            self.vocab_size - 1
        )  # as pi[gen_token] yields a value between 0 to (vocab_size - 1)
        return 1 - torch.abs(normalized - unif_noise)

    def null_distn(self, shape):
        unif_noise = np.random.rand(*shape)
        pi_wt = np.random.randint(self.vocab_size)
        normalized = pi_wt / (self.vocab_size - 1)
        return 1 - np.abs((normalized - unif_noise))


class RedGreenWatermark(BaseWatermark):
    """
    Implements Red-Green watermarking scheme
    """

    def __init__(self, vocab_size, prf_func: Callable[[torch.Tensor], int], key=None, **kwargs):
        super.__init__(vocab_size, prf_func, key, **kwargs)
        self.green_list_size = kwargs.get("green_list_size", 0.25)
        self.delta = kwargs.get("delta", 2.0)

    def decoding_func(self, probs: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        green_list_len = round(self.vocab_size * self.green_list_size)
        pi = torch.randperm(self.vocab_size, generator=g)  # random permutation (vocab_size, )
        logits = torch.log(probs)
        logits[pi[:green_list_len]] += self.delta
        probs_new = torch.softmax(logits, dim=0)  # apply softmax on logit scale
        return torch.multinomial(probs_new, 1)

    def pivot_statistic_func(self, gen_token: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        green_list_len = round(self.vocab_size * self.green_list_size)
        pi = torch.randperm(self.vocab_size, generator=g)  # random permutation (vocab_size, )
        normalized = (int(gen_token in pi[:green_list_len]) - self.green_list_size) / (
            self.green_list_size * (1 - self.green_list_size)
        ) ** 0.5
        return normalized

    def null_distn(self, shape):
        binom_noise = np.random.binomial(n=1, p=self.green_list_size, size=shape)
        normalized = (binom_noise - self.green_list_size) / (self.green_list_size * (1 - self.green_list_size)) ** 0.5
        return normalized


class PermuteFlipWatermark(BaseWatermark):
    """
    Reference: Permute-and-Flip Watermarking - arXiv:2402.05864
    """

    def __init__(self, vocab_size, prf_func: Callable[[torch.Tensor], int], key=None, **kwargs):
        super().__init__(vocab_size, prf_func, key, **kwargs)
        self.temperature = kwargs.get("temperature", 1)

    def decoding_func(self, probs: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        rt = torch.rand(self.vocab_size, generator=g)
        logits = torch.log(probs)
        biased_logits = logits / self.temperature - torch.log(rt)
        return torch.argmax(biased_logits)

    def pivot_statistic_func(self, gen_token: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        rt = torch.rand(self.vocab_size, generator=g)
        return -torch.log(rt[gen_token])

    def null_distn(self, shape):
        unif_noise = np.random.rand(*shape)
        return -np.log(unif_noise)
