"""Implement other PRF functions, so, hashing schemes.

Can be hooked into existing WatermarkLogitsProcessor as modified base class WatermarkBase
"""

import torch
from itertools import combinations
from functools import cache

# Key properties of a hashing scheme
props = {
    "prf_type": str,  # string name of the underlying PRF mapping multiple token ids to a random seed
    "context_width": int,  # this is h in the paper, how many previous tokens should be considered for each PRF
    "self_salt": bool,  # Use the rules laid in robust-watermarking to use the token itself to seed and possibly reject its own list
    "hash_key": int,  # integer, large prime, used to move seed away from low-entrop bit sequences in PRF chosen above
}


def seeding_scheme_lookup(seeding_scheme: str):
    if not isinstance(seeding_scheme, str):
        raise ValueError("Seeding scheme should be a string summarizing the procedure.")
    if seeding_scheme == "simple_1" or seeding_scheme == "lefthash":
        # Default, simple bigram hash  # alias for ff-additive_prf-1-False-15485863
        prf_type = "additive_prf"
        context_width = 1
        self_salt = False
        hash_key = 15485863
    elif seeding_scheme == "algorithm-3" or seeding_scheme == "selfhash":
        prf_type = "anchored_minhash_prf"
        context_width = 4
        self_salt = True
        hash_key = 15485863
    elif seeding_scheme == "skipgram":
        prf_type = "skipgram_prf"
        context_width = 5
        self_salt = False
        hash_key = 15485863
    elif seeding_scheme.startswith(
        "ff"
    ):  # freeform seeding scheme API - only use for experimenting
        # expects strings of the form ff-additive_prf-4-True-hash or ff-additive_prf-5-True (hash key is optional)
        split_scheme = seeding_scheme.split("-")
        prf_type = str(split_scheme[1])
        context_width = int(split_scheme[2])
        self_salt = split_scheme[3] == "True"
        if len(split_scheme) == 5:
            hash_key = int(split_scheme[4])
        else:
            hash_key = 15485863
    else:
        raise ValueError(f"Invalid seeding scheme name {seeding_scheme} given. Try  'simple_1'?")

    assert prf_type in prf_lookup.keys()
    return prf_type, context_width, self_salt, hash_key


def multiplicative_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.prod().item()


def additive_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.sum().item()


def minfunc_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    # not a great idea for non-random input ids as in text
    return salt_key * input_ids.min().item()


def simple_skip_prf(input_ids: torch.LongTensor, salt_key: int, k=2) -> int:
    # k is the skip distance
    return hashint(salt_key * input_ids[::k]).prod().item()


def skipgram_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    # maximum distance skipgram within context
    return hashint(salt_key * input_ids[0]).item()


def anchored_skipgram_prf(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    # maximum distance skipgram within context
    return (hashint(salt_key * input_ids[0]) * hashint(salt_key * input_ids[anchor])).item()


def minhash_prf(input_ids: torch.LongTensor, salt_key: int) -> int:
    # slightly less not the greatest idea for non-random input ids as in text
    return hashint(salt_key * input_ids).min().item()


def anchored_minhash_prf(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    # Anchor to one key to produce a min over pairs again
    return (salt_key * hashint(input_ids) * hashint(input_ids[anchor])).min().item()


def minskipgram_prf(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    # min over all skipgrams in context, k=2 is all pairs
    skipgrams = torch.as_tensor(list(combinations(hashint(salt_key * input_ids), 2)))
    return skipgrams.prod(dim=1).min().item()


def noncomm_prf(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    key = torch.as_tensor(salt_key, dtype=torch.long)
    for entry in input_ids:
        key *= hashint(key * entry)
        key %= 2**32
    return key.item()


def position_prf(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    return (
        (salt_key * input_ids * torch.arange(1, len(input_ids) + 1, device=input_ids.device))
        .sum()
        .item()
    )


prf_lookup = {
    "multiplicative_prf": multiplicative_prf,
    "additive_prf": additive_prf,
    "minfunc_prf": minfunc_prf,
    "simple_skip_prf": simple_skip_prf,
    "skipgram_prf": skipgram_prf,
    "anchored_skipgram_prf": anchored_skipgram_prf,
    "minhash_prf": minhash_prf,
    "anchored_minhash_prf": anchored_minhash_prf,
    "minskipgram_prf": minskipgram_prf,
    "noncomm_prf": noncomm_prf,
    "position_prf": position_prf,
}

# Generate a global permute table once at startup
rng = torch.Generator(device=torch.device("cpu"))
rng.manual_seed(2971215073)  # fib47 is prime
table_size = 1_000_003
fixed_table = torch.randperm(
    1_000_003, device=torch.device("cpu"), generator=rng
)  # actually faster than I thought


def hashint(integer_tensor: torch.LongTensor) -> torch.LongTensor:
    """Sane version, in the end we only need a small permutation table."""
    return (
        fixed_table[integer_tensor.cpu() % table_size] + 1
    )  # minor cheat here, this function always return CPU values


def _hashint_avalanche_tensor(integer_tensor: torch.LongTensor):
    """http://burtleburtle.net/bob/hash/integer.html, ported into pytorch, runs on tensors. Apparently a decent avalanche."""
    i = integer_tensor.to(torch.int32).clone()  # or torch.int16?
    i -= i << 6
    i ^= i >> 17
    i -= i << 9
    i ^= i << 4
    i -= i << 3
    i ^= i << 10
    i ^= i >> 15
    return i.to(torch.long)


@cache
def _hashint_avalanche_int(integer: int):
    """http://burtleburtle.net/bob/hash/integer.html, runs in base python, caches based on access.
    Does this make sense for signed 64bit ints?"""
    i = integer % (2**32)
    i -= i << 6
    i ^= i >> 17
    i -= i << 9
    i ^= i << 4
    i -= i << 3
    i ^= i << 10
    i ^= i >> 15
    return i
