"""A reward that needs no model quality: 0 or 1, a fixed function of the sample."""

import hashlib


async def seeded_random_reward(args, sample, **kwargs) -> float:
    key = f"{args.rollout_seed}:{sample.index}:{sample.response}".encode()
    return float(hashlib.sha256(key).digest()[0] & 1)
