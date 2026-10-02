"""IcePop after MiMo-V2.6's large-scale RL recipe without R3: the IcePop arm plus.

Frozen: this pins the run started before the clean recipes (IcePop r03), so its
retries and handoffs keep exactly the configuration it trained with. It shares the
IcePop arm's app, volume and W&B group, and changes from it:

- Top-p sampling whose support the trainer replays, and group-mean advantages
  without std normalization.
- Masking bounds [0.2, 5], MiMo-V2.6's, wider below than IcePop's [0.5, 5] default.
- Prompt-mean loss aggregation: a token mean within each prompt's rollouts, then an
  equal-weight mean over prompts.
- A frozen MoE router, which keeps expert load from drifting during RL.
- The B300 trainer the run started on.
"""

from dataclasses import replace

from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero_icepop import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero_icepop import (
    IcePopMiles,
    arguments,
)
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero_icepop import modal as _fleet

modal = replace(_fleet, gpu="B300")


class _Miles(IcePopMiles):
    rollout_top_p = 0.97
    # Only bounds the returned support; top-p sets it in practice.
    rollout_top_k = 4096
    disable_grpo_std_normalization = True
    tis_clip_low = 0.2
    calculate_per_token_loss = False
    prompt_mean_loss = True
    freeze_moe_router = True


miles = _Miles(**arguments())
