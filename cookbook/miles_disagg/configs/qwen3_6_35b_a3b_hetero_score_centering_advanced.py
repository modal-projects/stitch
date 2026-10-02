"""Score centering with top-p sampling: the score-centering arm plus.

Frozen: this pins the run started before the clean recipes (score centering r03,
on a B200 trainer from step 110), so its retries and handoffs keep exactly the
configuration it trained with. It shares the score-centering arm's app, volume and
W&B group, and changes from it:

- Top-p sampling bounded by top-k: the recorded top-128 candidates cover the whole
  realized support, with room for ties at the cutoff, so the expectation is exact.
- The B200 trainer the run moved to, pinned.
"""

from dataclasses import replace

from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero_score_centering import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero_score_centering import (
    ScoreCenteringMiles,
    arguments,
)
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero_score_centering import (
    modal as _fleet,
)

modal = replace(_fleet, gpu="B200")


class _Miles(ScoreCenteringMiles):
    rollout_top_p = 0.97
    rollout_top_k = 64


miles = _Miles(**arguments())
