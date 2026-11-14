"""Qwen3.5-9B at lr 2e-6 past step 80 with on-policy distillation from the 27B (harbor_opd.py): arm C
(qwen3_5_9b_harbor_from80_c: mixture-9b-v2, rollout seed 45, 40 steps from lr2x-from40-a's step-80 export) plus the
teacher's per-token reverse KL subtracted from the advantages."""

from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import OPD_TEACHER_URL, run

globals().update(
    run(
        "lr2x-from80-opd",
        2e-6,
        45,
        start="qwen3-5-9b-lr2x-from40-a-step80-opd",
        opd={"coef": 0.1, "url": OPD_TEACHER_URL},
    )
)
