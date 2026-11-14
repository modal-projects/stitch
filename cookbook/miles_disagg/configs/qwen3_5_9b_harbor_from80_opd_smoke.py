"""Qwen3.5-9B on-policy distillation smoke run: 3 steps of the OPD arm (qwen3_5_9b_harbor_from80_opd) to check the
teacher path and measure the distillation KL against the advantages before the coefficient is fixed."""

from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_5_9b_harbor_lr import OPD_TEACHER_URL, run

globals().update(
    run(
        "lr2x-from80-opd-smoke",
        2e-6,
        45,
        start="qwen3-5-9b-lr2x-from40-a-step80-opd",
        steps=3,
        opd={"coef": 0.1, "url": OPD_TEACHER_URL},
    )
)
