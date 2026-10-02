"""IcePop on the heterogeneous fleet: vanilla GRPO with masked importance weights.

Each token's policy-gradient term is weighted by its detached trainer/sampler
probability ratio, and tokens whose ratio leaves [0.5, 5] are dropped (arXiv:2510.18855;
the same default as INTELLECT-3). Sampling, advantages and loss aggregation are
vanilla GRPO's.
"""

from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero import *  # noqa: F403
from cookbook.miles_disagg.configs.qwen3_6_35b_a3b_hetero import HeteroMiles, arguments

APP_NAME = "stitch-qwen36-hetero-icepop"
EXPERIMENT_VOLUME_NAME = APP_NAME


class IcePopMiles(HeteroMiles):
    use_tis = True
    custom_tis_function_path = (
        "miles.backends.training_utils.loss_hub.corrections.icepop_function"
    )
    tis_clip_low = 0.5
    tis_clip = 5.0

    wandb_group = "qwen36-hetero-icepop"
    prometheus_run_name = wandb_group


miles = IcePopMiles(**arguments())
