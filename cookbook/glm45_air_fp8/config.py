"""Config for the glm45_air_fp8 example — GLM-4.5-Air trained in bf16, served in
native HF FP8 through a disaggregated sglang rollout pool.

Three things live here:
  - identity + path constants both deployment halves key off;
  - ModalConfig (Modal infra) and MilesConfig (miles training args), the two config
    surfaces of the app;
  - the concrete ``modal`` / ``miles`` instances for this experiment.

Every public, non-callable MilesConfig attribute becomes a miles CLI arg via
``cli_args`` (miles wraps Megatron's parser, so Megatron args pass straight through);
``environment`` / ``async_mode`` / ``miles_model_script`` are launcher instructions,
not CLI args. A different model or precision is a different example, not an edit here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

# ── identity + paths (both halves key off these) ─────────────────────────────────
APP_NAME = "stitch-glm45-air-fp8"          # the Modal app / Flash pool
SERVER_CLS_NAME = "Server"                 # the Flash-served rollout replica class
DELTA_VOLUME_NAME = "stitch-delta-glm45-air-fp8"  # the Store's Modal Volume
DELTA_BULLETIN_ROOT = "/delta-bulletin"    # Store root: `latest` + <run_id>/ chains
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"  # engine's per-host materialized checkpoint
SIDECAR_COMMIT_MODE = "quiesce"            # fp8 reload is exact; draining buys clean per-version attribution

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")
PREP_PATH = Path("/prep")  # <PREP>/<tag>/{bf16 masters, fp8 served base, torch_dist ref_load}

MODEL_TAG = "glm45-air-bf16"
SOURCE_MODEL = "zai-org/GLM-4.5-Air"          # bf16 masters + trainer arch
ROLLOUT_SOURCE_MODEL = "zai-org/GLM-4.5-Air-FP8"  # the served FP8 base

# R3 routing-replay needs the dropless Megatron dispatch fix applied at trainer start.
MEGATRON_RUNTIME_PATCHES = ["/root/cookbook/glm45_air_fp8/patches/megatron-r3-dispatch.patch"]

GPUType = Literal["H100", "H200", "B200", "B300", "A100"]

# MilesConfig fields that are launcher instructions, not miles CLI args.
_MILES_SKIP = {"environment", "async_mode", "miles_model_script"}
# MilesConfig fields miles reads as YAML files; inline dicts are materialized before launch.
YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config", "te_precision_config_file")

# sglang server flags (precision comes from the served checkpoint, not a --quantization flag).
SGLANG_SERVER_ARGS = {
    "--dtype": "auto",
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm45",
    "--dist-timeout": "3600",
    "--context-length": "32768",
    "--mem-fraction-static": "0.7",
    "--chunked-prefill-size": "8192",
    "--max-prefill-tokens": "16384",
    "--piecewise-cuda-graph-max-tokens": "2048",  # avoid H200 cold-start graph-compile hangs
    "--model-loader-extra-config": '{"enable_multithread_load":true,"num_threads":8}',
    "--skip-server-warmup": "",
}


class ModalConfig:
    """Modal infrastructure: GPU model, regions, rollout-pool sizing, prep topology."""

    gpu: GPUType = "B200"
    memory: tuple[int, int] | None = None
    cloud: str | None = None
    region: str | None = None
    rollout_min_containers: int = 2
    rollout_max_containers: int | None = None
    # Flash autoscaler target: keep well below the sglang engine concurrency so Flash
    # adds containers instead of packing requests until KV saturates and they stall.
    rollout_target_inputs: int | None = None
    proxy_regions: list[str] = ["us-west"]
    rollout_ephemeral_disk_mib: int | None = None
    rollout_memory_mib: int | None = None
    torch_dist_prep_nodes: int = 2
    torch_dist_prep_gpus_per_node: int = 8
    torch_dist_convert_extra_args: str = ""
    torch_dist_prep_ephemeral_disk_mib: int | None = None
    trainer_ephemeral_disk_mib: int | None = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class MilesConfig:
    """miles training config. Subclass and set class attributes; all public,
    non-callable, non-skip attributes become miles CLI args via ``cli_args``."""

    environment: dict = {}
    async_mode: bool = False       # True -> train_async.py
    miles_model_script: str = ""   # shell script (relative to the miles root) that defines MODEL_ARGS

    def __init__(self, **kwargs: Any) -> None:
        self.environment = dict(type(self).environment)  # fresh per instance; never mutate the class default
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _fields(self) -> dict[str, Any]:
        """Merged fields across the class hierarchy; instance attrs win."""
        fields: dict[str, Any] = {}
        for cls in reversed(type(self).__mro__):
            if cls is object:
                continue
            fields.update(
                {
                    k: v
                    for k, v in vars(cls).items()
                    if not k.startswith("_")
                    and not callable(v)
                    and not isinstance(v, (classmethod, staticmethod, property))
                }
            )
        fields.update(vars(self))
        return {k: v for k, v in fields.items() if k not in _MILES_SKIP}

    def cli_args(self) -> list[str]:
        """miles CLI args: field_name -> --field-name; True -> bare flag; False/None ->
        omitted; list -> --flag v1 v2; else --flag value."""
        out: list[str] = []
        for key, val in self._fields().items():
            if val is None or val is False:
                continue
            flag = f"--{key.replace('_', '-')}"
            if val is True:
                out.append(flag)
            elif isinstance(val, list):
                out += [flag] + [str(v) for v in val]
            else:
                out += [flag, str(val)]
        return out

    def prepare_data(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} has no prepare_data()")

    def to_payload(self) -> dict[str, Any]:
        """Flatten to plain data so launch_train can ship a config to the deployed
        Trainer — new or edited experiments run without a redeploy."""
        return {
            "fields": self._fields(),
            "environment": dict(self.environment),
            "async_mode": self.async_mode,
            "miles_model_script": self.miles_model_script,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MilesConfig":
        cfg = cls(**payload["fields"])
        cfg.environment = dict(payload["environment"])
        cfg.async_mode = payload["async_mode"]
        cfg.miles_model_script = payload["miles_model_script"]
        return cfg


class _Miles(MilesConfig):
    miles_model_script = "scripts/models/glm4.5-106B-A12B.sh"

    hf_checkpoint = f"{PREP_PATH}/{MODEL_TAG}/fp8"        # served FP8 base
    ref_load = f"{PREP_PATH}/{MODEL_TAG}/torch_dist"      # trainer weights (raw-mode Megatron)
    megatron_to_hf_mode = "raw"
    model_name = "glm4moe"

    actor_num_nodes = 4
    actor_num_gpus_per_node = 8
    num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0                 # external rollout: the framework runs no local engines
    rollout_num_gpus_per_engine = 4
    rollout_endpoint_url = None          # filled at launch from the pool gateway
    use_miles_router = True

    # The three plug points stitch fills (resolved by miles inside the trainer process):
    custom_rollout_request_hook_path = "cookbook.glm45_air_fp8.hooks.gated_rollout_request_hook"
    custom_update_weight_post_write_path = "cookbook.glm45_air_fp8.hooks.commit_and_wake"
    custom_config_path = {
        "rollout_request_weight_version_mode": "min",
        "rollout_request_weight_version_lag": 1,
        "rollout_request_retry_attempts": 900,
        "rollout_request_retry_sleep": 1.0,
        "rollout_session_affinity_header": "Modal-Session-ID",
        "rollout_request_timeout_secs": 300,
    }

    async_mode = True
    update_weights_interval = 1
    update_weight_transfer_mode = "disk-delta"
    update_weight_delta_encoding = "xor"
    update_weight_delta_checksum = "xxh3-128"
    update_weight_disk_dir = DELTA_BULLETIN_ROOT  # run-scoped at launch to <root>/<run_id>

    prompt_data = f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl"
    input_key = "prompt"
    label_key = "label"
    apply_chat_template = True
    rollout_shuffle = True
    balance_data = True
    rm_type = "deepscaler"
    eval_interval = None

    num_rollout = 10
    save_interval = None  # miles forces a final save when set, regardless of interval; leave off
    rollout_batch_size = 16
    rollout_max_response_len = 4096
    rollout_temperature = 0.8
    n_samples_per_prompt = 4
    global_batch_size = 64
    use_dynamic_global_batch_size = True
    sglang_server_concurrency = 128
    use_rollout_routing_replay = False

    tensor_model_parallel_size = 1
    sequence_parallel = True
    pipeline_model_parallel_size = 4
    context_parallel_size = 1
    expert_model_parallel_size = 8
    expert_tensor_parallel_size = 1
    decoder_last_pipeline_num_layers = 10
    use_dynamic_batch_size = True
    max_tokens_per_gpu = 8192
    recompute_granularity = "full"
    recompute_method = "uniform"
    recompute_num_layers = 1
    attention_dropout = 0.0
    hidden_dropout = 0.0
    accumulate_allreduce_grads_in_fp32 = True
    attention_softmax_in_fp32 = True
    no_check_for_nan_in_loss_and_grad = True

    optimizer = "adam"
    lr = 1e-6
    lr_decay_style = "constant"
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    optimizer_cpu_offload = True
    overlap_cpu_optimizer_d2h_h2d = True
    use_precision_aware_optimizer = True

    advantage_estimator = "gspo"
    eps_clip = 4e-4
    eps_clip_high = None
    use_kl_loss = False
    kl_loss_coef = 0.0
    kl_loss_type = "low_var_kl"
    entropy_coef = 0.0
    use_tis = True

    environment = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
        "NVSHMEM_DISABLE_NCCL": "1",
        "NCCL_TIMEOUT_MS": "360000000",
    }

    def prepare_data(self) -> None:
        from datasets import load_dataset

        ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(50000, ds.num_rows)))
        ds = ds.map(lambda ex: {"label": ex["reward_model"]["ground_truth"]})
        ds = ds.select_columns(["prompt", "label"])
        ds.to_json(f"{DATA_PATH}/dapo-math-17k/dapo-math-17k.jsonl")


modal = ModalConfig(
    gpu="H200",
    region="us",
    memory=1_048_576,
    rollout_min_containers=2,
    rollout_max_containers=4,   # start at 2; scale to 4 mid-run to exercise elastic join
    rollout_target_inputs=32,
    proxy_regions=["us-west"],
    rollout_ephemeral_disk_mib=819_200,
    torch_dist_prep_nodes=4,
    torch_dist_prep_gpus_per_node=8,
    torch_dist_convert_extra_args=(
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 4 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--decoder-last-pipeline-num-layers 10"
    ),
    torch_dist_prep_ephemeral_disk_mib=819_200,
)

miles = _Miles()
