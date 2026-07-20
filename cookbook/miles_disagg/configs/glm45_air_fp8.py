"""GLM-4.5-Air: bf16 trainer, served in native HF FP8 through the disaggregated pool."""

from __future__ import annotations

from cookbook.common.config import ModalConfig
from cookbook.common.constants import DATA_PATH, PREP_PATH
from cookbook.miles_disagg.config import MilesConfig

APP_NAME = "stitch-glm45-air-fp8"
DELTA_VOLUME_NAME = "stitch-delta-glm45-air-fp8"
DELTA_BULLETIN_ROOT = "/delta-bulletin"
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"
SIDECAR_COMMIT_MODE = "in_place"  # reload without pausing generation — no serving gap; weights shift under in-flight decodes, RL-tolerated
SIDECAR_FLUSH_CACHE_ON_COMMIT = False

MODEL_TAG = "glm45-air-bf16"
SOURCE_MODEL = "zai-org/GLM-4.5-Air"
ROLLOUT_SOURCE_MODEL = "zai-org/GLM-4.5-Air-FP8"
SERVED_CHECKPOINT_FORMAT = "fp8"
USE_MODAL_TORCH_DIST_WRAPPER = True
DISABLE_HF_XET = True
DISABLE_HF_TRANSFER = True

MEGATRON_RUNTIME_PATCHES = ["/root/cookbook/miles_disagg/patches/megatron-r3-dispatch.patch"]

SGLANG_SERVER_ARGS = {
    # Dense FP8 reloads the full checkpoint each step; fastsafetensors splits files across TP
    # ranks (each reads 1/N) to cut bytes under gVisor's byte-copy-bound read. nogds is forced
    # via SGLANG_FASTSAFETENSORS_NOGDS in the serving image (GDS/nvidia-fs is absent under gVisor).
    "--load-format": "fastsafetensors",
    "--dtype": "auto",
    "--reasoning-parser": "glm45",
    "--tool-call-parser": "glm45",
    "--dist-timeout": "3600",
    "--context-length": "32768",
    "--mem-fraction-static": "0.7",
    "--chunked-prefill-size": "8192",
    "--max-prefill-tokens": "16384",
    "--cuda-graph-max-bs-prefill": "2048",  # avoid H200 cold-start graph-compile hangs
    "--skip-server-warmup": "",
}


class _Miles(MilesConfig):
    miles_model_script = "scripts/models/glm4.5-106B-A12B.sh"

    hf_checkpoint = f"{PREP_PATH}/{MODEL_TAG}/fp8"
    ref_load = f"{PREP_PATH}/{MODEL_TAG}/torch_dist"
    megatron_to_hf_mode = "raw"
    model_name = "glm4moe"

    actor_num_nodes = 4
    actor_num_gpus_per_node = 8
    num_gpus_per_node = 8
    colocate = False
    rollout_num_gpus = 0                 # external rollout: framework runs no local engines
    rollout_num_gpus_per_engine = 4
    rollout_endpoint_url = None          # filled at launch from the pool gateway
    use_miles_router = True

    custom_rollout_request_hook_path = "cookbook.common.hooks.gated_rollout_request_hook"
    custom_update_weight_post_write_path = "cookbook.common.hooks.commit_and_wake"
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
    save_interval = None  # miles forces a final save when set, regardless of interval
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
    memory=(1024, int(2 * 1024 * 1024)),
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
