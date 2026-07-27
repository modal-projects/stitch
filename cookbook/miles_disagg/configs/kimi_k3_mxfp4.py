"""Kimi K3 rollout server using its native MXFP4 checkpoint."""

from cookbook.common.config import ModalConfig
from cookbook.common.serving_image import SGLangRuntime

ROLLOUT_SOURCE_MODEL = "moonshotai/Kimi-K3"
ROLLOUT_SOURCE_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
ROLLOUT_NUM_GPUS_PER_ENGINE = 8
SGLANG_DELTA_UPDATE_MODE = "cpu"

SGLANG_RUNTIME = SGLangRuntime(
    image=(
        "lmsysorg/sglang@"
        "sha256:81a9c00654b3e4c7c681a4728a64fcb4853aa698dc9fea1959bbf4eb26bfb2e5"
    ),
    repository="https://github.com/modal-projects/sglang.git",
    branch="stitch-sglang-kimi-k3",
    commit="1a87b07642368bbc9a7985edea73548fb3d07203",
)

SGLANG_SERVER_ARGS = {
    "--trust-remote-code": "",
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--enable-cpu-weight-cache": "",
    "--cpu-weight-cache-max-compile-group-gb": "8",
    "--weight-loader-drop-cache-after-load": "",
    "--dist-timeout": "3600",
    "--context-length": "1048576",
    "--max-running-requests": "32",
    "--cuda-graph-max-bs-decode": "32",
    "--mem-fraction-static": "0.85",
    "--kv-cache-dtype": "fp8_e4m3",
    "--mamba-ssm-dtype": "bfloat16",
    "--mamba-radix-cache-strategy": "extra_buffer_lazy",
    "--chunked-prefill-size": "16384",
    "--schedule-policy": "lpm",
    "--mm-feature-transport": "cuda_ipc",
    "--mm-processor-worker-num": "2",
    "--mm-io-worker-num": "16",
    "--reasoning-parser": "kimi_k3",
    "--tool-call-parser": "kimi_k3",
}

modal = ModalConfig(
    gpu="B300",
    rollout_target_inputs=32,
    rollout_ephemeral_disk_mib=2 * 1024 * 1024,
    # CPU updates retain the canonical checkpoint plus all TP rank images.
    rollout_memory_mib=(1024 * 1024, 4 * 1024 * 1024),
)
