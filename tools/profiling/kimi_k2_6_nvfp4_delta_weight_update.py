"""Profile one Kimi K2.6 NVFP4 delta weight update on Modal.

The entrypoint prepares the pinned serving checkpoint with the Miles NVFP4
recipe, builds a standardized element-wise synthetic delta, and runs one
verified update.

    MODAL_FUNCTION_RUNTIME=runc uv run --extra modal modal run -d \
      tools/profiling/kimi_k2_6_nvfp4_delta_weight_update.py \
      --update-mode cpu
"""

from __future__ import annotations

from pathlib import Path

import modal

from cookbook.common.constants import CHECKPOINTS_PATH, HF_CACHE_PATH
from cookbook.common.serving_image import build_serving_image
from cookbook.miles_disagg import prep, trainer_image
from cookbook.miles_disagg.configs import kimi_k2_6_nvfp4 as model
from tools.profiling._delta_weight_update import (
    WeightUpdateSpec,
    modal_runtime_label,
    parse_update_mode,
    run_delta_weight_update,
)
from tools.profiling._synthetic_delta import (
    SyntheticDeltaSpec,
    prepare_standard_delta,
    synthetic_delta_profile_id,
)

APP_NAME = "profile-kimi-k2-6-nvfp4-delta-weight-update"
EXPERIMENT = "kimi_k2_6_nvfp4"
DELTA_MOUNT = "/synthetic-delta"
DELTA_SPEC = SyntheticDeltaSpec(
    checkpoint_format="nvfp4",
    quantized_value_density=0.003,
    high_precision_value_density=0.01,
    # Text-only RL leaves the vision encoder and projector fixed.
    immutable_prefixes=("vision_tower.", "mm_projector."),
)
DELTA_ID = f"kimi-k2-6/{model.SOURCE_REVISION}/{synthetic_delta_profile_id(DELTA_SPEC)}"
DELTA_SOURCE_DIR = f"{DELTA_MOUNT}/{DELTA_ID}"
BASE_CHECKPOINT_DIR = str(model.ROLLOUT_CHECKPOINT_PATH)
LOCAL_CHECKPOINT_ROOT = "/local-checkpoint/kimi-k2-6-nvfp4"
LOCAL_TARGET_CHECKPOINT_DIR = f"{LOCAL_CHECKPOINT_ROOT}/target"
LOCAL_CANONICAL_CHECKPOINT_DIR = f"{LOCAL_CHECKPOINT_ROOT}/canonical"
SGLANG_CACHE_PATH = "/root/.cache/sglang"

SGLANG_SERVER_ARGS = {
    "--served-model-name": model.SOURCE_MODEL,
    "--load-format": "fastsafetensors",
    "--model-loader-extra-config": '{"enable_gds":false}',
    "--weight-loader-drop-cache-after-load": "",
    "--cpu-weight-cache-canonical-checkpoint-dir": LOCAL_CANONICAL_CHECKPOINT_DIR,
    "--trust-remote-code": "",
    "--tool-call-parser": "kimi_k2",
    "--reasoning-parser": "kimi_k2",
    "--dist-timeout": "3600",
    "--kv-cache-dtype": "fp8_e4m3",
    "--attention-backend": "tokenspeed_mla",
    "--context-length": "32768",
    "--mem-fraction-static": "0.80",
    "--chunked-prefill-size": "16384",
    "--schedule-conservativeness": "0.5",
    "--schedule-policy": "lpm",
    "--max-running-requests": "32",
    "--decode-log-interval": "100",
    "--cuda-graph-max-bs-decode": "32",
    "--random-seed": "42",
    "--skip-server-warmup": "",
}

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, version=2
)
checkpoint_volume = modal.Volume.from_name(
    "miles-checkpoints", create_if_missing=True, version=2
)
delta_volume = modal.Volume.from_name(
    "stitch-synthetic-deltas", create_if_missing=True, version=2
)
sglang_cache_volume = modal.Volume.from_name(
    "sglang-cache", create_if_missing=True, version=2
)
prep_image = trainer_image.build_trainer_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
)
serving_image = build_serving_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
).add_local_dir(
    str(Path(__file__).resolve().parents[1]),
    remote_path="/root/tools",
    ignore=["**/__pycache__", "**/*.pyc"],
)


@app.function(
    image=prep_image,
    gpu=f"{model.modal.gpu}:1",
    memory=model.modal.trainer_memory_mib,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(CHECKPOINTS_PATH): checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=6 * 60 * 60,
)
def prepare_base() -> None:
    prep.prepare_checkpoints(model, checkpoint_volume)


@app.function(
    image=serving_image,
    cpu=64,
    memory=(64 * 1024, 512 * 1024),
    volumes={
        str(CHECKPOINTS_PATH): checkpoint_volume.read_only(),
        DELTA_MOUNT: delta_volume,
    },
    timeout=6 * 60 * 60,
)
def prepare_delta() -> dict:
    return prepare_standard_delta(
        BASE_CHECKPOINT_DIR,
        DELTA_SOURCE_DIR,
        spec=DELTA_SPEC,
        commit=delta_volume.commit,
    )


@app.function(
    image=serving_image,
    gpu="B300:4",
    cpu=64,
    memory=(1024 * 1024, 3 * 1024 * 1024),
    # Disk mode retains both the immutable base and a complete mutable target.
    ephemeral_disk=1_572_864,
    volumes={
        str(CHECKPOINTS_PATH): checkpoint_volume.read_only(),
        DELTA_MOUNT: delta_volume.read_only(),
        SGLANG_CACHE_PATH: sglang_cache_volume,
    },
    timeout=4 * 60 * 60,
)
def benchmark(update_mode: str, runtime: str, sample_id: str) -> dict:
    return run_delta_weight_update(
        WeightUpdateSpec(
            model_name="Kimi K2.6 NVFP4",
            base_checkpoint_dir=BASE_CHECKPOINT_DIR,
            local_target_checkpoint_dir=LOCAL_TARGET_CHECKPOINT_DIR,
            server_args=SGLANG_SERVER_ARGS,
        ),
        source_dir=DELTA_SOURCE_DIR,
        target_version=1,
        update_mode=parse_update_mode(update_mode),
        runtime=runtime,
        sample_id=sample_id,
    )


@app.local_entrypoint()
def main(update_mode: str = "cpu", sample_id: str = "1") -> None:
    prepare_base.remote()
    prepare_delta.remote()
    benchmark.remote(parse_update_mode(update_mode), modal_runtime_label(), sample_id)
