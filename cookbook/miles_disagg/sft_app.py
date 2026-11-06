"""Supervised fine-tuning with Miles on one trainer machine, without a rollout pool.

The recipe's `miles` config runs Miles' SFT path with --debug-train-only, so Miles starts no SGLang engine, and this
app has no rollout Server. RUN_ID (minted when unset) scopes the run's checkpoints, HF exports and logs on the recipe's
run Volume; a retry resumes from the newest checkpoint there.

    EXPERIMENT_CONFIG=<recipe> MODAL_ENVIRONMENT=<env> uv run --extra modal modal run --detach -m cookbook.miles_disagg.sft_app
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

import modal

from cookbook.common import ray_cluster
from cookbook.common.constants import (
    CHECKPOINTS_PATH,
    HF_CACHE_PATH,
    KERNEL_CACHE_PATH,
    MINUTES,
    RAY_PORT,
    STITCH_PATH,
)
from cookbook.miles_disagg import trainer_image
from cookbook.miles_disagg.config import MilesConfig, train_command
from cookbook.miles_disagg.trainer_image import MEGATRON_PATH

EXPERIMENT = os.environ["EXPERIMENT_CONFIG"]
RUN_ID = os.environ.setdefault("RUN_ID", uuid.uuid4().hex[:8])
DATA_ENV = os.environ.get("STITCH_DATA_ENV") or None
exp = importlib.import_module(f"cookbook.miles_disagg.configs.{EXPERIMENT}")
modal_cfg, miles_cfg = exp.modal, exp.miles
if miles_cfg.n_train_nodes != 1 or not getattr(miles_cfg, "debug_train_only", False):
    raise ValueError("sft_app runs one trainer machine with --debug-train-only")
VOLUME_ENV = getattr(exp, "EXPERIMENT_VOLUME_ENV", None)
RUN_DIR = STITCH_PATH / RUN_ID

image = trainer_image.build_trainer_image(
    hf_cache_path=str(HF_CACHE_PATH),
    experiment=EXPERIMENT,
    run_id=RUN_ID,
    miles_repo_ref=exp.MILES_REPO_REF,
    miles_image_tag=exp.MILES_IMAGE_TAG,
    extra_env={"STITCH_DATA_ENV": DATA_ENV} if DATA_ENV else {},
)
run_volume = modal.Volume.from_name(
    exp.EXPERIMENT_VOLUME_NAME, environment_name=VOLUME_ENV, create_if_missing=True, version=2
)
volumes = {
    str(CHECKPOINTS_PATH): modal.Volume.from_name(
        "miles-checkpoints", environment_name=VOLUME_ENV, create_if_missing=True, version=2
    ),
    str(STITCH_PATH): run_volume,
    str(KERNEL_CACHE_PATH): modal.Volume.from_name(
        modal_cfg.kernel_cache_volume, create_if_missing=True, version=2
    ),
    **getattr(exp, "TRAINER_EXTRA_VOLUMES", {}),
}
secrets = (
    [modal.Secret.from_name("wandb-secret", environment_name=DATA_ENV)]
    if getattr(miles_cfg, "use_wandb", False)
    else []
)
app = modal.App(f"{exp.APP_NAME}-{RUN_ID}")


def run_paths(cfg: MilesConfig, run_dir: Path) -> MilesConfig:
    """Save to and resume from the run's directory: the base checkpoint (ref_load) seeds a fresh run."""
    cfg.save = cfg.load = str(run_dir / "checkpoints")
    cfg.save_hf = str(run_dir / cfg.save_hf)
    return cfg


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:{miles_cfg.actor_num_gpus_per_node}",
    cpu=modal_cfg.trainer_cpu,
    memory=modal_cfg.trainer_memory_mib,
    cloud=modal_cfg.cloud,
    region=modal_cfg.region,
    volumes=volumes,
    secrets=secrets,
    ephemeral_disk=modal_cfg.trainer_ephemeral_disk_mib,
    timeout=24 * 60 * MINUTES,
    retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=2.0, max_delay=60.0),
    single_use_containers=True,
    include_source=False,
)
def train(payload: dict) -> None:
    cfg = run_paths(MilesConfig.from_payload(payload), RUN_DIR)
    _, _, ip = ray_cluster.get_modal_cluster_context(1)
    ray_cluster.start_ray_node(
        0,
        ip,
        ip,
        n_nodes=1,
        ray_port=RAY_PORT,
        extra_env={
            "MILES_HOST_IP": ip,
            "PYTHONPATH": f"{MEGATRON_PATH}:{os.environ.get('PYTHONPATH', '')}",
            "TRITON_CACHE_DIR": str(KERNEL_CACHE_PATH / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(KERNEL_CACHE_PATH / "inductor"),
            **cfg.environment,
        },
    )
    cmd = train_command(cfg)
    print(f"SFT {EXPERIMENT}: run={RUN_ID}\nCommand: {cmd}", flush=True)
    log_path = RUN_DIR / "logs" / f"sft-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.log"
    with tempfile.TemporaryDirectory() as tmp:
        local_log = os.path.join(tmp, "train.log")
        try:
            subprocess.run(["bash", "-lc", f"set -o pipefail; ({cmd}) 2>&1 | tee {local_log}"], check=True)
        finally:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_log, log_path)
            run_volume.commit()


@app.local_entrypoint()
def main() -> None:
    print(f"SFT run {RUN_ID} on {app.name}: checkpoints in {exp.EXPERIMENT_VOLUME_NAME}:/{RUN_ID}", flush=True)
    train.remote(miles_cfg.to_payload())
