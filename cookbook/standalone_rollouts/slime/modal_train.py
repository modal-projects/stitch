"""Optional SLIME trainer for testing the standalone API-shim provider.

This module imports the provider app and adds trainer-only functions to it. A
deploy of `cookbook.standalone_rollouts.modal_serve` publishes only the rollout
provider; a deploy of this module publishes both the provider and the trainer.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import tempfile
from pathlib import Path

import modal
import modal.experimental

from cookbook.standalone_rollouts import modal_serve as provider_app
from cookbook.standalone_rollouts.slime import hooks
from cookbook.slime_disagg import helpers
from cookbook.slime_disagg.configs.base import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    SlimeConfig,
)


TRAINER_CONFIG = os.environ.get(
    "TRAINER_CONFIG", os.environ.get("EXPERIMENT_CONFIG", "moonlight_slime_trainer")
)
exp = importlib.import_module(
    f"cookbook.standalone_rollouts.slime.configs.{TRAINER_CONFIG}"
)

modal_cfg = exp.modal
slime_cfg = exp.slime

APP_NAME = provider_app.APP_NAME
MODEL_NAME = slime_cfg.hf_checkpoint
N_TRAIN_NODES = helpers.training_nodes(slime_cfg)

MINUTES = 60
RAY_PORT = 6379

SLIME_IMAGE_TAG = "slimerl/slime:nightly-dev-20260527a"
SLIME_ROOT = "/root/slime"
SLIME_REPO_URL = "https://github.com/modal-projects/slime.git"
# Pin to an exact commit (see cookbook/slime_disagg/modal_train.py): the cached
# clone layer otherwise leaves the container on a stale slime.
SLIME_REPO_REF = "ebfe153949b1a69c39e92f947ed5d475166dd724"

trainer_image = (
    modal.Image.from_registry(SLIME_IMAGE_TAG)
    .entrypoint([])
    .run_commands(f"rm -rf {HF_CACHE_PATH}")
    .run_commands(
        f"rm -rf {SLIME_ROOT}"
        f" && git clone --depth 1 {SLIME_REPO_URL} {SLIME_ROOT}"
        f" && cd {SLIME_ROOT}"
        f" && git fetch --depth 1 origin {SLIME_REPO_REF}"
        f" && git checkout FETCH_HEAD"
        f" && python3 -m pip install --no-deps -e {SLIME_ROOT}"
    )
    # The base image installs megatron-core as a PEP 660 *strict* editable that
    # hides megatron.training (which slime's megatron backend imports). Reinstall
    # in compat mode so the whole source tree is importable (mirrors slime_disagg).
    .run_commands(
        "cd /root/Megatron-LM"
        " && python3 -m pip install --no-deps -e . --config-settings editable_mode=compat"
    )
    .pip_install(
        # Disk-delta encode side (slime.utils.disk_delta) compresses with
        # zstandard and checksums with xxhash (xxh3-128 default) / blake3.
        "zstandard",
        "xxhash",
        "blake3",
    )
    .env(
        {
            "EXPERIMENT_CONFIG": TRAINER_CONFIG,
            "TRAINER_CONFIG": TRAINER_CONFIG,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "STITCH_SHIM_TRANSPORT_ROOT": str(provider_app.S3_TRANSPORT_MOUNT_PATH),
        }
    )
    .add_local_python_source("stitch")
    .add_local_dir(
        Path(__file__).parents[2],
        remote_path="/root/cookbook",
        ignore=["**/__pycache__"],
    )
)

control_image = (
    modal.Image.debian_slim()
    .env(
        {
            "STITCH_SHIM_TRANSPORT_ROOT": str(provider_app.S3_TRANSPORT_MOUNT_PATH),
            "STITCH_SHIM_API_BASE_URL": "http://preflight.invalid",
        }
    )
    .add_local_python_source("stitch")
    .add_local_dir(
        Path(__file__).parents[2],
        remote_path="/root/cookbook",
        ignore=["**/__pycache__"],
    )
)

# Dev iteration: SLIME_LOCAL_DIR overlays a local slime checkout onto the image's
# cloned fork (installed editable at /root/slime), so fork edits take effect on
# container start with no image rebuild. Unset by default.
if slime_local := os.environ.get("SLIME_LOCAL_DIR"):
    trainer_image = trainer_image.add_local_dir(
        slime_local,
        remote_path=SLIME_ROOT,
        ignore=[".git", "**/__pycache__", "**/*.pyc"],
    )

hf_cache_volume = provider_app.hf_cache_volume
data_volume = modal.Volume.from_name("slime-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("slime-checkpoints", create_if_missing=True)

train_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
    str(provider_app.S3_TRANSPORT_MOUNT_PATH): provider_app.s3_transport_mount,
}
reloadable_train_volumes = (hf_cache_volume, data_volume, checkpoints_volume)

app = provider_app.app
utility_app = provider_app.utility_app
preflight_app = modal.App(f"{APP_NAME}-preflight")


@app.cls(
    image=trainer_image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}",
    memory=modal_cfg.memory,
    cloud=modal_cfg.cloud,
    region=modal_cfg.region,
    volumes=train_volumes,
    secrets=[modal.Secret.from_name(exp.SHIM_SECRET_NAME)],
    timeout=24 * 60 * MINUTES,
    startup_timeout=20 * MINUTES,
    scaledown_window=30 * MINUTES,
    experimental_options={"efa_enabled": True},
    include_source=False,
)
@modal.experimental.clustered(N_TRAIN_NODES, rdma=True)
class Trainer:
    @modal.enter()
    def start_ray(self) -> None:
        rank, master_addr, my_ip = helpers.get_modal_cluster_context(N_TRAIN_NODES)
        self.rank = rank
        os.environ.update(
            {
                "SLIME_HOST_IP": my_ip,
                "SGLANG_HOST_IP": my_ip,
                "HOST_IP": my_ip,
                "MASTER_ADDR": master_addr,
                "RAY_ADDRESS": f"{master_addr}:{RAY_PORT}",
                "no_proxy": f"127.0.0.1,{master_addr},{my_ip}",
                "NO_PROXY": f"127.0.0.1,{master_addr},{my_ip}",
                **_trainer_environment(),
            }
        )
        if rank == 0:
            helpers.start_ray_head(my_ip, N_TRAIN_NODES, ray_port=RAY_PORT)
        else:
            helpers.start_ray_worker(my_ip, master_addr, ray_port=RAY_PORT)

    @modal.method()
    def train(self, experiment: str, payload: dict) -> None:
        for volume in reloadable_train_volumes:
            volume.reload()
        if self.rank != 0:
            return

        _validate_trainer_environment()
        provider_url = (
            os.environ.get("STITCH_SHIM_API_BASE_URL") or provider_app.frontdoor_url()
        ).rstrip("/")
        os.environ["STITCH_SHIM_API_BASE_URL"] = provider_url
        cfg = SlimeConfig.from_payload(payload)
        run = importlib.import_module(
            f"cookbook.standalone_rollouts.slime.configs.{experiment}"
        )
        if helpers.training_nodes(cfg) != N_TRAIN_NODES:
            raise ValueError(
                f"experiment {experiment!r} needs {helpers.training_nodes(cfg)} node(s) but this app "
                f"was deployed with {N_TRAIN_NODES}; redeploy with TRAINER_CONFIG={experiment}"
            )
        if cfg.environment != slime_cfg.environment:
            print(
                f"WARNING: experiment {experiment!r} changes `environment`, which only "
                f"takes effect after a redeploy restarts the Ray cluster."
            )

        cfg.rollout_endpoint_url = provider_url
        cfg.api_shim_base_url = provider_url
        # One deployed prefix is one append-only chain. Starting an independent
        # run on reused control/upload state is a correctness error, not a reset.
        hooks.assert_clean_transport(hooks.ShimConfig.from_env(cfg))
        cfg.custom_config_path = _custom_config(
            cfg, rollout_num_engines=getattr(run, "ROLLOUT_NUM_ENGINES", 1)
        )
        helpers.prepare_slime_config(cfg, tempfile.mkdtemp())
        cmd = helpers.build_train_cmd(cfg, SLIME_ROOT)

        print(
            f"Training {experiment}: nodes={N_TRAIN_NODES}, rollout_endpoint={cfg.rollout_endpoint_url}"
        )
        print(f"Command: {cmd}")
        subprocess.run(["bash", "-lc", cmd], check=True)


@utility_app.function(
    image=trainer_image,
    volumes={str(DATA_PATH): data_volume},
    timeout=2 * 60 * MINUTES,
    secrets=[modal.Secret.from_name(exp.HF_SECRET_NAME)],
    include_source=False,
)
def prepare_dataset() -> None:
    data_volume.reload()
    slime_cfg.prepare_data()
    data_volume.commit()


@preflight_app.function(
    image=control_image,
    volumes={
        str(provider_app.S3_TRANSPORT_MOUNT_PATH): provider_app.s3_transport_mount
    },
    secrets=[modal.Secret.from_name(exp.SHIM_SECRET_NAME)],
    timeout=5 * MINUTES,
    include_source=False,
)
def check_transport_clean() -> None:
    try:
        hooks.assert_clean_transport(hooks.ShimConfig.from_env())
    except Exception as exc:
        print(f"VERDICT=FAIL clean_transport=0 error={type(exc).__name__}:{exc}")
        raise
    print("VERDICT=PASS clean_transport=1")


@preflight_app.local_entrypoint()
def launch_train(experiment: str = TRAINER_CONFIG) -> None:
    from modal.exception import NotFoundError

    run = importlib.import_module(
        f"cookbook.standalone_rollouts.slime.configs.{experiment}"
    )
    check_transport_clean.remote()
    trainer = modal.Cls.from_name(APP_NAME, Trainer.__name__)()
    try:
        call = trainer.train.spawn(experiment, run.slime.to_payload())
    except NotFoundError:
        raise SystemExit(
            f"App {APP_NAME!r} is not deployed. Run:\n"
            "  uv run --extra modal modal deploy -e <env> --strategy recreate "
            "-m cookbook.standalone_rollouts.slime.modal_train::app"
        )
    print(f"Spawned train({experiment!r}) on {APP_NAME}: {call.object_id}")


@preflight_app.local_entrypoint()
def print_trainer_secret_template() -> None:
    print(
        "\n".join(
            [
                f"modal secret create -e <env> {exp.SHIM_SECRET_NAME} \\",
                "  STITCH_SHIM_API_KEY=... \\",
                "  STITCH_SHIM_PROVIDER_MODEL=moonlight \\",
                "  STITCH_SHIM_PROVIDER_DEPLOYMENT=rollout-prod \\",
                "  STITCH_SHIM_BASE_SNAPSHOT_IDENTITY=...",
            ]
        )
    )


def _validate_trainer_environment() -> None:
    required = ("STITCH_SHIM_TRANSPORT_ROOT",)
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing required runtime setting(s): {', '.join(missing)}. "
            f"Create/update the {exp.SHIM_SECRET_NAME!r} Modal secret or redeploy the app."
        )
    transport_root = Path(os.environ["STITCH_SHIM_TRANSPORT_ROOT"])
    if not transport_root.exists():
        raise RuntimeError(f"Transport root is not mounted: {transport_root}")


def _trainer_environment() -> dict[str, str]:
    env = {str(key): str(value) for key, value in slime_cfg.environment.items()}
    existing = env.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
    paths = ["/root/Megatron-LM", "/root"]
    paths.extend(path for path in existing.split(os.pathsep) if path)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return env


def _custom_config(cfg: SlimeConfig, *, rollout_num_engines: int) -> dict:
    current = getattr(cfg, "custom_config_path", None)
    if isinstance(current, dict):
        data = dict(current)
    elif current:
        raise ValueError(
            "This trainer app expects custom_config_path to be unset or a dict before materialization."
        )
    else:
        data = {}
    data["rollout_num_engines"] = int(rollout_num_engines)
    return data
