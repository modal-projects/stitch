"""Disaggregated SLIME training on Modal.

A Modal Flash pool of SGLang servers handles rollouts; a clustered Trainer
runs SLIME on Ray and publishes sparse weight deltas through a Modal Volume
bulletin board that the rollout servers sync from.

Run all commands as modules from the repo root, e.g.:

    uv run --extra modal modal deploy -m cookbook.slime_disagg.modal_train
"""

from __future__ import annotations

import importlib
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import modal
import modal.experimental

from cookbook.slime_disagg import helpers
from cookbook.slime_disagg.configs.base import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    SlimeConfig,
)
from stitch.providers.modal import resolve_flash_gateway_url

# The one deploy-time knob: which experiment config this app is built around.
# `modal deploy` takes no function arguments, so the selection has to come from
# the environment; the image records the same value so containers reconstruct
# the identical app. Everything else is configured in the experiment modules.
EXPERIMENT = os.environ.get("EXPERIMENT_CONFIG", "qwen3_4b_delta_flash")
exp = importlib.import_module(f"cookbook.slime_disagg.configs.{EXPERIMENT}")

modal_cfg = exp.modal
slime_cfg = exp.slime

APP_NAME = exp.APP_NAME
MODEL_NAME = slime_cfg.hf_checkpoint
ROLLOUT_CONCURRENCY = slime_cfg.sglang_server_concurrency
N_TRAIN_NODES = helpers.training_nodes(slime_cfg)

MINUTES = 60
SIDECAR_PORT = 8000
SGLANG_PORT = 8001
RAY_PORT = 6379
SERVER_STARTUP_TIMEOUT = 35 * MINUTES
# Ephemeral host-local full HF checkpoint the sidecar patches in place per delta
# (seeded once from the base; rebuilt from base on a cold container).
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"

SLIME_IMAGE_TAG = "slimerl/slime:nightly-dev-20260527a"
SLIME_ROOT = "/root/slime"
# Fork branch with the generic HTTP rollout endpoint and publish-only
# disk-delta hooks that this example drives.
SLIME_REPO_URL = "https://github.com/modal-projects/slime.git"
# Pin to an exact commit, not the branch tip: the build's `git fetch ... &&
# checkout` is a cached image layer, so a moving branch tip silently leaves the
# container on a stale slime. Bump this SHA to roll slime forward.
SLIME_REPO_REF = "11bb0fa48aa37d5c54fe297143c6bc1d40f311bf"

image = (
    modal.Image.from_registry(SLIME_IMAGE_TAG)
    .entrypoint([])
    # The base image bakes in an HF cache; remove it so it cannot shadow the
    # cache volume mounted at the same path.
    .run_commands(f"rm -rf {HF_CACHE_PATH}")
    # Replace the bundled slime with the fork branch.
    .run_commands(
        f"rm -rf {SLIME_ROOT}"
        f" && git clone --depth 1 {SLIME_REPO_URL} {SLIME_ROOT}"
        f" && cd {SLIME_ROOT}"
        f" && git fetch --depth 1 origin {SLIME_REPO_REF}"
        f" && git checkout FETCH_HEAD"
        f" && python3 -m pip install --no-deps -e {SLIME_ROOT}"
    )
    # The base image installs megatron-core as a PEP 660 *strict* editable that
    # exposes only `megatron.core`, hiding `megatron.training` (which slime's
    # megatron backend imports) even though the full tree exists on disk at
    # /root/Megatron-LM. Reinstall in compat editable mode so a .pth puts the
    # whole source tree on the path and `megatron.training` is importable.
    .run_commands(
        "cd /root/Megatron-LM"
        " && python3 -m pip install --no-deps -e . --config-settings editable_mode=compat"
    )
    .pip_install(
        "autoinference-utils==0.2.0",  # SGLang server lifecycle for the rollout pool
        "fastapi",  # stitch sidecar
        "httpx",  # stitch sidecar
        "uvicorn",  # stitch sidecar
        # The sidecar applies disk deltas host-side via slime.utils.disk_delta,
        # which decompresses with zstandard and checksums with xxhash (xxh3-128
        # default) or blake3. slime is installed --no-deps, so add them here.
        "zstandard",
        "xxhash",
        "blake3",
    )
    .env(
        {
            "EXPERIMENT_CONFIG": EXPERIMENT,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    # Local source is mounted when containers start rather than copied into the
    # image, so code changes never trigger an image rebuild. Modal puts /root
    # on PYTHONPATH, which also makes both packages importable from
    # subprocesses (the sidecar, Ray workers). The whole cookbook package is
    # mounted (not just the per-trainer subdir) so the trainer and the
    # `python3 -m cookbook.sidecar` subprocess can import the shared cookbook
    # spine (helpers/hooks/sidecar).
    .add_local_python_source("stitch")
    .add_local_dir(
        Path(__file__).parent.parent,
        remote_path="/root/cookbook",
        ignore=["**/__pycache__"],
    )
)

# Dev iteration: SLIME_LOCAL_DIR overlays a local slime checkout onto the image's
# cloned fork (installed editable at /root/slime), so fork edits take effect on
# container start with no image rebuild or push. Unset by default, so the committed
# example always builds from the pinned SLIME_REPO_REF.
if slime_local := os.environ.get("SLIME_LOCAL_DIR"):
    image = image.add_local_dir(
        slime_local,
        remote_path=SLIME_ROOT,
        ignore=[".git", "**/__pycache__", "**/*.pyc"],
    )


# The rollout pool may need a different serving stack than the trainer — e.g. a
# Blackwell SGLang build that serves native-INT4 Kimi K2.6, which the slime
# trainer image does not provide. An experiment opts in by defining
# build_serving_image(...); otherwise the pool reuses the trainer image (the
# Qwen/Moonlight bf16 and Moonlight-INT4 experiments do). The pool installs no
# trainer package — the delta decode/apply lives in the engine behind
# /pull_weights.
def _select_server_image() -> modal.Image:
    builder = getattr(exp, "build_serving_image", None)
    if builder is None:
        return image
    return builder(
        hf_cache_path=str(HF_CACHE_PATH),
        experiment=EXPERIMENT,
        delta_volume_name=exp.DELTA_VOLUME_NAME,
    )


server_image = _select_server_image()
if slime_local and server_image is not image:
    # Mirror the trainer's dev-iteration overlay onto a dedicated serving image.
    server_image = server_image.add_local_dir(
        slime_local,
        remote_path=SLIME_ROOT,
        ignore=[".git", "**/__pycache__", "**/*.pyc"],
    )

with server_image.imports():
    from autoinference_utils.endpoint import SGLangEndpoint, warmup_chat_completions


hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("slime-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("slime-checkpoints", create_if_missing=True)
delta_volume = modal.Volume.from_name(
    exp.DELTA_VOLUME_NAME, create_if_missing=True, version=2
)

train_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
    exp.DELTA_BULLETIN_ROOT: delta_volume,
}

app = modal.App(APP_NAME)

SGLANG_SERVER_ARGS = {
    "--served-model-name": MODEL_NAME,
    "--dtype": "bfloat16",
    "--cuda-graph-max-bs": str(ROLLOUT_CONCURRENCY),
    "--max-running-requests": str(ROLLOUT_CONCURRENCY),
    "--trust-remote-code": "",
    # Engine-side /pull_weights reads published versions off the Modal Volume;
    # the hook reloads it first (object stores lack cross-host read-after-write
    # consistency). Reads DELTA_VOLUME_NAME from the serving image env.
    "--custom-pull-weights-pre-read-hook": "stitch.providers.modal.pull_weights_pre_read_hook",
    **exp.SGLANG_SERVER_ARGS,
}

WARMUP_PAYLOAD = {
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Reply with exactly OK."}],
    "max_tokens": 8,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}


@app.cls(
    image=server_image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.rollout_num_gpus_per_engine}",
    cloud=modal_cfg.cloud,
    region=modal_cfg.region,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        exp.DELTA_BULLETIN_ROOT: delta_volume,
    },
    min_containers=modal_cfg.rollout_min_containers,
    timeout=40 * MINUTES,
    scaledown_window=15 * MINUTES,
    include_source=False,
)
@modal.experimental.http_server(
    port=SIDECAR_PORT,
    proxy_regions=modal_cfg.proxy_regions,
    exit_grace_period=25,
    startup_timeout=SERVER_STARTUP_TIMEOUT,
)
@modal.concurrent(target_inputs=ROLLOUT_CONCURRENCY)
class Server:
    """One SGLang rollout server plus the stitch weight-sync sidecar.

    The sidecar proxies rollout traffic, reloads the delta Volume, and applies
    published weight versions so requests pinned to a version are served by
    matching weights.
    """

    @modal.enter()
    def startup(self) -> None:
        self.endpoint = SGLangEndpoint(
            model_path=MODEL_NAME,
            worker_port=SGLANG_PORT,
            tp=slime_cfg.rollout_num_gpus_per_engine,
            extra_server_args=SGLANG_SERVER_ARGS,
            health_timeout=SERVER_STARTUP_TIMEOUT,
            health_poll_interval=10.0,
        )
        self.endpoint.start()
        warmup_chat_completions(
            port=SGLANG_PORT,
            payload=WARMUP_PAYLOAD,
            successful_requests=2,
            request_timeout=120.0,
            max_attempts_per_request=3,
        )
        # The engine materializes each delta version into LOCAL_CHECKPOINT_PATH
        # itself via /pull_weights (seeded from its own --model-path); the
        # sidecar only drives the sync.
        self.sidecar = helpers.start_sglang_sidecar(
            sidecar_port=SIDECAR_PORT,
            sglang_port=SGLANG_PORT,
            bulletin_root=exp.DELTA_BULLETIN_ROOT,
            local_checkpoint_dir=LOCAL_CHECKPOINT_PATH,
            volume_name=exp.DELTA_VOLUME_NAME,
            commit_mode=exp.SIDECAR_COMMIT_MODE,
            debug_requests=getattr(exp, "SIDECAR_DEBUG_REQUESTS", False),
        )
        helpers.wait_http(
            f"http://127.0.0.1:{SIDECAR_PORT}/health",
            self.sidecar,
            SERVER_STARTUP_TIMEOUT,
        )
        print(
            f"Rollout server ready: model={MODEL_NAME}, target_inputs={ROLLOUT_CONCURRENCY}"
        )

    @modal.exit()
    def stop(self) -> None:
        helpers.terminate_process(getattr(self, "sidecar", None))
        if hasattr(self, "endpoint"):
            self.endpoint.stop()


@app.cls(
    image=image,
    gpu=f"{modal_cfg.gpu}:{slime_cfg.actor_num_gpus_per_node}",
    memory=modal_cfg.memory,
    cloud=modal_cfg.cloud,
    region=modal_cfg.region,
    volumes=train_volumes,
    timeout=24 * 60 * MINUTES,
    startup_timeout=20 * MINUTES,
    scaledown_window=30 * MINUTES,
    experimental_options={"efa_enabled": True},
    include_source=False,
)
@modal.experimental.clustered(N_TRAIN_NODES, rdma=True)
class Trainer:
    """SLIME actor cluster. The Ray cluster comes up once per container in
    enter(), so back-to-back training runs reuse it instead of rebuilding it."""

    @modal.enter()
    def start_ray(self) -> None:
        rank, master_addr, my_ip = helpers.get_modal_cluster_context(N_TRAIN_NODES)
        self.rank = rank
        # Ray actors inherit the raylet's environment, so everything the
        # training processes need must be exported before `ray start`.
        os.environ.update(
            {
                "SLIME_HOST_IP": my_ip,
                "SGLANG_HOST_IP": my_ip,
                "HOST_IP": my_ip,
                "MASTER_ADDR": master_addr,
                "RAY_ADDRESS": f"{master_addr}:{RAY_PORT}",
                "no_proxy": f"127.0.0.1,{master_addr},{my_ip}",
                "NO_PROXY": f"127.0.0.1,{master_addr},{my_ip}",
                **slime_cfg.environment,
            }
        )
        if rank == 0:
            helpers.start_ray_head(my_ip, N_TRAIN_NODES, ray_port=RAY_PORT)
        else:
            helpers.start_ray_worker(my_ip, master_addr, ray_port=RAY_PORT)

    @modal.method()
    def train(self, experiment: str, payload: dict) -> None:
        """Run one training job from a SlimeConfig payload (see to_payload()).

        The config arrives as data instead of a module name, so launch_train
        can resolve experiments from the local working tree and new or edited
        configs run without a redeploy.
        """
        for volume in train_volumes.values():
            volume.reload()
        # Rank 0 drives the run; the cluster stays up until its call returns,
        # so the other ranks only need their Ray workers, started in enter().
        if self.rank != 0:
            return

        cfg = SlimeConfig.from_payload(payload)
        if helpers.training_nodes(cfg) != N_TRAIN_NODES:
            raise ValueError(
                f"experiment {experiment!r} needs {helpers.training_nodes(cfg)} node(s) but this app "
                f"was deployed with {N_TRAIN_NODES}; deploy it as its own app with EXPERIMENT_CONFIG={experiment}"
            )
        if cfg.environment != slime_cfg.environment:
            # Ray inherited the deploy-time environment when enter() started it.
            print(
                f"WARNING: experiment {experiment!r} changes `environment`, which only "
                f"takes effect after a redeploy restarts the Ray cluster."
            )

        cfg.rollout_endpoint_url = resolve_flash_gateway_url(APP_NAME, Server.__name__)
        # Fresh run id per launch: slime writes this run's chain under a partition
        # (<bulletin_root>/<run_id>/weight_v{N}/), while the canonical pointer at
        # <bulletin_root>/latest is self-identifying (<run_id>/weight_vN). So a new
        # run never overwrites a finished run's version dirs and its pointer move is
        # a forward step, not a colliding rewind — no manual bulletin reset needed.
        run_id = uuid.uuid4().hex[:12]
        cfg.update_weight_disk_dir = f"{exp.DELTA_BULLETIN_ROOT}/{run_id}"
        # stitch's publish hooks read these off the slime args namespace.
        hook_knobs = {
            "update_weight_delta_volume_name": exp.DELTA_VOLUME_NAME,
            "rollout_modal_flash_app_name": APP_NAME,
            "rollout_modal_flash_server_cls_name": Server.__name__,
            "run_id": run_id,
        }
        cfg.custom_config_path = hook_knobs
        # prepare_slime_config materializes the YAML_CONFIG_FIELDS dicts (including
        # custom_config_path) to temp YAML file PATHS on cfg — keep hook_knobs for
        # the claim below, which needs the mapping, not the path.
        helpers.prepare_slime_config(cfg, tempfile.mkdtemp())
        cmd = helpers.build_train_cmd(cfg, SLIME_ROOT)

        # Claim the pool for this run *before* slime starts publishing: write the
        # empty pointer (<run_id>/weight_v000000) and wake the pool so every
        # replica resets to base now, closing the window where a replica could
        # reconcile to a finished run's stale high-water version.
        from cookbook.slime_disagg import hooks

        hooks.claim_pool(
            SimpleNamespace(update_weight_disk_dir=cfg.update_weight_disk_dir, **hook_knobs)
        )

        print(
            f"Training {experiment}: nodes={N_TRAIN_NODES}, rollout_endpoint={cfg.rollout_endpoint_url}"
        )
        print(f"Command: {cmd}")
        subprocess.run(["bash", "-lc", cmd], check=True)


@app.function(
    image=image,
    volumes={str(HF_CACHE_PATH): hf_cache_volume},
    timeout=2 * 60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    include_source=False,
)
def download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_NAME)
    hf_cache_volume.commit()


@app.function(
    image=image,
    volumes={str(DATA_PATH): data_volume},
    timeout=2 * 60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    include_source=False,
)
def prepare_dataset() -> None:
    data_volume.reload()
    slime_cfg.prepare_data()
    data_volume.commit()


@app.local_entrypoint()
def launch_train(experiment: str = EXPERIMENT) -> None:
    """Resolve an experiment from the local working tree and spawn it on the
    deployed app. Training args ship as data, so new or edited configs run
    without a redeploy; infrastructure changes (GPU, nodes, pool size,
    Volume names) still require one."""
    from modal.exception import NotFoundError

    run = importlib.import_module(f"cookbook.slime_disagg.configs.{experiment}")
    if run.DELTA_VOLUME_NAME != exp.DELTA_VOLUME_NAME:
        raise SystemExit(
            f"Experiment {experiment!r} owns Volume {run.DELTA_VOLUME_NAME!r}, but app {APP_NAME!r} "
            f"mounts {exp.DELTA_VOLUME_NAME!r}. Deploy it as its own app with EXPERIMENT_CONFIG={experiment}."
        )

    try:
        trainer = modal.Cls.from_name(APP_NAME, Trainer.__name__)()
        call = trainer.train.spawn(experiment, run.slime.to_payload())
    except NotFoundError:
        raise SystemExit(
            f"App {APP_NAME!r} is not deployed. Run:\n"
            f"  uv run --extra modal modal deploy -m cookbook.slime_disagg.modal_train"
        )
    print(f"Spawned train({experiment!r}) on {APP_NAME}: {call.object_id}")


@app.local_entrypoint()
def smoke_flash_pool(
    weight_version: int = 0, timeout_seconds: int = 30 * MINUTES
) -> None:
    """Check that the deployed Flash pool serves completions at the expected
    weight version, via the gateway and each container directly."""
    helpers.smoke_flash_pool(
        app_name=APP_NAME,
        cls_name=Server.__name__,
        model_name=MODEL_NAME,
        weight_version=weight_version,
        expect_min_containers=modal_cfg.rollout_min_containers,
        timeout_seconds=timeout_seconds,
    )
