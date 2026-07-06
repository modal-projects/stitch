"""Disaggregated miles training on Modal (NVFP4, Blackwell end to end).

A Modal Flash pool of SGLang servers handles rollouts; a clustered Trainer runs
miles on Ray and publishes XOR weight deltas through a Modal Volume bulletin
board that the rollout servers sync from. The miles twin of
cookbook/slime_disagg/modal_train.py — same two-half architecture, same stitch
sidecar/bulletin machinery; the trainer is miles instead of slime and the
precision is NVFP4 (so BOTH halves are Blackwell — see the config docstrings).

Run all commands as modules from the repo root, e.g.:

    uv run --extra modal modal deploy -m cookbook.miles_disagg.modal_train
"""

from __future__ import annotations

import importlib
import os
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import modal
import modal.experimental

from cookbook.miles_disagg import helpers
from cookbook.miles_disagg.configs.base import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    HF_CACHE_PATH,
    PREP_PATH,
    MilesConfig,
)
from stitch.providers.modal import resolve_flash_gateway_url

# The one deploy-time knob: which experiment config this app is built around.
EXPERIMENT = os.environ.get("EXPERIMENT_CONFIG", "moonlight_nvfp4_disagg")
exp = importlib.import_module(f"cookbook.miles_disagg.configs.{EXPERIMENT}")

modal_cfg = exp.modal
miles_cfg = exp.miles

# The rollout Server pool warm-boots `min_containers` replicas the moment this app
# is materialized — by a deploy OR by `modal run` of ANY function in it. Those
# replicas serve the prepared rollout base (miles_cfg.hf_checkpoint), so before that
# base exists they crash-loop on a missing --model-path. prepare_checkpoints builds
# the base, so it must run with the pool down: POOL_MIN_CONTAINERS=0.
POOL_MIN_CONTAINERS = int(os.environ.get("POOL_MIN_CONTAINERS", modal_cfg.rollout_min_containers))

APP_NAME = exp.APP_NAME
MODEL_NAME = miles_cfg.hf_checkpoint
ROLLOUT_CONCURRENCY = modal_cfg.rollout_target_inputs or miles_cfg.sglang_server_concurrency
N_TRAIN_NODES = helpers.training_nodes(miles_cfg)

MINUTES = 60
SIDECAR_PORT = 8000
SGLANG_PORT = 8001
RAY_PORT = 6379
SERVER_STARTUP_TIMEOUT = 35 * MINUTES
LOCAL_CHECKPOINT_PATH = "/local-checkpoint"

# radixark/miles bakes Megatron-LM (miles-main: native --fp4-format NVFP4
# BlockScaling) + TransformerEngine. NVFP4 QAT requires TE >= 2.7.0.dev0 and
# Blackwell; verify the image's TE on a warm B200 during bring-up.
MILES_IMAGE_TAG = "radixark/miles:latest"
MILES_ROOT = "/root/miles"
# megatron.core is pip-installed in the base image, but megatron.training lives
# only in this source tree; miles' own launcher puts it on PYTHONPATH, and so
# must we (we run train_async.py directly rather than via execute_train).
MEGATRON_PATH = "/root/Megatron-LM"
# Fork commit with the disaggregated-rollout features (opaque HTTP endpoint,
# publish-only disk-delta, request hook) plus the NVFP4 and GLM-Air FP8 fixes
# this cookbook needs. Pin to an exact commit, not the branch tip (cached image
# layer); push the ref to modal-projects/miles before deploying.
MILES_REPO_URL = "https://github.com/modal-projects/miles.git"
MILES_REPO_REF = "852fb9a5cfb3c4630691b643ed354a9703ff3722"

# Build-time bake of the megatron R3 dispatch fix (see the .run_commands call
# below). Kept as a string so the build step has no host-file dependency; the
# replace targets a single, stable line and reports how many sites it hit.
_R3_DISPATCH_TARGET = "/root/Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py"
_R3_DISPATCH_OLD = "self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk"
_R3_DISPATCH_NEW = (
    "self.num_out_tokens = num_local_tokens_per_expert.sum()\n"
    '            self._maybe_update_cuda_sync_point("before_permutation_1")'
)
_R3_DISPATCH_BAKE_PY = (
    "import pathlib;"
    f"p=pathlib.Path({_R3_DISPATCH_TARGET!r});"
    "s=p.read_text();"
    f"n=s.count({_R3_DISPATCH_OLD!r});"
    f"p.write_text(s.replace({_R3_DISPATCH_OLD!r}, {_R3_DISPATCH_NEW!r}));"
    "print(f'[R3 bake] patched {n} dispatch site(s) in token_dispatcher.py')"
)

TORCH_DIST_CONVERT_WRAPPER = "/root/convert_hf_to_torch_dist_modal.py"

image = (
    modal.Image.from_registry(MILES_IMAGE_TAG)
    .entrypoint([])
    # RDMA/EFA userspace stack. The miles Dockerfile installs only nvtop/rsync/etc;
    # without the full ibverbs stack NCCL silently falls back to TCP sockets on the
    # multi-node trainer (catastrophic throughput) and emits libibverbs driver
    # warnings. These let the AWS-OFI/libfabric path NCCL uses under rdma=True bind EFA.
    .apt_install(
        "libibverbs-dev",
        "libibverbs1",
        "libhwloc-dev",
        "libnl-route-3-200",
    )
    # The base image bakes in an HF cache; remove it so it cannot shadow the
    # cache volume mounted at the same path.
    .run_commands(f"rm -rf {HF_CACHE_PATH}")
    # Replace the bundled miles with the fork branch (keeps the baked Megatron-LM
    # + TE, which provide the NVFP4 training path).
    .run_commands(
        f"rm -rf {MILES_ROOT}"
        f" && git clone {MILES_REPO_URL} {MILES_ROOT}"
        f" && cd {MILES_ROOT}"
        f" && git fetch origin {MILES_REPO_REF}"
        f" && git checkout FETCH_HEAD"
        f" && python3 -m pip install --no-deps -e {MILES_ROOT}"
    )
    # R3 routing-replay fix for the baked radixark/Megatron-LM. The dropless MoE
    # token dispatcher computes num_out_tokens = routing_map.size(0) * topk, which
    # assumes topk distinct experts/token — false under routing replay (duplicate
    # / -1 experts collapse in the boolean map), causing the EP all-to-all to
    # crash with "Split sizes doesn't match total dim 0 size". Derive it from the
    # actual per-expert counts instead (identical to size*topk in the dense
    # non-replay case). Idempotent: a no-op once the fork itself ships the fix.
    .run_commands(f"python3 -c {shlex.quote(_R3_DISPATCH_BAKE_PY)}")
    .pip_install(
        "fastapi",  # stitch sidecar (rollout pool reuses this image only if no serving image)
        "httpx",
        "uvicorn",
        # miles applies disk deltas host-side via miles.utils.disk_delta; ensure
        # its codec deps are present even when miles is reinstalled --no-deps.
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
)

if getattr(exp, "USE_MODAL_TORCH_DIST_WRAPPER", False):
    image = image.add_local_file(
        "cookbook/miles_disagg/convert_hf_to_torch_dist_modal.py",
        TORCH_DIST_CONVERT_WRAPPER,
        copy=True,
    )

# Local source mounted at container start (no rebuild on code edits). Modal puts
# /root on PYTHONPATH, so both packages import from subprocesses (the sidecar, Ray
# workers). MUST come after any .run_commands() above (Modal forbids run_commands
# after a non-copy local mount). The whole cookbook package is mounted (not just
# the per-trainer subdir) so the trainer and the `python3 -m
# cookbook.miles_disagg.sidecar` subprocess can import the shared cookbook spine
# (helpers/hooks/sidecar) the thin adapters delegate to.
image = image.add_local_python_source("stitch").add_local_dir(
    Path(__file__).parent.parent,
    remote_path="/root/cookbook",
    ignore=["**/__pycache__"],
)

# Dev iteration: MILES_LOCAL_DIR overlays a local miles checkout onto the image's
# cloned fork, so fork edits (e.g. the NVFP4 export-dispatch fix) take effect on
# container start with no image rebuild or push. Unset by default.
if miles_local := os.environ.get("MILES_LOCAL_DIR"):
    image = image.add_local_dir(
        miles_local,
        remote_path=MILES_ROOT,
        ignore=[".git", "**/__pycache__", "**/*.pyc"],
    )


def _select_server_image() -> modal.Image:
    """The rollout pool needs a Blackwell SGLang build that serves NVFP4; an
    experiment opts in via build_serving_image(...). Either way the pool pins the
    trainer's exact miles ref so the sidecar's disk_delta matches the encoder."""
    builder = getattr(exp, "build_serving_image", None)
    if builder is None:
        return image
    return builder(
        trainer_repo_url=MILES_REPO_URL,
        trainer_repo_ref=MILES_REPO_REF,
        trainer_root=MILES_ROOT,
        hf_cache_path=str(HF_CACHE_PATH),
        experiment=EXPERIMENT,
    )


server_image = _select_server_image()
if miles_local and server_image is not image:
    server_image = server_image.add_local_dir(
        miles_local,
        remote_path=MILES_ROOT,
        ignore=[".git", "**/__pycache__", "**/*.pyc"],
    )

with server_image.imports():
    from autoinference_utils.endpoint import SGLangEndpoint, warmup_chat_completions


hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("miles-data", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("miles-checkpoints", create_if_missing=True)
prep_volume = modal.Volume.from_name("miles-prep-checkpoints", create_if_missing=True)
# Persists SGLang's kernel caches across cold starts. SGLang nests flashinfer's
# FP4 MoE autotuner + JIT cache (and DeepGemm etc.) under its own cache dir, so we
# mount the whole tree. Without it every elastic scale-up re-runs `[AutoTuner]
# Tuning trtllm_fp4_block_scale_moe` and recompiles kernels (minutes per Server).
sglang_cache_volume = modal.Volume.from_name("miles-sglang-cache", create_if_missing=True)
delta_volume = modal.Volume.from_name(exp.DELTA_VOLUME_NAME, create_if_missing=True, version=2)

# SGLang's default cache dir ($HOME/.cache/sglang), which holds the nested
# flashinfer/DeepGemm caches; mounted so they survive container teardown.
SGLANG_CACHE_PATH = "/root/.cache/sglang"

train_volumes = {
    str(HF_CACHE_PATH): hf_cache_volume,
    str(DATA_PATH): data_volume,
    str(CHECKPOINTS_PATH): checkpoints_volume,
    str(PREP_PATH): prep_volume,
    exp.DELTA_BULLETIN_ROOT: delta_volume,
}

app = modal.App(APP_NAME)

SGLANG_SERVER_ARGS = {
    "--served-model-name": MODEL_NAME,
    "--dtype": "bfloat16",
    "--cuda-graph-max-bs": str(ROLLOUT_CONCURRENCY),
    "--max-running-requests": str(ROLLOUT_CONCURRENCY),
    "--trust-remote-code": "",
    # NVFP4 is driven by the served checkpoint's own quant config — no
    # --quantization flag. MLA/cache/routing extras come from the experiment.
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
    gpu=f"{modal_cfg.gpu}:{miles_cfg.rollout_num_gpus_per_engine}",
    cloud=modal_cfg.cloud,
    region=modal_cfg.region,
    volumes={
        str(HF_CACHE_PATH): hf_cache_volume,
        str(PREP_PATH): prep_volume,
        SGLANG_CACHE_PATH: sglang_cache_volume,
        exp.DELTA_BULLETIN_ROOT: delta_volume,
    },
    min_containers=POOL_MIN_CONTAINERS,
    max_containers=getattr(modal_cfg, "rollout_max_containers", None),
    timeout=40 * MINUTES,
    scaledown_window=15 * MINUTES,
    # The sidecar copies the full served base (~591 GB for K2.6) to
    # /local-checkpoint on ephemeral disk; Modal's default is too small.
    ephemeral_disk=modal_cfg.rollout_ephemeral_disk_mib,
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
        helpers.apply_sglang_runtime_patches(list(getattr(exp, "SGLANG_RUNTIME_PATCHES", [])))
        self.endpoint = SGLangEndpoint(
            model_path=MODEL_NAME,
            worker_port=SGLANG_PORT,
            tp=miles_cfg.rollout_num_gpus_per_engine,
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
        # The served base is a prepared directory (MODEL_NAME is its absolute
        # path); deltas are applied host-side onto a copy of it.
        self.sidecar = helpers.start_sglang_sidecar(
            sidecar_port=SIDECAR_PORT,
            sglang_port=SGLANG_PORT,
            bulletin_root=exp.DELTA_BULLETIN_ROOT,
            local_checkpoint_dir=LOCAL_CHECKPOINT_PATH,
            base_checkpoint_dir=MODEL_NAME,
            volume_name=exp.DELTA_VOLUME_NAME,
            commit_mode=exp.SIDECAR_COMMIT_MODE,
            debug_requests=getattr(exp, "SIDECAR_DEBUG_REQUESTS", False),
        )
        helpers.wait_http(
            f"http://127.0.0.1:{SIDECAR_PORT}/health",
            self.sidecar,
            SERVER_STARTUP_TIMEOUT,
        )
        print(f"Rollout server ready: model={MODEL_NAME}, target_inputs={ROLLOUT_CONCURRENCY}")

    @modal.exit()
    def stop(self) -> None:
        helpers.terminate_process(getattr(self, "sidecar", None))
        if hasattr(self, "endpoint"):
            self.endpoint.stop()


# Multi-node clustering (RDMA + EFA) is applied only when N_TRAIN_NODES > 1:
# Modal requires clustered B200 functions to use all 8 GPUs per node, so a
# single-node trainer (e.g. the Moonlight de-risk on B200:4) runs as a plain cls.
_TRAINER_KWARGS = dict(
    image=image,
    gpu=f"{modal_cfg.gpu}:{miles_cfg.actor_num_gpus_per_node}",
    memory=modal_cfg.memory,
    cloud=modal_cfg.cloud,
    region=modal_cfg.region,
    volumes=train_volumes,
    # Ray (address="auto") spills objects + writes logs under /tmp/ray on the node's
    # ephemeral disk; Modal's default is far too small for a multi-hour 128-GPU run and
    # progressively ENOSPC'd. Give the B200:8 nodes' local NVMe room.
    ephemeral_disk=modal_cfg.trainer_ephemeral_disk_mib,
    timeout=24 * 60 * MINUTES,
    startup_timeout=20 * MINUTES,
    scaledown_window=30 * MINUTES,
    include_source=False,
)
if N_TRAIN_NODES > 1:
    _TRAINER_KWARGS["experimental_options"] = {"efa_enabled": True}


class Trainer:
    """miles actor cluster. The Ray cluster comes up once per container in
    enter(), so back-to-back training runs reuse it instead of rebuilding it."""

    @modal.enter()
    def start_ray(self) -> None:
        rank, master_addr, my_ip = helpers.get_modal_cluster_context(N_TRAIN_NODES)
        self.rank = rank
        # Per-node host-RAM trace: the trainer can OOM-kill at host-RAM exhaustion
        # (the publish/update_weights gather is the peak), and Modal's kill leaves no
        # peak behind. This makes `modal app logs -f` show which node/phase blows the
        # ~1.95 TiB budget. Runs on every node (SPMD enter()).
        helpers.start_host_mem_monitor()
        os.environ.update(
            {
                "MILES_HOST_IP": my_ip,
                "SGLANG_HOST_IP": my_ip,
                "HOST_IP": my_ip,
                "MASTER_ADDR": master_addr,
                "RAY_ADDRESS": f"{master_addr}:{RAY_PORT}",
                "no_proxy": f"127.0.0.1,{master_addr},{my_ip}",
                "NO_PROXY": f"127.0.0.1,{master_addr},{my_ip}",
                # Expose megatron.training (source-only) before `ray start`, so both
                # the launch subprocess and the Ray actors inherit it. Prepend to
                # preserve Modal's existing PYTHONPATH (/pkg, /root for stitch).
                "PYTHONPATH": f"{MEGATRON_PATH}:{os.environ.get('PYTHONPATH', '')}",
                **miles_cfg.environment,
            }
        )
        if rank == 0:
            helpers.start_ray_head(my_ip, N_TRAIN_NODES, ray_port=RAY_PORT)
        else:
            helpers.start_ray_worker(my_ip, master_addr, ray_port=RAY_PORT)

    @modal.method()
    def train(self, experiment: str, payload: dict) -> None:
        """Run one training job from a MilesConfig payload (see to_payload())."""
        for volume in train_volumes.values():
            volume.reload()

        cfg = MilesConfig.from_payload(payload)
        # te_precision_config_file is re-read on every Ray actor during model build,
        # so it must exist at an identical local path on all 16 nodes. Materialize it
        # here (SPMD: train() runs on every container) before the rank-0 gate; node 0
        # then embeds this deterministic path in the args. (prepare_miles_config below
        # skips it once it's a path string, not a dict.)
        helpers.materialize_node_local_yaml(cfg, "te_precision_config_file")
        if self.rank != 0:
            return

        if helpers.training_nodes(cfg) != N_TRAIN_NODES:
            raise ValueError(
                f"experiment {experiment!r} needs {helpers.training_nodes(cfg)} node(s) but this app "
                f"was deployed with {N_TRAIN_NODES}; deploy it as its own app with EXPERIMENT_CONFIG={experiment}"
            )
        if cfg.environment != miles_cfg.environment:
            print(
                f"WARNING: experiment {experiment!r} changes `environment`, which only "
                f"takes effect after a redeploy restarts the Ray cluster."
            )

        cfg.rollout_endpoint_url = resolve_flash_gateway_url(APP_NAME, Server.__name__)
        # Fresh run id per launch: miles writes this run's chain under a partition
        # (<bulletin_root>/<run_id>/weight_v{N}/), while the canonical pointer at
        # <bulletin_root>/latest is self-identifying (<run_id>/weight_vN), so a new
        # run is a forward pointer move, never a colliding rewind.
        run_id = uuid.uuid4().hex[:12]
        cfg.update_weight_disk_dir = f"{exp.DELTA_BULLETIN_ROOT}/{run_id}"
        if getattr(cfg, "save_interval", None) is None:
            cfg.load = None
            cfg.save = None
            cfg.save_hf = None
        else:
            cfg.load = f"{CHECKPOINTS_PATH}/{run_id}/checkpoints"
            cfg.save = f"{CHECKPOINTS_PATH}/{run_id}/checkpoints"
        # Merge the dynamic bulletin identity into custom_config_path (which already
        # carries the request-gating knobs). miles setattr's every key onto args,
        # so the publish + request hooks read them via getattr(args, ...).
        existing = dict(getattr(cfg, "custom_config_path", {}) or {})
        custom_config = {
            **existing,
            "update_weight_delta_volume_name": exp.DELTA_VOLUME_NAME,
            "rollout_modal_flash_app_name": APP_NAME,
            "rollout_modal_flash_server_cls_name": Server.__name__,
            "run_id": run_id,
        }
        cfg.custom_config_path = custom_config
        helpers.prepare_miles_config(cfg, tempfile.mkdtemp())
        cmd = helpers.build_train_cmd(cfg, MILES_ROOT)

        # Claim the pool for this run *before* miles starts publishing: write the
        # empty pointer (<run_id>/weight_v000000) and wake the pool so every
        # replica resets to base now, closing the window where a replica could
        # reconcile to a finished run's stale high-water version.
        from cookbook.miles_disagg import hooks

        hooks.claim_pool(SimpleNamespace(update_weight_disk_dir=cfg.update_weight_disk_dir, **custom_config))

        print(f"Training {experiment}: nodes={N_TRAIN_NODES}, rollout_endpoint={cfg.rollout_endpoint_url}")
        print(f"Command: {cmd}")
        # Tee the full training output to a committed Volume file so failures are
        # observable after the fact — the app-logs buffer scrolls past tracebacks.
        # set -o pipefail preserves the trainer's exit code through the tee.
        log_path = f"{CHECKPOINTS_PATH}/{run_id}/train.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        teed = f"set -o pipefail; ({cmd}) 2>&1 | tee {log_path}"
        try:
            subprocess.run(["bash", "-lc", teed], check=True)
        finally:
            try:
                checkpoints_volume.commit()
                print(f"Train log committed to volume miles-checkpoints at {run_id}/train.log")
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: could not commit train log: {exc}")


# Apply clustering only for genuine multi-node trainers (see _TRAINER_KWARGS).
if N_TRAIN_NODES > 1:
    Trainer = modal.experimental.clustered(N_TRAIN_NODES, rdma=True)(Trainer)
Trainer = app.cls(**_TRAINER_KWARGS)(Trainer)


@app.function(
    image=image,
    gpu=f"{modal_cfg.gpu}:1",
    volumes={str(HF_CACHE_PATH): hf_cache_volume, str(PREP_PATH): prep_volume},
    timeout=6 * 60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    include_source=False,
)
def prepare_checkpoints() -> None:
    """Build the bf16 masters + served NVFP4 base on a GPU (NVFP4 quant needs CUDA).

    Lifecycle (see the config docstrings):
      1. masters (bf16): if the source ships quantized (Kimi INT4), dequantize
         with tools/convert_kimi_int4_to_bf16.py; if bf16 (Moonlight), the
         downloaded checkpoint IS the masters.
      2. served NVFP4 base: tools/convert_hf_to_nvfp4.py over the masters, so the
         served packing == the trainer's export packing by construction.
    """
    if getattr(exp, "DISABLE_HF_XET", False):
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
    if getattr(exp, "DISABLE_HF_TRANSFER", False):
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

    from huggingface_hub import snapshot_download

    prep_volume.reload()
    tag = exp.MODEL_TAG
    bf16_dir = f"{PREP_PATH}/{tag}/bf16"
    fp8_dir = f"{PREP_PATH}/{tag}/fp8"
    nvfp4_dir = f"{PREP_PATH}/{tag}/nvfp4"
    served_checkpoint_format = getattr(exp, "SERVED_CHECKPOINT_FORMAT", "nvfp4")
    if served_checkpoint_format not in {"bf16", "fp8", "nvfp4"}:
        raise SystemExit(f"Unsupported SERVED_CHECKPOINT_FORMAT={served_checkpoint_format!r}")
    tools = f"{MILES_ROOT}/tools"

    def _staged(final_dir: str, build) -> None:
        """Build into a sibling .partial dir and atomically rename on success, so
        an interrupted multi-hour step never leaves a half-built dir that the
        `already done?` check below would mistake for complete (this run's first
        attempt died mid-dequant with 7/64 files written)."""
        if os.path.isdir(final_dir) and os.listdir(final_dir):
            print(f"reusing existing {final_dir}")
            return
        partial = f"{final_dir}.partial"
        subprocess.run(["rm", "-rf", partial], check=True)
        os.makedirs(partial, exist_ok=True)
        build(partial)
        os.rename(partial, final_dir)

    src = snapshot_download(exp.SOURCE_MODEL)
    is_int4 = "int4" in exp.SOURCE_MODEL.lower() or _is_int4(src)

    # 1. masters (bf16)
    def _build_bf16(out: str) -> None:
        if is_int4:
            subprocess.run(
                ["python", f"{tools}/convert_kimi_int4_to_bf16.py", "--model-dir", src, "--output-dir", out],
                check=True,
            )
        else:
            # bf16 source IS the masters; dereference (-L) so the prep dir holds
            # real files, not symlinks into the (separate) HF cache volume.
            subprocess.run(f"cp -aL {src}/. {out}/", shell=True, check=True)
        # convert_kimi_int4_to_bf16 copies the source config verbatim, so the
        # dequantized masters still carry the source's INT4 quantization_config
        # (for Kimi VLMs it's nested under text_config). Strip it: the masters are
        # bf16, and convert_hf_to_nvfp4 inherits text_config verbatim — leaving it
        # would make the served NVFP4 base claim compressed-tensors INT4 there and
        # diverge from the clean served base (transformers/SGLang could then pick
        # the wrong quant loader for the language model).
        _strip_stale_quant_config(os.path.join(out, "config.json"))

    _staged(bf16_dir, _build_bf16)

    if served_checkpoint_format == "bf16":
        prep_volume.commit()
        print(f"Prepared masters={bf16_dir} served_base={bf16_dir}")
        return

    if served_checkpoint_format == "fp8":
        fp8_source_model = getattr(exp, "ROLLOUT_SOURCE_MODEL", None)
        if not fp8_source_model:
            raise SystemExit("SERVED_CHECKPOINT_FORMAT='fp8' requires ROLLOUT_SOURCE_MODEL")

        def _build_fp8(out: str) -> None:
            fp8_src = snapshot_download(fp8_source_model)
            subprocess.run(f"cp -aL {fp8_src}/. {out}/", shell=True, check=True)

        _staged(fp8_dir, _build_fp8)
        prep_volume.commit()
        print(f"Prepared masters={bf16_dir} served_base={fp8_dir}")
        return

    # 2. served NVFP4 base (miles' TE-direct quantizer). bf16
    # carve-outs for the dense first / last layers must match the trainer's
    # --num-layers-at-start/end-in-bf16 so the served base == the export layout.
    _nvfp4_carveouts = []
    if (n := getattr(miles_cfg, "num_layers_at_start_in_bf16", None)) is not None:
        _nvfp4_carveouts += ["--num-layers-at-start-in-bf16", str(n)]
    if (n := getattr(miles_cfg, "num_layers_at_end_in_bf16", None)) is not None:
        _nvfp4_carveouts += ["--num-layers-at-end-in-bf16", str(n)]
    _staged(
        nvfp4_dir,
        lambda out: subprocess.run(
            ["python", f"{tools}/convert_hf_to_nvfp4.py", "--model-dir", bf16_dir, "--save-dir", out, *_nvfp4_carveouts],
            check=True,
        ),
    )
    prep_volume.commit()
    print(f"Prepared masters={bf16_dir} served_base={nvfp4_dir}")


def _strip_stale_quant_config(config_path: str) -> None:
    """Remove any `quantization_config` (top-level AND text_config-nested) from an
    HF config.json. Used to clean dequantized bf16 masters whose config still
    claims the source quant scheme."""
    import json

    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        cfg = json.load(f)
    removed = bool(cfg.pop("quantization_config", None))
    if isinstance(cfg.get("text_config"), dict):
        removed = bool(cfg["text_config"].pop("quantization_config", None)) or removed
    if removed:
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"stripped stale quantization_config from {config_path}")


def _is_int4(model_dir: str) -> bool:
    import json

    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        return False
    with open(cfg_path) as f:
        cfg = json.load(f) or {}
    # VLMs (Kimi K2.x are KimiK25ForConditionalGeneration) nest the quant config
    # under text_config; convert_kimi_int4_to_bf16 reads it the same way.
    qc = (cfg.get("text_config") or {}).get("quantization_config") or cfg.get("quantization_config") or {}
    return qc.get("quant_method") == "compressed-tensors"


# Raw-mode training loads a Megatron torch_dist checkpoint, not HF (miles' HF load
# is bridge-only). Convert the bf16 masters -> torch_dist with convert_hf_to_torch_dist
# (raw build from MODEL_ARGS + the KimiK25 mbridge weight loader). The full 1T K2.6
# won't fit an 8-way split (~250 GB/GPU), so this runs clustered across
# TORCH_DIST_PREP_NODES x 8 B200 via torchrun (convert auto-derives pp from world size).
TORCH_DIST_PREP_NODES = modal_cfg.torch_dist_prep_nodes
_TORCH_DIST_GPUS_PER_NODE = modal_cfg.torch_dist_prep_gpus_per_node


def prepare_torch_dist() -> None:
    """Build {tag}/torch_dist (the raw-mode ref_load) from the {tag}/bf16 masters."""
    rank, master_addr, _my_ip = helpers.get_modal_cluster_context(TORCH_DIST_PREP_NODES)
    prep_volume.reload()
    tag = exp.MODEL_TAG
    bf16_dir = f"{PREP_PATH}/{tag}/bf16"
    torch_dist_dir = f"{PREP_PATH}/{tag}/torch_dist"
    if os.path.exists(os.path.join(torch_dist_dir, "latest_checkpointed_iteration.txt")):
        print(f"reusing existing torch_dist {torch_dist_dir}")
        return
    if not miles_cfg.miles_model_script:
        raise SystemExit("prepare_torch_dist requires miles_model_script (MODEL_ARGS)")
    use_modal_wrapper = TORCH_DIST_PREP_NODES > 1 and getattr(exp, "USE_MODAL_TORCH_DIST_WRAPPER", False)
    convert_script = TORCH_DIST_CONVERT_WRAPPER if use_modal_wrapper else f"{MILES_ROOT}/tools/convert_hf_to_torch_dist.py"
    # source the model script for ${MODEL_ARGS[@]}, then torchrun the conversion.
    inner = (
        f"source {MILES_ROOT}/{miles_cfg.miles_model_script} && "
        f"PYTHONPATH={MEGATRON_PATH} torchrun"
        f" --nnodes {TORCH_DIST_PREP_NODES} --node-rank {rank}"
        f" --master-addr {master_addr} --master-port 29500"
        f" --nproc-per-node {_TORCH_DIST_GPUS_PER_NODE}"
        f" {convert_script} ${{MODEL_ARGS[@]}}"
        f" --hf-checkpoint {bf16_dir} --save {torch_dist_dir} --megatron-to-hf-mode raw"
        f" {modal_cfg.torch_dist_convert_extra_args}"
    )
    env = os.environ.copy()
    if use_modal_wrapper:
        env["SKIP_RELEASE_RENAME"] = "1"
    subprocess.run(["bash", "-c", inner], check=True, env=env)
    # Each node wrote its own distcp shards to its own Volume mount, so EVERY node must
    # commit — a rank-0-only commit drops nodes 1..N-1's shards and the checkpoint loads
    # short a shard (FileNotFoundError on __<rank>_<n>.distcp). Rank 0 additionally
    # committed .metadata/common.pt and the iteration tracker. Disjoint files across nodes
    # merge cleanly on the Volume.
    prep_volume.commit()
    if rank == 0:
        print(f"Prepared torch_dist={torch_dist_dir}")


_torch_dist_fn_kwargs = dict(
    image=image,
    gpu=f"{modal_cfg.gpu}:{_TORCH_DIST_GPUS_PER_NODE}",
    volumes={str(HF_CACHE_PATH): hf_cache_volume, str(PREP_PATH): prep_volume},
    timeout=6 * 60 * MINUTES,
    # Headroom for the Modal Volume's local write cache while each node buffers its full
    # distcp shard set (~700 GB for the 1T save) until commit. The served-pool default
    # leaves no room and rank 0's commit hits ENOSPC, so allow a dedicated larger value.
    ephemeral_disk=(modal_cfg.torch_dist_prep_ephemeral_disk_mib or modal_cfg.rollout_ephemeral_disk_mib),
    secrets=[modal.Secret.from_name("huggingface-secret")],
    include_source=False,
)
if TORCH_DIST_PREP_NODES > 1:
    prepare_torch_dist = modal.experimental.clustered(TORCH_DIST_PREP_NODES, rdma=True)(prepare_torch_dist)
    _torch_dist_fn_kwargs["experimental_options"] = {"efa_enabled": True}
prepare_torch_dist = app.function(**_torch_dist_fn_kwargs)(prepare_torch_dist)


@app.function(
    image=image,
    volumes={str(DATA_PATH): data_volume},
    timeout=2 * 60 * MINUTES,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    include_source=False,
)
def prepare_dataset() -> None:
    data_volume.reload()
    miles_cfg.prepare_data()
    data_volume.commit()


@app.local_entrypoint()
def launch_train(experiment: str = EXPERIMENT) -> None:
    """Resolve an experiment from the local working tree and spawn it on the
    deployed app. Training args ship as data, so new or edited configs run
    without a redeploy; infrastructure changes still require one."""
    from modal.exception import NotFoundError

    run = importlib.import_module(f"cookbook.miles_disagg.configs.{experiment}")
    if run.DELTA_VOLUME_NAME != exp.DELTA_VOLUME_NAME:
        raise SystemExit(
            f"Experiment {experiment!r} owns Volume {run.DELTA_VOLUME_NAME!r}, but app {APP_NAME!r} "
            f"mounts {exp.DELTA_VOLUME_NAME!r}. Deploy it as its own app with EXPERIMENT_CONFIG={experiment}."
        )

    try:
        trainer = modal.Cls.from_name(APP_NAME, Trainer.__name__)()
        call = trainer.train.spawn(experiment, run.miles.to_payload())
    except NotFoundError:
        raise SystemExit(
            f"App {APP_NAME!r} is not deployed. Run:\n"
            f"  uv run --extra modal modal deploy -m cookbook.miles_disagg.modal_train"
        )
    print(f"Spawned train({experiment!r}) on {APP_NAME}: {call.object_id}")


@app.local_entrypoint()
def smoke_flash_pool(weight_version: int = 0, timeout_seconds: int = 30 * MINUTES) -> None:
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
