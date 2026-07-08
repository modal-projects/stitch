"""Base configuration classes and volume mount paths for the miles example.

Two separate concerns:

  ModalConfig  — Modal infrastructure (GPU model, regions, rollout pool size)
  MilesConfig  — miles training arguments

Each experiment module defines one instance of each, plus a handful of
module-level constants (APP_NAME, DELTA_VOLUME_NAME, ...) that name the
Modal resources the experiment owns. All non-private, non-callable
attributes on a MilesConfig subclass become miles CLI args automatically
via cli_args() (miles wraps Megatron's parser, so Megatron args like
``--fp4-format`` pass straight through). The 'environment' field is the only
exception — it is exported into the Ray runtime environment, not passed to
miles directly.

This is the miles twin of cookbook/slime_disagg/configs/base.py. The
reflection mechanism is identical; only the names and the skip set differ
(miles uses ``miles_model_script``).
"""

from pathlib import Path
from typing import Any, Literal

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")
# Derived checkpoints the prepare step builds on a GPU (see modal_train):
#   <PREP>/<name>-bf16   bf16 masters (dequantized from the published INT4)
#   <PREP>/<name>-nvfp4  served NVFP4 base (miles convert_hf_to_nvfp4 of the masters)
PREP_PATH = Path("/prep")

# ── Types ─────────────────────────────────────────────────────────────────────

GPUType = Literal["H100", "H200", "B200", "B300", "A100"]

# Fields on MilesConfig that are NOT miles CLI args.
_MILES_SKIP = {"environment", "async_mode", "miles_model_script"}

# MilesConfig fields that miles reads as YAML files at runtime. Experiments may
# set these as inline dicts; the launcher materializes them to temp YAML files
# before building the CLI command. miles' --custom-config-path setattr's every
# key in the YAML onto the args namespace (arguments.py), which is how the
# bulletin/gating knobs below reach the hooks.
YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config", "te_precision_config_file")


class ModalConfig:
    """Modal infrastructure configuration."""

    gpu: GPUType = "B200"  # NVFP4 QAT + NVFP4 serving are both Blackwell-only
    memory: tuple[int, int] | None = None  # per-container memory in MiB (request, limit)
    cloud: str | None = None  # e.g. "aws", "gcp"
    region: str | None = None  # e.g. "us-east-2"
    rollout_min_containers: int = 2  # warm Flash rollout containers
    rollout_max_containers: int | None = None  # cap Flash rollout containers; None = no explicit cap
    # Flash autoscaler target: concurrent inputs (requests) per container before it
    # scales OUT. None = use sglang_server_concurrency (legacy). Set it well below the
    # SGLang engine concurrency so Flash adds containers instead of packing requests
    # onto a few until their KV cache saturates and requests 502/stall.
    rollout_target_inputs: int | None = None
    proxy_regions: list[str] = ["us-west"]  # Flash gateway proxy regions
    # Ephemeral disk (MiB) for the rollout Server. The sidecar materializes a
    # writable local copy of the served base (~full model size) onto ephemeral
    # disk; large models exceed Modal's default. None = Modal default (fine for
    # small bases like Moonlight).
    rollout_ephemeral_disk_mib: int | None = None
    # Host-RAM request (MiB) for the rollout Server. Reloads read the whole
    # local checkpoint through the page cache; when it doesn't fit, every
    # reload pays capacity misses at disk speed (measured ~120s of iter_wait
    # on K2.6's 595 GB base). Request checkpoint size + engine headroom so the
    # base stays resident. None = Modal default.
    rollout_memory_mib: int | None = None
    # Nodes x GPUs/node for the prepare_torch_dist conversion. The full 1T K2.6
    # needs >=2 nodes (8-way OOMs); a small proxy (e.g. 2-layer) fits 1 GPU (the
    # convert auto-derives pp=world_size and asserts pp <= num_layers).
    torch_dist_prep_nodes: int = 2
    torch_dist_prep_gpus_per_node: int = 8
    # Extra parallelism args for the prepare_torch_dist conversion. Large MoE needs
    # explicit EP sharding so experts fit (else convert's auto-pp=world_size/EP1 OOMs).
    torch_dist_convert_extra_args: str = ""
    # Ephemeral disk (MiB) for the prepare_torch_dist conversion containers. Each node
    # buffers its whole distcp shard set in the Volume's local write cache until commit;
    # the 1T K2.6 save is ~700 GB/node, so the served-pool default (~800 GB) leaves no
    # headroom and rank 0's commit (shards + .metadata + common.pt) hits ENOSPC. None =
    # fall back to rollout_ephemeral_disk_mib.
    torch_dist_prep_ephemeral_disk_mib: int | None = None
    # Ephemeral disk (MiB) for the Trainer nodes. miles' ray.init(address="auto") uses
    # default spill+log dirs under /tmp/ray, so over a multi-hour run Ray logs + object
    # spill (rollout batches + per-publish full-model gathers) accumulate on the node's
    # disk; Modal's default is far too small and progressively ENOSPC'd (`No space left
    # on device` writing /tmp/ray/.../logs). None = Modal default.
    trainer_ephemeral_disk_mib: int | None = None

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class MilesConfig:
    """Base miles training configuration.

    Subclass and set class attributes to configure an experiment. All
    attributes (except those in _MILES_SKIP) are forwarded to miles as CLI
    args. Each experiment must be fully self-contained — no inherited defaults
    beyond this base class.

    Fields in _MILES_SKIP are launcher instructions, not miles CLI args:
      environment        — exported into the Ray runtime environment
      async_mode         — selects train_async.py vs train.py
      miles_model_script — path relative to the miles root to a shell script
                           that defines MODEL_ARGS for model architecture;
                           sourced before running the train command
    """

    # Launcher instructions — not passed to miles CLI (see _MILES_SKIP).
    environment: dict = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
    }
    async_mode: bool = False  # True → use train_async.py
    miles_model_script: str = ""  # shell script path relative to the miles root

    def __init__(self, **kwargs: Any) -> None:
        # Fresh environment dict per instance — never mutate the class-level default.
        self.environment = dict(type(self).environment)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _fields(self) -> dict[str, Any]:
        """Merged field dict from the class hierarchy; instance attrs win."""
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
        """miles CLI arguments derived from this config.

        Conversion rules:
          field_name → --field-name  (underscore to hyphen)
          True       → --flag        (no value)
          False/None → omitted
          list       → --flag v1 v2 ...
          other      → --flag value
        """
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
        """Flatten to plain data for sending to a deployed Trainer.

        launch_train resolves config modules locally and ships the result, so
        the deployed app never imports them — new or edited experiments run
        without a redeploy.
        """
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
