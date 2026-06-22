"""Base configuration classes and volume mount paths.

Two separate concerns:

  ModalConfig  — Modal infrastructure (GPU model, regions, rollout pool size)
  SlimeConfig  — SLIME training arguments

Each experiment module defines one instance of each, plus a handful of
module-level constants (APP_NAME, DELTA_VOLUME_NAME, ...) that name the
Modal resources the experiment owns. All non-private, non-callable
attributes on a SlimeConfig subclass become SLIME CLI args automatically
via cli_args(). The 'environment' field is the only exception — it is
exported into the Ray runtime environment, not passed to SLIME directly.
"""

from pathlib import Path
from typing import Any, Literal

# ── Volume mount paths ────────────────────────────────────────────────────────

HF_CACHE_PATH = Path("/root/.cache/huggingface")
DATA_PATH = Path("/data")
CHECKPOINTS_PATH = Path("/checkpoints")

# ── Types ─────────────────────────────────────────────────────────────────────

GPUType = Literal["H100", "H200", "B200", "B300", "A100"]

# Fields on SlimeConfig that are NOT SLIME CLI args.
_SLIME_SKIP = {"environment", "async_mode", "slime_model_script"}

# SlimeConfig fields that SLIME reads as YAML files at runtime.
# Experiments may set these as inline dicts; the launcher materializes
# them to temp YAML files before building the CLI command.
YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config")


class ModalConfig:
    """Modal infrastructure configuration."""

    gpu: GPUType = "H200"
    memory: tuple[int, int] | None = (
        None  # per-container memory in MiB (request, limit)
    )
    cloud: str | None = None  # e.g. "aws", "gcp"
    region: str | None = None  # e.g. "us-east-2"
    rollout_min_containers: int = 4  # warm Flash rollout containers
    proxy_regions: list[str] = ["us-east"]  # Flash gateway proxy regions

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


class SlimeConfig:
    """Base SLIME training configuration.

    Subclass and set class attributes to configure an experiment.
    All attributes (except those in _SLIME_SKIP) are forwarded to SLIME as
    CLI args. Each experiment must be fully self-contained — no inherited
    defaults beyond this base class.

    Fields in _SLIME_SKIP are launcher instructions, not SLIME CLI args:
      environment        — exported into the Ray runtime environment
      async_mode         — selects train_async.py vs train.py
      slime_model_script — path relative to /root/slime to a shell script that
                           defines MODEL_ARGS for model architecture; sourced
                           before running the train command
    """

    # Launcher instructions — not passed to SLIME CLI (see _SLIME_SKIP).
    environment: dict = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_NVLS_ENABLE": "1",
    }
    async_mode: bool = False  # True → use train_async.py
    slime_model_script: str = ""  # shell script path relative to /root/slime

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
        return {k: v for k, v in fields.items() if k not in _SLIME_SKIP}

    def cli_args(self) -> list[str]:
        """SLIME CLI arguments derived from this config.

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
            "slime_model_script": self.slime_model_script,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SlimeConfig":
        cfg = cls(**payload["fields"])
        cfg.environment = dict(payload["environment"])
        cfg.async_mode = payload["async_mode"]
        cfg.slime_model_script = payload["slime_model_script"]
        return cfg
