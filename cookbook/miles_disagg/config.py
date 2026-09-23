"""``MilesConfig`` — miles training arguments as a reflected config (self-contained).

Every public, non-callable attribute becomes a miles CLI arg via ``cli_args`` (miles
wraps Megatron's parser, so Megatron args pass straight through); ``environment`` /
``async_mode`` / ``megatron_model_type`` are launcher instructions, not CLI args. The
Modal-infra half of an experiment is ``common.config.ModalConfig``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cookbook.common.config import validate_serving_config
from cookbook.miles_disagg import nvfp4

_MILES_SKIP = {"environment", "async_mode", "megatron_model_type"}
# Fields miles reads as YAML files; inline dicts are materialized before launch.
# (te_precision_config_file is handled separately in app.py — it needs an identical
# node-local path on every Ray actor, not a per-launch tmpdir.)
YAML_CONFIG_FIELDS = ("custom_config_path",)


class MilesConfig:
    """Subclass and set class attributes; all public, non-callable, non-skip attributes
    become miles CLI args via ``cli_args``."""

    environment: dict = {}
    async_mode: bool = False  # True -> train_async.py
    megatron_model_type: str = ""

    def __init__(self, **kwargs: Any) -> None:
        self.environment = dict(
            type(self).environment
        )  # fresh per instance; never mutate the class default
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def n_train_nodes(self) -> int:
        """Trainer node count: actor nodes, plus critic nodes for PPO/critic setups."""
        nodes = int(getattr(self, "actor_num_nodes", 1))
        if (
            getattr(self, "use_critic", False)
            or getattr(self, "advantage_estimator", None) == "ppo"
        ):
            nodes += int(getattr(self, "critic_num_nodes", nodes))
        return nodes

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
            elif isinstance(val, dict):
                out += [flag, json.dumps(val, separators=(",", ":"))]
            else:
                out += [flag, str(val)]
        return out

    def prepare_data(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} has no prepare_data()")

    def to_payload(self) -> dict[str, Any]:
        """Flatten to plain data so the launcher can ship a config to the deployed Trainer
        — new or edited experiments run without a redeploy."""
        return {
            "fields": self._fields(),
            "environment": dict(self.environment),
            "async_mode": self.async_mode,
            "megatron_model_type": self.megatron_model_type,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> MilesConfig:
        cfg = cls(**payload["fields"])
        cfg.environment = dict(payload["environment"])
        cfg.async_mode = payload["async_mode"]
        cfg.megatron_model_type = payload["megatron_model_type"]
        return cfg


def validate_recipe(recipe: Any) -> None:
    """Check checkpoint and deployment agreements before constructing Modal images."""
    cfg = recipe.miles
    validate_serving_config(recipe, gpus_per_engine=cfg.rollout_num_gpus_per_engine)
    served = Path(recipe.ROLLOUT_CHECKPOINT_PATH)
    masters = Path(recipe.BF16_CHECKPOINT_PATH)
    if Path(cfg.hf_checkpoint) != served:
        raise ValueError("miles.hf_checkpoint must match ROLLOUT_CHECKPOINT_PATH")
    served_format = getattr(recipe, "SERVED_CHECKPOINT_FORMAT", "nvfp4")
    if served_format == "bf16":
        if served != masters:
            raise ValueError(
                "BF16 serving requires ROLLOUT_CHECKPOINT_PATH == BF16_CHECKPOINT_PATH"
            )
    elif served_format == "nvfp4":
        if served == masters:
            raise ValueError(
                "Quantized ROLLOUT_CHECKPOINT_PATH must differ from BF16_CHECKPOINT_PATH"
            )
    else:
        raise ValueError(f"Unsupported SERVED_CHECKPOINT_FORMAT: {served_format!r}")
    if getattr(cfg, "megatron_to_hf_mode", None) == "raw":
        reference = getattr(recipe, "TORCH_DIST_CHECKPOINT_PATH", None)
        ref_load = getattr(cfg, "ref_load", None)
        if not reference or not ref_load or Path(ref_load) != Path(reference):
            raise ValueError(
                "Raw export requires miles.ref_load == TORCH_DIST_CHECKPOINT_PATH"
            )
    if served_format == "nvfp4":
        nvfp4.validate_environments(recipe)
