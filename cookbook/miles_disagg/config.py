"""``MilesConfig`` — miles training arguments as a reflected config (self-contained).

Every public, non-callable attribute becomes a miles CLI arg via ``cli_args`` (miles
wraps Megatron's parser, so Megatron args pass straight through); ``environment`` /
``async_mode`` / ``miles_model_script`` are launcher instructions, not CLI args. The
Modal-infra half of an experiment is ``common.config.ModalConfig``.
"""

from __future__ import annotations

from typing import Any

# Fields that are launcher instructions, not miles CLI args.
_MILES_SKIP = {"environment", "async_mode", "miles_model_script"}
# Fields miles reads as YAML files; inline dicts are materialized before launch.
# (te_precision_config_file is handled separately in app.py — it needs an identical
# node-local path on every Ray actor, not a per-launch tmpdir.)
YAML_CONFIG_FIELDS = ("eval_config", "custom_config_path", "sglang_config")


class MilesConfig:
    """Subclass and set class attributes; all public, non-callable, non-skip attributes
    become miles CLI args via ``cli_args``."""

    environment: dict = {}
    async_mode: bool = False       # True -> train_async.py
    miles_model_script: str = ""   # shell script (relative to the miles root) defining MODEL_ARGS

    def __init__(self, **kwargs: Any) -> None:
        self.environment = dict(type(self).environment)  # fresh per instance; never mutate the class default
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def n_train_nodes(self) -> int:
        """Trainer node count: actor nodes, plus critic nodes for PPO/critic setups."""
        nodes = int(getattr(self, "actor_num_nodes", 1))
        if getattr(self, "use_critic", False) or getattr(self, "advantage_estimator", None) == "ppo":
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
            "miles_model_script": self.miles_model_script,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MilesConfig":
        cfg = cls(**payload["fields"])
        cfg.environment = dict(payload["environment"])
        cfg.async_mode = payload["async_mode"]
        cfg.miles_model_script = payload["miles_model_script"]
        return cfg
