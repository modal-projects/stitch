"""A run's config, and creating or stopping the run's app on Modal."""

import dataclasses
import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import modal

from stitch.control import Clocks
from stitch.sglang import SGLangConfig


@dataclass(frozen=True)
class PoolConfig:
    """The engines' Modal Server: hardware, and Modal's own autoscaling."""

    gpu: str
    cpu: float | None = None
    memory: int | None = None
    min_containers: int = 1
    max_containers: int | None = None
    buffer_containers: int | None = None
    target_concurrency: float | None = None
    scaledown_window: int | None = None
    # An engine's start: loading the checkpoint, then catching up through every version.
    startup_timeout: int = 3600
    exit_grace_period: int = 0
    experimental_options: dict[str, str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """`image`: the engines' Docker image, with an SGLang that serves weight deltas, built on with `image_commands`.
    `checkpoint` (version 0) and `versions` are paths under `volumes` (mount path -> Volume)."""

    run_id: str
    image: str
    volumes: dict[str, str]
    checkpoint: str
    versions: str
    engine: SGLangConfig
    pool: PoolConfig
    image_commands: list[str] = field(default_factory=list)
    clocks: Clocks = Clocks()

    @property
    def app_name(self) -> str:
        return f"stitch-{self.run_id}"

    @property
    def versions_volume(self) -> str:
        """The Volume the trainer commits each version to and the engines reload: STITCH_VERSIONS_VOLUME."""
        return self.volumes[self.mount_of(self.versions)]

    def mount_of(self, path: str) -> str:
        """The mount in `volumes` that `path` is under."""
        mounts = [mount for mount in self.volumes if Path(path).is_relative_to(mount)]
        if not mounts:
            raise ValueError(
                f"{path} is under none of the run's volumes {sorted(self.volumes)}"
            )
        return max(mounts, key=len)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RunConfig":
        fields = json.loads(text)
        return cls(
            **{
                **fields,
                "engine": SGLangConfig(**fields["engine"]),
                "pool": PoolConfig(**fields["pool"]),
                "clocks": Clocks(**fields["clocks"]),
            }
        )


@dataclass(frozen=True)
class Run:
    url: str  # the control service: /start, /versions, /status
    generate_url: str  # the engines' Server
    app_name: str

    def status(self) -> dict:
        """The head, how many engines serve each version, and how many are catching up: before the first /start,
        every engine that has loaded the checkpoint and waits to open."""
        return modal.Cls.from_name(self.app_name, "Control")().status.remote()

    def stop(self) -> None:
        """Stops the run's app; a run already stopped stays stopped."""
        try:
            app = modal.App.lookup(self.app_name)
        except modal.exception.NotFoundError:
            return
        app.stop()


def create_run(config: RunConfig) -> Run:
    """Deploys the run's app, or finds it deployed with this config; another config needs a new id."""
    checkpoint_volume = config.volumes[config.mount_of(config.checkpoint)]
    if config.versions_volume == checkpoint_volume:
        warnings.warn(
            f"versions and the checkpoint share the Volume {checkpoint_volume}: an engine sees a version only after "
            "reloading that Volume, and its first reload drops the cache of the whole checkpoint load (~0.08 s per GB "
            "read), which delays the run's first version. Put versions on another Volume.",
            stacklevel=2,
        )
    control = modal.Cls.from_name(config.app_name, "Control")()
    try:
        deployed = control.run_config.remote()
    except modal.exception.NotFoundError:
        deployed = None
    if deployed is None:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "deploy",
                "-m",
                "stitch.app",
                "--name",
                config.app_name,
            ],
            env={**os.environ, "STITCH_RUN": config.to_json()},
            check=True,
        )
    elif deployed != config.to_json():
        raise ValueError(
            f"run {config.run_id} is deployed with another config; a new config is a new run id"
        )
    return Run(
        url=control.api.get_web_url(),
        generate_url=modal.Server.from_name(config.app_name, "EnginePool").get_url(),
        app_name=config.app_name,
    )
