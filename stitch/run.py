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
    """The engines' Modal Server: hardware, and Modal's autoscaling, or with `autoscale` the control service's."""

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
    exit_grace_period: int = 3600
    experimental_options: dict[str, str | bool | int] = field(default_factory=dict)
    autoscale: bool = False

    def __post_init__(self):
        if self.autoscale and self.target_concurrency is not None:
            raise ValueError(
                "autoscale sets the engine count; Modal's target_concurrency would set it too"
            )
        if self.autoscale and self.min_containers < 1:
            raise ValueError(
                "autoscale measures what the engines produce, so it keeps at least one"
            )


@dataclass(frozen=True)
class RunConfig:
    """`image`: the engines' Docker image, with an SGLang that serves weight deltas, built on with `image_commands`.
    `checkpoint` (where every publication chain starts) and `versions` are paths under `volumes`
    (mount path -> Volume). `compile_cache`: the Volume where engines keep the kernels they compile and the
    tactics they tune, one archive per image, GPU and model, shared by every run that names it; None compiles
    in every engine. `adapters`: where a trainer publishes adapters that requests name and engines load on
    demand, on a mount of its own: engines see each one as a local copy at the same path. Each adapter is a directory
    a trainer's link points to; with `adapters_in_place` each link is one adapter, named by its path under `adapters`,
    whose new directories replace it in place. Adapters rotate through SGLang's LoRA slots, so not with attention DP."""

    run_id: str
    image: str
    volumes: dict[str, str]
    checkpoint: str
    versions: str
    engine: SGLangConfig
    pool: PoolConfig
    image_commands: list[str] = field(default_factory=list)
    compile_cache: str | None = "stitch-compile-cache"
    adapters: str | None = None
    adapters_in_place: bool = False
    clocks: Clocks = Clocks()

    def __post_init__(self):
        if self.adapters_in_place and not self.adapters:
            raise ValueError(
                "adapters_in_place updates the adapters under `adapters`, which is unset"
            )
        if self.adapters and self.engine.attention_dp:
            raise ValueError(
                "adapters rotate through SGLang's LoRA slots, but attention DP serves only "
                "pinned adapters: serve `adapters` without attention DP"
            )
        if self.adapters and self.mount_of(self.adapters) in (
            self.mount_of(self.versions),
            self.mount_of(self.checkpoint),
        ):
            raise ValueError(
                f"adapters {self.adapters} need a mount of their own: engines read them only as local copies"
            )

    @property
    def app_name(self) -> str:
        return f"stitch-{self.run_id}"

    @property
    def versions_volume(self) -> str:
        """The Volume the trainer commits each version to and the engines reload: STITCH_VERSIONS_VOLUME."""
        return self.volumes[self.mount_of(self.versions)]

    @property
    def adapters_volume(self) -> str | None:
        """The Volume the trainer commits each adapter to: STITCH_ADAPTERS_VOLUME."""
        return self.volumes[self.mount_of(self.adapters)] if self.adapters else None

    @property
    def server_options(self) -> dict[str, str | bool | int]:
        """The engines' Server options: Modal's KV-aware router, pinning each request to one of the engine's DP ranks."""
        options = {"kv_aware_routing": True, **self.pool.experimental_options}
        ranks = self.engine.dp_ranks
        if ranks > 1:
            given = options.setdefault("kv_aware_routing_dp_size", ranks)
            if given != ranks:
                raise ValueError(
                    f"kv_aware_routing_dp_size {given} is not the engine's {ranks} DP ranks"
                )
        return options

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
