"""One run's app on Modal, built from the run config in STITCH_RUN: control service and engines."""

import dataclasses
import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import modal

from stitch.autoscale import Autoscaler
from stitch.control import ControlService, Failed, Refused
from stitch.engine import AdapterStage, ControlClient, EngineContainer
from stitch.run import RunConfig
from stitch.sglang import SGLang

PORT = 8000
logger = logging.getLogger(__name__)
# Stitch's own records from INFO up, such as the autoscaler's decisions; other libraries keep their levels.
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
)
logging.getLogger("stitch").addHandler(_handler)
logging.getLogger("stitch").setLevel(logging.INFO)
logging.getLogger("stitch").propagate = False

config = RunConfig.from_json(os.environ["STITCH_RUN"])
ENV = {"STITCH_RUN": os.environ["STITCH_RUN"]}
# Engines only read the run's Volumes; a read-only mount also reloads without committing first.
volumes = {
    mount: modal.Volume.from_name(name).with_mount_options(read_only=True)
    for mount, name in config.volumes.items()
}
versions_volume = volumes[config.mount_of(config.versions)]
COMPILE_CACHE_MOUNT = "/compile-cache"
compile_cache_volume = (
    modal.Volume.from_name(config.compile_cache, create_if_missing=True, version=2)
    if config.compile_cache
    else None
)
# Engines see published adapters only as local copies at the paths requests name, made from a private mount.
ADAPTER_SOURCE_MOUNT = "/adapter-source"
adapters_mount = config.mount_of(config.adapters) if config.adapters else None
engine_volumes = {
    ADAPTER_SOURCE_MOUNT if mount == adapters_mount else mount: volume
    for mount, volume in volumes.items()
}
if compile_cache_volume:
    engine_volumes[COMPILE_CACHE_MOUNT] = compile_cache_volume
image = (
    modal.Image.from_registry(config.image)
    .run_commands(
        *config.image_commands,
        "python3 -m pip install --no-deps autoinference-utils==0.3.0",
        # autoinference-utils launches the engine with `python`; drop these two lines once it
        # launches with the running interpreter.
        "python -c 'import sglang' || ln -sf \"$(command -v python3)\" /usr/local/bin/python",
        "python -c 'import sglang'",
    )
    .add_local_python_source("stitch")
)
app = modal.App(config.app_name, image=image)


@app.cls(
    cpu=1,
    memory=2048,
    min_containers=0,
    max_containers=1,
    scaledown_window=600,
    timeout=math.ceil(2 * config.clocks.barrier),
    env=ENV,
)
@modal.concurrent(max_inputs=1000)
class Control:
    @modal.enter()
    def enter(self):
        self.service = ControlService(config.clocks)
        self.pool = modal.Server.from_name(config.app_name, "EnginePool")
        self.autoscaler = None
        if config.pool.autoscale:
            # A restarted service continues from the count it last set.
            count = self.pool.autoscaler.settings().max_containers
            self.autoscaler = Autoscaler(
                config.pool.min_containers, config.pool.max_containers, count
            )
        threading.Thread(target=self._size_pool, daemon=True).start()

    @modal.method()
    def open(self, engine_id: str, version: int, start_id: str | None) -> dict:
        return self.service.open(engine_id, version, start_id)

    @modal.method()
    def next(
        self,
        engine_id: str,
        version: int,
        seen: int,
        start_id: str | None,
        service_id: str | None,
        seen_changes: int,
    ) -> dict:
        return self.service.next(
            engine_id, version, seen, start_id, service_id, seen_changes
        )

    @modal.method()
    def served(self, engine_id: str, version: int, stats: dict, start_id: str) -> dict:
        return self.service.served(engine_id, version, stats, start_id)

    @modal.method()
    def synced(
        self,
        engine_id: str,
        version: int,
        changes: int,
        start_id: str | None,
        service_id: str,
    ) -> dict:
        return self.service.synced(engine_id, version, changes, start_id, service_id)

    @modal.method()
    def fail(self, engine_id: str, version: int) -> None:
        self.service.fail(engine_id, version)

    @modal.method()
    def report_load(self, engine_id: str, load: dict) -> None:
        self.service.report_load(engine_id, load)

    @modal.method()
    def leave(self, engine_id: str) -> None:
        self.service.leave(engine_id)

    @modal.method()
    def run_config(self) -> str:
        return os.environ["STITCH_RUN"]

    @modal.method()
    def status(self) -> dict:
        return self.service.status()

    @modal.asgi_app(requires_proxy_auth=True)
    def api(self):
        from fastapi import FastAPI, HTTPException

        web, service = FastAPI(), self.service

        @web.post("/start")
        def start(body: dict) -> dict:
            return service.start(int(body["version"]), body.get("publication"))

        @web.post("/versions")
        def versions(body: dict) -> dict:
            try:
                return service.publish(
                    int(body["version"]),
                    int(body["base_version"]),
                    int(body["start"]),
                    body["start_id"],
                    body.get("publication"),
                )
            except Refused as error:
                raise HTTPException(409, str(error)) from None
            except Failed as error:
                raise HTTPException(422, str(error)) from None

        @web.post("/adapters")
        def adapters(body: dict) -> dict:
            if not config.adapters or not Path(body["path"]).is_relative_to(
                config.adapters
            ):
                raise HTTPException(
                    422,
                    f"{body['path']} is outside the run's adapters {config.adapters}",
                )
            try:
                return service.publish_adapter(
                    body["path"], int(body["start"]), body["start_id"]
                )
            except Refused as error:
                raise HTTPException(409, str(error)) from None

        @web.post("/steps")
        def steps(body: dict) -> dict:
            if self.autoscaler is not None:
                self.autoscaler.observe_step(
                    time.monotonic(),
                    float(body["seconds"]),
                    float(body["waited"]),
                    float(body["held"]),
                    int(body["consumed"]),
                    int(body["produced"]),
                )
            return {}

        @web.get("/status")
        def status(
            start: int | None = None,
            start_id: str | None = None,
            head: int | None = None,
            publication: str | None = None,
        ) -> dict:
            try:
                return service.status(
                    start,
                    start_id,
                    head,
                    None if publication is None else json.loads(publication),
                )
            except Refused as error:
                raise HTTPException(409, str(error)) from None

        return web

    def _size_pool(self) -> None:
        """Keeps the run's engine count, or auto mode's, with a minimum of 0 while the trainer is away."""
        asked: dict[str, int] = {}
        while True:
            away = self.service.idle_seconds() > config.clocks.idle
            if self.autoscaler is None:
                settings = {"min_containers": 0 if away else config.pool.min_containers}
            else:
                self.autoscaler.observe_serving(
                    time.monotonic(), self.service.serving()
                )
                count = self.autoscaler.count
                settings = {
                    "min_containers": 0 if away else count,
                    "max_containers": count,
                }
            changed = {
                key: value for key, value in settings.items() if asked.get(key) != value
            }
            if changed:
                self.pool.autoscaler.update(**changed)
                asked |= changed
            time.sleep(config.clocks.wait)


# Each cache's environment variable and its folder; XDG_CACHE_HOME covers FlashInfer's and CuTe DSL's.
CACHE_FOLDERS = {
    "SGLANG_CACHE_DIR": "sglang",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "inductor",
    "CUDA_CACHE_PATH": "nv",
    "XDG_CACHE_HOME": "xdg",
    "TILELANG_CACHE_DIR": "tilelang",
    "DG_JIT_CACHE_DIR": "deep_gemm",
}
LOCAL_COMPILE_CACHE = Path("/root/compile-cache")


class CompileCache:
    """This image, GPU and model's compiled kernels and tuned tactics, kept as one archive in the compile cache:
    reading thousands of small files through a Volume costs more than compiling them."""

    def __init__(self, volume: modal.Volume):
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()[0]
        parts = (config.image, gpu, Path(config.checkpoint).name)
        name = "__".join(re.sub(r"[^A-Za-z0-9._-]", "_", part) for part in parts)
        self.volume, self.archive = volume, Path(COMPILE_CACHE_MOUNT, f"{name}.tar")
        self.restored = 0

    def restore(self) -> dict[str, str]:
        """Unpacks the archive onto local disk; the env that points every cache there."""
        LOCAL_COMPILE_CACHE.mkdir(parents=True, exist_ok=True)
        if self.archive.is_file():
            try:
                subprocess.run(
                    ["tar", "-xf", str(self.archive), "-C", str(LOCAL_COMPILE_CACHE)],
                    check=True,
                )
            except subprocess.CalledProcessError:
                logger.warning("the compile cache was not restored", exc_info=True)
        self.restored = _file_count(LOCAL_COMPILE_CACHE)
        return {
            variable: str(LOCAL_COMPILE_CACHE / folder)
            for variable, folder in CACHE_FOLDERS.items()
        }

    def save(self) -> None:
        """Packs the caches into the archive when this engine added to them; an engine that cannot still serves."""
        try:
            if _file_count(LOCAL_COMPILE_CACHE) == self.restored:
                return
            partial = self.archive.with_name(
                f"{self.archive.name}.{os.environ['MODAL_TASK_ID']}"
            )
            subprocess.run(
                ["tar", "-cf", str(partial), "-C", str(LOCAL_COMPILE_CACHE), "."],
                check=True,
            )
            partial.rename(self.archive)
            self.volume.commit()
        except Exception:
            logger.warning("the compile cache was not saved", exc_info=True)


def _file_count(folder: Path) -> int:
    return sum(len(files) for _, _, files in os.walk(folder))


def _node(cls):
    """An engine on every GPU of its machine runs as a one-node cluster, so the machine is its alone
    (Modal: up to 4 A10s or 8 of any other type in a container; a cluster takes one GPU type)."""
    gpu_type, _, count = config.pool.gpu.upper().partition(":")
    if "+" in gpu_type or int(count or 1) != (4 if gpu_type == "A10" else 8):
        return cls
    return modal.clustered(size=1)(cls)


@app.server(
    gpu=config.pool.gpu,
    port=PORT,
    startup_timeout=config.pool.startup_timeout,
    min_containers=config.pool.min_containers,
    # Auto mode starts at its minimum, and the control service sets the count from there.
    max_containers=config.pool.min_containers
    if config.pool.autoscale
    else config.pool.max_containers,
    buffer_containers=config.pool.buffer_containers,
    target_concurrency=config.pool.target_concurrency,
    scaledown_window=config.pool.scaledown_window,
    exit_grace_period=config.pool.exit_grace_period,
    cpu=config.pool.cpu,
    memory=config.pool.memory,
    volumes=engine_volumes,
    env=ENV,
    experimental_options=config.server_options,
)
@_node
class EnginePool:
    @modal.enter()
    def enter(self):
        cache = CompileCache(compile_cache_volume) if compile_cache_volume else None
        engine = SGLang(
            config.checkpoint,
            version=-1,
            port=PORT,
            config=dataclasses.replace(
                config.engine,
                env={**(cache.restore() if cache else {}), **config.engine.env},
            ),
            startup_timeout=config.pool.startup_timeout,
            engine_id=os.environ["MODAL_TASK_ID"],
        )
        control = ControlClient(Control(), os.environ["MODAL_TASK_ID"], config.clocks)
        adapter_stage = None
        if adapters_mount:
            adapter_stage = AdapterStage(
                Path(ADAPTER_SOURCE_MOUNT),
                Path(adapters_mount),
                Path(config.adapters),
                volumes[adapters_mount].reload,
            )
        self.container = EngineContainer(
            engine,
            control,
            Path(config.versions),
            versions_volume.reload,
            adapter_stage,
            config.adapters_in_place,
        )
        self.container.enter()
        if cache:
            threading.Thread(target=cache.save, daemon=True).start()

    @modal.exit()
    def exit(self):
        self.container.exit()
