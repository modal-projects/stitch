"""The run app test: a run's app on Modal, created as a trainer creates it, with engines that start
before the run does and catch up to its first version before Modal routes them traffic. The test publishes
each later version as a trainer does (written to the run's folder, then /versions) while traffic
flows through the engines' URL; on the way Modal's autoscaler removes one engine, and later the
control service is stopped. Then the trainer goes quiet past the run's idle window, which lowers the
engines' minimum to 0 until its next call, and last it publishes a version no engine can apply.
Per version: the barrier the trainer waits on and each engine's sync stats. SGLang checks each
version's checksums as it applies it, so an engine that reports a version holds its bytes. Neither
URL answers without the proxy token, and with it the engines' URL reaches no endpoint that drives
the engine and shows no admin key. With --autoscale, Modal's own load-based scaling instead (_autoscale).

    modal run tests/engine/run_app.py \
        --engine qwen3.6-35b-a3b-fp8 --run 1101-120000
"""

import contextlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import modal

# The repo root here; "/" in the container, where this file is /root/run_app.py.
sys.path.insert(0, str(Path(__file__).parents[1].parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "e2e"))
import harness  # noqa: E402
from engine_configs import DELTA_SETS, ENGINES  # noqa: E402
from engine_test import CONCURRENCY, Traffic, post  # noqa: E402

NUM_ENGINES = 2
IDLE_S = 60
# Requests in flight on a killed engine fail; from this long after the kill on, none may.
KILL_WINDOW_S = 60
# Before any engine serves, the test checks the engines' URL turns requests away this often.
PROBE_EVERY_S = 60

app = modal.App(
    "stitch-run-app-test",
    image=modal.Image.debian_slim(python_version="3.12")
    .env({"E2E_REVISIONS": json.dumps(harness.REVISIONS)})
    .add_local_python_source("harness", "configs", "engine_configs", "engine_test"),
)


@app.cls(
    cpu=8,
    memory=16384,
    timeout=6 * 3600,
    max_containers=1,
    scaledown_window=1200,
    volumes={harness.DATA: harness.data},
    secrets=[modal.Secret.from_name("stitch-proxy-auth")],
)
@modal.concurrent(max_inputs=4)
class Driver:
    """The trainer's side, inside Modal: writes and publishes versions, and sends traffic."""

    url: str = modal.parameter()
    generate_url: str = modal.parameter()
    deltas: str = modal.parameter()
    versions: str = modal.parameter()

    @modal.enter()
    def enter(self):
        self.headers = {
            "Modal-Key": os.environ["MODAL_PROXY_TOKEN_ID"],
            "Modal-Secret": os.environ["MODAL_PROXY_TOKEN_SECRET"],
        }
        self.traffic = Traffic(CONCURRENCY, self.generate_url, self.headers)
        self.running = contextlib.ExitStack()
        self.kills: list[float] = []

    @modal.method()
    def write(self, version: int) -> None:
        self._write(version)

    @modal.method()
    def start(self, version: int) -> dict:
        """/start as a trainer sends it; each later version carries the start it names."""
        reply = post("/start", {"version": version}, 900, self.url, self.headers)
        self.run_start = {"start": version, "start_id": reply["reply"]["start_id"]}
        return reply

    @modal.method()
    def send_versions(self, version: int) -> dict:
        return self._versions(version)

    @modal.method()
    def status(self) -> dict:
        return _get(f"{self.url}/status", self.headers)

    @modal.method()
    def publish(self, version: int) -> dict:
        self._write(version)
        return self._versions(version)

    @modal.method()
    def publish_unappliable(self, version: int) -> dict:
        """The previous version's files relabelled as `version`: complete and next in line, but
        applying them undoes the previous version, so every engine's checksum check fails."""
        previous = Path(self.deltas, f"weight_v{version - 1:06d}")
        manifest = json.loads(previous.joinpath("manifest.json").read_text())
        manifest.update(base_version=version - 1, target_version=version)
        self._write(version, previous, json.dumps(manifest))
        return self._versions(version)

    def _versions(self, version: int) -> dict:
        """/versions as a trainer sends it: again, as Miles' http_utils sends a request again, on
        no answer, 408, 429 or 5xx."""
        statuses = []
        while True:
            reply = post(
                "/versions",
                {"version": version, "base_version": version - 1, **self.run_start},
                900,
                self.url,
                self.headers,
            )
            statuses.append(reply["status"])
            again = reply["status"] in (0, 408, 429) or reply["status"] >= 500
            if not again or len(statuses) > 60:
                return {"version": version, "statuses": statuses, **reply}
            time.sleep(5)

    @modal.method()
    def wait_serving(self, engines: int, timeout: float) -> dict:
        """Waits until `engines` serve the run's newest version. Until one does, the engines' URL
        is probed: an engine not yet open must not get the request."""
        started, probes, first_s, probed = time.monotonic(), Counter(), None, 0.0
        while True:
            status = _get(f"{self.url}/status", self.headers)
            if status["status"] != 200:
                raise RuntimeError(f"/status: {status}")
            head, serving = status["reply"]["head"], status["reply"]["serving"]
            count = serving.get(str(head), 0)
            if count and first_s is None:
                first_s = time.monotonic() - started
            if count >= engines:
                return {
                    "probes": dict(probes),
                    "first_s": first_s,
                    "all_s": time.monotonic() - started,
                }
            if time.monotonic() - started > timeout:
                raise TimeoutError(f"after {timeout} s: {status['reply']}")
            if not count and time.monotonic() - probed >= PROBE_EVERY_S:
                probed = time.monotonic()
                probe = {"text": "Hi", "sampling_params": {"max_new_tokens": 1}}
                reply = post("/generate", probe, 60, self.generate_url, self.headers)
                probes[reply["status"]] += 1
            time.sleep(10)

    @modal.method()
    def unauthenticated(self) -> dict[str, int]:
        """What a caller without the proxy token gets from each URL."""
        probe = {"text": "Hi", "sampling_params": {"max_new_tokens": 1}}
        return {
            "control": _get(f"{self.url}/status", {})["status"],
            "engines": post("/generate", probe, 60, self.generate_url)["status"],
        }

    @modal.method()
    def security(self) -> dict:
        """What a caller holding the proxy token reaches through the engines' URL."""
        url, headers = self.generate_url, self.headers
        paused = post("/pause_generation", {"mode": "in_place"}, 60, url, headers)
        if paused["status"] == 200:
            # It must not have worked; let the engine serve again before failing the test.
            post("/continue_generation", {}, 60, url, headers)
        updated = post("/update_weights_from_delta", {}, 60, url, headers)
        info = _get(f"{url}/server_info", headers)["reply"]
        return {
            "pause_status": paused["status"],
            "update_status": updated["status"],
            "server_info_admin_api_key": info.get("admin_api_key"),
        }

    @modal.method()
    def start_traffic(self) -> None:
        self.running.enter_context(self.traffic)

    @modal.method()
    def note_kill(self) -> None:
        self.kills.append(time.time())

    @modal.method()
    def stop_traffic(self) -> dict:
        """Requests done, and the statuses of those that failed: within the window after an engine
        was killed, and otherwise."""
        self.running.close()
        after_kill, otherwise = Counter(), Counter()
        for when, status in self.traffic.failed:
            near = any(0 <= when - kill <= KILL_WINDOW_S for kill in self.kills)
            (after_kill if near else otherwise)[status] += 1
        return {
            "done": len(self.traffic.done),
            "failed_after_kill": dict(after_kill),
            "failed": dict(otherwise),
        }

    @modal.method()
    def save(self, result: dict, name: str) -> None:
        Path(self.versions).parent.joinpath(name).write_text(
            json.dumps(result, indent=1, default=str)
        )
        harness.data.commit()

    def _write(
        self, version: int, source: Path | None = None, manifest: str | None = None
    ):
        """As a trainer writes a version: its owner files, then the manifest, which marks it
        complete; the Volume also commits in the background, so the order matters."""
        source = source or Path(self.deltas, f"weight_v{version:06d}")
        target = Path(self.versions, f"weight_v{version:06d}")
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("manifest.json"))
        harness.data.commit()
        target.joinpath("manifest.json").write_text(
            manifest or source.joinpath("manifest.json").read_text()
        )
        harness.data.commit()


def _get(url: str, headers: dict[str, str]) -> dict:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as reply:
            status, text = reply.status, reply.read().decode()
    except urllib.error.HTTPError as error:
        status, text = error.code, error.read().decode()
    return {
        "status": status,
        "reply": json.loads(text) if text.startswith(("{", "[")) else text,
    }


def _wait_registered(pool: modal.Server, count: int, driver) -> float:
    """Seconds until Modal routes to `count` engine containers."""
    started = time.monotonic()
    while len(pool._experimental_list_containers()) != count:
        if time.monotonic() - started > 3600:
            raise TimeoutError(f"the pool did not reach {count} containers in 3600 s")
        # A trainer call each round, so the run's idle window does not open while the test waits.
        driver.status.remote()
        time.sleep(2)
    return time.monotonic() - started


def _idle_window(pool: modal.Server, driver, idle: float) -> dict:
    """Seconds until the engines' minimum drops to 0 with the trainer quiet, then until one trainer
    call restores it; the control service checks once a minute."""
    started = time.monotonic()
    _wait_min_containers(pool, 0, idle + 180)
    lowered = time.monotonic() - started
    driver.status.remote()
    started = time.monotonic()
    _wait_min_containers(pool, NUM_ENGINES, 180)
    return {"lowered_s": lowered, "restored_s": time.monotonic() - started}


def _wait_min_containers(pool: modal.Server, count: int, timeout: float) -> None:
    started = time.monotonic()
    while pool.autoscaler.settings().min_containers != count:
        if time.monotonic() - started > timeout:
            raise TimeoutError(
                f"the engines' minimum did not reach {count} in {timeout} s"
            )
        time.sleep(5)


def _control_container(app_name: str, engines: list[str]) -> str:
    """The run app's one container that is not one of its engines."""
    app_id = modal.App.lookup(app_name).app_id
    listed = _modal_cli("container", "list", "--app-id", app_id, "--json")
    containers = {c["container_id"] for c in json.loads(listed)}
    others = containers - set(engines)
    if len(others) != 1:
        raise RuntimeError(f"not one control container in {containers}")
    return others.pop()


def _modal_cli(*args: str) -> str:
    command = [sys.executable, "-m", "modal", *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


@app.local_entrypoint()
def main(
    run: str,
    engine: str = "qwen3.6-35b-a3b-fp8",
    join_at: int = 3,
    autoscale: bool = False,
):
    if autoscale:
        return _autoscale(run, engine, join_at)
    # Stitch runs here, where the trainer would run it; the driver's container has none of it.
    from stitch.control import Clocks
    from stitch.run import PoolConfig, RunConfig, create_run
    from stitch.sglang import SGLangConfig

    config = ENGINES[engine]
    # Runs launched together may start in the same second; the suffix keeps their apps apart.
    stamp = f"{datetime.now().strftime('%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    run_dir = Path(harness.DATA, "engine", config.delta_set, run)
    report = json.loads(
        b"".join(harness.data.read_file(f"engine/{config.delta_set}/{run}/report.json"))
    )
    if report["mode"] != "correctness":
        raise SystemExit(
            "the last version must fail every engine's checksum check: use a correctness set"
        )
    last_version = report["versions"]
    kill_engine_at, kill_control_at = join_at + 3, join_at + 5
    if last_version <= kill_control_at:
        raise SystemExit(
            f"{last_version} versions; the test needs {kill_control_at + 1}"
        )
    run_config = RunConfig(
        run_id=f"engine-test-{stamp}",
        image=harness.MILES_IMAGE,
        image_commands=harness.ENGINE_COMMANDS,
        volumes={
            harness.CHECKPOINTS: harness.checkpoints.name,
            harness.DATA: harness.data.name,
        },
        checkpoint=f"{harness.CHECKPOINTS}/{DELTA_SETS[config.delta_set].rollout_checkpoint}",
        versions=str(run_dir / f"run-app-{stamp}"),
        engine=SGLangConfig(
            server_args=harness.server_args(config.engine_args), env=config.env
        ),
        pool=PoolConfig(
            gpu=f"{config.gpu}:{config.gpus}",
            cpu=32,
            memory=262144,
            min_containers=NUM_ENGINES,
            max_containers=NUM_ENGINES,
        ),
        clocks=Clocks(idle=IDLE_S),
    )
    result = {"run_id": run_config.run_id, "join_at": join_at, "syncs": []}
    stitch_run = create_run(run_config)
    driver = Driver(
        url=stitch_run.url,
        generate_url=stitch_run.generate_url,
        deltas=str(run_dir / "deltas"),
        versions=run_config.versions,
    )
    pool = modal.Server.from_name(run_config.app_name, "EnginePool")
    try:
        result["unauthenticated"] = driver.unauthenticated.remote()
        for version in range(1, join_at + 1):
            driver.write.remote(version)
        result["start"] = driver.start.remote(join_at)
        result["open"] = driver.wait_serving.remote(NUM_ENGINES, timeout=3600)
        result["security"] = driver.security.remote()
        refused = driver.send_versions.remote(join_at + 2)
        result["refused_status"] = refused["status"]

        driver.start_traffic.remote()
        for version in range(join_at + 1, last_version + 1):
            # An engine's id is its container's.
            serving = (
                sorted(result["syncs"][-1]["reply"]["engines"])
                if result["syncs"]
                else []
            )
            if version == kill_engine_at:
                driver.note_kill.remote()
                pool.autoscaler.update(min_containers=1, max_containers=1)
                result["scaled_down_s"] = _wait_registered(pool, 1, driver)
            if version == kill_engine_at + 1:
                pool.autoscaler.update(
                    min_containers=NUM_ENGINES, max_containers=NUM_ENGINES
                )
                # A departing engine still calls the control service while it shuts down, but
                # Modal routes to a container only once it has caught up and opened.
                result["replaced_s"] = _wait_registered(pool, NUM_ENGINES, driver)
                result["replaced"] = driver.wait_serving.remote(
                    NUM_ENGINES, timeout=3600
                )
            if version == kill_control_at:
                killed = _control_container(run_config.app_name, serving)
                _modal_cli("container", "stop", "--yes", killed)
                result["killed_control"] = killed
            result["syncs"].append(driver.publish.remote(version))
            if result["syncs"][-1]["status"] != 200:
                raise RuntimeError(f"/versions {version}: {result['syncs'][-1]}")
        result["idle"] = _idle_window(pool, driver, run_config.clocks.idle)
        result["traffic"] = driver.stop_traffic.remote()
        unappliable = driver.publish_unappliable.remote(last_version + 1)
        result["unappliable"] = {
            k: unappliable[k] for k in ("statuses", "status", "reply")
        }
    except Exception as error:
        result["error"] = repr(error)
    finally:
        stitch_run.stop()
    driver.save.remote(result, f"run-app-{stamp}.json")

    print(json.dumps(result, indent=1, default=str)[:8000])
    # A gone engine is dropped within a lease, and a restarted control service waits one lease.
    failures = _failures(result, kill_engine_at, run_config.clocks.lease + 15)
    if failures:
        raise SystemExit(f"run app test FAILED: {failures}")
    print("run app test passed")


def _failures(result: dict, kill_engine_at: int, longest_barrier_s: float) -> list[str]:
    if result.get("error"):
        return [result["error"]]
    failures = []
    if set(result["unauthenticated"].values()) != {401}:
        failures.append(f"without the proxy token: {result['unauthenticated']}")
    if set(result["open"]["probes"]) - {503}:
        failures.append(f"probes before an engine opened: {result['open']['probes']}")
    security = result["security"]
    if (security["pause_status"], security["update_status"]) != (401, 401):
        failures.append(f"engine control through the URL: {security}")
    if security["server_info_admin_api_key"] != "<redacted>":
        failures.append("the admin key shows in /server_info")
    if result["refused_status"] != 409:
        failures.append(f"a version skipping one: HTTP {result['refused_status']}")
    for sync in result["syncs"]:
        engines = sync["reply"].get("engines", {}) if sync["status"] == 200 else {}
        expected = (
            range(1, NUM_ENGINES + 1)
            if sync["version"] == kill_engine_at
            else [NUM_ENGINES]
        )
        at_version = all(e["version"] == sync["version"] for e in engines.values())
        if len(engines) not in expected or not at_version:
            failures.append(f"version {sync['version']}: {sync}")
        elif sync["reply"]["seconds"] > longest_barrier_s:
            failures.append(
                f"version {sync['version']}: barrier {sync['reply']['seconds']:.1f} s"
            )
    if result["unappliable"]["status"] != 422:
        failures.append(f"a version no engine can apply: {result['unappliable']}")
    if not result["traffic"]["done"] or result["traffic"]["failed"]:
        failures.append(f"traffic: {result['traffic']}")
    return failures


@app.function(
    cpu=8,
    memory=16384,
    timeout=3600,
    volumes={harness.CHECKPOINTS: harness.checkpoints, harness.DATA: harness.data},
)
def copy_checkpoint(source: str, target: str) -> None:
    """A checkpoint copied once beside the runs' versions, so one Volume holds both."""
    if Path(target, ".complete").exists():
        return
    shutil.copytree(source, target, dirs_exist_ok=True)
    Path(target, ".complete").touch()
    harness.data.commit()


def _autoscale(run: str, engine: str, join_at: int) -> None:
    """Modal's own scaling on load, on the README's one-Volume layout (checkpoint and versions on one
    Volume): traffic grows the pool from one engine to two, the joiner catches up and takes every
    version published under load, and once traffic stops the pool shrinks back to one, which takes
    the last versions alone. No request may fail on the way.

        modal run tests/engine/run_app.py --autoscale \
            --engine qwen3.6-35b-a3b-nvfp4 --run 1111-004043
    """
    from stitch.run import PoolConfig, RunConfig, create_run
    from stitch.sglang import SGLangConfig

    config = ENGINES[engine]
    stamp = f"{datetime.now().strftime('%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    one_volume = Path(harness.DATA, "one-volume")
    checkpoint = DELTA_SETS[config.delta_set].rollout_checkpoint
    copy_checkpoint.remote(
        f"{harness.CHECKPOINTS}/{checkpoint}",
        str(one_volume / "checkpoints" / checkpoint),
    )
    report = json.loads(
        b"".join(harness.data.read_file(f"engine/{config.delta_set}/{run}/report.json"))
    )
    last_version = report["versions"]
    run_config = RunConfig(
        run_id=f"autoscale-{stamp}",
        image=harness.MILES_IMAGE,
        image_commands=harness.ENGINE_COMMANDS,
        volumes={harness.DATA: harness.data.name},
        checkpoint=str(one_volume / "checkpoints" / checkpoint),
        versions=str(one_volume / "runs" / f"autoscale-{stamp}"),
        engine=SGLangConfig(
            server_args=harness.server_args(config.engine_args), env=config.env
        ),
        pool=PoolConfig(
            gpu=f"{config.gpu}:{config.gpus}",
            cpu=32,
            memory=262144,
            min_containers=1,
            max_containers=NUM_ENGINES,
            target_concurrency=CONCURRENCY // NUM_ENGINES,
            scaledown_window=60,
        ),
    )
    result = {"run_id": run_config.run_id, "syncs": []}
    stitch_run = create_run(run_config)
    driver = Driver(
        url=stitch_run.url,
        generate_url=stitch_run.generate_url,
        deltas=str(Path(harness.DATA, "engine", config.delta_set, run, "deltas")),
        versions=run_config.versions,
    )
    pool = modal.Server.from_name(run_config.app_name, "EnginePool")
    under_load = range(join_at + 1, last_version - 1)
    try:
        for version in range(1, join_at + 1):
            driver.write.remote(version)
        driver.start.remote(join_at)
        result["open"] = driver.wait_serving.remote(1, timeout=3600)
        driver.start_traffic.remote()
        result["grew_s"] = _wait_registered(pool, NUM_ENGINES, driver)
        result["joined"] = driver.wait_serving.remote(NUM_ENGINES, timeout=3600)
        for version in under_load:
            result["syncs"].append(driver.publish.remote(version))
        result["traffic"] = driver.stop_traffic.remote()
        result["shrank_s"] = _wait_registered(pool, 1, driver)
        for version in range(under_load.stop, last_version + 1):
            result["syncs"].append(driver.publish.remote(version))
    except Exception as error:
        result["error"] = repr(error)
    finally:
        stitch_run.stop()
    driver.save.remote(result, f"autoscale-{stamp}.json")

    print(json.dumps(result, indent=1, default=str)[:8000])
    failures = [result["error"]] if result.get("error") else []
    for sync in result["syncs"]:
        engines = sync["reply"].get("engines", {}) if sync["status"] == 200 else {}
        expected = NUM_ENGINES if sync["version"] in under_load else 1
        if len(engines) != expected or any(
            e["version"] != sync["version"] for e in engines.values()
        ):
            failures.append(f"version {sync['version']}: {sync}")
    if "traffic" in result and (
        not result["traffic"]["done"] or result["traffic"]["failed"]
    ):
        failures.append(f"traffic: {result['traffic']}")
    if failures:
        raise SystemExit(f"autoscale test FAILED: {failures}")
    print("autoscale test passed")
