"""The engine container against the real control service, with a fake engine."""

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import modal
import pytest

import stitch.engine
from stitch.control import Clocks, ControlService, Failed, Refused
from stitch.engine import (
    ADAPTER_VERSION_FILE,
    PUBLICATION_FIELDS,
    AdapterStage,
    ControlClient,
    EngineContainer,
    publication_chain,
)

CLOCKS = Clocks(patience=2, lease=0.3)
STREAM, PLAN = "s", "plan"


class FakeEngine:
    """Weights at a committed (stream, version), (None, 0) for the checkpoint. `fail_at`: applying that
    publication fails; `hang_at`: applying it never returns. Every publication must build on its weights."""

    def __init__(self, fail_at=None, on_apply=None, apply_s: float = 0, hang_at=None):
        self.version, self.point = -1, (None, 0)
        self.applied, self.read, self.aborted = [], [], False
        self.fail_at, self.on_apply, self.apply_s = fail_at, on_apply, apply_s
        self.hang_at = hang_at

    def committed(self, plan_digest):
        assert plan_digest == PLAN
        return self.point

    def apply(self, publication, version):
        self.read.append(Path(publication["manifest_path"]).read_text())
        time.sleep(self.apply_s)
        target = publication["target_version"]
        if target == self.hang_at:
            threading.Event().wait()
        if target == self.fail_at:
            raise RuntimeError(f"version {target} failed")
        assert (
            publication["base_stream_id"],
            publication["base_version"],
        ) == self.point
        self.point = (publication["stream_id"], target)
        self.applied.append(target)
        if version is not None:
            self.version = version
        if self.on_apply:
            self.on_apply(target)
        return {"applied": target}

    def abort_all(self):
        self.aborted = True

    def stop(self):
        pass


class Call:
    """A Modal function call as ControlClient uses one: answered when asked for its result."""

    def __init__(self, method, args):
        self.method, self.args, self.cancelled = method, args, False

    def get(self, timeout):
        return self.method(*self.args)

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def versions(tmp_path):
    """One stream's publications 1-8, each built on the one before; the first on the checkpoint."""
    base = (None, 0)
    for version in range(1, 9):
        _publish_files(tmp_path, version, base)
        base = (STREAM, version)
    return tmp_path


@pytest.fixture
def service():
    service = ControlService(CLOCKS)
    service.started -= CLOCKS.lease  # up long enough to know its engines
    return service


@pytest.fixture
def no_sleep(monkeypatch):
    clock = SimpleNamespace(monotonic=time.monotonic, sleep=lambda seconds: None)
    monkeypatch.setattr(stitch.engine, "time", clock)


def _publish_files(
    versions: Path, target: int, base=(None, 0), stream: str = STREAM, note: str = ""
) -> dict:
    """A sealed publication's manifest, as the trainer writes it; returns its announcement."""
    manifest = f"{stream}/weight_v{target:06d}/manifest.json"
    record = {
        "stream_id": stream,
        "base_stream_id": base[0],
        "base_version": base[1],
        "target_version": target,
        "plan_digest": PLAN,
        "payload_checksum_format": "sha256",
        "note": note,
    }
    path = versions / manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return _announced(versions, manifest)


def _announced(versions: Path, manifest: str) -> dict:
    """The trainer's announcement of the publication at `manifest`."""
    data = (versions / manifest).read_bytes()
    record = json.loads(data)
    return {
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(data).hexdigest(),
        **{key: record[key] for key in PUBLICATION_FIELDS},
    }


def _pub(versions: Path, version: int, stream: str = STREAM) -> dict:
    return _announced(versions, f"{stream}/weight_v{version:06d}/manifest.json")


def _start(service: ControlService, version: int, publication: dict) -> tuple[int, str]:
    """The trainer's /start: what it sends back with each version."""
    return version, service.start(version, publication)["start_id"]


def _modal_class(**methods) -> SimpleNamespace:
    """Methods behind the shape of a Modal class instance: method.spawn(...).get(timeout)."""
    return SimpleNamespace(
        **{
            name: SimpleNamespace(spawn=lambda *args, method=method: Call(method, args))
            for name, method in methods.items()
        }
    )


CONTROL_METHODS = ("open", "next", "served", "synced", "fail")


def _container(
    service,
    engine,
    versions,
    reload=lambda: None,
    adapter_stage=None,
    adapters_in_place=False,
):
    methods = {name: getattr(service, name) for name in CONTROL_METHODS}
    control = ControlClient(_modal_class(**methods), "e0", CLOCKS)
    return EngineContainer(
        engine, control, versions, reload, adapter_stage, adapters_in_place
    )


def _run(*loops) -> list[str]:
    """Runs an open engine's loops; returns why they ended."""
    ended = []

    def run(loop):
        try:
            loop()
        except RuntimeError as error:
            ended.append(str(error))

    for loop in loops:
        threading.Thread(target=run, args=(loop,), daemon=True).start()
    return ended


def test_an_engine_waits_for_the_run_to_start(service, versions):
    engine = FakeEngine()
    container = _container(service, engine, versions)
    opening = threading.Thread(target=container.open, daemon=True)
    opening.start()
    opening.join(0.3)
    assert opening.is_alive() and engine.applied == []

    service.start(2, _pub(versions, 2))
    opening.join(5)
    assert engine.applied == [1, 2]
    assert service.status()["serving"] == {2: 1}


def test_an_engine_catches_up_along_the_chain_and_opens_at_the_newest(
    service, versions
):
    service.start(3, _pub(versions, 3))
    engine = FakeEngine()
    _container(service, engine, versions).open()

    assert engine.applied == [1, 2, 3] and engine.version == 3
    assert service.status()["serving"] == {3: 1}


def test_a_version_published_during_catch_up_is_applied_before_it_opens(
    service, versions
):
    start = _start(service, 3, _pub(versions, 3))
    engine = FakeEngine(
        on_apply=lambda version: (
            version == 2 and service.publish(4, 3, *start, _pub(versions, 4))
        )
    )
    _container(service, engine, versions).open()

    assert engine.applied == [1, 2, 3, 4]
    assert service.status()["serving"] == {4: 1}


def test_an_open_engine_follows_each_version_and_reports_its_stats(service, versions):
    start = _start(service, 3, _pub(versions, 3))
    engine = FakeEngine()
    container = _container(service, engine, versions)
    container.open()
    ended = _run(container.listen, container.follow)

    service.publish(4, 3, *start, _pub(versions, 4))
    stats = service.publish(5, 4, *start, _pub(versions, 5))["engines"]["e0"]["stats"]

    assert engine.applied == [1, 2, 3, 4, 5]
    assert stats["applied"] == 5
    service.start(2, _pub(versions, 2))
    time.sleep(0.3)
    assert ended == ["the control service ended this engine: the run resumed at 2"]


class FakeLoRAEngine:
    """Serves whichever adapter version it applied last; it holds no committed weights. `fail_at`:
    applying that version fails."""

    def __init__(self, fail_at=None):
        self.version, self.applied, self.fail_at = -1, [], fail_at

    def committed(self, plan_digest):
        raise AssertionError("a LoRA version builds on no committed weights")

    def apply(self, publication, version):
        if version == self.fail_at:
            raise RuntimeError(f"version {version} failed")
        self.applied.append((publication["lora_path"], version))
        self.version = version
        return {}


def _lora(version: int) -> dict:
    return {"kind": "lora", "lora_name": "a", "path": f"a_run_v{version}"}


def test_an_engine_takes_only_the_newest_lora_version_then_each_one_after(
    service, versions
):
    start = _start(service, 3, _lora(3))
    engine = FakeLoRAEngine()
    container = _container(service, engine, versions)
    container.open()
    _run(container.listen, container.follow)

    service.publish(4, 3, *start, _lora(4))

    assert engine.applied == [
        (str(versions / "a_run_v3"), 3),
        (str(versions / "a_run_v4"), 4),
    ]
    assert service.status()["serving"] == {4: 1}


def test_a_lora_apply_that_fails_names_the_version_folder(service, versions):
    start = _start(service, 3, _lora(3))
    container = _container(service, FakeLoRAEngine(fail_at=4), versions)
    container.open()
    ended = _run(container.listen, container.follow)

    with pytest.raises(Failed, match="every open engine failed to apply version 4"):
        service.publish(4, 3, *start, _lora(4))

    time.sleep(0.3)
    assert "version 4 from a_run_v4: version 4 failed" in ended


def test_an_engine_stays_heard_while_a_version_applies_longer_than_the_lease(
    service, versions
):
    start = _start(service, 3, _pub(versions, 3))
    engine = FakeEngine()
    container = _container(service, engine, versions)
    container.open()
    engine.apply_s = 3 * CLOCKS.lease
    _run(container.listen, container.follow)

    reply = service.publish(4, 3, *start, _pub(versions, 4))

    assert reply["engines"]["e0"]["version"] == 4
    assert reply["seconds"] >= engine.apply_s


def _volume(tmp_path):
    """What the trainer has written, and this container's view of it, which a reload refreshes."""
    trainer, view = tmp_path / "trainer", tmp_path / "view"
    trainer.mkdir()
    view.mkdir()
    return trainer, view, lambda: shutil.copytree(trainer, view, dirs_exist_ok=True)


def test_a_version_is_read_from_a_view_reloaded_after_it_was_announced(
    service, tmp_path
):
    """A trainer on two nodes commits version 4 node by node: a reload made while applying 3 sees 4's
    manifest before the other node's write."""
    trainer, view, reload = _volume(tmp_path)
    for version in (1, 2):
        _publish_files(
            trainer, version, (STREAM, version - 1) if version > 1 else (None, 0)
        )
    reload()
    start = _start(service, 2, _pub(trainer, 2))
    engine = FakeEngine()
    container = _container(service, engine, view, reload)
    container.open()
    _run(container.listen, container.follow)

    _publish_files(trainer, 3, (STREAM, 2))
    _publish_files(trainer, 4, (STREAM, 3), note="first node only")
    service.publish(3, 2, *start, _pub(trainer, 3))
    four = _publish_files(trainer, 4, (STREAM, 3), note="both nodes")
    service.publish(4, 3, *start, four)

    assert engine.applied == [1, 2, 3, 4]
    assert json.loads(engine.read[-1])["note"] == "both nodes"


def test_versions_that_skip_numbers_are_applied_in_order(service, tmp_path):
    _publish_files(tmp_path, 2)
    _publish_files(tmp_path, 4, (STREAM, 2))
    _publish_files(tmp_path, 6, (STREAM, 4))
    start = _start(service, 4, _pub(tmp_path, 4))
    engine = FakeEngine()
    container = _container(service, engine, tmp_path)
    container.open()
    _run(container.listen, container.follow)

    service.publish(6, 4, *start, _pub(tmp_path, 6))

    assert engine.applied == [2, 4, 6]


def test_an_apply_that_fails_fails_the_version(service, versions):
    start = _start(service, 3, _pub(versions, 3))
    engine = FakeEngine(fail_at=4)
    container = _container(service, engine, versions)
    container.open()
    ended = _run(container.listen, container.follow)

    with pytest.raises(Failed, match="every open engine failed to apply version 4"):
        service.publish(4, 3, *start, _pub(versions, 4))

    assert service.failed == {"e0": 4} and engine.version == 3
    time.sleep(0.3)
    assert set(ended) == {
        f"version 4 from {STREAM}/weight_v000004/manifest.json: version 4 failed",
        "the control service ended this engine: failed to apply version 4",
    }


def test_an_engine_past_a_resumed_run_stops_catching_up(service, versions):
    service.start(3, _pub(versions, 3))
    engine = FakeEngine(
        on_apply=lambda version: version == 2 and service.start(1, _pub(versions, 1))
    )

    with pytest.raises(RuntimeError, match="from before the run's start at 1"):
        _container(service, engine, versions).open()


def test_a_failed_version_is_reported_then_requests_are_aborted_and_the_container_ends(
    service, versions, monkeypatch
):
    ended = []
    monkeypatch.setattr(stitch.engine, "terminate_container", lambda: ended.append(1))
    start = _start(service, 3, _pub(versions, 3))
    engine = FakeEngine(fail_at=4)
    container = _container(service, engine, versions)
    container.open()
    _run(container.listen)
    published = []

    def publish():
        try:
            service.publish(4, 3, *start, _pub(versions, 4))
        except Failed as error:
            published.append(str(error))

    threading.Thread(target=publish, daemon=True).start()
    container._run_or_exit(container.follow)

    assert service.failed == {"e0": 4}
    assert engine.aborted and ended == [1]
    time.sleep(0.3)
    assert published == ["every open engine failed to apply version 4"]


def test_a_leaving_container_ends_its_loops_without_calling_again_or_alarming(
    versions, monkeypatch, caplog
):
    ended, calls = [], []
    monkeypatch.setattr(stitch.engine, "terminate_container", lambda: ended.append(1))

    def closed(*args):
        calls.append(args)
        raise modal.exception.ClientClosed("the container's Modal client closed")

    engine = FakeEngine()
    control = ControlClient(_modal_class(next=closed), "e0", CLOCKS)
    container = EngineContainer(engine, control, versions, lambda: None)
    container.exit()

    container._run_or_exit(container.listen)

    assert len(calls) == 1 and ended == [] and not engine.aborted
    assert "cannot follow" not in caplog.text and "did not answer" not in caplog.text


def test_a_leaving_container_tells_the_control_service_without_waiting(versions):
    left, waited = [], []

    def spawn(engine_id):
        left.append(engine_id)
        return SimpleNamespace(get=lambda **_: waited.append(engine_id))

    control = ControlClient(
        SimpleNamespace(leave=SimpleNamespace(spawn=spawn)), "e0", CLOCKS
    )
    container = EngineContainer(FakeEngine(), control, versions, lambda: None)

    container.exit()

    assert (left, waited) == (["e0"], [])


def test_a_version_cut_off_by_the_container_leaving_is_not_reported_failed(
    service, versions
):
    _start(service, 3, _pub(versions, 3))
    engine = FakeEngine(fail_at=4)
    container = _container(service, engine, versions)
    container.open()
    container.exit()

    container.head, container.publication = 4, _pub(versions, 4)
    with pytest.raises(RuntimeError, match="version 4"):
        container._catch_up()

    assert service.failed == {}


def test_a_hung_apply_ends_the_container_after_the_patience_without_failing_the_version(
    service, versions, monkeypatch
):
    ended = []
    monkeypatch.setattr(stitch.engine, "terminate_container", lambda: ended.append(1))
    start = _start(service, 3, _pub(versions, 3))
    engine = FakeEngine(hang_at=4)
    container = _container(service, engine, versions)
    container.open()
    for loop in (container.listen, container.follow):
        threading.Thread(
            target=container._run_or_exit, args=(loop,), daemon=True
        ).start()
    started = time.monotonic()

    reply = service.publish(4, 3, *start, _pub(versions, 4))

    assert reply["engines"] == {}
    assert time.monotonic() - started >= CLOCKS.patience
    time.sleep(0.3)
    assert engine.aborted and ended == [1] and engine.version == 3


def test_under_a_new_start_the_volume_is_reloaded_before_the_next_apply(
    service, tmp_path
):
    """A resume at 3 publishes a new stream on version 3; this container's view of the Volume predates it."""
    trainer, view, reload = _volume(tmp_path)
    for version in (1, 2, 3, 4):
        _publish_files(
            trainer, version, (STREAM, version - 1) if version > 1 else (None, 0)
        )
    reload()
    _start(service, 3, _pub(trainer, 3))
    engine = FakeEngine()
    container = _container(service, engine, view, reload)
    container.open()
    _run(container.listen, container.follow)

    start = _start(service, 3, _publish_files(trainer, 1, (STREAM, 3), stream="t"))
    service.publish(4, 3, *start, _publish_files(trainer, 2, ("t", 1), stream="t"))

    assert engine.point == ("t", 2) and engine.version == 4
    assert json.loads(engine.read[-1])["stream_id"] == "t"


def test_a_joiner_after_a_resume_replays_the_resumed_stream_to_its_point_then_the_new_one(
    service, versions
):
    """Versions 4-8 of the first stream came after the save the run resumed from: no engine applies them."""
    start = _start(service, 3, _publish_files(versions, 1, (STREAM, 3), stream="t"))
    service.publish(4, 3, *start, _publish_files(versions, 2, ("t", 1), stream="t"))
    engine = FakeEngine()
    _container(service, engine, versions).open()

    assert engine.applied == [1, 2, 3, 1, 2] and engine.point == ("t", 2)
    assert engine.version == 4


def test_weights_the_run_resumed_before_are_off_its_chain(versions):
    head = _publish_files(versions, 1, (STREAM, 3), stream="t")

    assert [
        p["target_version"] for p in publication_chain(versions, head, (None, 0))
    ] == [1, 2, 3, 1]
    with pytest.raises(RuntimeError, match="not on the run's publications to t:1"):
        publication_chain(versions, head, (STREAM, 4))


def test_a_head_manifest_unlike_its_announcement_is_refused(versions):
    announced = _pub(versions, 4)
    _publish_files(versions, 4, (STREAM, 3), note="rewritten")

    with pytest.raises(RuntimeError, match="is not the announced publication"):
        publication_chain(versions, announced, (STREAM, 3))


def test_an_apply_begun_before_a_resume_ends_the_engine_once_it_finishes(
    service, versions
):
    before = _start(service, 3, _pub(versions, 3))
    engine = FakeEngine()
    container = _container(service, engine, versions)
    container.open()
    ended = _run(container.listen, container.follow)
    engine.apply_s = 0.5
    refused = []

    def publish_before():
        try:
            service.publish(4, 3, *before, _pub(versions, 4))
        except Refused as error:
            refused.append(str(error))

    threading.Thread(target=publish_before, daemon=True).start()
    time.sleep(0.1)  # applying version 4 from before

    resumed = _start(service, 3, _pub(versions, 3))
    reply = service.publish(4, 3, *resumed, _pub(versions, 4))

    assert reply["engines"] == {}
    assert refused == ["the call is from another start than the run's"]
    time.sleep(0.3)
    assert set(ended) == {
        "the control service ended this engine: at 4 from before the run's start at 3"
    }


def test_a_call_with_no_answer_in_time_is_sent_again_and_left_to_end(no_sleep):
    calls = []

    class Lost(Call):
        def get(self, timeout):
            assert timeout == 2 * CLOCKS.wait
            if len(calls) == 1:
                raise TimeoutError
            return super().get(timeout)

    def spawn(*args):
        calls.append(Lost(lambda engine_id, version, start_id: {"head": version}, args))
        return calls[-1]

    service = SimpleNamespace(open=SimpleNamespace(spawn=spawn))
    assert ControlClient(service, "e0", CLOCKS).open(3, None) == {"head": 3}
    assert len(calls) == 2 and not calls[0].cancelled


def test_an_unreachable_control_service_is_retried_within_the_patience(no_sleep):
    attempts = []

    def spawn(*args):
        attempts.append(args)
        if len(attempts) < 3:
            raise modal.exception.ConnectionError("restarting")
        return Call(lambda engine_id, version, start_id: {"head": version}, args)

    service = SimpleNamespace(open=SimpleNamespace(spawn=spawn))
    assert ControlClient(service, "e0", CLOCKS).open(3, None) == {"head": 3}
    assert len(attempts) == 3


def test_a_control_service_unreachable_past_the_patience_raises(no_sleep):
    def spawn(*args):
        raise modal.exception.ConnectionError("gone")

    service = SimpleNamespace(open=SimpleNamespace(spawn=spawn))
    clocks = Clocks(patience=0, lease=0.3)
    with pytest.raises(modal.exception.ConnectionError):
        ControlClient(service, "e0", clocks).open(3, None)


def _adapters(tmp_path):
    """The run's adapters on a Volume another run also uses, staged from a mount each reload copies."""
    volume, mount, local = tmp_path / "volume", tmp_path / "mount", tmp_path / "local"
    published = volume / "run"
    published.mkdir(parents=True)
    _save(volume / "other-run", "z", "1")

    def reload():
        shutil.rmtree(mount, ignore_errors=True)
        shutil.copytree(volume, mount, symlinks=True)

    reload()
    return published, AdapterStage(mount, local, local / "run", reload), local / "run"


def _save(published: Path, name: str, commit: str, version: int | None = None) -> str:
    """Miles' commit: a new directory .<name>.<commit>, then <name> switched to it and the old one deleted."""
    directory = published / f".{name}.{commit}"
    directory.mkdir(parents=True)
    (directory / "adapter_model.safetensors").write_bytes(f"{name}@{commit}".encode())
    if version is not None:
        (directory / ADAPTER_VERSION_FILE).write_text(json.dumps({"version": version}))
    link = published / name
    previous = os.readlink(link) if link.is_symlink() else None
    staging = published / f"{directory.name}.link"
    staging.symlink_to(directory.name)
    os.replace(staging, link)
    if previous is not None:
        shutil.rmtree(published / previous)
    return directory.name


def _delete(published: Path, name: str) -> None:
    directory = published / os.readlink(published / name)
    (published / name).unlink()
    shutil.rmtree(directory)


class FakeAdapterEngine(FakeEngine):
    """Records each adapter call, with whether the directory it named last for that adapter was still there."""

    def __init__(self):
        super().__init__()
        self.adapter_calls, self.paths = [], {}

    def update_adapter(self, name, path, version):
        previous = self.paths.get(name)
        kept = previous is None or Path(previous).exists()
        self.adapter_calls.append(("update", name, Path(path).name, version, kept))
        self.paths[name] = path

    def remove_adapter(self, name):
        self.adapter_calls.append(("remove", name))
        self.paths.pop(name)


def test_the_volume_holds_the_directories_links_point_to_and_nothing_else(tmp_path):
    published, stage, local = _adapters(tmp_path)
    _save(published, "a", "1")
    _save(published, "a", "2")
    _save(published, "b", "1")
    # Still being written: no link points to it yet.
    (published / ".c.1").mkdir()
    a, b = str(local / ".a.2"), str(local / ".b.1")

    assert stage.current(None, links=True) == {
        str(local / "a"): a,
        str(local / "b"): b,
    }
    assert stage.current(None, links=False) == {a: a, b: b}
    assert stage.current([str(local / "b"), str(local / "c")], links=True) == {
        str(local / "b"): b,
        str(local / "c"): None,
    }
    assert stage.current([a, str(local / ".a.1")], links=False) == {
        a: a,
        str(local / ".a.1"): None,
    }


def test_a_directory_gone_from_the_volume_before_its_copy_is_not_held(tmp_path):
    published, stage, local = _adapters(tmp_path)
    kept = str(local / _save(published, "a", "1"))
    stage.reload()

    assert stage.copy([kept, str(local / ".b.1")]) == {kept}
    assert not (local / ".b.1").exists()


def test_a_burst_of_adapters_is_copied_at_once(tmp_path, monkeypatch):
    published, stage, local = _adapters(tmp_path)
    names = [_save(published, name, "1") for name in ("a", "b", "c")]
    stage.reload()
    together = threading.Barrier(len(names), timeout=5)
    copytree = shutil.copytree

    def copy_once_all_started(source, target, *args, **kwargs):
        if Path(target).parent == local:
            together.wait()
        return copytree(source, target, *args, **kwargs)

    monkeypatch.setattr(stitch.engine.shutil, "copytree", copy_once_all_started)

    stage.copy(str(local / name) for name in names)

    assert all((local / name).exists() for name in names)


def test_an_engine_copies_every_adapter_on_the_volume_before_it_opens(
    service, versions, tmp_path
):
    published, stage, local = _adapters(tmp_path)
    saved = [_save(published, name, "1") for name in ("a", "b")]
    _start(service, -1, None)
    container = _container(service, FakeEngine(), versions, adapter_stage=stage)

    container.open()

    assert sorted(container.adapters) == sorted(str(local / name) for name in saved)
    assert all((local / name).exists() for name in saved)


def test_a_serving_engine_copies_a_saved_adapter_and_drops_a_deleted_one_before_each_publication_returns(
    service, versions, tmp_path
):
    published, stage, local = _adapters(tmp_path)
    _, start_id = _start(service, -1, None)
    container = _container(service, FakeEngine(), versions, adapter_stage=stage)
    container.open()
    _run(container.listen, container.follow)
    a = str(local / _save(published, "a", "1"))

    service.publish_adapter(a, -1, start_id)
    assert Path(a).exists()

    _delete(published, "a")
    service.publish_adapter(a, -1, start_id)
    assert not Path(a).exists() and container.adapters == {}


def test_in_place_an_adapter_switches_to_each_save_then_drops_its_old_directory(
    service, versions, tmp_path
):
    published, stage, local = _adapters(tmp_path)
    _, start_id = _start(service, -1, None)
    engine = FakeAdapterEngine()
    container = _container(
        service, engine, versions, adapter_stage=stage, adapters_in_place=True
    )
    container.open()
    _run(container.listen, container.follow)
    x = str(local / "x")

    for commit, version in (("c1", 1), ("c2", 2)):
        _save(published, "x", commit, version)
        service.publish_adapter(x, -1, start_id)
    _delete(published, "x")
    service.publish_adapter(x, -1, start_id)

    assert engine.adapter_calls == [
        ("update", "x", ".x.c1", 1, True),
        ("update", "x", ".x.c2", 2, True),
        ("remove", "x"),
    ]
    assert not (local / ".x.c1").exists() and not (local / ".x.c2").exists()


def test_in_place_a_joining_engine_defines_every_adapter_before_it_opens(
    service, versions, tmp_path
):
    published, stage, local = _adapters(tmp_path)
    _save(published, "x", "c1", 3)
    _start(service, -1, None)
    engine = FakeAdapterEngine()
    container = _container(
        service, engine, versions, adapter_stage=stage, adapters_in_place=True
    )

    container.open()

    assert engine.adapter_calls == [("update", "x", ".x.c1", 3, True)]


def test_after_a_control_restart_an_engine_syncs_every_adapter_then_waits(
    versions, tmp_path
):
    published, stage, local = _adapters(tmp_path)
    services, calls = [ControlService(CLOCKS)], []
    services[0].started -= CLOCKS.lease
    _, start_id = _start(services[0], -1, None)

    def calling(name):
        def method(*args):
            calls.append(name)
            return getattr(services[-1], name)(*args)

        return method

    methods = {name: calling(name) for name in CONTROL_METHODS}
    control = ControlClient(_modal_class(**methods), "e0", CLOCKS)
    container = EngineContainer(FakeEngine(), control, versions, lambda: None, stage)
    container.open()
    _run(container.listen, container.follow)
    # Saved while the service restarts: no service hears of it.
    a = str(local / _save(published, "a", "1"))
    restarted = ControlService(CLOCKS)
    restarted.started -= CLOCKS.lease
    restarted.status(-1, start_id, -1)
    services.append(restarted)

    time.sleep(3 * CLOCKS.wait)
    calls.clear()
    time.sleep(5 * CLOCKS.wait)

    assert Path(a).exists()
    assert calls.count("next") <= 7
    b = str(local / _save(published, "b", "1"))
    restarted.publish_adapter(b, -1, start_id)
    assert Path(b).exists()


def test_an_engine_opens_only_after_syncing_adapters_saved_during_its_sync(
    service, versions, tmp_path, monkeypatch
):
    published, stage, local = _adapters(tmp_path)
    a = str(local / _save(published, "a", "1"))
    _, start_id = _start(service, -1, None)
    copy, saved = stage.copy, []

    def copy_then_save_b(paths):
        held = copy(paths)
        if not saved:
            saved.append(str(local / _save(published, "b", "1")))
            threading.Thread(
                target=service.publish_adapter,
                args=(saved[0], -1, start_id),
                daemon=True,
            ).start()
            time.sleep(0.1)
        return held

    monkeypatch.setattr(stage, "copy", copy_then_save_b)
    container = _container(service, FakeEngine(), versions, adapter_stage=stage)

    container.open()

    assert Path(a).exists() and Path(saved[0]).exists()
