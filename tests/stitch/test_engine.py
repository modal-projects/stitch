"""The engine container against the real control service, with a fake engine."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import modal
import pytest

import stitch.engine
from stitch.control import Clocks, ControlService, Failed, Refused
from stitch.engine import ControlClient, EngineContainer

CLOCKS = Clocks(patience=2, lease=0.3)


class FakeEngine:
    """`fail_prepare_at`: preparing that version fails; `hang_at`: applying that version never returns."""

    def __init__(
        self,
        fail_at=None,
        on_apply=None,
        apply_s: float = 0,
        hang_at=None,
        fail_prepare_at=None,
    ):
        self.version, self.applied, self.read, self.aborted = 0, [], [], False
        self.fail_at, self.on_apply, self.apply_s = fail_at, on_apply, apply_s
        self.hang_at, self.fail_prepare_at = hang_at, fail_prepare_at

    def prepare(self, manifest):
        self.read.append(manifest.read_text())
        if manifest.parent.name == f"weight_v{self.fail_prepare_at or 0:06d}":
            raise RuntimeError(
                "SGLang /prepare_weights_from_delta: [Errno 5] Input/output error"
            )
        return {"read_s": 0.0}

    def apply(self, manifest, version):
        time.sleep(self.apply_s)
        if version == self.hang_at:
            threading.Event().wait()
        if version == self.fail_at:
            raise RuntimeError(f"version {version} failed")
        self.applied.append(version)
        self.version = version
        if self.on_apply:
            self.on_apply(version)
        return {"applied": version}

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
    for version in range(1, 9):
        _write_manifest(tmp_path, version)
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


def _write_manifest(versions: Path, version: int, text: str = "{}") -> None:
    (versions / f"weight_v{version:06d}").mkdir(exist_ok=True)
    (versions / f"weight_v{version:06d}" / "manifest.json").write_text(text)


def _start(service: ControlService, version: int) -> tuple[int, str]:
    """The trainer's /start: what it sends back with each version."""
    return version, service.start(version)["start_id"]


def _modal_class(**methods) -> SimpleNamespace:
    """Methods behind the shape of a Modal class instance: method.spawn(...).get(timeout)."""
    return SimpleNamespace(
        **{
            name: SimpleNamespace(spawn=lambda *args, method=method: Call(method, args))
            for name, method in methods.items()
        }
    )


def _container(service, engine, versions, reload=lambda: None):
    methods = {
        name: getattr(service, name) for name in ("open", "next", "served", "fail")
    }
    control = ControlClient(_modal_class(**methods), "e0", CLOCKS)
    return EngineContainer(engine, control, versions, reload)


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

    service.start(2)
    opening.join(5)
    assert engine.applied == [1, 2]
    assert service.status()["serving"] == {2: 1}


def test_an_engine_catches_up_one_version_at_a_time_and_opens_at_the_newest(
    service, versions
):
    service.start(3)
    engine = FakeEngine()
    _container(service, engine, versions).open()

    assert engine.applied == [1, 2, 3]
    assert service.status()["serving"] == {3: 1}


def test_a_version_published_during_catch_up_is_applied_before_it_opens(
    service, versions
):
    start = _start(service, 3)
    engine = FakeEngine(
        on_apply=lambda version: version == 2 and service.publish(4, 3, *start)
    )
    _container(service, engine, versions).open()

    assert engine.applied == [1, 2, 3, 4]
    assert service.status()["serving"] == {4: 1}


def test_an_open_engine_follows_each_version_and_reports_its_stats(service, versions):
    start = _start(service, 3)
    engine = FakeEngine()
    container = _container(service, engine, versions)
    container.open()
    ended = _run(container.listen, container.follow)

    service.publish(4, 3, *start)
    stats = service.publish(5, 4, *start)["engines"]["e0"]["stats"]

    assert engine.applied == [1, 2, 3, 4, 5]
    assert stats["apply"]["applied"] == 5
    service.start(2)
    time.sleep(0.3)
    assert ended == ["the control service ended this engine: the run resumed at 2"]


def test_an_engine_stays_heard_while_a_version_applies_longer_than_the_lease(
    service, versions
):
    start = _start(service, 3)
    engine = FakeEngine()
    container = _container(service, engine, versions)
    container.open()
    engine.apply_s = 3 * CLOCKS.lease
    _run(container.listen, container.follow)

    reply = service.publish(4, 3, *start)

    assert reply["engines"]["e0"]["version"] == 4
    assert reply["seconds"] >= engine.apply_s


def test_a_version_is_read_from_a_view_reloaded_after_it_was_announced(
    service, tmp_path
):
    """A trainer on two nodes commits version 4 node by node: a reload made while applying 3 sees 4's
    manifest before the other node's file."""
    volume = {version: "{}" for version in (1, 2)}

    def reload():
        for version, text in volume.items():
            _write_manifest(tmp_path, version, text)

    start = _start(service, 2)
    engine = FakeEngine()
    container = _container(service, engine, tmp_path, reload)
    container.open()
    _run(container.listen, container.follow)

    volume |= {3: "{}", 4: "first node only"}
    service.publish(3, 2, *start)
    volume[4] = "both nodes"
    service.publish(4, 3, *start)

    assert engine.applied == [1, 2, 3, 4]
    assert engine.read[-1] == "both nodes"


def test_versions_that_skip_numbers_are_applied_in_order(service, tmp_path):
    for version in (2, 4, 6):
        _write_manifest(tmp_path, version)
    start = _start(service, 4)
    engine = FakeEngine()
    container = _container(service, engine, tmp_path)
    container.open()
    _run(container.listen, container.follow)

    service.publish(6, 4, *start)

    assert engine.applied == [2, 4, 6]


def test_a_prepare_that_fails_fails_the_version(service, versions):
    start = _start(service, 3)
    engine = FakeEngine(fail_prepare_at=4)
    container = _container(service, engine, versions)
    container.open()
    ended = _run(container.listen, container.follow)

    with pytest.raises(Failed, match="every open engine failed to apply version 4"):
        service.publish(4, 3, *start)

    assert service.failed == {"e0": 4} and engine.version == 3
    time.sleep(0.3)
    assert set(ended) == {
        f"version 4 from {versions}/weight_v000004/manifest.json: "
        "SGLang /prepare_weights_from_delta: [Errno 5] Input/output error",
        "the control service ended this engine: failed to apply version 4",
    }


def test_an_engine_past_a_resumed_run_stops_catching_up(service, versions):
    service.start(3)
    engine = FakeEngine(on_apply=lambda version: version == 2 and service.start(1))

    with pytest.raises(RuntimeError, match="from before the run's start at 1"):
        _container(service, engine, versions).open()


def test_a_failed_version_is_reported_then_requests_are_aborted_and_the_container_ends(
    service, versions, monkeypatch
):
    ended = []
    monkeypatch.setattr(stitch.engine, "terminate_container", lambda: ended.append(1))
    start = _start(service, 3)
    engine = FakeEngine(fail_at=4)
    container = _container(service, engine, versions)
    container.open()
    _run(container.listen)
    published = []

    def publish():
        try:
            service.publish(4, 3, *start)
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
    _start(service, 3)
    engine = FakeEngine(fail_at=4)
    container = _container(service, engine, versions)
    container.open()
    container.exit()

    container.head = 4
    with pytest.raises(RuntimeError, match="version 4"):
        container._apply_next()

    assert service.failed == {}


def test_a_hung_apply_ends_the_container_after_the_patience_without_failing_the_version(
    service, versions, monkeypatch
):
    ended = []
    monkeypatch.setattr(stitch.engine, "terminate_container", lambda: ended.append(1))
    start = _start(service, 3)
    engine = FakeEngine(hang_at=4)
    container = _container(service, engine, versions)
    container.open()
    for loop in (container.listen, container.follow):
        threading.Thread(
            target=container._run_or_exit, args=(loop,), daemon=True
        ).start()
    started = time.monotonic()

    reply = service.publish(4, 3, *start)

    assert reply["engines"] == {}
    assert time.monotonic() - started >= CLOCKS.patience
    time.sleep(0.3)
    assert engine.aborted and ended == [1] and engine.version == 3


def test_under_a_new_start_the_volume_is_reloaded_before_the_next_apply(
    service, tmp_path
):
    """Version 4, written before the trainer resumed at 3, is still in this container's view of the
    Volume after the trainer removed it and wrote another 4."""
    volume = {version: "{}" for version in (1, 2, 3)} | {4: "removed"}

    def reload():
        for version, text in volume.items():
            _write_manifest(tmp_path, version, text)

    _start(service, 3)
    engine = FakeEngine()
    container = _container(service, engine, tmp_path, reload)
    container.open()
    _run(container.listen, container.follow)

    volume[4] = "new"
    start = _start(service, 3)
    service.publish(4, 3, *start)

    assert engine.read[-1] == "new"


def test_an_apply_begun_before_a_resume_ends_the_engine_once_it_finishes(
    service, versions
):
    before = _start(service, 3)
    engine = FakeEngine()
    container = _container(service, engine, versions)
    container.open()
    ended = _run(container.listen, container.follow)
    engine.apply_s = 0.5
    refused = []

    def publish_before():
        try:
            service.publish(4, 3, *before)
        except Refused as error:
            refused.append(str(error))

    threading.Thread(target=publish_before, daemon=True).start()
    time.sleep(0.1)  # applying version 4 from before

    resumed = _start(service, 3)
    reply = service.publish(4, 3, *resumed)

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
