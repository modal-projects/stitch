"""The control service: who may open, the barrier, its clocks, and what a restart relearns."""

import threading
import time

import pytest

from stitch.control import Clocks, ControlService, Failed, Refused

CLOCKS = Clocks(patience=0.6, lease=0.3)


def _service(restarted: bool = False) -> ControlService:
    """A service up for a lease already, unless `restarted`."""
    service = ControlService(CLOCKS)
    if not restarted:
        service.started -= CLOCKS.lease
    return service


def _start(service: ControlService, version: int) -> tuple[int, str]:
    """The trainer's /start: what it sends back with each version."""
    return version, service.start(version)["start_id"]


class Engine:
    """An open engine: one thread keeps a next call open, another applies each newer version at
    once unless `stuck`, and reports it served; `fail_at` reports that version failed and leaves,
    `die` stops both. An engine that opened before the service restarted (`opened`) just keeps
    following; `start_id` is the start it knew when it applied `version`."""

    def __init__(
        self,
        service,
        engine_id,
        version,
        stuck=False,
        fail_at=None,
        opened=False,
        start_id=None,
    ):
        self.service, self.id, self.head = service, engine_id, version
        self.applied, self.start_id = (version, start_id), start_id
        self.stuck, self.fail_at, self.exit, self.alive = stuck, fail_at, None, True
        self.newer = threading.Condition()
        if not opened:
            reply = service.open(engine_id, version, start_id)
            assert reply["open"]
            self.start_id = reply["start_id"]
        for loop in (self._listen, self._follow):
            threading.Thread(target=loop, daemon=True).start()

    @property
    def version(self):
        return self.applied[0]

    def die(self):
        self.alive = False

    def _listen(self):
        while self.alive:
            version, start_id = self.applied
            reply = self.service.next(self.id, version, self.head, start_id)
            if not self.alive:
                return
            if "exit" in reply:
                self.exit, self.alive = reply["exit"], False
                return
            with self.newer:
                self.head, self.start_id = (
                    reply["head"],
                    reply["start_id"] or self.start_id,
                )
                self.newer.notify_all()

    def _follow(self):
        while self.alive:
            with self.newer:
                if self.stuck or not self.newer.wait_for(
                    lambda: self.head > self.version, timeout=0.05
                ):
                    continue
            if not self.alive:
                return
            if self.version + 1 == self.fail_at:
                self.service.fail(self.id, self.fail_at)
                self.alive = False
                return
            version, start_id = self.version + 1, self.start_id
            self.applied = (version, start_id)
            reply = self.service.served(
                self.id, version, {"applied": version}, start_id
            )
            if "exit" in reply:
                self.exit, self.alive = reply["exit"], False


def test_an_engine_waits_for_the_run_to_start():
    service = _service()
    assert service.open("e0", 0, None) == {"open": False, "head": 0, "start_id": None}

    start_id = service.start(3)["start_id"]
    assert service.open("e0", 3, start_id) == {
        "open": True,
        "head": 3,
        "start_id": start_id,
    }


def test_an_engine_opens_only_at_the_newest_version():
    service = _service()
    _start(service, 3)

    assert service.open("e0", 1, None)["open"] is False
    assert service.status()["catching_up"] == 1
    assert service.open("e0", 3, None)["open"] is True
    assert service.status()["serving"] == {3: 1}


def test_an_engine_past_a_resumed_run_exits():
    service = _service()
    _start(service, 5)
    engine = Engine(service, "e0", 5)

    service.start(3)
    time.sleep(0.2)
    assert engine.exit == "the run resumed at 3"
    assert "exit" in service.open("e1", 7, None)


def test_a_version_applied_before_a_resume_is_cut_off_after_its_new_bytes_are_published():
    service = _service()
    before = _start(service, 5)
    assert service.open("e0", 5, None)["open"]
    # The trainer resumes at 5 while e0 applies version 6 from before; e1 applies the new 6.
    resumed = _start(service, 5)
    published = []
    threading.Thread(
        target=lambda: published.append(service.publish(6, 5, *resumed)), daemon=True
    ).start()
    time.sleep(0.05)
    service.served("e1", 6, {}, resumed[1])

    assert service.served("e0", 6, {}, before[1]) == {
        "exit": "at 6 from before the run's start at 5"
    }
    time.sleep(0.2)
    assert list(published[0]["engines"]) == ["e1"]


def test_a_version_from_another_start_is_refused():
    service = _service()
    start = _start(service, 3)

    with pytest.raises(Refused, match="from another start"):
        service.publish(4, 3, 3, "an earlier start")
    assert service.publish(4, 3, *start)["version"] == 4


def test_a_publish_still_waiting_when_the_run_restarts_is_refused():
    service = _service()
    start = _start(service, 3)
    Engine(service, "e0", 3, stuck=True)
    refused = []

    def publish():
        try:
            service.publish(4, 3, *start)
        except Refused as error:
            refused.append(str(error))

    waiting = threading.Thread(target=publish, daemon=True)
    waiting.start()
    time.sleep(0.05)
    service.start(3)
    waiting.join(timeout=1)

    assert refused == ["the call is from another start than the run's"]


def test_a_publish_waits_until_every_open_engine_serves_it():
    service = _service()
    start = _start(service, 3)
    Engine(service, "e0", 3)
    Engine(service, "e1", 3)

    reply = service.publish(4, 3, *start)

    assert reply["version"] == 4
    assert reply["engines"] == {
        "e0": {"version": 4, "stats": {"applied": 4}},
        "e1": {"version": 4, "stats": {"applied": 4}},
    }


def test_a_resend_waits_on_the_same_barrier_and_an_older_version_answers_at_once():
    service = _service()
    start = _start(service, 3)
    Engine(service, "e0", 3)
    service.publish(4, 3, *start)

    assert service.publish(4, 3, *start)["engines"]["e0"]["version"] == 4
    started = time.monotonic()
    assert service.publish(2, 1, *start)["engines"]["e0"]["version"] == 4
    assert time.monotonic() - started < 0.1


def test_a_version_may_skip_numbers_when_it_builds_on_the_newest():
    service = _service()
    start = _start(service, 3)

    service.publish(5, 3, *start)

    assert service.head == 5
    with pytest.raises(Refused, match="does not follow the run's 5"):
        service.publish(8, 7, *start)
    assert service.head == 5


def test_a_version_built_on_another_than_the_newest_is_refused():
    service = _service()
    start = _start(service, 3)

    with pytest.raises(Refused, match=r"version 4 \(base 2\) does not follow"):
        service.publish(4, 2, *start)


def test_with_no_open_engine_a_publish_answers_at_once():
    service = _service()
    start = _start(service, 3)
    started = time.monotonic()

    assert service.publish(4, 3, *start)["engines"] == {}
    assert time.monotonic() - started < 0.1


def test_a_live_engine_not_at_the_newest_within_the_patience_is_told_to_exit():
    service = _service()
    start = _start(service, 3)
    Engine(service, "e0", 3)
    late = Engine(service, "e1", 3, stuck=True)

    reply = service.publish(4, 3, *start)

    assert list(reply["engines"]) == ["e0"]
    time.sleep(0.2)
    assert late.exit == f"not at version 4 within {CLOCKS.patience} s"


def test_an_open_engine_not_heard_from_is_dropped_after_the_lease():
    service = _service()
    _start(service, 3)
    Engine(service, "e0", 3)
    service.open("e1", 3, None)  # opens, then never calls again

    assert service.status()["serving"] == {3: 2}
    time.sleep(CLOCKS.lease + 0.1)
    assert service.status()["serving"] == {3: 1}


def test_a_dead_engine_holds_a_publish_for_the_lease_not_the_patience():
    service = _service()
    start = _start(service, 3)
    Engine(service, "e0", 3)
    Engine(service, "e1", 3).die()
    started = time.monotonic()

    assert list(service.publish(4, 3, *start)["engines"]) == ["e0"]
    assert time.monotonic() - started < CLOCKS.patience


def test_an_engine_that_leaves_holds_no_publish_and_its_call_in_flight_ends():
    service = _service()
    start = _start(service, 3)
    Engine(service, "e0", 3)
    leaving = Engine(service, "e1", 3, stuck=True)
    time.sleep(0.05)
    service.leave("e1")
    started = time.monotonic()

    assert list(service.publish(4, 3, *start)["engines"]) == ["e0"]
    assert time.monotonic() - started < CLOCKS.lease
    time.sleep(0.1)
    assert leaving.exit == "it left"


def test_an_older_report_arriving_late_does_not_move_an_engine_back():
    service = _service()
    _, start_id = _start(service, 5)
    service.served("e0", 5, {}, start_id)
    service.next("e0", 4, 5, start_id)

    assert service.status()["serving"] == {5: 1}


def test_a_version_every_open_engine_failed_fails_the_publish():
    service = _service()
    start = _start(service, 3)
    Engine(service, "e0", 3, fail_at=4)
    Engine(service, "e1", 3, fail_at=4)

    with pytest.raises(Failed, match="every open engine failed to apply version 4"):
        service.publish(4, 3, *start)


def test_an_engine_gone_quiet_before_the_publish_does_not_hide_a_failed_version():
    service = _service()
    start = _start(service, 3)
    Engine(service, "e0", 3, fail_at=4)
    # Opens, then goes quiet; nothing reads the list meanwhile.
    service.open("e1", 3, None)
    time.sleep(CLOCKS.lease + 0.1)

    with pytest.raises(Failed, match="every open engine failed to apply version 4"):
        service.publish(4, 3, *start)


def test_a_resend_of_a_version_every_open_engine_failed_fails_again():
    service = _service()
    start = _start(service, 3)
    Engine(service, "e0", 3, fail_at=4)

    for _ in range(2):
        with pytest.raises(Failed, match="every open engine failed"):
            service.publish(4, 3, *start)


def test_a_version_some_engine_applied_is_published_without_the_ones_that_failed():
    service = _service()
    start = _start(service, 3)
    Engine(service, "e0", 3)
    Engine(service, "e1", 3, fail_at=4)

    assert list(service.publish(4, 3, *start)["engines"]) == ["e0"]


def test_a_restarted_service_moves_no_engine_until_the_trainer_names_its_start():
    service = _service(restarted=True)

    assert service.open("e0", 2, "t1") == {"open": False, "head": 2, "start_id": None}
    assert service.status()["head"] is None


def test_a_restarted_service_learns_the_start_from_the_trainers_next_version():
    service = _service(restarted=True)
    # Both served 4 before the restart: e0 under the run's start at 2, e1 under an earlier one.
    kept = Engine(service, "e0", 4, opened=True, start_id="t1")
    cut = Engine(service, "e1", 4, opened=True, start_id="t0")
    time.sleep(0.1)

    reply = service.publish(5, 4, 2, "t1")

    assert reply["engines"] == {"e0": {"version": 5, "stats": {"applied": 5}}}
    assert kept.exit is None
    assert cut.exit == "at 4 from before the run's start at 2"
    assert service.status()["head"] == 5


def test_a_restarted_service_waits_a_lease_for_engines_before_any_barrier():
    service = _service(restarted=True)
    published = []
    threading.Thread(
        target=lambda: published.append(service.publish(5, 4, 2, "t1")), daemon=True
    ).start()
    time.sleep(CLOCKS.lease / 2)
    # Its call reaches the new service only now.
    Engine(service, "e0", 4, opened=True, start_id="t1")

    time.sleep(CLOCKS.lease + 0.2)
    assert published[0]["engines"] == {"e0": {"version": 5, "stats": {"applied": 5}}}


def test_a_restarted_service_takes_the_run_from_the_trainers_first_version():
    service = _service(restarted=True)

    assert service.publish(1, 0, 0, "t1")["engines"] == {}
    assert service.head == 1


def test_a_service_restarted_before_any_version_takes_the_run_from_the_trainers_status_poll():
    service = _service(restarted=True)

    service.status(2, "t1", 2)

    assert service.open("e0", 2, None) == {"open": True, "head": 2, "start_id": "t1"}
    with pytest.raises(Refused, match="from another start"):
        service.status(2, "t0", 2)


def test_trainer_calls_reset_the_idle_clock():
    service = _service()
    time.sleep(0.1)
    assert service.idle_seconds() >= 0.1

    service.status()
    assert service.idle_seconds() < 0.05
