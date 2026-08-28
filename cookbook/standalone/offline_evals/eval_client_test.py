"""Unit tests for the eval client, with mocked HTTP and sub-second durations:
pinned sessions, 409 backoff / 503 Retry-After retry accounting, conflict-gap
records, per-response stamp assertions, and the session tombstone contract."""

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from cookbook.standalone.offline_evals import eval_client as ec


class _FakeResponse:
    def __init__(self, status: int, payload: dict, headers: dict | None = None) -> None:
        self.status = status
        self._payload = payload
        self.headers = headers or {}

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, harness: "_Harness") -> None:
        self._harness = harness

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    def post(
        self, url: str, json: dict | None = None, **kwargs: object
    ) -> _FakeResponse:
        return self._harness.handle(url, json or {})

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        return self._harness.handle_get(url)


class _Harness:
    """Fake Router. Chat completions succeed, stamped with the pin; session
    tombstones are accepted and recorded."""

    def __init__(
        self,
        conflicts_before_success: int = 0,
        chat_responder: Callable[[dict], _FakeResponse] | None = None,
    ) -> None:
        self.conflicts_remaining = conflicts_before_success
        # Optional full control over /v1/chat/completions responses (e.g. 409s
        # that end based on a fake clock rather than a fixed count).
        self.chat_responder = chat_responder
        self.ended_sessions: list[str] = []

    def handle_get(self, url: str) -> _FakeResponse:
        raise AssertionError(f"unexpected GET url: {url}")

    def handle(self, url: str, body: dict) -> _FakeResponse:
        if url.endswith("/v1/chat/completions"):
            if self.chat_responder is not None:
                return self.chat_responder(body)
            pinned = body["weight_version"]["exact_version"]
            if self.conflicts_remaining > 0:
                self.conflicts_remaining -= 1
                return _FakeResponse(409, {})
            return _FakeResponse(
                200,
                {
                    "weight_version_start": pinned,
                    "weight_version_end": pinned,
                    "choices": [],
                },
            )
        if "/sessions/" in url and url.endswith("/end"):
            session_id = url.split("/sessions/")[1][: -len("/end")]
            self.ended_sessions.append(session_id)
            return _FakeResponse(200, {})
        raise AssertionError(f"unexpected url: {url}")


def _patch_http(monkeypatch: pytest.MonkeyPatch, harness: _Harness) -> None:
    monkeypatch.setattr(ec.aiohttp, "ClientSession", lambda: _FakeSession(harness))


def _make_client(
    tmp_path: Path, num_evals: int = 2, **overrides: object
) -> ec.EvalClient:
    kwargs = dict(
        router_url="http://router",
        num_evals=num_evals,
        eval_minutes=0.005,  # 0.3s main phase
        straggler_minutes=0.01,  # 0.6s straggler window
        num_sessions=3,
        num_stragglers=1,
        think_seconds=0.01,
        base_version=1,
        results_path=tmp_path / "results.jsonl",
    )
    kwargs.update(overrides)
    return ec.EvalClient(**kwargs)  # type: ignore[arg-type]


def _cli_construct(monkeypatch, extra_argv: list[str]) -> dict:
    constructed: dict = {}

    class _RecordingClient:
        def __init__(self, **kwargs: object) -> None:
            constructed.update(kwargs)

        async def run(self) -> None:
            pass

    monkeypatch.setattr(ec, "EvalClient", _RecordingClient)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_client",
            "--router-url",
            "http://router",
            *extra_argv,
        ],
    )
    asyncio.run(ec.main())
    return constructed


def test_eval_409_retry_and_all_straggler_main_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """409s and retries are recorded per attempt; when every session is a
    straggler the main phase is still time-based."""
    harness = _Harness(conflicts_before_success=1)
    _patch_http(monkeypatch, harness)
    client = _make_client(tmp_path, num_evals=1, num_sessions=1, num_stragglers=0)
    asyncio.run(client.run())
    assert any(m.status_code == 409 for m in client.metrics), "409 must be recorded"
    assert any(m.is_retry for m in client.metrics), "retry attempt must be recorded"
    assert any(m.status_code == 200 for m in client.metrics)
    assert client.eval_stats[1]["conflict_count"] >= 1
    assert client.eval_stats[1]["retry_count"] >= 1

    harness = _Harness()
    _patch_http(monkeypatch, harness)
    client = _make_client(tmp_path, num_evals=1, num_sessions=3, num_stragglers=3)
    started = time.time()
    events: list[tuple[str, float]] = []
    monkeypatch.setattr(
        client, "print_event", lambda event: events.append((event, time.time()))
    )
    asyncio.run(client.run())
    main_end = next(t for name, t in events if name == "eval 1 main phase end")
    assert main_end - started >= client.eval_duration * 0.9, (
        "when every session is a straggler the main phase is still time-based"
    )
    assert any(m.timestamp > main_end for m in client.metrics)
    assert client.results_path.exists()


def test_eval_timeouts_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--request-timeout must reach aiohttp so stragglers survive a paused engine."""
    default_client = _make_client(tmp_path)
    assert default_client.request_timeout == 420.0
    client = _make_client(tmp_path, request_timeout=333.0)
    assert client.request_timeout == 333.0
    captured: dict = {}

    class _Session:
        def post(self, url, json=None, headers=None, timeout=None):
            captured["timeout"] = timeout
            captured["headers"] = headers
            return _FakeResponse(200, {})

    status, _, _, _ = asyncio.run(client._send_request(_Session(), "s1", 3))
    assert status == 200
    assert captured["timeout"].total == 333.0
    headers = captured["headers"]
    assert headers["modal-session-id"] == "s1"
    assert headers["stitch-exact-version"] == "3"

    constructed = _cli_construct(monkeypatch, ["--request-timeout", "240"])
    assert constructed["request_timeout"] == 240.0
    constructed = _cli_construct(monkeypatch, [])
    assert constructed["request_timeout"] == 420.0


def _install_fake_clock(monkeypatch: pytest.MonkeyPatch, t0: float) -> list[float]:
    """Drive eval_client time/sleep from a mutable clock: only sleeps advance it."""
    clock = [t0]
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        clock[0] += seconds
        await real_sleep(0)

    monkeypatch.setattr(ec.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(ec.time, "time", lambda: clock[0])
    return clock


def test_session_409_backoff_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capped exponential backoff: ~2s start, doubling to a ~15s cap, jittered."""
    monkeypatch.setattr(ec.random, "uniform", lambda lo, hi: 1.0)
    assert ec.session_409_backoff_seconds(1) == pytest.approx(2.0)
    assert ec.session_409_backoff_seconds(2) == pytest.approx(4.0)
    assert ec.session_409_backoff_seconds(3) == pytest.approx(8.0)
    assert ec.session_409_backoff_seconds(4) == pytest.approx(15.0)
    assert ec.session_409_backoff_seconds(100) == pytest.approx(15.0)
    # Jitter scales the capped base within [0.5, 1.5].
    monkeypatch.setattr(ec.random, "uniform", lambda lo, hi: lo)
    assert ec.session_409_backoff_seconds(3) == pytest.approx(8.0 * 0.5)
    monkeypatch.setattr(ec.random, "uniform", lambda lo, hi: hi)
    assert ec.session_409_backoff_seconds(3) == pytest.approx(8.0 * 1.5)


def test_session_409_budget_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Budget defaults to the straggler window; ctor/CLI overrides reach the client."""
    default_client = _make_client(tmp_path)
    assert default_client.session_409_budget_seconds == pytest.approx(
        default_client.straggler_duration
    )
    client = _make_client(tmp_path, session_409_budget_seconds=45.0)
    assert client.session_409_budget_seconds == 45.0
    constructed = _cli_construct(monkeypatch, ["--session-409-budget-seconds", "45"])
    assert constructed["session_409_budget_seconds"] == 45.0
    constructed = _cli_construct(monkeypatch, [])
    assert constructed["session_409_budget_seconds"] is None


def test_session_survives_rollout_length_409_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """A session on the default budget (= straggler window) rides out a ~6 min
    409 rollout gap and resumes; the gap is recorded resolved, not a failure."""
    monkeypatch.setattr(ec.random, "uniform", lambda lo, hi: 1.0)
    t0 = 1_000_000.0
    gap_seconds = 360.0

    def chat_responder(body: dict) -> _FakeResponse:
        if clock[0] < t0 + gap_seconds:
            return _FakeResponse(409, {})
        pinned = body["weight_version"]["exact_version"]
        return _FakeResponse(
            200,
            {
                "weight_version_start": pinned,
                "weight_version_end": pinned,
                "choices": [],
            },
        )

    harness = _Harness(chat_responder=chat_responder)
    _patch_http(monkeypatch, harness)
    client = _make_client(
        tmp_path,
        num_evals=1,
        num_sessions=1,
        num_stragglers=1,
        straggler_minutes=10,
        think_seconds=5.0,
    )
    assert client.session_409_budget_seconds == 600.0  # default: straggler window
    clock = _install_fake_clock(monkeypatch, t0)
    asyncio.run(
        client.eval_session(
            eval_num=1,
            session_id="sess-1",
            pinned_version=1,
            straggler_deadline=t0 + 400,
        )
    )
    statuses = [m.status_code for m in client.metrics]
    assert 409 in statuses
    assert statuses[-1] == 200, "session must resume after the gap"
    assert len(client.conflict_gaps) == 1
    gap = client.conflict_gaps[0]
    assert gap["resolved"] is True
    assert gap["status"] == 409
    assert gap["duration_s"] >= gap_seconds
    assert gap["attempts"] > 20
    assert gap["session_id"] == "sess-1"
    assert gap["eval_num"] == 1

    client.compute_stats()
    client.write_results()
    out = capsys.readouterr().out
    assert "GAPS (transient service interruptions, resolved): 1" in out
    assert "UNRESOLVED" not in out
    records = [
        json.loads(line) for line in client.results_path.read_text().splitlines()
    ]
    gap_records = [r for r in records if "conflict_gap" in r]
    assert len(gap_records) == 1
    assert gap_records[0]["conflict_gap"]["resolved"] is True
    assert gap_records[0]["conflict_gap"]["attempts"] == gap["attempts"]


def test_session_409_budget_exhaustion_records_unresolved_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """When the cumulative 409 budget runs out, the 409 is the request's final
    status and the gap is recorded unresolved."""
    monkeypatch.setattr(ec.random, "uniform", lambda lo, hi: 1.0)
    t0 = 1_000_000.0
    harness = _Harness(chat_responder=lambda body: _FakeResponse(409, {}))
    _patch_http(monkeypatch, harness)
    client = _make_client(
        tmp_path,
        num_evals=1,
        num_sessions=1,
        num_stragglers=1,
        think_seconds=5.0,
        session_409_budget_seconds=10.0,
    )
    _install_fake_clock(monkeypatch, t0)
    asyncio.run(
        client.eval_session(
            eval_num=1,
            session_id="sess-1",
            pinned_version=1,
            straggler_deadline=t0 + 30,
        )
    )
    assert client.metrics
    assert all(m.status_code == 409 for m in client.metrics)
    first_request = [m for m in client.metrics if m.request_num == 1]
    assert len(first_request) == 4, "1 initial attempt + 3 backoff retries"
    assert first_request[-1].status_code == 409, "409 stands as the final status"
    assert client.conflict_gaps
    assert all(not gap["resolved"] for gap in client.conflict_gaps)
    first_gap = client.conflict_gaps[0]
    assert first_gap["attempts"] == 4
    assert first_gap["duration_s"] == pytest.approx(10.0)

    client.compute_stats()
    client.write_results()
    out = capsys.readouterr().out
    assert "UNRESOLVED gaps (budget exhausted)" in out
    records = [
        json.loads(line) for line in client.results_path.read_text().splitlines()
    ]
    gap_records = [r for r in records if "conflict_gap" in r]
    assert len(gap_records) == len(client.conflict_gaps)
    assert all(not r["conflict_gap"]["resolved"] for r in gap_records)


def test_session_503_honors_retry_after(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 503 retries after the server's Retry-After and resumes; the streak is
    recorded as a resolved 503 gap. The 409 backoff applies only to 409s."""
    t0 = 1_000_000.0
    unavailable_seconds = 7.0

    def chat_responder(body: dict) -> _FakeResponse:
        if clock[0] < t0 + unavailable_seconds:
            return _FakeResponse(503, {}, headers={"Retry-After": "3"})
        pinned = body["weight_version"]["exact_version"]
        return _FakeResponse(
            200,
            {
                "weight_version_start": pinned,
                "weight_version_end": pinned,
                "choices": [],
            },
        )

    harness = _Harness(chat_responder=chat_responder)
    _patch_http(monkeypatch, harness)
    client = _make_client(
        tmp_path,
        num_evals=1,
        num_sessions=1,
        num_stragglers=1,
        think_seconds=1.0,
        session_409_budget_seconds=30.0,
    )
    clock = _install_fake_clock(monkeypatch, t0)
    asyncio.run(
        client.eval_session(
            eval_num=1,
            session_id="sess-1",
            pinned_version=2,
            straggler_deadline=t0 + 30,
        )
    )
    statuses = [m.status_code for m in client.metrics]
    assert statuses[:4] == [503, 503, 503, 200]
    # Retry-After: 3 paces these retries; the 2s/4s backoff applies only to 409s.
    timestamps = [m.timestamp for m in client.metrics[:4]]
    assert timestamps[1] - timestamps[0] == pytest.approx(3.0)
    assert timestamps[2] - timestamps[1] == pytest.approx(3.0)
    client.compute_stats()
    assert client.eval_stats[1]["unavailable_count"] == 3
    assert len(client.conflict_gaps) == 1
    gap = client.conflict_gaps[0]
    assert gap["resolved"] is True
    assert gap["status"] == 503
    assert gap["attempts"] == 3


def test_session_tombstoned_on_completion_and_give_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every trajectory ends with exactly one best-effort tombstone call,
    whether the session ran to its deadline or gave up on a transient streak."""
    harness = _Harness()
    _patch_http(monkeypatch, harness)
    client = _make_client(tmp_path, num_evals=1, num_sessions=3, num_stragglers=1)
    asyncio.run(client.run())
    assert len(harness.ended_sessions) == 3
    assert set(harness.ended_sessions) == {m.session_id for m in client.metrics}

    harness = _Harness(chat_responder=lambda body: _FakeResponse(409, {}))
    _patch_http(monkeypatch, harness)
    client = _make_client(
        tmp_path,
        num_evals=1,
        num_sessions=1,
        num_stragglers=1,
        session_409_budget_seconds=0.0,
    )
    asyncio.run(client.run())
    assert len(harness.ended_sessions) == 1, "give-up still tombstones"


def test_stamp_violation_recorded_per_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 200 whose stamps are missing or off the pin is a recorded violation,
    and the summary carries the violation list."""
    pinned_seen: list[int] = []

    def chat_responder(body: dict) -> _FakeResponse:
        pinned = body["weight_version"]["exact_version"]
        pinned_seen.append(pinned)
        # Serve one version behind the pin (a stale replica).
        return _FakeResponse(
            200,
            {
                "weight_version_start": pinned - 1,
                "weight_version_end": pinned - 1,
                "choices": [],
            },
        )

    harness = _Harness(chat_responder=chat_responder)
    _patch_http(monkeypatch, harness)
    client = _make_client(tmp_path, num_evals=1, num_sessions=1, num_stragglers=0)
    asyncio.run(client.run())
    assert client.stamp_violations, "every 200 was off the pin"
    assert all(v["pinned_version"] == 1 for v in client.stamp_violations)
    assert all(v["weight_version_start"] == 0 for v in client.stamp_violations)
    records = [
        json.loads(line) for line in client.results_path.read_text().splitlines()
    ]
    violation_records = [r for r in records if "stamp_violation" in r]
    assert len(violation_records) == len(client.stamp_violations)
    summary = next(r for r in records if "summary" in r)["summary"]
    assert len(summary["stamp_violations"]) == len(client.stamp_violations)
