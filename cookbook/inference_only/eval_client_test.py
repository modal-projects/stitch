"""Unit tests for the eval client orchestration, with mocked HTTP and sub-second
durations: overlapping evals, 409/retry accounting, and assertion behavior."""

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from cookbook.inference_only import eval_client as ec


class _FakeResponse:
    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self._payload = payload

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

    def post(self, url: str, json: dict | None = None, **kwargs: object) -> _FakeResponse:
        return self._harness.handle(url, json or {})

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        return self._harness.handle_get(url)


class _Harness:
    """Fake Router + Publisher. Chat completions succeed, stamped with the pin.

    Publisher endpoints are async: POST returns 202 {job_id}, GET /job/{id}
    reports pending (job_pending_polls times) then success/failure, and
    GET /status reports the store pointer from the published list.
    """

    def __init__(
        self,
        conflicts_before_success: int = 0,
        publish_ok: bool = True,
        publish_retryable_409s: int = 0,
        job_pending_polls: int = 0,
        job_fails: bool = False,
        job_never_completes: bool = False,
        poll_non_200: bool = False,
        chat_responder: Callable[[dict], _FakeResponse] | None = None,
    ) -> None:
        self.conflicts_remaining = conflicts_before_success
        self.publish_ok = publish_ok
        self.publish_retryable_remaining = publish_retryable_409s
        self.job_pending_polls = job_pending_polls
        self.job_fails = job_fails
        self.job_never_completes = job_never_completes
        self.poll_non_200 = poll_non_200
        # Optional full control over /v1/chat/completions responses (e.g. 409s
        # that end based on a fake clock rather than a fixed count).
        self.chat_responder = chat_responder
        self.fabricated: list[int] = []
        self.delta_fabricated: list[dict] = []
        self.published: list[int] = []
        self.publish_bodies: list[dict] = []
        self.jobs: dict[str, dict] = {}
        self._job_seq = 0

    def _new_job(self, result: dict) -> str:
        self._job_seq += 1
        job_id = f"fc-{self._job_seq}"
        self.jobs[job_id] = {"result": result, "pending": self.job_pending_polls}
        return job_id

    def handle_get(self, url: str) -> _FakeResponse:
        if url.endswith("/loads"):
            # Replicas hold leases at the latest published version (or the base).
            applied = max(self.published) if self.published else 1
            return _FakeResponse(
                200,
                [
                    {
                        "task_id": f"ta-{i}",
                        "upstream": f"h{i}:8000",
                        "load": 1,
                        "applied_version": applied,
                        "leases": {str(applied): 2},
                    }
                    for i in range(3)
                ],
            )
        if "/job/" in url:
            job_id = url.rsplit("/job/", 1)[1]
            if self.poll_non_200:
                return _FakeResponse(503, {})
            job = self.jobs[job_id]
            if self.job_never_completes or job["pending"] > 0:
                job["pending"] -= 1
                return _FakeResponse(200, {"job_id": job_id, "status": "pending"})
            if self.job_fails:
                return _FakeResponse(
                    200, {"job_id": job_id, "status": "failure", "error": "boom"}
                )
            return _FakeResponse(
                200,
                {"job_id": job_id, "status": "success", "result": job["result"]},
            )
        if url.endswith("/status"):
            latest = max(self.published) if self.published else None
            return _FakeResponse(
                200,
                {"latest_version": latest, "staged_versions": sorted(self.published)},
            )
        raise AssertionError(f"unexpected GET url: {url}")

    def handle(self, url: str, body: dict) -> _FakeResponse:
        if url.endswith("/fabricate"):
            self.fabricated.append(body["new_version"])
            job_id = self._new_job(
                {"path": f"/fake/weight_v{body['new_version']:06d}"}
            )
            return _FakeResponse(202, {"status": "accepted", "job_id": job_id})
        if url.endswith("/fabricate_delta"):
            self.delta_fabricated.append(
                {"base_version": body["base_version"], "new_version": body["new_version"]}
            )
            job_id = self._new_job(
                {"path": f"/fake/weight_v{body['new_version']:06d}"}
            )
            return _FakeResponse(202, {"status": "accepted", "job_id": job_id})
        if url.endswith("/publish"):
            if not self.publish_ok:
                return _FakeResponse(500, {"detail": "boom"})
            self.publish_bodies.append(body)
            if self.publish_retryable_remaining > 0:
                self.publish_retryable_remaining -= 1
                return _FakeResponse(
                    409, {"status": "retryable", "detail": "fleet mixed"}
                )
            self.published.append(body["version"])
            job_id = self._new_job({"status": "published", "version": body["version"]})
            return _FakeResponse(202, {"status": "accepted", "job_id": job_id})
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
        raise AssertionError(f"unexpected url: {url}")


def _patch_http(monkeypatch: pytest.MonkeyPatch, harness: _Harness) -> None:
    monkeypatch.setattr(ec.aiohttp, "ClientSession", lambda: _FakeSession(harness))


def _make_client(tmp_path: Path, num_evals: int = 2, **overrides: object) -> ec.EvalClient:
    kwargs = dict(
        router_url="http://router",
        publisher_url="http://publisher",
        run_id="test-run",
        num_evals=num_evals,
        eval_minutes=0.005,  # 0.3s main phase
        straggler_minutes=0.01,  # 0.6s straggler window
        num_sessions=3,
        num_stragglers=1,
        think_seconds=0.01,
        base_version=1,
        results_path=tmp_path / "results.jsonl",
        job_poll_interval=0.01,
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
            "--publisher-url",
            "http://publisher",
            "--run-id",
            "test-run",
            *extra_argv,
        ],
    )
    asyncio.run(ec.main())
    return constructed


def test_eval_orchestration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Overlap, abort-on-publish-failure, pending jobs, delta chain, retryable
    409, and all-stragglers still last a full main phase."""
    harness = _Harness()
    _patch_http(monkeypatch, harness)
    client = _make_client(tmp_path)
    asyncio.run(client.run())
    assert harness.published == [2]
    eval1_ts = [m.timestamp for m in client.metrics if m.eval_num == 1]
    eval2_ts = [m.timestamp for m in client.metrics if m.eval_num == 2]
    assert eval1_ts and eval2_ts
    assert min(eval2_ts) < max(eval1_ts), "eval 2 starts while eval 1 stragglers run"
    t0 = client.publish_accepted_at[2]
    window = [
        m
        for m in client.metrics
        if m.status_code == 200 and t0 <= m.timestamp <= t0 + client.straggler_duration
    ]
    served = {m.weight_version_start for m in window}
    assert {1, 2} <= served
    assert client.results_path.exists()

    for kwargs in (
        {"publish_ok": False},
        {"job_fails": True},
        {"job_never_completes": True, "timeout": 0.05},
        {"poll_non_200": True, "timeout": 0.05},
        {"publish_retryable_409s": 1000, "timeout": 0.05},
    ):
        timeout = kwargs.pop("timeout", None)
        harness = _Harness(**kwargs)
        _patch_http(monkeypatch, harness)
        extra = {} if timeout is None else {"publish_timeout_seconds": timeout}
        client = _make_client(tmp_path, **extra)
        with pytest.raises(RuntimeError, match="aborting run before eval 2"):
            asyncio.run(client.run())
        assert all(m.eval_num == 1 for m in client.metrics), kwargs
        if kwargs.get("publish_retryable_409s"):
            assert harness.published == []

    harness = _Harness(job_pending_polls=2)
    _patch_http(monkeypatch, harness)
    client = _make_client(tmp_path)
    asyncio.run(client.run())
    assert harness.published == [2]
    assert client.publish_accepted_at[2] is not None

    harness = _Harness()
    _patch_http(monkeypatch, harness)
    client = _make_client(tmp_path, num_evals=3, base_version=0, delta=True)
    asyncio.run(client.run())
    assert harness.delta_fabricated == [
        {"base_version": 0, "new_version": 1},
        {"base_version": 1, "new_version": 2},
    ]
    assert harness.fabricated == []
    assert harness.published == [1, 2]
    for body, version in zip(harness.publish_bodies, [1, 2], strict=True):
        assert body["source"] == f"/fake/weight_v{version:06d}"
        assert "delta" not in body
    for next_eval in (2, 3):
        t0 = client.publish_accepted_at[next_eval]
        window = [
            m
            for m in client.metrics
            if m.status_code == 200
            and t0 <= m.timestamp <= t0 + client.straggler_duration
        ]
        served = {m.weight_version_start for m in window}
        assert {next_eval - 2, next_eval - 1} <= served

    harness = _Harness(conflicts_before_success=1)
    _patch_http(monkeypatch, harness)
    client = _make_client(tmp_path, num_evals=1, num_sessions=1, num_stragglers=0)
    asyncio.run(client.run())
    assert any(m.status_code == 409 for m in client.metrics), "409 must be recorded"
    assert any(m.is_retry for m in client.metrics), "retry attempt must be recorded"
    assert any(m.status_code == 200 for m in client.metrics)
    assert client.eval_stats[1]["conflict_count"] >= 1
    assert client.eval_stats[1]["retry_count"] >= 1

    harness = _Harness(publish_retryable_409s=2)
    _patch_http(monkeypatch, harness)
    client = _make_client(tmp_path)
    asyncio.run(client.run())
    assert harness.published == [2]
    assert len(harness.publish_bodies) == 3
    assert harness.fabricated == [2]

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


def test_eval_timeouts_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    status, _, _ = asyncio.run(client._send_request(_Session(), "s1", 3))
    assert status == 200
    assert captured["timeout"].total == 333.0
    headers = captured["headers"]
    assert headers["modal-session-id"] == "s1"
    assert headers["stitch-exact-version"] == "3"

    constructed = _cli_construct(monkeypatch, ["--request-timeout", "240"])
    assert constructed["request_timeout"] == 240.0
    constructed = _cli_construct(monkeypatch, [])
    assert constructed["request_timeout"] == 420.0


def _metric(
    ts: float, eval_num: int, pinned: int, status: int = 200,
    start: int | None = None, end: int | None = None,
) -> ec.RequestMetrics:
    return ec.RequestMetrics(
        timestamp=ts,
        session_id="sess",
        eval_num=eval_num,
        pinned_version=pinned,
        request_num=1,
        status_code=status,
        latency_ms=1.0,
        weight_version_start=start,
        weight_version_end=end,
        is_retry=False,
    )


def test_eval_assertions(tmp_path: Path) -> None:
    client = _make_client(tmp_path, num_evals=1)
    client.metrics.append(_metric(time.time(), 1, 1, start=None, end=None))
    with pytest.raises(AssertionError, match="missing version stamp"):
        client.verify_assertions()

    client = _make_client(tmp_path, num_evals=1)
    client.metrics.append(_metric(time.time(), 1, 1, start=2, end=2))
    with pytest.raises(AssertionError, match="pin violated"):
        client.verify_assertions()

    client = _make_client(tmp_path, num_evals=2)
    t0 = time.time()
    client.publish_accepted_at[2] = t0
    client.metrics.append(_metric(t0 + 0.05, 2, 2, start=2, end=2))
    with pytest.raises(AssertionError, match=r"no 200 served at v1"):
        client.verify_assertions()

    client = _make_client(tmp_path, num_evals=2)
    t0 = time.time()
    client.publish_accepted_at[2] = t0
    client.metrics.append(_metric(t0 + 0.05, 1, 1, start=1, end=1))
    with pytest.raises(AssertionError, match=r"no 200 served at v2"):
        client.verify_assertions()

    client = _make_client(tmp_path, num_evals=2)
    t0 = time.time()
    client.publish_accepted_at[2] = t0
    client.metrics.append(_metric(t0 + 0.05, 1, 1, start=1, end=1))
    client.metrics.append(_metric(t0 + 0.06, 2, 2, start=2, end=2))
    client.verify_assertions()


def test_eval_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fleet snapshots, drain events, boundary rollout, and session deadlines."""
    harness = _Harness()
    _patch_http(monkeypatch, harness)
    client = _make_client(
        tmp_path,
        registry_url="http://registry",
        fleet_poll_interval=0.01,
    )
    asyncio.run(client.run())
    assert client.fleet_snapshots, "poller must have recorded snapshots"
    records = [json.loads(line) for line in client.results_path.read_text().splitlines()]
    snapshots = [r for r in records if "fleet_snapshot" in r]
    metrics = [
        r
        for r in records
        if "fleet_snapshot" not in r
        and "boundary_rollout" not in r
        and "drain_event" not in r
        and "conflict_gap" not in r
        and "summary" not in r
    ]
    assert snapshots, "results file must contain fleet_snapshot records"
    assert all("timestamp" in s for s in snapshots)
    assert len(metrics) == len(client.metrics)
    assert sum(s["request_count"] for s in client.eval_stats.values()) == len(metrics)
    # The terminal summary record closes out a completed run.
    assert records[-1]["summary"]["aborted"] is False
    assert records[-1]["summary"]["conflict_gaps"] == []
    assert records[-1]["summary"]["eval_stats"]
    # Boundary rollout is recorded, not asserted.
    assert client.boundary_rollout[2]["fleet_first_applied_at"] is not None
    boundary_records = [r for r in records if "boundary_rollout" in r]
    assert boundary_records and boundary_records[0]["boundary_rollout"]["new_version"] == 2

    client = _make_client(tmp_path, num_evals=2)
    t0 = time.time()
    client.publish_accepted_at[2] = t0
    client.fleet_snapshots.append(
        {
            "fleet_snapshot": [
                {
                    "task_id": "ta-0",
                    "upstream": "h0:8000",
                    "load": 0,
                    "applied_version": 1,
                    "leases": {"1": 3},
                }
            ],
            "timestamp": t0 + 0.01,
        }
    )
    client.compute_boundary_rollout()
    assert client.boundary_rollout[2]["fleet_first_applied_at"] is None
    assert client.boundary_rollout[2]["publish_accepted_at"] == t0

    client.metrics.append(_metric(t0 - 0.5, 2, 2, start=2, end=2))
    client.metrics.append(_metric(t0 + 0.05, 2, 1, start=1, end=1))
    client.metrics.append(_metric(t0 + 0.10, 2, 2, start=2, end=2))
    client.compute_boundary_rollout()
    assert client.boundary_rollout[2]["first_new_version_200_at"] == pytest.approx(
        t0 + 0.10
    )

    client = _make_client(tmp_path, num_evals=2)
    t0 = time.time()
    client.publish_accepted_at[2] = t0
    client.metrics.append(_metric(t0 + 0.05, 2, 1, start=1, end=1))
    client.compute_boundary_rollout()
    assert client.boundary_rollout[2]["first_new_version_200_at"] is None

    client = _make_client(tmp_path, registry_url="http://registry")

    def loads(draining: bool) -> list[dict]:
        entry: dict = {
            "task_id": "ta-0",
            "upstream": "h0:8000",
            "load": 1,
            "applied_version": 1,
            "leases": {"1": 2},
        }
        if draining:
            entry.update(
                {"draining": True, "sync_state": "catching_up", "target_version": 2}
            )
        return [entry]

    client._record_fleet_snapshot(loads(draining=False))
    client._record_fleet_snapshot(loads(draining=True))
    client._record_fleet_snapshot(loads(draining=True))
    events = [r for r in client.fleet_snapshots if "drain_event" in r]
    assert len(events) == 1
    assert events[0]["drain_event"] == {
        "task_id": "ta-0",
        "applied_version": 1,
        "target_version": 2,
        "sync_state": "catching_up",
    }
    assert "timestamp" in events[0]
    client.write_results()
    records = [
        json.loads(line) for line in client.results_path.read_text().splitlines()
    ]
    assert sum(1 for r in records if "drain_event" in r) == 1

    # Straggler sessions run to main_deadline + straggler window; main-phase
    # sessions stop at main_deadline.
    client = _make_client(tmp_path, num_sessions=3, num_stragglers=2)
    recorded: list[float] = []

    async def fake_eval_session(
        self: ec.EvalClient,
        eval_num: int,
        session_id: str,
        pinned_version: int,
        straggler_deadline: float,
    ) -> None:
        recorded.append(straggler_deadline)

    monkeypatch.setattr(ec.EvalClient, "eval_session", fake_eval_session)

    async def start() -> float:
        main_tasks, straggler_tasks, main_deadline = client._start_sessions(1, 1)
        assert len(straggler_tasks) == 2
        await asyncio.gather(*main_tasks, *straggler_tasks)
        return main_deadline

    main_deadline = asyncio.run(start())
    assert len(recorded) == 3
    assert recorded[0] == pytest.approx(main_deadline + client.straggler_duration), (
        "stragglers run to the end of the straggler window"
    )
    assert recorded[1] == pytest.approx(main_deadline + client.straggler_duration)
    assert recorded[2] == pytest.approx(main_deadline), (
        "main-phase sessions stop at the main-phase deadline"
    )


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
    assert gap["duration_s"] >= gap_seconds
    assert gap["attempts"] > 20
    assert gap["session_id"] == "sess-1"
    assert gap["eval_num"] == 1

    client.compute_stats()
    client.write_results()
    out = capsys.readouterr().out
    assert "GAPS (409 service interruptions, resolved): 1" in out
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
    assert "UNRESOLVED 409 gaps (budget exhausted)" in out
    records = [
        json.loads(line) for line in client.results_path.read_text().splitlines()
    ]
    gap_records = [r for r in records if "conflict_gap" in r]
    assert len(gap_records) == len(client.conflict_gaps)
    assert all(not r["conflict_gap"]["resolved"] for r in gap_records)


def test_abort_preserves_records_and_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed fabricate/publish aborts the run, but the JSONL keeps every
    record produced so far plus a terminal summary with aborted=true."""
    harness = _Harness(publish_ok=False)
    _patch_http(monkeypatch, harness)
    client = _make_client(tmp_path)
    with pytest.raises(RuntimeError, match="aborting run before eval 2"):
        asyncio.run(client.run())
    assert client.results_path.exists()
    records = [
        json.loads(line) for line in client.results_path.read_text().splitlines()
    ]
    metric_records = [r for r in records if "status_code" in r]
    assert metric_records, "eval-1 request metrics must survive the abort"
    assert all(r["eval_num"] == 1 for r in metric_records)
    assert len(metric_records) == len(client.metrics)
    summary = records[-1]["summary"]
    assert summary["aborted"] is True
    assert summary["eval_stats"]
    assert "version_stats" in summary
    assert summary["conflict_gaps"] == []
