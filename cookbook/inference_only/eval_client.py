"""Eval client: concurrent version-pinned sessions against the router, with
fabricate/publish at eval boundaries and overlap assertions at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
import uuid
from dataclasses import asdict, dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

SESSION_409_BACKOFF_START_SECONDS = 2.0
SESSION_409_BACKOFF_CAP_SECONDS = 15.0
FLEET_POLL_INTERVAL_SECONDS = 10.0
JOB_POLL_INTERVAL_SECONDS = 10.0
PUBLISH_TIMEOUT_SECONDS = 7200.0


def session_409_backoff_seconds(attempt: int) -> float:
    """Jittered capped exponential backoff for the ``attempt``-th 409 retry
    (1-based): ~2s, ~4s, ~8s, then capped at ~15s, each x uniform(0.5, 1.5)."""
    # Cap the shift so a long 409 streak can't overflow the float multiply.
    shift = min(attempt - 1, 8)
    base = min(
        SESSION_409_BACKOFF_CAP_SECONDS,
        SESSION_409_BACKOFF_START_SECONDS * 2**shift,
    )
    return base * random.uniform(0.5, 1.5)


@dataclass
class RequestMetrics:
    timestamp: float
    session_id: str
    eval_num: int
    pinned_version: int
    request_num: int
    status_code: int
    latency_ms: float
    weight_version_start: int | None
    weight_version_end: int | None
    is_retry: bool


class EvalClient:
    def __init__(
        self,
        router_url: str,
        publisher_url: str,
        run_id: str,
        num_evals: int,
        eval_minutes: int,
        straggler_minutes: int,
        num_sessions: int,
        num_stragglers: int,
        think_seconds: float,
        base_version: int,
        results_path: Path,
        registry_url: str | None = None,
        delta: bool = False,
        fleet_poll_interval: float = FLEET_POLL_INTERVAL_SECONDS,
        publish_timeout_seconds: float = PUBLISH_TIMEOUT_SECONDS,
        job_poll_interval: float = JOB_POLL_INTERVAL_SECONDS,
        request_timeout: float = 420.0,
        session_409_budget_seconds: float | None = None,
    ):
        self.router_url = router_url.rstrip("/")
        self.publisher_url = publisher_url.rstrip("/")
        self.registry_url = registry_url.rstrip("/") if registry_url else None
        self.run_id = run_id
        self.num_evals = num_evals
        self.eval_duration = eval_minutes * 60
        self.straggler_duration = straggler_minutes * 60
        self.num_sessions = num_sessions
        self.num_stragglers = num_stragglers
        self.think_seconds = think_seconds
        self.base_version = base_version
        self.delta = delta
        self.results_path = results_path
        self.fleet_poll_interval = fleet_poll_interval
        self.publish_timeout_seconds = publish_timeout_seconds
        self.job_poll_interval = job_poll_interval
        # Straggler requests must survive a paused engine: a first cpu-mode delta
        # stage pauses generation for the full image compile (minutes).
        self.request_timeout = request_timeout
        # Per-session cumulative wall-clock budget for riding out 409s (a
        # rollout's staging/compile gap). Defaults to the straggler window.
        self.session_409_budget_seconds = (
            session_409_budget_seconds
            if session_409_budget_seconds is not None
            else self.straggler_duration
        )

        self.start_time = time.time()
        self.metrics: list[RequestMetrics] = []
        self.conflict_gaps: list[dict[str, Any]] = []
        self.eval_stats: dict[int, dict[str, Any]] = {}
        self.version_stats: dict[int, int] = {}
        self.publish_accepted_at: dict[int, float] = {}
        # Raw registry /loads snapshots; kept OUT of self.metrics so the metrics
        # and assertion code (which iterates RequestMetrics only) never sees them.
        self.fleet_snapshots: list[dict[str, Any]] = []
        self._drained_task_ids: set[str] = set()
        self.boundary_rollout: dict[int, dict[str, Any]] = {}
        self._eval_loop_completed = False
        self._results_file: TextIOWrapper | None = None
        self._results_finalized = False

    def _open_results_file(self) -> None:
        """Open (truncating) the results JSONL for incremental writes."""
        self._results_file = open(self.results_path, "w")

    def _close_results_file(self) -> None:
        if self._results_file is not None:
            self._results_file.close()
            self._results_file = None

    def _emit_record(self, record: dict[str, Any]) -> None:
        """Append one JSONL record and flush, so an abort never loses it."""
        if self._results_file is not None:
            self._results_file.write(json.dumps(record) + "\n")
            self._results_file.flush()

    def elapsed_str(self) -> str:
        elapsed = int(time.time() - self.start_time)
        return f"{elapsed // 60:02d}:{elapsed % 60:02d}"

    def print_event(self, event: str) -> None:
        print(f"[t+{self.elapsed_str()}] {event}")

    async def _post_and_poll_job(
        self,
        session: aiohttp.ClientSession,
        path: str,
        payload: dict[str, Any],
        retry_retryable_409: bool = False,
    ) -> dict[str, Any] | None:
        """POST a publisher job (202 + job_id) and poll ``/job/{id}``.

        A 200 no-op is immediate success. With ``retry_retryable_409``, a 409
        ``{"status": "retryable"}`` is retried until ``publish_timeout_seconds``.
        """
        deadline = time.time() + self.publish_timeout_seconds
        attempt = 0
        while True:
            attempt += 1
            try:
                async with session.post(
                    f"{self.publisher_url}{path}",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    data = await resp.json() if resp.status in (200, 202, 409) else {}
                    if resp.status == 200 and data.get("status") == "no-op":
                        return data
                    if resp.status == 202:
                        job_id = data.get("job_id")
                        if not job_id:
                            logger.error("%s response missing job_id", path)
                            return None
                        break
                    if (
                        retry_retryable_409
                        and resp.status == 409
                        and data.get("status") == "retryable"
                        and time.time() < deadline
                    ):
                        logger.info(
                            "%s attempt %d got 409 retryable; retrying in %.1fs",
                            path,
                            attempt,
                            self.job_poll_interval,
                        )
                        await asyncio.sleep(self.job_poll_interval)
                        continue
                    logger.error("%s failed: %d %s", path, resp.status, data)
                    return None
            except Exception as exc:
                logger.error("%s error: %r", path, exc)
                return None

        while True:
            await asyncio.sleep(self.job_poll_interval)
            # Check the deadline on every iteration, including after non-200
            # responses, poll errors, and non-terminal statuses (e.g. "unknown")
            # that `continue` below, so an outage can't loop forever.
            if time.time() >= deadline:
                logger.error(
                    "job %s (%s) not done after %.0fs; giving up",
                    job_id,
                    path,
                    self.publish_timeout_seconds,
                )
                return None
            try:
                async with session.get(
                    f"{self.publisher_url}/job/{job_id}",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
            except Exception:
                continue
            status = data.get("status")
            if status == "success":
                return data.get("result") or {}
            if status == "failure":
                logger.error("job %s (%s) failed: %s", job_id, path, data.get("error"))
                return None
            if time.time() >= deadline:
                logger.error(
                    "job %s (%s) still %s after %.0fs; giving up",
                    job_id,
                    path,
                    status,
                    self.publish_timeout_seconds,
                )
                return None

    async def _pointer_at_least(self, session: aiohttp.ClientSession, version: int) -> bool:
        """True when /status's store pointer has advanced to `version`."""
        try:
            async with session.get(
                f"{self.publisher_url}/status",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    logger.error("/status got status %d", resp.status)
                    return False
                data = await resp.json()
        except Exception as exc:
            logger.error("/status error: %r", exc)
            return False
        latest = data.get("latest_version")
        return latest is not None and latest >= version

    async def fabricate_and_publish(self, version: int) -> bool:
        """Fabricate ``version`` then publish it. True only if the job succeeds
        and ``/status`` shows the pointer at ``version``. ``--delta`` fabricates
        a XOR delta on the current pin (version-1); otherwise copies the base.
        """
        async with aiohttp.ClientSession() as session:
            if self.delta:
                fab = await self._post_and_poll_job(
                    session,
                    "/fabricate_delta",
                    {
                        "run_id": self.run_id,
                        "base_version": version - 1,
                        "new_version": version,
                    },
                )
            else:
                fab = await self._post_and_poll_job(
                    session,
                    "/fabricate",
                    {
                        "run_id": self.run_id,
                        "from_version": self.base_version,
                        "new_version": version,
                    },
                )
            if fab is None:
                return False
            model_dir = fab.get("path")
            if not model_dir:
                logger.error("fabricate v%d result missing path", version)
                return False

            pub = await self._post_and_poll_job(
                session,
                "/publish",
                {
                    "run_id": self.run_id,
                    "version": version,
                    "source": model_dir,
                },
                # /publish 409s with {"status": "retryable"} while the fleet is
                # mixed; retry the POST until the publish timeout. Fabricate
                # calls above are unaffected.
                retry_retryable_409=True,
            )
            if pub is None:
                return False

            # The boundary is real only when the pointer actually moved.
            if not await self._pointer_at_least(session, version):
                logger.error(
                    "publish v%d job succeeded but the store pointer has not advanced",
                    version,
                )
                return False

        logger.info("fabricated and published version %d", version)
        return True

    async def _send_request(
        self, session: aiohttp.ClientSession, session_id: str, pinned_version: int
    ) -> tuple[int, int | None, int | None]:
        """Send one pinned request; return (status, weight_version_start, weight_version_end)."""
        headers = {
            "modal-session-id": session_id,
            "stitch-exact-version": str(pinned_version),
        }
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 10,
            "weight_version": {"exact_version": pinned_version},
        }
        try:
            async with session.post(
                f"{self.router_url}/v1/chat/completions",
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.request_timeout),
            ) as resp:
                status_code = resp.status
                data = await resp.json() if status_code < 400 else {}
                return (
                    status_code,
                    data.get("weight_version_start"),
                    data.get("weight_version_end"),
                )
        except asyncio.TimeoutError:
            return 504, None, None
        except Exception as exc:
            logger.debug("request failed: %r", exc)
            return 500, None, None

    async def eval_session(
        self,
        eval_num: int,
        session_id: str,
        pinned_version: int,
        straggler_deadline: float,
    ) -> None:
        """Pin one session to ``pinned_version`` until ``straggler_deadline``.
        Every attempt is recorded, including 409s (``is_retry`` after the first).

        409s are retried with jittered capped exponential backoff while the
        session's cumulative 409 budget (wall-clock) lasts; when the budget is
        exhausted the 409 stands as the request's final status. Each 409 streak
        is recorded as a conflict gap: resolved when a non-409 response arrives
        (service resumed), unresolved when the budget ran out first.
        """
        async with aiohttp.ClientSession() as session:
            request_num = 0
            conflict_started_at: float | None = None
            conflict_attempts = 0
            conflict_budget_used = 0.0

            while True:
                now = time.time()
                if now > straggler_deadline:
                    break

                request_num += 1
                attempt = 0
                while True:
                    request_start = time.time()
                    status_code, weight_version_start, weight_version_end = (
                        await self._send_request(session, session_id, pinned_version)
                    )
                    latency_ms = (time.time() - request_start) * 1000

                    metric = RequestMetrics(
                        timestamp=request_start,
                        session_id=session_id,
                        eval_num=eval_num,
                        pinned_version=pinned_version,
                        request_num=request_num,
                        status_code=status_code,
                        latency_ms=latency_ms,
                        weight_version_start=weight_version_start,
                        weight_version_end=weight_version_end,
                        is_retry=attempt > 0,
                    )
                    self.metrics.append(metric)
                    self._emit_record(asdict(metric))

                    if status_code == 409:
                        now = time.time()
                        if conflict_started_at is None:
                            conflict_started_at = request_start
                        conflict_attempts += 1
                        streak_elapsed = now - conflict_started_at
                        remaining_budget = (
                            self.session_409_budget_seconds
                            - conflict_budget_used
                            - streak_elapsed
                        )
                        if remaining_budget > 0:
                            attempt += 1
                            await asyncio.sleep(
                                min(
                                    session_409_backoff_seconds(attempt),
                                    remaining_budget,
                                )
                            )
                            continue
                        # Budget exhausted: the 409 is the request's final status.
                        conflict_budget_used += streak_elapsed
                        self._record_conflict_gap(
                            session_id=session_id,
                            eval_num=eval_num,
                            started_at=conflict_started_at,
                            ended_at=now,
                            attempts=conflict_attempts,
                            resolved=False,
                        )
                        conflict_started_at = None
                        conflict_attempts = 0
                        break

                    if conflict_started_at is not None:
                        now = time.time()
                        conflict_budget_used += now - conflict_started_at
                        self._record_conflict_gap(
                            session_id=session_id,
                            eval_num=eval_num,
                            started_at=conflict_started_at,
                            ended_at=now,
                            attempts=conflict_attempts,
                            resolved=True,
                        )
                        conflict_started_at = None
                        conflict_attempts = 0
                    break

                await asyncio.sleep(self.think_seconds)

    def _record_conflict_gap(
        self,
        session_id: str,
        eval_num: int,
        started_at: float,
        ended_at: float,
        attempts: int,
        resolved: bool,
    ) -> None:
        """Record one session 409 streak (in-memory + JSONL conflict_gap record)."""
        gap = {
            "session_id": session_id,
            "eval_num": eval_num,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": ended_at - started_at,
            "attempts": attempts,
            "resolved": resolved,
        }
        self.conflict_gaps.append(gap)
        self._emit_record({"conflict_gap": gap, "timestamp": ended_at})
        outcome = "resolved" if resolved else "UNRESOLVED (409 budget exhausted)"
        self.print_event(
            f"session {session_id[:8]} 409 gap {outcome} after "
            f"{gap['duration_s']:.1f}s ({attempts} attempts)"
        )

    def _start_sessions(
        self, eval_num: int, pinned_version: int
    ) -> tuple[list[asyncio.Task], list[asyncio.Task], float]:
        """Start this eval's sessions; return (main_phase_tasks, straggler_tasks,
        main_phase_deadline)."""
        now = time.time()
        main_deadline = now + self.eval_duration
        straggler_deadline = main_deadline + self.straggler_duration

        self.print_event(f"eval {eval_num} start (version {pinned_version})")

        main_tasks: list[asyncio.Task] = []
        straggler_tasks: list[asyncio.Task] = []
        for i in range(self.num_sessions):
            session_id = str(uuid.uuid4())
            is_straggler = i < self.num_stragglers
            if is_straggler:
                session_straggler_deadline = straggler_deadline
            else:
                session_straggler_deadline = main_deadline
            task = asyncio.create_task(
                self.eval_session(
                    eval_num=eval_num,
                    session_id=session_id,
                    pinned_version=pinned_version,
                    straggler_deadline=session_straggler_deadline,
                )
            )
            (straggler_tasks if is_straggler else main_tasks).append(task)
        return main_tasks, straggler_tasks, main_deadline

    async def run(self) -> None:
        """Run all evals; each eval's stragglers overlap the next eval's sessions.

        The results JSONL is opened (truncated) up front and every record is
        flushed as it is produced, so an abort leaves a valid file with all
        records so far; the finally block always appends the boundary-rollout
        records and a terminal summary carrying an ``aborted`` flag.
        """
        pending_stragglers: list[asyncio.Task] = []
        self._open_results_file()
        fleet_task: asyncio.Task | None = (
            asyncio.create_task(self.poll_fleet()) if self.registry_url else None
        )
        try:
            for eval_num in range(1, self.num_evals + 1):
                pinned_version = self.base_version + eval_num - 1

                if eval_num > 1:
                    ok = await self.fabricate_and_publish(pinned_version)
                    if not ok:
                        raise RuntimeError(
                            f"fabricate/publish of version {pinned_version} failed; "
                            f"aborting run before eval {eval_num}"
                        )
                    self.publish_accepted_at[eval_num] = time.time()
                    self.print_event(f"checkpoint {pinned_version} published")

                main_tasks, straggler_tasks, main_deadline = self._start_sessions(
                    eval_num, pinned_version
                )
                pending_stragglers.extend(straggler_tasks)
                await asyncio.gather(*main_tasks)
                # Main-phase boundary is time-based: gather(main_tasks) returns
                # instantly when every session is a straggler (main_tasks empty).
                remaining = main_deadline - time.time()
                if remaining > 0:
                    await asyncio.sleep(remaining)
                self.print_event(f"eval {eval_num} main phase end")

            if pending_stragglers:
                self.print_event("waiting for final stragglers")
                await asyncio.gather(*pending_stragglers)

            self._eval_loop_completed = True
            self.verify_assertions()

        except Exception as exc:
            logger.error("eval run failed: %r", exc, exc_info=True)
            raise
        finally:
            # Finalize metrics even on an abort; each step is guarded so a
            # mid-run failure can't crash the finally and lose the summary.
            try:
                self.compute_stats()
            except Exception:
                logger.exception("compute_stats failed while finalizing results")
            try:
                self.compute_boundary_rollout()
            except Exception:
                logger.exception(
                    "compute_boundary_rollout failed while finalizing results"
                )
            self.write_results()
            if fleet_task is not None:
                fleet_task.cancel()
                await asyncio.gather(fleet_task, return_exceptions=True)

    async def poll_fleet(self) -> None:
        """Poll the registry's /loads and append fleet_snapshot records.

        Snapshots carry the per-replica applied_version/leases timeline; they are
        stored separately from request metrics and merged into the results JSONL
        at write time under the 'fleet_snapshot' record type.
        """
        assert self.registry_url is not None
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(
                        f"{self.registry_url}/loads",
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            self._record_fleet_snapshot(await resp.json())
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
                await asyncio.sleep(self.fleet_poll_interval)

    def _record_fleet_snapshot(self, loads: list[dict[str, Any]]) -> None:
        """Append a fleet_snapshot record, plus a drain_event the first time each
        container entry shows draining truthy (one record per task_id, ever)."""
        now = time.time()
        self.fleet_snapshots.append({"fleet_snapshot": loads, "timestamp": now})
        self._emit_record(self.fleet_snapshots[-1])
        for replica in loads:
            task_id = replica.get("task_id")
            if (
                replica.get("draining")
                and task_id is not None
                and task_id not in self._drained_task_ids
            ):
                self._drained_task_ids.add(task_id)
                self.fleet_snapshots.append(
                    {
                        "drain_event": {
                            "task_id": task_id,
                            "applied_version": replica.get("applied_version"),
                            "target_version": replica.get("target_version"),
                            "sync_state": replica.get("sync_state"),
                        },
                        "timestamp": now,
                    }
                )
                self._emit_record(self.fleet_snapshots[-1])
                self.print_event(
                    f"container {task_id} draining "
                    f"(applied={replica.get('applied_version')} "
                    f"target={replica.get('target_version')} "
                    f"sync_state={replica.get('sync_state')})"
                )

    def compute_boundary_rollout(self) -> None:
        """Per boundary, the earliest fleet snapshot showing the new version applied.

        Recorded, NOT asserted: when the overlap assertion fails this distinguishes
        'fleet never rolled' from 'rolled but no traffic routed'. Snapshots are
        appended in time order, so the first match is the earliest.
        """
        for next_eval in range(2, self.num_evals + 1):
            new_version = self.base_version + next_eval - 1
            first_applied_at: float | None = None
            for snapshot in self.fleet_snapshots:
                if "fleet_snapshot" not in snapshot:
                    continue
                if any(
                    replica.get("applied_version") == new_version
                    for replica in snapshot["fleet_snapshot"]
                ):
                    first_applied_at = snapshot["timestamp"]
                    break
            published_at = self.publish_accepted_at.get(next_eval)
            first_new_version_200_at: float | None = None
            if published_at is not None:
                for m in self.metrics:
                    if (
                        m.status_code == 200
                        and m.weight_version_start == new_version
                        and m.timestamp >= published_at
                    ):
                        first_new_version_200_at = m.timestamp
                        break
            self.boundary_rollout[new_version] = {
                "old_version": new_version - 1,
                "new_version": new_version,
                "publish_accepted_at": published_at,
                "fleet_first_applied_at": first_applied_at,
                "first_new_version_200_at": first_new_version_200_at,
            }

    def compute_stats(self) -> None:
        per_eval: dict[int, dict[str, Any]] = {}
        per_version: dict[int, int] = {}

        for metric in self.metrics:
            if metric.eval_num not in per_eval:
                per_eval[metric.eval_num] = {
                    "request_count": 0,
                    "success_count": 0,
                    "conflict_count": 0,
                    "retry_count": 0,
                    "versions_served": set(),
                }
            per_eval[metric.eval_num]["request_count"] += 1
            if metric.status_code < 400:
                per_eval[metric.eval_num]["success_count"] += 1
            if metric.status_code == 409:
                per_eval[metric.eval_num]["conflict_count"] += 1
            if metric.is_retry:
                per_eval[metric.eval_num]["retry_count"] += 1
            if metric.weight_version_start is not None:
                per_eval[metric.eval_num]["versions_served"].add(
                    metric.weight_version_start
                )
            if metric.weight_version_end is not None:
                per_eval[metric.eval_num]["versions_served"].add(
                    metric.weight_version_end
                )

            if metric.weight_version_start is not None:
                per_version[metric.weight_version_start] = (
                    per_version.get(metric.weight_version_start, 0) + 1
                )

        for stats in per_eval.values():
            stats["versions_served"] = sorted(stats["versions_served"])

        self.eval_stats = per_eval
        self.version_stats = per_version
        logger.info("per-eval stats: %s", per_eval)
        logger.info("per-version stats: %s", per_version)

    def verify_assertions(self) -> None:
        """Verify pinning held on every 200 and both versions served at each boundary.

        Raises AssertionError with full details if any check fails.
        """
        failures: list[str] = []

        # 1) Every 200 response must carry BOTH version stamps, equal to the pin.
        #    A None stamp on a 200 is a failure, not a skip.
        for m in self.metrics:
            if m.status_code != 200:
                continue
            if m.weight_version_start is None or m.weight_version_end is None:
                failures.append(
                    f"200 response missing version stamp: eval={m.eval_num} "
                    f"session={m.session_id[:8]} req={m.request_num} "
                    f"pinned={m.pinned_version} start={m.weight_version_start} "
                    f"end={m.weight_version_end}"
                )
            elif (
                m.weight_version_start != m.pinned_version
                or m.weight_version_end != m.pinned_version
            ):
                failures.append(
                    f"pin violated: eval={m.eval_num} session={m.session_id[:8]} "
                    f"req={m.request_num} pinned={m.pinned_version} "
                    f"served start={m.weight_version_start} end={m.weight_version_end}"
                )

        # Overlap window: >=1 success at N_k and >=1 at N_{k+1}.
        for next_eval in range(2, self.num_evals + 1):
            old_version = self.base_version + next_eval - 2
            new_version = self.base_version + next_eval - 1
            t0 = self.publish_accepted_at.get(next_eval)
            if t0 is None:
                failures.append(
                    f"boundary v{old_version}->v{new_version}: no accepted publish "
                    f"timestamp recorded for v{new_version}"
                )
                continue
            t1 = t0 + self.straggler_duration
            window_successes = [
                m
                for m in self.metrics
                if m.status_code == 200 and t0 <= m.timestamp <= t1
            ]
            if not any(
                m.weight_version_start == old_version for m in window_successes
            ):
                failures.append(
                    f"boundary v{old_version}->v{new_version}: no 200 served at "
                    f"v{old_version} in overlap window [{t0:.3f}, {t1:.3f}] "
                    f"({self.straggler_duration:.1f}s straggler window)"
                )
            if not any(
                m.weight_version_start == new_version for m in window_successes
            ):
                failures.append(
                    f"boundary v{old_version}->v{new_version}: no 200 served at "
                    f"v{new_version} in overlap window [{t0:.3f}, {t1:.3f}] "
                    f"({self.straggler_duration:.1f}s straggler window)"
                )

        if failures:
            details = "\n".join(f"  - {f}" for f in failures)
            print(f"\n[SUMMARY] ASSERTIONS FAILED ({len(failures)}):\n{details}")
            raise AssertionError(f"eval assertions FAILED:\n{details}")

        logger.info(
            "ASSERTIONS PASSED: every 200 served its pinned version (start and end), "
            "and every boundary had both versions serving concurrently"
        )

    def write_results(self) -> None:
        """Emit the boundary-rollout and terminal summary records, then print the
        stdout summary.

        Request metrics, fleet snapshots/drain events, and conflict gaps are
        already written incrementally as they are produced (see run(),
        eval_session, _record_fleet_snapshot), so records appear in arrival
        order rather than sorted by timestamp — incremental writes are
        naturally chronological. When called without a preceding run() (e.g.
        tests), the in-memory records are flushed first so the file is complete.
        """
        if self._results_finalized:
            return
        self._results_finalized = True
        if self._results_file is None:
            self._open_results_file()
            for metric in self.metrics:
                self._emit_record(asdict(metric))
            for snapshot in self.fleet_snapshots:
                self._emit_record(snapshot)
            for gap in self.conflict_gaps:
                self._emit_record({"conflict_gap": gap, "timestamp": gap["ended_at"]})
        for info in self.boundary_rollout.values():
            self._emit_record(
                {
                    "boundary_rollout": info,
                    "timestamp": info["publish_accepted_at"] or self.start_time,
                }
            )
        self._emit_record(
            {
                "summary": {
                    "eval_stats": self.eval_stats,
                    "version_stats": self.version_stats,
                    "conflict_gaps": self.conflict_gaps,
                    "aborted": not self._eval_loop_completed,
                },
                "timestamp": time.time(),
            }
        )
        self._close_results_file()

        logger.info("results written to %s", self.results_path)
        print(f"\n[SUMMARY] results in {self.results_path}")
        print(f"  Total requests: {len(self.metrics)}")
        print(f"  Per-eval: {self.eval_stats}")
        print(f"  Per-version: {self.version_stats}")
        resolved_gaps = [gap for gap in self.conflict_gaps if gap["resolved"]]
        unresolved_gaps = [gap for gap in self.conflict_gaps if not gap["resolved"]]
        # Resolved gaps are service interruptions the sessions rode out, NOT
        # failures; unresolved gaps (409 budget exhausted) are called out apart.
        if resolved_gaps:
            print(f"  GAPS (409 service interruptions, resolved): {len(resolved_gaps)}")
            for gap in resolved_gaps:
                print(
                    f"    session {gap['session_id'][:8]} eval {gap['eval_num']}: "
                    f"{gap['duration_s']:.1f}s, {gap['attempts']} attempts"
                )
        if unresolved_gaps:
            print(f"  UNRESOLVED 409 gaps (budget exhausted): {len(unresolved_gaps)}")
            for gap in unresolved_gaps:
                print(
                    f"    session {gap['session_id'][:8]} eval {gap['eval_num']}: "
                    f"{gap['duration_s']:.1f}s, {gap['attempts']} attempts, "
                    f"still 409 at budget end"
                )
        for new_version, info in sorted(self.boundary_rollout.items()):
            old_version = info["old_version"]
            applied_at = info["fleet_first_applied_at"]
            published_at = info["publish_accepted_at"]
            if applied_at is None:
                print(
                    f"  boundary v{old_version}->v{new_version}: fleet never showed "
                    f"v{new_version} applied (no fleet snapshot)"
                )
            else:
                delay = (
                    f"{applied_at - published_at:.1f}s after publish accept"
                    if published_at is not None
                    else "publish accept time unknown"
                )
                print(
                    f"  boundary v{old_version}->v{new_version}: v{new_version} "
                    f"first seen applied in fleet at t+{applied_at - self.start_time:.1f}s "
                    f"({delay})"
                )
            first_200_at = info["first_new_version_200_at"]
            if first_200_at is None:
                print(
                    f"  boundary v{old_version}->v{new_version}: no 200 served at "
                    f"v{new_version} after publish accept"
                )
            else:
                first_200_delay = (
                    f"{first_200_at - published_at:.1f}s after publish accept"
                    if published_at is not None
                    else "publish accept time unknown"
                )
                print(
                    f"  boundary v{old_version}->v{new_version}: first v{new_version} "
                    f"200 at t+{first_200_at - self.start_time:.1f}s ({first_200_delay})"
                )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Version-pinned eval client")
    parser.add_argument("--router-url", required=True, help="Router endpoint URL")
    parser.add_argument("--publisher-url", required=True, help="Publisher endpoint URL")
    parser.add_argument(
        "--registry-url",
        default=None,
        help="Router-registry endpoint URL; if set, poll /loads every 10s and "
        "record fleet_snapshot records (applied_version/leases timeline)",
    )
    parser.add_argument("--run-id", required=True, help="Stitch run ID")
    parser.add_argument("--evals", type=int, default=3, help="Number of evals")
    parser.add_argument(
        "--eval-minutes", type=float, default=20, help="Minutes per eval"
    )
    parser.add_argument(
        "--straggler-minutes", type=float, default=5, help="Extra straggler minutes"
    )
    parser.add_argument(
        "--sessions", type=int, default=6, help="Concurrent sessions per eval"
    )
    parser.add_argument(
        "--stragglers", type=int, default=2, help="Sessions that persist to straggler phase"
    )
    parser.add_argument(
        "--think-seconds", type=float, default=3, help="Sleep between requests per session"
    )
    parser.add_argument("--base-version", type=int, default=1, help="Base version number")
    parser.add_argument(
        "--delta",
        action="store_true",
        help="Boundaries call /fabricate_delta (base = current pin) instead of /fabricate",
    )
    parser.add_argument(
        "--results", type=Path, default=Path("eval_results.jsonl"), help="Results file"
    )
    parser.add_argument(
        "--publish-timeout-seconds",
        type=float,
        default=PUBLISH_TIMEOUT_SECONDS,
        help="Max seconds to poll a fabricate/publish job before aborting",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=420.0,
        help="Per-request total timeout in seconds; must outlast a paused engine "
        "during a first-delta cpu stage",
    )
    parser.add_argument(
        "--session-409-budget-seconds",
        type=float,
        default=None,
        help="Per-session cumulative wall-clock budget for retrying 409s with "
        "backoff; defaults to the straggler window",
    )

    args = parser.parse_args()

    client = EvalClient(
        router_url=args.router_url,
        publisher_url=args.publisher_url,
        registry_url=args.registry_url,
        run_id=args.run_id,
        num_evals=args.evals,
        eval_minutes=args.eval_minutes,
        straggler_minutes=args.straggler_minutes,
        num_sessions=args.sessions,
        num_stragglers=args.stragglers,
        think_seconds=args.think_seconds,
        base_version=args.base_version,
        delta=args.delta,
        results_path=args.results,
        publish_timeout_seconds=args.publish_timeout_seconds,
        request_timeout=args.request_timeout,
        session_409_budget_seconds=args.session_409_budget_seconds,
    )

    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
