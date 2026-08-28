"""Eval client: concurrent version-pinned sessions against the router.

The pin contract: one ``pinned_version`` per eval drives both the
``weight_version.exact_version`` request body field and the
``stitch-exact-version`` header; one ``Modal-Session-Id`` per trajectory keeps
the router's session pin stable for the trajectory's whole life. A trajectory
ends by tombstone (best-effort ``POST /sessions/{id}/end`` on completion,
success or give-up); the router's TTL is only the backstop for clients that
never call it.

Transient responses are ridden out, not failed: 409s retry with jittered
capped exponential backoff and 503s retry after the server's ``Retry-After``,
both under one per-session wall-clock budget. Every 2xx response's version
stamps are checked against the pin; violations are recorded per response.
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
DEFAULT_RETRY_AFTER_SECONDS = 1.0
# Deadline arithmetic leaves sub-nanosecond float dust in the remaining budget
# after a capped sleep; treat anything below a microsecond as exhausted.
BUDGET_EPSILON_SECONDS = 1e-6


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
        num_evals: int,
        eval_minutes: int,
        straggler_minutes: int,
        num_sessions: int,
        num_stragglers: int,
        think_seconds: float,
        base_version: int,
        results_path: Path,
        request_timeout: float = 420.0,
        session_409_budget_seconds: float | None = None,
    ):
        self.router_url = router_url.rstrip("/")
        self.num_evals = num_evals
        self.eval_duration = eval_minutes * 60
        self.straggler_duration = straggler_minutes * 60
        self.num_sessions = num_sessions
        self.num_stragglers = num_stragglers
        self.think_seconds = think_seconds
        self.base_version = base_version
        self.results_path = results_path
        # Straggler requests must survive a paused engine: a first cpu-mode delta
        # stage pauses generation for the full image compile (minutes).
        self.request_timeout = request_timeout
        # Per-session cumulative wall-clock budget for riding out transient
        # 409/503 responses (a rollout's staging/compile gap). Defaults to the
        # straggler window.
        self.session_409_budget_seconds = (
            session_409_budget_seconds
            if session_409_budget_seconds is not None
            else self.straggler_duration
        )

        self.start_time = time.time()
        self.metrics: list[RequestMetrics] = []
        self.conflict_gaps: list[dict[str, Any]] = []
        self.stamp_violations: list[dict[str, Any]] = []
        self.eval_stats: dict[int, dict[str, Any]] = {}
        self.version_stats: dict[int, int] = {}
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

    async def _send_request(
        self, session: aiohttp.ClientSession, session_id: str, pinned_version: int
    ) -> tuple[int, int | None, int | None, float | None]:
        """Send one pinned request; return (status, weight_version_start,
        weight_version_end, retry_after_seconds)."""
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
                retry_after: float | None = None
                if status_code == 503:
                    raw = resp.headers.get("Retry-After")
                    if raw is not None:
                        try:
                            retry_after = float(raw)
                        except ValueError:
                            retry_after = None
                return (
                    status_code,
                    data.get("weight_version_start"),
                    data.get("weight_version_end"),
                    retry_after,
                )
        except asyncio.TimeoutError:
            return 504, None, None, None
        except Exception as exc:
            logger.debug("request failed: %r", exc)
            return 500, None, None, None

    async def _end_session(
        self, session: aiohttp.ClientSession, session_id: str
    ) -> None:
        """Best-effort session tombstone (one retry): tell the router this
        trajectory is done so its pin can be released immediately. The router
        endpoint is idempotent, so any failure here is safe to swallow."""
        for attempt in (1, 2):
            try:
                async with session.post(
                    f"{self.router_url}/sessions/{session_id}/end",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status < 500:
                        return
            except Exception as exc:
                logger.debug("session tombstone failed: %r", exc)
            if attempt == 1:
                await asyncio.sleep(0.5)
        self.print_event(f"session {session_id[:8]} tombstone gave up")

    async def eval_session(
        self,
        eval_num: int,
        session_id: str,
        pinned_version: int,
        straggler_deadline: float,
    ) -> None:
        """Pin one session to ``pinned_version`` until ``straggler_deadline``.
        Every attempt is recorded, including retries (``is_retry`` after the
        first).

        409s are retried with jittered capped exponential backoff and 503s
        after the server's ``Retry-After``, both while the session's cumulative
        transient-error budget (wall-clock) lasts; when the budget is exhausted
        the transient status stands as the request's final status. Each streak
        is recorded as a conflict gap: resolved when a non-transient response
        arrives (service resumed), unresolved when the budget ran out first.

        The session is tombstoned at the router when the trajectory ends, however
        it ends; the router's TTL is only the backstop.
        """
        async with aiohttp.ClientSession() as session:
            try:
                await self._eval_session_requests(
                    session, eval_num, session_id, pinned_version, straggler_deadline
                )
            finally:
                await self._end_session(session, session_id)

    async def _eval_session_requests(
        self,
        session: aiohttp.ClientSession,
        eval_num: int,
        session_id: str,
        pinned_version: int,
        straggler_deadline: float,
    ) -> None:
        request_num = 0
        conflict_started_at: float | None = None
        conflict_attempts = 0
        conflict_budget_used = 0.0
        conflict_status = 0

        while True:
            now = time.time()
            if now > straggler_deadline:
                break

            request_num += 1
            attempt = 0
            while True:
                request_start = time.time()
                (
                    status_code,
                    weight_version_start,
                    weight_version_end,
                    retry_after,
                ) = await self._send_request(session, session_id, pinned_version)
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
                if status_code == 200:
                    self._check_stamps(metric, session_id=session_id, eval_num=eval_num)

                if status_code in (409, 503):
                    now = time.time()
                    if conflict_started_at is None:
                        conflict_started_at = request_start
                        conflict_status = status_code
                    conflict_attempts += 1
                    streak_elapsed = now - conflict_started_at
                    remaining_budget = (
                        self.session_409_budget_seconds
                        - conflict_budget_used
                        - streak_elapsed
                    )
                    if remaining_budget > BUDGET_EPSILON_SECONDS:
                        attempt += 1
                        delay = (
                            session_409_backoff_seconds(attempt)
                            if status_code == 409
                            else (retry_after or DEFAULT_RETRY_AFTER_SECONDS)
                        )
                        await asyncio.sleep(min(delay, remaining_budget))
                        continue
                    # Budget exhausted: the status is the request's final status.
                    conflict_budget_used += streak_elapsed
                    self._record_conflict_gap(
                        session_id=session_id,
                        eval_num=eval_num,
                        started_at=conflict_started_at,
                        ended_at=now,
                        attempts=conflict_attempts,
                        resolved=False,
                        status=conflict_status,
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
                        status=conflict_status,
                    )
                    conflict_started_at = None
                    conflict_attempts = 0
                break

            await asyncio.sleep(self.think_seconds)

    def _check_stamps(
        self, metric: RequestMetrics, *, session_id: str, eval_num: int
    ) -> None:
        """Per-response stamp assertion: a 200 must carry both version stamps,
        both equal to the pin. Violations are recorded, never retried away."""
        if (
            metric.weight_version_start == metric.pinned_version
            and metric.weight_version_end == metric.pinned_version
        ):
            return
        violation = {
            "session_id": session_id,
            "eval_num": eval_num,
            "request_num": metric.request_num,
            "pinned_version": metric.pinned_version,
            "weight_version_start": metric.weight_version_start,
            "weight_version_end": metric.weight_version_end,
            "timestamp": metric.timestamp,
        }
        self.stamp_violations.append(violation)
        self._emit_record({"stamp_violation": violation, "timestamp": metric.timestamp})
        self.print_event(
            f"session {session_id[:8]} STAMP VIOLATION: pinned "
            f"{metric.pinned_version}, served "
            f"{metric.weight_version_start}->{metric.weight_version_end}"
        )

    def _record_conflict_gap(
        self,
        session_id: str,
        eval_num: int,
        started_at: float,
        ended_at: float,
        attempts: int,
        resolved: bool,
        status: int,
    ) -> None:
        """Record one session transient-error streak (in-memory + JSONL)."""
        gap = {
            "session_id": session_id,
            "eval_num": eval_num,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_s": ended_at - started_at,
            "attempts": attempts,
            "resolved": resolved,
            "status": status,
        }
        self.conflict_gaps.append(gap)
        self._emit_record({"conflict_gap": gap, "timestamp": ended_at})
        outcome = "resolved" if resolved else f"UNRESOLVED ({status} budget exhausted)"
        self.print_event(
            f"session {session_id[:8]} {status} gap {outcome} after "
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
        records so far; the finally block always appends a terminal summary
        carrying an ``aborted`` flag.
        """
        pending_stragglers: list[asyncio.Task] = []
        self._open_results_file()
        try:
            for eval_num in range(1, self.num_evals + 1):
                pinned_version = self.base_version + eval_num - 1

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
            self.write_results()

    def compute_stats(self) -> None:
        per_eval: dict[int, dict[str, Any]] = {}
        per_version: dict[int, int] = {}

        for metric in self.metrics:
            if metric.eval_num not in per_eval:
                per_eval[metric.eval_num] = {
                    "request_count": 0,
                    "success_count": 0,
                    "conflict_count": 0,
                    "unavailable_count": 0,
                    "retry_count": 0,
                    "versions_served": set(),
                }
            per_eval[metric.eval_num]["request_count"] += 1
            if metric.status_code < 400:
                per_eval[metric.eval_num]["success_count"] += 1
            if metric.status_code == 409:
                per_eval[metric.eval_num]["conflict_count"] += 1
            if metric.status_code == 503:
                per_eval[metric.eval_num]["unavailable_count"] += 1
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

    def write_results(self) -> None:
        """Emit the terminal summary record, then print the stdout summary.

        Request metrics and conflict gaps are already written incrementally as
        they are produced (see run() and eval_session), so records appear in
        arrival order rather than sorted by timestamp — incremental writes are
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
            for gap in self.conflict_gaps:
                self._emit_record({"conflict_gap": gap, "timestamp": gap["ended_at"]})
            for violation in self.stamp_violations:
                self._emit_record(
                    {"stamp_violation": violation, "timestamp": violation["timestamp"]}
                )
        self._emit_record(
            {
                "summary": {
                    "eval_stats": self.eval_stats,
                    "version_stats": self.version_stats,
                    "conflict_gaps": self.conflict_gaps,
                    "stamp_violations": self.stamp_violations,
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
        if self.stamp_violations:
            print(
                f"  STAMP VIOLATIONS (200s off the pin): {len(self.stamp_violations)}"
            )
        resolved_gaps = [gap for gap in self.conflict_gaps if gap["resolved"]]
        unresolved_gaps = [gap for gap in self.conflict_gaps if not gap["resolved"]]
        # Resolved gaps are service interruptions the sessions rode out, NOT
        # failures; unresolved gaps (budget exhausted) are called out apart.
        if resolved_gaps:
            print(
                f"  GAPS (transient service interruptions, resolved): "
                f"{len(resolved_gaps)}"
            )
            for gap in resolved_gaps:
                print(
                    f"    session {gap['session_id'][:8]} eval {gap['eval_num']}: "
                    f"{gap['duration_s']:.1f}s, {gap['attempts']} attempts"
                )
        if unresolved_gaps:
            print(f"  UNRESOLVED gaps (budget exhausted): {len(unresolved_gaps)}")
            for gap in unresolved_gaps:
                print(
                    f"    session {gap['session_id'][:8]} eval {gap['eval_num']}: "
                    f"{gap['duration_s']:.1f}s, {gap['attempts']} attempts, "
                    f"still {gap['status']} at budget end"
                )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Version-pinned eval client")
    parser.add_argument("--router-url", required=True, help="Router endpoint URL")
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
        "--stragglers",
        type=int,
        default=2,
        help="Sessions that persist to straggler phase",
    )
    parser.add_argument(
        "--think-seconds",
        type=float,
        default=3,
        help="Sleep between requests per session",
    )
    parser.add_argument(
        "--base-version", type=int, default=1, help="Base version number"
    )
    parser.add_argument(
        "--results", type=Path, default=Path("eval_results.jsonl"), help="Results file"
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
        help="Per-session cumulative wall-clock budget for retrying transient "
        "409/503 responses; defaults to the straggler window",
    )

    args = parser.parse_args()

    client = EvalClient(
        router_url=args.router_url,
        num_evals=args.evals,
        eval_minutes=args.eval_minutes,
        straggler_minutes=args.straggler_minutes,
        num_sessions=args.sessions,
        num_stragglers=args.stragglers,
        think_seconds=args.think_seconds,
        base_version=args.base_version,
        results_path=args.results,
        request_timeout=args.request_timeout,
        session_409_budget_seconds=args.session_409_budget_seconds,
    )

    await client.run()


if __name__ == "__main__":
    asyncio.run(main())
