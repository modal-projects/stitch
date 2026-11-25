"""Auto mode's next rule in shadow: the engine count that the trainer and the engines' load call for, logged."""

import logging
import math
import statistics
import threading
from collections import deque
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Defaults, to become run settings once the solver sets the count.
STRESSED_SECONDS, IDLE_SECONDS = 300, 900
PREFIX_KEPT, RETRACTED_SHARE, QUEUED_SHARE, IDLE_RUNNING_SHARE = 0.9, 0.05, 0.5, 0.25


class Step(NamedTuple):
    seconds: float
    waited: float
    consumed: int


class InBand(NamedTuple):
    groups_per_engine_second: float
    slot_seconds_per_group: float
    generating_share: float


class Decision(NamedTuple):
    engines: int
    cell: str
    reason: str
    capacity: int | None
    sessions: int | None
    slot_engines: int | None


class Solver:
    def __init__(self, minimum: int, maximum: int | None):
        self.minimum, self.maximum = minimum, maximum
        self.rollout_loads: deque[tuple[float, dict]] = deque(maxlen=30)
        self.steps: deque[Step] = deque(maxlen=3)
        self.in_band: InBand | None = None
        self.serving: int | None = None
        self.serving_since = -math.inf
        self.decision: Decision | None = None
        self.lock = threading.Lock()

    def observe_rollout_load(self, now: float, load: dict) -> None:
        with self.lock:
            self.rollout_loads.append((now, load))

    def observe_step(self, seconds: float, waited: float, consumed: int) -> None:
        with self.lock:
            self.steps.append(Step(seconds, waited, consumed))

    def decide(
        self,
        now: float,
        engine_loads: dict[str, list[tuple[float, dict]]],
        count: int,
        serving: int,
    ) -> Decision:
        """`count` is the engine count set now, of which `serving` serve."""
        with self.lock:
            slots = _slots(engine_loads)
            rollout_reports = _window(self.rollout_loads, now, STRESSED_SECONDS)
            stressed = _stressed(now, engine_loads, rollout_reports)
            idle = _idle(now, engine_loads, slots)
            if serving != self.serving:
                self.serving, self.serving_since = serving, now
            recent = [
                (at, load)
                for at, load in self.rollout_loads
                if at >= now - STRESSED_SECONDS
            ]
            if (
                rollout_reports
                and self.serving_since <= now - STRESSED_SECONDS
                and not stressed
                and not idle
                and serving
                and len(recent) > 1
            ):
                # Each report counts what finished since the one before, so the first opens the span.
                span = recent[-1][0] - recent[0][0]
                finished = [load for _, load in recent[1:]]
                if groups := _sum(finished, "finished_groups"):
                    turn_seconds = _sum(finished, "turn_seconds")
                    self.in_band = InBand(
                        groups / span / serving,
                        turn_seconds / groups,
                        turn_seconds / _sum(finished, "session_seconds"),
                    )
            starving = self._starving(now)
            consumption = self._consumption()
            needed = slot_engines = None
            if consumption is not None and self.in_band is not None:
                needed = math.ceil(consumption / self.in_band.groups_per_engine_second)
                if slots is not None:
                    slot_engines = math.ceil(
                        consumption * self.in_band.slot_seconds_per_group / slots
                    )
            target, reason = _target(
                starving,
                stressed,
                idle,
                needed=needed,
                load_engines=_engines_for(engine_loads, slots, now, STRESSED_SECONDS),
                peak_engines=_engines_for(engine_loads, slots, now, IDLE_SECONDS),
                count=count,
            )
            if self.maximum is not None:
                target = min(self.maximum, target)
            target = max(self.minimum, target)
            trainer_state = {
                None: "no trainer",
                True: "trainer starving",
                False: "trainer fed",
            }
            engines_state = "stressed" if stressed else "idle" if idle else "in band"
            cell = f"{trainer_state[starving]}, engines {engines_state}"
            capacity = None if slots is None else serving * slots
            sessions = None
            if capacity is not None and self.in_band is not None:
                sessions = math.floor(capacity / self.in_band.generating_share)
            decision = Decision(target, cell, reason, capacity, sessions, slot_engines)
            previous, self.decision = self.decision, decision
            if previous is None or previous[:2] != decision[:2]:
                logger.info(
                    "shadow: engines %d -> %d (%s): %s", count, target, cell, reason
                )
            return decision

    def _starving(self, now: float) -> bool | None:
        """Whether the trainer is starving; None without a trainer."""
        if not self.rollout_loads and not self.steps:
            return None
        at, load = self.rollout_loads[-1] if self.rollout_loads else (now, {})
        waiting = now - at <= 120 and bool(load.get("waiting"))
        return waiting or bool(self.steps and self.steps[-1].waited)

    def _consumption(self) -> float | None:
        """Groups the trainer consumes per second it trains."""
        training = sum(step.seconds - step.waited for step in self.steps)
        if training <= 0:
            return None
        return sum(step.consumed for step in self.steps) / training


def _target(
    starving, stressed, idle, *, needed, load_engines, peak_engines, count
) -> tuple[int, str]:
    if stressed:
        if starving is False:
            return count, f"over-admitted: {stressed}"
        if added := needed or load_engines:
            return max(count, added), stressed
        return count, f"{stressed}; nothing to solve the count from yet"
    if idle:
        if starving:
            return count, "demand-limited: more engines would not feed the trainer"
        return min(count, max(needed or 0, peak_engines)), "idle"
    if starving:
        if needed:
            return max(count, needed), "short of what the trainer uses"
        return count, "starving; no window in band to solve from yet"
    return count, ""


def _window(reports, now: float, seconds: float) -> list[dict] | None:
    if not reports or reports[0][0] > now - seconds:
        return None
    return [load for at, load in reports if at >= now - seconds]


def _sum(reports: list[dict], key: str) -> float:
    return sum(load.get(key, 0) for load in reports)


def _stressed(now: float, engine_loads, rollout_reports: list[dict] | None) -> str:
    if rollout_reports and (prefix := _sum(rollout_reports, "prefix_tokens")):
        kept = 1 - _sum(rollout_reports, "uncached_prefix_tokens") / prefix
        if kept < PREFIX_KEPT:
            return f"prefix kept {kept:.2f}"
    windows = [
        reports
        for loads in engine_loads.values()
        if (reports := _window(loads, now, STRESSED_SECONDS))
    ]
    if not windows:
        return ""
    queued = statistics.median(
        sum(load["queued"] > 0 for load in reports) / len(reports)
        for reports in windows
    )
    if queued >= QUEUED_SHARE:
        return f"a queue on {queued:.0%} of the median engine's reports"
    if not rollout_reports or not _sum(rollout_reports, "prefix_tokens"):
        prompt = sum(r[-1]["prompt_tokens"] - r[0]["prompt_tokens"] for r in windows)
        retracted = sum(
            r[-1]["retracted_tokens"] - r[0]["retracted_tokens"] for r in windows
        )
        if prompt and retracted / prompt > RETRACTED_SHARE:
            return f"retraction recomputes {retracted / prompt:.0%} of prefill"
    return ""


def _idle(now: float, engine_loads, slots: int | None) -> bool:
    windows = [_window(loads, now, IDLE_SECONDS) for loads in engine_loads.values()]
    if slots is None or not windows or None in windows:
        return False
    return all(
        load["running"] < IDLE_RUNNING_SHARE * slots and not load["queued"]
        for reports in windows
        for load in reports
    )


def _slots(engine_loads) -> int | None:
    """What an engine ran while requests queued (SGLang may cap it below the run's slots); before a queue, those."""
    loads = [load for reports in engine_loads.values() for _, load in reports]
    if ran := max(
        (round(load["running"]) for load in loads if load["queued"]), default=None
    ):
        return ran
    return next(
        (load["running_slots"] for load in loads if load.get("running_slots")), None
    )


def _engines_for(
    engine_loads, slots: int | None, now: float, seconds: float
) -> int | None:
    peaks = [
        max(
            load["running"] + load["queued"]
            for at, load in reports
            if at >= now - seconds
        )
        for reports in engine_loads.values()
        if reports and reports[-1][0] >= now - seconds
    ]
    if slots is None or not peaks:
        return None
    return math.ceil(sum(peaks) / slots)
