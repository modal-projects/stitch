"""Auto mode: the engine count that the trainer and the engines' load call for, each move checked before the next."""

import logging
import math
import statistics
import threading
from collections import deque
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Defaults, to become run settings.
STRESSED_SECONDS, IDLE_SECONDS = 300, 900
PREFIX_KEPT, RETRACTED_SHARE, QUEUED_SHARE, IDLE_RUNNING_SHARE = 0.9, 0.05, 0.5, 0.25
# Production changes count from this share up: of what a move promised, or of the production a refused move saw.
SIGNIFICANT = 0.25


class Step(NamedTuple):
    seconds: float
    waited: float
    held: float
    consumed: int
    produced: int


class InBand(NamedTuple):
    groups_per_engine_second: float
    slot_seconds_per_group: float | None
    generating_share: float | None


class Move(NamedTuple):
    kind: str
    before: int
    production: float
    promised: float
    served_at: float | None = None


class Decision(NamedTuple):
    engines: int
    cell: str
    reason: str
    capacity: int | None
    sessions: int | None
    slot_engines: int | None


class Solver:
    def __init__(self, minimum: int, maximum: int | None, count: int | None):
        """`count`: the engines it sets from, None to only log what it would set."""
        self.minimum, self.maximum, self.count = minimum, maximum, count
        self.rollout_loads: deque[tuple[float, dict]] = deque(maxlen=30)
        self.steps: deque[Step] = deque(maxlen=3)
        self.in_band: InBand | None = None
        self.serving: int | None = None
        self.serving_since = -math.inf
        self.move: Move | None = None
        self.rollout_limited: tuple[int, float] | None = None
        self.stress_stuck: int | None = None
        self.refused_shrink: tuple[int, float] | None = None
        self.decision: Decision | None = None
        self.lock = threading.Lock()

    def observe_rollout_load(self, now: float, load: dict) -> None:
        with self.lock:
            self.rollout_loads.append((now, load))

    def observe_step(
        self, seconds: float, waited: float, held: float, consumed: int, produced: int
    ) -> None:
        with self.lock:
            self.steps.append(Step(seconds, waited, held, consumed, produced))

    def decide(
        self,
        now: float,
        engine_loads: dict[str, list[tuple[float, dict]]],
        serving: int,
    ) -> Decision:
        with self.lock:
            count = serving if self.count is None else self.count
            slots = _slots(engine_loads)
            rollout_reports = _window(self.rollout_loads, now, STRESSED_SECONDS)
            stressed = _stressed(now, engine_loads, rollout_reports)
            idle = _idle(now, engine_loads, slots)
            if not stressed:
                self.stress_stuck = None
            if serving != self.serving:
                self.serving, self.serving_since = serving, now
            recent = [
                (at, load)
                for at, load in self.rollout_loads
                if at >= now - STRESSED_SECONDS
            ]
            steady = (
                self.serving_since <= now - STRESSED_SECONDS
                and not stressed
                and not idle
                and serving
            )
            producing = sum(step.seconds - step.held for step in self.steps)
            if steady and rollout_reports and len(recent) > 1:
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
            elif (
                steady
                and not self.rollout_loads
                and producing > 0
                and self.serving_since <= now - sum(step.seconds for step in self.steps)
            ):
                produced = sum(step.produced for step in self.steps)
                self.in_band = InBand(produced / producing / serving, None, None)
            starving = self._starving(now)
            production = self._production(recent)
            consumption = self._consumption()
            needed = slot_engines = None
            if consumption is not None and self.in_band is not None:
                needed = math.ceil(consumption / self.in_band.groups_per_engine_second)
                if (
                    slots is not None
                    and self.in_band.slot_seconds_per_group is not None
                ):
                    slot_engines = math.ceil(
                        consumption * self.in_band.slot_seconds_per_group / slots
                    )
            if self.count is not None and serving != count:
                target, reason = count, f"{serving} of {count} engines serve"
                if self.move is not None:
                    self.move = self.move._replace(served_at=None)
            elif self.count is not None and self.move is not None:
                reason = self._follow_move(now, stressed or starving, production)
                target = self.count
            else:
                target, reason = _target(
                    starving,
                    stressed,
                    idle,
                    needed=needed,
                    load_engines=_engines_for(
                        engine_loads, slots, now, STRESSED_SECONDS
                    ),
                    peak_engines=_engines_for(engine_loads, slots, now, IDLE_SECONDS),
                    count=count,
                )
                if self.maximum is not None and target > self.maximum:
                    target, reason = self.maximum, f"{reason}; at the maximum"
                target = max(self.minimum, target)
                if self.count is not None and target != count:
                    target, reason = self._start_move(
                        count, target, reason, "stress" if stressed else "", production
                    )
            trainer_state = {
                None: "no trainer",
                True: "trainer starving",
                False: "trainer fed",
            }
            engines_state = "stressed" if stressed else "idle" if idle else "in band"
            cell = f"{trainer_state[starving]}, engines {engines_state}"
            capacity = None if slots is None else serving * slots
            sessions = None
            if capacity is not None and self.in_band and self.in_band.generating_share:
                sessions = math.floor(capacity / self.in_band.generating_share)
            decision = Decision(target, cell, reason, capacity, sessions, slot_engines)
            previous, self.decision = self.decision, decision
            if previous is None or previous[:3] != decision[:3]:
                logger.info(
                    "%sengines %d -> %d (%s): %s",
                    "shadow: " if self.count is None else "",
                    count,
                    target,
                    cell,
                    reason,
                )
            return decision

    def _start_move(
        self, count: int, target: int, reason: str, stress: str, production: float
    ) -> tuple[int, str]:
        if target < count:
            kind = "shrink"
            refused = self.refused_shrink
            if refused and target <= refused[0] and _close(production, refused[1]):
                return (
                    count,
                    f"{reason}; shrinking to {refused[0]} stressed or starved it before",
                )
        elif stress:
            kind = "stress"
            if self.stress_stuck == count:
                return count, f"{reason}; adds for stress stopped at {count}"
        else:
            kind = "add"
            limited = self.rollout_limited
            if limited and limited[0] == count and _close(production, limited[1]):
                return count, f"{reason}; rollout-limited at {count}"
        per_engine = self.in_band.groups_per_engine_second if self.in_band else 0.0
        self.count, self.move = (
            target,
            Move(kind, count, production, (target - count) * per_engine),
        )
        return target, reason

    def _follow_move(self, now: float, worse: str | bool, production: float) -> str:
        move = self.move
        if move.served_at is None:
            self.move = move = move._replace(served_at=now)
        if move.kind == "undo":
            self.move = None
            return f"back at {self.count}"
        if now - move.served_at < STRESSED_SECONDS:
            return f"checking the move from {move.before}"
        self.move = None
        if move.kind == "shrink":
            if worse:
                self.refused_shrink = (self.count, move.production)
                return self._undo(
                    move, "the shrink stressed the engines or starved the trainer"
                )
            return "the shrink held"
        gain = production - move.production
        if gain >= SIGNIFICANT * move.promised:
            return (
                f"production rose {gain:.3g}/s against {move.promised:.3g}/s promised"
            )
        if move.kind == "stress":
            self.stress_stuck = self.count
            return f"production rose {gain:.3g}/s against {move.promised:.3g}/s: adds for stress stop here"
        self.rollout_limited = (move.before, move.production)
        return self._undo(
            move,
            f"production rose {gain:.3g}/s against {move.promised:.3g}/s: rollout-limited",
        )

    def _undo(self, move: Move, reason: str) -> str:
        self.count, self.move = move.before, Move("undo", self.count, 0.0, 0.0)
        return f"undone: {reason}"

    def _starving(self, now: float) -> bool | None:
        """Whether each of the trainer's last steps waited or it waited in half the window; None without one."""
        if not self.rollout_loads and not self.steps:
            return None
        steps_waited = len(self.steps) == self.steps.maxlen and all(
            step.waited for step in self.steps
        )
        if reports := _window(self.rollout_loads, now, STRESSED_SECONDS):
            waiting = sum(bool(load.get("waiting")) for load in reports)
            return steps_waited or waiting * 2 >= len(reports)
        return steps_waited

    def _production(self, recent: list[tuple[float, dict]]) -> float:
        if len(recent) > 1:
            return _sum([load for _, load in recent[1:]], "finished_groups") / (
                recent[-1][0] - recent[0][0]
            )
        producing = sum(step.seconds - step.held for step in self.steps)
        return (
            sum(step.produced for step in self.steps) / producing
            if producing > 0
            else 0.0
        )

    def _consumption(self) -> float | None:
        """Groups the trainer consumes per second it trains."""
        training = sum(step.seconds - step.waited for step in self.steps)
        if training <= 0:
            return None
        return sum(step.consumed for step in self.steps) / training


def _close(production: float, seen: float) -> bool:
    return abs(production - seen) < SIGNIFICANT * seen


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
    """What an engine runs at once as SGLang reports it; without that, the most it ran while requests queued."""
    loads = [load for reports in engine_loads.values() for _, load in reports]
    if reported := next(
        (load["running_slots"] for load in loads if load.get("running_slots")), None
    ):
        return reported
    return (
        max((round(load["running"]) for load in loads if load["queued"]), default=0)
        or None
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
