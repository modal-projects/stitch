"""Auto mode: the engines whose output keeps up with what the trainer takes, each move judged once its effect is in."""

import logging
import math
import threading
from collections import deque
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Defaults, to become run settings: output kept spare over what the trainer takes; how long a change takes to show in
# the output until the rollout reports its own group lifetime; the span output is averaged over, and the least span
# of it an engine's output is measured from.
HEADROOM = 0.1
LAG, WINDOW, MEASURED = 900, 900, 600
# The trainer is short when it waited at least this share of its last steps.
SHORT = 0.05
# A load report counts as a hold from this many seconds of holding off: less is the rollout's own scheduling between
# its waits, which no engine's output shows.
HELD = 1.0
# An add counts from this share of the output it promised; an undone move is not retried while the trainer's demand
# stays within this share of what it was.
SIGNIFICANT = 0.25


class Step(NamedTuple):
    at: float
    seconds: float
    waited: float
    consumed: int


class Minute(NamedTuple):
    at: float
    produced: int
    held: float


class Load(NamedTuple):
    at: float
    filled: float


class Move(NamedTuple):
    kind: str
    before: int
    output: float
    served_at: float | None = None


class Decision(NamedTuple):
    engines: int
    reason: str
    capacity: int | None


class Solver:
    def __init__(self, minimum: int, maximum: int | None, count: int | None):
        """`count`: the engines it sets from, None to only log what it would set."""
        self.minimum, self.maximum, self.count = minimum, maximum, count
        self.steps: deque[Step] = deque(maxlen=3)
        self.minutes: deque[Minute] = deque(maxlen=240)
        self.loads: deque[Load] = deque(maxlen=240)
        self.lag = LAG
        self.serving: int | None = None
        self.serving_since = -math.inf
        # Groups per second one engine puts out, last measured while the rollout was held only by the engines.
        self.per_engine: float | None = None
        self.move: Move | None = None
        self.undone: dict[tuple[str, int], float] = {}
        self.decision: Decision | None = None
        self.lock = threading.Lock()

    def observe_step(self, now: float, seconds: float, waited: float, consumed: int) -> None:
        with self.lock:
            self.steps.append(Step(now, seconds, waited, consumed))

    def observe_load(self, now: float, produced: int, held: float = 0.0, lifetime: float | None = None) -> None:
        """`produced`: groups put since the rollout's start; `held`: seconds held off since; `lifetime`: a group's."""
        with self.lock:
            if self.minutes and produced < self.minutes[-1].produced:
                # The rollout restarted and counts from its new start.
                self.minutes.clear()
            self.minutes.append(Minute(now, produced, held))
            if lifetime:
                self.lag = lifetime

    def decide(self, now: float, serving: int, load: dict | None) -> Decision:
        """`load`: the serving engines' requests running and queued, and their slots, summed."""
        with self.lock:
            if serving != self.serving:
                self.serving, self.serving_since = serving, now
                if self.move is not None:
                    self.move = self.move._replace(served_at=None)
            if load and load["slots"] and serving:
                self.loads.append(Load(now, (load["running"] + load["queued"]) * serving / load["slots"]))
            self._measure(now)
            count = serving if self.count is None else self.count
            if self.count is not None and self.move is not None and serving == count:
                reason = self._follow_move(now, self.move)
                target = self.count
            else:
                target, reason = self._solve(now, serving, count)
            decision = Decision(target, reason, load["slots"] if load else None)
            previous, self.decision = self.decision, decision
            if previous is None or previous[:2] != decision[:2]:
                prefix = "shadow: " if self.count is None else ""
                logger.info("%sengines %d -> %d: %s", prefix, count, target, reason)
            return decision

    def _measure(self, now: float) -> None:
        # Output shows a change in the serving engines, or in the rollout holding off, a group's lifetime later.
        since = max(self.serving_since, self._last_held()) + self.lag
        start = max(since, now - WINDOW)
        if now - start >= MEASURED and (output := self._output(start, now)) is not None and self.serving:
            self.per_engine = output / self.serving

    def _solve(self, now: float, serving: int, count: int) -> tuple[int, str]:
        demand, waited = self._demand(), self._waited(self.serving_since + self.lag)
        if demand is None:
            return count, "no training step yet"
        if not self.per_engine:
            return count, "no output measured from the serving engines yet"
        if waited is None:
            return count, "no training step on the serving engines yet"
        held = self._held(now - self.lag)
        needed = demand * (1 + HEADROOM) / self.per_engine
        reason = (
            f"the trainer waited {waited:.0%} of its last steps and takes {demand:.3g} groups/s, "
            f"an engine puts out {self.per_engine:.3g}/s: {needed:.2f} needed"
        )
        if waited >= SHORT and held:
            return count, f"{reason}; the rollout held off admitting, so more engines would not be used"
        if waited >= SHORT:
            kind, target = "add", math.ceil(needed)
        elif held:
            # Ahead of the trainer the rollout held off: the surplus engines go, but none the requests on them fill.
            kind, target = "shrink", math.ceil(needed)
            reason = f"{reason}; the rollout held off admitting"
            if needed <= count - 1 and (filled := self._filled(now - WINDOW) or 0.0) * (1 + HEADROOM) > target:
                target = math.ceil(filled * (1 + HEADROOM))
                reason = f"{reason}; the requests on the engines fill {filled:.2f}"
        else:
            return count, reason
        if target < self.minimum:
            target, reason = self.minimum, f"{reason}; at the minimum"
        if self.maximum is not None and target > self.maximum:
            target, reason = self.maximum, f"{reason}; at the maximum"
        if serving != count:
            reason = f"{reason}; {serving} of {count} engines serve"
        if kind == "add":
            ready = target > count and serving == count
        else:
            # Only a surplus of at least one engine goes: any less is the band between two counts.
            ready = needed <= count - 1 and target < count and target <= serving
        if not ready:
            return count, reason
        if self.count is None:
            return target, reason
        if (undone := self.undone.get((kind, count))) is not None and _close(demand, undone):
            return count, f"{reason}; the {kind} from {count} was undone"
        output = self._output(max(self.serving_since + self.lag, now - WINDOW), now) or 0.0
        self.count, self.move = target, Move(kind, count, output)
        return target, reason

    def _follow_move(self, now: float, move: Move) -> str:
        if move.served_at is None:
            self.move = move = move._replace(served_at=now)
        if move.kind == "undo":
            self.move = None
            return f"back at {self.count}"
        start = (move.served_at or now) + self.lag
        if now - start < MEASURED:
            return f"checking the {move.kind} from {move.before}"
        self.move = None
        output, demand = self._output(start, now) or 0.0, self._demand() or 0.0
        if move.kind == "add":
            if self._held(start):
                return "the add held: the rollout got ahead and held off admitting"
            promised = (self.count - move.before) * move.output / move.before
            outcome = f"output rose {output - move.output:.3g}/s of {promised:.3g}/s promised"
            return outcome if output - move.output >= SIGNIFICANT * promised else self._undo(move, outcome)
        if self._held(start):
            return "the shrink held: the rollout still held off admitting"
        outcome = f"output {output:.3g}/s against the trainer's {demand:.3g}/s"
        return f"the shrink held: {outcome}" if output >= demand else self._undo(move, outcome)

    def _undo(self, move: Move, outcome: str) -> str:
        self.undone[(move.kind, move.before)] = self._demand() or 0.0
        self.count, self.move = move.before, Move("undo", self.count, 0.0)
        return f"undone: {outcome}"

    def _demand(self) -> float | None:
        """Groups the trainer takes per second it does not wait."""
        training = sum(step.seconds - step.waited for step in self.steps)
        return sum(step.consumed for step in self.steps) / training if training > 0 else None

    def _waited(self, since: float) -> float | None:
        """The share of its last steps begun since `since` the trainer waited for groups."""
        steps = [step for step in self.steps if step.at - step.seconds >= since]
        seconds = sum(step.seconds for step in steps)
        return sum(step.waited for step in steps) / seconds if seconds else None

    def _held(self, since: float) -> bool:
        return any(minute.held >= HELD for minute in self.minutes if minute.at >= since)

    def _last_held(self) -> float:
        return max((minute.at for minute in self.minutes if minute.held >= HELD), default=-math.inf)

    def _filled(self, since: float) -> float | None:
        """The engines the requests on them fill, averaged from `since`."""
        filled = [load.filled for load in self.loads if load.at >= since]
        return sum(filled) / len(filled) if filled else None

    def _output(self, since: float, now: float) -> float | None:
        """Groups put per second from `since`, at most WINDOW back."""
        reports = [minute for minute in self.minutes if minute.at >= max(since, now - WINDOW)]
        if len(reports) < 2:
            return None
        first, last = reports[0], reports[-1]
        return (last.produced - first.produced) / (last.at - first.at)


def _close(value: float, seen: float) -> bool:
    return abs(value - seen) < SIGNIFICANT * seen
