"""Auto mode: the engines whose output keeps up with what the trainer takes, each move checked before the next."""

import logging
import math
import threading
from collections import deque
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Defaults, to become run settings: output kept spare over what the trainer takes; how long after the serving engines
# change their output and the trainer's steps show them (a group's lifetime: groups started on them have to finish);
# the span output is averaged over; and how long an add has to draw requests onto its engines, also the shortest span
# output is measured over.
HEADROOM = 0.1
LAG, WINDOW, CHECK_SECONDS = 900, 900, 300
# The trainer is short when it waited at least this share of its last steps.
SHORT = 0.05
# An add counts when the requests grew at least this share as much as the engines; an undone move is not retried
# while the trainer's demand stays within this share of what it was.
SIGNIFICANT = 0.25


class Step(NamedTuple):
    at: float
    seconds: float
    waited: float
    consumed: int


class Load(NamedTuple):
    at: float
    serving: int
    filled: float


class Move(NamedTuple):
    kind: str
    before: int
    filled: float
    demand: float
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
        self.outputs: deque[tuple[float, int]] = deque(maxlen=WINDOW // 60 + 1)
        self.loads: deque[Load] = deque(maxlen=WINDOW // 60 + 1)
        self.serving: int | None = None
        self.serving_since = -math.inf
        self.move: Move | None = None
        self.undone: dict[tuple[str, int], float] = {}
        self.decision: Decision | None = None
        self.lock = threading.Lock()

    def observe_step(self, now: float, seconds: float, waited: float, consumed: int) -> None:
        with self.lock:
            self.steps.append(Step(now, seconds, waited, consumed))

    def observe_output(self, now: float, produced: int) -> None:
        """`produced`: the groups the rollout put into the trainer's buffer since its start."""
        with self.lock:
            if self.outputs and produced < self.outputs[-1][1]:
                # The rollout restarted and counts from its new start.
                self.outputs.clear()
            self.outputs.append((now, produced))

    def decide(self, now: float, serving: int, load: dict | None) -> Decision:
        """`load`: the serving engines' requests running and queued, and their slots, summed."""
        with self.lock:
            if serving != self.serving:
                self.serving, self.serving_since = serving, now
                if self.move is not None:
                    self.move = self.move._replace(served_at=None)
            if load and load["slots"] and serving:
                filled = (load["running"] + load["queued"]) * serving / load["slots"]
                self.loads.append(Load(now, serving, filled))
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

    def _solve(self, now: float, serving: int, count: int) -> tuple[int, str]:
        demand, waited = self._demand(), self._waited()
        per_engine, filled = self._per_engine(now), self._filled(now)
        if demand is None:
            return count, "no training step yet"
        if per_engine is None or filled is None:
            return count, "no output measured from the serving engines yet"
        if waited is None:
            return count, "no training step on the serving engines yet"
        needed = demand * (1 + HEADROOM) / per_engine
        reason = (
            f"the trainer waited {waited:.0%} of its last steps and takes {demand:.3g} groups/s, "
            f"an engine puts out {per_engine:.3g}/s: {needed:.2f} needed"
        )
        kind = "add" if waited >= SHORT else "shrink"
        if kind == "shrink" and needed < filled * (1 + HEADROOM):
            # Fewer engines than the requests on them fill would queue them until they end.
            needed = filled * (1 + HEADROOM)
            reason = f"{reason}; the requests on the engines fill {filled:.2f}"
        target = math.ceil(needed)
        if target < self.minimum:
            target, reason = self.minimum, f"{reason}; at the minimum"
        if self.maximum is not None and target > self.maximum:
            target, reason = self.maximum, f"{reason}; at the maximum"
        if serving != count:
            reason = f"{reason}; {serving} of {count} engines serve"
        if kind == "add":
            ready = target > count and serving == count
        else:
            ready = target < count and target <= serving
        if not ready:
            return count, reason
        if self.count is None:
            return target, reason
        if (undone := self.undone.get((kind, count))) is not None and _close(demand, undone):
            return count, f"{reason}; the {kind} from {count} was undone"
        self.count, self.move = target, Move(kind, count, filled, demand)
        return target, reason

    def _follow_move(self, now: float, move: Move) -> str:
        if move.served_at is None:
            self.move = move = move._replace(served_at=now)
        if move.kind == "undo":
            self.move = None
            return f"back at {self.count}"
        served_at = move.served_at or now
        if move.kind == "add":
            filled = self._filled(now, served_at)
            if now - served_at < CHECK_SECONDS or filled is None:
                return f"checking the add from {move.before}"
            self.move = None
            grew = (filled / move.filled - 1) / (self.count / move.before - 1) if move.filled else 1.0
            outcome = f"the requests grew {grew:.0%} as much as the engines"
            return outcome if grew >= SIGNIFICANT else self._undo(move, outcome)
        after = [step for step in self.steps if step.at > served_at]
        if not after:
            return f"checking the shrink from {move.before}"
        self.move = None
        waited = sum(step.waited for step in after) / sum(step.seconds for step in after)
        if waited >= SHORT:
            return self._undo(move, f"the trainer waited {waited:.0%} after it")
        return "the shrink held: the trainer stayed fed"

    def _undo(self, move: Move, outcome: str) -> str:
        self.undone[(move.kind, move.before)] = move.demand
        self.count, self.move = move.before, Move("undo", self.count, 0.0, 0.0)
        return f"undone: {outcome}"

    def _demand(self) -> float | None:
        """Groups the trainer takes per second it does not wait."""
        training = sum(step.seconds - step.waited for step in self.steps)
        return sum(step.consumed for step in self.steps) / training if training > 0 else None

    def _waited(self) -> float | None:
        """The share of its last steps the trainer waited, of those begun once output shows the serving engines."""
        steps = [step for step in self.steps if step.at - step.seconds >= self.serving_since + LAG]
        seconds = sum(step.seconds for step in steps)
        return sum(step.waited for step in steps) / seconds if seconds else None

    def _per_engine(self, now: float) -> float | None:
        """Groups put per second per serving engine, from LAG after the serving engines last changed."""
        since = max(self.serving_since + LAG, now - WINDOW)
        output = self._output(now, since) if now - since >= CHECK_SECONDS else None
        return output / self.serving if output and self.serving else None

    def _filled(self, now: float, since: float = -math.inf) -> float | None:
        """The engines the requests on them fill, averaged from `since`, at most WINDOW back."""
        filled = [load.filled for load in self.loads if load.at >= max(since, now - WINDOW)]
        return sum(filled) / len(filled) if filled else None

    def _output(self, now: float, since: float) -> float | None:
        """Groups put per second from `since`, at most WINDOW back."""
        reports = [(at, produced) for at, produced in self.outputs if at >= max(since, now - WINDOW)]
        if len(reports) < 2:
            return None
        (first_at, first), (last_at, last) = reports[0], reports[-1]
        return (last - first) / (last_at - first_at)


def _close(value: float, seen: float) -> bool:
    return abs(value - seen) < SIGNIFICANT * seen
