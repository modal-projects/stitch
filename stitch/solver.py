"""Auto mode: the engines whose output keeps up with what the trainer takes, each move checked before the next."""

import logging
import math
import threading
from collections import deque
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Defaults, to become run settings: output kept spare over what the trainer takes, and how long the serving engines
# hold before an add or a shrink reads their output; a move is checked after the first.
HEADROOM = 0.1
ADD_SECONDS, SHRINK_SECONDS = 300, 900
# An add counts from this share of the output it promised; an undone move is not retried while the engines needed
# stay within this share of what they were.
SIGNIFICANT = 0.25


class Step(NamedTuple):
    seconds: float
    waited: float
    consumed: int


class Move(NamedTuple):
    kind: str
    before: int
    output: float
    promised: float
    needed: float
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
        self.outputs: deque[tuple[float, int]] = deque(maxlen=SHRINK_SECONDS // 60 + 1)
        self.uses: deque[tuple[float, int, float]] = deque(
            maxlen=SHRINK_SECONDS // 60 + 1
        )
        self.serving: int | None = None
        self.serving_since = self.count_since = -math.inf
        self.move: Move | None = None
        self.undone: dict[tuple[str, int], float] = {}
        self.decision: Decision | None = None
        self.lock = threading.Lock()

    def observe_step(self, seconds: float, waited: float, consumed: int) -> None:
        with self.lock:
            self.steps.append(Step(seconds, waited, consumed))

    def observe_output(self, now: float, produced: int) -> None:
        """`produced`: the groups the rollout put into the trainer's buffer since its start."""
        with self.lock:
            if self.outputs and produced < self.outputs[-1][1]:
                # The rollout restarted and counts from its new start.
                self.outputs.clear()
            self.outputs.append((now, produced))

    def decide(self, now: float, serving: int, load: dict | None) -> Decision:
        """`load`: the serving engines' requests running and queued and their slots, summed, and their average use."""
        with self.lock:
            if serving != self.serving:
                self.serving, self.serving_since = serving, now
                if self.move is not None:
                    self.move = self.move._replace(served_at=None)
            if load:
                self.uses.append((now, serving, load["use"]))
            count = serving if self.count is None else self.count
            if self.count is not None and self.move is not None and serving == count:
                reason = self._follow_move(now, serving, self.move)
                target = self.count
            else:
                target, reason = self._solve(now, serving, count, load)
            decision = Decision(target, reason, load["slots"] if load else None)
            previous, self.decision = self.decision, decision
            if previous is None or previous[:2] != decision[:2]:
                logger.info(
                    "%sengines %d -> %d: %s",
                    "shadow: " if self.count is None else "",
                    count,
                    target,
                    reason,
                )
            return decision

    def _solve(
        self, now: float, serving: int, count: int, load: dict | None
    ) -> tuple[int, str]:
        demand, per_engine = self._demand(), self._per_engine(now)
        if demand is None:
            return count, "no training step yet"
        if per_engine is None:
            return count, "no output measured on the serving engines yet"
        needed = demand * (1 + HEADROOM) / per_engine
        target = math.ceil(needed)
        reason = (
            f"the trainer takes {demand:.3g} groups/s, a fully used engine puts out "
            f"{per_engine:.3g}/s, the engines are {self.uses[-1][2]:.0%} used: "
            f"{needed:.2f} needed"
        )
        if target < self.minimum:
            target, reason = self.minimum, f"{reason}; at the minimum"
        if self.maximum is not None and target > self.maximum:
            target, reason = self.maximum, f"{reason}; at the maximum"
        if target < count and load and load["slots"]:
            # Admitted requests stay where they are: fewer engines than they fill would queue them until they end.
            occupied = math.ceil(
                (load["running"] + load["queued"]) * serving / load["slots"]
            )
            if occupied > target:
                target = min(count, occupied)
                reason = f"{reason}; the requests on the engines fill {occupied}"
        if serving != count:
            reason = f"{reason}; {serving} of {count} engines serve"
        steady = now - max(self.serving_since, self.count_since)
        if target > count and serving == count and steady >= ADD_SECONDS:
            kind = "add"
        elif target < count and target <= serving and steady >= SHRINK_SECONDS:
            kind = "shrink"
        else:
            return count, reason
        if self.count is None:
            return target, reason
        if (undone := self.undone.get((kind, count))) is not None and _close(
            needed, undone
        ):
            return count, f"{reason}; the {kind} from {count} was undone"
        output = self._output(now, self.serving_since)
        # An add promises what its engines put out, up to what the trainer still lacks.
        promised = (
            min((target - count) * per_engine, demand * (1 + HEADROOM) - output)
            if kind == "add"
            else 0.0
        )
        self.count, self.move = target, Move(kind, count, output, promised, needed)
        self.count_since = now
        return target, reason

    def _follow_move(self, now: float, serving: int, move: Move) -> str:
        if move.served_at is None:
            self.move = move = move._replace(served_at=now)
        if move.kind == "undo":
            self.move = None
            return f"back at {self.count}"
        per_engine = self._per_engine(now)
        if now - (move.served_at or now) < ADD_SECONDS or per_engine is None:
            return f"checking the {move.kind} from {move.before}"
        self.move = None
        if move.kind == "add":
            gain = self._output(now, self.serving_since) - move.output
            outcome = f"output rose {gain:.3g}/s of {move.promised:.3g}/s promised"
            return (
                outcome
                if gain >= SIGNIFICANT * move.promised
                else self._undo(now, move, outcome)
            )
        needed = self._demand() * (1 + HEADROOM) / per_engine
        if math.ceil(needed) > self.count:
            return self._undo(now, move, f"{needed:.2f} needed after it")
        return f"the shrink held: {needed:.2f} needed"

    def _undo(self, now: float, move: Move, outcome: str) -> str:
        self.undone[(move.kind, move.before)] = move.needed
        self.count, self.move = move.before, Move("undo", self.count, 0.0, 0.0, 0.0)
        self.count_since = now
        return f"undone: {outcome}"

    def _demand(self) -> float | None:
        """Groups the trainer takes per second it does not wait."""
        training = sum(step.seconds - step.waited for step in self.steps)
        return (
            sum(step.consumed for step in self.steps) / training
            if training > 0
            else None
        )

    def _per_engine(self, now: float) -> float | None:
        """Groups a fully used engine puts out per second over the last SHRINK_SECONDS, across serving changes."""
        output = self._output(now, now - SHRINK_SECONDS)
        in_use = [
            engines * use
            for at, engines, use in self.uses
            if at >= now - SHRINK_SECONDS
        ]
        if not output or not sum(in_use):
            return None
        return output / (sum(in_use) / len(in_use))

    def _output(self, now: float, since: float) -> float | None:
        """Groups put per second from `since`, at most SHRINK_SECONDS back."""
        reports = [
            (at, produced)
            for at, produced in self.outputs
            if at >= max(since, now - SHRINK_SECONDS)
        ]
        if len(reports) < 2:
            return None
        (first_at, first), (last_at, last) = reports[0], reports[-1]
        return (last - first) / (last_at - first_at)


def _close(value: float, seen: float) -> bool:
    return abs(value - seen) < SIGNIFICANT * seen
