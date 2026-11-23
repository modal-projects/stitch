"""Auto mode's engine count, from the rollout's report of each training step."""

import logging
import math
import threading
from collections import deque
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Production changes count from this share up: of what a rise promised, or of the production a failed move saw.
SIGNIFICANT = 0.25


class Step(NamedTuple):
    need: float
    seconds: float
    waited: float
    produced: int


class Rise(NamedTuple):
    engines: int
    production: float
    need: float


class Fall(NamedTuple):
    engines: int
    production: float


class Autoscaler:
    """The engine count that keeps the trainer fed, moved halfway to the need three agreeing steps show."""

    def __init__(self, minimum: int, maximum: int | None, count: int, steps: int = 3):
        self.minimum, self.maximum, self.count = minimum, maximum, count
        self.steps: deque[Step] = deque(maxlen=steps)
        self.serving: int | None = None
        self.serving_since = self.count_since = -math.inf
        self.pending_rise: Rise | None = None
        self.stalled_production: float | None = None
        self.pending_fall: Fall | None = None
        self.starved: Fall | None = None
        self.lock = threading.Lock()

    def observe_serving(self, now: float, engines: int) -> None:
        with self.lock:
            if engines != self.serving:
                self.serving, self.serving_since = engines, now
                self.steps.clear()

    def observe_step(
        self,
        now: float,
        seconds: float,
        waited: float,
        held: float,
        consumed: int,
        produced: int,
    ) -> None:
        """Counts a step only if the serving engines and the count held through it."""
        with self.lock:
            steady = max(self.serving_since, self.count_since) <= now - seconds
            engines = self.serving
            if not (steady and engines and produced and max(waited, held) < seconds):
                return
            # Groups the trainer uses per second it did not wait, over those produced per second not held back.
            need = (
                engines * consumed * (seconds - held) / ((seconds - waited) * produced)
            )
            self.steps.append(Step(need, seconds, waited, produced))
            if len(self.steps) == self.steps.maxlen:
                self._judge(now, engines)

    def _judge(self, now: float, engines: int) -> None:
        needs = [step.need for step in self.steps]
        production = sum(step.produced for step in self.steps) / sum(
            step.seconds for step in self.steps
        )
        fall, self.pending_fall = self.pending_fall, None
        if (rise := self.pending_rise) is not None and engines > rise.engines:
            self.pending_rise = None
            # A rise promises production in proportion to the engines it adds, up to the need.
            promised = min(engines, rise.need) - rise.engines
            share = rise.production * promised / rise.engines
            if production - rise.production < SIGNIFICANT * share:
                self.stalled_production = rise.production
                self._set(now, rise.engines, f"undone: production {production:.3g}/s")
                return
        # Short only if the trainer waited every time: a buffer with groups to spare covers a slower step.
        waited = all(step.waited for step in self.steps)
        if waited and (fewest := math.ceil(min(needs))) > engines:
            if fall is not None:
                # The last fall left the trainer waiting.
                self.starved = fall
            stalled = self.stalled_production
            if (
                stalled is not None
                and abs(production - stalled) < SIGNIFICANT * stalled
            ):
                return
            count = self._bounded(engines + math.ceil((fewest - engines) / 2))
            if count > self.count:
                self.pending_rise = Rise(engines, production, min(needs))
                self.stalled_production = None
                self._set(now, count, f"needs {[round(need, 2) for need in needs]}")
            elif self.count == self.maximum:
                logger.info("the trainer waits with the maximum %d engines", self.count)
        elif (most := math.ceil(max(needs))) < engines:
            starved = self.starved
            if (
                starved is not None
                and engines == starved.engines
                and abs(production - starved.production)
                < SIGNIFICANT * starved.production
            ):
                return
            count = self._bounded(engines - math.ceil((engines - most) / 2))
            if count != self.count:
                self.stalled_production = None
                self.pending_fall = Fall(engines, production)
                self._set(now, count, f"needs {[round(need, 2) for need in needs]}")

    def _set(self, now: float, count: int, why: str) -> None:
        logger.info(
            "engines %d -> %d, %d serving: %s", self.count, count, self.serving, why
        )
        self.count, self.count_since = count, now
        self.steps.clear()

    def _bounded(self, engines: int) -> int:
        engines = max(self.minimum, engines)
        return engines if self.maximum is None else min(self.maximum, engines)
