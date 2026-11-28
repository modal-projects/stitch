"""Auto mode's engine count from the trainer's steps, the rollout's output and holds, and the engines' requests."""

import logging

from stitch.solver import Decision, Solver

SLOTS = 16


class _Run:
    """A trainer taking `demand` groups/s from engines putting out `per_engine` each; ahead, the rollout holds off."""

    def __init__(self, solver, *, demand, per_engine, limit=float("inf"), lifetime=None, paced=False, requests=None):
        """`paced`: the rollout holds off whatever the engines put out; `requests`: on the engines, however busy."""
        self.solver, self.demand, self.per_engine, self.limit = solver, demand, per_engine, limit
        self.lifetime, self.paced, self.requests = lifetime, paced, requests
        self.now, self.produced = 0.0, 0.0

    def minutes(self, count: int, *, serving: int) -> Decision:
        for _ in range(count):
            self.now += 60
            can = min(self.per_engine * serving, self.limit)
            output = min(can, self.demand)
            waited = 60 * (1 - output / self.demand)
            held = 30.0 if self.paced else 60 * (1 - output / can) if can else 0.0
            self.solver.observe_step(self.now, 60.0, waited, round(self.demand * (60 - waited)))
            self.produced += output * 60
            self.solver.observe_load(self.now, round(self.produced), held, self.lifetime)
            requests = self.requests if self.requests is not None else SLOTS / 2 * output / self.per_engine
            load = {
                "running": round(min(requests, SLOTS * serving)),
                "queued": round(max(0.0, requests - SLOTS * serving)),
                "slots": SLOTS * serving,
            }
            decision = self.solver.decide(self.now, serving, load)
        return decision


def test_without_a_training_step_it_holds():
    decision = Solver(1, 8, 1).decide(60.0, 1, None)

    assert decision == Decision(1, "no training step yet", None)


def test_output_is_read_only_a_group_lifetime_after_the_serving_engines_change():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)

    assert run.minutes(25, serving=1) == Decision(1, "no output measured from the serving engines yet", 16)


def test_a_short_trainer_adds_what_its_demand_over_one_engines_output_needs():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(25, serving=1)

    assert run.minutes(1, serving=1) == Decision(
        2,
        "the trainer waited 23% of its last steps and takes 0.26 groups/s, an engine puts out 0.2/s: 1.43 needed",
        16,
    )


def test_the_group_lifetime_the_rollout_reports_sets_the_wait():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2, lifetime=300.0)

    assert run.minutes(15, serving=1).engines == 1
    assert run.minutes(1, serving=1).engines == 2


def test_an_add_that_puts_the_rollout_ahead_of_the_trainer_holds_once_its_effect_is_in():
    run = _Run(Solver(1, 8, 1), demand=0.6, per_engine=0.2)
    run.minutes(26, serving=1)
    assert run.solver.count == 4

    assert run.minutes(25, serving=4).reason == "checking the add from 1"
    assert run.minutes(1, serving=4).reason == "the add held: the rollout got ahead and held off admitting"
    assert (run.solver.count, run.solver.move) == (4, None)


def test_an_add_short_of_what_is_needed_holds_once_its_output_rose():
    run = _Run(Solver(1, 3, 1), demand=0.8, per_engine=0.2)
    run.minutes(26, serving=1)
    assert run.solver.count == 3

    run.minutes(25, serving=3)

    assert run.minutes(1, serving=3).reason == "output rose 0.4/s of 0.4/s promised"


def test_an_add_the_rollout_cannot_use_is_undone_and_not_tried_again_while_the_demand_stays():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2, limit=0.2)
    run.minutes(26, serving=1)

    assert run.minutes(26, serving=2) == Decision(1, "undone: output rose 0/s of 0.2/s promised", 32)
    assert run.minutes(1, serving=1).reason == "back at 1"
    decision = run.minutes(26, serving=1)
    assert decision.engines == 1
    assert decision.reason.endswith("; the add from 1 was undone")

    run.demand = 0.52
    assert run.minutes(3, serving=1).engines == 3


def test_a_rollout_held_off_by_its_own_limits_gets_no_engines_it_would_not_use():
    """A staleness bound or the args hold the rollout off while the trainer still waits."""
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2, paced=True)
    run.solver.per_engine = 0.2

    decision = run.minutes(26, serving=1)

    assert decision.engines == 1
    assert decision.reason.endswith("the rollout held off admitting, so more engines would not be used")


def _held_off(requests=None):
    """Four engines that kept up with the trainer, until it came to take an eighth: the rollout holds off since."""
    run = _Run(Solver(1, 8, 4), demand=0.8, per_engine=0.2, requests=requests)
    run.minutes(26, serving=4)
    run.demand = 0.1
    return run


def test_a_rollout_held_off_ahead_of_the_trainer_gives_up_its_surplus_engines():
    """The second step at the new demand already brings what is needed one engine under the count."""
    run = _held_off()

    assert run.minutes(2, serving=4) == Decision(
        2,
        "the trainer waited 0% of its last steps and takes 0.333 groups/s, an engine puts out 0.2/s: 1.83 needed; "
        "the rollout held off admitting",
        64,
    )


def test_a_shrink_keeps_the_engines_the_requests_on_them_fill():
    run = _held_off(requests=40)

    decision = run.minutes(2, serving=4)

    assert decision.engines == 3
    assert decision.reason.endswith("the rollout held off admitting; the requests on the engines fill 2.50")


def test_a_shrink_holds_while_the_rollout_stays_ahead():
    run = _held_off()
    run.minutes(2, serving=4)

    assert run.minutes(25, serving=2).reason == "checking the shrink from 4"
    assert run.minutes(1, serving=2).reason == "the shrink held: the rollout still held off admitting"


def test_a_shrink_after_which_output_falls_below_the_trainers_demand_is_undone_and_not_tried_again():
    run = _held_off()
    run.minutes(2, serving=4)
    # Two engines alone put out less than their share of four did.
    run.per_engine = 0.04

    assert run.minutes(26, serving=2) == Decision(4, "undone: output 0.08/s against the trainer's 0.104/s", 32)
    run.per_engine = 0.2
    assert run.minutes(1, serving=4).reason == "back at 4"
    decision = run.minutes(26, serving=4)
    assert decision.engines == 4
    assert decision.reason.endswith("; the shrink from 4 was undone")


def test_engines_within_one_of_what_is_needed_stay():
    run = _Run(Solver(1, 8, 4), demand=0.8, per_engine=0.2)
    run.minutes(26, serving=4)
    run.demand = 0.6

    decision = run.minutes(3, serving=4)

    assert decision.engines == 4
    assert decision.reason.endswith("3.30 needed; the rollout held off admitting")


def test_while_engines_it_asked_for_start_it_neither_adds_nor_lowers_toward_them():
    run = _Run(Solver(1, 32, 8), demand=4.0, per_engine=0.2)
    run.minutes(26, serving=8)
    assert run.solver.count == 22

    run.demand = 5.0
    assert run.minutes(10, serving=8).engines == 22
    run.demand = 3.5
    decision = run.minutes(10, serving=8)
    assert decision.engines == 22
    assert decision.reason.endswith("19.25 needed; 8 of 22 engines serve")


def test_short_of_engines_it_asked_for_it_lowers_the_count_to_what_serves_once_the_rollout_holds_off():
    """A GPU shortage: the second engine never starts, and the trainer comes to take less."""
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(26, serving=1)
    run.demand = 0.1

    run.minutes(3, serving=1)

    assert (run.solver.count, run.solver.move.kind) == (1, "shrink")


def test_an_engine_lost_while_an_add_is_checked_restarts_the_check():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(26, serving=1)
    run.minutes(10, serving=2)
    run.minutes(1, serving=1)

    assert run.minutes(25, serving=2).reason == "checking the add from 1"
    assert run.minutes(1, serving=2).reason == "the add held: the rollout got ahead and held off admitting"


def test_a_restarted_rollout_measures_its_output_again():
    solver = Solver(1, 8, 1)
    solver.observe_step(1500.0, 60.0, 0.0, 26)
    load = {"running": 16, "queued": 0, "slots": 16}
    solver.decide(0.0, 1, load)
    solver.observe_load(1500.0, 500)
    solver.observe_load(1560.0, 3)

    reason = solver.decide(1560.0, 1, load).reason

    assert reason == "no output measured from the serving engines yet"


def test_at_the_maximum_it_holds_and_says_so():
    run = _Run(Solver(1, 2, 2), demand=0.8, per_engine=0.2)

    decision = run.minutes(26, serving=2)

    assert decision.engines == 2
    assert decision.reason.endswith("4.40 needed; at the maximum")


def test_at_the_minimum_it_stops_there_and_says_so():
    run = _held_off()
    run.solver.minimum = 3

    decision = run.minutes(2, serving=4)

    assert decision.engines == 3
    assert decision.reason.endswith("held off admitting; at the minimum")


def test_a_fixed_pool_logs_what_auto_mode_would_set_and_moves_nothing(caplog):
    caplog.set_level(logging.INFO, logger="stitch.solver")
    run = _Run(Solver(1, 8, None), demand=0.26, per_engine=0.2)

    decision = run.minutes(26, serving=1)

    assert (decision.engines, run.solver.count, run.solver.move) == (2, None, None)
    assert "shadow: engines 1 -> 2: the trainer waited 23%" in caplog.text
