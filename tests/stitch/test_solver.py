"""Auto mode's engine count, from the trainer's steps, the rollout's output and the requests on the engines."""

import logging

from stitch.solver import Decision, Solver

SLOTS = 16


class _Run:
    """A trainer taking `demand` groups a second it trains; each engine puts out `per_engine`, `fill` of it in use."""

    def __init__(self, solver: Solver, *, demand: float, per_engine: float, limit=float("inf"), fill=0.5):
        self.solver, self.demand, self.per_engine, self.limit, self.fill = solver, demand, per_engine, limit, fill
        self.now, self.produced = 0.0, 0.0

    def minutes(self, count: int, *, serving: int) -> Decision:
        for _ in range(count):
            self.now += 60
            output = min(self.per_engine * serving, self.limit)
            waited = 60 * max(0.0, 1 - output / self.demand)
            self.solver.observe_step(self.now, 60.0, waited, round(self.demand * (60 - waited)))
            self.produced += output * 60
            self.solver.observe_output(self.now, round(self.produced))
            filled = self.fill * output / self.per_engine
            load = {"running": round(filled * SLOTS), "queued": 0, "slots": SLOTS * serving}
            decision = self.solver.decide(self.now, serving, load)
        return decision


def test_without_a_training_step_it_holds():
    decision = Solver(1, 8, 1).decide(60.0, 1, None)

    assert decision == Decision(1, "no training step yet", None)


def test_output_is_read_a_groups_lifetime_after_the_serving_engines_change():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)

    assert run.minutes(20, serving=1) == Decision(1, "no output measured from the serving engines yet", 16)


def test_a_short_trainer_adds_what_its_demand_over_one_engines_output_needs():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(20, serving=1)

    assert run.minutes(1, serving=1) == Decision(
        2,
        "the trainer waited 23% of its last steps and takes 0.26 groups/s, an engine puts out 0.2/s: 1.43 needed",
        16,
    )


def test_an_add_the_requests_move_onto_is_kept_and_the_next_waits_for_its_output():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(21, serving=1)

    assert run.minutes(5, serving=2).reason == "checking the add from 1"
    assert run.minutes(1, serving=2).reason == "the requests grew 100% as much as the engines"
    assert (run.solver.count, run.solver.move) == (2, None)
    assert run.minutes(14, serving=2).reason == "no output measured from the serving engines yet"


def test_an_add_the_rollout_cannot_use_is_undone_and_not_tried_again_while_the_demand_stays():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2, limit=0.2)
    run.minutes(21, serving=1)

    assert run.minutes(6, serving=2) == Decision(1, "undone: the requests grew 0% as much as the engines", 32)
    assert run.minutes(1, serving=1).reason == "back at 1"
    decision = run.minutes(21, serving=1)
    assert decision.engines == 1
    assert decision.reason.endswith("; the add from 1 was undone")

    run.demand = 0.52
    assert run.minutes(3, serving=1).engines == 3


def test_a_fed_trainer_shrinks_to_what_its_demand_needs_but_not_below_what_the_requests_fill():
    run = _Run(Solver(1, 8, 4), demand=0.1, per_engine=0.2)
    run.minutes(20, serving=4)

    assert run.minutes(1, serving=4) == Decision(
        3,
        "the trainer waited 0% of its last steps and takes 0.1 groups/s, an engine puts out 0.2/s: 0.55 needed; "
        "the requests on the engines fill 2.00",
        64,
    )


def test_engines_the_requests_fill_are_not_shrunk():
    run = _Run(Solver(1, 8, 4), demand=0.1, per_engine=0.2, fill=1.0)

    decision = run.minutes(21, serving=4)

    assert decision.engines == 4
    assert decision.reason.endswith("0.55 needed; the requests on the engines fill 4.00")


def test_a_shrink_held_once_the_next_step_finds_the_trainer_fed():
    run = _Run(Solver(1, 8, 4), demand=0.1, per_engine=0.2)
    run.minutes(21, serving=4)

    assert run.minutes(1, serving=3).reason == "checking the shrink from 4"
    assert run.minutes(1, serving=3).reason == "the shrink held: the trainer stayed fed"


def test_a_shrink_after_which_the_trainer_waits_is_undone_and_not_tried_again_while_the_demand_stays():
    run = _Run(Solver(1, 8, 4), demand=0.1, per_engine=0.2)
    run.minutes(21, serving=4)
    # Three engines alone put out less than their share of four did.
    run.per_engine = 0.03

    assert run.minutes(2, serving=3) == Decision(4, "undone: the trainer waited 10% after it", 48)
    run.per_engine = 0.2
    assert run.minutes(1, serving=4).reason == "back at 4"
    decision = run.minutes(21, serving=4)
    assert decision.engines == 4
    assert decision.reason.endswith("the requests on the engines fill 2.00; the shrink from 4 was undone")


def test_while_engines_it_asked_for_start_it_neither_adds_nor_lowers_toward_them():
    run = _Run(Solver(1, 32, 8), demand=4.0, per_engine=0.2)
    run.minutes(21, serving=8)
    assert run.solver.count == 22

    run.demand = 5.0
    assert run.minutes(10, serving=8).engines == 22
    run.demand = 3.5
    decision = run.minutes(10, serving=8)
    assert decision.engines == 22
    assert decision.reason.endswith("19.25 needed; 8 of 22 engines serve")


def test_short_of_engines_it_asked_for_it_lowers_the_count_to_what_serves_once_the_trainer_is_fed():
    """A GPU shortage: the second engine never starts, and the trainer comes to take less."""
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(21, serving=1)
    run.demand = 0.1

    assert run.minutes(2, serving=1).engines == 2
    run.minutes(1, serving=1)

    assert (run.solver.count, run.solver.move.kind) == (1, "shrink")


def test_an_engine_lost_while_an_add_is_checked_restarts_the_check():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(21, serving=1)
    run.minutes(3, serving=2)
    run.minutes(1, serving=1)

    assert run.minutes(5, serving=2).reason == "checking the add from 1"
    assert run.minutes(1, serving=2).reason == "the requests grew 100% as much as the engines"


def test_a_restarted_rollout_measures_its_output_again():
    solver = Solver(1, 8, 1)
    solver.observe_step(1200.0, 60.0, 0.0, 26)
    load = {"running": 16, "queued": 0, "slots": 16}
    solver.decide(0.0, 1, load)
    solver.observe_output(1200.0, 500)
    solver.observe_output(1260.0, 3)

    reason = solver.decide(1260.0, 1, load).reason

    assert reason == "no output measured from the serving engines yet"


def test_at_the_maximum_it_holds_and_says_so():
    run = _Run(Solver(1, 2, 2), demand=0.8, per_engine=0.2)

    decision = run.minutes(21, serving=2)

    assert decision.engines == 2
    assert decision.reason.endswith("4.40 needed; at the maximum")


def test_at_the_minimum_it_holds_and_says_so():
    run = _Run(Solver(3, 8, 3), demand=0.1, per_engine=0.2)

    decision = run.minutes(21, serving=3)

    assert decision.engines == 3
    assert decision.reason.endswith("the requests on the engines fill 1.50; at the minimum")


def test_a_fixed_pool_logs_what_auto_mode_would_set_and_moves_nothing(caplog):
    caplog.set_level(logging.INFO, logger="stitch.solver")
    run = _Run(Solver(1, 8, None), demand=0.26, per_engine=0.2)

    decision = run.minutes(21, serving=1)

    assert (decision.engines, run.solver.count, run.solver.move) == (2, None, None)
    assert "shadow: engines 1 -> 2: the trainer waited 23%" in caplog.text
