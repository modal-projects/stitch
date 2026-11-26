"""Auto mode's engine count, from the trainer's steps and the rollout's output, each move checked before the next."""

import logging

from stitch.solver import Decision, Solver

SLOTS = 32


class _Run:
    """A trainer taking `demand` groups a second; each serving engine puts out `per_engine` a second while the
    producer is not held, which it is for `held` of each minute. One report and one decision a minute."""

    def __init__(self, solver: Solver, *, demand: float, per_engine: float, held=0.0):
        self.solver, self.per_engine, self.held = solver, per_engine, held
        self.now, self.output = 0.0, {"produced": 0.0, "held": 0.0}
        self.steps(demand)

    def steps(self, demand: float) -> None:
        for _ in range(3):
            self.solver.observe_step(100.0, 0.0, round(demand * 100))

    def minutes(self, count: int, *, serving: int) -> Decision:
        for _ in range(count):
            self.now += 60
            unheld = 60 * (1 - self.held)
            self.output = {
                "produced": self.output["produced"]
                + self.per_engine * serving * unheld,
                "held": self.output["held"] + 60 - unheld,
            }
            self.solver.observe_output(self.now, self.output)
            decision = self.solver.decide(self.now, serving, SLOTS * serving)
        return decision


def test_without_a_training_step_it_holds():
    assert Solver(1, 8, 1).decide(60.0, 1, SLOTS) == Decision(
        1, "no training step yet", SLOTS
    )


def test_without_output_from_the_serving_engines_it_holds():
    solver = Solver(1, 8, 1)
    solver.observe_step(100.0, 0.0, 26)

    reason = solver.decide(60.0, 1, SLOTS).reason

    assert reason == "no output measured on the serving engines yet"


def test_a_trainer_taking_more_than_the_engines_put_out_adds_once_they_served_five_minutes():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)

    assert run.minutes(5, serving=1).engines == 1
    assert run.minutes(1, serving=1) == Decision(
        2, "the trainer takes 0.26 groups/s, 1 engines put out 0.2/s: 1.43 needed", 32
    )


def test_an_add_that_raised_the_output_is_kept():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(6, serving=1)

    assert run.minutes(5, serving=2).reason == "checking the add from 1"
    assert run.minutes(1, serving=2).reason == "output rose 0.2/s of 0.2/s promised"
    assert (run.solver.count, run.solver.move) == (2, None)


def test_an_add_that_did_not_raise_the_output_is_undone_and_not_tried_again_while_the_need_stays():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(6, serving=1)
    # Two engines put out what one did: something other than the engines limits the rollout.
    run.per_engine = 0.1

    assert run.minutes(6, serving=2) == Decision(
        1, "undone: output rose 0/s of 0.2/s promised", 64
    )
    run.per_engine = 0.2
    assert run.minutes(1, serving=1).reason == "back at 1"
    decision = run.minutes(6, serving=1)
    assert decision.engines == 1
    assert decision.reason.endswith("1.43 needed; the add from 1 was undone")

    run.steps(0.52)
    assert run.minutes(1, serving=1).engines == 3


def test_engines_held_by_a_full_buffer_shrink_to_what_the_trainer_takes_after_fifteen_minutes():
    run = _Run(Solver(1, 8, 4), demand=0.1, per_engine=0.2, held=0.875)

    assert run.minutes(15, serving=4).engines == 4
    assert run.minutes(1, serving=4) == Decision(
        1, "the trainer takes 0.1 groups/s, 4 engines put out 0.8/s: 0.55 needed", 128
    )


def test_engines_whose_extra_groups_go_stale_shrink_too():
    run = _Run(Solver(1, 8, 4), demand=0.1, per_engine=0.2)

    assert run.minutes(16, serving=4).engines == 1


def test_a_shrink_that_leaves_too_few_engines_is_undone_and_not_tried_again_while_the_need_stays():
    run = _Run(Solver(1, 8, 4), demand=0.1, per_engine=0.2, held=0.875)
    run.minutes(16, serving=4)
    # One engine alone puts out less than a quarter of what four did.
    run.per_engine, run.held = 0.08, 0.0

    assert run.minutes(5, serving=1).reason == "checking the shrink from 4"
    assert run.minutes(1, serving=1) == Decision(4, "undone: 1.38 needed after it", 32)
    run.per_engine, run.held = 0.2, 0.875
    assert run.minutes(1, serving=4).reason == "back at 4"
    decision = run.minutes(16, serving=4)
    assert decision.engines == 4
    assert decision.reason.endswith("0.55 needed; the shrink from 4 was undone")


def test_while_engines_it_asked_for_start_it_adds_no_more():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(6, serving=1)
    run.steps(0.8)

    decision = run.minutes(10, serving=1)

    assert decision.engines == 2
    assert decision.reason.endswith("4.40 needed; 1 of 2 engines serve")


def test_short_of_engines_it_asked_for_it_may_still_lower_the_count():
    """A GPU shortage: the second engine never starts, and the trainer comes to take less."""
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(6, serving=1)
    run.steps(0.1)

    run.minutes(10, serving=1)

    assert (run.solver.count, run.solver.move.kind) == (1, "shrink")


def test_an_engine_lost_while_an_add_is_checked_restarts_the_check():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(6, serving=1)
    run.minutes(3, serving=2)
    run.minutes(1, serving=1)

    assert run.minutes(5, serving=2).reason == "checking the add from 1"
    assert run.minutes(1, serving=2).reason == "output rose 0.2/s of 0.2/s promised"


def test_a_restarted_rollout_measures_its_output_again():
    solver = Solver(1, 8, 1)
    solver.observe_step(100.0, 0.0, 26)
    solver.decide(0.0, 1, SLOTS)
    solver.observe_output(60.0, {"produced": 500, "held": 0.0})
    solver.observe_output(120.0, {"produced": 3, "held": 0.0})

    reason = solver.decide(120.0, 1, SLOTS).reason

    assert reason == "no output measured on the serving engines yet"


def test_at_the_maximum_it_holds_and_says_so():
    run = _Run(Solver(1, 2, 2), demand=0.8, per_engine=0.2)

    assert run.minutes(6, serving=2) == Decision(
        2,
        "the trainer takes 0.8 groups/s, 2 engines put out 0.4/s: 4.40 needed; at the maximum",
        64,
    )


def test_a_fixed_pool_logs_what_auto_mode_would_set_and_moves_nothing(caplog):
    caplog.set_level(logging.INFO, logger="stitch.solver")
    run = _Run(Solver(1, 8, None), demand=0.26, per_engine=0.2)

    decision = run.minutes(6, serving=1)

    assert (decision.engines, run.solver.count, run.solver.move) == (2, None, None)
    assert "shadow: engines 1 -> 2: the trainer takes 0.26 groups/s" in caplog.text
