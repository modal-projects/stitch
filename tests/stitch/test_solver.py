"""Auto mode's engine count, from the trainer's steps, the rollout's output and the engines' load."""

import logging

from stitch.solver import Decision, Solver

SLOTS = 32


class _Run:
    """A trainer taking `demand` groups a second, and serving engines that run `use` of their slots with `queued`
    requests waiting, each fully used engine putting out `per_engine` groups a second; an engine with requests waiting
    runs all it can. One minute per decision."""

    def __init__(self, solver: Solver, *, demand: float, per_engine: float, use=1.0):
        self.solver, self.per_engine, self.use, self.queued = solver, per_engine, use, 0
        self.now, self.produced = 0.0, 0.0
        self.steps(demand)

    def steps(self, demand: float) -> None:
        for _ in range(3):
            self.solver.observe_step(100.0, 0.0, round(demand * 100))

    def minutes(self, count: int, *, serving: int) -> Decision:
        for _ in range(count):
            self.now += 60
            use = 1.0 if self.queued else self.use
            self.produced += self.per_engine * serving * use * 60
            self.solver.observe_output(self.now, round(self.produced))
            load = {
                "running": round(self.use * SLOTS * serving),
                "queued": self.queued,
                "slots": SLOTS * serving,
                "use": use,
            }
            decision = self.solver.decide(self.now, serving, load)
        return decision


def test_without_a_training_step_it_holds():
    decision = Solver(1, 8, 1).decide(60.0, 1, None)

    assert decision == Decision(1, "no training step yet", None)


def test_without_output_from_the_serving_engines_it_holds():
    solver = Solver(1, 8, 1)
    solver.observe_step(100.0, 0.0, 26)

    reason = solver.decide(
        60.0, 1, {"running": 32, "queued": 0, "slots": 32, "use": 1.0}
    ).reason

    assert reason == "no output measured on the serving engines yet"


def test_a_trainer_taking_more_than_the_engines_put_out_adds_once_they_served_five_minutes():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)

    assert run.minutes(5, serving=1).engines == 1
    assert run.minutes(1, serving=1) == Decision(
        2,
        "the trainer takes 0.26 groups/s, a fully used engine puts out 0.2/s, the engines are 100% "
        "used: 1.43 needed",
        32,
    )


def test_an_add_that_raised_the_output_is_kept():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(6, serving=1)
    # Two engines, each half used: they put out what the trainer takes.
    run.use = 0.65

    assert run.minutes(5, serving=2).reason == "checking the add from 1"
    assert run.minutes(1, serving=2).reason == "output rose 0.06/s of 0.086/s promised"
    assert (run.solver.count, run.solver.move) == (2, None)


def test_an_add_that_did_not_raise_the_output_is_undone_and_not_tried_again_while_the_need_stays():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(6, serving=1)
    # Something other than the engines limits the rollout: two engines put out what one did.
    run.use = 0.5

    assert run.minutes(6, serving=2) == Decision(
        1, "undone: output rose 0/s of 0.086/s promised", 64
    )
    run.use = 1.0
    assert run.minutes(1, serving=1).reason == "back at 1"
    decision = run.minutes(6, serving=1)
    assert decision.engines == 1
    assert decision.reason.endswith("1.43 needed; the add from 1 was undone")

    run.steps(0.52)
    assert run.minutes(1, serving=1).engines == 3


def test_engines_the_trainer_does_not_need_shrink_after_fifteen_minutes():
    run = _Run(Solver(1, 8, 4), demand=0.1, per_engine=0.2, use=0.125)

    assert run.minutes(15, serving=4).engines == 4
    assert run.minutes(1, serving=4) == Decision(
        1,
        "the trainer takes 0.1 groups/s, a fully used engine puts out 0.2/s, the engines are 12% "
        "used: 0.55 needed",
        128,
    )


def test_a_shrink_leaves_the_engines_the_requests_on_them_fill():
    run = _Run(Solver(1, 8, 4), demand=0.1, per_engine=0.2, use=0.5)
    run.queued = 16

    decision = run.minutes(16, serving=4)

    assert decision.engines == 3
    assert decision.reason.endswith("0.55 needed; the requests on the engines fill 3")


def test_a_shrink_the_trainer_outgrows_is_undone_and_not_tried_again_while_the_need_stays():
    run = _Run(Solver(1, 8, 4), demand=0.1, per_engine=0.2, use=0.125)
    run.minutes(16, serving=4)
    run.steps(0.3)
    run.use = 0.5

    assert run.minutes(5, serving=1).reason == "checking the shrink from 4"
    assert run.minutes(1, serving=1) == Decision(4, "undone: 1.65 needed after it", 32)
    run.steps(0.1)
    run.use = 0.125
    assert run.minutes(1, serving=4).reason == "back at 4"
    decision = run.minutes(16, serving=4)
    assert decision.engines == 4
    assert decision.reason.endswith("0.55 needed; the shrink from 4 was undone")


def test_engines_with_requests_waiting_count_as_fully_used():
    """Long contexts fill an engine's cache before its slots: half its slots run and the rest of its requests wait."""
    run = _Run(Solver(1, 16, 5), demand=0.15, per_engine=0.02, use=0.5)
    run.queued = 20

    decision = run.minutes(6, serving=5)

    assert decision.engines == 9
    assert decision.reason.endswith("the engines are 100% used: 8.25 needed")


def test_the_output_measure_holds_across_a_change_in_serving_engines():
    run = _Run(Solver(1, 8, None), demand=0.26, per_engine=0.2)
    run.minutes(6, serving=1)

    reason = run.minutes(1, serving=2).reason

    assert reason.startswith(
        "the trainer takes 0.26 groups/s, a fully used engine puts out 0.2"
    )


def test_while_engines_it_asked_for_start_it_neither_adds_nor_lowers_toward_them():
    run = _Run(Solver(1, 32, 8), demand=4.0, per_engine=0.2)
    run.minutes(6, serving=8)
    assert run.solver.count == 22

    run.steps(5.0)
    assert run.minutes(10, serving=8).engines == 22
    run.steps(3.5)
    decision = run.minutes(10, serving=8)
    assert decision.engines == 22
    assert decision.reason.endswith("19.25 needed; 8 of 22 engines serve")


def test_short_of_engines_it_asked_for_it_lowers_the_count_to_what_serves_once_that_is_enough():
    """A GPU shortage: the second engine never starts, and the trainer comes to take less."""
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(6, serving=1)
    run.steps(0.1)

    assert run.minutes(14, serving=1).engines == 2
    run.minutes(1, serving=1)

    assert (run.solver.count, run.solver.move.kind) == (1, "shrink")


def test_a_reading_that_dips_after_an_add_does_not_lower_the_count_before_the_add_is_checked():
    run = _Run(Solver(1, 16, 8), demand=1.6, per_engine=0.2)
    run.minutes(6, serving=8)
    assert run.solver.count == 9
    # The ninth engine is still starting when the trainer comes to take less.
    run.steps(1.3)

    decision = run.minutes(10, serving=8)

    assert decision.engines == 9
    assert decision.reason.endswith("7.15 needed; 8 of 9 engines serve")


def test_at_the_minimum_it_holds_and_says_so():
    run = _Run(Solver(8, 16, 8), demand=0.26, per_engine=0.2)

    decision = run.minutes(16, serving=8)

    assert decision.engines == 8
    assert decision.reason.endswith("1.43 needed; at the minimum")


def test_an_engine_lost_while_an_add_is_checked_restarts_the_check():
    run = _Run(Solver(1, 8, 1), demand=0.26, per_engine=0.2)
    run.minutes(6, serving=1)
    run.use = 0.65
    run.minutes(3, serving=2)
    run.minutes(1, serving=1)

    assert run.minutes(5, serving=2).reason == "checking the add from 1"
    assert run.minutes(1, serving=2).reason == "output rose 0.06/s of 0.086/s promised"


def test_a_restarted_rollout_measures_its_output_again():
    solver = Solver(1, 8, 1)
    solver.observe_step(100.0, 0.0, 26)
    load = {"running": 32, "queued": 0, "slots": 32, "use": 1.0}
    solver.decide(0.0, 1, load)
    solver.observe_output(60.0, 500)
    solver.observe_output(120.0, 3)

    reason = solver.decide(120.0, 1, load).reason

    assert reason == "no output measured on the serving engines yet"


def test_at_the_maximum_it_holds_and_says_so():
    run = _Run(Solver(1, 2, 2), demand=0.8, per_engine=0.2)

    assert run.minutes(6, serving=2) == Decision(
        2,
        "the trainer takes 0.8 groups/s, a fully used engine puts out 0.2/s, the engines are 100% "
        "used: 4.40 needed; at the maximum",
        64,
    )


def test_a_fixed_pool_logs_what_auto_mode_would_set_and_moves_nothing(caplog):
    caplog.set_level(logging.INFO, logger="stitch.solver")
    run = _Run(Solver(1, 8, None), demand=0.26, per_engine=0.2)

    decision = run.minutes(6, serving=1)

    assert (decision.engines, run.solver.count, run.solver.move) == (2, None, None)
    assert "shadow: engines 1 -> 2: the trainer takes 0.26 groups/s" in caplog.text
