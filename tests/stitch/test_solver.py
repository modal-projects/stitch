"""Auto mode's engine count, from the engines' and the rollout's load reports, each move checked before the next."""

import logging

from stitch.solver import Decision, Solver

NOW = 3600.0


def _engine(running, queued=0, *, slots: int | None = 32, prompt=1000, retracted=0):
    return [
        (
            NOW - 30 * age,
            {
                "running": running,
                "queued": queued,
                "prompt_tokens": prompt * (40 - age),
                "retracted_tokens": retracted * (40 - age),
                "running_slots": slots,
            },
        )
        for age in range(40, -1, -1)
    ]


def _rollout(solver, ages, *, kept=1.0, waiting=0.0):
    for age in ages:
        solver.observe_rollout_load(
            NOW - 60 * age,
            {
                "finished_groups": 3,
                "turn_seconds": 7200.0,
                "session_seconds": 14400.0,
                "prefix_tokens": 10_000,
                "uncached_prefix_tokens": round(10_000 * (1 - kept)),
                "waiting": waiting,
            },
        )


def _solver(*, waited=0.0, maximum=16) -> Solver:
    """A trainer that uses 8 groups a 100 s step, and a window in band from 15 to 10 minutes ago."""
    solver = Solver(1, maximum, None)
    for _ in range(3):
        solver.observe_step(100.0 + waited, waited, 0.0, 8, 8)
    _rollout(solver, range(15, 9, -1))
    for age in (900, 600):
        solver.decide(NOW - age, {"e": _engine(16)}, 4)
    return solver


def test_a_starved_trainer_with_stressed_engines_adds_the_engines_it_uses_at_the_window_in_band():
    solver = _solver()
    _rollout(solver, range(9, -1, -1), kept=0.6, waiting=5.0)

    decision = solver.decide(NOW, {"e": _engine(16)}, 4)

    assert decision == Decision(
        7, "trainer starving, engines stressed", "prefix kept 0.60", 128, 256, 6
    )


def test_engines_below_their_slots_are_solved_from_what_one_engine_produced():
    solver = Solver(1, 16, None)
    for _ in range(3):
        solver.observe_step(100.0, 0.0, 0.0, 15, 15)
    for age in range(10, -1, -1):
        solver.observe_rollout_load(
            NOW - 60 * age,
            {
                "finished_groups": 6,
                "turn_seconds": 6000.0,
                "session_seconds": 6000.0,
                "waiting": 5.0,
            },
        )

    solver.decide(NOW - 300, {"e": _engine(100, slots=256)}, 1)
    decision = solver.decide(NOW, {"e": _engine(100, slots=256)}, 1)

    assert decision == Decision(
        2,
        "trainer starving, engines in band",
        "short of what the trainer uses",
        256,
        256,
        1,
    )


def test_a_window_across_a_change_in_serving_engines_teaches_nothing():
    solver = _solver()
    _rollout(solver, range(9, -1, -1), waiting=5.0)

    decision = solver.decide(NOW, {"e": _engine(16)}, 8)

    assert decision[:3] == (
        8,
        "trainer starving, engines in band",
        "short of what the trainer uses",
    )


def test_a_fed_trainer_with_stressed_engines_holds_and_reports_over_admission():
    solver = _solver()
    _rollout(solver, range(9, -1, -1), kept=0.6)

    decision = solver.decide(NOW, {"e": _engine(16)}, 4)

    assert decision[:3] == (
        4,
        "trainer fed, engines stressed",
        "over-admitted: prefix kept 0.60",
    )


def test_a_starved_trainer_with_idle_engines_holds_as_demand_limited():
    solver = _solver()
    _rollout(solver, range(9, -1, -1), waiting=5.0)

    decision = solver.decide(NOW, {"e": _engine(4), "f": _engine(4)}, 2)

    assert decision[:3] == (
        2,
        "trainer starving, engines idle",
        "demand-limited: more engines would not feed the trainer",
    )


def test_idle_engines_under_a_fed_trainer_shrink_to_what_the_trainer_uses():
    solver = _solver()
    _rollout(solver, range(9, -1, -1))

    decision = solver.decide(NOW, {"e": _engine(4), "f": _engine(4)}, 8)

    assert decision[:3] == (7, "trainer fed, engines idle", "idle")


def test_idle_engines_without_a_trainer_shrink_no_lower_than_their_peak_load():
    solver = Solver(1, 16, None)
    engines = {name: _engine(7) for name in "abcdefghij"}

    decision = solver.decide(NOW, engines, 10)

    assert decision[:3] == (3, "no trainer, engines idle", "idle")


def test_without_a_trainer_queued_engines_add_toward_the_load_they_hold():
    solver = Solver(1, 16, None)

    decision = solver.decide(NOW, {"e": _engine(32, 16), "f": _engine(32, 16)}, 2)

    assert decision[:3] == (
        3,
        "no trainer, engines stressed",
        "a queue on 100% of the median engine's reports",
    )


def test_without_a_prefix_signal_retraction_marks_the_engines_stressed():
    solver = Solver(1, 16, None)

    decision = solver.decide(NOW, {"e": _engine(16, retracted=60)}, 1)

    assert decision.reason == "retraction recomputes 6% of prefill"


def test_reports_short_of_a_window_judge_nothing():
    solver = Solver(1, 16, None)
    young = {"e": _engine(32, 16)[-5:]}

    assert solver.decide(NOW, young, 2)[:3] == (
        2,
        "no trainer, engines in band",
        "",
    )


def test_without_reported_slots_an_engine_holds_what_it_ran_while_requests_queued():
    solver = Solver(1, 16, None)

    decision = solver.decide(NOW, {"e": _engine(96, 28, slots=None)}, 1)

    assert (decision.engines, decision.capacity) == (2, 96)
    reported = solver.decide(NOW, {"e": _engine(96, 28, slots=96)}, 1)
    assert reported.capacity == 96


def test_the_count_stays_within_the_pools_bounds():
    solver = _solver(maximum=5)
    _rollout(solver, range(9, -1, -1), kept=0.6, waiting=5.0)

    decision = solver.decide(NOW, {"e": _engine(16)}, 4)

    assert (decision.engines, decision.reason) == (
        5,
        "prefix kept 0.60; at the maximum",
    )


def test_a_decision_is_logged_only_when_its_count_or_cell_changes(caplog):
    solver = _solver()
    _rollout(solver, range(9, -1, -1), kept=0.6, waiting=5.0)

    with caplog.at_level(logging.INFO, logger="stitch.solver"):
        for _ in range(3):
            solver.decide(NOW, {"e": _engine(16)}, 4)

    assert caplog.text.count("shadow: engines 4 -> 7") == 1


def _engines_at(t: float, count: int, running: int = 16) -> dict:
    reports = [
        (
            t - 30 * age,
            {
                "running": running,
                "queued": 0,
                "prompt_tokens": 0,
                "retracted_tokens": 0,
                "running_slots": 32,
            },
        )
        for age in range(40, -1, -1)
    ]
    return {f"e{index}": reports for index in range(count)}


def _minutes(solver, first: int, last: int, *, groups: int, waiting: float) -> None:
    for minute in range(first, last + 1):
        solver.observe_rollout_load(
            60.0 * minute,
            {
                "finished_groups": groups,
                "turn_seconds": 600.0,
                "session_seconds": 1200.0,
                "waiting": waiting,
            },
        )


def _starved_at_one_engine() -> Solver:
    """One engine finishing 3 groups a minute while the trainer, waiting, consumes 0.08 a second: 2 engines needed."""
    solver = Solver(1, 16, 1)
    for _ in range(3):
        solver.observe_step(100.0, 0.0, 0.0, 8, 8)
    _minutes(solver, 0, 10, groups=3, waiting=5.0)
    solver.decide(300.0, _engines_at(300.0, 1), 1)
    return solver


def test_a_move_waits_for_its_engines_then_keeps_the_count_its_production_backs():
    solver = _starved_at_one_engine()

    assert solver.decide(600.0, _engines_at(600.0, 1), 1)[:3] == (
        2,
        "trainer starving, engines in band",
        "short of what the trainer uses",
    )
    assert (
        solver.decide(660.0, _engines_at(660.0, 1), 1).reason == "1 of 2 engines serve"
    )
    assert (
        solver.decide(720.0, _engines_at(720.0, 2), 2).reason
        == "checking the move from 1"
    )
    _minutes(solver, 11, 18, groups=6, waiting=0.0)

    kept = solver.decide(1080.0, _engines_at(1080.0, 2), 2)

    assert (kept.engines, kept.reason) == (
        2,
        "production rose 0.05/s against 0.05/s promised",
    )
    assert solver.count == 2


def test_an_add_production_does_not_back_is_undone_and_not_retried_while_production_stays():
    solver = _starved_at_one_engine()
    solver.decide(600.0, _engines_at(600.0, 1), 1)
    solver.decide(720.0, _engines_at(720.0, 2), 2)
    _minutes(solver, 11, 18, groups=3, waiting=5.0)

    undone = solver.decide(1080.0, _engines_at(1080.0, 2), 2)
    assert (undone.engines, undone.reason) == (
        1,
        "undone: production rose 0/s against 0.05/s: rollout-limited",
    )
    assert solver.decide(1140.0, _engines_at(1140.0, 1), 1).reason == "back at 1"
    _minutes(solver, 19, 25, groups=3, waiting=5.0)

    held = solver.decide(1500.0, _engines_at(1500.0, 1), 1)

    assert (held.engines, held.reason) == (
        1,
        "short of what the trainer uses; rollout-limited at 1",
    )


def test_a_shrink_that_starves_the_trainer_is_undone_and_not_tried_again():
    solver = Solver(1, 16, 4)
    for _ in range(3):
        solver.observe_step(100.0, 0.0, 0.0, 1, 1)
    _minutes(solver, 0, 20, groups=3, waiting=0.0)
    solver.decide(600.0, _engines_at(600.0, 4), 4)
    solver.decide(1200.0, _engines_at(1200.0, 4, running=16), 4)

    shrink = solver.decide(1260.0, _engines_at(1260.0, 4, running=4), 4)
    assert (shrink.engines, shrink.reason) == (1, "idle")
    solver.decide(1320.0, _engines_at(1320.0, 1, running=4), 1)
    _minutes(solver, 21, 27, groups=3, waiting=5.0)

    undone = solver.decide(1680.0, _engines_at(1680.0, 1, running=4), 1)
    assert (undone.engines, undone.reason) == (
        4,
        "undone: the shrink stressed the engines or starved the trainer",
    )
    solver.decide(1740.0, _engines_at(1740.0, 4, running=4), 4)
    _minutes(solver, 28, 45, groups=3, waiting=0.0)

    refused = solver.decide(2700.0, _engines_at(2700.0, 4, running=4), 4)

    assert (refused.engines, refused.reason) == (
        4,
        "idle; shrinking to 1 stressed or starved it before",
    )


def test_one_minute_of_waiting_does_not_move_the_count():
    solver = Solver(1, 16, 1)
    for _ in range(3):
        solver.observe_step(100.0, 0.0, 0.0, 8, 8)
    _minutes(solver, 0, 9, groups=3, waiting=0.0)
    _minutes(solver, 10, 10, groups=3, waiting=5.0)
    solver.decide(300.0, _engines_at(300.0, 1), 1)

    decision = solver.decide(600.0, _engines_at(600.0, 1), 1)

    assert (decision.engines, decision.cell) == (1, "trainer fed, engines in band")


def test_without_the_rollouts_load_the_steps_tell_what_one_engine_produced():
    solver = Solver(1, 16, 1)
    for _ in range(3):
        solver.observe_step(100.0, 50.0, 0.0, 8, 4)
    solver.decide(300.0, _engines_at(300.0, 1), 1)

    decision = solver.decide(600.0, _engines_at(600.0, 1), 1)

    assert decision[:3] == (
        4,
        "trainer starving, engines in band",
        "short of what the trainer uses",
    )
