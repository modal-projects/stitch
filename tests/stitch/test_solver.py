"""Auto mode's next rule in shadow, from the engines' and the rollout's load reports."""

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
    solver = Solver(1, maximum)
    for _ in range(3):
        solver.observe_step(100.0 + waited, waited, 8)
    _rollout(solver, range(15, 9, -1))
    for age in (900, 600):
        solver.decide(NOW - age, {"e": _engine(16)}, count=4, serving=4)
    return solver


def test_a_starved_trainer_with_stressed_engines_adds_the_engines_it_uses_at_the_window_in_band():
    solver = _solver()
    _rollout(solver, range(9, -1, -1), kept=0.6, waiting=5.0)

    decision = solver.decide(NOW, {"e": _engine(16)}, count=4, serving=4)

    assert decision == Decision(
        7, "trainer starving, engines stressed", "prefix kept 0.60", 128, 256, 6
    )


def test_engines_below_their_slots_are_solved_from_what_one_engine_produced():
    solver = Solver(1, 16)
    for _ in range(3):
        solver.observe_step(100.0, 0.0, 15)
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

    solver.decide(NOW - 300, {"e": _engine(100, slots=256)}, count=1, serving=1)
    decision = solver.decide(NOW, {"e": _engine(100, slots=256)}, count=1, serving=1)

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

    decision = solver.decide(NOW, {"e": _engine(16)}, count=8, serving=8)

    assert decision[:3] == (
        8,
        "trainer starving, engines in band",
        "short of what the trainer uses",
    )


def test_a_fed_trainer_with_stressed_engines_holds_and_reports_over_admission():
    solver = _solver()
    _rollout(solver, range(9, -1, -1), kept=0.6)

    decision = solver.decide(NOW, {"e": _engine(16)}, count=4, serving=4)

    assert decision[:3] == (
        4,
        "trainer fed, engines stressed",
        "over-admitted: prefix kept 0.60",
    )


def test_a_starved_trainer_with_idle_engines_holds_as_demand_limited():
    solver = _solver()
    _rollout(solver, range(9, -1, -1), waiting=5.0)

    decision = solver.decide(
        NOW, {"e": _engine(4), "f": _engine(4)}, count=4, serving=2
    )

    assert decision[:3] == (
        4,
        "trainer starving, engines idle",
        "demand-limited: more engines would not feed the trainer",
    )


def test_idle_engines_under_a_fed_trainer_shrink_to_what_the_trainer_uses():
    solver = _solver()
    _rollout(solver, range(9, -1, -1))

    decision = solver.decide(
        NOW, {"e": _engine(4), "f": _engine(4)}, count=8, serving=8
    )

    assert decision[:3] == (7, "trainer fed, engines idle", "idle")


def test_idle_engines_without_a_trainer_shrink_no_lower_than_their_peak_load():
    solver = Solver(1, 16)
    engines = {name: _engine(7) for name in "abcdefghij"}

    decision = solver.decide(NOW, engines, count=10, serving=10)

    assert decision[:3] == (3, "no trainer, engines idle", "idle")


def test_without_a_trainer_queued_engines_add_toward_the_load_they_hold():
    solver = Solver(1, 16)

    decision = solver.decide(
        NOW, {"e": _engine(32, 16), "f": _engine(32, 16)}, count=2, serving=2
    )

    assert decision[:3] == (
        3,
        "no trainer, engines stressed",
        "a queue on 100% of the median engine's reports",
    )


def test_without_a_prefix_signal_retraction_marks_the_engines_stressed():
    solver = Solver(1, 16)

    decision = solver.decide(NOW, {"e": _engine(16, retracted=60)}, count=1, serving=1)

    assert decision.reason == "retraction recomputes 6% of prefill"


def test_reports_short_of_a_window_judge_nothing():
    solver = Solver(1, 16)
    young = {"e": _engine(32, 16)[-5:]}

    assert solver.decide(NOW, young, count=2, serving=2)[:3] == (
        2,
        "no trainer, engines in band",
        "",
    )


def test_an_engine_holds_what_it_runs_while_requests_queue_whatever_the_slots_set():
    solver = Solver(1, 16)

    decision = solver.decide(NOW, {"e": _engine(96, 28, slots=256)}, count=1, serving=1)

    assert (decision.engines, decision.capacity) == (2, 96)
    assert (
        solver.decide(
            NOW, {"e": _engine(24, 8, slots=None)}, count=1, serving=1
        ).capacity
        == 24
    )


def test_the_count_stays_within_the_pools_bounds():
    solver = _solver(maximum=5)
    _rollout(solver, range(9, -1, -1), kept=0.6, waiting=5.0)

    assert solver.decide(NOW, {"e": _engine(16)}, count=4, serving=4).engines == 5


def test_a_decision_is_logged_only_when_its_count_or_cell_changes(caplog):
    solver = _solver()
    _rollout(solver, range(9, -1, -1), kept=0.6, waiting=5.0)

    with caplog.at_level(logging.INFO, logger="stitch.solver"):
        for _ in range(3):
            solver.decide(NOW, {"e": _engine(16)}, count=4, serving=4)

    assert caplog.text.count("shadow: engines 4 -> 7") == 1
