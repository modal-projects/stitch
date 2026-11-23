"""Auto mode's engine count from each training step's report."""

from stitch.autoscale import Autoscaler

# The trainer uses 8 groups per step and trains for 25 s.
CONSUMED, TRAIN = 8, 25.0


class _Pool:
    """Modal runs the autoscaler's count from the next step; the engines make `rate(engines)` groups a second."""

    def __init__(
        self,
        rate,
        count: int,
        minimum=1,
        maximum: int | None = 16,
        paced=False,
        window_held=0.0,
    ):
        self.autoscaler = Autoscaler(minimum, maximum, count)
        self.rate, self.paced, self.window_held, self.now = (
            rate,
            paced,
            window_held,
            0.0,
        )

    def step(self) -> int:
        """The engines that ran the step."""
        engines = self.autoscaler.count
        self.autoscaler.observe_serving(self.now, engines)
        rate = self.rate(engines)
        if rate * TRAIN < CONSUMED:
            seconds, held, produced = CONSUMED / rate, 0.0, CONSUMED
        elif self.paced:
            # Held back by a full buffer for the time the trainer did not need, and by any tight window.
            seconds, produced = TRAIN, CONSUMED
            held = TRAIN - CONSUMED / rate + self.window_held
        else:
            seconds, held, produced = TRAIN, 0.0, round(rate * TRAIN)
        self.now += seconds
        self.autoscaler.observe_step(
            self.now, seconds, seconds - TRAIN, held, CONSUMED, produced
        )
        return engines

    def steps(self, count: int) -> list[int]:
        return [self.step() for _ in range(count)]


def linear(engines: int) -> float:
    # Eight engines make exactly what the trainer uses.
    return 0.04 * engines


def test_a_starved_trainer_moves_the_count_halfway_to_its_need_until_it_is_fed():
    pool = _Pool(linear, count=2)

    assert pool.steps(15) == [2] * 3 + [5] * 3 + [7] * 3 + [8] * 6


def test_a_surplus_moves_the_count_halfway_down_until_it_is_used():
    pool = _Pool(linear, count=16)

    assert pool.steps(15) == [16] * 3 + [12] * 3 + [10] * 3 + [9] * 3 + [8] * 3


def test_a_rise_that_raised_nothing_is_undone_and_not_retried():
    def client_bound(engines):
        # The rollout keeps three engines' worth of requests in flight.
        return 0.04 * min(engines, 3)

    pool = _Pool(client_bound, count=3)

    assert pool.steps(15) == [3] * 3 + [6] * 3 + [3] * 9


def test_a_rise_past_a_small_shortfall_pays_by_feeding_the_trainer():
    # One engine makes 0.3 groups a second, the trainer uses 0.32, and two make no more than that.
    pool = _Pool(lambda engines: 0.3 * engines, count=1, paced=True)

    assert pool.steps(9) == [1] * 3 + [2] * 6


def test_a_rollout_held_back_by_a_full_buffer_moves_the_count_halfway_down():
    pool = _Pool(linear, count=12, paced=True)

    assert pool.steps(15) == [12] * 3 + [10] * 3 + [9] * 3 + [8] * 6


def test_a_fall_that_starved_the_trainer_is_not_retried():
    # A tight window holds the producer back even with exactly the engines the trainer needs.
    pool = _Pool(linear, count=8, paced=True, window_held=10.0)

    assert pool.steps(18) == [8] * 3 + [6] * 3 + [7] * 3 + [8] * 9


def test_a_trainer_waiting_at_the_maximum_is_logged(caplog):
    with caplog.at_level("INFO", logger="stitch.autoscale"):
        _Pool(linear, count=2, maximum=2).steps(3)

    assert "the trainer waits with the maximum 2 engines" in caplog.text


def test_once_production_moves_a_stalled_count_rises_again():
    pool = _Pool(lambda engines: 0.04 * min(engines, 3), count=3)
    pool.steps(9)

    pool.rate = lambda engines: 0.06 * min(engines, 6)

    assert pool.steps(6)[-1] > 3


def test_the_count_stays_within_the_run_s_bounds():
    assert _Pool(linear, count=2, maximum=4).steps(9)[-1] == 4
    assert _Pool(linear, count=16, minimum=12).steps(9)[-1] == 12


class _Steps:
    """An autoscaler fed 100 s steps directly."""

    def __init__(self, serving: int):
        self.autoscaler = Autoscaler(1, 16, serving)
        self.now = 0.0
        self.autoscaler.observe_serving(self.now, serving)

    def step(self, waited=0.0, produced=8, serving=None) -> int:
        if serving is not None:
            self.autoscaler.observe_serving(self.now + 50, serving)
        self.now += 100
        self.autoscaler.observe_step(self.now, 100, waited, 0.0, 8, produced)
        return self.autoscaler.count


def test_production_wobbling_a_group_around_the_trainers_use_holds_the_count():
    steps = _Steps(serving=2)

    assert [steps.step(produced=p) for p in (7, 7, 7, 9, 7, 7, 7)][-1] == 2


def test_one_step_that_disagrees_holds_the_count():
    steps = _Steps(serving=2)

    assert [steps.step(waited=w) for w in (75, 0, 75)][-1] == 2


def test_the_count_moves_toward_the_smallest_need_of_the_steps():
    steps = _Steps(serving=2)

    # Needs 8, 4 and 8.
    assert [steps.step(waited=w) for w in (75, 50, 75)][-1] == 3


def test_a_step_the_serving_engines_changed_in_does_not_count():
    steps = _Steps(serving=4)
    steps.step(produced=32)
    steps.step(produced=32, serving=5)

    assert [steps.step(produced=32) for _ in range(2)][-1] == 4
    # Need 1.25 on five serving engines: halfway down from five.
    assert steps.step(produced=32) == 3


def test_a_step_that_produced_nothing_says_nothing():
    steps = _Steps(serving=2)

    assert [steps.step(waited=50, produced=0) for _ in range(3)][-1] == 2
