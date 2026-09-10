import time

import pytest

from cookbook.miles_disagg.checkpoint_priority import CheckpointIO


def test_old_completion_cannot_release_a_new_delta():
    io = CheckpointIO()
    first, second = io.pause(), io.pause()
    io.resume(first)
    with pytest.raises(TimeoutError, match="upload deadline"):
        with io.operation(time.monotonic() + 0.02):
            pytest.fail("new delta lost priority")
    io.resume(second)
    with io.operation(time.monotonic() + 1):
        assert not io.quiescent
    assert io.quiescent
