from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("miles") is None,
    reason="requires the Miles trainer image",
)


def test_basic_logging_does_not_import_training_stack():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class RejectTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'megatron', 'transformer_engine', 'wandb'}:
            raise AssertionError(f'basic logging imported {fullname}')

sys.meta_path.insert(0, RejectTrainingImports())
from miles.utils.logging_utils import configure_logger, configure_logger_raw
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from types import SimpleNamespace
configure_logger_raw('session_server')
configure_logger(SimpleNamespace(save_debug_event_data=None), source=MainProcessIdentity())
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_event_logging_still_writes_valid_events(tmp_path, monkeypatch):
    from miles.utils.audit_utils.event_logger import logger, models
    from miles.utils.audit_utils.process_identity import MainProcessIdentity
    from miles.utils.logging_utils import configure_logger

    monkeypatch.setattr(logger, "_event_logger", None)
    configure_logger(
        SimpleNamespace(save_debug_event_data=str(tmp_path)),
        source=MainProcessIdentity(),
    )
    logger.get_event_logger().log(models.MetricEvent, {"metrics": {"loss": 1.25}})

    event = json.loads((tmp_path / "main.jsonl").read_text())
    assert event["type"] == "metric"
    assert event["source"]["component"] == "main"
    assert event["metrics"] == {"loss": 1.25}


@pytest.mark.parametrize("creation_fails", [False, True])
def test_rollout_manager_initialization_is_awaited(monkeypatch, creation_fails):
    from miles.ray import placement_group

    ready_ref = object()
    manager = SimpleNamespace(ready=SimpleNamespace(remote=lambda: ready_ref))
    actor_type = SimpleNamespace(
        options=lambda **kwargs: SimpleNamespace(remote=lambda args, pg: manager)
    )
    awaited = []

    def get(ref):
        awaited.append(ref)
        if creation_fails:
            raise RuntimeError("session pool failed to initialize")

    monkeypatch.setattr(placement_group, "RolloutManager", actor_type)
    monkeypatch.setattr(placement_group.ray, "get", get)
    args = SimpleNamespace(
        pin_rollout_manager_to_head=False,
        num_rollout=500,
        check_weight_update_equal=False,
        offload_rollout=False,
    )
    if creation_fails:
        with pytest.raises(RuntimeError, match="session pool failed to initialize"):
            placement_group.create_rollout_manager(args, None)
    else:
        assert placement_group.create_rollout_manager(args, None) == (manager, None)
    assert awaited == [ready_ref]
