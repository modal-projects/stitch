from __future__ import annotations

from types import SimpleNamespace

import pytest

from cookbook.common import launch as common_launch
from cookbook.miles_disagg import launch


def _resumable() -> SimpleNamespace:
    return SimpleNamespace(
        save_interval=20,
        save_hf="hf_checkpoints/weight_v{rollout_id:06d}",
    )


def _launch(
    monkeypatch,
    *,
    argv_resume_from: str | None,
    miles: object | None = None,
    pool_reachable: bool = False,
) -> tuple[dict, list[str]]:
    """Run launch.main() with the config and app modules stubbed out."""
    spawned: dict = {}
    prints: list[str] = []
    exp = SimpleNamespace(miles=miles if miles is not None else _resumable())
    run = SimpleNamespace(APP_NAME="app-run")

    def import_module(name: str):
        return exp if name.endswith(".configs.test") else run

    def spawn(actual_run):
        spawned["run"] = actual_run
        spawned["run_id"] = launch.os.environ["RUN_ID"]
        return SimpleNamespace(object_id="fc-1")

    monkeypatch.setenv("EXPERIMENT_CONFIG", "test")
    monkeypatch.setattr(launch.importlib, "import_module", import_module)
    monkeypatch.setattr(common_launch, "deploy_pool_and_spawn", spawn)
    monkeypatch.setattr(common_launch, "spawn_on_pool", spawn)
    monkeypatch.setattr(common_launch, "pool_reachable", lambda _run: pool_reachable)
    monkeypatch.setattr(
        launch,
        "_cancel_recorded_trainer_call",
        lambda run: spawned.update(cancelled=True),
    )
    monkeypatch.setattr(
        launch,
        "_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(resume_from=argv_resume_from)
        ),
    )
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: prints.append(" ".join(map(str, args))),
    )

    launch.main()
    return spawned, prints


def test_fresh_launch_mints_a_run_id(monkeypatch) -> None:
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.setattr(
        launch.uuid, "uuid4", lambda: SimpleNamespace(hex="feedc0dedeadbeef")
    )

    spawned, _prints = _launch(monkeypatch, argv_resume_from=None)

    assert spawned["run_id"] == "feedc0de"
    assert spawned["run"].APP_NAME == "app-run"
    assert "cancelled" not in spawned


def test_fresh_launch_honors_explicit_run_id(monkeypatch) -> None:
    monkeypatch.setenv("RUN_ID", "hero-run")

    spawned, _prints = _launch(monkeypatch, argv_resume_from=None)

    assert spawned["run_id"] == "hero-run"


def test_fresh_launch_refuses_to_roll_over_a_live_pool(monkeypatch) -> None:
    # An explicitly reused RUN_ID with a live pool is a resume, not a fresh
    # deploy: redeploying would silently replace the running fleet.
    monkeypatch.setenv("RUN_ID", "hero-run")

    with pytest.raises(SystemExit, match="resume it with --resume-from"):
        _launch(monkeypatch, argv_resume_from=None, pool_reachable=True)


def test_resume_cancels_the_recorded_call_and_reuses_the_run_id(monkeypatch) -> None:
    monkeypatch.delenv("RUN_ID", raising=False)

    spawned, _prints = _launch(monkeypatch, argv_resume_from="old-run")

    assert spawned["run_id"] == "old-run"
    assert spawned["cancelled"] is True


def test_resume_requires_a_resumable_config(monkeypatch) -> None:
    with pytest.raises(ValueError, match="requires save_hf"):
        _launch(
            monkeypatch,
            argv_resume_from="old-run",
            miles=SimpleNamespace(save_interval=20, save_hf=None),
        )


def test_fresh_launch_warns_but_proceeds_without_a_resumable_config(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RUN_ID", "smoke")

    spawned, prints = _launch(
        monkeypatch,
        argv_resume_from=None,
        miles=SimpleNamespace(save_interval=None, save_hf=None),
    )

    assert spawned["run_id"] == "smoke"
    assert any("not resumable" in line for line in prints)


def _patch_modal_call(monkeypatch, call: object) -> None:
    import modal

    monkeypatch.setattr(
        modal.Volume, "from_name", classmethod(lambda *_args, **_kwargs: object())
    )
    monkeypatch.setattr(
        modal.FunctionCall, "from_id", classmethod(lambda *_args, **_kwargs: call)
    )


def test_cancel_waits_until_the_call_settles(monkeypatch) -> None:
    from modal.types import InputStatus

    events: list[str] = []
    graphs = iter(
        [
            [SimpleNamespace(status=InputStatus.PENDING, children=[])],
            [SimpleNamespace(status=InputStatus.FAILURE, children=[])],
        ]
    )

    class _Call:
        def cancel(self, *, terminate_containers: bool) -> None:
            events.append(f"cancel(terminate={terminate_containers})")

        def get_call_graph(self):
            events.append("graph")
            return next(graphs)

    run = SimpleNamespace(exp=SimpleNamespace(EXPERIMENT_VOLUME_NAME="runs"))
    monkeypatch.setenv("RUN_ID", "old-run")
    monkeypatch.setattr(launch.time, "sleep", lambda _s: None)
    _patch_modal_call(monkeypatch, _Call())
    monkeypatch.setattr(launch, "read_trainer_call", lambda _volume, _run_id: "fc-old")

    launch._cancel_recorded_trainer_call(run)

    assert events == ["cancel(terminate=True)", "graph", "graph"]


def test_cancel_is_a_noop_before_the_first_spawn(monkeypatch) -> None:
    run = SimpleNamespace(exp=SimpleNamespace(EXPERIMENT_VOLUME_NAME="runs"))
    monkeypatch.setenv("RUN_ID", "old-run")
    _patch_modal_call(monkeypatch, None)
    monkeypatch.setattr(launch, "read_trainer_call", lambda _volume, _run_id: None)

    launch._cancel_recorded_trainer_call(run)  # nothing recorded: no cancel
