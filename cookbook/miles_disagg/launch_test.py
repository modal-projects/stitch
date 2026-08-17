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
) -> tuple[dict, list[str]]:
    """Run launch.main() with the config and app modules stubbed out."""
    spawned: dict = {}
    prints: list[str] = []
    exp = SimpleNamespace(miles=miles if miles is not None else _resumable())
    run = SimpleNamespace(APP_NAME="app-run")

    def import_module(name: str):
        return exp if name.endswith(".configs.test") else run

    def ensure(actual_run):
        spawned["run"] = actual_run
        spawned["run_id"] = launch.os.environ["RUN_ID"]
        return SimpleNamespace(object_id="fc-1")

    monkeypatch.setenv("EXPERIMENT_CONFIG", "test")
    monkeypatch.setattr(launch.importlib, "import_module", import_module)
    monkeypatch.setattr(common_launch, "spawn_on_pool", ensure)
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


def test_fresh_launch_honors_explicit_run_id(monkeypatch) -> None:
    monkeypatch.setenv("RUN_ID", "hero-run")

    spawned, _prints = _launch(monkeypatch, argv_resume_from=None)

    assert spawned["run_id"] == "hero-run"


def test_resume_reuses_the_run_id(monkeypatch) -> None:
    monkeypatch.delenv("RUN_ID", raising=False)

    spawned, _prints = _launch(monkeypatch, argv_resume_from="old-run")

    assert spawned["run_id"] == "old-run"


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
