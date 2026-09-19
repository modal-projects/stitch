from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import modal
import pytest

from cookbook.common import process
from cookbook.miles_disagg import trainer_image
from cookbook.miles_disagg.configs import qwen3_4b_math


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_4b_math")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.setattr(modal.App, "cls", lambda *args, **kwargs: lambda cls: cls)
    monkeypatch.setattr(modal.App, "server", lambda *args, **kwargs: lambda cls: cls)
    monkeypatch.setattr(modal, "enter", lambda *args, **kwargs: lambda fn: fn)
    monkeypatch.setattr(modal, "method", lambda *args, **kwargs: lambda fn: fn)
    name = "cookbook.miles_disagg.app"
    previous = sys.modules.pop(name, None)
    try:
        yield importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)
        if previous is not None:
            sys.modules[name] = previous


def test_warm_trainer_entry_never_restores_checkpoint(app, monkeypatch):
    events = []
    monkeypatch.setattr(
        app, "STORE_DEPLOYMENT", SimpleNamespace(bootstrap_credentials=lambda: None)
    )
    monkeypatch.setattr(
        app.ray_cluster, "get_modal_cluster_context", lambda _: (0, "head", "head")
    )
    monkeypatch.setattr(process, "apply_git_patches", lambda *args: None)
    monkeypatch.setattr(process, "start_host_mem_monitor", lambda: None)
    monkeypatch.setattr(
        app.ray_cluster, "start_ray_node", lambda *args, **kwargs: events.append("ray")
    )
    monkeypatch.setattr(
        app,
        "prepare_attempt",
        lambda *args, **kwargs: pytest.fail("warm entry rewound checkpoint"),
    )

    app.Trainer().start_ray()

    assert events == ["ray"]


@pytest.mark.parametrize("reload_fails", [False, True])
def test_training_restores_then_waits_for_every_mount_before_launch(
    app, monkeypatch, reload_fails
):
    events = []
    monkeypatch.setattr(app.launch, "materialize_node_local_yaml", lambda *args: None)
    monkeypatch.setattr(
        app, "prepare_attempt", lambda *args, **kwargs: events.append("restore")
    )
    monkeypatch.setattr(
        app, "train_volumes", {"/stitch": SimpleNamespace(object_id="vo-test")}
    )

    def reload(volumes, *, n_nodes):
        assert volumes == ["vo-test"]
        assert n_nodes == app.miles_cfg.n_train_nodes
        events.append("reload")
        if reload_fails:
            raise RuntimeError("worker mount unavailable")

    class BeforeMilesLaunch(Exception):
        pass

    def pool(*args):
        assert events == ["restore", "reload"]
        events.append("launch")
        raise BeforeMilesLaunch

    monkeypatch.setattr(app.ray_cluster, "reload_volumes_on_nodes", reload)
    monkeypatch.setattr(app, "ModalFlashPool", pool)
    trainer = app.Trainer()
    trainer.rank = 0
    expected = RuntimeError if reload_fails else BeforeMilesLaunch
    with pytest.raises(expected):
        trainer.train(app.miles_cfg.to_payload())
    assert events == (
        ["restore", "reload"] if reload_fails else ["restore", "reload", "launch"]
    )


@pytest.mark.parametrize("resume", [False, True])
def test_resume_restores_checkpoint_scheduler(app, monkeypatch, resume):
    point = (
        SimpleNamespace(
            trainer_checkpoint="/saved/checkpoints",
            rollout_checkpoint="/saved/hf",
            version=2,
        )
        if resume
        else None
    )
    monkeypatch.setattr(app.launch, "materialize_node_local_yaml", lambda *args: None)
    monkeypatch.setattr(app, "prepare_attempt", lambda *args, **kwargs: point)
    monkeypatch.setattr(
        app, "train_volumes", {"/stitch": SimpleNamespace(object_id="vo-test")}
    )
    monkeypatch.setattr(
        app.ray_cluster, "reload_volumes_on_nodes", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        app,
        "ModalFlashPool",
        lambda *args: SimpleNamespace(gateway_url=lambda: "http://rollout"),
    )
    monkeypatch.setattr(app.launch, "resolve_config", lambda *args, **kwargs: None)

    class CapturedCommand(Exception):
        pass

    def capture(cfg):
        assert ("--use-checkpoint-opt-param-scheduler" in cfg.cli_args()) == resume
        if resume:
            assert cfg.load == point.trainer_checkpoint
            assert cfg.hf_checkpoint == point.rollout_checkpoint
        raise CapturedCommand

    monkeypatch.setattr(app, "_build_train_cmd", capture)
    trainer = app.Trainer()
    trainer.rank = 0
    with pytest.raises(CapturedCommand):
        trainer.train(app.miles_cfg.to_payload())


def test_worker_does_not_restore_or_independently_reload(app, monkeypatch):
    events = []
    monkeypatch.setattr(app.launch, "materialize_node_local_yaml", lambda *args: None)
    monkeypatch.setattr(
        app, "prepare_attempt", lambda *args, **kwargs: pytest.fail("worker restore")
    )
    monkeypatch.setattr(
        app.ray_cluster,
        "reload_volumes_on_nodes",
        lambda *args, **kwargs: pytest.fail("worker barrier"),
    )
    monkeypatch.setattr(
        app.ray_cluster,
        "hold_worker_node",
        lambda *args, **kwargs: events.append("hold"),
    )
    trainer = app.Trainer()
    trainer.rank, trainer.master_addr = 1, "head"

    trainer.train(app.miles_cfg.to_payload())

    assert events == ["hold"]


@pytest.mark.parametrize("entrypoint", ["app", "prep_app"])
@pytest.mark.parametrize(
    "recipe", ["qwen3_4b_math", "qwen3_6_35b_a3b_nvfp4", "glm5_3_nvfp4"]
)
def test_maintained_recipes_import_through_deployment_entrypoints(
    monkeypatch, entrypoint, recipe
):
    monkeypatch.setenv("EXPERIMENT_CONFIG", recipe)
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    module = f"cookbook.miles_disagg.{entrypoint}"
    monkeypatch.delitem(sys.modules, module, raising=False)

    app = importlib.import_module(module)

    assert app.exp.__name__ == f"cookbook.miles_disagg.configs.{recipe}"
    assert app.modal_cfg is app.exp.modal
    assert app.miles_cfg is app.exp.miles


@pytest.mark.parametrize("entrypoint", ["app", "prep_app"])
def test_invalid_recipe_fails_before_image_construction(monkeypatch, entrypoint):
    recipe = SimpleNamespace(**vars(qwen3_4b_math))
    recipe.SOURCE_REVISION = "main"
    monkeypatch.setitem(
        sys.modules, "cookbook.miles_disagg.configs.invalid_test_recipe", recipe
    )
    monkeypatch.setenv("EXPERIMENT_CONFIG", "invalid_test_recipe")
    monkeypatch.setenv("RUN_ID", "test-run")
    module = f"cookbook.miles_disagg.{entrypoint}"
    monkeypatch.delitem(sys.modules, module, raising=False)

    def build_image(**_kwargs):
        pytest.fail("invalid recipe reached image construction")

    monkeypatch.setattr(trainer_image, "build_trainer_image", build_image)
    with pytest.raises(ValueError, match="SOURCE_REVISION"):
        importlib.import_module(module)
