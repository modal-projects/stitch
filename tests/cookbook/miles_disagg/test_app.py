import importlib
import os
import sys
from types import SimpleNamespace

import pytest

from cookbook.common import storage
from cookbook.common.constants import KERNEL_CACHE_PATH
from cookbook.miles_disagg import trainer_image
from cookbook.miles_disagg.configs import qwen3_4b_math


@pytest.mark.parametrize("entrypoint", ["app", "prep_app"])
@pytest.mark.parametrize(
    "recipe",
    [
        "qwen3_4b_math",
        "qwen3_6_35b_a3b_mimo_code",
        "qwen3_6_35b_a3b_hetero_grpo",
        "qwen3_6_35b_a3b_hetero_icepop",
        "qwen3_6_35b_a3b_hetero_score_centering",
        "qwen3_6_35b_a3b_hetero_score_centering_mis",
        "qwen3_6_35b_a3b_swebench_pro",
        "qwen3_6_35b_a3b_nvfp4",
        "glm5_3_nvfp4",
    ],
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


def test_trainer_mounts_kernel_cache_volume(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_6_35b_a3b_nvfp4")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)

    app = importlib.import_module("cookbook.miles_disagg.app")

    assert app.train_volumes[str(KERNEL_CACHE_PATH)].name == "kernel-cache"


def test_rollout_pool_environment_is_applied_before_engine_start(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_6_35b_a3b_hetero_grpo")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)

    app = importlib.import_module("cookbook.miles_disagg.app")
    pool = next(pool for pool in app.ROLLOUT_POOL_CONFIGS if pool.environment)
    actor = app._RolloutServer()
    actor.rollout_pool_config = pool
    monkeypatch.setattr(
        app,
        "STORE_DEPLOYMENT",
        SimpleNamespace(
            bootstrap_credentials=lambda: None,
            hook_config=lambda _namespace: {"stitch_store_backend": "modal-volume"},
        ),
    )
    monkeypatch.setattr(app, "_boot_checkpoint", lambda *_args: ("checkpoint", 0))
    observed = {}

    def serve_startup(*_args, **_kwargs):
        observed.update({key: os.environ.get(key) for key in pool.environment})

    monkeypatch.setattr(app.server, "serve_startup", serve_startup)

    actor.startup()

    assert observed == pool.environment


def test_multi_view_pool_claims_use_view_scoped_update_directories(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_6_35b_a3b_hetero_grpo")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)

    app = importlib.import_module("cookbook.miles_disagg.app")
    claimed = []
    monkeypatch.setattr(
        "cookbook.common.hooks.claim_pool",
        lambda args, *, boot_version: claimed.append((args, boot_version)),
    )

    app._claim_rollout_pools(
        {"update_weight_disk_dir": str(app.UPDATES_DIR)}, boot_version=7
    )

    assert [args.update_weight_view for args, _ in claimed] == [
        "bf16",
        "fp8",
        "nvfp4",
    ]
    assert [args.update_weight_disk_dir for args, _ in claimed] == [
        str(app.UPDATES_DIR / "bf16"),
        str(app.UPDATES_DIR / "fp8"),
        str(app.UPDATES_DIR / "nvfp4"),
    ]
    assert [version for _, version in claimed] == [7, 7, 7]


def test_multi_view_replica_boots_from_its_latest_complete_export(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_6_35b_a3b_hetero_grpo")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)

    app = importlib.import_module("cookbook.miles_disagg.app")
    fp8 = tmp_path / "hf_checkpoints/weight_v000019/fp8"
    fp8.mkdir(parents=True)
    (fp8 / ".complete").touch()

    store = SimpleNamespace(
        read_pointer=lambda: app.VersionRef("test-run", 20),
        refresh=lambda: None,
    )
    monkeypatch.setattr(app, "RUN_DIR", tmp_path)
    monkeypatch.setattr(
        app, "STORE_DEPLOYMENT", SimpleNamespace(backend=storage.MODAL_VOLUME)
    )
    monkeypatch.setattr(app.storage, "create_store", lambda *_args, **_kwargs: store)

    assert app._boot_checkpoint(
        {"stitch_store_backend": storage.MODAL_VOLUME}, "fp8"
    ) == (
        str(fp8),
        20,
    )


def test_heterogeneous_rollout_sources_name_every_pool_and_its_view(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_6_35b_a3b_hetero_grpo")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)

    app = importlib.import_module("cookbook.miles_disagg.app")

    # Miles matches each sample's reported source against this list, so both must
    # name every pool, and the view after the last colon must be the pool's view.
    assert list(app.ROLLOUT_SOURCES) == list(app.ROLLOUT_SERVER_NAMES)
    for pool in app.ROLLOUT_POOL_CONFIGS:
        source = app.ROLLOUT_SOURCES[pool.name]
        assert source == f"{pool.name}:{pool.weight_view}"
        assert source.rsplit(":", 1)[1] in app.ROLLOUT_WEIGHT_VIEWS
    assert len(set(app.ROLLOUT_SOURCES.values())) == len(app.ROLLOUT_POOL_CONFIGS)


def test_heterogeneous_readiness_is_checked_per_pool(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_6_35b_a3b_hetero_grpo")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)

    app = importlib.import_module("cookbook.miles_disagg.app")
    waits = []
    monkeypatch.setattr(
        app,
        "await_pool_ready",
        lambda pool, **kwargs: waits.append((pool.cls_name, kwargs)),
    )

    latest = app.VersionRef("test-run", 4)
    app.await_rollout_ready(latest=latest)

    assert waits == [
        (
            pool.name,
            {"replica_floor": pool.min_containers, "latest": latest},
        )
        for pool in app.ROLLOUT_POOL_CONFIGS
    ]


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


def _hetero_app(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_CONFIG", "qwen3_6_35b_a3b_hetero_grpo")
    monkeypatch.setenv("RUN_ID", "test-run")
    monkeypatch.delenv("STITCH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("MILES_LOCAL_DIR", raising=False)
    monkeypatch.delitem(sys.modules, "cookbook.miles_disagg.app", raising=False)
    return importlib.import_module("cookbook.miles_disagg.app")


def test_local_checkpoints_mirror_the_run_directory_layout(monkeypatch):
    """Miles saves under a node-local root shaped like the run directory, so each file's
    path relative to it is its Volume path relative to the run directory."""
    app = _hetero_app(monkeypatch)
    cfg = SimpleNamespace(
        save_interval=10, save_hf="hf_checkpoints/weight_v{rollout_id:06d}"
    )

    root = app._set_save_paths(cfg)

    assert root.parent == app.Path("/tmp/stitch-local-checkpoints/test-run")
    assert root.name.startswith("attempt-")
    assert cfg.save == str(root / "checkpoints")
    assert cfg.save_hf == str(root / "hf_checkpoints/weight_v{rollout_id:06d}")
    # A retry writes to a fresh directory.
    assert app._set_save_paths(SimpleNamespace(save_interval=10, save_hf=None)) != root


def test_volume_checkpoints_save_to_the_run_directory(monkeypatch):
    app = _hetero_app(monkeypatch)
    monkeypatch.setattr(app.modal_cfg, "trainer_local_checkpoint_dir", None)
    cfg = SimpleNamespace(
        save_interval=10, save_hf="hf_checkpoints/weight_v{rollout_id:06d}"
    )

    assert app._set_save_paths(cfg) is None
    assert cfg.save == str(app.RUN_DIR / "checkpoints")
    assert cfg.save_hf == str(app.RUN_DIR / "hf_checkpoints/weight_v{rollout_id:06d}")


def test_runs_without_saves_write_nothing(monkeypatch):
    app = _hetero_app(monkeypatch)
    cfg = SimpleNamespace(save_interval=None, save_hf="hf_checkpoints/x")

    assert app._set_save_paths(cfg) is None
    assert cfg.save is None and cfg.save_hf is None
