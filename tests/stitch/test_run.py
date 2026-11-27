"""A run's config: what the app's containers rebuild it from, and one config per run id."""

import dataclasses
from types import SimpleNamespace

import pytest

import stitch.run
from stitch.control import Clocks
from stitch.run import PoolConfig, Run, RunConfig, create_run
from stitch.sglang import SGLangConfig

CONFIG = RunConfig(
    run_id="r1",
    image="me/sglang:1",
    volumes={"/checkpoints": "checkpoints", "/data": "runs"},
    checkpoint="/checkpoints/base",
    versions="/data/r1/versions",
    engine=SGLangConfig(
        server_args={"--tp-size": "4"}, env={"A": "1"}, pause_mode="retract"
    ),
    pool=PoolConfig(gpu="H200:4", cpu=16, memory=262144),
    image_commands=["pip install my-plugin"],
    clocks=Clocks(patience=60),
)


def test_a_run_config_comes_back_whole_from_its_json():
    assert RunConfig.from_json(CONFIG.to_json()) == CONFIG


def test_auto_mode_refuses_modals_target_concurrency():
    with pytest.raises(ValueError, match="target_concurrency"):
        PoolConfig(gpu="H200:4", autoscale=True, target_concurrency=8)


def test_a_removed_engine_may_finish_its_requests_for_modals_longest_grace():
    assert PoolConfig(gpu="H200:4").exit_grace_period == 3600


def test_requests_route_and_engines_run_where_modal_puts_them_unless_pinned():
    pool = PoolConfig(gpu="H200:4")

    assert (pool.routing_region, pool.compute_region) == ("us-east", None)


def test_auto_mode_leaves_the_count_to_the_control_service():
    with pytest.raises(ValueError, match="target_concurrency"):
        PoolConfig(gpu="H200:4", autoscale=True, target_concurrency=8)


def test_auto_mode_keeps_an_engine_to_measure():
    with pytest.raises(ValueError, match="at least one"):
        PoolConfig(gpu="H200:4", autoscale=True, min_containers=0)


def test_adapters_need_a_mount_of_their_own():
    with pytest.raises(ValueError, match="a mount of their own"):
        dataclasses.replace(CONFIG, adapters="/data/r1/adapters")

    on_their_own = dataclasses.replace(
        CONFIG,
        volumes={**CONFIG.volumes, "/adapters": "adapters"},
        adapters="/adapters/r1",
    )
    assert RunConfig.from_json(on_their_own.to_json()) == on_their_own
    assert on_their_own.adapters_volume == "adapters" and CONFIG.adapters_volume is None


def test_adapters_refuse_attention_dp():
    with pytest.raises(ValueError, match="without attention DP"):
        dataclasses.replace(
            CONFIG,
            volumes={**CONFIG.volumes, "/adapters": "adapters"},
            adapters="/adapters/r1",
            engine=SGLangConfig(server_args={"--tp-size": "4", "--attn-dp-size": "4"}),
        )


def test_a_run_id_already_deployed_with_another_config_is_refused(monkeypatch):
    deployed = dataclasses.replace(CONFIG, clocks=Clocks()).to_json()
    control = SimpleNamespace(run_config=SimpleNamespace(remote=lambda: deployed))
    monkeypatch.setattr(
        stitch.run.modal.Cls, "from_name", lambda app_name, name: lambda: control
    )

    with pytest.raises(ValueError, match="a new config is a new run id"):
        create_run(CONFIG)


def test_a_run_reports_its_control_service_status(monkeypatch):
    status = {"head": None, "serving": {}, "catching_up": 2}
    control = SimpleNamespace(status=SimpleNamespace(remote=lambda: status))
    monkeypatch.setattr(
        stitch.run.modal.Cls,
        "from_name",
        lambda app_name, name: (
            (lambda: control) if (app_name, name) == ("stitch-r1", "Control") else None
        ),
    )

    assert (
        Run(
            url="https://control", generate_url="https://engines", app_name="stitch-r1"
        ).status()
        == status
    )


def test_stopping_a_run_already_stopped_changes_nothing(monkeypatch):
    def lookup(app_name):
        raise stitch.run.modal.exception.NotFoundError(f"App '{app_name}' not found")

    monkeypatch.setattr(stitch.run.modal.App, "lookup", lookup)

    Run(
        url="https://control", generate_url="https://engines", app_name="stitch-r1"
    ).stop()


def test_the_versions_volume_is_the_one_mounted_closest_above_them():
    nested = dataclasses.replace(
        CONFIG, volumes={"/data": "runs", "/data/r1": "r1-versions"}
    )
    assert CONFIG.versions_volume == "runs"
    assert nested.versions_volume == "r1-versions"


def _deployed(monkeypatch, config):
    """create_run finds `config` deployed, so it only looks up the run's URLs."""
    control = SimpleNamespace(
        run_config=SimpleNamespace(remote=config.to_json),
        api=SimpleNamespace(get_web_url=lambda: "https://control"),
    )
    monkeypatch.setattr(
        stitch.run.modal.Cls, "from_name", lambda app_name, name: lambda: control
    )
    monkeypatch.setattr(
        stitch.run.modal.Server,
        "from_name",
        lambda app_name, name: SimpleNamespace(get_url=lambda: "https://engines"),
    )


def test_versions_on_the_checkpoints_volume_warn(monkeypatch, recwarn):
    shared = dataclasses.replace(CONFIG, versions="/checkpoints/r1/versions")
    _deployed(monkeypatch, shared)

    with pytest.warns(UserWarning, match="share the Volume checkpoints"):
        create_run(shared)

    _deployed(monkeypatch, CONFIG)
    recwarn.clear()
    create_run(CONFIG)
    assert not recwarn.list


def test_the_router_pins_requests_to_the_engines_dp_ranks():
    dp = dataclasses.replace(
        CONFIG,
        engine=SGLangConfig(server_args={"--tp-size": "4", "--attn-dp-size": "4"}),
    )
    assert CONFIG.server_options == {"kv_aware_routing": True}
    assert dp.server_options == {
        "kv_aware_routing": True,
        "kv_aware_routing_dp_size": 4,
    }
    same = dataclasses.replace(
        dp,
        pool=dataclasses.replace(
            dp.pool, experimental_options={"kv_aware_routing_dp_size": 4}
        ),
    )
    assert same.server_options == dp.server_options
    other = dataclasses.replace(
        dp,
        pool=dataclasses.replace(
            dp.pool, experimental_options={"kv_aware_routing_dp_size": 2}
        ),
    )
    with pytest.raises(ValueError, match="not the engine's 4 DP ranks"):
        _ = other.server_options
