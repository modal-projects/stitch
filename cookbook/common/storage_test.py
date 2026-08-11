from __future__ import annotations

from pathlib import Path

import pytest

from cookbook.common.storage import (
    MODAL_VOLUME,
    S3,
    StoreDeployment,
    create_store,
)
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.stores.s3 import S3Store


def test_volume_is_the_default_deployment() -> None:
    deployment = StoreDeployment.from_environment(
        {"STITCH_S3_SECRET_NAME": "ignored-with-volume-backend"}
    )

    assert deployment.backend == MODAL_VOLUME
    assert deployment.image_environment == {"STITCH_STORE_BACKEND": MODAL_VOLUME}
    assert deployment.extra_packages == ()
    assert deployment.hook_config("experiment/run-a", {}) == {
        "stitch_store_backend": MODAL_VOLUME
    }


def test_s3_deployment_separates_image_and_secret_settings() -> None:
    deployment = StoreDeployment.from_environment(
        {
            "STITCH_STORE_BACKEND": S3,
            "STITCH_S3_SECRET_NAME": "stitch-s3",
        }
    )

    assert deployment.image_environment == {
        "STITCH_STORE_BACKEND": S3,
        "STITCH_S3_SECRET_NAME": "stitch-s3",
    }
    assert deployment.extra_packages == ("boto3",)
    assert deployment.hook_config(
        "experiment/run-a",
        {
            "S3_ROOT": "s3://bucket/prefix/",
            "S3_ENDPOINT_URL": "https://s3.example.test",
        },
    ) == {
        "stitch_store_backend": S3,
        "stitch_s3_root": "s3://bucket/prefix/experiment/run-a",
        "stitch_s3_endpoint_url": "https://s3.example.test",
    }


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"STITCH_STORE_BACKEND": "other"}, "STITCH_STORE_BACKEND"),
        ({"STITCH_STORE_BACKEND": S3}, "STITCH_S3_SECRET_NAME"),
    ],
)
def test_rejects_an_invalid_deployment(
    environment: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        StoreDeployment.from_environment(environment)


def test_s3_hook_config_requires_a_root() -> None:
    deployment = StoreDeployment(S3, "stitch-s3")

    with pytest.raises(ValueError, match="S3_ROOT"):
        deployment.hook_config("experiment/run-a", {})


def test_s3_bootstraps_modal_web_identity(tmp_path: Path) -> None:
    deployment = StoreDeployment(S3, "stitch-s3")
    environment = {
        "MODAL_IDENTITY_TOKEN": "short-lived-token",
        "AWS_ROLE_ARN": "arn:aws:iam::123456789012:role/stitch",
    }
    token_path = tmp_path / "token"

    deployment.bootstrap_credentials(environment, token_path)

    assert token_path.read_text() == "short-lived-token"
    assert token_path.stat().st_mode & 0o777 == 0o600
    assert environment["AWS_WEB_IDENTITY_TOKEN_FILE"] == str(token_path)


def test_create_store_selects_the_backend(tmp_path: Path) -> None:
    volume = create_store(
        MODAL_VOLUME,
        local_root=tmp_path / "volume",
        run_id="run-a",
    )
    s3 = create_store(
        S3,
        local_root=tmp_path / "cache",
        run_id="run-a",
        s3_root="s3://bucket/experiment/run-a",
    )

    assert isinstance(volume, ModalVolumeStore)
    assert isinstance(s3, S3Store)
    assert s3.cache_dir == tmp_path / "cache"
    assert s3.run_id == "run-a"


def test_s3_trainer_updates_are_node_local() -> None:
    run_dir = Path("/stitch/run-a")

    assert StoreDeployment(MODAL_VOLUME).updates_dir(run_dir) == run_dir / "updates"
    assert StoreDeployment(S3, "stitch-s3").updates_dir(run_dir) == Path(
        "/tmp/stitch-publications/run-a/updates"
    )
