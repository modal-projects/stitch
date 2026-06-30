from __future__ import annotations

import unittest
from argparse import Namespace
from unittest import mock

from cookbook import trainer_helpers


def _cfg(*, async_mode: bool, model_script: str | None) -> Namespace:
    cfg = Namespace(async_mode=async_mode, trainer_model_script=model_script)
    cfg.cli_args = lambda: ["--foo", "bar baz"]  # a value with a space to prove quoting
    return cfg


class BuildTrainCmdTest(unittest.TestCase):
    def test_plain_python_invocation_without_model_script(self) -> None:
        cmd = trainer_helpers.build_train_cmd(
            _cfg(async_mode=False, model_script=None), "/root/slime", model_script_attr="trainer_model_script"
        )
        self.assertEqual(cmd, "python3 /root/slime/train.py --foo 'bar baz'")

    def test_async_mode_selects_train_async(self) -> None:
        cmd = trainer_helpers.build_train_cmd(
            _cfg(async_mode=True, model_script=None), "/root/slime", model_script_attr="trainer_model_script"
        )
        self.assertTrue(cmd.startswith("python3 /root/slime/train_async.py "))

    def test_sources_model_script_and_passes_model_args(self) -> None:
        cmd = trainer_helpers.build_train_cmd(
            _cfg(async_mode=False, model_script="scripts/qwen.sh"),
            "/root/slime",
            model_script_attr="trainer_model_script",
        )
        # bash -c wrapping a `source <script> && python3 ... ${MODEL_ARGS[@]} ...`.
        self.assertTrue(cmd.startswith("bash -c "))
        self.assertIn("source /root/slime/scripts/qwen.sh", cmd)
        self.assertIn("${MODEL_ARGS[@]}", cmd)


class SmokeFlashPoolTest(unittest.TestCase):
    def test_warm_floor_checks_containers_then_gateway(self) -> None:
        server_info = {"current_version": 3}
        completion = {"weight_version_start": 3, "weight_version_end": 3}
        with mock.patch.object(trainer_helpers, "resolve_flash_gateway_url", return_value="http://gw"), \
            mock.patch.object(trainer_helpers, "discover_flash_targets", return_value=["http://c1"]), \
            mock.patch.object(trainer_helpers, "_get_json", return_value=server_info), \
            mock.patch.object(trainer_helpers, "_post_json", return_value=completion):
            trainer_helpers.smoke_flash_pool(
                app_name="a", cls_name="Server", model_name="m", weight_version=3,
                expect_min_containers=1, timeout_seconds=5, wake_on_demand=False,
            )

    def test_warm_floor_version_ahead_raises_immediately(self) -> None:
        with mock.patch.object(trainer_helpers, "resolve_flash_gateway_url", return_value="http://gw"), \
            mock.patch.object(trainer_helpers, "discover_flash_targets", return_value=["http://c1"]), \
            mock.patch.object(trainer_helpers, "_get_json", return_value={"current_version": 9}):
            with self.assertRaises(trainer_helpers.VersionAheadError):
                trainer_helpers.smoke_flash_pool(
                    app_name="a", cls_name="Server", model_name="m", weight_version=3,
                    expect_min_containers=1, timeout_seconds=5, wake_on_demand=False,
                )

    def test_wake_on_demand_completes_then_confirms_containers(self) -> None:
        completion = {"weight_version_start": 2, "weight_version_end": 2}
        with mock.patch.object(trainer_helpers, "resolve_flash_gateway_url", return_value="http://gw"), \
            mock.patch.object(trainer_helpers, "discover_flash_targets", return_value=["http://c1"]), \
            mock.patch.object(trainer_helpers, "_post_json", return_value=completion), \
            mock.patch.object(trainer_helpers, "_get_json", return_value={"current_version": 2}):
            trainer_helpers.smoke_flash_pool(
                app_name="a", cls_name="Server", model_name="m", weight_version=2,
                expect_min_containers=0, timeout_seconds=5, wake_on_demand=True,
            )


if __name__ == "__main__":
    unittest.main()
