from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from cookbook.standalone_rollouts.slime import hooks
from stitch.protocol import RolloutPoolState, RolloutReplicaState


class ShimConfigTest(unittest.TestCase):
    def test_uses_rollout_endpoint_as_base_url_fallback_and_zstd_defaults(self) -> None:
        args = Namespace(rollout_endpoint_url="http://provider/")

        with mock.patch.dict("os.environ", {}, clear=True):
            cfg = hooks.ShimConfig.from_env(args)

        self.assertEqual(cfg.api_base_url, "http://provider")
        # The disk-delta branch ships zstd compression + xxh3-128 checksums.
        self.assertEqual(cfg.compression_format, "zstd")
        self.assertEqual(cfg.checksum_format, "xxh3-128")


class AnnounceAndWaitTest(unittest.TestCase):
    def test_copies_to_transport_then_announces_on_rank_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = root / "local" / "weight_v000003"
            version_dir.mkdir(parents=True)
            (version_dir / "model-00000-of-00001.safetensors").write_bytes(b"delta")
            (version_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            transport = root / "transport"

            posted: list[tuple[str, str]] = []

            def fake_post(cfg, *, identity, previous_identity) -> None:
                posted.append((identity, previous_identity))

            ready = RolloutPoolState(
                replicas=[
                    RolloutReplicaState(readiness=True, current_snapshot_identity="weight_v000003")
                ]
            )
            args = Namespace(api_shim_base_url="http://provider", api_shim_transport_root=str(transport))
            with mock.patch.object(hooks, "_distributed_rank", return_value=0), mock.patch.object(
                hooks, "_post_hot_load", fake_post
            ), mock.patch.object(hooks, "_get_hot_load_state", return_value=ready):
                hooks.announce_and_wait(args, str(version_dir), [])

            # The local version dir is copied to the transport (no rename)...
            self.assertEqual(
                (transport / "weight_v000003" / "model-00000-of-00001.safetensors").read_bytes(), b"delta"
            )
            self.assertTrue((transport / "weight_v000003" / "model.safetensors.index.json").exists())
            # ...then the version is announced (v2 as predecessor).
            self.assertEqual(posted, [("weight_v000003", "weight_v000002")])

    def test_is_noop_off_rank_zero(self) -> None:
        posted: list[int] = []
        args = Namespace(api_shim_base_url="http://provider")
        with mock.patch.object(hooks, "_distributed_rank", return_value=1), mock.patch.object(
            hooks, "_post_hot_load", lambda *a, **k: posted.append(1)
        ):
            hooks.announce_and_wait(args, "/work/weight_v000003", [])

        self.assertEqual(posted, [])

    def test_skips_baseline_non_version_dir(self) -> None:
        posted: list[int] = []
        args = Namespace(api_shim_base_url="http://provider")
        with mock.patch.object(hooks, "_distributed_rank", return_value=0), mock.patch.object(
            hooks, "_post_hot_load", lambda *a, **k: posted.append(1)
        ):
            # _capture_baseline calls the hook with the disk-dir root, not a
            # weight_v{N} dir — it must be a no-op, not an error.
            hooks.announce_and_wait(args, "/mnt/stitch-s3-transport", [])
        self.assertEqual(posted, [])


class RolloutRequestHookTest(unittest.TestCase):
    def test_skips_pin_without_rollout_id_but_sets_affinity(self) -> None:
        # PR #5's request carries no rollout_id: the hook must not crash, must
        # skip the version pin, and still apply session affinity.
        args = Namespace(
            api_shim_rollout_request_weight_version_mode="exact",
            rollout_endpoint_url="http://provider",
        )
        sample = Namespace(session_id="grp-1")
        request = {"url": "u", "payload": {}, "headers": None, "max_retries": 60, "retry_sleep": 1.0}
        with mock.patch.dict("os.environ", {}, clear=True):
            hooks.rollout_request_weight_version_hook(args, sample, request)
        self.assertNotIn("weight_version", request["payload"])
        self.assertEqual(request["headers"]["x-session-affinity"], "grp-1")

    def test_attaches_auth_headers(self) -> None:
        # The front door enforces auth on inference too, so every rollout request
        # must carry the provider auth headers.
        args = Namespace(
            api_shim_rollout_request_weight_version_mode="none",
            rollout_endpoint_url="http://provider",
        )
        sample = Namespace(session_id=None)
        request = {"payload": {}, "headers": None}
        env = {
            "STITCH_SHIM_API_KEY": "k",
            "STITCH_SHIM_PROVIDER_MODEL": "qwen3-4b",
            "STITCH_SHIM_PROVIDER_DEPLOYMENT": "prod",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            hooks.rollout_request_weight_version_hook(args, sample, request)
        self.assertEqual(request["headers"]["Authorization"], "Bearer k")
        self.assertEqual(request["headers"]["Provider-Model"], "qwen3-4b")
        self.assertEqual(request["headers"]["Provider-Deployment"], "prod")

    def test_pins_exact_when_rollout_id_supplied(self) -> None:
        args = Namespace(
            api_shim_rollout_request_weight_version_mode="exact",
            api_shim_rollout_request_version_lag=0,
            rollout_endpoint_url="http://provider",
        )
        sample = Namespace(session_id=None)
        request = {"payload": {}, "headers": None, "rollout_id": 3}
        with mock.patch.dict("os.environ", {}, clear=True):
            hooks.rollout_request_weight_version_hook(args, sample, request)
        self.assertEqual(request["payload"]["weight_version"], {"exact_version": 3})


if __name__ == "__main__":
    unittest.main()
