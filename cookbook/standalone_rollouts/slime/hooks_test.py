from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from cookbook.standalone_rollouts.ledger import (
    DeltaFormats,
    IdentityLedger,
    save_ledger_data,
)
from cookbook.standalone_rollouts.slime import hooks
from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import RolloutPoolState, RolloutReplicaState


class ShimConfigTest(unittest.TestCase):
    def test_requires_base_identity_and_uses_flat_transport(self) -> None:
        args = Namespace(
            rollout_endpoint_url="http://provider/",
            api_shim_transport_root="/transport",
        )
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "BASE_SNAPSHOT_IDENTITY"):
                hooks.ShimConfig.from_env(args)

        args.api_shim_base_snapshot_identity = "customer-base"
        with mock.patch.dict("os.environ", {}, clear=True):
            cfg = hooks.ShimConfig.from_env(args)
        self.assertEqual(cfg.api_base_url, "http://provider")
        self.assertEqual(cfg.base_snapshot_identity, "customer-base")
        self.assertEqual(
            cfg.transport_path_for_identity("opaque-a"), Path("/transport/opaque-a")
        )
        self.assertEqual(cfg.previous_identity_for_version(1), "customer-base")
        self.assertEqual(cfg.previous_identity_for_version(2), "weight_v000001")


class CleanTransportTest(unittest.TestCase):
    def test_accepts_only_seeded_base_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = IdentityLedger.new("customer-base")
            save_ledger_data(root, ledger.to_dict())
            FilesystemBulletinBoard(root, layout="slime").write_latest(None, 0)
            cfg = hooks.ShimConfig(
                api_base_url="http://provider",
                base_snapshot_identity="customer-base",
                transport_root=root,
            )
            hooks.assert_clean_transport(cfg)

    def test_rejects_missing_mismatched_advanced_or_dirty_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = hooks.ShimConfig(
                api_base_url="http://provider",
                base_snapshot_identity="customer-base",
                transport_root=root,
            )
            with self.assertRaises(RuntimeError):
                hooks.assert_clean_transport(cfg)

            save_ledger_data(root, IdentityLedger.new("other-base").to_dict())
            with self.assertRaises(RuntimeError):
                hooks.assert_clean_transport(cfg)

            formats = DeltaFormats(
                delta_encoding="xor",
                compression_format="zstd",
                checksum_format="xxh3-128",
            )
            ledger = IdentityLedger.new("customer-base")
            ledger.append_delta("old-delta", "customer-base", formats)
            save_ledger_data(root, ledger.to_dict())
            FilesystemBulletinBoard(root, layout="slime").write_latest(None, 1)
            with self.assertRaises(RuntimeError):
                hooks.assert_clean_transport(cfg)

            save_ledger_data(root, IdentityLedger.new("customer-base").to_dict())
            FilesystemBulletinBoard(root, layout="slime").write_latest(None, 0)
            (root / "orphan-upload").mkdir()
            with self.assertRaises(RuntimeError):
                hooks.assert_clean_transport(cfg)


class AnnounceAndWaitTest(unittest.TestCase):
    def test_copies_flat_then_announces_and_waits_on_opaque_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = root / "local" / "weight_v000003"
            version_dir.mkdir(parents=True)
            (version_dir / "model-00000-of-00001.safetensors").write_bytes(b"delta")
            (version_dir / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": {}}), encoding="utf-8"
            )
            transport = root / "transport"
            posted: list[tuple[str, str]] = []

            def fake_post(_cfg, *, identity, previous_identity) -> None:
                posted.append((identity, previous_identity))

            ready = RolloutPoolState(
                replicas=[
                    RolloutReplicaState(
                        readiness=True,
                        current_snapshot_identity="weight_v000003",
                    )
                ]
            )
            args = Namespace(
                api_shim_base_url="http://provider",
                api_shim_transport_root=str(transport),
                api_shim_base_snapshot_identity="customer-base",
            )
            with (
                mock.patch.object(hooks, "_distributed_rank", return_value=0),
                mock.patch.object(hooks, "_post_hot_load", fake_post),
                mock.patch.object(hooks, "_get_hot_load_state", return_value=ready),
            ):
                hooks.announce_and_wait(args, str(version_dir), [])

            self.assertTrue(
                (transport / "weight_v000003" / "model.safetensors.index.json").exists()
            )
            self.assertEqual(posted, [("weight_v000003", "weight_v000002")])

    def test_is_noop_off_rank_zero_or_for_baseline_dir(self) -> None:
        posted: list[int] = []
        args = Namespace(
            api_shim_base_url="http://provider",
            api_shim_base_snapshot_identity="base",
        )
        with (
            mock.patch.object(hooks, "_distributed_rank", return_value=1),
            mock.patch.object(
                hooks, "_post_hot_load", lambda *args, **kwargs: posted.append(1)
            ),
        ):
            hooks.announce_and_wait(args, "/work/weight_v000003", [])
        with (
            mock.patch.object(hooks, "_distributed_rank", return_value=0),
            mock.patch.object(
                hooks, "_post_hot_load", lambda *args, **kwargs: posted.append(1)
            ),
        ):
            hooks.announce_and_wait(args, "/work/checkpoints", [])
        self.assertEqual(posted, [])

    def test_never_uploads_slime_v0_as_a_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = root / "weight_v000000"
            version_dir.mkdir()
            args = Namespace(
                api_shim_base_url="http://provider",
                api_shim_transport_root=str(root / "transport"),
                api_shim_base_snapshot_identity="base",
            )
            with mock.patch.object(hooks, "_distributed_rank", return_value=0):
                with self.assertRaisesRegex(RuntimeError, "v0"):
                    hooks.announce_and_wait(args, str(version_dir), [])


class UploadTest(unittest.TestCase):
    def test_exact_retry_does_not_rewrite_and_difference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root / "source", root / "destination"
            source.mkdir()
            (source / "index.json").write_bytes(b"same")
            hooks._copy_version_to_transport(source, destination)
            first = (destination / "index.json").stat().st_mtime_ns

            hooks._copy_version_to_transport(source, destination)
            self.assertEqual((destination / "index.json").stat().st_mtime_ns, first)

            (source / "index.json").write_bytes(b"different")
            with self.assertRaises(FileExistsError):
                hooks._copy_version_to_transport(source, destination)
            self.assertEqual((destination / "index.json").read_bytes(), b"same")


class PostHotLoadTest(unittest.TestCase):
    def test_posts_no_run_id_and_validates_acceptance(self) -> None:
        cfg = hooks.ShimConfig(
            api_base_url="http://provider",
            base_snapshot_identity="base",
        )
        responses = [
            {"accepted": True, "identity": "opaque-a"},
            {"accepted": False, "identity": "opaque-a"},
        ]
        calls: list[dict] = []

        def request(*_args, **kwargs):
            calls.append(kwargs["payload"])
            return responses.pop(0)

        with mock.patch.object(hooks, "_request_json", request):
            hooks._post_hot_load(cfg, identity="opaque-a", previous_identity="base")
            with self.assertRaises(RuntimeError):
                hooks._post_hot_load(cfg, identity="opaque-a", previous_identity="base")
        self.assertNotIn("run_id", calls[0])
        self.assertEqual(calls[0]["identity"], "opaque-a")


class RolloutRequestHookTest(unittest.TestCase):
    def test_sets_retries_affinity_and_auth_headers(self) -> None:
        args = Namespace(
            rollout_endpoint_url="http://provider",
            api_shim_base_snapshot_identity="base",
        )
        sample = Namespace(session_id="grp-1")
        request = {"payload": {}, "headers": None}
        env = {
            "STITCH_SHIM_API_KEY": "k",
            "STITCH_SHIM_PROVIDER_MODEL": "moonlight",
            "STITCH_SHIM_PROVIDER_DEPLOYMENT": "prod",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            hooks.rollout_request_weight_version_hook(args, sample, request)
        self.assertEqual(request["headers"]["Authorization"], "Bearer k")
        self.assertEqual(request["headers"]["x-session-affinity"], "grp-1")
        self.assertEqual(request["max_retries"], 60)


if __name__ == "__main__":
    unittest.main()
