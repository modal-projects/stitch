from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from cookbook.standalone_rollouts.slime import hooks


class SlimeTransportHooksTest(unittest.TestCase):
    def test_config_uses_rollout_endpoint_as_shim_base_url_fallback(self) -> None:
        args = Namespace(
            api_shim_transport_root="/tmp/transport",
            rollout_http_endpoint_url="http://provider/",
        )

        with mock.patch.dict("os.environ", {}, clear=True):
            cfg = hooks.ShimConfig.from_env(args)

        self.assertEqual(cfg.api_base_url, "http://provider")

    def test_copy_delta_to_transport_copies_rank_prefixed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = root / "versions" / "weight_v000001"
            version_dir.mkdir(parents=True)
            (version_dir / "rank0000_delta.safetensors").write_bytes(b"rank-0")
            (version_dir / "rank0001_delta.safetensors").write_bytes(b"rank-1")
            transport_root = root / "transport"
            args = Namespace(
                api_shim_transport_root=str(transport_root),
                api_shim_base_url="http://provider",
            )

            with mock.patch.object(hooks, "_distributed_rank", return_value=0):
                hooks.copy_delta_to_transport(args, str(version_dir), [])

            destination = transport_root / "weight_v000001"
            self.assertEqual(
                (destination / "rank0000_delta.safetensors").read_bytes(), b"rank-0"
            )
            self.assertFalse((destination / "rank0001_delta.safetensors").exists())

    def test_copy_delta_to_transport_copies_all_files_without_distributed_rank(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = root / "versions" / "weight_v000001"
            version_dir.mkdir(parents=True)
            (version_dir / "rank0000_delta.safetensors").write_bytes(b"rank-0")
            (version_dir / "manifest.json").write_text("{}", encoding="utf-8")
            transport_root = root / "transport"
            args = Namespace(
                api_shim_transport_root=str(transport_root),
                api_shim_base_url="http://provider",
            )

            with mock.patch.object(hooks, "_distributed_rank", return_value=None):
                hooks.copy_delta_to_transport(args, str(version_dir), [])

            destination = transport_root / "weight_v000001"
            self.assertEqual(
                (destination / "rank0000_delta.safetensors").read_bytes(), b"rank-0"
            )
            self.assertEqual(
                (destination / "manifest.json").read_text(encoding="utf-8"), "{}"
            )

    def test_copy_delta_to_transport_replaces_stale_target_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = root / "versions" / "weight_v000001"
            version_dir.mkdir(parents=True)
            (version_dir / "rank0000_delta.safetensors").write_bytes(b"new")
            transport_root = root / "transport"
            destination = transport_root / "weight_v000001"
            destination.mkdir(parents=True)
            (destination / "rank0000_delta.safetensors").write_bytes(b"stale")
            args = Namespace(
                api_shim_transport_root=str(transport_root),
                api_shim_base_url="http://provider",
            )

            with mock.patch.object(hooks, "_distributed_rank", return_value=0):
                hooks.copy_delta_to_transport(args, str(version_dir), [])

            self.assertEqual(
                (destination / "rank0000_delta.safetensors").read_bytes(), b"new"
            )

    def test_copy_delta_to_transport_skips_files_for_other_distributed_rank(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = root / "versions" / "weight_v000001"
            version_dir.mkdir(parents=True)
            (version_dir / "rank0000_delta.safetensors").write_bytes(b"rank-0")
            transport_root = root / "transport"
            args = Namespace(
                api_shim_transport_root=str(transport_root),
                api_shim_base_url="http://provider",
            )

            with mock.patch.object(hooks, "_distributed_rank", return_value=1):
                hooks.copy_delta_to_transport(args, str(version_dir), [])

            self.assertFalse((transport_root / "weight_v000001").exists())


if __name__ == "__main__":
    unittest.main()
