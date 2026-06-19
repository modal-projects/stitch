from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from stitch.protocol import VersionManifest, read_latest
from stitch.trainers.slime import publish_delta_version, rollout_request_weight_version_hook


class SlimeHooksTest(unittest.TestCase):
    def test_rollout_request_hook_adds_exact_policy_retry_and_affinity(self) -> None:
        args = Namespace(
            rollout_request_weight_version_mode="exact",
            rollout_request_weight_version_lag=1,
            rollout_request_retry_attempts=240,
            rollout_request_retry_sleep=0.25,
        )
        sample = Namespace(session_id="session-1")
        request = {
            "payload": {},
            "headers": None,
            "max_retries": 60,
            "retry_sleep": 1.0,
            "rollout_id": 3,
            "evaluation": False,
        }

        rollout_request_weight_version_hook(args, sample, request)

        self.assertEqual(request["payload"]["weight_version"], {"exact_version": 2})
        self.assertEqual(request["max_retries"], 240)
        self.assertEqual(request["retry_sleep"], 0.25)
        self.assertEqual(request["headers"]["x-session-affinity"], "session-1")

    def test_rollout_request_hook_uses_configured_affinity_header(self) -> None:
        args = Namespace(
            rollout_request_weight_version_mode="exact",
            rollout_session_affinity_header="Modal-Session-ID",
        )
        sample = Namespace(session_id="group-7")
        request = {
            "payload": {},
            "headers": None,
            "max_retries": 60,
            "retry_sleep": 1.0,
            "rollout_id": 3,
            "evaluation": False,
        }

        rollout_request_weight_version_hook(args, sample, request)

        self.assertEqual(request["headers"]["Modal-Session-ID"], "group-7")
        self.assertNotIn("x-session-affinity", request["headers"])

    def test_rollout_request_hook_can_add_min_policy(self) -> None:
        args = Namespace(rollout_request_weight_version_mode="min")
        sample = Namespace(session_id=None)
        request = {
            "payload": {},
            "headers": None,
            "max_retries": 60,
            "retry_sleep": 1.0,
            "rollout_id": 3,
            "evaluation": False,
        }

        rollout_request_weight_version_hook(args, sample, request)

        self.assertEqual(request["payload"]["weight_version"], {"min_required_version": 3})
        self.assertIsNone(request["headers"])

    def test_rollout_request_hook_degrades_without_rollout_id(self) -> None:
        # PR #5's request carries no rollout_id: skip the pin (no crash), still
        # apply the retry budget and session affinity.
        args = Namespace(
            rollout_request_weight_version_mode="exact",
            rollout_request_retry_attempts=240,
        )
        sample = Namespace(session_id="grp-9")
        request = {"payload": {}, "headers": None, "max_retries": 60, "retry_sleep": 1.0}

        rollout_request_weight_version_hook(args, sample, request)

        self.assertNotIn("weight_version", request["payload"])
        self.assertEqual(request["max_retries"], 240)
        self.assertEqual(request["headers"]["x-session-affinity"], "grp-9")

    def test_publish_delta_version_writes_manifest_and_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = root / "versions" / "weight_v000001"
            version_dir.mkdir(parents=True)
            args = Namespace(
                update_weight_delta_root=str(root),
                update_weight_delta_dir=str(root / "versions"),
                hf_checkpoint="Qwen/Qwen3-4B",
                run_id="run-1",
            )

            refs = publish_delta_version(args, str(version_dir), ["b.safetensors", "a.safetensors"], 1, [])

            manifest = VersionManifest.read(version_dir / "manifest.json")
            self.assertEqual(refs, [])
            self.assertEqual(read_latest(root), 1)
            self.assertEqual(manifest.version, 1)
            self.assertEqual(manifest.base_version, 0)
            self.assertEqual(manifest.transition_files, ["a.safetensors", "b.safetensors"])
            self.assertEqual(manifest.artifacts[0].kind, "transition")
            self.assertEqual(manifest.metadata["trainer"], "slime")

    def test_publish_delta_version_lifts_slime_index(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = root / "versions" / "weight_v000002"
            version_dir.mkdir(parents=True)
            (version_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "version": "000002",
                            "base_version": "000001",
                            "delta_encoding": "xor",
                            "compression_format": "zstd",
                            "checksum_format": "xxh3-128",
                        },
                        "weight_map": {"w": "model-00001-of-00001.safetensors"},
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace(
                update_weight_delta_root=str(root),
                update_weight_delta_dir=str(root / "versions"),
                hf_checkpoint="Qwen/Qwen3-4B",
                run_id="run-2",
            )

            # files arg is ignored when the engine wrote a canonical index.
            publish_delta_version(args, str(version_dir), [], 2, [])

            manifest = VersionManifest.read(version_dir / "manifest.json")
            self.assertEqual(read_latest(root), 2)
            self.assertEqual(manifest.base_version, 1)
            self.assertEqual(manifest.backend, "disk_delta")
            self.assertEqual(manifest.compression_format, "zstd")
            self.assertEqual(manifest.checksum_format, "xxh3-128")
            self.assertEqual(manifest.transition_files, ["model-00001-of-00001.safetensors"])


if __name__ == "__main__":
    unittest.main()
