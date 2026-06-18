from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import (
    Artifact,
    RolloutPoolState,
    RolloutReplicaState,
    VersionManifest,
    WeightVersionPolicy,
    evaluate_version_policy,
    parse_weight_identity,
    read_latest,
    weight_identity,
)


class ProtocolTest(unittest.TestCase):
    def test_manifest_round_trips_extended_and_legacy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            board = FilesystemBulletinBoard(root)
            manifest = VersionManifest(
                version=3,
                base_version=2,
                backend="sparse_delta",
                load_format="delta",
                transition_files=["rank0000_flush000000.safetensors"],
                artifacts=[
                    Artifact(
                        kind="transition",
                        path="rank0000_flush000000.safetensors",
                        checksum="sha256:abc",
                    )
                ],
                run_id="run-1",
                base_model="Qwen/Qwen3-4B",
            )

            board.publish_manifest(manifest)
            loaded = board.read_manifest(3)

            self.assertEqual(read_latest(root), 3)
            self.assertEqual(loaded.version, 3)
            self.assertEqual(loaded.base_version, 2)
            self.assertEqual(loaded.transition_artifact_paths(), ["rank0000_flush000000.safetensors"])
            self.assertEqual(loaded.artifacts[0].checksum, "sha256:abc")
            self.assertEqual(loaded.run_id, "run-1")

    def test_weight_identity_round_trips(self) -> None:
        self.assertEqual(weight_identity(0), "weight_v000000")
        self.assertEqual(weight_identity(123), "weight_v000123")
        self.assertEqual(parse_weight_identity("weight_v000123"), 123)
        self.assertIsNone(parse_weight_identity("base"))
        self.assertIsNone(parse_weight_identity("weight_vxyz"))

    def test_evaluate_version_policy(self) -> None:
        self.assertIsNone(evaluate_version_policy(5, WeightVersionPolicy()))
        self.assertIsNone(evaluate_version_policy(5, WeightVersionPolicy(exact_version=5)))
        self.assertEqual(
            evaluate_version_policy(4, WeightVersionPolicy(exact_version=5))["error"]["type"],
            "WeightVersionNotReady",
        )
        self.assertEqual(
            evaluate_version_policy(6, WeightVersionPolicy(exact_version=5))["error"]["type"],
            "WeightVersionTooOld",
        )
        self.assertEqual(
            evaluate_version_policy(4, WeightVersionPolicy(min_required_version=5))["error"]["type"],
            "WeightVersionNotReady",
        )
        self.assertIsNone(evaluate_version_policy(5, WeightVersionPolicy(min_required_version=5)))

    def test_weight_version_policy_ignores_malformed_payload(self) -> None:
        self.assertEqual(WeightVersionPolicy.from_payload({}), WeightVersionPolicy())
        self.assertEqual(WeightVersionPolicy.from_payload({"weight_version": 7}), WeightVersionPolicy())
        self.assertEqual(
            WeightVersionPolicy.from_payload({"weight_version": {"min_required_version": "5", "exact_version": 6}}),
            WeightVersionPolicy(min_required_version=5, exact_version=6),
        )

    def test_rollout_pool_state_parses_snapshot_readiness(self) -> None:
        state = RolloutPoolState.from_dict(
            {
                "replicas": [
                    {
                        "replica_id": "a",
                        "readiness": True,
                        "current_snapshot_identity": "ckpt-2",
                        "zone": "us-east-1a",
                    },
                    {
                        "replica_id": "b",
                        "readiness": True,
                        "current_snapshot_identity": "ckpt-2",
                    },
                    {
                        "replica_id": "c",
                        "readiness": False,
                        "current_snapshot_identity": "ckpt-1",
                        "readiness_reason": "downloading weights",
                    },
                ]
            }
        )

        self.assertEqual(state.ready_count(target_snapshot_identity="ckpt-2"), 2)
        self.assertEqual(state.readiness_fraction(target_snapshot_identity="ckpt-2"), 2 / 3)
        self.assertTrue(state.is_ready(target_snapshot_identity="ckpt-2", threshold=0.5))
        self.assertFalse(state.is_ready(target_snapshot_identity="ckpt-2", threshold=1.0))
        self.assertEqual(state.replicas[0].metadata["zone"], "us-east-1a")
        self.assertEqual(state.replicas[2].readiness_reason, "downloading weights")

    def test_rollout_pool_state_matches_integer_versions(self) -> None:
        state = RolloutPoolState(
            replicas=[
                RolloutReplicaState(readiness=True, current_version=7),
                RolloutReplicaState(readiness=True, current_version=6),
            ]
        )

        self.assertEqual(state.ready_count(target_version=7), 1)
        self.assertEqual(state.ready_count(), 2)
        self.assertFalse(RolloutPoolState().is_ready())
        self.assertEqual(
            RolloutPoolState.from_dict(state.to_dict()),
            state,
        )


if __name__ == "__main__":
    unittest.main()
