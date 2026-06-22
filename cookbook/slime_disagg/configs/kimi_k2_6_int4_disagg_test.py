"""Contract tests for the Kimi K2.6 native-INT4 disaggregated config.

These assert the invariants that make the INT4-QAT + disk-delta loop coherent —
they are pure data checks and import no modal/slime, so they run anywhere.
"""

from __future__ import annotations

import unittest

from cookbook.slime_disagg.configs import kimi_k2_6_int4_disagg as cfg
from cookbook.slime_disagg.configs.base import SlimeConfig


class KimiK26Int4ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.slime = cfg.slime
        self.args = cfg.slime.cli_args()

    def test_is_slime_config_and_cli_args_build(self) -> None:
        self.assertIsInstance(self.slime, SlimeConfig)
        self.assertTrue(all(isinstance(a, str) for a in self.args))

    def test_int4_qat_wired(self) -> None:
        env = self.slime.environment
        self.assertEqual(env.get("OPEN_TRAINING_INT4_FAKE_QAT_FLAG"), "1")
        # The QAT simulation grouping MUST match the served checkpoint's
        # compressed-tensors group_size (the load-bearing INT4 invariant).
        self.assertEqual(env.get("OPEN_TRAINING_INT4_GROUP_SIZE"), cfg.INT4_GROUP_SIZE)

    def test_disaggregated_disk_delta_contract(self) -> None:
        self.assertFalse(self.slime.colocate)
        self.assertEqual(self.slime.rollout_num_gpus, 0)
        self.assertEqual(self.slime.rollout_num_gpus_per_engine, 4)  # B200:4
        self.assertTrue(self.slime.async_mode)
        self.assertEqual(self.slime.update_weight_mode, "delta")
        self.assertEqual(self.slime.update_weight_transport, "disk")
        self.assertEqual(self.slime.update_weight_delta_encoding, "xor")

    def test_trainer_footprint_matches_recipe(self) -> None:
        # world = TP8 * PP8 * CP4 = 256 = actor_num_nodes(32) * gpus_per_node(8).
        self.assertEqual(self.slime.actor_num_nodes, 32)
        self.assertEqual(self.slime.actor_num_gpus_per_node, 8)
        world = (
            self.slime.tensor_model_parallel_size
            * self.slime.pipeline_model_parallel_size
            * self.slime.context_parallel_size
        )
        self.assertEqual(world, self.slime.actor_num_nodes * self.slime.actor_num_gpus_per_node)

    def test_no_fp4_leakage_and_routing_replay(self) -> None:
        keys = cfg.SGLANG_SERVER_ARGS
        # INT4 is driven by the checkpoint's compressed-tensors config, never a
        # quantization flag — an FP4 leak would mismatch the disk-delta bytes.
        self.assertNotIn("--quantization", keys)
        self.assertNotIn("modelopt_fp4", " ".join(keys))
        # Routing replay needs the pool to emit per-token routed experts.
        self.assertIn("--enable-return-routed-experts", keys)
        self.assertTrue(self.slime.use_rollout_routing_replay)

    def test_mla_does_not_force_flash_attention(self) -> None:
        # MLA models must not set --attention-backend flash on the trainer.
        self.assertNotIn("--attention-backend", self.args)

    def test_launcher_only_fields_excluded_from_cli(self) -> None:
        # environment / async_mode / slime_model_script are launcher instructions,
        # not SLIME CLI args.
        for flag in ("--environment", "--async-mode", "--slime-model-script"):
            self.assertNotIn(flag, self.args)
        self.assertEqual(self.slime.slime_model_script, "scripts/models/kimi-k2-thinking.sh")

    def test_serving_image_builder_is_lazy_and_callable(self) -> None:
        # Present so modal_train deploys the dedicated B200 image; lazy so importing
        # this config needs no modal SDK (verified by this test importing cleanly).
        self.assertTrue(callable(cfg.build_serving_image))


if __name__ == "__main__":
    unittest.main()
