from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cookbook.standalone_rollouts.delta_view import merge_index_metadata, rebuild_delta_view
from cookbook.standalone_rollouts.ledger import IdentityLedger


class RebuildDeltaViewTest(unittest.TestCase):
    def _transport(self, tmp: str) -> Path:
        transport = Path(tmp) / "transport"
        transport.mkdir()
        (transport / "latest").write_text("weight_v000001", encoding="utf-8")
        # Customer uploaded to identity-named dirs, not weight_vN.
        for identity in ("base-ckpt", "ckpt-100"):
            d = transport / identity
            d.mkdir()
            (d / "model-00001-of-00001.safetensors").write_bytes(b"shard")
            (d / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": 5}, "weight_map": {}}), encoding="utf-8"
            )
        return transport

    def test_view_presents_derived_index_and_reads_through(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport = self._transport(tmp)
            ledger = IdentityLedger()
            ledger.record("base-ckpt", previous=None)  # v0
            ledger.record("ckpt-100", previous="base-ckpt")  # v1
            merge_index_metadata(
                transport / "ckpt-100" / "model.safetensors.index.json",
                {"version": "000001", "base_version": "000000"},
            )
            view = Path(tmp) / "view"

            rebuild_delta_view(view, transport, ledger)

            # The decoder sees the derived (normalized) index under the HF name,
            # while the customer's uploaded index is untouched.
            index = json.loads(
                (view / "weight_v000001" / "model.safetensors.index.json").read_text()
            )
            self.assertEqual(index["metadata"]["version"], "000001")
            self.assertEqual(index["metadata"]["total_size"], 5)
            raw = json.loads(
                (transport / "ckpt-100" / "model.safetensors.index.json").read_text()
            )
            self.assertNotIn("version", raw["metadata"])
            # Shards and the latest pointer resolve through the view; a base
            # with no derived index presents its own upload as-is.
            self.assertEqual(
                (view / "weight_v000001" / "model-00001-of-00001.safetensors").read_bytes(),
                b"shard",
            )
            self.assertEqual((view / "latest").read_text(), "weight_v000001")
            base_index = json.loads(
                (view / "weight_v000000" / "model.safetensors.index.json").read_text()
            )
            self.assertEqual(base_index["metadata"], {"total_size": 5})

    def test_rebuild_is_idempotent_and_picks_up_new_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport = self._transport(tmp)
            view = Path(tmp) / "view"

            ledger = IdentityLedger()
            ledger.record("base-ckpt", previous=None)
            rebuild_delta_view(view, transport, ledger)
            self.assertFalse((view / "weight_v000001").exists())

            # A later signal adds v1; a rebuild picks it up without disturbing v0.
            ledger.record("ckpt-100", previous="base-ckpt")
            rebuild_delta_view(view, transport, ledger)
            rebuild_delta_view(view, transport, ledger)  # idempotent second pass
            self.assertTrue((view / "weight_v000001").is_dir())
            self.assertTrue((view / "weight_v000000").is_dir())


if __name__ == "__main__":
    unittest.main()
