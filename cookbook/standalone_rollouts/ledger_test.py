from __future__ import annotations

import unittest

from cookbook.standalone_rollouts.ledger import (
    DeltaFormats,
    IdentityLedger,
    LedgerConflict,
    LedgerCorruption,
    LedgerRewind,
    is_valid_identity,
)


FORMATS = DeltaFormats(
    delta_encoding="xor",
    compression_format="zstd",
    checksum_format="xxh3-128",
)


class IdentityTest(unittest.TestCase):
    def test_identity_is_opaque_but_must_be_one_safe_path_component(self) -> None:
        self.assertTrue(is_valid_identity("customer-checkpoint:abc_123"))
        for identity in (
            "",
            ".",
            "..",
            "latest",
            "identities.json",
            ".stitch",
            "nested/checkpoint",
            "nested\\checkpoint",
            "contains\x00nul",
            "unpaired-surrogate-\ud800",
            "x" * 256,
        ):
            with self.subTest(identity=identity):
                self.assertFalse(is_valid_identity(identity))


class LineageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = IdentityLedger.new("base")

    def test_new_ledger_is_seeded_with_configured_base(self) -> None:
        self.assertEqual(self.ledger.head_version, 0)
        self.assertEqual(self.ledger.identity_for(0), "base")
        self.assertEqual(self.ledger.version_for("base"), 0)

    def test_deltas_form_one_contiguous_chain(self) -> None:
        first = self.ledger.append_delta("delta-a", "base", FORMATS)
        second = self.ledger.append_delta("delta-b", "delta-a", FORMATS)
        self.assertTrue(first.is_new)
        self.assertEqual(first.entry.version, 1)
        self.assertEqual(second.entry.version, 2)
        self.assertEqual(second.entry.previous_snapshot_identity, "delta-a")

    def test_exact_head_retry_is_idempotent(self) -> None:
        first = self.ledger.append_delta("delta-a", "base", FORMATS)
        retry = self.ledger.append_delta("delta-a", "base", FORMATS)
        self.assertTrue(first.is_new)
        self.assertFalse(retry.is_new)
        self.assertEqual(retry.entry, first.entry)

    def test_retry_must_preserve_parent_and_formats(self) -> None:
        self.ledger.append_delta("delta-a", "base", FORMATS)
        different_formats = DeltaFormats(
            delta_encoding="xor",
            compression_format="zstd",
            checksum_format="blake3",
        )
        for parent, formats in (("other", FORMATS), ("base", different_formats)):
            with self.subTest(parent=parent, formats=formats):
                with self.assertRaises(LedgerConflict):
                    self.ledger.append_delta("delta-a", parent, formats)

    def test_old_identity_retry_is_a_rewind(self) -> None:
        self.ledger.append_delta("delta-a", "base", FORMATS)
        self.ledger.append_delta("delta-b", "delta-a", FORMATS)
        with self.assertRaises(LedgerRewind):
            self.ledger.append_delta("delta-a", "base", FORMATS)

    def test_unknown_old_or_self_parent_is_rejected(self) -> None:
        self.ledger.append_delta("delta-a", "base", FORMATS)
        for identity, parent in (
            ("unknown-parent", "missing"),
            ("fork", "base"),
            ("self-parent", "self-parent"),
        ):
            with self.subTest(identity=identity, parent=parent):
                with self.assertRaises(LedgerConflict):
                    self.ledger.append_delta(identity, parent, FORMATS)

    def test_only_configured_base_can_be_confirmed(self) -> None:
        retry = self.ledger.confirm_base("base")
        self.assertFalse(retry.is_new)
        with self.assertRaises(LedgerConflict):
            self.ledger.confirm_base("other-full-snapshot")

    def test_base_confirmation_after_a_delta_is_a_rewind(self) -> None:
        self.ledger.append_delta("delta-a", "base", FORMATS)
        with self.assertRaises(LedgerRewind):
            self.ledger.confirm_base("base")
        with self.assertRaises(LedgerConflict):
            self.ledger.append_delta("base", "base", FORMATS)


class FormatTest(unittest.TestCase):
    def test_only_decoder_supported_formats_are_accepted(self) -> None:
        self.assertEqual(DeltaFormats.defaults().checksum_format, "adler32")
        for kwargs in (
            {"delta_encoding": "overwrite"},
            {"compression_format": "gzip"},
            {"checksum_format": "sha256"},
        ):
            values = {
                "delta_encoding": "xor",
                "compression_format": "zstd",
                "checksum_format": "adler32",
                **kwargs,
            }
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    DeltaFormats(**values)


class SerializationTest(unittest.TestCase):
    def test_round_trip_uses_an_ordered_versioned_schema(self) -> None:
        ledger = IdentityLedger.new("base")
        ledger.append_delta("delta-a", "base", FORMATS)
        data = ledger.to_dict()
        self.assertEqual(
            data,
            {
                "schema_version": 1,
                "base_identity": "base",
                "deltas": [
                    {
                        "identity": "delta-a",
                        "formats": {
                            "delta_encoding": "xor",
                            "compression_format": "zstd",
                            "checksum_format": "xxh3-128",
                        },
                    },
                ],
            },
        )
        restored = IdentityLedger.from_dict(data, expected_base_identity="base")
        self.assertEqual(restored.to_dict(), data)

    def test_expected_base_must_match_persisted_base(self) -> None:
        data = IdentityLedger.new("base-a").to_dict()
        with self.assertRaises(LedgerCorruption):
            IdentityLedger.from_dict(data, expected_base_identity="base-b")

    def test_malformed_persisted_state_never_coerces_or_seeds(self) -> None:
        corruptions = [
            {},
            {
                "schema_version": True,
                "base_identity": "base",
                "deltas": [],
            },
            {
                "schema_version": 2,
                "base_identity": "base",
                "deltas": [],
            },
            {
                "schema_version": 1,
                "base_identity": "base",
                "deltas": {},
            },
            {
                "schema_version": 1,
                "base_identity": "base",
                "deltas": [
                    {
                        "identity": "delta",
                        "formats": FORMATS.to_dict(),
                        "extra": True,
                    },
                ],
            },
            {
                "schema_version": 1,
                "base_identity": "base",
                "deltas": [
                    {
                        "identity": "base",
                        "formats": FORMATS.to_dict(),
                    },
                ],
            },
        ]
        for data in corruptions:
            with self.subTest(data=data):
                with self.assertRaises(LedgerCorruption):
                    IdentityLedger.from_dict(data, expected_base_identity="base")


if __name__ == "__main__":
    unittest.main()
