from __future__ import annotations

import unittest

from stitch.engines.sglang import compose_extra_key, parse_extra_key_version


class ExtraKeyTest(unittest.TestCase):
    def test_compose_extra_key_round_trips_and_is_position_fixed(self) -> None:
        self.assertEqual(compose_extra_key(0), "wv0;")
        self.assertEqual(compose_extra_key(7, "my-key"), "wv7;my-key")
        self.assertEqual(parse_extra_key_version(compose_extra_key(12, None)), 12)
        self.assertEqual(parse_extra_key_version(compose_extra_key(3, "wv9;decoy")), 3)
        # The user key cannot shift or forge the version segment.
        self.assertEqual(parse_extra_key_version("wv1;anything;else"), 1)
        self.assertIsNone(parse_extra_key_version("plain-user-key"))
        self.assertIsNone(parse_extra_key_version("wv12"))  # no terminator
        self.assertIsNone(parse_extra_key_version("wvx;k"))


if __name__ == "__main__":
    unittest.main()
