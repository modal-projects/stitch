from __future__ import annotations

import unittest
from argparse import Namespace
from unittest import mock

from cookbook import rollout_control


class DistributedRankTest(unittest.TestCase):
    def test_none_when_torch_distributed_unavailable(self) -> None:
        # No torch dist initialized in tests (torch may be absent entirely): the
        # probe must degrade to None, which the hooks treat as "single writer".
        self.assertIsNone(rollout_control.distributed_rank())


class ReadSettingTest(unittest.TestCase):
    def test_args_beats_env_beats_default(self) -> None:
        args = Namespace(api_key="from-args")
        with mock.patch.dict("os.environ", {"API_KEY": "from-env"}, clear=True):
            self.assertEqual(
                rollout_control.read_setting(args, "api_key", "API_KEY", default="d"), "from-args"
            )
        with mock.patch.dict("os.environ", {"API_KEY": "from-env"}, clear=True):
            self.assertEqual(
                rollout_control.read_setting(Namespace(), "api_key", "API_KEY", default="d"),
                "from-env",
            )
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                rollout_control.read_setting(Namespace(), "api_key", "API_KEY", default="d"), "d"
            )

    def test_empty_string_for_absent_optional_and_raises_when_required(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(rollout_control.read_setting(None, "x", "X"), "")
            with self.assertRaises(RuntimeError):
                rollout_control.read_setting(None, "x", "X", required=True)


class SessionAffinityTest(unittest.TestCase):
    def test_stamps_header_without_overriding_explicit_value(self) -> None:
        request: dict = {"headers": {"x-aff": "explicit"}}
        rollout_control.apply_session_affinity(request, "sess-1", "x-aff")
        self.assertEqual(request["headers"]["x-aff"], "explicit")  # setdefault, not override

        request = {}
        rollout_control.apply_session_affinity(request, "sess-1", "x-aff")
        self.assertEqual(request["headers"]["x-aff"], "sess-1")

    def test_noop_for_falsy_session_id(self) -> None:
        request: dict = {"headers": {"a": "b"}}
        rollout_control.apply_session_affinity(request, None, "x-aff")
        self.assertEqual(request["headers"], {"a": "b"})
        self.assertNotIn("x-aff", request["headers"])


if __name__ == "__main__":
    unittest.main()
