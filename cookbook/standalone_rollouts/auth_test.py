from __future__ import annotations

import unittest

from cookbook.standalone_rollouts.auth import authorize


GOOD = {"authorization": "Bearer secret"}


class AuthorizeTest(unittest.TestCase):
    def test_unset_api_key_fails_closed(self) -> None:
        for headers in ({}, GOOD, {"authorization": "Bearer anything"}):
            rejection = authorize(
                headers, api_key=None, provider_model=None, provider_deployment=None
            )
            self.assertIsNotNone(rejection)
            self.assertEqual(rejection[0], 503)

    def test_correct_bearer_is_allowed(self) -> None:
        self.assertIsNone(
            authorize(
                GOOD, api_key="secret", provider_model=None, provider_deployment=None
            )
        )

    def test_missing_or_wrong_bearer_is_401(self) -> None:
        for headers in (
            {},
            {"authorization": "Bearer wrong"},
            {"authorization": "secret"},
        ):
            rejection = authorize(
                headers, api_key="secret", provider_model=None, provider_deployment=None
            )
            self.assertEqual(rejection, (401, "unauthorized"))

    def test_optional_provider_headers_are_enforced_when_configured(self) -> None:
        self.assertEqual(
            authorize(
                GOOD,
                api_key="secret",
                provider_model="moonlight",
                provider_deployment=None,
            )[0],
            400,
        )
        self.assertIsNone(
            authorize(
                {**GOOD, "provider-model": "moonlight"},
                api_key="secret",
                provider_model="moonlight",
                provider_deployment=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
