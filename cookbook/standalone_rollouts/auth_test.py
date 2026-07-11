from __future__ import annotations

import unittest

from cookbook.standalone_rollouts.auth import authorize


GOOD = {"authorization": "Bearer secret"}


class AuthorizeTest(unittest.TestCase):
    def test_unset_api_key_fails_closed_for_every_request(self) -> None:
        # The Modal server is unauthenticated; a missing key must reject, not
        # skip the check and serve an open endpoint.
        for headers in ({}, GOOD, {"authorization": "Bearer anything"}):
            rejection = authorize(
                headers, api_key=None, provider_model=None, provider_deployment=None
            )
            self.assertIsNotNone(rejection)
            self.assertEqual(rejection[0], 503)

    def test_correct_bearer_is_allowed(self) -> None:
        self.assertIsNone(
            authorize(GOOD, api_key="secret", provider_model=None, provider_deployment=None)
        )

    def test_missing_or_wrong_bearer_is_401(self) -> None:
        for headers in ({}, {"authorization": "Bearer wrong"}, {"authorization": "secret"}):
            rejection = authorize(
                headers, api_key="secret", provider_model=None, provider_deployment=None
            )
            self.assertEqual(rejection, (401, "unauthorized"))

    def test_provider_model_enforced_only_when_configured(self) -> None:
        # Not configured -> not checked.
        self.assertIsNone(
            authorize(GOOD, api_key="secret", provider_model=None, provider_deployment=None)
        )
        # Configured + mismatch -> 400.
        rejection = authorize(
            GOOD, api_key="secret", provider_model="moonlight", provider_deployment=None
        )
        self.assertEqual(rejection[0], 400)
        # Configured + match -> allowed.
        self.assertIsNone(
            authorize(
                {**GOOD, "provider-model": "moonlight"},
                api_key="secret",
                provider_model="moonlight",
                provider_deployment=None,
            )
        )

    def test_provider_deployment_enforced_only_when_configured(self) -> None:
        rejection = authorize(
            GOOD, api_key="secret", provider_model=None, provider_deployment="rollout-prod"
        )
        self.assertEqual(rejection[0], 400)
        self.assertIsNone(
            authorize(
                {**GOOD, "provider-deployment": "rollout-prod"},
                api_key="secret",
                provider_model=None,
                provider_deployment="rollout-prod",
            )
        )


if __name__ == "__main__":
    unittest.main()
