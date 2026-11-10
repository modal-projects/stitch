"""SGLang as Stitch's engine: one server in this container, run by autoinference-utils."""

import contextlib
import json
import secrets
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from autoinference_utils.endpoint import SGLangEndpoint

PAUSE_MODES = ("in_place", "retract", "abort")


@dataclass(frozen=True)
class SGLangConfig:
    """How each engine runs SGLang."""

    server_args: dict[str, str]
    env: dict[str, str] = field(default_factory=dict)
    pause_mode: str = "in_place"

    def __post_init__(self):
        if self.pause_mode not in PAUSE_MODES:
            raise ValueError(
                f"pause mode {self.pause_mode!r} is not one of {PAUSE_MODES}"
            )


class SGLang:
    """Serves a checkpoint; the endpoints that drive the engine need an admin key only this holds."""

    def __init__(
        self,
        checkpoint: str,
        version: int,
        port: int,
        config: SGLangConfig,
        startup_timeout: int,
    ):
        self.version, self.config = version, config
        self.admin_key = secrets.token_urlsafe(32)
        self.endpoint = SGLangEndpoint(
            model_path=checkpoint,
            worker_port=port,
            extra_server_args={
                # SGLang's own default, as Miles' engines get; autoinference-utils would pick fastokens.
                "--tokenizer-backend": "huggingface",
                # Modal keeps a routing key on one engine; with DP attention each rank has its own prefix
                # cache, so the engine keeps the key on one rank too.
                "--load-balance-method": "routing_key",
                **config.server_args,
                "--enable-weight-delta": "",
                "--weight-version": str(version),
                "--admin-api-key": self.admin_key,
            },
            health_timeout=startup_timeout,
            log_requests_level=-1,
            watch=False,
        )
        # Compresses replies since egress is billed per byte the engine sends; the run's env can override it.
        self.endpoint.extra_env = {
            "SGLANG_ENABLE_RESPONSE_COMPRESSION": "1",
            **config.env,
        }

    def start(self) -> None:
        self.endpoint.start()

    def warm_up(self) -> None:
        """Builds the kernels SGLang compiles on first use, which would stall the first requests."""
        body = {
            "text": ["Hello"] * 16,
            "sampling_params": {"temperature": 1.0, "max_new_tokens": 16},
        }
        self._post("/generate", body)

    def watch(self) -> None:
        """From here on, SGLang exiting or failing its health checks for 60 s ends the container."""
        self.endpoint.watch(failure_timeout=60)

    def prepare(self, manifest: Path) -> dict | None:
        return self._post(
            "/prepare_weights_from_delta", {"manifest": str(manifest)}
        ).get("stats")

    def apply(self, manifest: Path, version: int) -> dict | None:
        self._post("/pause_generation", {"mode": self.config.pause_mode})
        try:
            updated = self._post(
                "/update_weights_from_delta",
                {
                    "manifest": str(manifest),
                    "weight_version": str(version),
                    "flush_cache": self.config.pause_mode != "in_place",
                },
            )
        except BaseException:
            with contextlib.suppress(OSError, RuntimeError):
                self._post("/continue_generation", {})
            raise
        self._post("/continue_generation", {})
        self.version = version
        return updated.get("stats")

    def abort_all(self) -> None:
        """Ends every request in flight with finish_reason abort."""
        self._post("/abort_request", {"abort_all": True}, timeout=60)

    def stop(self) -> None:
        """Lets the requests in flight finish, then stops SGLang."""
        self.endpoint.stop()

    def _post(self, path: str, body: dict, timeout: float | None = None) -> dict:
        request = urllib.request.Request(
            f"{self.endpoint.base_url}{path}",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.admin_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                data = reply.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"SGLang {path}: HTTP {error.code} {error.read().decode()[:2000]}"
            ) from None
        result = json.loads(data) if data.startswith((b"{", b"[")) else {}
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(f"SGLang {path}: {result.get('message')}")
        return result
