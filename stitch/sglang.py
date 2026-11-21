"""SGLang as Stitch's engine: one server in this container, run by autoinference-utils."""

import asyncio
import json
import secrets
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

from autoinference_utils.endpoint import SGLangEndpoint

PAUSE_MODES = ("in_place", "retract")


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

    @property
    def dp_ranks(self) -> int:
        """The engine's DP ranks as SGLang counts them: replicas times attention-DP groups."""
        args = self.server_args
        replicas = args.get("--dp-size", args.get("--data-parallel-size", "1"))
        groups = args.get(
            "--attn-dp-size", args.get("--attention-data-parallel-size", "1")
        )
        return int(replicas) * int(groups)


class _Receiver:
    """The GPU-delta controls of this container's SGLang, as Miles' activation posts them."""

    def __init__(self, sglang: "SGLang"):
        self.sglang = sglang

    async def _make_request(self, endpoint: str, payload: dict) -> dict:
        return await asyncio.to_thread(self.sglang._post, f"/{endpoint}", payload)


class SGLang:
    """Serves a checkpoint; the endpoints that drive the engine need an admin key only this holds."""

    def __init__(
        self,
        checkpoint: str,
        version: int,
        port: int,
        config: SGLangConfig,
        startup_timeout: int,
        engine_id: str,
    ):
        self.version, self.config, self.engine_id = version, config, engine_id
        # The receiver's ranks, as the latest `committed` read them.
        self.participants: list[dict] = []
        self.admin_key = secrets.token_urlsafe(32)
        self.endpoint = SGLangEndpoint(
            model_path=checkpoint,
            worker_port=port,
            extra_server_args={
                # SGLang's own default, as Miles' engines get; autoinference-utils would pick fastokens.
                "--tokenizer-backend": "huggingface",
                **config.server_args,
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

    def committed(self, plan_digest: str) -> tuple[str | None, int]:
        from miles.backends.training_utils.weight_update.protocols.gpu_delta.session import (
            committed_endpoint,
        )

        info = self._post("/get_gpu_delta_info", {"engine_id": self.engine_id})
        self.participants = [row["identity"] for row in info["participants"]]
        return committed_endpoint(info, self.participants, plan_digest)

    def apply(self, publication: dict, version: int | None) -> dict | None:
        """Miles' activation of this engine, on the receiver ranks `committed` read."""
        from miles.backends.training_utils.weight_update.protocols.gpu_delta.session import (
            activate_engine,
        )

        reply = asyncio.run(
            activate_engine(
                _Receiver(self),
                publication,
                self.participants,
                uuid.uuid4().hex,
                None if version is None else str(version),
                None,
                pause_mode=self.config.pause_mode,
            )
        )
        if version is not None:
            self.version = version
        return reply["timings"]

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
