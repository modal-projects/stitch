"""Shared rollout-Router logic: run the stitch routing service in front of the
Server pool, and the ``python -m cookbook.common.router`` entrypoint it launches.

Same split as ``common/server.py``: Modal requires the ``@app.cls`` skeleton at
module scope in each framework's ``app.py``; its ``@enter``/``@exit`` delegate
here. The router is a single always-on CPU container — the radix trie, load
counters, and online tuner are in-process state, so the class MUST be pinned to
``min_containers=1, max_containers=1``.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from . import process
from .constants import SIDECAR_PORT

ROUTER_MODULE = "cookbook.common.router"


def start_router(
    *,
    router_port: int,
    app_name: str,
    server_cls: str,
    model_name: str | None,
    policy: str,
    hyperparameters: dict[str, float] | None = None,
    tuner: dict[str, Any] | None = None,
) -> subprocess.Popen:
    """Launch the routing service as a subprocess (mirrors process.start_sidecar)."""
    cmd = [
        "python3", "-m", ROUTER_MODULE,
        "--host", "0.0.0.0", "--port", str(router_port),
        "--app-name", app_name,
        "--server-cls", server_cls,
        "--policy", policy,
    ]
    if model_name:
        cmd += ["--model", model_name]
    if hyperparameters:
        cmd += ["--hyperparameters", json.dumps(hyperparameters)]
    if tuner:
        cmd += ["--tuner", json.dumps(tuner)]
    print("Starting router:", " ".join(cmd))
    return subprocess.Popen(cmd, start_new_session=True)


def serve_router_startup(
    replica: Any,
    *,
    app_name: str,
    server_cls: str,
    model_name: str | None,
    policy: str,
    hyperparameters: dict[str, float] | None,
    tuner: dict[str, Any] | None,
    startup_timeout: int,
) -> None:
    """Start the router on the Router replica (from ``@modal.enter``)."""
    replica.router = start_router(
        router_port=SIDECAR_PORT, app_name=app_name, server_cls=server_cls,
        model_name=model_name, policy=policy, hyperparameters=hyperparameters, tuner=tuner,
    )
    process.wait_http(f"http://127.0.0.1:{SIDECAR_PORT}/health", replica.router, startup_timeout)
    print(f"Rollout router ready: policy={policy}, pool={app_name}.{server_cls}")


def serve_router_stop(replica: Any) -> None:
    """Tear down the router (from ``@modal.exit``)."""
    process.terminate_process(getattr(replica, "router", None))


def main() -> None:
    import argparse

    from stitch.pools.modal_flash import ModalFlashPool
    from stitch.pools.union import UnionPool
    from stitch.routing import serve_router

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=SIDECAR_PORT)
    p.add_argument("--app-name", required=True)
    p.add_argument("--server-cls", default="Server",
                   help="comma-separated for a multi-class (e.g. per-region) pool")
    p.add_argument("--policy", default="session-affinity")
    p.add_argument("--model", default=None, help="HF id for prompt tokenization (omit for byte fallback)")
    p.add_argument("--hyperparameters", default=None, help="JSON dict of gorgo weight overrides")
    p.add_argument("--tuner", default=None, help="JSON dict, POST /router/tune body shape")
    args = p.parse_args()

    tokenizer_factory = None
    if args.model:
        def tokenizer_factory():  # lazy: transformers import + HF cache read at startup
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(args.model)

    members = [ModalFlashPool(args.app_name, cls.strip()) for cls in args.server_cls.split(",") if cls.strip()]
    pool = members[0] if len(members) == 1 else UnionPool(members)
    serve_router(
        pool,
        host=args.host, port=args.port,
        policy=args.policy,
        tokenizer_factory=tokenizer_factory,
        hyperparameters=json.loads(args.hyperparameters) if args.hyperparameters else None,
        tuner_config=json.loads(args.tuner) if args.tuner else None,
    )


if __name__ == "__main__":
    main()
