"""The rollout router: a policy-driven load balancer in front of a Pool's replicas.

``create_router_app`` builds the ASGI app; ``serve_router`` is the uvicorn
entrypoint. The router discovers replicas through the ``Pool`` port, scrapes
each one's ``/metrics`` (SGLang gauges, forwarded by the sidecar), ``/server_info``
(readiness, applied version), and RTT (timed ``/health``, answered by the
sidecar locally), then routes every rollout request via the configured gorgo
policy and streams the response back.

Division of labor with the per-replica sidecar (``stitch.service``): the router
forwards bodies untouched — ``weight_version`` popping, request/response
version stamping, and the 409 constraint gate all stay in the sidecar. The
router only *pre-filters* replicas that would 409 (using scraped server_info),
so the trainer's existing retry loop remains the recovery path.

Admin endpoints live under ``/router/*`` so the proxy surface stays
transparent to gateway-shaped clients (probes hitting ``/server_info`` etc.).

No ``from __future__ import annotations`` here for the same reason as
``stitch.service``: FastAPI introspects the handlers' ``Request`` annotation
at runtime and can't resolve a stringized create_router_app-local import.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Callable

from gorgo.measure import consume_sse_stream
from gorgo.policy import POLICY_REGISTRY, normalize_policy
from gorgo.policy.gorgo import merge_update, validate_update
from gorgo.tuner import OnlineTuner
from stitch.pools.base import Pool
from stitch.routing.state import RouterState, parse_metrics_text
from stitch.routing.tokens import ROUTED_PATHS, extract_token_ids
from stitch.versions import VersionConstraint

logger = logging.getLogger(__name__)

DEFAULT_AFFINITY_HEADER = "Modal-Session-ID"
SCRAPE_TIMEOUT_SECONDS = 2.0

# Hop-by-hop / rewritten headers the router never forwards upstream.
_DROP_HEADERS = {"host", "content-length", "connection"}


def _usage_tokens(data: Any) -> tuple[int | None, int | None]:
    """Pull (prompt_tokens, completion_tokens) out of a buffered JSON response
    body — OpenAI ``usage`` or SGLang ``meta_info`` shape, top-level or on the
    first batch element."""
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return None, None
    for block in (data.get("usage"), data.get("meta_info")):
        if isinstance(block, dict):
            pt, ct = block.get("prompt_tokens"), block.get("completion_tokens")
            if isinstance(pt, int) and isinstance(ct, int):
                return pt, ct
    return None, None


def create_router_app(
    pool: Pool,
    *,
    policy: str = "session-affinity",
    affinity_header: str = DEFAULT_AFFINITY_HEADER,
    tokenizer_factory: Callable[[], Any] | None = None,
    hyperparameters: dict[str, float] | None = None,
    tuner_config: dict[str, Any] | None = None,
    metrics_interval: float = 30.0,
    discovery_interval: float = 30.0,
    upstream_timeout: float | None = 3600.0,
    upstream_transport: Any | None = None,
):
    """The rollout router app. ``tuner_config`` is a ``POST /router/tune``-shaped
    dict applied at startup (e.g. ``{"enabled": True, "objective_metric":
    "neg_p95_e2e"}``); the tuner stays off when omitted. ``upstream_transport``
    is a test seam: an ``httpx`` transport routing "replica" traffic to a stub
    ASGI app instead of the network."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    import httpx

    state = RouterState(policy=policy, hyperparameters=hyperparameters)
    if tuner_config:
        err = state.tuner.configure(
            dict(tuner_config),
            active_policy=state.policy,
            current_defaults=state.hyperparameters["defaults"],
        )
        if err:
            raise ValueError(f"invalid tuner_config: {err}")

    timeout = httpx.Timeout(upstream_timeout, connect=10.0)
    pooled: dict[str, Any] = {}

    def client() -> Any:
        c = pooled.get("client")
        if c is None:
            c = httpx.AsyncClient(
                timeout=timeout,
                trust_env=False,
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
                transport=upstream_transport,
            )
            pooled["client"] = c
        return c

    def tokenizer() -> Any | None:
        return pooled.get("tokenizer")

    async def _scrape_one(url: str) -> None:
        c = client()
        # RTT probe: the sidecar answers /health locally, so this times the
        # network leg without touching the engine.
        try:
            t0 = time.perf_counter_ns()
            r = await c.get(f"{url}/health", timeout=SCRAPE_TIMEOUT_SECONDS)
            r.raise_for_status()
            state.update_rtt(url, (time.perf_counter_ns() - t0) / 1e9)
        except Exception:
            pass
        try:
            r = await c.get(f"{url}/server_info", timeout=SCRAPE_TIMEOUT_SECONDS)
            state.update_server_info(url, r.json() if r.is_success else None)
        except Exception:
            state.update_server_info(url, None)
        try:
            t0 = time.perf_counter_ns()
            r = await c.get(f"{url}/metrics", timeout=SCRAPE_TIMEOUT_SECONDS)
            if r.is_success:
                state.update_metrics(
                    url, parse_metrics_text(r.text), (time.perf_counter_ns() - t0) / 1e9
                )
        except Exception:
            pass  # keep the stale snapshot; the local counters bridge the gap

    async def _scrape_all() -> None:
        urls = list(state.replica_urls)
        if urls:
            await asyncio.gather(*(_scrape_one(u) for u in urls))

    async def _discover() -> None:
        try:
            urls = await asyncio.to_thread(pool.discover_replicas)
            state.set_replicas(urls)
        except Exception:
            logger.exception("replica discovery failed; keeping current set")

    async def _discovery_loop() -> None:
        while True:
            await asyncio.sleep(discovery_interval)
            await _discover()

    async def _scrape_loop() -> None:
        while True:
            await asyncio.sleep(metrics_interval)
            await _scrape_all()

    @asynccontextmanager
    async def lifespan(_app: Any):
        if tokenizer_factory is not None:
            pooled["tokenizer"] = await asyncio.to_thread(tokenizer_factory)
        # Prime discovery + one scrape before serving so the first request
        # doesn't random-fallback on missing metrics.
        await _discover()
        await _scrape_all()
        tasks = [
            asyncio.create_task(_discovery_loop()),
            asyncio.create_task(_scrape_loop()),
        ]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: PERF203
                    pass
            c = pooled.pop("client", None)
            if c is not None:
                await c.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "replicas": len(state.replica_urls)}

    # ---- admin surface under /router/* --------------------------------

    @app.get("/router/replicas")
    async def router_replicas() -> dict[str, Any]:
        return {
            "replicas": list(state.replica_urls),
            "server_info": state.replica_states,
        }

    @app.get("/router/stats")
    async def router_stats() -> dict[str, Any]:
        return state.stats_payload()

    @app.get("/router/policy")
    async def get_router_policy() -> dict[str, Any]:
        return {"policy": state.policy, "available": sorted(POLICY_REGISTRY)}

    @app.post("/router/policy")
    async def post_router_policy(request: Request) -> Any:
        data = await request.json()
        name = normalize_policy(str((data or {}).get("policy", "")))
        if name not in POLICY_REGISTRY:
            return JSONResponse(
                {"error": f"unknown policy {name!r}", "available": sorted(POLICY_REGISTRY)},
                status_code=400,
            )
        state.policy = name
        return {"policy": state.policy}

    @app.get("/router/hyperparameters")
    async def get_hyperparameters() -> dict[str, Any]:
        return {"hyperparameters": state.hyperparameters}

    @app.api_route("/router/hyperparameters", methods=["POST", "PUT", "PATCH"])
    async def post_hyperparameters(request: Request) -> Any:
        data = await request.json()
        update, err = validate_update(data, known_targets=set(state.replica_urls))
        if err:
            return JSONResponse({"error": err}, status_code=400)
        state.hyperparameters = merge_update(
            state.hyperparameters, update, replace=request.method == "PUT"
        )
        return {"hyperparameters": state.hyperparameters}

    @app.get("/router/tune")
    async def get_tune() -> dict[str, Any]:
        return {"auto_tune": state.tuner.status(buffered_samples=len(state.samples))}

    @app.post("/router/tune")
    async def post_tune(request: Request) -> Any:
        data = await request.json()
        err = state.tuner.configure(
            data if isinstance(data, dict) else {},
            active_policy=state.policy,
            current_defaults=state.hyperparameters["defaults"],
        )
        if err:
            return JSONResponse({"error": err}, status_code=400)
        return {"auto_tune": state.tuner.status(buffered_samples=len(state.samples))}

    @app.get("/router/samples")
    async def get_samples() -> dict[str, Any]:
        return {
            "buffered_samples": len(state.samples),
            "recent": list(state.samples)[-50:],
        }

    @app.get("/router/calibrated_rates")
    async def get_calibrated_rates() -> dict[str, Any]:
        return {"calibrated_rates": state.tuner.calibration.rates()}

    @app.post("/router/flush")
    async def post_flush() -> dict[str, Any]:
        # Clears the router's own indexes only. Replica KV caches are owned
        # by each sidecar's reconciler — never flushed from here.
        state.radix_trie.clear()
        state.samples.clear()
        state.tuner.samples_since_last_apply = 0
        return {"radix_trie_cleared": True, "samples_cleared": True}

    # ---- the proxy ------------------------------------------------------

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(path: str, request: Request) -> Response:
        route = path.strip("/")
        body = await request.body()
        payload: dict[str, Any] | None = None
        if body and request.headers.get("content-type", "").startswith("application/json"):
            parsed = await request.json()
            payload = parsed if isinstance(parsed, dict) else None

        is_routed = route in ROUTED_PATHS
        constraint = VersionConstraint.from_payload(payload) if is_routed else None
        token_ids: list[int] = []
        request_tokens = 0
        if is_routed:
            token_ids, request_tokens = extract_token_ids(route, payload, tokenizer())

        try:
            target, effective_policy, _scores = state.select(
                token_ids=token_ids,
                request_tokens=request_tokens,
                affinity_key=request.headers.get(affinity_header),
                constraint=constraint,
            )
        except LookupError:
            return JSONResponse(
                {"error": {"type": "NoReplicas", "message": "no rollout replicas discovered"}},
                status_code=503,
            )

        handle = (
            state.dispatch(target, token_ids=token_ids, request_tokens=request_tokens)
            if is_routed
            else None
        )
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP_HEADERS}
        headers["x-stitch-router-policy"] = effective_policy
        request_start_ns = time.perf_counter_ns()

        upstream = client().build_request(
            request.method,
            f"{target}/{path}",
            content=body,  # untouched: the sidecar owns weight_version/stamping
            headers=headers,
            params=request.query_params,
        )
        try:
            resp = await client().send(upstream, stream=True)
        except Exception as exc:
            if handle is not None:
                handle.release()
            return JSONResponse(
                {"error": {"type": "UpstreamUnreachable", "message": str(exc)[:200]}},
                status_code=502,
            )

        content_type = resp.headers.get("content-type", "")

        if content_type.startswith("text/event-stream"):
            # Tee: bytes go to the client the moment they arrive; the same
            # chunks feed the SSE parser for TTFT/usage without adding
            # latency (gorgo's proxy pattern).
            queue: asyncio.Queue[bytes | None] = asyncio.Queue()

            async def sink(chunk: bytes) -> None:
                await queue.put(chunk)

            async def parse() -> None:
                try:
                    ttft_ns, _out, ptok, ctok, _meta = await consume_sse_stream(
                        resp, request_start_ns=request_start_ns, chunk_sink=sink
                    )
                    if handle is not None and resp.status_code < 300:
                        state.record_success(
                            handle,
                            token_ids=token_ids,
                            ttft_ns=ttft_ns,
                            total_ns=time.perf_counter_ns() - request_start_ns,
                            prompt_tokens=ptok,
                            completion_tokens=ctok,
                        )
                finally:
                    await queue.put(None)

            async def stream_body():
                parse_task = asyncio.create_task(parse())
                try:
                    while True:
                        chunk = await queue.get()
                        if chunk is None:
                            break
                        yield chunk
                    await parse_task
                finally:
                    parse_task.cancel()
                    if handle is not None:
                        handle.release()
                    await resp.aclose()

            return StreamingResponse(
                stream_body(), status_code=resp.status_code, media_type="text/event-stream"
            )

        # Buffered path (the stitch sidecar always buffers): read fully,
        # account, and relay. TTFT collapses to E2E here — which is why the
        # router's tuner objective defaults to an E2E metric.
        try:
            data = await resp.aread()
            total_ns = time.perf_counter_ns() - request_start_ns
            if handle is not None and resp.status_code < 300:
                parsed_body: Any = None
                if content_type.startswith("application/json"):
                    try:
                        parsed_body = resp.json()
                    except ValueError:
                        parsed_body = None
                ptok, ctok = _usage_tokens(parsed_body)
                state.record_success(
                    handle,
                    token_ids=token_ids,
                    ttft_ns=total_ns,
                    total_ns=total_ns,
                    prompt_tokens=ptok,
                    completion_tokens=ctok,
                )
            return Response(
                content=data,
                status_code=resp.status_code,
                media_type=content_type or None,
            )
        finally:
            if handle is not None:
                handle.release()
            await resp.aclose()

    return app


def serve_router(
    pool: Pool,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    **kwargs: Any,
) -> None:
    """Run the router under uvicorn. ``kwargs`` forward to ``create_router_app``;
    the deployment supplies the concrete Pool (and tokenizer factory)."""
    import uvicorn

    uvicorn.run(create_router_app(pool, **kwargs), host=host, port=port, log_level="info")
