"""``RouterState`` — the routing brain behind the router service.

Composes the gorgo package's primitives (policy registry, radix trie, online
tuner, calibration) with stitch-side replica lifecycle: discovery churn,
readiness/constraint candidate filtering from each replica's ``/server_info``,
the dispatch counters every load-aware policy reads, and the sample buffer the
online tuner scores. Pure state + logic — the HTTP layer (``routing.service``)
owns transport and calls in from a single event loop, so nothing here locks.

Counter lifecycle (mirrors the GORGO proxy): ``dispatch()`` increments the
three per-target counters at routing time and returns a release handle whose
``release()`` decrements them exactly once — the service calls it from a
``finally`` around the proxied request. On 2xx the service also calls
``record_success()`` to tag the trie and feed the tuner/calibration.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from typing import Any

from gorgo.measure import NS_PER_S
from gorgo.policy import POLICY_REGISTRY, ReplicaSnapshot, RouteContext, normalize_policy
from gorgo.policy.gorgo import make_default_store, merge_update, prune_per_target
from gorgo.radix_trie import RadixTrie
from gorgo.tuner import OnlineTuner
from stitch.versions import SyncState, VersionConstraint, VersionRef

logger = logging.getLogger(__name__)

MAX_REQUEST_SAMPLES = 1000
NETWORK_RTT_EWMA_ALPHA = 0.3

# SGLang Prometheus gauges scraped into a ReplicaSnapshot.
_METRIC_KEYS = {
    "num_running_reqs": "sglang:num_running_reqs",
    "num_queue_reqs": "sglang:num_queue_reqs",
    "num_used_tokens": "sglang:num_used_tokens",
    "gen_throughput": "sglang:gen_throughput",
    "utilization": "sglang:utilization",
}


def parse_metrics_text(text: str) -> dict[str, float]:
    """Parse a Prometheus exposition snippet into ``{metric_name: value}``,
    dropping label suffixes and skipping non-numeric or commented lines."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            value = float(parts[1])
        except ValueError:
            continue
        out[parts[0].split("{")[0]] = value
    return out


class ReleaseHandle:
    """Exactly-once decrement of the dispatch counters for one request.

    ``queued_at_dispatch`` snapshots the target's queued-token counter as the
    routing decision saw it — the calibration regressor needs the dispatch-time
    load, not whatever the counter reads when the response finishes.
    """

    __slots__ = (
        "_state",
        "target",
        "request_tokens",
        "uncached_tokens",
        "queued_at_dispatch",
        "policy",
        "_released",
    )

    def __init__(
        self,
        state: "RouterState",
        target: str,
        request_tokens: int,
        uncached: int,
        queued_at_dispatch: int,
        policy: str,
    ):
        self._state = state
        self.target = target
        self.request_tokens = request_tokens
        self.uncached_tokens = uncached
        self.queued_at_dispatch = queued_at_dispatch
        # The policy active at dispatch — samples carry it so post-hoc analysis
        # attributes a request to the policy that actually routed it, even when
        # the policy is switched live mid-flight (A/B benchmarking).
        self.policy = policy
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        s = self._state
        if self.target in s.endpoints_queued_tokens:
            s.endpoints_queued_tokens[self.target] = max(
                0, s.endpoints_queued_tokens[self.target] - self.request_tokens
            )
        if self.target in s.endpoints_queued_uncached_tokens:
            s.endpoints_queued_uncached_tokens[self.target] = max(
                0, s.endpoints_queued_uncached_tokens[self.target] - self.uncached_tokens
            )
        if self.target in s.endpoints_inflight_requests:
            s.endpoints_inflight_requests[self.target] = max(
                0, s.endpoints_inflight_requests[self.target] - 1
            )


class RouterState:
    def __init__(
        self,
        *,
        policy: str = "session-affinity",
        hyperparameters: dict[str, Any] | None = None,
        tuner: OnlineTuner | None = None,
    ) -> None:
        self.policy = normalize_policy(policy)
        if self.policy not in POLICY_REGISTRY:
            raise ValueError(f"unknown routing policy: {policy!r}")
        self.replica_urls: list[str] = []
        self.endpoints_queued_tokens: dict[str, int] = {}
        self.endpoints_queued_uncached_tokens: dict[str, int] = {}
        self.endpoints_inflight_requests: dict[str, int] = {}
        self.live_metrics: dict[str, ReplicaSnapshot] = {}
        self.replica_states: dict[str, dict] = {}  # url -> last /server_info body
        self.network_rtt_ewma: dict[str, float] = {}
        self.radix_trie = RadixTrie()
        store = make_default_store()
        if hyperparameters:
            # Config-supplied weight overrides land in the defaults branch.
            store = merge_update(store, {"defaults": dict(hyperparameters)}, replace=False)
        self.hyperparameters = store
        self.samples: deque[dict] = deque(maxlen=MAX_REQUEST_SAMPLES)
        self.tuner = tuner or OnlineTuner()
        self.total_requests = 0
        self.fallback_counts: dict[str, int] = {}

    # -- replica membership -------------------------------------------

    def set_replicas(self, urls: list[str]) -> tuple[list[str], list[str]]:
        """Reconcile the live replica set after a discovery pass. New URLs
        get zeroed counters; removed URLs drop all per-target state
        including their radix-trie tags (their KV cache is gone)."""
        new = list(dict.fromkeys(urls))
        added = [u for u in new if u not in self.endpoints_queued_tokens]
        removed = [u for u in self.replica_urls if u not in set(new)]
        self.replica_urls = new
        for u in added:
            self.endpoints_queued_tokens[u] = 0
            self.endpoints_queued_uncached_tokens[u] = 0
            self.endpoints_inflight_requests[u] = 0
        for u in removed:
            self.endpoints_queued_tokens.pop(u, None)
            self.endpoints_queued_uncached_tokens.pop(u, None)
            self.endpoints_inflight_requests.pop(u, None)
            self.live_metrics.pop(u, None)
            self.replica_states.pop(u, None)
            self.network_rtt_ewma.pop(u, None)
            self.radix_trie.remove_endpoint(u)
        if added or removed:
            prune_per_target(self.hyperparameters, set(new))
            logger.info("replicas: %d live (+%d/-%d)", len(new), len(added), len(removed))
        return added, removed

    # -- background scrape sinks ---------------------------------------

    def update_metrics(self, url: str, metrics: dict[str, float], scrape_latency: float) -> None:
        self.live_metrics[url] = ReplicaSnapshot(
            num_running_reqs=int(metrics.get(_METRIC_KEYS["num_running_reqs"], 0)),
            num_queue_reqs=int(metrics.get(_METRIC_KEYS["num_queue_reqs"], 0)),
            num_used_tokens=int(metrics.get(_METRIC_KEYS["num_used_tokens"], 0)),
            latency=scrape_latency,
            network_rtt=self.network_rtt_ewma.get(url, 0.0),
            gen_throughput=metrics.get(_METRIC_KEYS["gen_throughput"], 0.0),
            utilization=metrics.get(_METRIC_KEYS["utilization"], 0.0),
        )

    def update_rtt(self, url: str, rtt_seconds: float) -> None:
        prev = self.network_rtt_ewma.get(url)
        ewma = (
            rtt_seconds
            if prev is None
            else NETWORK_RTT_EWMA_ALPHA * rtt_seconds + (1 - NETWORK_RTT_EWMA_ALPHA) * prev
        )
        self.network_rtt_ewma[url] = ewma
        snap = self.live_metrics.get(url)
        if snap is not None:
            snap.network_rtt = ewma

    def update_server_info(self, url: str, info: dict | None) -> None:
        """Record a replica's ``/server_info`` (or ``None`` on scrape failure —
        an unreachable replica must stop passing the readiness filter)."""
        if info is None:
            self.replica_states.pop(url, None)
        else:
            self.replica_states[url] = info

    # -- routing --------------------------------------------------------

    def _applied_version(self, url: str) -> int | None:
        applied = (self.replica_states.get(url) or {}).get("applied")
        if not applied:
            return None
        try:
            return VersionRef.parse(applied).version
        except ValueError:
            return None

    def candidates(self, constraint: VersionConstraint | None) -> tuple[list[str], str | None]:
        """Filter replicas to those likely to accept the request.

        Drops replicas whose last ``/server_info`` said not-ready or ERROR,
        and — when the request carries a weight_version constraint — those
        whose applied version fails it (they would 409). A replica with no
        server_info yet is kept: assuming reachable beats stalling a fresh
        pool. If filtering empties the set, falls back to the full list and
        lets the sidecar 409 (the trainer's retry loop owns recovery);
        returns the filter tag for the effective-policy trace.
        """
        urls = self.replica_urls
        if not urls:
            return [], None
        ready = []
        for u in urls:
            info = self.replica_states.get(u)
            if info is not None and (
                not info.get("ready") or info.get("sync_state") == SyncState.ERROR.value
            ):
                continue
            ready.append(u)
        if constraint is not None and (
            constraint.min_version is not None or constraint.exact_version is not None
        ):
            satisfying = []
            for u in ready:
                applied = self._applied_version(u)
                # No server_info yet -> unknown version; keep (see above).
                if applied is None and u not in self.replica_states:
                    satisfying.append(u)
                elif constraint.satisfied_by(applied):
                    satisfying.append(u)
            ready = satisfying
        if not ready:
            return list(urls), "constraint-filter-empty"
        return ready, None

    def select(
        self,
        *,
        token_ids: list[int],
        request_tokens: int,
        affinity_key: str | None,
        constraint: VersionConstraint | None,
    ) -> tuple[str, str, dict[str, float] | None]:
        """Pick a target replica. Returns ``(target, effective_policy,
        scores)`` where ``effective_policy`` records any fallback taken
        (missing metrics, empty candidates, constraint-filter bail)."""
        if not self.replica_urls:
            raise LookupError("no replicas discovered")
        candidates, filter_tag = self.candidates(constraint)
        if len(candidates) == 1:
            tag = f"single-replica{':' + filter_tag if filter_tag else ''}"
            return candidates[0], tag, None

        pdef = POLICY_REGISTRY[self.policy]
        effective = self.policy
        if pdef.needs_metrics and any(u not in self.live_metrics for u in candidates):
            # Same degradation as the GORGO proxy: no full metrics coverage
            # yet -> random rather than systematically favoring the scraped
            # subset.
            target = random.choice(candidates)
            return target, f"random-fallback:missing-metrics:{self.policy}", None

        ctx = RouteContext(
            replica_urls=candidates,
            metrics=self.live_metrics,
            endpoints_queued_tokens=self.endpoints_queued_tokens,
            endpoints_queued_uncached_tokens=self.endpoints_queued_uncached_tokens,
            endpoints_inflight_requests=self.endpoints_inflight_requests,
            radix_trie=self.radix_trie,
            token_ids=token_ids,
            request_tokens=request_tokens,
            hyperparameters=self.hyperparameters,
            affinity_key=affinity_key,
        )
        decision = pdef.fn(ctx)
        if decision.fallback_reason:
            effective = f"random-fallback:internal:{self.policy}:{decision.fallback_reason}"
            self.fallback_counts[decision.fallback_reason] = (
                self.fallback_counts.get(decision.fallback_reason, 0) + 1
            )
        if filter_tag:
            effective = f"{effective}:{filter_tag}"
        return decision.target, effective, decision.scores

    # -- dispatch lifecycle ----------------------------------------------

    def dispatch(self, target: str, *, token_ids: list[int], request_tokens: int) -> ReleaseHandle:
        """Increment the per-target counters for a request being sent now."""
        cached = (
            self.radix_trie.cached_prefix_length(token_ids, target) if token_ids else 0
        )
        uncached = max(0, request_tokens - cached)
        queued_at_dispatch = self.endpoints_queued_tokens.get(target, 0)
        if target in self.endpoints_queued_tokens:
            self.endpoints_queued_tokens[target] += request_tokens
            self.endpoints_queued_uncached_tokens[target] += uncached
            self.endpoints_inflight_requests[target] += 1
        self.total_requests += 1
        return ReleaseHandle(self, target, request_tokens, uncached, queued_at_dispatch, self.policy)

    def record_success(
        self,
        handle: ReleaseHandle,
        *,
        token_ids: list[int],
        ttft_ns: int | None,
        total_ns: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        """Post-completion bookkeeping for a 2xx response: tag the trie with
        the prompt (the target's KV cache now holds it), append a tuning
        sample, and feed the calibration accumulator + online tuner."""
        if token_ids:
            self.radix_trie.insert(token_ids, endpoint=handle.target)

        if (
            ttft_ns is None
            or prompt_tokens is None
            or prompt_tokens <= 0
            or completion_tokens is None
            or completion_tokens <= 0
        ):
            return

        snap = self.live_metrics.get(handle.target)
        if snap is None:
            ping_rtt_s = 0.0
        elif snap.network_rtt > 0.0:
            ping_rtt_s = snap.network_rtt
        else:
            ping_rtt_s = snap.latency

        ttft_s = ttft_ns / NS_PER_S
        total_s = total_ns / NS_PER_S
        # Cached-at-dispatch was measured before this request's own insert.
        uncached = max(1, handle.uncached_tokens)
        self.samples.append(
            {
                "ping_seconds": ping_rtt_s,
                "ttft_seconds": ttft_s,
                "total_seconds": total_s,
                "prefill_seconds": max(ttft_s - ping_rtt_s, 0.0),
                "decode_seconds": max(total_s - ttft_s, 0.0),
                "prompt_tokens": prompt_tokens,
                "uncached_tokens": uncached,
                "completion_tokens": completion_tokens,
                "target": handle.target,
                "policy": handle.policy,
                "recorded_at_monotonic": time.monotonic(),
            }
        )
        if self.tuner.mode == "calibrate" and self.tuner.enabled:
            self.tuner.calibration.add(
                target=handle.target,
                uncached_at_dispatch=handle.uncached_tokens,
                queued_at_dispatch=handle.queued_at_dispatch,
                ttft_ms=ttft_s * 1000.0,
            )
        new_store = self.tuner.on_sample(self.samples, self.hyperparameters, policy=self.policy)
        if new_store is not None:
            self.hyperparameters = new_store

    # -- diagnostics ------------------------------------------------------

    def stats_payload(self) -> dict:
        return {
            "policy": self.policy,
            "replicas": {
                u: {
                    "queued_tokens": self.endpoints_queued_tokens.get(u, 0),
                    "queued_uncached_tokens": self.endpoints_queued_uncached_tokens.get(u, 0),
                    "inflight_requests": self.endpoints_inflight_requests.get(u, 0),
                    "network_rtt_s": self.network_rtt_ewma.get(u),
                    "ready": (self.replica_states.get(u) or {}).get("ready"),
                    "applied": (self.replica_states.get(u) or {}).get("applied"),
                    "sync_state": (self.replica_states.get(u) or {}).get("sync_state"),
                    "metrics": (
                        {
                            "num_running_reqs": s.num_running_reqs,
                            "num_queue_reqs": s.num_queue_reqs,
                            "num_used_tokens": s.num_used_tokens,
                            "gen_throughput": s.gen_throughput,
                            "utilization": s.utilization,
                            "scrape_latency_s": s.latency,
                        }
                        if (s := self.live_metrics.get(u)) is not None
                        else None
                    ),
                }
                for u in self.replica_urls
            },
            "radix_trie": {
                "num_sequences": self.radix_trie.num_sequences,
                "total_tokens_inserted": self.radix_trie.total_tokens_inserted,
                "unique_tokens": self.radix_trie.unique_token_count(),
                "nodes": self.radix_trie.node_count(),
            },
            "hyperparameters": self.hyperparameters,
            "total_requests": self.total_requests,
            "fallback_counts": dict(self.fallback_counts),
            "buffered_samples": len(self.samples),
        }
