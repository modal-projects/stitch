import pytest

from stitch.routing.state import RouterState, parse_metrics_text
from stitch.versions import VersionConstraint

A, B, C = "http://a", "http://b", "http://c"

METRICS_TEXT = """\
# HELP sglang:num_running_reqs running
sglang:num_running_reqs{model="m"} 3
sglang:num_queue_reqs{model="m"} 1
sglang:num_used_tokens{model="m"} 512
sglang:gen_throughput{model="m"} 42.5
sglang:utilization{model="m"} 0.7
not a metric line
sglang:bogus{x="y"} nope
"""


def ready_info(version: int = 1, *, ready: bool = True, sync_state: str = "IDLE") -> dict:
    return {
        "ready": ready,
        "applied": f"run-x/weight_v{version:06d}",
        "sync_state": sync_state,
        "active_requests": 0,
    }


def make_state(policy="least-request", urls=(A, B)) -> RouterState:
    s = RouterState(policy=policy)
    s.set_replicas(list(urls))
    for u in urls:
        s.update_rtt(u, 0.01)
        s.update_metrics(u, parse_metrics_text(METRICS_TEXT), scrape_latency=0.02)
        s.update_server_info(u, ready_info())
    return s


def test_parse_metrics_text():
    m = parse_metrics_text(METRICS_TEXT)
    assert m["sglang:num_running_reqs"] == 3
    assert m["sglang:utilization"] == 0.7
    assert "sglang:bogus" not in m


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        RouterState(policy="nonsense")


def test_set_replicas_churn():
    s = make_state()
    s.radix_trie.insert([1, 2, 3], endpoint=A)
    s.hyperparameters["per_target"][A] = {"rtt_weight": 2.0}
    added, removed = s.set_replicas([B, C])
    assert added == [C] and removed == [A]
    assert A not in s.endpoints_queued_tokens
    assert s.endpoints_queued_tokens[C] == 0
    assert s.radix_trie.cached_prefix_length([1, 2, 3], A) == 0
    assert A not in s.hyperparameters["per_target"]


def test_dispatch_and_release_exactly_once():
    s = make_state()
    h = s.dispatch(A, token_ids=[1, 2, 3], request_tokens=3)
    assert s.endpoints_queued_tokens[A] == 3
    assert s.endpoints_inflight_requests[A] == 1
    h.release()
    h.release()  # idempotent
    assert s.endpoints_queued_tokens[A] == 0
    assert s.endpoints_inflight_requests[A] == 0


def test_dispatch_uncached_accounting_uses_trie():
    s = make_state()
    s.radix_trie.insert([1, 2, 3, 4], endpoint=A)
    h = s.dispatch(A, token_ids=[1, 2, 3, 4, 5, 6], request_tokens=6)
    assert h.uncached_tokens == 2
    assert s.endpoints_queued_uncached_tokens[A] == 2
    h.release()
    assert s.endpoints_queued_uncached_tokens[A] == 0


def test_release_survives_replica_removal():
    s = make_state()
    h = s.dispatch(A, token_ids=[], request_tokens=10)
    s.set_replicas([B])
    h.release()  # must not raise or resurrect counters
    assert A not in s.endpoints_queued_tokens


def test_candidates_filters_not_ready_and_error():
    s = make_state(urls=(A, B, C))
    s.update_server_info(A, ready_info(ready=False))
    s.update_server_info(B, ready_info(sync_state="ERROR", ready=True))
    cands, tag = s.candidates(None)
    assert cands == [C] and tag is None


def test_candidates_filters_constraint():
    s = make_state(urls=(A, B))
    s.update_server_info(A, ready_info(version=3))
    s.update_server_info(B, ready_info(version=7))
    cands, tag = s.candidates(VersionConstraint(min_version=5))
    assert cands == [B] and tag is None
    # Exact pin no replica satisfies -> unfiltered fallback with tag.
    cands, tag = s.candidates(VersionConstraint(exact_version=99))
    assert cands == [A, B] and tag == "constraint-filter-empty"


def test_candidates_keeps_unscraped_replicas():
    s = make_state(urls=(A, B))
    s.update_server_info(B, None)  # scrape failed -> unknown
    # B was dropped from replica_states entirely, so it's treated as
    # possibly-fine (fresh pool case) rather than excluded.
    cands, _ = s.candidates(VersionConstraint(min_version=1))
    assert set(cands) == {A, B}


def test_select_single_candidate_short_circuits():
    s = make_state(urls=(A, B))
    s.update_server_info(B, ready_info(ready=False))
    target, effective, scores = s.select(
        token_ids=[], request_tokens=0, affinity_key=None, constraint=None
    )
    assert target == A and effective.startswith("single-replica")


def test_select_missing_metrics_falls_back_random():
    s = RouterState(policy="gorgo")
    s.set_replicas([A, B])
    target, effective, _ = s.select(
        token_ids=[1], request_tokens=1, affinity_key=None, constraint=None
    )
    assert target in (A, B)
    assert effective.startswith("random-fallback:missing-metrics")


def test_select_gorgo_prefers_cached_replica():
    s = make_state(policy="gorgo")
    prompt = list(range(500))
    s.radix_trie.insert(prompt, endpoint=B)
    target, effective, scores = s.select(
        token_ids=prompt, request_tokens=len(prompt), affinity_key=None, constraint=None
    )
    assert target == B and effective == "gorgo"
    assert scores is not None and scores[B] < scores[A]


def test_select_session_affinity_sticky():
    s = make_state(policy="session-affinity", urls=(A, B, C))
    t1, eff, _ = s.select(token_ids=[], request_tokens=0, affinity_key="k1", constraint=None)
    t2, _, _ = s.select(token_ids=[], request_tokens=0, affinity_key="k1", constraint=None)
    assert t1 == t2 and eff == "session-affinity"


def test_select_no_replicas_raises():
    s = RouterState()
    with pytest.raises(LookupError):
        s.select(token_ids=[], request_tokens=0, affinity_key=None, constraint=None)


def test_record_success_inserts_trie_and_buffers_sample():
    s = make_state(policy="gorgo")
    prompt = [1, 2, 3, 4]
    h = s.dispatch(A, token_ids=prompt, request_tokens=4)
    s.record_success(
        h,
        token_ids=prompt,
        ttft_ns=50_000_000,
        total_ns=200_000_000,
        prompt_tokens=4,
        completion_tokens=10,
    )
    h.release()
    assert s.radix_trie.cached_prefix_length(prompt, A) == 4
    assert len(s.samples) == 1
    sample = s.samples[0]
    assert sample["target"] == A and sample["ttft_seconds"] == pytest.approx(0.05)
    assert sample["policy"] == "gorgo"  # dispatch-time policy, not completion-time


def test_sample_policy_survives_live_flip():
    s = make_state(policy="gorgo")
    h = s.dispatch(A, token_ids=[1], request_tokens=1)
    s.policy = "session-affinity"  # A/B driver flips mid-flight
    s.record_success(
        h, token_ids=[1], ttft_ns=1_000_000, total_ns=2_000_000, prompt_tokens=1, completion_tokens=1
    )
    h.release()
    assert s.samples[-1]["policy"] == "gorgo"


def test_record_success_skips_sample_without_usage():
    s = make_state()
    h = s.dispatch(A, token_ids=[1], request_tokens=1)
    s.record_success(
        h, token_ids=[1], ttft_ns=None, total_ns=1, prompt_tokens=None, completion_tokens=None
    )
    assert len(s.samples) == 0
    assert s.radix_trie.cached_prefix_length([1], A) == 1  # trie still tagged


def test_tuner_writes_weights_through_record_success():
    s = make_state(policy="gorgo")
    assert (
        s.tuner.configure(
            {"window_size": 4, "hop_size": 2},
            active_policy="gorgo",
            current_defaults=s.hyperparameters["defaults"],
        )
        is None
    )
    before = dict(s.hyperparameters["defaults"])
    for i in range(12):
        h = s.dispatch(A, token_ids=[i], request_tokens=1)
        s.record_success(
            h,
            token_ids=[i],
            ttft_ns=10_000_000 + i,
            total_ns=20_000_000 + i,
            prompt_tokens=1,
            completion_tokens=2,
        )
        h.release()
    assert s.tuner.applied_count >= 2
    assert s.hyperparameters["defaults"] != before


def test_stats_payload_shape():
    s = make_state()
    s.dispatch(A, token_ids=[1, 2], request_tokens=2)
    p = s.stats_payload()
    assert p["replicas"][A]["queued_tokens"] == 2
    assert p["replicas"][A]["ready"] is True
    assert p["replicas"][A]["metrics"]["num_running_reqs"] == 3
    assert p["radix_trie"]["num_sequences"] == 0
    assert p["total_requests"] == 1
