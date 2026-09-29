from tools.fleet.traffic import SHAPES, _common_prefix_len, summarize


def test_rl_agentic_shape_exercises_long_sessions() -> None:
    shape = SHAPES["rl_agentic"]

    assert shape.turns == (40, 80)
    assert shape.tool_tokens[0] > 0
    assert shape.think_seconds[0] > 0
    assert shape.top_p == 0.95
    assert shape.top_k == 1024
    assert shape.return_routed_experts
    assert shape.track_token_ids


def test_fixed_decode_shape_holds_generated_work_constant() -> None:
    shape = SHAPES["fixed_decode"]

    assert shape.prompt_tokens[0] == shape.prompt_tokens[1]
    assert shape.max_tokens[0] == shape.max_tokens[1]
    assert shape.ignore_eos


def test_common_prefix_len_uses_exact_token_identity() -> None:
    assert _common_prefix_len([1, 2, 3], [1, 2, 4, 5]) == 2
    assert _common_prefix_len([1, 2], [1, 2, 3]) == 2


def test_summary_reports_useful_token_throughput() -> None:
    rows = [
        {
            "shape": "rl_agentic",
            "latency": 2.0,
            "status": 200,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "response_bytes": 400,
            "json_parse_s": 0.1,
            "sglang_e2e_latency_s": 1.5,
        },
        {
            "shape": "rl_agentic",
            "latency": 4.0,
            "status": 500,
            "error": "failed",
        },
    ]

    result = summarize(rows, elapsed=10.0)

    assert result["successful_requests"] == 1
    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 20
    assert result["requests_per_s"] == 0.1
    assert result["completion_tokens_per_s"] == 2.0
    assert result["total_tokens_per_s"] == 12.0
    assert result["rl_agentic"]["response_bytes_total"] == 400
    assert result["rl_agentic"]["response_bytes_per_request"] == 400
    assert result["rl_agentic"]["json_parse_seconds_total"] == 0.1
    assert result["rl_agentic"]["sglang_e2e_seconds_total"] == 1.5
