# Sampler throughput benchmark

This benchmark answers one question for each (GPU, weight precision) sampler
configuration: how much output can one GPU sustain on our agentic SWE workload while
turn latency stays under a fixed bound (p90 10 s or 20 s by default)? The model is Qwen3.6-35B-A3B. Each
configuration is tuned fairly over a few serving variants.

The results feed one blog figure, with two numbers per configuration:

- output throughput per GPU, relative to B200 BF16;
- relative cost per token at Modal list prices.

| File | What it holds |
| --- | --- |
| `configs.py` | The matrix: 13 rows (12 (GPU, precision) pairs plus B200 BF16 with an FP8 KV cache), each with 2–4 tuning variants and its list price. |
| `replay.py` | The closed-loop load generator: pure Python and httpx. |
| `sweep.py` | The load ramp, stop rules, CSV rows, and the throughput-at-speed summary. It also has a CLI. |
| `app.py` | The Modal app: engines of one configuration (one Server per variant) and a CPU driver per variant. |
| `launch.py` | Deploys one configuration, sweeps its variants in parallel, and stops the app. `--plan` previews a launch. |
| `fake_server.py` | A stand-in for SGLang (streamed or whole responses), used by the tests and by local dry runs. |

Traces come from the agent's opt-in trajectory dump, `MODAL_SWE_TRAJECTORY_DUMP_DIR` in
`modal_swe/agent.py`.

## Method

**Why replay recorded trajectories.** The rollout workload has a shape that synthetic
fixed-length benchmarks miss:

- contexts run to tens of thousands of tokens and grow every turn;
- each turn writes only hundreds to a few thousand tokens;
- nearly every prompt token was already cached by the previous turn of the same episode;
- the hybrid Gated DeltaNet layers keep per-request "mamba" state beside the KV cache.

Replaying real episodes reproduces the distributions of context length, turn length and
cache reuse.

Each request asks for exactly the recorded number of tokens: `max_tokens` is the
recorded completion length, with `ignore_eos`. So every configuration decodes the same
token counts whatever text its precision samples, and all configurations are compared on
identical work.

Sampling is temperature 1.0, top-p 0.97, top-k 64. Every request returns per-token
logprobs, as rollout requests do; `--no-logprobs` turns that off.

**Why the generated outputs go back into the context.** Turn k carries:

- the recorded system, user and tool messages;
- the assistant outputs this replay generated, not the recorded ones.

So the engine's prefix cache sees the tokens it produced itself, as it does under the
agent. A turn whose response the agent rejected (a format error) never entered the
context, and it does not enter it here either.

**Why sessions start mid-trajectory.** An episode runs for tens of minutes, but a
measurement point lasts five. If every session started at turn 0, a point would see
only short contexts.

- A new session starts at a uniformly random turn of a random trajectory, with the
  recorded context up to that turn. Long contexts are therefore present from the first
  minute.
- When a trajectory ends, the session moves on to another one, starting from its first
  turn, as a new episode would.
- Each trajectory run puts a unique tag line at the start of its task message. Sessions
  then share the system prompt and the tools, as real episodes do, and nothing after
  them. Without the tag, two sessions replaying one trajectory would share recorded
  prefixes that no two real episodes share.

**Why a bound on turn latency.** Batching more sessions raises an engine's output until
its KV cache fills, at the cost of each turn's latency. In rollout, turn latency adds
directly to how long an episode takes. So configurations are compared at equal per-turn
latency, as a service-level objective: the 90th-percentile time from sending a turn's
request to its last token, which includes queueing and prefill. There are two default
bounds:

- 10 s at p90, latency-bound;
- 20 s at p90, throughput-bound.

For reference, B200 BF16 ran 3.8-5.5 s at p90 with 8-16 sessions per engine and 38 s
once its KV cache was full at 32 (validation, 2026-10-07).

An agent turn writes only about 77 tokens, so per-request decode speed (first streamed
token to last) is a short, noisy measurement and is no longer the default.
`--speed-metric` still accepts `decode_tok_s_p50` and `e2e_tok_s_p50` (higher is faster;
targets in tok/s; both need `--stream`) and `latency_s_p50` (lower is faster; targets in
seconds).

**Token prompts by default, as training sends turns.** Miles' session server sends each
turn's prompt as token IDs (TITO) with the messages, so SGLang never re-renders or
re-tokenizes the conversation, and reads the completion's IDs from `meta_info`. The
replay does the same: it renders
and tokenizes a trajectory's context once, with the served checkpoint's own tokenizer and
chat template (`preserve_thinking`, as Miles' Qwen3.6 TITO tokenizer sets it), then adds
each completion's IDs (from `meta_info`), the template's end of turn, and the newly
appended messages, tokenized on their own. On the 36 recorded trajectories (5,550 turns)
the token prompts come within 8-28 tokens of the prompt lengths training recorded, and
cost the client under 1 ms per turn; each window reports `prompt_vs_recorded`.
`--text-prompts` sends messages alone instead. Then each engine re-tokenizes about 45K
tokens per turn on one CPU core, which capped B300 NVFP4 at about 4.7 turns/s per
engine, KV cache 4% full, whatever the session count (2026-10-07).

**Engine CPU per turn is checked against training.** SGLang's main process handles
every request on one core, so a request shape that costs it more CPU than training's
caps fast GPUs below what they serve in training. Each point reports
`server_cpu_s_per_request`; a live B300 NVFP4 training engine spent about 0.011 s per
turn (5.4 turns/s, 2026-10-07). Asking for 128 OpenAI-format top log-probabilities per
output token (`--top-logprobs 128`) cost about 0.66 s per turn and capped B300 NVFP4 at
1.5 turns/s, so the default asks for none.

**Whole responses by default.** The replay reads each response whole, as the training
agent does (LiteLLM without streaming), which costs the client one parse per request
instead of one per token. Streaming every token kept one driver from loading four B200
engines: its event loop woke 0.46-0.72 s late and output per GPU came out ~25% low
(2026-10-07). `--stream` restores streaming, with TTFT and per-request decode speed.

**Load model.** The loop is closed: N concurrent sessions per engine, with zero think
time, so N is the number of concurrent requests. Tool execution is left out because it
is not sampler work.

This makes the benchmark's session counts the engine's concurrent requests, not the
fleet's sessions per engine. The fleet's sessions spend part of their time in tool
calls, so a real engine serves more sessions than the concurrency measured here.

**One point of the ramp.** For each session count per engine (`--sessions`; by default
8, 16, 20, 24, 28, 32, 48, 64, 96, 128, 192, 256, capped at the variant's batch
ceiling; the 4-session steps resolve the KV-cache cliff that B200 BF16 hits between 16 and
32), a point
runs four steps:

1. Grow the running replay to the new session count. Existing sessions keep their place
   in their trajectories.
2. Warm up for at least 2 minutes. Warm-up continues until every new session has
   finished its first request, which is its long first prefill, for at most 10 minutes.
3. Measure one 3-minute window.
4. Write one CSV row.

**What a window counts.**

- Output tokens are counted as they stream in, or as each whole response arrives.
- Every other statistic covers the requests that finish inside the window: prompt
  tokens and their cached share, requests/s, per-request decode tok/s (p10/p50/p90)
  and TTFT (p50/p90) when streaming, turn latency (p50/p90), errors by kind, mean
  requests in flight, and the load generator's event-loop lag (p90 and max).
- When the engine exports Prometheus `/metrics`, the window also records SGLang's own
  counters (as rates) and gauges (as means): generation and prompt tokens/s, running
  and queued requests, token usage, and cache hit rate.

**When the ramp stops.** It stops at the first point where:

- the load generator's event loop wakes more than 0.1 s late at p90 (`client_bound`):
  the point measured the client, not the engines, and the summary skips it; or
- the speed metric misses the loosest target (by default, p90 turn latency above 20 s);
  or
- two points in a row each add less than 3% output over the best earlier point (the
  engine is saturated, for instance by KV capacity; one such point can be noise at
  4-session steps); or
- more than 5% of requests fail.

**How the summary is computed.** `sweep.summarize` reads the CSV rows and uses each
variant's most recent run.

For each target, a variant's throughput is the most output per GPU that any load
sustains while meeting it (turn latency at or under the bound, or a rate at or over it).
It is the better of two values: the best measured point that meets the target, and a
linear interpolation, in the speed metric, between the two points that straddle it. Each value is labelled with one of three kinds:

- `crossed`: the ramp slowed past the target, so the value is bracketed.
- `lower_bound`: every point ran faster than the target, because the ramp saturated or
  ran out of sessions first. The value is the best point's, and it is a lower bound.
- `unreached`: even the lightest load ran slower than the target.

A configuration's figure is its best variant at each target. Two numbers are reported
against B200 BF16:

- relative throughput: throughput divided by B200 BF16's;
- relative cost per token: (price / throughput) divided by B200 BF16's (price /
  throughput).

The summary also reports the absolute cost in $ per million output tokens.

**Tuning variants (`configs.py`).** Each row starts from the serving configuration a
fleet already runs on that GPU and precision. That is a hetero training pool, the B200
BF16 pool, or the eval's B300 BF16 pool, each at the same engine size (TP). A variant
changes at most one knob:

- **KV cache dtype**, FP8 E4M3 vs BF16. The B200 BF16 baseline keeps a BF16 KV cache in
  every variant: it is the trainer-matched reference.
- **Full-attention backend**, among those valid for the architecture:
  - SM80: `flashinfer`, `triton`
  - SM90: `fa3`, `flashinfer`, `triton`
  - SM100: `trtllm_mha`, `triton`, `fa4` (SGLang v0.5.20 rejects FlashInfer full
    attention on Blackwell datacenter GPUs for hybrid GDN models; upstream re-allowed it
    on 2026-09-19 in sgl-project/sglang#36340). The B200/B300 rows sweep no attention
    variant: in the 2026-10-07 boot checks FA4 crashed at startup on B200
    ("SM100 forward with head_dim=256 does not support seqused_q/seqused_k"), and Triton
    served 3-4x less output per GPU than TRT-LLM MHA at 8 sessions per engine on every
    B200/B300 row.
  - SM120: `flashinfer`, `triton`
- **Decode batch ceiling** (`cg-N`): `--max-running-requests` and
  `--cuda-graph-max-bs-decode` move together, as the fleet sets them.
- **Speculative decoding.** Neither draft was trained against the RL policy, so any gain
  is a lower bound.
  - `spec-dflash` (B200 and B300 rows): z-lab's DFlash draft for the base model
    (`z-lab/Qwen3.6-35B-A3B-DFlash`, staged on the `sglang-cache` volume under
    `dflash/`) with the model card's Blackwell settings, block size 8 and an FA4 draft.
    In the 2026-10-07 boot checks it served, but accepted only 1.5-1.8 tokens per step at
    temperature 1.0, and served less than `base` at the same latency (B300 NVFP4,
    16 sessions per engine: 1,459 vs 1,850 tok/s per GPU). On the base model, which the
    draft matches, it still served less than `base` on every B200/B300 row at the 10 s
    bound (step-0 anchor, 2026-10-08; B200 BF16 at 20 sessions per engine: 688 vs 1,487
    tok/s per GPU). At 16-32 sessions per engine the engine is compute-bound, and drafting
    and verifying cost more than the accepted tokens save.
  - `spec-mtp`: the checkpoint's own MTP layer (every export carries it), as Miles runs
    Qwen3.5. It never served in our SGLang build: an illegal memory access on B200 at
    startup or under load, even with Miles' float32 state and Triton GDN kernels, and
    `NotImplementedError` in draft verification on FA3 (H100/H200) and on A100. The
    sweep leaves it out.

Every variant raises the fleet's batch ceiling, so the ramp can push an engine past the
speed targets. Two rows have no fleet behind them: B200 FP8 and RTX PRO 6000 FP8. They
use `hetero._pool` defaults.

**Engines.** Each engine runs SGLang alone on the container's public port. It uses its
pool's server arguments, plus the flags `server.serve_startup` adds:

- `--weight-update-staging`
- `--weight-version`

The Stitch sidecar is left out for two reasons: nothing publishes weights during the
benchmark, and the sidecar buffers each response, which would hide TTFT.

The driver addresses each replica through the pool URL with the `modal-flash-upstream`
header, so a session stays on its engine. `--gateway` lets Flash route requests instead.
Streams keep bytes flowing, so the Flash cutoff for silent responses (~350 s) only
applies while a request waits for its first token.

## 1. Record traces (eval smoke, 30–40 tasks, 1 sample each)

Run the eval's smoke path on the base BF16 checkpoint with the dump directory set. The
directory must be under `/stitch`, which is where the eval volume is mounted in the eval
driver. `eval_app` passes the variable to the driver image, and the agent's Ray workers
inherit it.

Launch from a clean worktree of a commit that contains the dump flag.

```bash
MODAL_FUNCTION_RUNTIME=runc MODAL_ENVIRONMENT=stitch-dev \
MODAL_SWE_TRAJECTORY_DUMP_DIR=/stitch/bench-traces/qwen36-swebench-pro-base-bf16-36x1 \
  uv run --extra modal python -m cookbook.miles_disagg.eval_launch \
    --spec swebench_pro_hetero --versions 0 --views bf16 --smoke 36x1 --engines 1
```

- **Hardware:** one B300 engine (`EvalB300BF16`, TP1) and the eval driver's H100 host.
- **Time:** about 1–1.5 h, bounded by the longest episode. An episode is limited to 500
  steps and 4,800 s, plus grading.
- **Cost:** about $11 for the B300 and $6 for the driver host, roughly $17 in all.
- **Choosing the smoke size:** `eval_launch` skips a point whose smoke metrics are
  already complete, so pick a size (here 36x1) that has never been run for the base
  BF16 point.
- **Output:** one JSON file per episode, named `<instance>.s<sample>.<id>.json`.
  - Each file holds the instance id, sample index, exit status and reward.
  - It holds the ordered messages. Each model-call message also carries its request's
    `prompt_tokens`, `completion_tokens` and `cached_tokens`.
  - It holds the duration of every model request.
  - An episode that aborted and was rerun leaves one file per attempt. A truncated
    attempt is still valid workload.

Check the count with:

```bash
modal volume ls stitch-swebench-pro-eval bench-traces/qwen36-swebench-pro-base-bf16-36x1
```

## 2. Validate on B200 BF16 alone

Preview the launch first. This touches nothing on Modal:

```bash
uv run --extra modal python -m cookbook.miles_disagg.bench.launch \
  --config b200-bf16 --variants base --engines 1 --plan
```

Then launch:

```bash
MODAL_FUNCTION_RUNTIME=runc MODAL_ENVIRONMENT=stitch-dev \
  uv run --extra modal python -m cookbook.miles_disagg.bench.launch \
    --config b200-bf16 --variants base --engines 1 \
    --traces bench-traces/qwen36-swebench-pro-base-bf16-36x1
```

The launcher does the following:

1. Refuses to run without runc.
2. Refuses to run if the app is already deployed.
3. Deploys the app `stitch-bench-b200-bf16`.
4. Spawns one driver per variant.
5. Prints each point as it is measured.
6. Stops the app when the drivers finish. Pass `--keep-app` to keep it.

Results go to the `stitch-sampler-bench` volume under `b200-bf16/<run_id>/`, as two
files per variant: `<variant>.csv` and `<variant>.manifest.json`.

**Cost of the validation:** one B200 for at most about 55 min (20 min boot plus 7 points
of 5 min). That is at most $5.73; early stops shorten it.

**What to check:**

- At 32 sessions, p50 decode should be near the fleet's calibration of about 94 tok/s
  per request at 32 sessions (see `configs/qwen3_6_35b_a3b_b200_bf16.py`).
- `server_cache_hit_rate` should be high, and `uncached_prompt_tok_s` should be small
  next to `prompt_tok_s`. Otherwise the replay's context is not matching the cache.
- `errors` should be 0.
- `in_flight_mean` should be close to `sessions`.
- `loop_lag_s_p90` should be well under 0.1 s (the ramp stops at `client_bound`
  otherwise). Otherwise the driver is too slow for its load.
- `completion_tokens_p50` should match the traces.

## 3. The full matrix

Launch each configuration as above, with all of its variants. Each variant gets its own
engine(s) and driver, so a configuration's variants run side by side. To run one
configuration twice at once, use `--tag`.

The upper bounds below assume 1 engine per variant and the whole ramp. They come from
`--plan`.

| Config | Variants | GPUs | Wall (h) | GPU-h | $ |
| --- | --- | --- | --- | --- | --- |
| B200 BF16 | 3 | 3 | 0.92 | 2.6 | 16 |
| B200 FP8 | 4 | 4 | 1.08 | 3.8 | 24 |
| B200 NVFP4 W4A16 | 4 | 4 | 1.08 | 3.8 | 24 |
| B300 BF16 | 4 | 4 | 1.08 | 3.8 | 27 |
| B300 NVFP4 W4A16 | 4 | 4 | 1.08 | 3.8 | 27 |
| H200 BF16 | 4 | 4 | 0.75 | 2.8 | 13 |
| H200 FP8 | 4 | 4 | 0.92 | 3.5 | 16 |
| H100 BF16 (TP2) | 4 | 8 | 0.83 | 6.2 | 24 |
| H100 FP8 | 4 | 4 | 0.75 | 2.8 | 11 |
| A100 80GB BF16 (TP2) | 4 | 8 | 0.75 | 5.7 | 14 |
| RTX PRO 6000 BF16 (TP2) | 4 | 8 | 0.75 | 5.7 | 17 |
| RTX PRO 6000 FP8 | 4 | 4 | 0.75 | 2.8 | 9 |

The whole matrix costs at most about 48 GPU-hours, or about $225. The driver CPU
containers add little.

Some engines run a configuration that has never been booted on this SGLang fork:

- the B200 FP8 and RTX PRO 6000 FP8 rows;
- every `attn-*` variant;
- `kv-*` variants that a fleet does not already run.

Boot-check those with a single variant before you launch their whole row.

## 4. Summarize

Download the results, then summarize them:

```bash
modal volume get stitch-sampler-bench b200-bf16 ./bench-results/
python -m cookbook.miles_disagg.bench.sweep summarize bench-results/**/*.csv \
  --out bench-results/summary.csv
```

The summary has one row per configuration. For each target `T`, it has these columns:

- `tok_s_per_gpu@T`
- `kind@T`
- `variant@T`
- `usd_per_mtok@T`
- `rel_throughput@T`
- `rel_cost_per_token@T`

## Local dry run (no GPU)

```bash
python -m cookbook.miles_disagg.bench.fake_server --port 8000 --rate 80 --capacity 16 \
  --synthetic-traces /tmp/bench-traces &
python -m cookbook.miles_disagg.bench.sweep run --url http://127.0.0.1:8000 \
  --traces /tmp/bench-traces --out /tmp/points.csv --config b200-bf16 \
  --sessions 4,8,16,32 --warmup 5 --window 10
```

The tests (`tests/cookbook/miles_disagg/bench/`) run the same loop against the
in-process fake.
