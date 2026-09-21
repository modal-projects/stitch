# Tools

These tools validate Stitch's weight-update path and inspect an existing rollout
fleet. They are measurement utilities, not deployment recipes.

## Weight-update validation

Each profile prepares a pinned checkpoint and deterministic XOR-delta lineage,
then validates ordinary, folded, repeated, missing-target, and clean-reload
behavior. The benchmark checks live rank weights against an independent native
load while measuring preparation, engine pause, host resources, and generation
during staging.

| Checkpoint | Profile |
| --- | --- |
| GLM-4.5-Air FP8 | [`glm45_air_fp8.py`](weight_update/profiles/glm45_air_fp8.py) |
| GLM-5.2 FP8 | [`glm5_2_fp8.py`](weight_update/profiles/glm5_2_fp8.py) |
| GLM-5.2 mixed NVFP4/BF16 | [`glm5_2_nvfp4.py`](weight_update/profiles/glm5_2_nvfp4.py) |
| GLM-5.3-Flash native FP8 | [`glm5_3_flash_fp8.py`](weight_update/profiles/glm5_3_flash_fp8.py) |
| GLM-5.3 FP8 | [`glm5_3_fp8.py`](weight_update/profiles/glm5_3_fp8.py) |
| GLM-5.3 mixed NVFP4/BF16 | [`glm5_3_nvfp4.py`](weight_update/profiles/glm5_3_nvfp4.py) |
| Kimi K2.6 NVFP4 | [`kimi_k2_6_nvfp4.py`](weight_update/profiles/kimi_k2_6_nvfp4.py) |
| Kimi K3 MXFP4 | [`kimi_k3_mxfp4.py`](weight_update/profiles/kimi_k3_mxfp4.py) |

Select the Modal environment explicitly:

```bash
MODAL_ENVIRONMENT=your-environment
uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" -d \
  tools/weight_update/profiles/kimi_k3_mxfp4.py \
  --update-mode cpu --canonical-storage disk
```

Profiles expose checkpoint preparation and mode-specific options through
`--help`. CPU staging requires `--canonical-storage memory|disk`; disk staging
does not accept that option. Synthetic deltas validate SGLang application, not
trainer export or production throughput. Compare results only for the same
checkpoint, runtime, topology, delta shape, storage mode, and host class.

## Fleet diagnostics

[`fleet/app.py`](fleet/app.py) drives synthetic traffic and polls convergence
against an already deployed pool in the same Modal environment. Results are
stored by tag on the `stitch-probe-results` Volume.

```bash
POOL_APP=your-pool-app
uv run --extra modal modal deploy -e "$MODAL_ENVIRONMENT" -m tools.fleet.app

uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m tools.fleet.app::poll --pool-app "$POOL_APP" --tag demo

uv run --extra modal modal run -e "$MODAL_ENVIRONMENT" \
  -m tools.fleet.app::traffic --pool-app "$POOL_APP" \
  --shape agentic --concurrency 32 --duration 1800 --tag demo
```

The traffic probe records end-to-end latency rather than time to first token;
its synthetic workload and sampled version polling are diagnostic signals, not
CI thresholds.
