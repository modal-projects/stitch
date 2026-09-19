# Delta-update profiling

These scripts measure model-specific SGLang delta updates independently of the
[curated cookbook recipes](../../cookbook/README.md). Keep architecture and
checkpoint-format coverage here even when there is no maintained training recipe.
A profiler is a measurement tool, not a claim that its workload, training settings,
or deployment sizing is a tuned reference configuration.

| Model / checkpoint | Entrypoint |
| --- | --- |
| GLM-4.5-Air FP8 | [glm45_air_fp8_delta_weight_update.py](glm45_air_fp8_delta_weight_update.py) |
| GLM-5.2 FP8 | [glm5_2_fp8_delta_weight_update.py](glm5_2_fp8_delta_weight_update.py) |
| GLM-5.2 mixed NVFP4/BF16 | [glm5_2_nvfp4_delta_weight_update.py](glm5_2_nvfp4_delta_weight_update.py) |
| GLM-5.3 FP8 | [glm5_3_fp8_delta_weight_update.py](glm5_3_fp8_delta_weight_update.py) |
| GLM-5.3 mixed NVFP4/BF16 | [glm5_3_nvfp4_delta_weight_update.py](glm5_3_nvfp4_delta_weight_update.py) |
| Kimi K2.6 NVFP4 | [kimi_k2_6_nvfp4_delta_weight_update.py](kimi_k2_6_nvfp4_delta_weight_update.py) |
| Kimi K3 MXFP4 | [kimi_k3_mxfp4_delta_weight_update.py](kimi_k3_mxfp4_delta_weight_update.py) |

Confirm authentication with `modal profile current`, then pass the target
environment explicitly. For example:

```bash
uv run --extra modal modal run -e <env> -d \
  tools/profiling/kimi_k3_mxfp4_delta_weight_update.py \
  --update-mode cpu --canonical-storage disk
```

Entrypoints document checkpoint preparation and expose their options through
`--help`. GLM-5.3 profilers use the prepared cookbook checkpoints; the other
entrypoints own their preparation or download settings. GLM-5.2 NVFP4 expects
its DFlash artifact on the existing `dflash-checkpoints` Volume in the selected
Modal environment, mounted read-only at `/draft`. Profiler-only settings
belong with these scripts rather than in the training recipe catalog.

The shared runner checks one replica through initialization, target staging,
commit, and generation. Synthetic deltas cover FP8, NVFP4, and MXFP4; they do not
establish trainer-export correctness, fleet convergence, or production workload
performance. Compare timings only with the same checkpoint, runtime, update mode,
canonical-storage choice, and delta shape. CPU mode with memory-backed canonical
storage requires enough RAM for both the canonical checkpoint and rank images.

Fingerprints require repeatable tokens and text, plus finite, repeatable logprobs
where supported. DSpark rejects logprob requests, so its fingerprint is limited
to tokens and text; that result does not establish logprob correctness.
