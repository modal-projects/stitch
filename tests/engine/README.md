# Engine tests

A delta set is a model in one precision: its rollout checkpoint and ten versions written as a trainer
publishes them (a simulated trainer, Miles' quantizers and Miles' weight-delta encoder). An engine
config serves a delta set on some hardware; every engine config of that model and precision shares it.

The mode is chosen when a delta set is generated:
- **perf**: the files a production trainer writes. The engine applies each version as Stitch syncs
  it: prepare while serving, then pause in place, update, continue.
- **correctness**: each changed tensor also carries its checksum, which SGLang checks as it applies,
  and at the end the engine must hash byte for byte like a fresh engine on the expected checkpoint.

```
modal run tests/engine/generate.py --delta-set glm5.3-nvfp4 --mode perf
modal run tests/engine/engine_test.py --engine glm5.3-nvfp4 --run <run>
```

`run_app.py` runs a Stitch run on Modal against a correctness set: engines catch up, take versions
under traffic, lose an engine and the control service, go idle, and refuse a version that fails their
checksums.

## Full models

| Model | Precision | Delta size | Changed | Prepare | Apply | Prepare, checksums | Apply, checksums | Pinned host memory |
|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | NVFP4 | 81 MB | 0.17% | 2.5 s | 38 ms | 2.6 s | 31 ms | 192 MiB |
| DeepSeek-V4.1-Flash | NVFP4 | 240 MB | 0.055% | 3.6 s | 72 ms | 2.5 s | 94 ms | 960 MiB |
| GLM-5.3-Flash | NVFP4 | 198 MB | 0.053% | 3.3 s | 63 ms | 3.9 s | 82 ms | 576 MiB |
| GLM-5.3 | NVFP4 | 536 MB | 0.072% | 11.4 s | 107 ms | 16.1 s | 128 ms | 1.8 GiB |

B200, mean of versions 2 to 10 under traffic. Pinned host memory is per GPU and held for the engine's life:
the staging area grows to 1.25 times the largest version a GPU has staged, in 64 MiB steps, and never shrinks.
