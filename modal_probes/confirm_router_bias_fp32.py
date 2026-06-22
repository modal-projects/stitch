"""Probe: confirm Megatron keeps the MoE router's expert_bias buffer in fp32, and
show WHY (so we know whether exporting it as bf16 for the rollout is correct).

The M1 disk-delta export crashed XORing a 256-byte tensor against a 128-byte base
for `e_score_correction_bias`. The base is uniformly bf16 (per inspect_moonlight_weights),
and `64 experts x 4 bytes (fp32) = 256` vs `64 x 2 (bf16) = 128`. moonlight.sh sets
`--moe-router-dtype fp32`, so the hypothesis is: Megatron keeps the router `expert_bias`
in fp32 and slime's deepseekv3 converter exported it without casting to the bf16
checkpoint dtype.

This is CPU-only source inspection of megatron-core's router -- it (1) confirms the
fp32 path without a GPU build, and (2) dumps `_maintain_float32_expert_bias` + the
expert_bias buffer registration + its update, so we can judge whether the fp32 is a
training-side accumulation concern (=> bf16 export to the rollout is the correct HF
representation) or something the inference path also depends on.

    m run -m modal_probes.confirm_router_bias_fp32::confirm
"""

from __future__ import annotations

import modal

SLIME_IMAGE_TAG = "slimerl/slime:nightly-dev-20260527a"

image = modal.Image.from_registry(SLIME_IMAGE_TAG).entrypoint([])
app = modal.App("confirm-router-bias-fp32")


@app.function(image=image, timeout=10 * 60)
def confirm() -> None:
    import importlib
    import inspect

    router_mod = importlib.import_module("megatron.core.transformer.moe.router")
    print(f"router source: {inspect.getsourcefile(router_mod)}\n")

    # 1) Dump _maintain_float32_expert_bias from whatever class defines it.
    printed = set()
    for cls_name in dir(router_mod):
        cls = getattr(router_mod, cls_name)
        if not isinstance(cls, type):
            continue
        method = getattr(cls, "_maintain_float32_expert_bias", None)
        if method is None or method in printed:
            continue
        printed.add(method)
        print(f"===== {cls_name}._maintain_float32_expert_bias =====")
        try:
            print(inspect.getsource(method))
        except (OSError, TypeError) as exc:
            print(f"(could not getsource: {exc})")

    # 2) Show every line in the module that touches expert_bias creation / update / dtype.
    print("===== router.py lines: expert_bias / register_buffer / bias update / dtype =====")
    for i, line in enumerate(inspect.getsource(router_mod).splitlines(), 1):
        s = line.strip()
        if ("expert_bias" in s or "register_buffer" in s) and ("#" != s[:1] if s else False):
            print(f"{i:5}: {s}")
