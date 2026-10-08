"""The benchmark matrix: one row per (GPU, weight precision) that can serve Qwen3.6-35B-A3B
rollouts, each with a few tuning variants.

A row starts from the serving configuration the heterogeneous fleet already runs on that
GPU and precision (``configs.qwen3_6_35b_a3b_hetero``), the B200 BF16 fleet
(``configs.qwen3_6_35b_a3b_b200_bf16``) or the SWE-bench Pro eval's B300 BF16 pool. Rows
no fleet runs (B200 FP8, RTX PRO 6000 FP8) are built with the same ``_pool`` defaults
and say so in ``source``. Engine size (GPUs per engine, i.e. TP) is the fleet's.

A variant changes at most one serving knob of its row's ``base``:

- the KV cache dtype (FP8 E4M3 vs BF16). The B200 BF16 baseline keeps a BF16 KV cache in
  every variant: it is the trainer-matched sampler the other rows are compared with;
- the full-attention backend, among those valid for the GPU architecture;
- the decode batch ceiling: ``--max-running-requests`` and ``--cuda-graph-max-bs-decode``
  together, as the fleet sets them, so a running batch never leaves CUDA graphs.

Every variant raises the batch ceiling above the fleet's (which is sized to its rollout
session target) so the sweep can load an engine past the per-request speed targets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from cookbook.common.config import RolloutPoolConfig
from cookbook.common.constants import SGLANG_CACHE_PATH
from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_b200_bf16 as b200_bf16
from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_hetero as hetero

# Modal list prices, $ per GPU-hour.
GPU_PRICES = {
    "B300": 7.10,
    "B200": 6.25,
    "H200": 4.54,
    "H100": 3.95,
    "RTX-PRO-6000": 3.03,
    "A100-80GB": 2.50,
}
GPU_LABELS = {
    "B300": "B300",
    "B200": "B200",
    "H200": "H200",
    "H100": "H100",
    "RTX-PRO-6000": "RTX PRO 6000",
    "A100-80GB": "A100 80GB",
}
ARCHITECTURES = {
    "A100-80GB": "sm80",
    "H100": "sm90",
    "H200": "sm90",
    "B200": "sm100",
    "B300": "sm100",
    "RTX-PRO-6000": "sm120",
}
# Full-attention backends SGLang offers each architecture. FA3 needs Hopper;
# TRT-LLM MHA is Blackwell datacenter only (on SM120 it rejects an FP8 KV cache). On
# Blackwell datacenter parts SGLang v0.5.20 allows only TRT-LLM MHA, Triton or FA4 for
# hybrid GDN models such as Qwen3.6 (an engine given FlashInfer fails its startup
# assertion). Boot checks (2026-10-07) found FA4 crashing at startup on B200 for this
# model's head_dim of 256, and Triton serving 3-4x less output per GPU than TRT-LLM MHA at
# 8 sessions per engine on every B200/B300 row, so those rows sweep no attention variant.
ATTENTION_BACKENDS = {
    "sm80": ("flashinfer", "triton"),
    "sm90": ("fa3", "flashinfer", "triton"),
    "sm100": ("trtllm_mha", "triton", "fa4"),
    "sm120": ("flashinfer", "triton"),
}
KV_CACHE_DTYPES = {"fp8_e4m3": "fp8", "bf16": "bf16", "bfloat16": "bf16"}
PRECISION_LABELS = {"bf16": "BF16", "fp8": "FP8", "nvfp4": "NVFP4 W4A16"}
# Architectures with FP8 tensor cores; NVFP4 needs Blackwell datacenter parts.
FP8_ARCHITECTURES = {"sm90", "sm100", "sm120"}
NVFP4_ARCHITECTURES = {"sm100"}

BASELINE = "b200-bf16"
MAX_VARIANTS = 5
# Speculative decoding with the checkpoint's own MTP head, as Miles runs Qwen3.5
# (scripts/run_qwen3_5_35b_a3b_mtp.py). Our RL runs train with --mtp-num-layers 0, so
# every checkpoint carries the base model's MTP head unchanged: the draft is untrained
# against the RL policy, and its acceptance (hence the gain) is a lower bound.
# Miles serves MTP with SGLang's defaults for the Gated DeltaNet layers: float32 state and
# triton kernels. Our pools' bfloat16 state failed at startup (H200 FP8: "initial_state
# must be float32"; B200: illegal memory access), and flashinfer's SM100 decode kernel
# requires bfloat16 state. Draft verification also needs the full-attention backend to
# support it: FA3 (H100/H200) raised NotImplementedError (boot checks, 2026-10-07).
SPECULATIVE_MTP = {
    "--speculative-algorithm": "EAGLE",
    "--speculative-num-steps": "2",
    "--speculative-eagle-topk": "1",
    "--speculative-num-draft-tokens": "3",
    "--mamba-ssm-dtype": "float32",
    "--linear-attn-prefill-backend": "triton",
    "--linear-attn-decode-backend": "triton",
}
# DFlash block-diffusion drafting with z-lab's draft for the base model, on the model
# card's Blackwell settings (block 8 for high concurrency, an fa4 draft). Like MTP, the
# draft never saw the RL policy, so its acceptance (hence the gain) is a lower bound. The
# draft is staged on the engines' cache volume: sglang-cache:/dflash/<name>.
DFLASH_DRAFT = "Qwen3.6-35B-A3B-DFlash"
SPECULATIVE_DFLASH = {
    "--speculative-algorithm": "DFLASH",
    "--speculative-draft-model-path": f"{SGLANG_CACHE_PATH}/dflash/{DFLASH_DRAFT}",
    "--speculative-dflash-block-size": "8",
    "--speculative-draft-attention-backend": "fa4",
}
# Chat template options for client-side rendering, as Miles' Qwen3.6 TITO tokenizer sets
# them: earlier turns keep their reasoning, so appending a turn never re-renders one.
CHAT_TEMPLATE_KWARGS = {"preserve_thinking": True}
# The recipe whose harness recorded the traces and whose checkpoint views the engines
# serve; base points (version 0) are shared by every hetero arm.
DEFAULT_EXPERIMENT = "qwen3_6_35b_a3b_hetero_grpo"


def gpu_family(gpu: str) -> str:
    """The priced GPU class of a Modal GPU request (``H100!`` pins H100 without the
    automatic H200 upgrade; it is still an H100)."""
    return gpu.rstrip("!")


@dataclass(frozen=True)
class Variant:
    """One serving configuration of a row: the row's pool with at most one knob changed,
    and the decode batch ceiling the sweep may load each engine to."""

    name: str
    max_batch: int
    kv_cache_dtype: str | None = None
    attention_backend: str | None = None
    speculative: bool = False
    dflash: bool = False
    note: str = ""

    def sglang_overrides(self) -> dict[str, str]:
        overrides = {
            "--max-running-requests": str(self.max_batch),
            "--cuda-graph-max-bs-decode": str(self.max_batch),
        }
        if self.kv_cache_dtype is not None:
            overrides["--kv-cache-dtype"] = self.kv_cache_dtype
        if self.attention_backend is not None:
            overrides["--attention-backend"] = self.attention_backend
        if self.speculative:
            overrides.update(SPECULATIVE_MTP)
        if self.dflash:
            overrides.update(SPECULATIVE_DFLASH)
        return overrides


@dataclass(frozen=True)
class BenchConfig:
    """A row of the matrix. ``precision`` is the served weight view: ``bf16``, ``fp8``
    (block-scaled FP8) or ``nvfp4`` (NVFP4 weights, BF16 activations)."""

    key: str
    gpu: str
    precision: str
    pool: RolloutPoolConfig
    variants: tuple[Variant, ...]
    source: str
    baseline: bool = False

    @property
    def family(self) -> str:
        return gpu_family(self.gpu)

    @property
    def arch(self) -> str:
        return ARCHITECTURES[self.family]

    @property
    def label(self) -> str:
        return f"{GPU_LABELS[self.family]} {PRECISION_LABELS[self.precision]}"

    @property
    def price_per_gpu_hour(self) -> float:
        return GPU_PRICES[self.family]

    @property
    def gpus_per_engine(self) -> int:
        return self.pool.gpus_per_engine

    @property
    def weight_view(self) -> str:
        return str(self.pool.weight_view)

    def variant(self, name: str) -> Variant:
        for variant in self.variants:
            if variant.name == name:
                return variant
        known = ", ".join(v.name for v in self.variants)
        raise KeyError(f"{self.key} has no variant {name!r}; it has {known}")

    def server_args(self, variant: Variant) -> dict[str, str]:
        return {**self.pool.sglang_args, **variant.sglang_overrides()}

    def bench_pool(self, variant: Variant, *, engines: int) -> RolloutPoolConfig:
        """The variant as a fixed fleet of ``engines`` engines. Flash's concurrency
        target is the batch ceiling, which no sweep point exceeds."""
        if not 1 <= engines <= 8:
            raise ValueError(f"engines must be 1..8, got {engines}")
        return replace(
            self.pool,
            name=server_name(self, variant),
            sglang_args=self.server_args(variant),
            target_inputs=variant.max_batch,
            min_containers=engines,
            max_containers=engines,
        )


def server_name(config: BenchConfig, variant: Variant) -> str:
    """A variant's Modal Server class: ``BenchB200Bf16Base``."""
    words = re.split(r"[^0-9A-Za-z]+", f"{config.key}-{variant.name}")
    return "Bench" + "".join(word[:1].upper() + word[1:] for word in words if word)


def app_name(config: BenchConfig, tag: str = "") -> str:
    name = f"stitch-bench-{config.key}" + (f"-{tag}" if tag else "")
    if len(name) > 64 or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ValueError(f"invalid bench app name {name!r}")
    return name


def _variants(
    max_batch: int,
    *,
    kv_cache_dtype: str | None = None,
    attention_backend: str | None = None,
    batch: int | None = None,
    speculative: bool = False,
    dflash: bool = False,
) -> tuple[Variant, ...]:
    """``base`` plus one variant per alternative given, in the order KV cache,
    attention backend, batch ceiling, speculative decoding (MTP, then DFlash)."""
    variants = [Variant("base", max_batch)]
    if kv_cache_dtype is not None:
        variants.append(
            Variant(
                f"kv-{KV_CACHE_DTYPES[kv_cache_dtype]}",
                max_batch,
                kv_cache_dtype=kv_cache_dtype,
            )
        )
    if attention_backend is not None:
        variants.append(
            Variant(
                f"attn-{attention_backend.replace('_', '-')}",
                max_batch,
                attention_backend=attention_backend,
            )
        )
    if batch is not None:
        variants.append(Variant(f"cg-{batch}", batch))
    if speculative:
        variants.append(Variant("spec-mtp", max_batch, speculative=True))
    if dflash:
        variants.append(Variant("spec-dflash", max_batch, dflash=True))
    return tuple(variants)


def _training_pool(name: str) -> RolloutPoolConfig:
    return next(pool for pool in hetero.modal.rollout_pools if pool.name == name)


def _row(
    key: str,
    pool: RolloutPoolConfig,
    variants: tuple[Variant, ...],
    source: str,
    *,
    baseline: bool = False,
) -> BenchConfig:
    gpu = pool.gpu if isinstance(pool.gpu, str) else pool.gpu[0]
    return BenchConfig(
        key=key,
        gpu=gpu,
        precision=str(pool.weight_view),
        pool=pool,
        variants=variants,
        source=source,
        baseline=baseline,
    )


def _eval_b300_bf16_pool() -> RolloutPoolConfig:
    # The SWE-bench Pro eval's BF16 pool (eval_configs.swebench_pro_hetero.POOLS["bf16"]),
    # built the same way here so the matrix does not import the eval spec.
    return hetero._pool(
        name="EvalB300BF16",
        gpu="B300",
        weight_view="bf16",
        attention_backend="trtllm_mha",
        kv_cache_dtype="bfloat16",
        moe_runner_backend="triton",
    )


def _matrix() -> dict[str, BenchConfig]:
    rows = (
        _row(
            "b200-bf16",
            b200_bf16.modal.rollout_pools[0],
            _variants(128, batch=64, speculative=True, dflash=True),
            "qwen3_6_35b_a3b_b200_bf16 ServerB200BF16",
            baseline=True,
        ),
        # The baseline's weights with the FP8 KV cache the mixed pool uses: the same GPU,
        # with only the cache precision that correction allows changed.
        _row(
            "b200-bf16-kv-fp8",
            replace(
                b200_bf16.modal.rollout_pools[0],
                name="ServerB200BF16KVFP8",
                sglang_args={
                    **b200_bf16.modal.rollout_pools[0].sglang_args,
                    "--kv-cache-dtype": "fp8_e4m3",
                },
            ),
            _variants(128, batch=64),
            "qwen3_6_35b_a3b_b200_bf16 ServerB200BF16 with an FP8 KV cache",
        ),
        _row(
            "b200-fp8",
            hetero._pool(
                name="ServerB200FP8",
                gpu="B200",
                weight_view="fp8",
                attention_backend="trtllm_mha",
            ),
            _variants(
                128,
                kv_cache_dtype="bf16",
                batch=256,
            ),
            "new: hetero._pool defaults (no fleet serves FP8 on B200)",
        ),
        _row(
            "b200-nvfp4",
            _training_pool("ServerB200NVFP4W4A16"),
            _variants(
                128,
                kv_cache_dtype="bf16",
                batch=256,
                speculative=True,
                dflash=True,
            ),
            "qwen3_6_35b_a3b_hetero ServerB200NVFP4W4A16",
        ),
        _row(
            "b300-bf16",
            _eval_b300_bf16_pool(),
            _variants(
                128,
                kv_cache_dtype="fp8_e4m3",
                batch=256,
            ),
            "swebench_pro_hetero eval pool EvalB300BF16",
        ),
        _row(
            "b300-nvfp4",
            _training_pool("ServerB300NVFP4W4A16"),
            _variants(
                128,
                kv_cache_dtype="bf16",
                batch=256,
                speculative=True,
                dflash=True,
            ),
            "qwen3_6_35b_a3b_hetero ServerB300NVFP4W4A16",
        ),
        _row(
            "h200-bf16",
            _training_pool("ServerH200BF16"),
            _variants(
                64,
                kv_cache_dtype="bf16",
                attention_backend="flashinfer",
                batch=32,
                speculative=True,
            ),
            "qwen3_6_35b_a3b_hetero ServerH200BF16",
        ),
        _row(
            "h200-fp8",
            _training_pool("ServerH200FP8"),
            _variants(
                128,
                kv_cache_dtype="bf16",
                attention_backend="flashinfer",
                batch=64,
                speculative=True,
            ),
            "qwen3_6_35b_a3b_hetero ServerH200FP8",
        ),
        _row(
            "h100-bf16",
            _training_pool("ServerH100BF16TP2"),
            _variants(
                64,
                kv_cache_dtype="bf16",
                attention_backend="flashinfer",
                batch=96,
                speculative=True,
            ),
            "qwen3_6_35b_a3b_hetero ServerH100BF16TP2",
        ),
        _row(
            "h100-fp8",
            _training_pool("ServerH100FP8"),
            _variants(
                64,
                kv_cache_dtype="bf16",
                attention_backend="flashinfer",
                batch=32,
                speculative=True,
            ),
            "qwen3_6_35b_a3b_hetero ServerH100FP8",
        ),
        _row(
            "a100-bf16",
            _training_pool("ServerA100BF16TP2"),
            # No Triton attention: the fleet stores its KV cache in FP8, which only
            # FlashInfer's kernel unpacks on A100 (Triton: "type fp8e4nv not supported
            # in this architecture", 2026-10-07).
            _variants(
                64,
                kv_cache_dtype="bf16",
                batch=32,
                speculative=True,
            ),
            "qwen3_6_35b_a3b_hetero ServerA100BF16TP2",
        ),
        _row(
            "rtx-pro-6000-bf16",
            _training_pool("ServerRTXPRO6000BF16TP2"),
            _variants(
                64,
                kv_cache_dtype="bf16",
                attention_backend="triton",
                batch=32,
                speculative=True,
            ),
            "qwen3_6_35b_a3b_hetero ServerRTXPRO6000BF16TP2",
        ),
        _row(
            "rtx-pro-6000-fp8",
            hetero._pool(
                name="ServerRTXPRO6000FP8",
                gpu="RTX-PRO-6000",
                weight_view="fp8",
                # As the BF16 pool: TRT-LLM MHA on SM120 rejects an FP8 KV cache.
                attention_backend="flashinfer",
            ),
            _variants(64, kv_cache_dtype="bf16", attention_backend="triton", batch=32),
            "new: hetero._pool defaults (no fleet serves FP8 on RTX PRO 6000)",
        ),
    )
    return {row.key: row for row in rows}


CONFIGS: dict[str, BenchConfig] = _matrix()


def config(key: str) -> BenchConfig:
    try:
        return CONFIGS[key]
    except KeyError:
        raise KeyError(
            f"no bench config {key!r}; choose one of {', '.join(CONFIGS)}"
        ) from None


def select_variants(bench: BenchConfig, names: str = "") -> tuple[Variant, ...]:
    """The comma-separated ``names`` of ``bench``'s variants, or all of them."""
    if not names.strip():
        return bench.variants
    chosen = tuple(bench.variant(name.strip()) for name in names.split(","))
    if len({variant.name for variant in chosen}) != len(chosen):
        raise ValueError(f"duplicate variants in {names!r}")
    return chosen


def validate(bench: BenchConfig) -> None:
    """Reject a row whose variants an engine of its GPU could not serve as written."""
    if bench.family not in GPU_PRICES:
        raise ValueError(f"{bench.key}: no list price for {bench.family}")
    if bench.precision not in PRECISION_LABELS:
        raise ValueError(f"{bench.key}: unknown precision {bench.precision!r}")
    if bench.precision == "fp8" and bench.arch not in FP8_ARCHITECTURES:
        raise ValueError(f"{bench.key}: {bench.family} has no FP8 tensor cores")
    if bench.precision == "nvfp4":
        if bench.arch not in NVFP4_ARCHITECTURES:
            raise ValueError(f"{bench.key}: NVFP4 needs Blackwell datacenter GPUs")
        if bench.pool.sglang_args.get("--quantization") != "modelopt_fp4":
            raise ValueError(f"{bench.key}: NVFP4 serves with modelopt_fp4")
    elif "--quantization" in bench.pool.sglang_args:
        raise ValueError(f"{bench.key}: only NVFP4 rows set --quantization")
    if not 1 <= len(bench.variants) <= MAX_VARIANTS:
        raise ValueError(
            f"{bench.key}: 1..{MAX_VARIANTS} variants, not {len(bench.variants)}"
        )
    if bench.variants[0].name != "base":
        raise ValueError(f"{bench.key}: the first variant must be 'base'")
    if len({variant.name for variant in bench.variants}) != len(bench.variants):
        raise ValueError(f"{bench.key}: duplicate variant names")
    for variant in bench.variants:
        args = bench.server_args(variant)
        where = f"{bench.key}/{variant.name}"
        if variant.max_batch < 1:
            raise ValueError(f"{where}: max_batch must be positive")
        if args["--attention-backend"] not in ATTENTION_BACKENDS[bench.arch]:
            raise ValueError(
                f"{where}: {args['--attention-backend']} is not offered on {bench.arch}"
            )
        if args["--kv-cache-dtype"] not in KV_CACHE_DTYPES:
            raise ValueError(
                f"{where}: unknown KV cache dtype {args['--kv-cache-dtype']}"
            )
        if bench.baseline and KV_CACHE_DTYPES[args["--kv-cache-dtype"]] != "bf16":
            raise ValueError(f"{where}: the baseline keeps a BF16 KV cache")
        if args["--tp"] != str(bench.gpus_per_engine):
            raise ValueError(f"{where}: --tp disagrees with the engine's GPU count")
        # FlashInfer's linear attention needs SM90+.
        if bench.arch == "sm80" and args["--linear-attn-decode-backend"] != "triton":
            raise ValueError(f"{where}: Ampere runs linear attention on Triton")


def validate_matrix(matrix: dict[str, BenchConfig] | None = None) -> None:
    matrix = CONFIGS if matrix is None else matrix
    baselines = [key for key, row in matrix.items() if row.baseline]
    if baselines != [BASELINE]:
        raise ValueError(f"exactly {BASELINE} must be the baseline, got {baselines}")
    for row in matrix.values():
        validate(row)


validate_matrix()
