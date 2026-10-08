"""The sampler benchmark's configuration matrix."""

import dataclasses

import pytest

from cookbook.miles_disagg.bench import configs
from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_b200_bf16 as b200_bf16
from cookbook.miles_disagg.configs import qwen3_6_35b_a3b_hetero as hetero

BATCH_FLAGS = {"--max-running-requests", "--cuda-graph-max-bs-decode"}


def test_matrix_has_exactly_the_requested_gpu_and_precision_rows():
    rows = {(row.family, row.precision) for row in configs.CONFIGS.values()}

    assert rows == {
        ("B200", "bf16"),
        ("B200", "fp8"),
        ("B200", "nvfp4"),
        ("B300", "bf16"),
        ("B300", "nvfp4"),
        ("H200", "bf16"),
        ("H200", "fp8"),
        ("H100", "bf16"),
        ("H100", "fp8"),
        ("A100-80GB", "bf16"),
        ("RTX-PRO-6000", "bf16"),
        ("RTX-PRO-6000", "fp8"),
    }
    # B200 BF16 weights appear twice: the matched baseline (BF16 KV cache) and the same
    # weights with an FP8 KV cache.
    assert len(configs.CONFIGS) == len(rows) + 1
    kv = {
        key: configs.CONFIGS[key].pool.sglang_args["--kv-cache-dtype"]
        for key in ("b200-bf16", "b200-bf16-kv-fp8")
    }
    assert kv == {"b200-bf16": "bf16", "b200-bf16-kv-fp8": "fp8_e4m3"}


def test_every_row_is_valid_and_priced_at_list():
    configs.validate_matrix()
    prices = {row.family: row.price_per_gpu_hour for row in configs.CONFIGS.values()}

    assert prices == {
        "B300": 7.10,
        "B200": 6.25,
        "H200": 4.54,
        "H100": 3.95,
        "RTX-PRO-6000": 3.03,
        "A100-80GB": 2.50,
    }


def test_baseline_is_b200_bf16_with_a_bf16_kv_cache_in_every_variant():
    baseline = configs.CONFIGS[configs.BASELINE]

    assert [row.key for row in configs.CONFIGS.values() if row.baseline] == [
        "b200-bf16"
    ]
    assert (baseline.family, baseline.precision) == ("B200", "bf16")
    for variant in baseline.variants:
        assert baseline.server_args(variant)["--kv-cache-dtype"] == "bf16"


def _training_pool(name):
    pools = (*hetero.modal.rollout_pools, *b200_bf16.modal.rollout_pools)
    return next(pool for pool in pools if pool.name == name)


@pytest.mark.parametrize(
    ("key", "pool_name"),
    [
        ("b200-bf16", "ServerB200BF16"),
        ("b200-nvfp4", "ServerB200NVFP4W4A16"),
        ("b300-nvfp4", "ServerB300NVFP4W4A16"),
        ("h200-bf16", "ServerH200BF16"),
        ("h200-fp8", "ServerH200FP8"),
        ("h100-bf16", "ServerH100BF16TP2"),
        ("h100-fp8", "ServerH100FP8"),
        ("a100-bf16", "ServerA100BF16TP2"),
        ("rtx-pro-6000-bf16", "ServerRTXPRO6000BF16TP2"),
    ],
)
def test_rows_a_fleet_serves_start_from_that_pool(key, pool_name):
    row = configs.CONFIGS[key]
    pool = _training_pool(pool_name)

    assert row.pool == pool
    assert row.gpus_per_engine == pool.gpus_per_engine
    assert row.pool.environment == pool.environment
    # base differs from the fleet only in its batch ceiling.
    base = row.server_args(row.variants[0])
    assert {k: v for k, v in base.items() if k not in BATCH_FLAGS} == {
        k: v for k, v in pool.sglang_args.items() if k not in BATCH_FLAGS
    }


def test_engine_sizes_follow_the_fleet():
    tp = {key: row.gpus_per_engine for key, row in configs.CONFIGS.items()}

    assert {key for key, size in tp.items() if size == 2} == {
        "a100-bf16",
        "rtx-pro-6000-bf16",
        "h100-bf16",
    }
    assert set(tp.values()) == {1, 2}


def test_b300_bf16_matches_the_eval_pool():
    from cookbook.miles_disagg.eval_configs import swebench_pro_hetero as spec

    assert configs.CONFIGS["b300-bf16"].pool == spec.POOLS["bf16"]


def test_nvfp4_rows_serve_w4a16_through_modelopt():
    for row in configs.CONFIGS.values():
        args = row.pool.sglang_args
        if row.precision == "nvfp4":
            assert args["--quantization"] == "modelopt_fp4"
            assert row.pool.environment == {
                "SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16": "1"
            }
            assert row.arch == "sm100"
        else:
            assert "--quantization" not in args


def test_each_variant_changes_at_most_one_knob_of_base():
    for row in configs.CONFIGS.values():
        assert 1 <= len(row.variants) <= configs.MAX_VARIANTS
        base = row.server_args(row.variants[0])
        names = [variant.name for variant in row.variants]
        assert names[0] == "base" and len(set(names)) == len(names)
        for variant in row.variants[1:]:
            args = row.server_args(variant)
            changed = {k for k in args if args[k] != base.get(k)}
            # The MTP set may match some of a row's base flags (its GDN kernels).
            speculative = "--speculative-algorithm" in changed and (
                changed <= set(configs.SPECULATIVE_MTP)
                or changed == set(configs.SPECULATIVE_DFLASH)
            )
            assert speculative or changed in (
                {"--kv-cache-dtype"},
                {"--attention-backend"},
                BATCH_FLAGS,
            ), (row.key, variant.name, changed)
            # The KV dtype variants flip between FP8 and BF16.
            if changed == {"--kv-cache-dtype"}:
                assert {
                    configs.KV_CACHE_DTYPES[base["--kv-cache-dtype"]],
                    configs.KV_CACHE_DTYPES[args["--kv-cache-dtype"]],
                } == {"fp8", "bf16"}


def test_variants_tie_running_requests_to_the_cuda_graph_batch():
    for row in configs.CONFIGS.values():
        for variant in row.variants:
            args = row.server_args(variant)
            assert args["--max-running-requests"] == str(variant.max_batch)
            assert args["--cuda-graph-max-bs-decode"] == str(variant.max_batch)
            # Above the fleet's ceiling, so the ramp can pass the speed targets.
            fleet = int(row.pool.sglang_args["--max-running-requests"])
            assert variant.max_batch >= 32 and (
                variant.name.startswith("cg-") or variant.max_batch >= fleet
            )


@pytest.mark.parametrize(
    ("key", "change", "message"),
    [
        ("b200-fp8", {"attention_backend": "fa3"}, "not offered on sm100"),
        ("h200-fp8", {"attention_backend": "trtllm_mha"}, "not offered on sm90"),
        ("a100-bf16", {"attention_backend": "fa3"}, "not offered on sm80"),
        ("b200-bf16", {"kv_cache_dtype": "fp8_e4m3"}, "baseline keeps a BF16 KV cache"),
        ("h100-fp8", {"kv_cache_dtype": "fp8_e5m2"}, "unknown KV cache dtype"),
    ],
)
def test_invalid_variants_are_rejected(key, change, message):
    row = configs.CONFIGS[key]
    bad = dataclasses.replace(row.variants[0], name="bad", **change)
    broken = dataclasses.replace(row, variants=(row.variants[0], bad))

    with pytest.raises(ValueError, match=message):
        configs.validate(broken)


def test_fp8_is_rejected_where_there_are_no_fp8_tensor_cores():
    row = configs.CONFIGS["h100-fp8"]
    a100 = dataclasses.replace(row, gpu="A100-80GB")

    with pytest.raises(ValueError, match="no FP8 tensor cores"):
        configs.validate(a100)


def test_bench_pool_is_a_fixed_fleet_named_per_variant():
    row = configs.CONFIGS["h100-bf16"]
    names = set()
    for variant in row.variants:
        pool = row.bench_pool(variant, engines=3)
        assert (pool.min_containers, pool.max_containers) == (3, 3)
        assert pool.target_inputs == variant.max_batch
        assert pool.gpu_request() == "H100!:2"
        assert pool.name.isidentifier() and pool.name.startswith("BenchH100Bf16")
        names.add(pool.name)
    assert len(names) == len(row.variants)
    with pytest.raises(ValueError):
        row.bench_pool(row.variants[0], engines=0)


def test_app_names_fit_modal():
    for row in configs.CONFIGS.values():
        assert len(configs.app_name(row, "tag-1")) <= 64
    with pytest.raises(ValueError):
        configs.app_name(configs.CONFIGS["b200-bf16"], "Bad_Tag")


def test_variant_selection():
    row = configs.CONFIGS["b200-nvfp4"]

    assert configs.select_variants(row) == row.variants
    assert [v.name for v in configs.select_variants(row, "kv-bf16, base")] == [
        "kv-bf16",
        "base",
    ]
    with pytest.raises(KeyError, match="no variant"):
        configs.select_variants(row, "nope")
    with pytest.raises(ValueError, match="duplicate"):
        configs.select_variants(row, "base,base")
    with pytest.raises(KeyError, match="no bench config"):
        configs.config("tpu-v5")


def test_the_rows_the_fleets_ran_also_measure_mtp_speculative_decoding():
    ran = {
        "b200-bf16",
        "b200-nvfp4",
        "b300-nvfp4",
        "h200-bf16",
        "h200-fp8",
        "h100-bf16",
        "h100-fp8",
        "a100-bf16",
        "rtx-pro-6000-bf16",
    }
    with_spec = {
        key
        for key, row in configs.CONFIGS.items()
        if any(variant.speculative for variant in row.variants)
    }

    assert with_spec == ran
    args = configs.CONFIGS["b200-bf16"].server_args(
        configs.CONFIGS["b200-bf16"].variant("spec-mtp")
    )
    assert args["--speculative-algorithm"] == "EAGLE"
    assert args["--max-running-requests"] == "128"


def test_dflash_drafts_only_on_blackwell_rows_from_the_engines_cache_volume():
    rows = {
        key
        for key, row in configs.CONFIGS.items()
        if any(variant.dflash for variant in row.variants)
    }

    assert rows == {"b200-bf16", "b200-nvfp4", "b300-nvfp4"}
    args = configs.CONFIGS["b300-nvfp4"].variant("spec-dflash").sglang_overrides()
    assert args["--speculative-algorithm"] == "DFLASH"
    assert args["--speculative-draft-model-path"].startswith(
        "/root/.cache/sglang/dflash/"
    )
