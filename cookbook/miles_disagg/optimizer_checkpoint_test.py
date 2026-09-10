"""Optimizer checkpoint regression tests for the patched trainer image and CUDA."""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module", autouse=True)
def process_group(tmp_path_factory):
    import torch.distributed as dist

    already_initialized = dist.is_initialized()
    if not already_initialized:
        path = tmp_path_factory.mktemp("optimizer-dist") / "rendezvous"
        dist.init_process_group(
            "gloo", init_method=f"file://{path}", rank=0, world_size=1
        )
    yield
    if not already_initialized:
        dist.destroy_process_group()


def _make_wrapper(gap, *, initialize):
    """Use the real checkpoint methods with a small single-rank buffer layout."""
    import torch.distributed as dist
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
    from megatron.core.optimizer.optimizer import param_group_identifier_keys
    from transformer_engine.pytorch.optimizers import FusedAdam

    device = "cuda"
    params = [
        torch.nn.Parameter(
            torch.arange(4, device=device, dtype=torch.float32)
            .add_(i + 1)
            .to(torch.bfloat16),
            requires_grad=True,
        )
        for i in range(3)
    ]
    groups = []
    for i, subset in enumerate((params[:2], params[2:])):
        group = {
            k: (i + 1 if k in ("lr_mult", "wd_mult") else False)
            for k in param_group_identifier_keys
        }
        groups.append({**group, "params": subset, "lr": 0.01 * (i + 1)})
    opt = FusedAdam(
        groups,
        master_weights=True,
        use_decoupled_grad=True,
        betas=(0.9, 0.98),
        weight_decay=0.1,
    )
    if initialize:
        for param in params:
            opt.initialize_state(param, False)
    wrapper = DistributedOptimizer.__new__(DistributedOptimizer)
    wrapper.optimizer = opt
    wrapper.config = SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=True,
        low_memory_resume=False,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        main_params_dtype=torch.float32,
        store_param_remainders=False,
        bf16=True,
        fp16=False,
    )
    wrapper.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    wrapper.grad_scaler = None
    wrapper.data_parallel_group = dist.group.WORLD
    wrapper.data_parallel_group_idx = 0
    wrapper.distributed_optimizer_instance_id = 0
    wrapper.model_param_group_index_map = {
        params[0]: (0, 0),
        params[1]: (0, 1),
        params[2]: (1, 0),
    }
    ranges = {}
    for i, p in enumerate(params):
        start = i * (4 + gap)
        ranges[p] = {
            "gbuf_local": Range(start, start + 4),
            "gbuf_world": Range(start, start + 4),
        }
    wrapper.gbuf_ranges = [{torch.float32: [{"param_map": ranges}]}]
    numel = 12 + 2 * gap
    wrapper.per_bucket_numel = [[numel]]
    wrapper.per_bucket_numel_unpadded = [[numel]]
    wrapper.buffers = [
        SimpleNamespace(
            buckets=[
                SimpleNamespace(
                    numel_unpadded=numel, grad_data=torch.zeros(numel, device=device)
                )
            ]
        )
    ]
    return wrapper, params


def _advance(wrapper, params, step):
    for index, param in enumerate(params):
        param.decoupled_grad = torch.full(
            param.shape,
            (index + 1) * 0.125 + step * 0.01,
            device="cuda",
            dtype=torch.float32,
        )
    wrapper.optimizer.step()


def _snapshot(wrapper, params):
    return [
        {
            name: tensor.clone()
            for name, tensor in wrapper.optimizer.state[param].items()
        }
        for param in params
    ]


@pytest.mark.parametrize("gap", [0, 2])
def test_optimizer_checkpoint_preserves_state_and_next_step(tmp_path, gap):
    from megatron.core.dist_checkpointing import serialization

    metadata = {"distrib_optim_sharding_type": "dp_reshardable"}
    source, params = _make_wrapper(gap, initialize=True)
    for step in range(3):
        _advance(source, params, step)
    expected = _snapshot(source, params)
    model_values = [param.detach().clone() for param in params]
    serialization.save(
        source.sharded_state_dict({}, metadata=metadata),
        str(tmp_path),
        sharded_strategy=("torch_dist", 1),
    )
    restored, new_params = _make_wrapper(gap, initialize=False)
    for param, value in zip(new_params, model_values, strict=True):
        param.data.copy_(value)
    loaded = serialization.load(
        restored.sharded_state_dict({}, is_loading=True, metadata=metadata),
        str(tmp_path),
    )
    entries = loaded["param_state"][0][torch.float32][0]
    assert [entry["padding"] for entry in entries] == (
        [False, False, False] if gap == 0 else [False, True, False, True, False]
    )

    restored.load_state_dict(copy.deepcopy(loaded))
    for actual, wanted in zip(_snapshot(restored, new_params), expected, strict=True):
        assert set(actual) == {"master_param", "exp_avg", "exp_avg_sq"}
        for name in wanted:
            torch.testing.assert_close(actual[name], wanted[name], rtol=0, atol=0)
    assert [group["step"] for group in restored.optimizer.param_groups] == [3, 3]
    _advance(source, params, 3)
    _advance(restored, new_params, 3)
    for param, restored_param in zip(params, new_params, strict=True):
        torch.testing.assert_close(param, restored_param, rtol=0, atol=0)
    for actual, wanted in zip(
        _snapshot(restored, new_params), _snapshot(source, params), strict=True
    ):
        for name in wanted:
            torch.testing.assert_close(actual[name], wanted[name], rtol=0, atol=0)

    malformed = copy.deepcopy(loaded)
    real_entry = next(
        entry
        for entry in malformed["param_state"][0][torch.float32][0]
        if not entry.get("padding", False)
    )
    real_entry["exp_avg"] = False
    with pytest.raises(AttributeError, match="dtype"):
        restored.load_state_dict(malformed)
