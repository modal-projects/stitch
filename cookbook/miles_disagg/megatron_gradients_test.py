"""Native Megatron norm and clipping contracts; run with two-rank torchrun."""

import os

import pytest

torch = pytest.importorskip("torch")
clip_grads = pytest.importorskip("megatron.core.optimizer.clip_grads")


@pytest.fixture(scope="module", autouse=True)
def process_group(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA and native fused multi-tensor kernels")
    if clip_grads.l2_norm_impl.__name__ == "local_multi_tensor_l2_norm":
        pytest.skip("The local fallback does not reproduce native dtype dispatch")
    owns_group = not torch.distributed.is_initialized()
    if owns_group:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if "RANK" in os.environ:
            torch.distributed.init_process_group("nccl")
        else:
            rendezvous = tmp_path_factory.mktemp("distributed") / "rendezvous"
            torch.distributed.init_process_group(
                "nccl", init_method=rendezvous.as_uri(), rank=0, world_size=1
            )
    yield
    if owns_group:
        torch.distributed.destroy_process_group()


DTYPES = [
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.bfloat16),
    (torch.float32, torch.float32),
]


def padded_grads(dtypes, numel, value):
    # Bound a regressed kernel's wider loads and detect writes past the logical view.
    bases = [torch.ones(4 * numel, device="cuda", dtype=dtype) for dtype in dtypes]
    return bases, [base[:numel].fill_(value) for base in bases]


@pytest.mark.parametrize("dtypes", DTYPES)
@pytest.mark.parametrize("layout", ["both", "rank0_empty", "disjoint", "all_empty"])
@pytest.mark.parametrize("value", [0.0, 1.0 / 64])
def test_norm(dtypes, layout, value):
    numel = 130
    bases, grads = padded_grads(dtypes, numel, value)
    rank = torch.distributed.get_rank()
    if layout == "all_empty" or (layout == "rank0_empty" and rank == 0):
        grads = []
    elif layout == "disjoint":
        grads = [grads[rank % 2]]
    reference = torch.zeros(1, device="cuda", dtype=torch.float64)
    for grad in grads:
        reference += grad.double().square().sum()
    torch.distributed.all_reduce(reference)
    actual = clip_grads.get_grad_norm_fp32(
        grads, grad_stats_parallel_group=torch.distributed.group.WORLD
    )
    assert float(actual) == pytest.approx(float(reference.sqrt()), rel=2e-6, abs=1e-8)
    assert all(torch.all(base[numel:] == 1) for base in bases)


@pytest.mark.parametrize("dtypes", DTYPES)
@pytest.mark.parametrize("clip", [False, True])
def test_decoupled_clipping(dtypes, clip):
    numel = 130
    bases, grads = padded_grads(dtypes, numel, 1.0 / 64)
    params = [torch.nn.Parameter(torch.zeros_like(grad)) for grad in grads]
    for param, grad in zip(params, grads, strict=True):
        param.decoupled_grad = grad
    params.append(torch.nn.Parameter(torch.zeros(1, device="cuda")))
    total_norm = 2.0
    max_norm = 1.0000005 if clip else 4.0
    coefficient = min(float(max_norm / (total_norm + 1e-6)), 1.0)
    expected = [base.clone() for base in bases]
    for base in expected:
        base[:numel].mul_(coefficient)
    clip_grads.clip_grad_by_total_norm_fp32(
        params, max_norm, total_norm, use_decoupled_grad=True
    )
    for actual, reference in zip(bases, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_empty_clipping(monkeypatch):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("Empty gradient lists must not launch a fused kernel")

    monkeypatch.setattr(clip_grads, "multi_tensor_applier", unexpected_launch)
    clip_grads.clip_grad_by_total_norm_fp32([], 1.0, 2.0, use_decoupled_grad=True)


def test_ordinary_fp32_clipping():
    param = torch.nn.Parameter(torch.zeros(130, device="cuda"))
    param.grad = torch.ones_like(param)
    clip_grads.clip_grad_by_total_norm_fp32([param], 1.0000005, 2.0)
    torch.testing.assert_close(param.grad, torch.full_like(param, 0.5), rtol=0, atol=0)
