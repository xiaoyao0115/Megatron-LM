# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import gc
import os
import weakref

import pytest
import torch

from megatron.core.tensor_parallel.layers import (
    LinearWithGradAccumulationAndAsyncCommunication,
    _grad_accum_fusion_available,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _grad_accum_fusion_available,
    reason="Requires CUDA and the fused weight-gradient extension",
)


@pytest.fixture(autouse=True)
def _local_cuda_device():
    """Keep independent unit-test workers on their assigned local GPUs."""
    with torch.cuda.device(int(os.environ.get("LOCAL_RANK", "0"))):
        yield


def _linear(inp, weight, fused=True):
    return LinearWithGradAccumulationAndAsyncCommunication.apply(
        inp, weight, None, fused, False, False, None, 0, None
    )


def _weight(zero_dummy):
    values = torch.arange(32 * 32, device="cuda").reshape(32, 32) % 7 / 16
    weight = torch.nn.Parameter(values.to(torch.bfloat16))
    weight.main_grad = torch.full_like(weight, 0.25, dtype=torch.float32)
    weight.grad_added_to_main_grad = False
    weight.zero_out_wgrad = zero_dummy
    return weight


def _input(offset=0):
    values = torch.arange(16 * 2 * 32, device="cuda").reshape(16, 2, 32) % 11 / 8
    return (values + offset).to(torch.bfloat16).requires_grad_(True)


@pytest.mark.parametrize("use_hooks", [False, True], ids=["plain", "hooks"])
@pytest.mark.parametrize("zero_dummy", [False, True], ids=["dummy", "zero_dummy"])
def test_fused_linear_keeps_weight_owner(use_hooks, zero_dummy):
    """Identity saved hooks must preserve fused wgrad and the original DDP flags."""
    weight = _weight(zero_dummy)
    inp = _input()
    expected_input = inp.detach().clone().requires_grad_(True)
    expected_weight = weight.detach().clone().requires_grad_(True)
    expected = torch.nn.functional.linear(expected_input, expected_weight)
    expected.sum().backward()
    hooks = (
        torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor, lambda tensor: tensor)
        if use_hooks
        else contextlib.nullcontext()
    )
    with hooks:
        actual = _linear(inp, weight)
    actual.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(inp.grad, expected_input.grad)
    # Fused wgrad accumulates in FP32; a BF16 leaf gradient has an extra rounding step.
    expected_main_grad = inp.detach().float().sum(dim=(0, 1)).expand_as(weight)
    torch.testing.assert_close(weight.main_grad, expected_main_grad + 0.25)
    assert weight.grad_added_to_main_grad is True
    assert weight.grad is not None, "DDP needs the dummy gradient to run AccumulateGrad"
    if zero_dummy:
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight.grad))


@pytest.mark.parametrize("use_hooks", [False, True], ids=["plain", "hooks"])
def test_nonfused_linear_saved_hooks_match_torch(use_hooks):
    """Normal returned parameter gradients keep their numerical behavior."""
    weight = _weight(False)
    inp = _input()
    expected_input = inp.detach().clone().requires_grad_(True)
    expected_weight = weight.detach().clone().requires_grad_(True)
    expected = torch.nn.functional.linear(expected_input, expected_weight)
    expected.sum().backward()
    hooks = (
        torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor, lambda tensor: tensor)
        if use_hooks
        else contextlib.nullcontext()
    )
    with hooks:
        actual = _linear(inp, weight, fused=False)
    actual.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(inp.grad, expected_input.grad)
    torch.testing.assert_close(weight.grad, expected_weight.grad)
    torch.testing.assert_close(weight.main_grad, torch.full_like(weight.main_grad, 0.25))
    assert weight.grad_added_to_main_grad is False


def test_fused_linear_reports_released_weight_owner():
    """A temporary weight wrapper cannot silently lose its fused-gradient owner."""
    leaf = _weight(False)
    weight = leaf * 1
    weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
    weight.grad_added_to_main_grad = False
    owner_ref = weakref.ref(weight)
    inp = _input()
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor.detach(), lambda x: x):
        output = _linear(inp, weight)
    del weight
    gc.collect()
    assert owner_ref() is None
    with pytest.raises(RuntimeError, match="original weight was released"):
        output.sum().backward()


@pytest.mark.parametrize("zero_dummy", [False, True], ids=["dummy", "zero_dummy"])
def test_fused_linear_slot_capture_with_retained_accumulator(zero_dummy):
    """A DDP-style retained accumulator must join capture through its dummy gradient."""
    from transformer_engine.pytorch import make_graphed_callables

    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = _weight(zero_dummy)

        def forward(self, inp):
            return _linear(inp, self.weight)

    module = Module()
    owner_stream = torch.cuda.Stream()
    owner_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(owner_stream):
        temporary = module.weight.expand_as(module.weight)
        accumulator = temporary.grad_fn.next_functions[0][0]
    torch.cuda.current_stream().wait_stream(owner_stream)
    del temporary
    graphed = make_graphed_callables(
        (module,),
        ((_input(),),),
        num_warmup_iters=2,
        _order=[1, -1],
        _num_layers_per_chunk=[1],
        _reuse_graph_input_output_buffers=True,
        _graph_memory_slots=((0, 1, 0, 0, 0, 0, 1),),
    )
    try:
        for offset in (0.0, 0.25, 0.5):
            module.weight.main_grad.zero_()
            module.weight.grad = None
            inp = _input(offset)
            expected_input = inp.detach().clone().requires_grad_(True)
            expected_weight = module.weight.detach().clone().requires_grad_(True)
            expected = torch.nn.functional.linear(expected_input, expected_weight)
            expected.sum().backward()
            actual = graphed[0](inp)
            actual.sum().backward()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(inp.grad, expected_input.grad)
            expected_main_grad = inp.detach().float().sum(dim=(0, 1)).expand_as(module.weight)
            torch.testing.assert_close(module.weight.main_grad, expected_main_grad)
            assert module.weight.grad_added_to_main_grad is True
            assert module.weight.grad is not None
    finally:
        graphed[0].reset()
        del accumulator
