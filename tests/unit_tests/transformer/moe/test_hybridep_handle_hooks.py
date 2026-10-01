# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""HybridEP's real autograd functions expose routing storage to saved-tensor hooks."""

import pytest
import torch

from megatron.core.transformer.moe import fused_a2a


@pytest.mark.parametrize("operation", ["dispatch", "combine"])
def test_native_handle_tensors_are_restored_through_saved_tensor_hooks(monkeypatch, operation):
    handle = (torch.tensor(2.0), None, 7, torch.tensor([0, 1], dtype=torch.int32), (3, 4))
    packed = []
    observed_backward_handles = []

    class Buffer:
        def dispatch_with_permute(self, *, hidden, probs=None, handle=None, **kwargs):
            if handle is not None:
                observed_backward_handles.append(handle)
                return hidden * handle[0], None, None, None, None
            return hidden * 2, probs * 2, None, torch.tensor([hidden.shape[0]]), initial_handle

        def combine_with_unpermute(self, *, hidden, handle, probs=None, **kwargs):
            if operation == "dispatch":
                observed_backward_handles.append(handle)
            return hidden * handle[0], None if probs is None else probs * handle[0]

    initial_handle = handle
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", Buffer())

    def pack(value):
        packed.append(value)
        return value.clone()

    x = torch.ones(4, 8, requires_grad=True)
    probabilities = torch.ones(4, 1, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        if operation == "dispatch":
            outputs = fused_a2a.HybridEPDispatch.apply(
                x,
                torch.ones(4, 1, dtype=torch.bool),
                probabilities,
                None,
                1,
                None,
                None,
                None,
                None,
                False,
                4,
                1,
                108,
                None,
                None,
            )
            loss = outputs[0].sum() + outputs[1].sum()
        else:
            loss = fused_a2a.HybridEPCombine.apply(x, handle, 4, 1, False).sum()
    assert len(packed) == 2 and packed[0] is handle[0] and packed[1] is handle[3]
    # Reuse the original routing storage after forward, as another live graph slot can do.
    handle[0].fill_(9)
    handle[3].fill_(-1)
    loss.backward()
    assert torch.equal(x.grad, torch.full_like(x, 2))
    if operation == "dispatch":
        assert torch.equal(probabilities.grad, torch.full_like(probabilities, 2))
    assert len(observed_backward_handles) == 1
    restored = observed_backward_handles[0]
    assert restored[1:3] == (None, 7) and restored[4] == (3, 4)
    assert torch.equal(restored[3], torch.tensor([0, 1], dtype=torch.int32))
