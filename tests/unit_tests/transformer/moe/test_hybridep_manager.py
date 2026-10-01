# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.token_dispatcher import _HybridEPManager


@pytest.mark.parametrize(
    ("expert_rank_capacity_factor", "expected_num_permuted_tokens"), [(None, None), (1.5, 128)]
)
def test_combine_only_releases_dynamic_token_count(
    monkeypatch, expert_rank_capacity_factor, expected_num_permuted_tokens
):
    manager = object.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(moe_permute_fusion_into_hybridep=False)
    manager.handle = object()
    manager.num_permuted_tokens = 128
    manager.pad_multiple = None
    manager.drop_and_pad = False
    manager.moe_expert_rank_capacity_factor = expert_rank_capacity_factor
    manager._original_num_tokens = None
    manager._padded_num_tokens = None

    def fake_hybrid_ep_combine(**kwargs):
        assert kwargs["handle"] is manager.handle
        assert kwargs["num_permuted_tokens"] == 128
        return kwargs["x"]

    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_combine", fake_hybrid_ep_combine
    )

    hidden_states = torch.empty(4, 8)
    assert manager.combine(hidden_states) is hidden_states
    assert manager.handle is None
    assert manager.num_permuted_tokens == expected_num_permuted_tokens


def _replay_manager(rows=512, *, padded_rows=None, dense=False):
    manager = object.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(
        moe_permute_fusion_into_hybridep=True,
        moe_hybridep_pad_variable_tokens=padded_rows is not None,
    )
    manager.num_experts = 32
    manager.router_topk = 6
    padded_rows = rows if padded_rows is None else padded_rows
    manager.token_probs = torch.ones(padded_rows, 32)
    manager.routing_map = None if dense else torch.ones(padded_rows, 32, dtype=torch.bool)
    manager.topk_idx = torch.zeros(padded_rows, 6, dtype=torch.int16) if dense else None
    manager._original_num_tokens = None
    manager._padded_num_tokens = None
    manager.num_permuted_tokens = 13568
    manager.drop_and_pad = False
    manager.moe_expert_rank_capacity_factor = 4.416666666666667
    manager.pad_multiple = 256
    manager.handle = object()
    return manager


@pytest.mark.parametrize("dense", [False, True])
def test_partial_replay_restores_counts_before_native_dispatch(monkeypatch, dense):
    from megatron.core.transformer.moe.token_dispatcher import MoEFlexTokenDispatcher

    manager = _replay_manager(dense=dense)
    dispatcher = object.__new__(MoEFlexTokenDispatcher)
    dispatcher._comm_manager = manager
    dispatcher.shared_experts = None
    banks = [_replay_manager(rows=rows, dense=dense) for rows in (512, 256)]
    calls = []

    def dispatch(self, hidden_states, async_finish, allocate_on_comm_stream):
        assert self._original_num_tokens == self._padded_num_tokens == hidden_states.shape[0]
        assert self.num_permuted_tokens == 13568
        calls.append((async_finish, allocate_on_comm_stream))
        self.dispatched_probs = self.token_probs
        return hidden_states

    monkeypatch.setattr(_HybridEPManager, "dispatch", dispatch)
    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_combine",
        lambda **kwargs: kwargs["x"],
    )
    for bank in (banks[0], banks[1], banks[0]):
        for attr in ("token_probs", "routing_map", "topk_idx"):
            dispatcher.set_cudagraph_attr("_comm_manager." + attr, getattr(bank, attr))
        tensor_ids = [
            id(getattr(manager, attr)) for attr in ("token_probs", "routing_map", "topk_idx")
        ]
        x = torch.empty(bank.token_probs.shape[0], 16)
        output, probabilities = dispatcher.token_dispatch(
            x, async_finish=False, allocate_on_comm_stream=False
        )
        assert output is x and probabilities is bank.token_probs
        assert manager.combine(output) is output
        assert manager._original_num_tokens is manager._padded_num_tokens is None
        assert manager.num_permuted_tokens == 13568
        assert tensor_ids == [
            id(getattr(manager, attr)) for attr in ("token_probs", "routing_map", "topk_idx")
        ]
    assert calls == [(False, False)] * 3


@pytest.mark.parametrize("dense", [False, True])
def test_restored_equalized_counts_trim_combine_to_original_input(monkeypatch, dense):
    manager = _replay_manager(rows=65, padded_rows=256, dense=dense)
    manager.restore_dispatch_token_counts(torch.empty(65, 16))
    assert manager._original_num_tokens == 65 and manager._padded_num_tokens == 256
    padded_output = torch.arange(256 * 16).reshape(256, 16)
    monkeypatch.setattr(
        "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_combine",
        lambda **kwargs: kwargs["x"],
    )
    actual = manager.combine(padded_output)
    assert torch.equal(actual, padded_output[:65])
    assert manager._original_num_tokens is manager._padded_num_tokens is None


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("_original_num_tokens", 512),
        ("_padded_num_tokens", 512),
        ("token_probs", None),
        ("token_probs", torch.zeros(512)),
        ("token_probs", torch.zeros(256, 32)),
        ("token_probs", torch.zeros(1024, 32)),
        ("token_probs", torch.zeros(512, 16)),
        ("routing_map", None),
        ("routing_map", torch.zeros(256, 32, dtype=torch.bool)),
        ("topk_idx", torch.zeros(512, 5, dtype=torch.int16)),
    ],
)
def test_replay_metadata_rejects_incomplete_or_stale_shapes(attribute, value):
    manager = _replay_manager()
    setattr(manager, attribute, value)
    original = manager._original_num_tokens, manager._padded_num_tokens
    with pytest.raises(RuntimeError):
        manager.restore_dispatch_token_counts(torch.empty(512, 16))
    assert (manager._original_num_tokens, manager._padded_num_tokens) == original
    assert manager.num_permuted_tokens == 13568
