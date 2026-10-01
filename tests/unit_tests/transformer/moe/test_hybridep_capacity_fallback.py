# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe import token_dispatcher
from megatron.core.transformer.moe.paged_stash import PagedStashRunner
from megatron.core.transformer.moe.token_dispatcher import _HybridEPManager


def _config(factor):
    return SimpleNamespace(
        moe_router_topk=2,
        moe_permute_fusion=True,
        moe_permute_fusion_into_hybridep=False,
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_expert_rank_capacity_factor=factor,
        moe_hybridep_pad_variable_tokens=False,
        moe_hybridep_routing_map_mode="bool",
        moe_flex_dispatcher_num_sms=32,
        moe_hybridep_num_blocks_permute=None,
        moe_hybridep_num_blocks_unpermute=None,
        moe_hybridep_num_sms_preprocessing=32,
        moe_use_grouped_tensor=True,
        use_transformer_engine_op_fuser=False,
        fp8=None,
        fp4=None,
        moe_paged_stash=False,
    )


def _runner(manager):
    """Use the actual retry preparation; unrelated model/stash work is empty."""
    runner = PagedStashRunner.__new__(PagedStashRunner)
    dispatcher = SimpleNamespace(
        _comm_manager=manager, reset_over_budget=lambda: manager.over_budget.zero_()
    )
    runner.moe_layers = [SimpleNamespace(token_dispatcher=dispatcher)]
    runner._configs_to_sync_moe_paged_stash = [manager.config]
    runner.stash_manager = SimpleNamespace(
        overflow=None, host_spill=None, release_stash_buffers=lambda: None
    )
    runner.model = []
    runner.optimizer = None
    runner.copy_main_params = False
    runner.forward_backward_func = lambda **kwargs: None
    return runner


@pytest.mark.parametrize("is_training", [True, False])
def test_first_dropless_retry_uses_new_native_counts(monkeypatch, is_training):
    manager = _HybridEPManager.__new__(_HybridEPManager)
    manager.config = _config(1.5)
    manager.group = object()
    manager.num_experts = manager.num_local_experts = 2
    manager.router_topk = 2
    manager.drop_and_pad = False
    manager.moe_expert_rank_capacity_factor = 1.5
    manager.num_permuted_tokens = 8
    manager.tokens_per_expert = torch.tensor([4, 4])
    manager.pad_multiple = 256
    manager.handle = None
    manager.over_budget = torch.ones(1, dtype=torch.bool)
    manager._original_num_tokens = manager._padded_num_tokens = None

    budgets = []
    native_counts = [torch.tensor([3, 9]), torch.tensor([7, 1])]

    def dispatch(**kwargs):
        budgets.append(kwargs['num_permuted_tokens'])
        counts = native_counts[len(budgets) - 1]
        values = torch.arange(int(counts.sum()) * 8).reshape(-1, 8).float()
        return values, torch.ones(values.shape[0]), None, counts, (torch.zeros(1),)

    monkeypatch.setattr(token_dispatcher, 'hybrid_ep_dispatch', dispatch)
    monkeypatch.setattr(token_dispatcher, 'hybrid_ep_combine', lambda **kwargs: kwargs['x'])
    _runner(manager).prepare_for_rerun(is_training=is_training)

    for counts in native_counts:
        manager.setup_metadata(torch.ones(4, 2, dtype=torch.bool), torch.ones(4, 2))
        dispatched = manager.dispatch(torch.ones(4, 8))
        assert budgets[-1] is None, 'the retry must not retain the failed static receive budget'
        torch.testing.assert_close(manager.get_number_of_tokens_per_expert(), counts)
        # Expert segmentation must cover the new output, including a larger first retry.
        experts = torch.split(dispatched, manager.tokens_per_expert.tolist())
        assert [expert.shape[0] for expert in experts] == counts.tolist()
        manager.combine(dispatched)
        assert manager.num_permuted_tokens is None
    assert not manager.over_budget.any()

    # The configured static policy is restored after retry; the next eager setup
    # must calculate its budget again, rather than reusing a dynamic token count.
    manager.moe_expert_rank_capacity_factor = manager.config.moe_expert_rank_capacity_factor
    manager.setup_metadata(torch.ones(4, 2, dtype=torch.bool), torch.ones(4, 2))
    assert manager.num_permuted_tokens == 256


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(token_dispatcher.hybrid_ep_dispatch is None, reason="HybridEP is required")
def test_real_hybridep_retry_matches_fresh_dropless():
    from megatron.core import parallel_state
    from megatron.core.transformer.moe.fused_a2a import reset_hybrid_ep_buffer
    from tests.unit_tests.test_utilities import Utils

    if Utils.world_size not in (4, 8):
        pytest.skip("requires 4 or 8 ranks with four-rank expert groups")
    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    group = parallel_state.get_expert_model_parallel_group()
    reset_hybrid_ep_buffer()
    try:
        tokens, width, experts = 256, 256, 8
        manager = _HybridEPManager(group, 2, experts, _config(1.0))
        routing = torch.zeros(tokens, experts, device='cuda', dtype=torch.bool)
        routing[:, :2] = True  # All ranks overload destination zero.
        values = (
            torch.arange(tokens * width, device='cuda').remainder(29).reshape(tokens, width)
            + group.rank()
        ).to(torch.bfloat16) / 8

        def run(comm):
            x = values.detach().clone().requires_grad_(True)
            probabilities = (routing.float() / 2).requires_grad_(True)
            comm.setup_metadata(routing, probabilities)
            dispatched = comm.dispatch(x)
            counts = comm.get_number_of_tokens_per_expert().clone()
            # Distinct expert scales make stale expert boundaries observable.
            scales = torch.arange(1, 3, device='cuda', dtype=torch.float32).repeat_interleave(
                counts.to(dispatched.device), output_size=int(counts.sum())
            )
            # Static receive buffers may have unused tail capacity.
            if scales.numel() < dispatched.shape[0]:
                scales = torch.nn.functional.pad(scales, (0, dispatched.shape[0] - scales.numel()))
            expert_output = (
                dispatched.float() * comm.dispatched_probs[:, None] * scales[:, None]
            ).to(torch.bfloat16)
            output = comm.combine(expert_output)
            output.backward(torch.full_like(output, 1 / 16))
            return tuple(t.detach().clone() for t in (output, x.grad, probabilities.grad, counts))

        run(manager)
        overflow = manager.over_budget.to(torch.int64)
        torch.distributed.all_reduce(overflow, group=group)
        assert overflow.item() == 1, 'the fixture must overflow exactly the overloaded destination'

        _runner(manager).prepare_for_rerun()
        retry = run(manager)
        fresh = run(_HybridEPManager(group, 2, experts, _config(None)))
        for actual, expected in zip(retry, fresh):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_counts = torch.full_like(
            retry[3], tokens * group.size() if group.rank() == 0 else 0
        )
        torch.testing.assert_close(retry[3], expected_counts, rtol=0, atol=0)
        torch.testing.assert_close(retry[0], values * 1.5, rtol=0, atol=0)
        torch.testing.assert_close(retry[1], torch.full_like(values, 1.5 / 16), rtol=0, atol=0)
        assert not manager.over_budget.any()
    finally:
        torch.cuda.synchronize()
        reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()
