# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpointed training preserves the ordinary router's reduction order."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.moe_utils import (
    get_default_pg_collection,
    topk_routing_with_score_function,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _logits():
    generator = torch.Generator(device="cpu").manual_seed(31415)
    return torch.randn((1280, 128), generator=generator).cuda()


@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("sort_topk", [None, False, True])
def test_topk_sort_default_and_explicit_order(grad_enabled, sort_topk):
    logits = _logits()
    expected_sort = grad_enabled if sort_topk is None else sort_topk
    values, expected_indices = torch.topk(logits, 8, dim=1, sorted=expected_sort)
    expected_probs = values.softmax(dim=-1)
    with torch.set_grad_enabled(grad_enabled):
        probs, indices = topk_routing_with_score_function(
            logits, 8, dense_output=True, sort_topk=sort_topk
        )
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(probs, expected_probs, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options",
    [
        {"score_function": "softmax"},
        {"score_function": "softmax", "use_pre_softmax": True},
        {"score_function": "sigmoid"},
        {"score_function": "sqrtsoftplus"},
    ],
)
def test_explicit_order_preserves_training_probabilities_and_gradients(options):
    logits = _logits().requires_grad_()
    weight = torch.linspace(-1, 1, 128, device="cuda")
    baseline_probs, baseline_map = topk_routing_with_score_function(logits, 8, **options)
    baseline_grad = torch.autograd.grad((baseline_probs * weight).square().sum(), logits)[0]
    for enabled in (False, True):
        with torch.set_grad_enabled(enabled):
            probs, routing_map = topk_routing_with_score_function(
                logits, 8, sort_topk=True, **options
            )
        torch.testing.assert_close(probs, baseline_probs, rtol=0, atol=0)
        torch.testing.assert_close(routing_map, baseline_map, rtol=0, atol=0)
        if enabled:
            gradient = torch.autograd.grad((probs * weight).square().sum(), logits)[0]
            torch.testing.assert_close(gradient, baseline_grad, rtol=0, atol=0)


def test_group_limited_order_is_unchanged():
    logits = _logits()
    baseline = topk_routing_with_score_function(logits, 8, num_groups=8, group_topk=2)
    for sort_topk in (False, True):
        with torch.no_grad():
            actual = topk_routing_with_score_function(
                logits, 8, num_groups=8, group_topk=2, sort_topk=sort_topk
            )
        for value, expected in zip(actual, baseline):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_unfused_sort_option_does_not_change_fused_api(monkeypatch):
    calls = []

    def fused_router(**kwargs):
        calls.append(kwargs)
        return kwargs["logits"], torch.ones_like(kwargs["logits"], dtype=torch.bool)

    monkeypatch.setattr(moe_utils, "HAVE_TE", True)
    monkeypatch.setattr(moe_utils, "fused_topk_with_score_function", fused_router)
    logits = _logits()
    baseline = topk_routing_with_score_function(logits, 8, fused=True)
    for sort_topk in (False, True):
        actual = topk_routing_with_score_function(logits, 8, fused=True, sort_topk=sort_topk)
        assert calls[-1].keys() == calls[0].keys()
        assert "sort_topk" not in calls[-1]
        assert actual[0] is baseline[0]


def test_checkpoint_order_records_and_replays_identical_probabilities():
    replay = RouterReplay()
    try:
        logits = _logits()
        expected = topk_routing_with_score_function(logits, 8)
        replay.set_router_replay_action(RouterReplayAction.RECORD)
        with torch.no_grad():
            recorded = topk_routing_with_score_function(
                logits, 8, sort_topk=True, router_replay=replay
            )
        replay.set_target_indices(replay.get_recorded_indices())
        replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        actual = topk_routing_with_score_function(logits, 8, router_replay=replay)
        for recorded_value, replayed_value, baseline in zip(recorded, actual, expected):
            torch.testing.assert_close(recorded_value, baseline, rtol=0, atol=0)
            torch.testing.assert_close(replayed_value, baseline, rtol=0, atol=0)
    finally:
        RouterReplay.global_router_replay_instances.remove(replay)


@pytest.fixture
def router():
    Utils.initialize_model_parallel(1, 1)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=128,
        moe_router_topk=8,
        moe_router_score_function="softmax",
        moe_router_load_balancing_type="none",
        moe_router_pre_softmax=False,
        use_cpu_initialization=True,
        add_bias_linear=False,
    )
    yield TopKRouter(config, pg_collection=get_default_pg_collection(), layer_number=1).cuda()
    Utils.destroy_model_parallel()


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_router_training_order_and_legacy_eval(router, training, grad_enabled):
    router.train(training)
    logits = _logits().view(1280, 1, 128)
    with torch.set_grad_enabled(training or grad_enabled):
        expected = topk_routing_with_score_function(logits.view(1280, 128), 8)
    with torch.set_grad_enabled(grad_enabled):
        actual = router.routing(logits)
    for value, baseline in zip(actual, expected):
        torch.testing.assert_close(value, baseline, rtol=0, atol=0)


def test_actual_router_checkpoint_matches_forward_and_backward(router):
    router.train()
    logits = _logits().view(1280, 1, 128).requires_grad_()
    upstream = torch.linspace(-1, 1, 128, device="cuda")
    expected_probs, expected_map = router.routing(logits)
    expected_gradient = torch.autograd.grad((expected_probs * upstream).square().sum(), logits)[0]
    probs, routing_map = checkpoint(router.routing, logits, use_reentrant=True)
    (probs * upstream).square().sum().backward()
    gradient = logits.grad
    torch.testing.assert_close(probs, expected_probs, rtol=0, atol=0)
    torch.testing.assert_close(routing_map, expected_map, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    assert torch.count_nonzero(gradient) > 0
