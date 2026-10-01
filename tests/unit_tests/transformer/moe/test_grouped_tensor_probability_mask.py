# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GroupedTensor capacity masking with real weighted activation and autograd."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from megatron.core.transformer.moe import experts


class _Offload:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *args):
        return False

    def group_offload(self, value, **kwargs):
        return value


class _NoHostScalar(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("Tensor values must not be extracted to Python")
        return func(*args, **(kwargs or {}))


class TestGroupedTensorProbabilityMask(unittest.TestCase):
    def setUp(self):
        adapters = patch.multiple(
            experts,
            skip_routed_expert_padding=lambda config: True,
            off_interface=lambda *args: _Offload(args[1]),
            apply_module=lambda module: module,
        )
        adapters.start()
        self.addCleanup(adapters.stop)
        self.config = SimpleNamespace(
            moe_apply_probs_on_input=False,
            delay_offload_until_cuda_graph=False,
            gated_linear_unit=True,
            moe_mlp_glu_interleave_size=None,
            use_te_activation_func=False,
            bias_activation_fusion=True,
            activation_func_fp8_input_store=False,
            activation_func_clamp_value=10.0,
        )
        self.model = SimpleNamespace(
            config=self.config,
            _with_fused_impl=False,
            _use_grouped_tensor=True,
            offload_expert_fc1=False,
            offload_moe_act=False,
            activation_recompute=False,
            activation_func=F.silu,
            linear_fc1=lambda x, splits: (x, None),
            linear_fc2=lambda x, splits: (x, None),
            _apply_bias=lambda out, *args: out,
        )
        self.model.forward = MethodType(experts.TEGroupedMLP.forward, self.model)
        generator = torch.Generator().manual_seed(2304)
        self.x = torch.randn(12, 16, generator=generator).to(torch.bfloat16)
        self.p = torch.rand(12, generator=generator)
        self.dy = torch.randn(12, 8, generator=generator).to(torch.bfloat16)

    def assertBits(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(
            torch.equal(
                actual.detach().contiguous().view(torch.uint8),
                expected.detach().contiguous().view(torch.uint8),
            )
        )

    def run_case(self, splits, *, reference=False, x=None, dy=None):
        x = (self.x if x is None else x).clone().requires_grad_(True)
        p = self.p.clone().requires_grad_(True)
        if reference:
            out = experts.weighted_bias_swiglu_impl(
                x, None, p.unsqueeze(-1), clamp_value=self.config.activation_func_clamp_value
            )
        else:
            out, bias = self.model.forward(x, splits, p)
            self.assertIsNone(bias)
        out.backward(self.dy if dy is None else dy)
        return out.detach(), x.grad, p.grad

    def test_dynamic_a_b_a_and_active_math_exact(self):
        splits = torch.tensor([4, 4, 4], dtype=torch.int64)
        split_address = splits.data_ptr()
        for clamp in (None, 10.0):
            self.config.activation_func_clamp_value = clamp
            reference = self.run_case(splits, reference=True)
            first = None
            for counts in ([4, 4, 4], [2, 1, 2], [4, 4, 4]):
                splits.copy_(torch.tensor(counts))
                actual = self.run_case(splits)
                active = sum(counts)
                self.assertEqual(splits.data_ptr(), split_address)
                for left, right in zip(actual, reference):
                    self.assertBits(left[:active], right[:active])
                self.assertTrue(
                    torch.equal(
                        actual[2][active:].view(torch.int32),
                        torch.zeros(12 - active, dtype=torch.int32),
                    )
                )
                if first is None:
                    first = actual
            for left, right in zip(first, actual):
                self.assertBits(left, right)

    def test_unused_nan_backward_selects_positive_zero(self):
        x = self.x.clone()
        x[5:] = float("nan")
        dy = self.dy.clone()
        dy[5:] = float("nan")
        splits = torch.tensor([2, 1, 2])
        reference = self.run_case(splits, reference=True, x=x, dy=dy)
        self.assertTrue(torch.isnan(reference[2][5:]).all())
        actual = self.run_case(splits, x=x, dy=dy)
        self.assertBits(actual[2][5:], torch.zeros(7))
        for left, right in zip(actual, reference):
            self.assertBits(left[:5], right[:5])

    def test_all_unused_rows_positive_zero(self):
        actual = self.run_case(
            torch.zeros(3, dtype=torch.int64), x=torch.full_like(self.x, float("nan"))
        )
        self.assertBits(actual[2], torch.zeros(12))

    def test_normalized_splits_no_host_scalar_and_no_input_mutation(self):
        splits = torch.tensor([2, 1, 2], dtype=torch.int32)
        saved = (splits.clone(), self.x.clone(), self.p.clone())
        observed = []

        def fc1(x, normalized):
            observed.append((normalized.dtype, normalized.device))
            return x, None

        self.model.linear_fc1 = fc1
        with _NoHostScalar():
            self.model.forward(self.x, splits, self.p)
        self.assertEqual(observed, [(torch.int64, self.x.device)])
        for actual, expected in zip((splits, self.x, self.p), saved):
            self.assertBits(actual, expected)

    def test_selected_probability_bits_and_noncontiguous_input(self):
        probs = torch.tensor([-0.0, 0.0, 1.0e-45, -1.0e-45, 0.5, 0.8] * 4)[::2]
        self.assertFalse(probs.is_contiguous())
        captured = []

        def activation(x, bias, weights, *args, **kwargs):
            captured.append(weights)
            return x[:, :8]

        with patch.object(experts, "weighted_bias_swiglu_impl", activation):
            self.model.forward(self.x, torch.tensor([2, 1, 2]), probs)
        self.assertBits(captured[0][:5, 0], probs[:5])
        self.assertBits(captured[0][5:, 0], torch.zeros(7))

    def test_non_grouped_path_does_not_add_mask(self):
        self.model._use_grouped_tensor = False
        captured = []

        def activation(x, bias, weights, *args, **kwargs):
            captured.append(weights)
            return x[:, :8]

        with patch.object(experts, "weighted_bias_swiglu_impl", activation):
            self.model.forward(self.x, torch.tensor([2, 1, 2]), self.p)
        self.assertBits(captured[0][:, 0], self.p)

    def test_fuser_returns_before_mask_or_split_normalization(self):
        self.model._with_fused_impl = True
        splits = object()
        sentinel = object()

        def fused(x, actual_splits, probs):
            self.assertIs(actual_splits, splits)
            self.assertIs(probs, self.p)
            return sentinel

        self.model._fused_forward = fused
        self.assertEqual(self.model.forward(self.x, splits, self.p), (sentinel, None))
