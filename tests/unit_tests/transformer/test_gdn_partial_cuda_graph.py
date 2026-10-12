# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Qwen3.5-style partial capture must include GDN, softmax attention and MoE prefixes."""

import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.context_parallel_layout.graph import (
    supports_packed_gdn_graph,
    validate_packed_gdn_graph_inputs,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import AttnBackend, CudaGraphModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.transformer.dcp_graph_test_utils import (
    assert_gradients_close,
    dcp_batch,
    dcp_test_context,
    dcp_train_step,
    track_graph_replays,
)

pytestmark = pytest.mark.internal
_CU_NAMES = ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")
_SCOPE = ["attn", "moe_router", "moe_preprocess"]


def _config(**overrides):
    values = dict(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=256,
        ffn_hidden_size=512,
        experimental_attention_variant="gdn",
        linear_attention_freq=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_cp_mode="chunkwise",
        cp_partition_mode="contiguous",
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_backend=AttnBackend.fused,
        gradient_accumulation_fusion=False,
        add_bias_linear=False,
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        qk_layernorm=True,
        attention_output_gate=True,
        num_moe_experts=4,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_shared_expert_gate=True,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_expert_rank_capacity_factor=1.25,
        moe_use_grouped_tensor=True,
        dynamic_context_parallel=True,
        max_seqlen_per_dp_cp_rank=512,
        thd_max_packed_sequences=8,
        sequence_packing_scheduler="default_dynamic_cp",
        pad_packed_seq_alignment="max",
        thd_tail_padding_policy="append_dummy_seq",
        cuda_graph_impl="transformer_engine",
        cuda_graph_modules=_SCOPE,
        cuda_graph_dynamic_microbatches=True,
        cuda_graph_warmup_steps=3,
    )
    values.update(overrides)
    return TransformerConfig(**values)


@pytest.mark.parametrize("spelling", ["gdn", "gated_delta_net"])
def test_partial_gdn_config_accepts_canonical_and_legacy_names(spelling):
    config = _config(experimental_attention_variant=spelling)
    assert config.experimental_attention_variant == "gdn"
    assert supports_packed_gdn_graph(config)
    assert config.cuda_graph_modules == [
        CudaGraphModule.attn,
        CudaGraphModule.moe_router,
        CudaGraphModule.moe_preprocess,
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"linear_cp_mode": "headwise"},
        {"cp_partition_mode": "zigzag"},
        {"tensor_model_parallel_size": 2, "sequence_parallel": True},
        {"max_seqlen_per_dp_cp_rank": 510},
        {"cuda_graph_impl": "local"},
    ],
)
def test_partial_gdn_config_rejects_unsupported_geometry(changes):
    with pytest.raises((ValueError, AssertionError)):
        _config(**changes)


def _params(config, size, *, valid=None, physical=None):
    end = config.max_seqlen_per_dp_cp_rank * size
    physical = physical or [0, 128 * size, end]
    valid = valid or physical
    tensors = []
    for boundaries in (valid, valid, physical, physical):
        tensors.append(
            torch.tensor(
                boundaries
                + [boundaries[-1]] * (config.thd_max_packed_sequences + 1 - len(boundaries)),
                dtype=torch.int32,
            )
        )
    return PackedSeqParams(
        qkv_format="thd",
        cp_partition_mode="contiguous",
        local_cp_size=size,
        cp_group=object(),
        **dict(zip(_CU_NAMES, tensors)),
    )


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32])
def test_partial_gdn_metadata_roundtrip_preserves_group_and_graph_backend(size):
    config = _config()
    params = _params(config, size, valid=[0, 100 * size, 480 * size])
    config._cuda_graph_capture_dynamic_cp = (size, params.cp_group)
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.self_attention = IdentityOp()
    kwargs = dict(packed_seq_params=params)
    layer._decompose_packed_seq_params_to_kwargs(kwargs)
    assert "packed_seq_params" not in kwargs
    layer._reconstruct_packed_seq_params_from_kwargs(kwargs)
    restored = kwargs["packed_seq_params"]
    assert restored.gdn_graph_safe
    assert restored.local_cp_size == size and restored.cp_group is params.cp_group
    for name in _CU_NAMES:
        assert getattr(restored, name) is getattr(params, name)


def test_partial_gdn_revalidates_mutated_packs():
    config = _config()
    params = _params(config, 4)
    validate_packed_gdn_graph_inputs(config, params)
    for name in _CU_NAMES:
        getattr(params, name)[1] += 1
    with pytest.raises(ValueError, match="64/2CP-aligned"):
        validate_packed_gdn_graph_inputs(config, params)
    params = _params(config, 4, physical=[0, 0, 2048])
    with pytest.raises(ValueError, match="empty physical"):
        validate_packed_gdn_graph_inputs(config, params)


def test_moe_only_scope_does_not_require_gdn_graph_geometry():
    config = _config(
        cuda_graph_modules=["moe_router"], tensor_model_parallel_size=2, sequence_parallel=True
    )
    assert not supports_packed_gdn_graph(config)
    params = _params(config, 4)
    config._cuda_graph_capture_dynamic_cp = (4, params.cp_group)
    layer = TransformerLayer.__new__(TransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.config = config
    layer.self_attention = IdentityOp()
    kwargs = dict(packed_seq_params=params)
    layer._decompose_packed_seq_params_to_kwargs(kwargs)
    layer._reconstruct_packed_seq_params_from_kwargs(kwargs)
    assert not getattr(kwargs["packed_seq_params"], "gdn_graph_safe", False)


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 4, reason="requires four ranks")
def test_partial_gdn_moe_mtp_dcp_replay_matches_eager(monkeypatch):
    with dcp_test_context(monkeypatch, expert_model_parallel_size=4) as groups:
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
        config = _config(expert_model_parallel_size=4, mtp_num_layers=1)
        spec = get_transformer_block_with_experimental_attention_variant_spec(config)
        mtp_spec = get_gpt_mtp_block_spec(config, spec, use_transformer_engine=True)
        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(42)
        model = GPTModel(
            config=config,
            transformer_layer_spec=spec,
            vocab_size=128,
            max_sequence_length=2048,
            position_embedding_type="rope",
            mtp_block_spec=mtp_spec,
            pg_collection=groups,
        ).cuda()
        # Match get_model()'s DDP initialization: persistent AccumulateGrad nodes
        # must not bind to the legacy default stream before graph capture.
        init_stream = torch.cuda.Stream()
        init_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(init_stream):
            _grad_accs = [
                parameter.expand_as(parameter).grad_fn.next_functions[0][0]
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
        torch.cuda.current_stream().wait_stream(init_stream)
        model.zero_grad_buffer = lambda: model.zero_grad(set_to_none=True)
        helper = None
        try:
            cases = [(size, lengths) for size in (1, 2, 4) for lengths in ((384,), (128, 256))]
            batches = [
                dcp_batch(config, groups, size, lengths, index)
                for index, (size, lengths) in enumerate(cases)
            ]
            references = []
            for index, batch in enumerate(batches):
                values, gradients = dcp_train_step(model, batch, index, cpu=True)
                assert torch.isfinite(values).all(), ("eager loss", index)
                for name, gradient in gradients.items():
                    assert torch.isfinite(gradient).all(), ("eager gradient", index, name)
                references.append((values, gradients))
                print(
                    f"GDN_PARTIAL_EAGER_REFERENCE rank={torch.distributed.get_rank()} "
                    f"case={index} local_valid_tokens={batch['loss_mask'].sum().item()}",
                    flush=True,
                )
            helper = TECudaGraphHelper(
                [model],
                config,
                seq_length=2048,
                micro_batch_size=1,
                pg_collection=groups,
                dynamic_cp_group_getter=parallel_state.get_dynamic_data_context_parallel_groups,
            )
            assert len(helper.flattened_callables) == 5, "all four layers and MTP must be captured"
            assert (
                sum(
                    type(layer.self_attention).__name__ == "GatedDeltaNet"
                    for layer in helper.flattened_callables
                )
                == 3
            )
            capture_bank = helper._capture_graph_bank
            import transformer_engine.pytorch.graph as te_graph

            copy_span = te_graph._copy_tensor_storage_span
            compact_alpha_copies = []

            def checked_copy_span(target, source, size_bytes):
                if source.dtype == torch.bfloat16 and source.shape == (1, 512, 4):
                    assert source.is_contiguous()
                    assert size_bytes == source.numel() * source.element_size()
                    compact_alpha_copies.append(size_bytes)
                return copy_span(target, source, size_bytes)

            monkeypatch.setattr(te_graph, "_copy_tensor_storage_span", checked_copy_span)

            def checked_capture_bank(callables, sample_args, kwargs):
                assert len(callables) == len(kwargs["sample_kwargs"]) == 15
                for callable_, inputs in zip(callables, kwargs["sample_kwargs"]):
                    cp_size = callable_._capture_context[0]
                    assert inputs["rotary_pos_emb"].shape[0] == 512 * cp_size
                return capture_bank(callables, sample_args, kwargs)

            monkeypatch.setattr(helper, "_capture_graph_bank", checked_capture_bank)
            helper.create_cudagraphs()
            assert compact_alpha_copies, "captured GDN alpha snapshots must use compact storage"
            print(f"GDN_PARTIAL_CAPTURE_COMPLETE rank={torch.distributed.get_rank()}", flush=True)
            assert helper.graphs_created()
            for layer in helper.flattened_callables:
                assert set(layer.cuda_graphs_by_dynamic_cp_size) == {1, 2, 4}
            replays = track_graph_replays(monkeypatch)
            for microbatch, index in enumerate((*range(6), *reversed(range(6)), 0)):
                before = len(replays)
                values, gradients = dcp_train_step(model, batches[index], microbatch, cpu=True)
                assert len(replays) - before >= 10, "every layer must replay forward AND backward"
                expected_values, expected_gradients = references[index]
                torch.testing.assert_close(values, expected_values, rtol=1e-3, atol=1e-3)
                assert_gradients_close(gradients, expected_gradients, index)
                print(
                    f"GDN_PARTIAL_REPLAY_PASS rank={torch.distributed.get_rank()} replay={microbatch} case={index}",
                    flush=True,
                )
            print("GDN_PARTIAL_PASS layers=4 mtp=1 scope=attn,moe_router,moe_preprocess cp=1,2,4")
        finally:
            torch.cuda.synchronize()
            if helper is not None:
                helper.delete_cuda_graphs()
