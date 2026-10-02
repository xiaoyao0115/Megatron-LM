# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Whole-chunk DCP graph banks must preserve packed metadata and replay ownership.

Run the numerical regression with two or four GPUs, for example::

    torchrun --nproc-per-node 2 -m pytest -xvs \
        tests/unit_tests/transformer/test_chunk_dcp_cuda_graph.py
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.chunk_cuda_graph import ChunkCudaGraphPostProcessBlock
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.transformer.dcp_graph_test_utils import (
    assert_gradients_close,
    dcp_batch,
    dcp_test_context,
)
from tests.unit_tests.transformer.dcp_graph_test_utils import dcp_train_step as _train_step
from tests.unit_tests.transformer.dcp_graph_test_utils import track_graph_replays

pytestmark = pytest.mark.internal

_CU_NAMES = ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")
_BLOCK_TYPES = (TransformerBlock, ChunkCudaGraphPostProcessBlock)
_REQUIRES_TWO_RANKS = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2 or torch.cuda.device_count() < 2,
    reason="requires torchrun with at least 2 GPUs",
)


def _metadata_block(block_type, *, cp_size=2, mtp_layers=0, pp_size=1):
    """Exercise the real adapters without constructing attention or MTP kernels."""
    block = block_type.__new__(block_type)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(
        dynamic_context_parallel=True,
        context_parallel_size=1,
        pipeline_model_parallel_size=pp_size,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        max_seqlen_per_dp_cp_rank=64,
        thd_max_packed_sequences=3,
        hidden_size=128,
        bf16=True,
        fp16=False,
        cuda_graph_impl="transformer_engine",
        cuda_graph_granularity="chunk",
        cuda_graph_dynamic_microbatches=True,
        sequence_packing_scheduler="default_dynamic_cp",
        cp_partition_mode="zigzag",
        _cuda_graph_capture_dynamic_cp=(cp_size, object()),
        enable_hyper_connections=False,
        moe_n_hash_layers=0,
        mtp_num_layers=mtp_layers or None,
        fine_grained_activation_offloading=False,
    )
    block.pre_process = False
    block.input_tensor = None
    object.__setattr__(block, "_owner", SimpleNamespace(mtp_process=bool(mtp_layers)))
    block.cuda_graph_manual_hooks = []
    block.cuda_graphs_by_dynamic_cp_size = {}
    block.cuda_graph_cp_groups_by_dynamic_cp_size = {}
    block._te_cuda_graph_static_hidden_inputs_by_dynamic_cp_size = {}
    return block


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("block_type", _BLOCK_TYPES)
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_chunk_static_inputs_use_capture_cp_capacity(block_type, cp_size):
    block = _metadata_block(block_type, cp_size=cp_size, mtp_layers=1)
    inputs = block.get_layer_static_inputs(seq_length=512, micro_batch_size=4)

    # DCP changes global packed capacity, while each rank retains the same local buffers.
    assert inputs["hidden_states"].shape == (32, 1, 128)
    for name in _CU_NAMES:
        torch.testing.assert_close(
            inputs[name],
            torch.tensor(
                [0, 64 * cp_size, 64 * cp_size, 64 * cp_size],
                dtype=torch.int32,
                device=inputs[name].device,
            ),
        )
    assert inputs["padding_mask"].shape == (1, 64)
    assert not inputs["padding_mask"].any()
    if block_type is ChunkCudaGraphPostProcessBlock:
        for name in ("input_ids", "position_ids", "labels", "loss_mask"):
            assert inputs[name].shape == (1, 64)


@pytest.mark.parametrize("block_type", _BLOCK_TYPES)
@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_chunk_capture_reconstructs_dcp_group_and_padding(block_type, cp_size):
    block = _metadata_block(block_type, cp_size=cp_size)
    # Distinct valid/physical endpoints expose lost inter-sequence padding semantics.
    valid = torch.tensor([0, 12 * cp_size, 40 * cp_size, 40 * cp_size], dtype=torch.int32)
    padded = torch.tensor([0, 16 * cp_size, 64 * cp_size, 64 * cp_size], dtype=torch.int32)
    tensors = dict(zip(_CU_NAMES, (valid, valid.clone(), padded, padded.clone())))
    kwargs = dict(tensors, unrelated="preserved")
    block._reconstruct_packed_seq_params_from_kwargs(kwargs)

    params = kwargs["packed_seq_params"]
    assert params.local_cp_size == cp_size
    assert params.cp_group is block.config._cuda_graph_capture_dynamic_cp[1]
    assert params.max_seqlen_q == params.max_seqlen_kv == 64 * cp_size
    assert params.qkv_format == "thd"
    assert params.pad_between_seqs
    assert params.cp_partition_mode == "zigzag"
    for name, tensor in tensors.items():
        assert getattr(params, name) is tensor
    assert kwargs["unrelated"] == "preserved"


def _install_fake_banks(block, slots=2):
    calls = []
    groups = {size: object() for size in (1, 2)}

    def graph(size, slot):
        def replay(hidden_states, **kwargs):
            assert "packed_seq_params" not in kwargs
            assert all(isinstance(kwargs[name], torch.Tensor) for name in _CU_NAMES)
            calls.append((size, slot, kwargs["is_first_microbatch"]))
            return hidden_states

        return replay

    block.cuda_graphs_by_dynamic_cp_size = {
        size: [graph(size, slot) for slot in range(slots)] for size in groups
    }
    block.cuda_graph_cp_groups_by_dynamic_cp_size = groups
    block.cuda_graphs = block.cuda_graphs_by_dynamic_cp_size[2]
    return calls, groups


def _replay_params(size, group):
    cu = torch.tensor([0, 64 * size, 64 * size, 64 * size], dtype=torch.int32)
    return PackedSeqParams(
        qkv_format="thd", local_cp_size=size, cp_group=group, **dict.fromkeys(_CU_NAMES, cu)
    )


@pytest.mark.parametrize("block_type", _BLOCK_TYPES)
def test_chunk_replay_selects_bank_before_metadata_decomposition(block_type):
    block = _metadata_block(block_type)
    calls, groups = _install_fake_banks(block)
    hidden = torch.ones(4, 1, 128)
    sizes = (1, 2, 2, 1, 2, 1)
    for microbatch, size in enumerate(sizes):
        block.current_microbatch = microbatch
        # A non-first PP chunk obtains its hidden states through set_input_tensor.
        block.input_tensor = hidden
        output = block._te_cuda_graph_replay(
            hidden_states=None, packed_seq_params=_replay_params(size, groups[size])
        )
        assert output is hidden
        assert block.cuda_graphs is block.cuda_graphs_by_dynamic_cp_size[size]
    assert calls == [(size, index % 2, index == 0) for index, size in enumerate(sizes)]

    with pytest.raises(RuntimeError, match="does not match"):
        block._te_cuda_graph_replay(hidden, packed_seq_params=_replay_params(1, object()))
    with pytest.raises(RuntimeError, match="local_cp_size"):
        block._te_cuda_graph_replay(hidden, packed_seq_params=_replay_params(4, object()))
    with pytest.raises(RuntimeError, match="local_cp_size"):
        block._te_cuda_graph_replay(hidden, packed_seq_params=None)
    assert len(calls) == len(sizes), "invalid metadata must fail before launching a graph"


@pytest.mark.parametrize("block_type", _BLOCK_TYPES)
@pytest.mark.parametrize("published_bound", [None, 8])
def test_chunk_pp_slot_wrap_requires_published_dcp_capacity(block_type, published_bound):
    block = _metadata_block(block_type, pp_size=2)
    calls, groups = _install_fake_banks(block)
    if published_bound is not None:
        block.config._cuda_graph_num_microbatches = published_bound
    block.current_microbatch = 3
    params = _replay_params(1, groups[1])
    if published_bound is None:
        with pytest.raises(AssertionError, match="microbatch"):
            block._te_cuda_graph_replay(torch.ones(1), packed_seq_params=params)
        assert not calls
    else:
        block._te_cuda_graph_replay(torch.ones(1), packed_seq_params=params)
        assert calls == [(1, 1, False)]
        block.current_microbatch = published_bound
        with pytest.raises(AssertionError, match="microbatch"):
            block._te_cuda_graph_replay(torch.ones(1), packed_seq_params=params)
        assert len(calls) == 1


@pytest.mark.parametrize("block_type", _BLOCK_TYPES)
def test_static_cp_chunk_keeps_pp_no_wrap_contract(block_type):
    block = _metadata_block(block_type, pp_size=2)
    calls, groups = _install_fake_banks(block)
    block.cuda_graphs_by_dynamic_cp_size = {}
    block.config.dynamic_context_parallel = False
    block.config._cuda_graph_num_microbatches = 8
    block.current_microbatch = 2
    with pytest.raises(AssertionError, match="microbatch"):
        block._te_cuda_graph_replay(torch.ones(1), packed_seq_params=_replay_params(1, groups[1]))
    assert not calls


def _build_dcp_model(groups, mtp_layers, granularity="chunk", max_cp_size=None):
    config = TransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=512,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        attention_backend=AttnBackend.fused,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        add_bias_linear=False,
        dynamic_context_parallel=True,
        max_seqlen_per_dp_cp_rank=64,
        thd_max_packed_sequences=4,
        sequence_packing_scheduler="default_dynamic_cp",
        pad_packed_seq_alignment="max",
        mtp_num_layers=mtp_layers or None,
        cuda_graph_impl="transformer_engine",
        cuda_graph_granularity=granularity,
        cuda_graph_modules=[] if granularity == "chunk" else ["attn"],
        max_dynamic_context_parallel_size=max_cp_size,
        cuda_graph_dynamic_microbatches=True,
    )
    spec = get_gpt_layer_with_transformer_engine_spec()
    mtp_spec = (
        get_gpt_mtp_block_spec(config, spec, use_transformer_engine=True) if mtp_layers else None
    )
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=128,
        max_sequence_length=64 * groups.dp_cp.size(),
        position_embedding_type="rope",
        mtp_block_spec=mtp_spec,
        pg_collection=groups,
    ).cuda()
    # TECudaGraphHelper accepts a DDP wrapper in training; this test owns leaf gradients directly.
    model.zero_grad_buffer = lambda: model.zero_grad(set_to_none=True)
    return model


def _dcp_batch(config, groups, size, local_lengths, seed):
    batch = dcp_batch(config, groups, size, local_lengths, seed)
    assert batch["input_ids"].shape == (1, 64)
    assert batch["packed_seq_params"].cu_seqlens_q.numel() == 5
    assert batch["packed_seq_params"].local_cp_size == size
    return batch


@_REQUIRES_TWO_RANKS
@pytest.mark.parametrize("mtp_layers", [0, 1])
def test_chunk_dcp_capture_replay_matches_eager(mtp_layers, monkeypatch):
    """Real decoder, LM loss and optional MTP F/B across changing CP banks and packs."""
    _check_dcp_capture_replay(mtp_layers, monkeypatch)


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 4, reason="requires four ranks")
@pytest.mark.parametrize("granularity", ["layer", "chunk"])
def test_dynamic_cp_max_capture_replay_matches_eager(granularity, monkeypatch):
    """CP1/2 native PG and graph banks must exclude the real four-rank DPxCP parent."""
    _check_dcp_capture_replay(0, monkeypatch, granularity=granularity, max_cp_size=2)


def _check_dcp_capture_replay(mtp_layers, monkeypatch, granularity="chunk", max_cp_size=None):
    with dcp_test_context(monkeypatch, max_dynamic_context_parallel_size=max_cp_size) as groups:
        helper = None
        try:
            getter = parallel_state.get_dynamic_data_context_parallel_groups
            model = _build_dcp_model(groups, mtp_layers, granularity, max_cp_size)
            maximum = groups.dp_cp.size() if max_cp_size is None else max_cp_size
            sizes = tuple(2**exponent for exponent in range(maximum.bit_length()))
            if max_cp_size is not None:
                assert groups.dp_cp.size() == 4
                assert set(parallel_state._DYNAMIC_DP_CP_GROUPS) == set(sizes)
                for size in sizes:
                    group = getter(group_size=size)
                    ranks = torch.distributed.get_process_group_ranks(group)
                    assert group.size() == size and ranks[0] % size == 0
                    assert ranks == list(range(ranks[0], ranks[0] + size))
            cases = [(size, lengths) for size in sizes for lengths in ((48,), (16, 32))]
            batches = [
                _dcp_batch(model.config, groups, size, lengths, index)
                for index, (size, lengths) in enumerate(cases)
            ]
            reference = [_train_step(model, batch, index) for index, batch in enumerate(batches)]

            helper = TECudaGraphHelper(
                [model],
                model.config,
                seq_length=64 * groups.dp_cp.size(),
                micro_batch_size=1,
                pg_collection=groups,
                dynamic_cp_group_getter=getter,
            )
            if granularity == "chunk":
                assert helper.flattened_callables == [model.decoder, model.postprocess_block]
            else:
                assert helper.flattened_callables == list(model.decoder.layers)
            assert helper._get_dynamic_cp_capture_sizes() == tuple(reversed(sizes))
            helper.create_cudagraphs()
            assert helper.graphs_created()
            for block in helper.flattened_callables:
                assert set(block.cuda_graphs_by_dynamic_cp_size) == set(sizes)
                assert all(len(bank) == 1 for bank in block.cuda_graphs_by_dynamic_cp_size.values())

            replays = track_graph_replays(monkeypatch)
            # Return to earlier banks and packs after intervening captures/replays overwrite buffers.
            order = (*range(len(batches)), *reversed(range(len(batches))), 0)
            for microbatch, index in enumerate(order):
                before = len(replays)
                values, gradients = _train_step(model, batches[index], microbatch)
                assert (
                    len(replays) - before >= 4
                ), "both captured modules must replay forward and backward"
                expected_values, expected_gradients = reference[index]
                torch.testing.assert_close(values, expected_values, rtol=1e-3, atol=1e-3)
                assert_gradients_close(gradients, expected_gradients, index)
            print(
                f"DCP_REPLAY_PASS granularity={granularity} mtp={mtp_layers} cp_sizes={sizes} microbatches={len(order)}"
            )
        finally:
            torch.cuda.synchronize()
            if helper is not None:
                helper.delete_cuda_graphs()
