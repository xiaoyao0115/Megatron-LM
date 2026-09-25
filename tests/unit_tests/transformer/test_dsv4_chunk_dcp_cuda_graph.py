# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small DSv4 eager/chunk-graph parity with identical weights, packs, and CP order.

Run with 2/4/8 GPUs and real DSv4 kernel dependencies. DSV4_CHUNK_CASES selects
comma-separated ``window``, ``indexer``, or ``full`` cases (default: ``full``).
These are PP1 checks, without an optimizer, MTP, FP8, or cross-rank experts.
"""

import hashlib
import inspect
import json
import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import get_batch_on_this_rank_for_sequence_packing
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper, set_current_microbatch
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.experimental_attention_variant.dsa_logging import (
    DSAIndexerLossLoggingHelper,
    initialize_dsa_metric_tracker,
)
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.internal

_LOCAL_TOKENS = 512
_VOCAB_SIZE = 128
_CASES = tuple(os.environ.get("DSV4_CHUNK_CASES", "full").split(","))
_CU_NAMES = ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")


def _probe_dependencies(with_indexer):
    """Fail when real kernels are absent; never substitute mocked attention kernels."""
    import cudnn
    import flash_mla
    import transformer_engine
    from cudnn import DSA
    from fast_hadamard_transform import hadamard_transform

    assert callable(flash_mla.flash_mla_sparse_fwd)
    assert callable(DSA.sparse_attention_backward_wrapper)
    assert callable(hadamard_transform)
    if with_indexer:
        assert "q_causal_offsets" in inspect.signature(DSA.indexer_forward_wrapper).parameters
    print(
        "DSV4_CHUNK_DEPENDENCIES "
        + json.dumps(
            {
                "torch": torch.__version__,
                "te": transformer_engine.__version__,
                "cudnn_module": cudnn.__file__,
                "flash_mla_module": flash_mla.__file__,
                "gpu": torch.cuda.get_device_name(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _build_model(groups, case):
    with_indexer, with_moe = case != "window", case == "full"
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=512,
        # The retained FlashMLA sparse-prefill kernel supports 64 or 128 heads.
        num_attention_heads=64,
        ffn_hidden_size=1024,
        q_lora_rank=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=512,
        o_groups=1,
        o_lora_rank=128,
        multi_latent_attention=True,
        qk_layernorm=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        rope_type="rope",
        mscale=1.0,
        mscale_all_dim=1.0,
        experimental_attention_variant="dsv4_hybrid",
        csa_compress_ratios=[0, 4 if with_indexer else 0],
        csa_window_size=128,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        # SM90 score recompute and indexer backward require a multiple of 128.
        dsa_indexer_topk=128,
        dsa_indexer_loss_coeff=1e-2 if with_indexer else 0.0,
        dsa_indexer_use_sparse_loss=True,
        dsa_kernel_backend="cudnn",
        dsa_indexer_precision="bf16",
        enable_hyper_connections=True,
        num_residual_streams=4,
        use_fused_mhc=True,
        num_moe_experts=4 if with_moe else None,
        moe_ffn_hidden_size=256 if with_moe else None,
        moe_router_topk=2,
        moe_grouped_gemm=with_moe,
        moe_token_dispatcher_type="alltoall",
        # Capacity = local_tokens: every token can reach any expert without dropping.
        moe_expert_capacity_factor=2.0 if with_moe else None,
        moe_pad_expert_input_to_capacity=with_moe,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=1e-2 if with_moe else 0.0,
        moe_router_enable_expert_bias=False,
        actual_vocab_size=_VOCAB_SIZE,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        attention_backend=AttnBackend.fused,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        add_bias_linear=False,
        dynamic_context_parallel=True,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="default_dynamic_cp",
        max_seqlen_per_dp_cp_rank=_LOCAL_TOKENS,
        pad_packed_seq_alignment="max",
        thd_max_packed_sequences=4,
        cuda_graph_impl="transformer_engine",
        cuda_graph_granularity="chunk",
        cuda_graph_dynamic_microbatches=True,
        cuda_graph_modules=[],
    )
    spec = get_transformer_block_with_experimental_attention_variant_spec(config)
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_LOCAL_TOKENS * groups.dp_cp.size(),
        position_embedding_type="rope",
        pg_collection=groups,
    ).cuda()
    model.zero_grad_buffer = lambda: model.zero_grad(set_to_none=True)
    initialize_dsa_metric_tracker(model, groups)
    DSAIndexerLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
    return model


def _batch(config, groups, cp_size, local_lengths, seed):
    lengths = [length * cp_size for length in local_lengths]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, _VOCAB_SIZE, (sum(lengths) + 1,), generator=generator).cuda()
    cu = torch.tensor([0, *lengths], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    raw = {
        "tokens": tokens[:-1].contiguous(),
        "labels": tokens[1:].contiguous(),
        "loss_mask": torch.ones(sum(lengths), device="cuda"),
        "position_ids": torch.cat([torch.arange(length) for length in lengths]).cuda(),
        "cu_seqlens": cu,
        "cu_seqlens_padded": cu.clone(),
        "max_seqlen": max(lengths),
        "local_cp_size": cp_size,
    }
    result = get_batch_on_this_rank_for_sequence_packing(
        iter([raw]), dynamic_cp=True, mtp_on_this_rank=False, pg_collection=groups, config=config
    )
    batch = dict(
        zip(
            (
                "input_ids",
                "labels",
                "loss_mask",
                "attention_mask",
                "position_ids",
                "packed_seq_params",
                "padding_mask",
            ),
            result,
        )
    )
    assert batch["input_ids"].shape == (1, _LOCAL_TOKENS)
    assert batch["packed_seq_params"].local_cp_size == cp_size
    assert batch["packed_seq_params"].cp_partition_mode == "contiguous"
    return batch


def _fingerprint(batch):
    digest = hashlib.sha256()
    tensors = [value for value in batch.values() if isinstance(value, torch.Tensor)]
    tensors.extend(getattr(batch["packed_seq_params"], name) for name in _CU_NAMES)
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _step(model, batch, microbatch):
    set_current_microbatch(model, microbatch)
    DSAIndexerLossLoggingHelper.clean_loss_in_tracker(preserve_groups=True)
    for parameter in model.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        else:
            parameter.grad.zero_()
    loss = model(**batch)
    values = loss.detach().float().cpu().clone()
    ((loss.float() * batch["loss_mask"]).sum() / batch["loss_mask"].sum()).backward()
    gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    indexer = DSAIndexerLossLoggingHelper.tracker.get("values")
    return values, gradients, None if indexer is None else indexer.detach().float().cpu().clone()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4, 8) or not torch.cuda.is_available(),
    reason="requires torchrun with 2, 4, or 8 CUDA ranks and real DSv4 kernels",
)
@pytest.mark.parametrize("case", _CASES)
def test_dsv4_chunk_dcp_same_weight_same_pack_replay(case, monkeypatch):
    """Capture real mHC/CSA/MoE and compare F/B after CP-bank and THD-pack switches."""
    if case not in ("window", "indexer", "full"):
        raise ValueError(f"Unknown DSV4_CHUNK_CASES entry: {case}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    assert world_size in (2, 4, 8), "launch this test with torchrun on 2, 4, or 8 GPUs"
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(context_parallel_size=1, dynamic_context_parallel=True)
    helper = None
    try:
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
        monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
        monkeypatch.setenv("NVTE_CUDA_GRAPH_SLOT_DEBUG", "1")
        _probe_dependencies(with_indexer=case != "window")
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        groups = ProcessGroupCollection.use_mpu_process_groups()
        getter = parallel_state.get_dynamic_data_context_parallel_groups
        destroy_num_microbatches_calculator()
        init_num_microbatches_calculator(
            rank=torch.distributed.get_rank(),
            global_batch_size=groups.dp.size(),
            micro_batch_size=1,
            data_parallel_size=groups.dp.size(),
        )
        DSAIndexerLossLoggingHelper.tracker.clear()
        model = _build_model(groups, case)
        weights = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
        sizes = tuple(2**exponent for exponent in range(groups.dp_cp.size().bit_length()))
        cases = [(size, lengths) for size in sizes for lengths in ((480,), (256, 224))]
        batches = [
            _batch(model.config, groups, size, lengths, index)
            for index, (size, lengths) in enumerate(cases)
        ]
        fingerprints = [_fingerprint(batch) for batch in batches]
        reference = []
        for index, (batch, (cp_size, local_lengths)) in enumerate(zip(batches, cases)):
            if torch.distributed.get_rank() == 0:
                print(
                    f"DSV4_CHUNK_EAGER case={case} index={index} cp_size={cp_size} "
                    f"local_lengths={local_lengths}",
                    flush=True,
                )
            reference.append(_step(model, batch, index))

        helper = TECudaGraphHelper(
            [model],
            model.config,
            seq_length=_LOCAL_TOKENS * groups.dp_cp.size(),
            micro_batch_size=1,
            pg_collection=groups,
            dynamic_cp_group_getter=getter,
        )
        assert helper.flattened_callables == [model.decoder, model.postprocess_block]
        helper.create_cudagraphs()
        assert helper.graphs_created()
        assert helper.num_microbatches == 1
        for block in (model.decoder, model.postprocess_block):
            assert set(block.cuda_graphs_by_dynamic_cp_size) == set(sizes)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.detach().cpu(), weights[name], rtol=0, atol=0)

        replays = []
        original_replay = torch.cuda.CUDAGraph.replay

        def counting_replay(graph, *args, **kwargs):
            replays.append(graph)
            return original_replay(graph, *args, **kwargs)

        monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counting_replay)
        order = (*range(len(batches)), *reversed(range(len(batches))), 0)
        max_relative_gradient_error = 0.0
        for microbatch, index in enumerate(order):
            before = len(replays)
            values, gradients, indexer = _step(model, batches[index], microbatch)
            assert len(replays) - before >= 4, "decoder and loss F/B must all replay"
            assert (
                _fingerprint(batches[index]) == fingerprints[index]
            ), "batch mutated during replay"
            expected_values, expected_gradients, expected_indexer = reference[index]
            torch.testing.assert_close(values, expected_values, rtol=1e-3, atol=1e-3)
            assert gradients.keys() == expected_gradients.keys()
            for name, expected in expected_gradients.items():
                error = (gradients[name] - expected).norm()
                max_relative_gradient_error = max(
                    max_relative_gradient_error, (error / expected.norm().clamp_min(1e-6)).item()
                )
                assert error <= 2e-2 * expected.norm() + 1e-6, (name, index, error.item())
            if expected_indexer is not None:
                torch.testing.assert_close(indexer, expected_indexer, rtol=2e-2, atol=1e-5)
        print(
            "DSV4_CHUNK_DCP_REPLAY_PASS "
            + json.dumps(
                {
                    "case": case,
                    "rank": torch.distributed.get_rank(),
                    "cp_sizes": sizes,
                    "physical_slots": 1,
                    "replay_microbatches": len(order),
                    "batch_sha256": fingerprints,
                    "max_relative_gradient_error": max_relative_gradient_error,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        torch.cuda.synchronize()
        if helper is not None:
            helper.delete_cuda_graphs()
        DSAIndexerLossLoggingHelper.tracker.clear()
        DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        destroy_moe_metrics_tracker()
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()
