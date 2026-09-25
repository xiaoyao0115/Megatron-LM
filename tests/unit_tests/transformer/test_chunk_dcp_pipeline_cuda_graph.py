# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise DCP chunk graph slot reuse under the real interleaved PP schedule.

    torchrun --nproc-per-node 4 -m pytest -xvs \
        tests/unit_tests/transformer/test_chunk_dcp_pipeline_cuda_graph.py
"""

import os
from collections import defaultdict

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import (
    _validate_cuda_graph_microbatch_count,
    get_batch_on_this_rank_for_sequence_packing,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _pipeline_model(groups):
    config = TransformerConfig(
        num_layers=4,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=512,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        attention_backend=AttnBackend.fused,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        add_bias_linear=False,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        microbatch_group_size_per_vp_stage=2,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
        variable_seq_lengths=True,
        dynamic_context_parallel=True,
        max_seqlen_per_dp_cp_rank=64,
        thd_max_packed_sequences=4,
        sequence_packing_scheduler="default_dynamic_cp",
        pad_packed_seq_alignment="max",
        cuda_graph_impl="transformer_engine",
        cuda_graph_granularity="chunk",
        cuda_graph_modules=[],
        cuda_graph_dynamic_microbatches=True,
    )
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    model = []
    for vp_stage in range(2):
        chunk = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=128,
            max_sequence_length=64 * groups.dp_cp.size(),
            pre_process=groups.pp.rank() == 0 and vp_stage == 0,
            post_process=groups.pp.rank() == 1 and vp_stage == 1,
            position_embedding_type="rope",
            pg_collection=groups,
            vp_stage=vp_stage,
        ).cuda()
        chunk.zero_grad_buffer = lambda chunk=chunk: chunk.zero_grad(set_to_none=True)
        model.append(chunk)
    return model


def _pipeline_batches(model, groups, count):
    per_chunk = []
    for chunk in model:
        batches = []
        for index in range(count):
            # Period three changes CP across a four-slot ring; period five changes the pack.
            size = (1, 2, 1)[index % 3]
            local_lengths = ((48,), (16, 32), (24, 24), (8, 40), (32, 16))[index % 5]
            lengths = [length * size for length in local_lengths]
            generator = torch.Generator(device="cpu").manual_seed(1000 + index)
            tokens = torch.randint(0, 128, (sum(lengths) + 1,), generator=generator).cuda()
            cu = torch.tensor([0, *lengths], dtype=torch.int32, device="cuda").cumsum(
                0, dtype=torch.int32
            )
            raw = dict(
                tokens=tokens[:-1].contiguous(),
                labels=tokens[1:].contiguous(),
                loss_mask=torch.ones(sum(lengths), dtype=torch.float32, device="cuda"),
                position_ids=torch.cat([torch.arange(length) for length in lengths]).cuda(),
                cu_seqlens=cu,
                cu_seqlens_padded=cu.clone(),
                max_seqlen=max(lengths),
                local_cp_size=size,
            )
            values = get_batch_on_this_rank_for_sequence_packing(
                iter([raw]),
                dynamic_cp=True,
                vpp_size=2,
                vp_stage=chunk.vp_stage,
                pg_collection=groups,
                config=chunk.config,
            )
            batches.append(
                dict(
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
                        values,
                    )
                )
            )
        per_chunk.append(batches)
    return per_chunk


def _run_schedule(model, groups, batches, count, replay_slots):
    for chunk in model:
        _validate_cuda_graph_microbatch_count(chunk.config, count)
        for parameter in chunk.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            else:
                parameter.grad.zero_()

    def forward_step(data_iterator, chunk):
        batch = next(data_iterator)
        block = chunk.decoder
        if block.cuda_graphs_by_dynamic_cp_size:
            size = batch["packed_seq_params"].local_cp_size
            slots = len(block.cuda_graphs_by_dynamic_cp_size[size])
            replay_slots[(chunk.vp_stage, block.current_microbatch % slots)].append(size)
        output = chunk(**batch)

        def loss_func(per_token_loss):
            mask = batch["loss_mask"]
            loss = (per_token_loss.float() * mask).sum() / mask.sum()
            return loss, {"loss": loss.detach().clone()}

        return output, loss_func

    losses = get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=[iter(chunk_batches[:count]) for chunk_batches in batches],
        model=model,
        num_microbatches=count,
        seq_length=64,
        micro_batch_size=1,
        forward_only=False,
        p2p_communicator=P2PCommunicator(pp_group=groups.pp, config=model[0].config),
        pg_collection=groups,
    )
    torch.cuda.synchronize()
    gradients = {
        (index, name): parameter.grad.detach().float().clone()
        for index, chunk in enumerate(model)
        for name, parameter in chunk.named_parameters()
        if parameter.grad is not None
    }
    return [item["loss"].float().clone() for item in losses], gradients


@pytest.mark.internal
@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 4 or torch.cuda.device_count() < 4,
    reason="requires torchrun with at least 4 GPUs for PP2 and DCP2",
)
def test_chunk_dcp_pp2_vpp2_async_replay_matches_eager(monkeypatch):
    """Compare local losses/grads while graph-backed outputs are sent asynchronously.

    No optimizer update or DP reduction is needed: both executions use the same parameters
    and compare every rank's accumulated gradients after the production 1F1B schedule.
    """
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        context_parallel_size=1,
        dynamic_context_parallel=True,
    )
    helper = None
    try:
        monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
        monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
        monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        groups = ProcessGroupCollection.use_mpu_process_groups()
        destroy_num_microbatches_calculator()
        init_num_microbatches_calculator(
            rank=torch.distributed.get_rank(),
            global_batch_size=12 * groups.dp.size(),
            micro_batch_size=1,
            data_parallel_size=groups.dp.size(),
        )
        model = _pipeline_model(groups)
        batches = _pipeline_batches(model, groups, 12)
        references = {
            count: _run_schedule(model, groups, batches, count, defaultdict(list))
            for count in (8, 12)
        }

        helper = TECudaGraphHelper(
            model,
            model[0].config,
            seq_length=64 * groups.dp_cp.size(),
            micro_batch_size=1,
            pg_collection=groups,
            dynamic_cp_group_getter=parallel_state.get_dynamic_data_context_parallel_groups,
        )
        helper.create_cudagraphs()
        assert helper.graphs_created()
        assert helper.num_microbatches < 8, "the numerical test must actually wrap physical slots"
        for chunk in model:
            assert chunk.config._cuda_graph_num_microbatches >= 12
            assert {8, 12} <= chunk.config._cuda_graph_allowed_microbatches

        graph_launches = []
        original_replay = torch.cuda.CUDAGraph.replay

        def counting_replay(graph, *args, **kwargs):
            graph_launches.append(graph)
            return original_replay(graph, *args, **kwargs)

        monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", counting_replay)
        for count in (8, 12, 8):
            replay_slots = defaultdict(list)
            before = len(graph_launches)
            losses, gradients = _run_schedule(model, groups, batches, count, replay_slots)
            callables = sum(1 + (chunk.postprocess_block is not None) for chunk in model)
            assert len(graph_launches) - before >= 2 * count * callables
            assert replay_slots
            for vp_stage in range(2):
                assert any(
                    len(set(sizes)) == 2
                    for (vp, _slot), sizes in replay_slots.items()
                    if vp == vp_stage
                ), "each chunk must reuse a physical slot with a different CP communicator"
            if count == 12:
                assert any(sizes == [1, 2, 1] for sizes in replay_slots.values())
            expected_losses, expected_gradients = references[count]
            assert len(losses) == len(expected_losses)
            for actual, expected in zip(losses, expected_losses):
                torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
            assert gradients.keys() == expected_gradients.keys()
            for name, expected in expected_gradients.items():
                error = (gradients[name] - expected).norm()
                assert error <= 2e-2 * expected.norm() + 1e-6, (name, count, error.item())
        print(
            f"CHUNK_DCP_PP_REPLAY_PASS pp=2 vpp=2 cp=1,2 "
            f"slots={helper.num_microbatches} counts=8,12,8 overlap_p2p=True"
        )
    finally:
        torch.cuda.synchronize()
        if helper is not None:
            helper.delete_cuda_graphs()
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()
