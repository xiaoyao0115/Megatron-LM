# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU invariants for the native power-of-two Dynamic CP upper bound."""

import random
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest

from megatron.core import parallel_state
from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.core.datasets.data_schedule_utils import align_sample_id_groups
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.training.arguments import _add_network_size_args


def _scheduler(parent=4, maximum=2, vpp=None, max_num_seqs=4):
    return DefaultDynamicCPScheduler(
        max_seqlen_per_dp_cp_rank=512,
        cp_size=1,
        dp_size=parent,
        min_cp_size=1,
        max_cp_size=maximum,
        max_num_seqs=max_num_seqs,
        microbatch_group_size_per_vp_stage=vpp,
    )


def _assert_schedule(batches, lengths, parent, maximum, max_num_seqs=4, local_tokens=512):
    consumed = []
    for batch in batches:
        assert len(batch) == parent
        rank = 0
        while rank < parent:
            ids = batch[rank]
            assert ids
            members = [index for index, other in enumerate(batch) if other == ids]
            size = len(members)
            assert size <= maximum and size & (size - 1) == 0
            assert rank % size == 0 and members == list(range(rank, rank + size))
            assert len(ids) <= max_num_seqs
            assert sum(lengths[index] for index in ids) <= size * local_tokens
            consumed.extend(ids)
            rank += size
    assert sorted(consumed) == list(range(len(lengths)))


def test_dynamic_cp_max_config_default_and_valid_range():
    assert (
        ModelParallelConfig(dynamic_context_parallel=True).max_dynamic_context_parallel_size is None
    )
    config = ModelParallelConfig(
        dynamic_context_parallel=True,
        min_dynamic_context_parallel_size=2,
        max_dynamic_context_parallel_size=4,
    )
    assert config.sequence_packing_scheduler == "default_dynamic_cp"


@pytest.mark.parametrize("minimum,maximum", [(1, 0), (1, -2), (1, 3), (4, 2), (3, 4)])
def test_dynamic_cp_max_config_rejects_invalid_limits(minimum, maximum):
    with pytest.raises(ValueError, match="powers of two|must be >="):
        ModelParallelConfig(
            dynamic_context_parallel=True,
            min_dynamic_context_parallel_size=minimum,
            max_dynamic_context_parallel_size=maximum,
        )


@pytest.mark.parametrize("maximum,expected_sizes", [(None, {1, 2, 4}), (2, {1, 2}), (4, {1, 2, 4})])
def test_dynamic_cp_max_native_groups(monkeypatch, maximum, expected_sizes):
    calls = []

    def create_group(ranks, **kwargs):
        calls.append(tuple(ranks))
        return tuple(ranks)

    monkeypatch.setattr(parallel_state, "create_group", create_group)
    groups = parallel_state.create_dynamic_dp_cp_groups(
        3, list(range(8)), None, max_cp_size=maximum
    )
    assert set(groups) == expected_sizes
    assert {len(ranks) for ranks in calls} == expected_sizes
    for ranks in calls:
        assert ranks == tuple(range(ranks[0], ranks[0] + len(ranks)))
        assert ranks[0] % len(ranks) == 0


@pytest.mark.parametrize("maximum", [0, 3, 16])
def test_dynamic_cp_max_native_groups_reject_invalid_parent_limit(maximum):
    with pytest.raises(ValueError, match="between the minimum"):
        parallel_state.create_dynamic_dp_cp_groups(0, list(range(8)), None, max_cp_size=maximum)


@pytest.mark.parametrize(
    "parent,maximum,count,vpp", [(4, 2, 4, None), (4, 2, 8, 4), (32, 4, 256, None)]
)
def test_dynamic_cp_max_scheduler_preserves_tail(parent, maximum, count, vpp):
    lengths = [512] * count
    batches = _scheduler(parent, maximum, vpp).get_groups_and_subsamples(list(enumerate(lengths)))
    _assert_schedule(batches, lengths, parent, maximum)
    assert vpp is None or len(batches) % vpp == 0


@pytest.mark.parametrize("seed", range(20))
def test_dynamic_cp_max_scheduler_conserves_mixed_lengths(seed):
    rng = random.Random(seed)
    lengths = [rng.choice([1, 16, 128, 256, 512, 768, 1024]) for _ in range(32)]
    batches = _scheduler().get_groups_and_subsamples(list(enumerate(lengths)))
    _assert_schedule(batches, lengths, 4, 2)


def test_dynamic_cp_max_scheduler_rejects_sequence_that_needs_larger_group():
    with pytest.raises(ValueError, match="requires CP size 4, exceeding max_cp_size=2"):
        _scheduler().get_groups_and_subsamples([(0, 1025), (1, 512)])


def test_dynamic_cp_max_scheduler_rejects_underfilled_parent_without_duplication():
    with pytest.raises(ValueError, match="Cannot fill"):
        _scheduler().get_groups_and_subsamples([(0, 32)])


def test_dynamic_cp_max_default_preserves_full_parent_tail():
    batches = _scheduler(maximum=None).get_groups_and_subsamples([(0, 32)])
    assert batches == [[[0], [0], [0], [0]]]


def test_dynamic_cp_max_vpp_repartitions_groups_without_exceeding_cap():
    batches = [[[3 * index, 3 * index + 1]] * 4 + [[3 * index + 2]] * 4 for index in range(3)]
    aligned = align_sample_id_groups(batches, 4, max_cp_size=4)
    assert len(aligned) == 4
    _assert_schedule(aligned, [512] * 9, 8, 4)


def test_dynamic_cp_max_vpp_reports_current_packing_limitation():
    # Repacking the seven samples into two microbatches is possible. This aligner only
    # expands the existing three-microbatch schedule, and four need at least eight samples.
    lengths = [768, 512, 16, 16, 256, 1024, 256]
    with pytest.raises(
        ValueError, match="Cannot expand the current packing to the next VPP multiple"
    ):
        _scheduler(vpp=2).get_groups_and_subsamples(list(enumerate(lengths)))


@pytest.mark.parametrize("maximum,expected", [(None, (4, 2, 1)), (2, (2, 1)), (1, (1,))])
def test_dynamic_cp_max_capture_banks_match_native_groups(maximum, expected):
    helper = object.__new__(TECudaGraphHelper)
    helper.dp_cp_group = SimpleNamespace(size=lambda: 4)
    helper.config = SimpleNamespace(
        min_dynamic_context_parallel_size=1, max_dynamic_context_parallel_size=maximum
    )
    assert helper._get_dynamic_cp_capture_sizes() == expected


def test_dynamic_cp_max_logical_microbatch_bound_accounts_for_reserved_tail():
    kwargs = dict(
        global_batch_size=4,
        total_dcp_gpus=4,
        max_seqlen_per_dp_cp_rank=512,
        max_sequence_length=512,
    )
    bound = TECudaGraphHelper._get_default_dynamic_cp_thd_max_num_microbatches
    assert bound(**kwargs) == 1
    assert bound(**kwargs, max_cp_size=2) == 2
    assert bound(**kwargs, max_cp_size=2, microbatch_group_size_per_vp_stage=4) == 4


def test_dynamic_cp_max_generated_cli_accepts_optional_integer():
    parser = ArgumentParser()
    _add_network_size_args(parser)
    assert parser.parse_args([]).max_dynamic_context_parallel_size is None
    args = parser.parse_args(["--max-dynamic-context-parallel-size", "4"])
    assert args.max_dynamic_context_parallel_size == 4
