# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.

from collections import deque
from functools import lru_cache
from math import ceil, log2
from typing import Callable, Dict, List, Optional, Tuple

import torch

from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices
from megatron.core.rerun_state_machine import RerunDataIterator


def get_cp_slice_for_thd(batch, cp_group):
    """Partition sequence data for context parallelism in THD format.

    Uses TE's THD partitioned indices to split the packed sequence across CP ranks.
    Only keys present in the batch are sliced.

    Args:
        batch: Dict with packed sequence data.
        cp_group: Context parallel process group.
    """
    cp_size = cp_group.size()
    if cp_size <= 1:
        return
    cp_rank = cp_group.rank()
    # Use whichever data field is available to determine total_tokens
    for _key in ['tokens', 'labels', 'loss_mask', 'position_ids']:
        if _key in batch and batch[_key] is not None:
            total_tokens = batch[_key].size(0)
            break
    else:
        raise ValueError("Cannot determine total_tokens: no data field found in batch")
    # Transformer Engine has a bug of cu_seqlens, we must treat cu_seqlens_padded as
    # cu_seqlens to get the correct result.
    # TODO: Revert this workaround once TE fixes the issue.
    cu_seqlens = batch["cu_seqlens_padded"]
    index = get_thd_partitioned_indices(cu_seqlens, total_tokens, cp_size, cp_rank)
    for key in ['tokens', 'position_ids', 'labels', 'loss_mask']:
        if key in batch:
            batch[key] = batch[key].index_select(0, index)


def _unpack_batch(batch: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """
    Unpacks the packed samples into a list of sub-samples.
    Since each sub-sample may be routed to different DPxCP ranks,
    we unpack the sample here to avoid unnecessarily transferring
    the entire packed sample.
    """
    batch_unpacked = []
    dev = batch[0]["cu_seqlens"].device
    original_seq_lens = []
    padded_seq_lens = []
    # Determine which data fields exist in the batch
    data_keys = [k for k in ["tokens", "labels", "loss_mask", "position_ids"] if k in batch[0]]
    for sample in batch:
        for key in sample.keys():
            if len(sample[key].shape) == 2:
                # squeeze the redundant batch dimension added by
                # default collate_fn in pytorch dataloader
                # we need a custom collate_fn for THD to avoid this
                # current THD does not support micro_batch_size > 1 due to sft_dataset.py and
                # data_loader in data_samples.py
                sample[key] = sample[key].squeeze(0)
        for sub_sample in range(sample["cu_seqlens"].shape[0] - 1):
            sub_sample_dict = {}
            start_idx = sample["cu_seqlens"][sub_sample]
            end_idx = sample["cu_seqlens"][sub_sample + 1]
            if end_idx - start_idx == 0:
                continue
            for key in data_keys:
                sub_sample_dict[key] = sample[key][start_idx:end_idx]
            # Since sft_dataset.py does not provide cu_seqlens_original,
            # we assume original_seq_len equals padded_seq_len here.
            # Ideally the dataset should define the pre-padding seq_len.
            seq_len = (end_idx - start_idx).item()
            original_seq_lens.append(seq_len)
            padded_seq_lens.append(seq_len)
            batch_unpacked.append(sub_sample_dict)

    # Single H2D transfer for all seq lens
    original_seq_lens_cuda = torch.tensor(original_seq_lens, device=dev)
    padded_seq_lens_cuda = torch.tensor(padded_seq_lens, device=dev)
    for i, sub_sample_dict in enumerate(batch_unpacked):
        sub_sample_dict["original_seq_len"] = original_seq_lens_cuda[i : i + 1]
        sub_sample_dict["padded_seq_len"] = padded_seq_lens_cuda[i : i + 1]

    return batch_unpacked


def _get_global_seqlens_and_ids(subsample_seqlens: torch.Tensor, dp_group):
    """
    Gathers the sequence lengths of all subsamples from all DP ranks and calculates global IDs.
    """
    # Collect the number of subsamples from all ranks
    num_local_subsamples = subsample_seqlens.shape[0]
    local_len = torch.tensor([num_local_subsamples], dtype=torch.int32).cuda()
    dp_subsample_count = [torch.zeros_like(local_len) for _ in range(dp_group.size())]
    torch.distributed.all_gather(dp_subsample_count, local_len, group=dp_group)

    # Find the max number of subsamples across all ranks and pad subsample_seqlens to max length
    dp_subsample_counts = torch.stack(dp_subsample_count, dim=0).cpu().view(-1)
    max_sub_samples = int(dp_subsample_counts.max().item())

    if num_local_subsamples < max_sub_samples:
        subsample_seqlens_padded = torch.cat(
            [
                subsample_seqlens,
                torch.zeros(max_sub_samples - num_local_subsamples, dtype=torch.int32).cuda(),
            ],
            dim=0,
        )
    else:
        subsample_seqlens_padded = subsample_seqlens

    # Gather the subsample_seqlens from all ranks
    seqlens_gathered = [torch.empty_like(subsample_seqlens_padded) for _ in range(dp_group.size())]
    torch.distributed.all_gather(seqlens_gathered, subsample_seqlens_padded, group=dp_group)

    # Trim each seqlens_gathered to the length of the correct sample
    for dp_rank, seqlen in enumerate(seqlens_gathered):
        seqlens_gathered[dp_rank] = seqlen[: dp_subsample_counts[dp_rank]]

    seqlens_gathered = torch.cat(seqlens_gathered, dim=0)
    seqlens_gathered = seqlens_gathered.cpu().tolist()

    # Calculate the offsets to assign unique global ID to each subsample.
    csum = torch.cumsum(dp_subsample_counts, dim=0, dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), csum], dim=0)

    # Calculate global ID for each subsample
    dp_rank = dp_group.rank()
    global_ids = torch.arange(len(seqlens_gathered), dtype=torch.int32).cuda()

    # Create a list of (global_id, seqlen) tuples for scheduling
    global_id_seqlens = [(i, seqlens_gathered[i]) for i in range(len(global_ids))]

    # Get the global IDs locally present on this rank
    start_idx = offsets[dp_rank]
    end_idx = offsets[dp_rank + 1]

    global_ids_this_rank = global_ids[start_idx:end_idx]

    return global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered


def _pack_sequences(
    samples: List,
    padded_lengths: torch.Tensor,
    original_lengths: torch.Tensor,
    local_cp_size: Optional[torch.Tensor],
    dev: torch.device,
) -> Dict[str, torch.Tensor]:
    """Pack multiple samples into a single packed sample."""

    def _pack_tensors(tensors):
        return torch.cat([t.reshape(-1) for t in tensors], dim=0)

    new_sample = {}
    for key in ['tokens', 'labels', 'loss_mask', 'position_ids']:
        if key in samples[0]:
            new_sample[key] = _pack_tensors([sample[key] for sample in samples])

    padded_lengths = padded_lengths.to(device=dev, dtype=torch.int32, non_blocking=True).reshape(-1)
    cu_seqlens_padded = torch.empty(padded_lengths.numel() + 1, device=dev, dtype=torch.int32)
    cu_seqlens_padded[0] = 0
    cu_seqlens_padded[1:] = torch.cumsum(padded_lengths, dim=0)
    max_seqlen = torch.max(padded_lengths).to(dtype=torch.int32)

    new_sample["cu_seqlens_padded"] = cu_seqlens_padded
    new_sample["max_seqlen"] = max_seqlen

    original_lengths = original_lengths.to(
        device=dev, dtype=torch.int32, non_blocking=True
    ).reshape(-1)
    cu_seqlens = torch.empty(original_lengths.numel() + 1, device=dev, dtype=torch.int32)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = torch.cumsum(original_lengths, dim=0).reshape(-1)
    new_sample["cu_seqlens"] = cu_seqlens

    if local_cp_size is not None:
        new_sample["local_cp_size"] = local_cp_size

    return new_sample


def broadcast_tensor(item, src_rank, group) -> None:
    """Broadcast a tensor from src_rank to all ranks in the group."""
    if item is not None:
        torch.distributed.broadcast(item, src_rank, group=group)


def broadcast_scalars(values: List, group, dev, dtype=torch.float32) -> List:
    """
    Broadcast scalar values from rank 0 to all ranks in the group.

    Args:
        values: List of scalar values to broadcast (only used on rank 0).
        group: The process group to broadcast within.
        dev: The device to use for the tensor.
        dtype: The data type for the tensor.

    Returns:
        List of broadcasted values.
    """
    if group.size() <= 1:
        return values

    src_rank = torch.distributed.get_process_group_ranks(group)[0]
    num_values = len(values)

    if group.rank() == 0:
        info_to_broadcast = torch.tensor(values, dtype=dtype, device=dev)
    else:
        info_to_broadcast = torch.zeros(num_values, dtype=dtype, device=dev)

    broadcast_tensor(info_to_broadcast, src_rank, group)

    if group.rank() != 0:
        values = info_to_broadcast.cpu().tolist()

    return values


def create_data_iterator(
    new_samples, tp_group, config, vpp_needs_data=None, is_dynamic_cp: bool = False
):
    """Handle virtual pipeline parallelism.

    For VPP, each PP rank needs a list of data iterators (one per VPP stage).
    VPP stages that need full data (first/last pipeline stage, or MTP) get
    full samples; others get metadata only (cu_seqlens, cu_seqlens_padded,
    max_seqlen).

    Args:
        new_samples: The packed samples after scheduling.
        tp_group: Tensor parallel process group.
        config: Model parallel config.
        vpp_needs_data: A list of booleans (one per VPP stage) indicating which
            VPP stages need full samples (data fields). None if VPP is disabled.
    """
    if (
        config.virtual_pipeline_model_parallel_size is not None
        and config.virtual_pipeline_model_parallel_size > 1
    ):
        vpp_size = config.virtual_pipeline_model_parallel_size
        if tp_group.rank() == 0:
            metadata_keys = ["max_seqlen", "cu_seqlens", "cu_seqlens_padded"]
            if is_dynamic_cp:
                metadata_keys.append("local_cp_size")
            new_data_iterator = []
            for i in range(vpp_size):
                if vpp_needs_data is not None and vpp_needs_data[i]:
                    new_data_iterator.append(RerunDataIterator(iter(new_samples)))
                else:
                    # Create independent metadata dicts to avoid shared-reference mutation
                    metadata = [
                        {k: sample[k] for k in metadata_keys if k in sample}
                        for sample in new_samples
                    ]
                    new_data_iterator.append(RerunDataIterator(iter(metadata)))
        else:
            new_data_iterator = [None for _ in range(vpp_size)]
    else:
        new_data_iterator = RerunDataIterator(iter(new_samples)) if tp_group.rank() == 0 else None

    return new_data_iterator


def reroute_samples_to_dcp_ranks(
    batch,
    global_ids_this_rank,
    global_id_seqlens,
    sample_id_groups,
    offsets,
    dp_group,
    tp_group,
    dp_cp_group,
    total_dcp_gpus,
):
    """
    Reroutes the sub-samples to the correct rank after scheduling.

    For each key in the batch dict, we perform an all-to-all communication
    to transfer the data to the correct ranks.
    """

    def _gid_to_src_rank(gid: int) -> int:
        dp_src_rank = torch.bucketize(gid, offsets[1:] - 1)
        dcp_rank = (
            torch.distributed.get_process_group_ranks(dp_group)[dp_src_rank] // tp_group.size()
        ) % dp_cp_group.size()
        return dcp_rank

    gid2local_id = {int(gid): i for i, gid in enumerate(global_ids_this_rank)}
    dcp_rank = dp_cp_group.rank()
    dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
    dp_ranks = [(r // tp_group.size()) % dp_cp_group.size() for r in dp_ranks]

    data_keys = batch[0].keys()

    # Create the send plan
    combined_sample_id_groups: List[List[int]] = [[] for _ in range(total_dcp_gpus)]
    for d in range(total_dcp_gpus):
        for sample_id_group in sample_id_groups:
            combined_sample_id_groups[d].extend(sample_id_group[d])
    for dest_rank in range(total_dcp_gpus):
        combined_sample_id_groups[dest_rank].sort()

    send_ids_sorted = [
        gid for d in dp_ranks for gid in combined_sample_id_groups[d] if gid in global_ids_this_rank
    ]

    send_num_split = [0] * total_dcp_gpus
    send_lens_split = [0] * total_dcp_gpus
    for dest_rank in range(total_dcp_gpus):
        if dest_rank in dp_ranks:
            send_seq_lens = [
                global_id_seqlens[gid][1]
                for gid in combined_sample_id_groups[dest_rank]
                if gid in global_ids_this_rank
            ]
            send_num_split[dest_rank] = len(send_seq_lens)
            send_lens_split[dest_rank] = sum(send_seq_lens)
        else:
            send_lens_split[dest_rank] = 0

    # Create the recv plan
    recv_sample_id_groups = [[] for _ in range(total_dcp_gpus)]
    for gid in combined_sample_id_groups[dcp_rank]:
        src_rank = _gid_to_src_rank(gid)
        recv_sample_id_groups[src_rank].append(gid)

    recv_lens_split = [0] * total_dcp_gpus
    for src_rank in range(total_dcp_gpus):
        recv_lens_split[src_rank] = sum(
            [global_id_seqlens[gid][1] for gid in recv_sample_id_groups[src_rank]]
        )

    recv_ids_sorted = [gid for d in range(total_dcp_gpus) for gid in recv_sample_id_groups[d]]
    recv_counts = [len(recv_sample_id_groups[d]) for d in range(total_dcp_gpus)]

    recv_samples = [{k: None for k in data_keys} for _ in range(sum(recv_counts))]

    def _pack_sample_by_key(key: str) -> torch.Tensor:
        flattened_tensors = []
        for gid in send_ids_sorted:
            t = batch[gid2local_id[gid]][key].to(torch.cuda.current_device(), non_blocking=True)
            flattened_tensors.append(t.reshape(-1))
        return (
            torch.cat(flattened_tensors, dim=0)
            if flattened_tensors
            else torch.empty(0, device=torch.cuda.current_device(), dtype=batch[0][key].dtype)
        )

    def _unpack_sample_by_key(key: str, recv_tensor: torch.Tensor):
        cursor = 0
        for i, gid in enumerate(recv_ids_sorted):
            sample_len = (
                1 if key in ["original_seq_len", "padded_seq_len"] else global_id_seqlens[gid][1]
            )
            recv_samples[i][key] = recv_tensor[cursor : cursor + sample_len]
            cursor += sample_len

    for key in data_keys:
        output_split_sizes, input_split_sizes = (
            (recv_counts, send_num_split)
            if key in ["original_seq_len", "padded_seq_len"]
            else (recv_lens_split, send_lens_split)
        )
        send_tensor = _pack_sample_by_key(key)
        recv_tensor_size = sum(output_split_sizes)
        recv_tensor = torch.empty(
            recv_tensor_size, device=torch.cuda.current_device(), dtype=send_tensor.dtype
        )
        torch.distributed.all_to_all_single(
            output=recv_tensor,
            input=send_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=dp_cp_group,
        )
        _unpack_sample_by_key(key, recv_tensor)

    recv_sample_with_id = {recv_id: recv_samples[i] for i, recv_id in enumerate(recv_ids_sorted)}
    return recv_sample_with_id


def build_packed_microbatches(
    samples_this_rank_with_id: Dict[int, Dict[str, torch.Tensor]],
    sample_id_groups: List[List[List[int]]],
    dcp_rank: int,
    dev: torch.device,
    is_dynamic_cp: bool = False,
) -> List[Dict[str, torch.Tensor]]:
    """Build packed samples for each microbatch.

    Args:
        samples_this_rank_with_id: Mapping from global sample ID to sample dict,
            as returned by reroute_samples_to_dcp_ranks.
        sample_id_groups: Per-microbatch, per-rank lists of global sample IDs.
        dcp_rank: This rank's index within the DP×CP group.
        dev: Target device.
        is_dynamic_cp: Whether dynamic context parallel is enabled.
    """
    num_micro_batches = len(sample_id_groups)
    seg_starts: List[int] = [0]
    original_lens_tensors = []
    padded_lens_tensors = []

    grouped_samples = [
        [
            samples_this_rank_with_id[sub_sample_id]
            for sub_sample_id in sample_id_groups[i][dcp_rank]
        ]
        for i in range(num_micro_batches)
    ]

    local_cp_sizes_gpu = None
    if is_dynamic_cp:
        local_cp_sizes_cpu: List[int] = []
        for i in range(num_micro_batches):
            sample_ids_this_group = sample_id_groups[i][dcp_rank]
            local_cp_sizes_cpu.append(
                len(
                    [
                        1
                        for sample_ids in sample_id_groups[i]
                        if sample_ids_this_group[0] in sample_ids
                    ]
                )
            )
        local_cp_sizes_gpu = torch.tensor(local_cp_sizes_cpu, dtype=torch.int32, device=dev)

    for i in range(num_micro_batches):
        samples = grouped_samples[i]
        seg_starts.append(seg_starts[-1] + len(samples))
        original_lens_tensors.extend([s["original_seq_len"].reshape(-1) for s in samples])
        padded_lens_tensors.extend([s["padded_seq_len"].reshape(-1) for s in samples])

    padded_lens_all_gpu = torch.cat(padded_lens_tensors, dim=0).to(dtype=torch.int32)
    original_lens_all_gpu = torch.cat(original_lens_tensors, dim=0).to(dtype=torch.int32)

    new_samples: List[Dict[str, torch.Tensor]] = []
    for i in range(num_micro_batches):
        samples = grouped_samples[i]
        lens_padded = padded_lens_all_gpu[seg_starts[i] : seg_starts[i + 1]]
        lens_original = original_lens_all_gpu[seg_starts[i] : seg_starts[i + 1]]
        local_cp_size = local_cp_sizes_gpu[i] if is_dynamic_cp else None
        new_sample = _pack_sequences(samples, lens_padded, lens_original, local_cp_size, dev)
        new_samples.append(new_sample)

    return new_samples


def get_batch_and_global_seqlens(data_iterator, num_microbatches, dp_group):
    """
    Get the batch and global sequence lengths.
    Each DP rank loads the same number of sequences, so we need to gather the sequence
    lengths from all ranks then we can schedule the sequences into groups.
    Args:
        data_iterator: The data iterator.
        num_microbatches: The number of microbatches.
        dp_group: The data parallel group.

    Returns:
        batch: The batch.
        global_id_seqlens: The global sequence lengths.
        global_ids_this_rank: The global IDs locally present on this rank.
    """

    batch_list = [next(data_iterator) for _ in range(num_microbatches)]

    batch = []
    for item in batch_list:
        if isinstance(item, dict):
            batch.append(item)
        elif isinstance(item, list):
            batch.extend(item)
        else:
            raise ValueError(f"Invalid item type: {type(item)}")

    # in sft_dataset.py, sequences are already packed before rescheduling,
    # so we need to unpack them here and repack after rescheduling.
    # This is only to adapt to the current megatron-lm sft_dataset.
    # If you implement your own dataset, just have __getitem__ return List[Dict]
    # and this step can be skipped.
    batch = _unpack_batch(batch)

    subsample_seqlens = torch.cat([sample["padded_seq_len"] for sample in batch]).to(
        dtype=torch.int32, device=torch.cuda.current_device()
    )

    global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered = (
        _get_global_seqlens_and_ids(subsample_seqlens, dp_group)
    )

    return batch, global_id_seqlens, global_ids_this_rank, offsets, seqlens_gathered


# =============================================================================
# Dynamic CP scheduling algorithms (used by DefaultDynamicCPScheduler)
# =============================================================================


def next_hdp_group(
    sample_seqlens: List[Tuple[int, int]],
    compute_estimator: Callable[[int], float],
    total_gpus: int,
    gpus_needed_fn: Callable[[int], int],
    make_buckets_equal_fn: Callable,
    max_seq_len_per_rank: float,
    get_total_workload_fn: Callable,
    delta: float = 0.05,
    strategy: str = "dp",
    eps_bucket: float = 0.10,
) -> Tuple[List[List[int]], List[Tuple[int, int]], List[float], List[List[int]]]:
    """Form one balanced micro-batch group across DPxCP ranks.

    This is a standalone version of the scheduling algorithm extracted from
    DefaultDynamicCPScheduler so it can live in a utils module.

    Extra args compared to the method version:
        gpus_needed_fn: callable(seq_len) -> int
        make_buckets_equal_fn: callable(sample_seqlens, compute_estimator) -> list[deque]
        max_seq_len_per_rank: max tokens per rank for packing
        get_total_workload_fn: callable(seq_len, cp_size) -> float
    """
    if not sample_seqlens:
        return (
            [[] for _ in range(total_gpus)],
            [],
            [0.0 for _ in range(total_gpus)],
            [[] for _ in range(total_gpus)],
        )

    buckets = make_buckets_equal_fn(sample_seqlens, compute_estimator)

    micro_batches = [[] for _ in range(total_gpus)]
    exec_times = [0.0 for _ in range(total_gpus)]
    sample_ids_per_gpu = [[] for _ in range(total_gpus)]
    packing_sequence_len = {}

    gpu_group_id = [None] * total_gpus
    group_members = {}
    group_size = {}
    next_gid = 0

    pp_cursor = 0
    prev_needed = None
    check_balance = False

    while buckets:
        sample_seq_tuple = bucket_idx = None
        needed = None

        scan_order = (
            range(len(buckets))
            if strategy == "dp"
            else [(pp_cursor + i) % len(buckets) for i in range(len(buckets))]
        )

        for idx in scan_order:
            if not buckets[idx]:
                continue
            cand_tuple = buckets[idx][0]
            cand_seq_len = cand_tuple[1]
            needed = gpus_needed_fn(cand_seq_len)

            candidate_gids = [gid for gid, sz in group_size.items() if sz == needed]
            free_ranks = [r for r, gid in enumerate(gpu_group_id) if gid is None]
            if candidate_gids or len(free_ranks) >= needed:
                sample_seq_tuple, bucket_idx = cand_tuple, idx
                break

        if sample_seq_tuple is None:
            break

        if strategy == "pp":
            pp_cursor = (bucket_idx + 1) % len(buckets)

        sample_id, seq_len = sample_seq_tuple
        needed = gpus_needed_fn(seq_len)
        if prev_needed is None:
            prev_needed = needed

        candidate_gids = [
            gid
            for gid, sz in group_size.items()
            if sz == needed and packing_sequence_len[gid] + seq_len / needed <= max_seq_len_per_rank
        ]
        if candidate_gids:
            best_gid, best_load = min(
                ((gid, max(exec_times[r] for r in group_members[gid])) for gid in candidate_gids),
                key=lambda t: t[1],
            )
        else:
            best_gid, best_load = None, float("inf")

        free_ranks = [r for r, gid in enumerate(gpu_group_id) if gid is None]
        if len(free_ranks) >= needed:
            free_sorted = sorted(free_ranks, key=lambda r: exec_times[r])
            new_members = free_sorted[:needed]
            new_load = exec_times[new_members[-1]]

            if new_load < best_load:
                best_gid = None
                chosen_members = new_members
            else:
                chosen_members = group_members[best_gid]
        else:
            if best_gid is None:
                break
            chosen_members = group_members[best_gid]

        if best_gid is None:
            best_gid = next_gid
            next_gid += 1
            group_members[best_gid] = chosen_members
            group_size[best_gid] = needed
            for r in chosen_members:
                gpu_group_id[r] = best_gid

        per_gpu_cost = compute_estimator(seq_len)

        packing_sequence_len[best_gid] = packing_sequence_len.get(best_gid, 0) + seq_len / needed
        for r in chosen_members:
            micro_batches[r].append(seq_len)
            exec_times[r] += per_gpu_cost
            sample_ids_per_gpu[r].append(sample_id)

        buckets[bucket_idx].popleft()

        while buckets and not buckets[0]:
            buckets.pop(0)
            pp_cursor %= max(1, len(buckets))

        if needed < prev_needed:
            check_balance = True

        if (
            check_balance
            and buckets
            and max(exec_times) - min(exec_times) <= delta * max(exec_times)
        ):
            break

    leftovers = []
    for b in buckets:
        for sample_seq_tuple in b:
            leftovers.append(sample_seq_tuple)

    def trim_overload():
        while True:
            cur_max = max(exec_times)
            cur_min = min(exec_times)
            cur_slack = cur_max - cur_min
            if cur_slack <= delta * cur_max:
                break
            if cur_min == 0:
                break

            max_r = exec_times.index(cur_max)
            gid = gpu_group_id[max_r]
            members = group_members[gid]

            if not micro_batches[max_r] or len(micro_batches[max_r]) <= 1:
                break

            seq = micro_batches[max_r][-1]
            per_gpu_cost = compute_estimator(seq)

            proj_times = exec_times[:]
            for r in members:
                proj_times[r] -= per_gpu_cost

            proj_slack = max(proj_times) - min(proj_times)

            if proj_slack < cur_slack:
                sample_id_to_remove = sample_ids_per_gpu[max_r][-1]
                for r in members:
                    micro_batches[r].pop()
                    exec_times[r] -= per_gpu_cost
                    sample_ids_per_gpu[r].pop()
                leftovers.append((sample_id_to_remove, seq))
            else:
                break

    # TODO(tailaim): uncomment this to support different ranks have different num_microbatches
    # trim_overload()

    total_work_before = sum(len(mb) for mb in micro_batches)

    def fill_empty_gpus(micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size):
        empty_gpus = [i for i in range(total_gpus) if not micro_batches[i]]
        if not empty_gpus:
            return (micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size)

        existing_group_sizes = set(group_size.values())
        assert (
            existing_group_sizes
        ), "There should be at least one group existing, cannot redistribute, "
        "try to increase 'max-seqlen-per-dp-cp-rank'."

        min_group_size = min(existing_group_sizes)
        next_power = min(min_group_size * 2, total_gpus)

        for gid, size in group_size.items():
            if size == min_group_size:
                members = group_members[gid]
                needed_count = next_power - min_group_size
                group_start_gpu = members[0]
                group_end_gpu = members[-1]
                empty_gpu = [idx for idx, work in enumerate(micro_batches) if not work][0]
                assert not all(
                    work for work in micro_batches[empty_gpu : empty_gpu + needed_count]
                ), "Empty GPUs were detected but not enough to expand."
                work_to_push = micro_batches[group_end_gpu + 1 : empty_gpu]
                exec_times_to_push = exec_times[group_end_gpu + 1 : empty_gpu]
                sample_ids_to_push = sample_ids_per_gpu[group_end_gpu + 1 : empty_gpu]

                new_micro_batches = [[]] * len(micro_batches)
                new_exec_times = [0.0] * len(exec_times)
                new_sample_ids_per_gpu = [[]] * len(sample_ids_per_gpu)

                for i in range(group_start_gpu):
                    new_micro_batches[i] = micro_batches[i]
                    new_exec_times[i] = exec_times[i]
                    new_sample_ids_per_gpu[i] = sample_ids_per_gpu[i]

                for i in range(group_start_gpu, group_end_gpu + needed_count + 1):
                    new_micro_batches[i] = micro_batches[group_end_gpu]
                    new_exec_times[i] = get_total_workload_fn(
                        micro_batches[group_end_gpu][0], next_power
                    )
                    new_sample_ids_per_gpu[i] = sample_ids_per_gpu[group_end_gpu]

                for i, work in enumerate(work_to_push):
                    new_micro_batches[group_end_gpu + needed_count + 1 + i] = work
                    new_exec_times[group_end_gpu + needed_count + 1 + i] = exec_times_to_push[i]
                    new_sample_ids_per_gpu[group_end_gpu + needed_count + 1 + i] = (
                        sample_ids_to_push[i]
                    )

                group_size[gid] = next_power
                group_members[gid] = list(range(members[0], members[-1] + needed_count + 1))
                for pushed_gid in group_size.keys():
                    if pushed_gid > gid:
                        group_members[pushed_gid] = [
                            x + needed_count for x in group_members[pushed_gid]
                        ]

                return (
                    new_micro_batches,
                    new_exec_times,
                    new_sample_ids_per_gpu,
                    group_members,
                    group_size,
                )

    empty_gpus = any([not micro_batches[i] for i in range(total_gpus)])
    while empty_gpus:
        micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size = fill_empty_gpus(
            micro_batches, exec_times, sample_ids_per_gpu, group_members, group_size
        )
        empty_gpus = any([not micro_batches[i] for i in range(total_gpus)])

    total_work_after = sum(len(mb) for mb in micro_batches)
    assert (
        total_work_after >= total_work_before
    ), f"Samples were removed: {total_work_before} -> {total_work_after}"

    return micro_batches, leftovers, exec_times, sample_ids_per_gpu


def next_hdp_group_v2(
    sample_seqlens: List[Tuple[int, int]],
    total_gpus: int,
    max_seq_len_per_rank: int,
    min_cp_size: int = 1,
    delta: float = 0.05,
    global_target: Optional[float] = None,
) -> Tuple[List[List[int]], List[Tuple[int, int]], List[float], List[List[int]]]:
    """V2-pack3 DCP scheduler: V2-orig 全机制 + packing-aware cap.

    直接 port v2orig_reference_impl.py (claude-data/2026-04-16) 的 V2-orig 算法,
    唯一改动: cap 公式从 tall²/cp_min × (1+δ) 改成 tall × MSLPR × (1+δ).

    V2-orig 关键机制 (相比之前的 V2-pack / V2-pack2 保留):
    - 无 V1 bucket, 纯 desc by L 遍历
    - 无 slack 终止, 遍历 remaining 直到 cap 拒绝或放完
    - Step 1: 第一条 tall seq 强制 cp_min 放 new group (不参与 argmin)
    - Step 2: 剩余 seq 循环 cp ∈ {cp_min, 2·cp_min, ..., total_gpus}
      - Option A: existing groups with sz == cp (严格相等, 每个 cp 一次)
      - Option B: new group at this cp
      - argmin proj_max across all candidates
      - cap 作为 hard 过滤
      - 无 valid -> leftover
    - Step 3: fill_empty 扩展 smallest group (V1 机制)

    cap 创新 (vs V2-orig):
      V2-orig cap = tall²/cp_min × 1.1 在短 seq mb (tall << MSLPR) 下过紧,
      导致 m=1536 σ=1.8 下 -27% regression.
      V2-pack3 cap = tall × MSLPR × 1.05 是 per-rank workload 的真正上界
      (tall pole 独占 cp_min 个 rank, 其他 seq 填剩余 MSLPR-tall/cp_min tokens,
       合计 tall × MSLPR). 在长 seq mb 下 ≈ V2-orig (tall/cp_min ≈ MSLPR),
       在短 seq mb 下宽 MSLPR/tall 倍, 修复灾难.

    global_target 参数历史遗留, 忽略.
    """
    if not sample_seqlens:
        return (
            [[] for _ in range(total_gpus)],
            [],
            [0.0 for _ in range(total_gpus)],
            [[] for _ in range(total_gpus)],
        )

    def cp_min_fn(L: int) -> int:
        return dcp_gpus_needed(L, max_seq_len_per_rank, min_cp_size)

    def wl(L: int, cp: int) -> float:
        return (L * L) / cp

    # Sorted desc by L (V2-orig style, 不用 bucket)
    sample_seqlens = sorted(sample_seqlens, key=lambda x: x[1], reverse=True)

    # V2-pack3: packing-aware cap = tall × MSLPR × (1+δ)
    local_tall = sample_seqlens[0][1]
    cap = float(local_tall) * float(max_seq_len_per_rank) * (1.0 + delta)

    micro_batches: List[List[int]] = [[] for _ in range(total_gpus)]
    exec_times: List[float] = [0.0] * total_gpus
    sample_ids_per_gpu: List[List[int]] = [[] for _ in range(total_gpus)]
    packing_sequence_len: dict = {}
    gpu_group_id: List[Optional[int]] = [None] * total_gpus
    group_members: dict = {}
    group_size: dict = {}
    next_gid = 0

    # Step 1: tall pole 强制 cp_min, 占 rank [0..cp0-1]
    sid0, L0 = sample_seqlens[0]
    cp0 = cp_min_fn(L0)
    gid0 = next_gid
    next_gid += 1
    members0 = list(range(cp0))
    group_members[gid0] = members0
    group_size[gid0] = cp0
    packing_sequence_len[gid0] = L0 / cp0
    per_gpu0 = wl(L0, cp0)
    for r in members0:
        gpu_group_id[r] = gid0
        micro_batches[r].append(L0)
        exec_times[r] += per_gpu0
        sample_ids_per_gpu[r].append(sid0)

    remaining = list(sample_seqlens[1:])
    leftovers: List[Tuple[int, int]] = []

    # Step 2: 逐条 desc 处理, 循环 cp 找 argmin proj_max
    idx = 0
    while idx < len(remaining):
        sid, seq_len = remaining[idx]
        cp_lo = cp_min_fn(seq_len)

        best = None  # (proj_max, cp, action, gid_or_None, members_or_None)
        cp = cp_lo
        while cp <= total_gpus:
            per_gpu_cost = wl(seq_len, cp)

            # Option A: add to existing group of size == cp
            for gid_c, sz in list(group_size.items()):
                if sz != cp:
                    continue
                if packing_sequence_len.get(gid_c, 0) + seq_len / cp > max_seq_len_per_rank:
                    continue
                members_c = group_members[gid_c]
                proj_max = 0.0
                m_set = set(members_c)
                for r_i, t in enumerate(exec_times):
                    nt = t + per_gpu_cost if r_i in m_set else t
                    if nt > proj_max:
                        proj_max = nt
                if proj_max > cap:
                    continue
                if best is None or proj_max < best[0]:
                    best = (proj_max, cp, 'add', gid_c, None)

            # Option B: new group from free ranks, size cp
            free = [r for r, g in enumerate(gpu_group_id) if g is None]
            if len(free) >= cp:
                chosen = sorted(free, key=lambda r: exec_times[r])[:cp]
                proj_max = 0.0
                ch_set = set(chosen)
                for r_i, t in enumerate(exec_times):
                    nt = t + per_gpu_cost if r_i in ch_set else t
                    if nt > proj_max:
                        proj_max = nt
                if proj_max <= cap:
                    if best is None or proj_max < best[0]:
                        best = (proj_max, cp, 'new', None, chosen)
            cp *= 2

        if best is None:
            leftovers.append((sid, seq_len))
            idx += 1
            continue

        _, cp_sel, action, gid_to, members_or_none = best
        per_gpu_cost = wl(seq_len, cp_sel)
        if action == 'add':
            members = group_members[gid_to]
            packing_sequence_len[gid_to] += seq_len / cp_sel
            for r in members:
                micro_batches[r].append(seq_len)
                exec_times[r] += per_gpu_cost
                sample_ids_per_gpu[r].append(sid)
        else:
            members = members_or_none
            g = next_gid
            next_gid += 1
            group_members[g] = members
            group_size[g] = cp_sel
            packing_sequence_len[g] = seq_len / cp_sel
            for r in members:
                gpu_group_id[r] = g
                micro_batches[r].append(seq_len)
                exec_times[r] += per_gpu_cost
                sample_ids_per_gpu[r].append(sid)
        idx += 1

    # 7) V1's fill_empty_gpus: expand smallest group, push others right
    def _fill_empty():
        nonlocal micro_batches, exec_times, sample_ids_per_gpu
        empty = [i for i in range(total_gpus) if not micro_batches[i]]
        if not empty:
            return False
        existing_sizes = set(group_size.values())
        if not existing_sizes:
            return False
        min_size = min(existing_sizes)
        next_power = min(min_size * 2, total_gpus)
        for gid, sz in list(group_size.items()):
            if sz != min_size:
                continue
            members = group_members[gid]
            step = next_power - min_size
            start, end = members[0], members[-1]
            empty_gpu = [i for i, mb in enumerate(micro_batches) if not mb][0]
            if end + 1 > empty_gpu:
                continue
            if end + step >= total_gpus:
                continue
            work_to_push = micro_batches[end + 1 : empty_gpu]
            exec_push = exec_times[end + 1 : empty_gpu]
            sids_push = sample_ids_per_gpu[end + 1 : empty_gpu]
            new_mb: List[List[int]] = [[]] * total_gpus
            new_et: List[float] = [0.0] * total_gpus
            new_sids: List[List[int]] = [[]] * total_gpus
            for i in range(start):
                new_mb[i] = micro_batches[i]
                new_et[i] = exec_times[i]
                new_sids[i] = sample_ids_per_gpu[i]
            for i in range(start, end + step + 1):
                new_mb[i] = micro_batches[end]
                new_et[i] = sum(wl(ll, next_power) for ll in micro_batches[end])
                new_sids[i] = sample_ids_per_gpu[end]
            for i, work in enumerate(work_to_push):
                new_mb[end + step + 1 + i] = work
                new_et[end + step + 1 + i] = exec_push[i]
                new_sids[end + step + 1 + i] = sids_push[i]
            group_size[gid] = next_power
            group_members[gid] = list(range(start, end + step + 1))
            for other_gid in list(group_size.keys()):
                if other_gid == gid:
                    continue
                if min(group_members[other_gid]) > end:
                    group_members[other_gid] = [
                        x + step for x in group_members[other_gid]
                    ]
            micro_batches = new_mb
            exec_times = new_et
            sample_ids_per_gpu = new_sids
            return True
        return False

    while any(not mb for mb in micro_batches):
        if not _fill_empty():
            break

    # DCP_SCHEDULER_DEBUG=1 开启 debug, =2 则连带打每 group 明细.
    import os as _os
    _dbg = int(_os.environ.get("DCP_SCHEDULER_DEBUG", "0"))
    if _dbg:
        _is_rank0 = True
        try:
            import torch.distributed as _dist
            # 严格 is True: 生产下 is_initialized() 返回 python bool; mock 环境下
            # 返回 MagicMock, 不等于 True, 保持 _is_rank0=True 让打印照常.
            if _dist.is_initialized() is True:
                _is_rank0 = (_dist.get_rank() == 0)
        except Exception:
            pass
        if _is_rank0:
            _n_in = len(sample_seqlens)
            _cp_histo = {}
            for _sz in group_size.values():
                _cp_histo[_sz] = _cp_histo.get(_sz, 0) + 1
            _emax = max(exec_times) if exec_times else 0.0
            _emin = min(exec_times) if exec_times else 0.0
            _emean = (sum(exec_times) / len(exec_times)) if exec_times else 0.0
            _util = (_emin / _emax * 100.0) if _emax > 0 else 0.0
            _tall = max((L for _, L in sample_seqlens), default=0)
            _cap_str = f"{cap:.3e}" if cap < float('inf') else "inf"
            print(
                f"[DCP-PACK] mb-done tall={_tall} cap={_cap_str} "
                f"n_in={_n_in} placed_groups={len(group_size)} leftover={len(leftovers)} "
                f"groups_by_cp={_cp_histo} "
                f"et_max={_emax/1e6:.2f}M et_min={_emin/1e6:.2f}M et_mean={_emean/1e6:.2f}M "
                f"util={_util:.1f}%",
                flush=True,
            )
            if _dbg >= 2:
                for _gid, _sz in group_size.items():
                    _members = group_members[_gid]
                    _pack = packing_sequence_len.get(_gid, 0)
                    _rank0_et = exec_times[_members[0]] if _members else 0.0
                    _seqs = micro_batches[_members[0]] if _members else []
                    print(
                        f"  [group {_gid}] cp={_sz} ranks=[{_members[0]}..{_members[-1]}] "
                        f"pack_len={_pack:.0f}/{max_seq_len_per_rank} "
                        f"rank_et={_rank0_et/1e6:.2f}M seqs={_seqs}",
                        flush=True,
                    )

    return micro_batches, leftovers, exec_times, sample_ids_per_gpu


def reorder_microbatches_for_pp(
    sample_id_groups: List[List[List[int]]],
    sample_id_to_seqlen: dict,
    exec_times_per_mb: Optional[List[List[float]]] = None,
) -> List[List[List[int]]]:
    """Reorder microbatches so lightest are at head and tail (PP bubble reduction).

    Bitonic arrangement: sorted asc → [L0, L1, L2, L3, L4] becomes [L0, L2, L4, L3, L1].
    Heaviest ends up in the middle; lightest at the edges.

    Weight = critical path of each microbatch.
    - If ``exec_times_per_mb`` is given (scheduler-provided per-rank cumulative
      workload), weight = max(exec_times[r]) over ranks — this is the real PP bubble
      driver (max single-rank time, not total work).
    - Fallback to sum of seq_len**2 (only correct when all seqs in a mb use the
      same cp_size, which is not generally true in DCP).
    """
    n = len(sample_id_groups)
    if n <= 2:
        return sample_id_groups

    if exec_times_per_mb is not None and len(exec_times_per_mb) == n:
        weights = [max(et) if et else 0.0 for et in exec_times_per_mb]
    else:
        weights = []
        for mb in sample_id_groups:
            seen = set()
            w = 0
            for sub in mb:
                for sid in sub:
                    if sid not in seen:
                        seen.add(sid)
                        L = sample_id_to_seqlen.get(int(sid), 0)
                        w += L * L
            weights.append(w)

    sorted_idx = sorted(range(n), key=lambda i: weights[i])
    result_order = [None] * n
    left, right = 0, n - 1
    toggle = 0
    for k in range(n):
        src = sorted_idx[k]
        if toggle == 0:
            result_order[left] = src
            left += 1
        else:
            result_order[right] = src
            right -= 1
        toggle ^= 1
    return [sample_id_groups[i] for i in result_order]


def align_sample_id_groups(sample_id_groups: List, microbatch_group_size_per_vp_stage: int) -> List:
    """Align len(sample_id_groups) to microbatch_group_size_per_vp_stage when VPP is enabled.

    Standalone version extracted from DefaultDynamicCPScheduler.
    """
    multiple = int(microbatch_group_size_per_vp_stage)
    remainder = (-len(sample_id_groups)) % multiple
    i = len(sample_id_groups) - 1

    def split_group(sample_id_group):
        total_hdp_ranks = len(sample_id_group)
        cu_ranks = [0]
        prev_cp_size = 0

        while cu_ranks[-1] != total_hdp_ranks:
            start_rank = cu_ranks[-1]
            sid0 = sample_id_group[start_rank][0]
            cp_size = 0
            for r in range(start_rank, total_hdp_ranks):
                if sid0 in sample_id_group[r]:
                    cp_size += 1
                else:
                    break
            assert (
                prev_cp_size == 0 or cp_size <= prev_cp_size
            ), f"split_group: CP size is not decreasing: prev={prev_cp_size}, cur={cp_size}"
            cu_ranks.append(start_rank + cp_size)
            prev_cp_size = cp_size
        if len(cu_ranks) == 2:
            return None, None

        k = 0
        while cu_ranks[k] < total_hdp_ranks // 2:
            k += 1

        old_mb = sample_id_group[: cu_ranks[k]] + [[] for _ in range(total_hdp_ranks - cu_ranks[k])]
        new_mb = sample_id_group[cu_ranks[k] :] + [[] for _ in range(cu_ranks[k])]
        old_mb = fill_empty_by_expanding_cp(old_mb)
        new_mb = fill_empty_by_expanding_cp(new_mb)
        return new_mb, old_mb

    def fill_empty_by_expanding_cp(sample_id_group):
        def fill_empty(sample_id_group):
            empty_size = sum(1 for x in sample_id_group if len(x) == 0)
            i = len(sample_id_group) - 1 - empty_size
            prev_cp_size = 0
            while i >= 0:
                sid0 = sample_id_group[i][0]
                cp_size = 0
                while sid0 in sample_id_group[i] and i >= 0:
                    cp_size += 1
                    i -= 1
                if cp_size > prev_cp_size and prev_cp_size != 0:
                    start_idx = i + 1 + cp_size
                    end_idx = -empty_size + prev_cp_size if -empty_size + prev_cp_size < 0 else None
                    sample_id_group[start_idx + 2 * prev_cp_size : end_idx] = sample_id_group[
                        start_idx + prev_cp_size : -empty_size
                    ]
                    sample_id_group[start_idx + prev_cp_size : start_idx + 2 * prev_cp_size] = (
                        sample_id_group[start_idx : start_idx + prev_cp_size]
                    )
                    break
                elif cp_size <= empty_size and i == -1:
                    end_idx = -empty_size + cp_size if -empty_size + cp_size < 0 else None
                    sample_id_group[2 * cp_size : end_idx] = sample_id_group[cp_size:-empty_size]
                    sample_id_group[cp_size : 2 * cp_size] = sample_id_group[0:cp_size]
                    break
                prev_cp_size = cp_size
            return sample_id_group

        while len(sample_id_group[-1]) == 0:
            sample_id_group = fill_empty(sample_id_group)
        return sample_id_group

    attempts_since_split = 0
    while remainder > 0:
        if i < 0:
            if attempts_since_split >= len(sample_id_groups):
                assert False, 'align_sample_id_groups: no tail microbatch has enough ids to split'
            i = len(sample_id_groups) - 1
        group1, group2 = split_group(sample_id_groups[i])
        if group1 is not None and group2 is not None:
            sample_id_groups[i] = group1
            sample_id_groups.append(group2)
            remainder -= 1
            attempts_since_split = 0
        else:
            attempts_since_split += 1
        i -= 1

    return sample_id_groups


# =============================================================================
# Workload estimation helpers for dynamic CP scheduling
# =============================================================================


@lru_cache(maxsize=128)
def dcp_gpus_needed(seq_len: int, max_seq_len_per_rank: int, min_cp_size: int = 1) -> int:
    """Number of GPUs needed, rounded up to the next power of 2, lower-bounded by min_cp_size."""
    raw = max(1, 2 ** ceil(log2(seq_len / max_seq_len_per_rank)))
    return max(min_cp_size, raw)


@lru_cache(maxsize=128)
def dcp_get_total_workload(
    seq_length: int, max_seq_len_per_rank: int, cp_size: Optional[int] = None, min_cp_size: int = 1
) -> float:
    """Estimate workload of a sub-sample for scheduling balance."""
    if cp_size is None:
        cp_size = dcp_gpus_needed(seq_length, max_seq_len_per_rank, min_cp_size)
    return (seq_length * seq_length) / cp_size


def dcp_make_buckets_equal(
    sample_seqlens: List[Tuple[int, int]],
    compute_estimator: Callable,
    max_seq_len_per_rank: int,
    min_cp_size: int = 1,
) -> List[deque]:
    """Split samples into buckets of roughly equal work, one per unique CP size."""
    seqlens = [seq_len for _, seq_len in sample_seqlens]
    k = len({dcp_gpus_needed(L, max_seq_len_per_rank, min_cp_size) for L in seqlens})

    work = []
    for _, s in sample_seqlens:
        cp_size = dcp_gpus_needed(s, max_seq_len_per_rank, min_cp_size)
        work.append(compute_estimator(s, cp_size))
    total_work = sum(work)
    target = total_work / k
    buckets, cur, cur_work = [], [], 0.0
    remaining_k = k

    for i, (sample_id, seq_len) in enumerate(sample_seqlens):
        w = compute_estimator(seq_len)
        projected = cur_work + w
        if cur and (
            projected > target * 1.1 or len(sample_seqlens) - i <= remaining_k - len(buckets)
        ):
            buckets.append(deque(cur))
            cur, cur_work = [], 0.0
            remaining_k -= 1
        cur.append((sample_id, seq_len))
        cur_work += w

    if cur:
        buckets.append(deque(cur))
    return buckets
