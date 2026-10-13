# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fixed-size communication for graph replay with changing packed CP layouts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.transformer_config import TransformerConfig


def supports_packed_gdn_graph(config: TransformerConfig) -> bool:
    """Whether the fixed-capacity GDN graph backend can represent this configuration."""
    capacity = config.max_seqlen_per_dp_cp_rank
    return (
        config.experimental_attention_variant == "gdn"
        and config.cuda_graph_impl == "transformer_engine"
        and config.sequence_packing_scheduler is not None
        and (config.linear_cp_mode, config.cp_partition_mode)
        in (("chunkwise", "contiguous"), ("headwise", "zigzag"))
        and config.tensor_model_parallel_size == 1
        and not config.gdn_pre_gated_delta_rule_fusion
        and not config.overlap_moe_expert_parallel_comm
        and not config.delay_wgrad_compute
        and capacity is not None
        and capacity > 0
        and capacity % 64 == 0
        and config.pad_packed_seq_alignment in ("max", capacity)
        and (config.thd_tail_padding_policy or "append_dummy_seq") == "append_dummy_seq"
    )


def validate_packed_gdn_graph_inputs(
    config: TransformerConfig, packed_seq_params: PackedSeqParams
) -> None:
    """Check each pack before replay, outside capture where host reads are legal.

    Chunkwise FLA graphs need 64-aligned physical sequences. Headwise graphs
    align them internally after the head exchange, preserving the input pack.
    Actual q/kv lengths may exclude padding,
    but self-attention requires matching query/key boundaries in both layouts.
    """
    if not supports_packed_gdn_graph(config):
        raise ValueError("Unsupported packed GDN graph configuration")
    cp_size = packed_seq_params.local_cp_size or config.context_parallel_size
    if config.linear_cp_mode == "headwise" and any(
        heads % (config.tensor_model_parallel_size * cp_size)
        for heads in (config.linear_num_key_heads, config.linear_num_value_heads)
    ):
        raise ValueError("GDN headwise graphs require CP to evenly divide the TP-local heads")
    capacity = config.max_seqlen_per_dp_cp_rank * cp_size
    arrays = (
        packed_seq_params.cu_seqlens_q,
        packed_seq_params.cu_seqlens_kv,
        packed_seq_params.cu_seqlens_q_padded,
        packed_seq_params.cu_seqlens_kv_padded,
    )
    if any(
        value is None
        or value.dtype != torch.int32
        or value.ndim != 1
        or value.numel() != config.thd_max_packed_sequences + 1
        for value in arrays
    ):
        raise ValueError("GDN graphs require fixed-capacity int32 q/kv packed metadata")
    actual_q, actual_kv, padded_q, padded_kv = torch.stack(arrays).cpu().tolist()
    if actual_q != actual_kv or padded_q != padded_kv:
        raise ValueError("GDN graphs require identical query and key sequence boundaries")
    if padded_q[0] != 0 or padded_q[-1] != capacity or actual_q[0] != 0:
        raise ValueError("GDN graph physical boundaries must span the full CP capacity")
    for index, (start, end) in enumerate(zip(padded_q, padded_q[1:])):
        if end < start or (end == start and end != capacity):
            raise ValueError("GDN graphs allow empty physical sequences only at the packed tail")
        valid_length = actual_q[index + 1] - actual_q[index]
        if not 0 <= valid_length <= end - start:
            raise ValueError("GDN graph valid lengths must fit their physical sequence lengths")
    if any(
        (config.linear_cp_mode == "chunkwise" and (end - start) % 64)
        or (end - start) % (2 * cp_size)
        for start, end in zip(padded_q, padded_q[1:])
    ):
        raise ValueError("GDN graph physical sequence lengths must be 64/2CP-aligned")


def pad_graph_headwise_sequences(value: torch.Tensor, cu_seqlens: torch.Tensor):
    """Align complete head-local sequences without changing the public THD pack.

    At most N-1 extra chunks cover rounding N sequences whose total is aligned.
    Extend the last physical sequence to this fixed bound; its extra outputs are
    discarded and its added decay/update gates must be zeroed by the caller.
    """
    lengths = cu_seqlens.diff()
    capacity = value.shape[0] + 64 * (cu_seqlens.numel() - 2)
    aligned_lengths = ((lengths + 63) // 64) * 64
    aligned_cu = torch.cat(
        (cu_seqlens.new_zeros(1), aligned_lengths.cumsum(0, dtype=cu_seqlens.dtype))
    )
    aligned_cu = torch.where(
        torch.arange(cu_seqlens.numel(), device=value.device) >= (lengths > 0).sum(),
        capacity,
        aligned_cu,
    )
    positions = torch.arange(value.shape[0], device=value.device)
    seq = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    indices = aligned_cu[seq].long() + positions - cu_seqlens[seq].long()
    padded = value.new_zeros((capacity, *value.shape[1:]))
    return padded.index_copy(0, indices, value), aligned_cu, indices


def _packed_layout_indices(cu_seqlens, local_tokens, cp_size, cp_rank):
    """Return both permutations into rank-major gathered contiguous/zigzag tensors."""
    positions = torch.arange(local_tokens, device=cu_seqlens.device, dtype=torch.long)
    seq = torch.searchsorted(cu_seqlens[1:] // cp_size, positions, right=True)
    bos = cu_seqlens[seq]
    half = (cu_seqlens[seq + 1] - bos) // (2 * cp_size)
    within = positions - bos // cp_size
    block = torch.where(within < half, cp_rank, 2 * cp_size - 1 - cp_rank)
    to_zigzag = bos + block * half + within % half

    positions = positions + cp_rank * local_tokens
    seq = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    bos = cu_seqlens[seq]
    half = (cu_seqlens[seq + 1] - bos) // (2 * cp_size)
    block = (positions - bos) // half
    source_rank = torch.minimum(block, 2 * cp_size - 1 - block)
    source_pos = bos // cp_size + (positions - bos) % half
    source_pos = source_pos + (block >= cp_size) * half
    to_contiguous = source_rank * local_tokens + source_pos
    return to_zigzag, to_contiguous


def _gather_and_select(value, indices, group):
    value = value.contiguous()
    gathered = value.new_empty((value.shape[0] * group.size(), *value.shape[1:]))
    torch.distributed.all_gather_into_tensor(gathered, value, group=group)
    return gathered.index_select(0, indices)


class _GraphPackedLayout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, cu_seqlens, group, to_zigzag):
        zigzag, contiguous = _packed_layout_indices(
            cu_seqlens, value.shape[0], group.size(), group.rank()
        )
        indices, inverse = (zigzag, contiguous) if to_zigzag else (contiguous, zigzag)
        ctx.save_for_backward(inverse)
        ctx.group = group
        return _gather_and_select(value, indices, group)

    @staticmethod
    def backward(ctx, grad_output):
        (inverse,) = ctx.saved_tensors
        return _gather_and_select(grad_output, inverse, ctx.group), None, None, None


def convert_graph_packed_layout(
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    group: torch.distributed.ProcessGroup,
    seq_dim: int,
    target_partition_mode: str,
) -> torch.Tensor:
    """Convert with fixed collective shapes and replay-dependent token permutations.

    Packed lengths must be divisible by ``2 * CP`` and total tokens must stay
    fixed. A gather costs more traffic than the eager variable-split all-to-all,
    but its captured communication remains valid for every pack in the graph bank.
    The backward uses the inverse permutation, without reducing duplicated data.
    """
    value = value.movedim(seq_dim, 0)
    result = _GraphPackedLayout.apply(value, cu_seqlens, group, target_partition_mode == "zigzag")
    return result.movedim(0, seq_dim).contiguous()
