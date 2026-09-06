# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared sizing and initialization for Transformer Engine's native CP transport."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

_ARENA_ALIGNMENT = 256
_GDN_FAMILY_VARIANTS = ("gdn", "gated_delta_net", "kda")


def _two_buffer_payload_bytes(message_bytes: int) -> int:
    """Return arena bytes for aligned send and receive buffers."""
    aligned_bytes = ((message_bytes + _ARENA_ALIGNMENT - 1) // _ARENA_ALIGNMENT) * _ARENA_ALIGNMENT
    return aligned_bytes + message_bytes


def config_uses_gdn_family(config: TransformerConfig) -> bool:
    """Return whether model construction can instantiate a GDN-family layer."""
    has_gdn_dimensions = all(
        getattr(config, name, None) is not None
        for name in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        )
    )
    return config.experimental_attention_variant in _GDN_FAMILY_VARIANTS or has_gdn_dimensions


def get_native_cp_payload_bytes(
    config: TransformerConfig,
    tp_size: int,
    kv_channels: Optional[Union[int, Sequence[int]]] = None,
    *,
    include_gdn_state: bool = False,
) -> int:
    """Calculate one arena that can be reused by standard and linear attention.

    The arena contains only two communication buffers, so its size is independent
    of the number of runtime CP groups and of the selected CP size.
    """
    if config.max_seqlen_per_dp_cp_rank is None:
        raise RuntimeError("Native CP transport requires max_seqlen_per_dp_cp_rank.")
    if tp_size < 1:
        raise ValueError(f"tp_size must be positive, got {tp_size}")

    if kv_channels is None:
        kv_channels = config.kv_channels
    kv_width = sum(kv_channels) if isinstance(kv_channels, (tuple, list)) else 2 * kv_channels
    local_query_groups = max(1, config.num_query_groups // tp_size)
    element_size = torch.empty((), dtype=config.params_dtype).element_size()

    # Standard attention exchanges a [KV, dKV] pair in each ring step.
    kv_bytes = config.max_seqlen_per_dp_cp_rank * local_query_groups * kv_width * element_size
    payload_bytes = _two_buffer_payload_bytes(2 * kv_bytes)

    if config.num_moe_experts:
        max_packed_sequences = max(1, config.thd_max_packed_sequences or 1)
        aux_bytes = config.num_moe_experts * max_packed_sequences * 8
        payload_bytes = max(payload_bytes, _two_buffer_payload_bytes(aux_bytes))

    if include_gdn_state or config_uses_gdn_family(config):
        local_value_heads = config.linear_num_value_heads // tp_size
        state_bytes = (
            local_value_heads
            * config.linear_key_head_dim
            * (config.linear_value_head_dim + config.linear_key_head_dim)
            * torch.empty((), dtype=torch.float32).element_size()
        )
        payload_bytes = max(payload_bytes, _two_buffer_payload_bytes(state_bytes))

        # The causal-conv boundary is normally much smaller than the recurrent
        # state, but include it explicitly so unusual configurations remain safe.
        local_conv_width = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        ) // tp_size
        conv_bytes = (config.linear_conv_kernel_dim - 1) * local_conv_width * element_size
        payload_bytes = max(payload_bytes, _two_buffer_payload_bytes(conv_bytes))

        # Hybrid Qwen layers keep the scheduler layout contiguous for GDN and
        # route the occasional standard-attention contiguous<->zigzag
        # permutation over the bounded peer ring. One rank-local hidden buffer
        # is forwarded at a time and reuses the same send/receive arena.
        layout_bytes = config.max_seqlen_per_dp_cp_rank * config.hidden_size * element_size
        payload_bytes = max(payload_bytes, _two_buffer_payload_bytes(layout_bytes))

    return payload_bytes


def initialize_native_cp_transport_for_config(
    parent_group,
    config: TransformerConfig,
    tp_size: int,
    kv_channels: Optional[Union[int, Sequence[int]]] = None,
    *,
    include_gdn_state: bool = False,
):
    """Initialize TE's parent transport with the model-wide maximum payload."""
    from transformer_engine.pytorch.attention.native_cp_transport import (
        initialize_native_cp_transport,
    )

    return initialize_native_cp_transport(
        parent_group,
        get_native_cp_payload_bytes(
            config, tp_size, kv_channels, include_gdn_state=include_gdn_state
        ),
    )
