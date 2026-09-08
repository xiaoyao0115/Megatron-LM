# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.extensions.native_cp_transport import get_native_cp_payload_bytes


def _qwen35_config(**overrides):
    values = {
        "max_seqlen_per_dp_cp_rank": 128,
        "hidden_size": 2048,
        "num_query_groups": 2,
        "kv_channels": 256,
        "params_dtype": torch.bfloat16,
        "num_moe_experts": 256,
        "thd_max_packed_sequences": None,
        "experimental_attention_variant": "gdn",
        "linear_attention_freq": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen35_gdn_state_determines_native_cp_arena_size():
    config = _qwen35_config()

    payload_bytes = get_native_cp_payload_bytes(config, tp_size=1)

    assert payload_bytes == 8 * 1024 * 1024


def test_native_cp_arena_scales_with_tp_local_gdn_heads():
    config = _qwen35_config()

    payload_bytes = get_native_cp_payload_bytes(config, tp_size=2)

    assert payload_bytes == 4 * 1024 * 1024


def test_attention_only_config_does_not_reserve_gdn_state():
    config = _qwen35_config(
        num_moe_experts=None,
        experimental_attention_variant=None,
        linear_attention_freq=None,
        linear_num_key_heads=None,
        linear_num_value_heads=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        linear_conv_kernel_dim=None,
    )

    payload_bytes = get_native_cp_payload_bytes(config, tp_size=1)

    assert payload_bytes == 1024 * 1024


def test_explicit_hybrid_gdn_dimensions_reserve_state_before_layer_construction():
    config = _qwen35_config(experimental_attention_variant=None, linear_attention_freq=None)

    payload_bytes = get_native_cp_payload_bytes(config, tp_size=1)

    assert payload_bytes == 8 * 1024 * 1024


def test_moe_aux_payload_scales_with_max_packed_sequences():
    config = _qwen35_config(
        max_seqlen_per_dp_cp_rank=1,
        num_query_groups=1,
        kv_channels=1,
        thd_max_packed_sequences=2048,
        experimental_attention_variant=None,
        linear_attention_freq=None,
        linear_num_key_heads=None,
        linear_num_value_heads=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        linear_conv_kernel_dim=None,
    )

    payload_bytes = get_native_cp_payload_bytes(config, tp_size=1)

    assert payload_bytes == 8 * 1024 * 1024
