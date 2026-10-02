# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native LM-head recompute: config, weight updates and real TE slot capture."""

import gc
import os
import sys
import weakref

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _config(recompute=True, chunk=False, **overrides):
    values = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        attention_backend=AttnBackend.unfused,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gradient_accumulation_fusion=False,
        add_bias_linear=False,
        recompute_granularity="selective" if recompute else None,
        recompute_modules=["lm_head"] if recompute else None,
        cuda_graph_impl="transformer_engine" if chunk else "none",
        cuda_graph_granularity="chunk" if chunk else "layer",
    )
    values.update(overrides)
    return TransformerConfig(**values)


def test_lm_head_cli_preserves_other_modules_and_normalizes_tp1(monkeypatch):
    """Use the real parser/validator, including historical TP1 --sequence-parallel."""
    from megatron.training.arguments import (
        core_transformer_config_from_args,
        parse_args,
        validate_args,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test",
            *f"""
            --num-layers 1 --hidden-size 64 --num-attention-heads 4
            --seq-length 16 --max-position-embeddings 16 --micro-batch-size 1
            --global-batch-size {os.environ.get("WORLD_SIZE", "1")}
            --train-iters 2 --lr 0.0001 --bf16 --mock-data
            --tokenizer-type NullTokenizer --vocab-size 128
            --untie-embeddings-and-output-weights --sequence-parallel
            --recompute-granularity selective --recompute-modules layernorm lm_head
        """.split(),
        ],
    )
    args = validate_args(parse_args())
    config = core_transformer_config_from_args(args)
    assert config.recompute_modules == ["layernorm", "lm_head"]
    assert config.tensor_model_parallel_size == 1 and not config.sequence_parallel
    assert config.params_dtype == torch.bfloat16


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"recompute_granularity": None}, "selective"),
        ({"params_dtype": torch.float32}, "BF16"),
        ({"tensor_model_parallel_size": 2}, "TP1"),
        ({"fp8": "e4m3"}, "FP8/FP4"),
        ({"mtp_num_layers": 1}, "MTP"),
        ({"cross_entropy_loss_fusion": True, "cross_entropy_fusion_impl": "native"}, "unfused"),
        (
            {
                "defer_embedding_wgrad_compute": True,
                "pipeline_model_parallel_size": 2,
                "pipeline_dtype": torch.bfloat16,
                "num_layers": 2,
                "gradient_accumulation_fusion": True,
            },
            "deferred",
        ),
        ({"distribute_saved_activations": True}, "sharding"),
        ({"cuda_graph_impl": "full_iteration"}, "Transformer Engine"),
        ({"use_mup": True}, "MuP"),
    ],
)
def test_lm_head_config_rejects_unsupported_paths(overrides, message):
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


@pytest.fixture
def model_parallel(monkeypatch):
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "1")
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(123)
    yield
    torch.cuda.synchronize()
    Utils.destroy_model_parallel()


def _model(recompute, chunk=False, fused=False, tied=False, layer_graph=False):
    overrides = {"gradient_accumulation_fusion": fused}
    if layer_graph:
        overrides["cuda_graph_impl"] = "transformer_engine"
    config = _config(recompute, chunk, **overrides)
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=8192,
        max_sequence_length=32,
        pre_process=False,
        post_process=True,
        share_embeddings_and_output_weights=tied,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    ).cuda()
    if fused:
        weight = model.output_layer.weight
        weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        weight.grad_added_to_main_grad = False
        weight.zero_out_wgrad = True
    return model


def _head(model, hidden, labels, **kwargs):
    return model._postprocess(
        hidden_states=hidden,
        input_ids=None,
        position_ids=None,
        labels=labels,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        **kwargs,
    )


def _inputs(offset):
    hidden = torch.arange(16 * 64, device="cuda").reshape(16, 1, 64) % 17 / 32
    hidden = (hidden + offset / 16).to(torch.bfloat16).requires_grad_(True)
    labels = (torch.arange(16, device="cuda").reshape(1, 16) * 37 + offset) % 8192
    return hidden, labels


def _clear_weight_grad(model):
    weight = model.output_layer.weight
    # Stable leaf-owned grad storage also works with TE's recyclable backward outputs.
    if weight.grad is None:
        weight.grad = torch.zeros_like(weight)
    else:
        weight.grad.zero_()
    if hasattr(weight, "main_grad"):
        weight.main_grad.zero_()
        weight.grad_added_to_main_grad = False


def _weight_grad(model):
    weight = model.output_layer.weight
    return weight.main_grad if model.config.gradient_accumulation_fusion else weight.grad


@pytest.mark.parametrize("backend", ["eager", "layer", "chunk"])
@pytest.mark.parametrize("fused", [False, True], ids=["returned-wgrad", "main-grad"])
def test_native_head_recompute_gradients_updates_and_forward_saves(model_parallel, backend, fused):
    """The original native math is unchanged, and softmax is not saved across F/B."""
    reference = _model(False, fused=fused)
    actual = _model(True, chunk=backend == "chunk", layer_graph=backend == "layer", fused=fused)
    actual.load_state_dict(reference.state_dict())
    ref_master = reference.output_layer.weight.detach().float().clone()
    act_master = actual.output_layer.weight.detach().float().clone()
    calls = []
    hook = actual.output_layer.register_forward_hook(lambda *args: calls.append(True))
    try:
        for step in range(2):
            _clear_weight_grad(reference)
            _clear_weight_grad(actual)
            for microbatch in range(2):
                h, labels = _inputs(step * 2 + microbatch)
                ref_h = h.detach().clone().requires_grad_(True)
                saved = {False: [], True: []}

                def record(tensor, enabled):
                    saved[enabled].append((tuple(tensor.shape), tensor.dtype))
                    return tensor

                with torch.autograd.graph.saved_tensors_hooks(
                    lambda tensor: record(tensor, False), lambda tensor: tensor
                ):
                    expected = _head(reference, ref_h, labels)
                rng = torch.cuda.get_rng_state()
                with torch.autograd.graph.saved_tensors_hooks(
                    lambda tensor: record(tensor, True), lambda tensor: tensor
                ):
                    result = _head(actual, h, labels)
                assert type(result.grad_fn).__name__ == (
                    "_CheckpointFunctionBackward"
                    if backend == "chunk"
                    else "CheckpointFunctionBackward"
                )
                softmax = ((16, 1, 8192), torch.float32)
                assert softmax in saved[False] and softmax not in saved[True]
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                result.mean().backward()
                expected.mean().backward()
                assert torch.equal(torch.cuda.get_rng_state(), rng)
                torch.testing.assert_close(h.grad, ref_h.grad, rtol=0, atol=0)
            torch.testing.assert_close(
                _weight_grad(actual), _weight_grad(reference), rtol=0, atol=0
            )
            if fused:
                assert actual.output_layer.weight.grad_added_to_main_grad
            with torch.no_grad():
                ref_master.add_(_weight_grad(reference).float(), alpha=-0.1)
                act_master.add_(_weight_grad(actual).float(), alpha=-0.1)
                reference.output_layer.weight.copy_(ref_master)
                actual.output_layer.weight.copy_(act_master)
            torch.testing.assert_close(act_master, ref_master, rtol=0, atol=0)
        assert len(calls) == 8, "four forwards and exactly four recomputes"
    finally:
        hook.remove()


def test_lm_head_runtime_guards_and_eval(model_parallel):
    with pytest.raises(ValueError, match="untied"):
        _model(True, tied=True)
    model = _model(True)
    h, labels = _inputs(0)
    with pytest.raises(ValueError, match="custom output"):
        _head(model, h, labels, output_processor=lambda **kwargs: None)
    with pytest.raises(ValueError, match="differentiable BF16"):
        _head(model, h.detach(), labels)
    model.eval()
    calls = []
    hook = model.output_layer.register_forward_hook(lambda *args: calls.append(True))
    try:
        with torch.no_grad():
            assert _head(model, h, labels).shape == labels.shape
            assert _head(model, h, None).shape == (1, 16, 8192)
        assert len(calls) == 2
    finally:
        hook.remove()


class _HeadOnly(torch.nn.Module):
    """Register only the actual head parameters while calling production GPT postprocess."""

    def __init__(self, owner):
        super().__init__()
        self.output_layer = owner.output_layer
        self.owner = weakref.ref(owner)

    def forward(self, hidden, labels):
        return _head(self.owner(), hidden, labels)


def test_lm_head_real_te_slot_capture_releases_vocab_save(model_parallel):
    """Real slot arenas shrink, graph labels refresh, and weight updates remain visible."""
    from transformer_engine.pytorch import make_graphed_callables

    arena_bytes = {}
    for enabled in (False, True):
        reference = _model(False, fused=True)
        owner = _model(enabled, chunk=True, fused=True)
        owner.load_state_dict(reference.state_dict())
        wrapper = _HeadOnly(owner)
        sample = _inputs(0)
        graphs = make_graphed_callables(
            (wrapper,),
            (sample,),
            num_warmup_iters=2,
            _order=[1, -1],
            _num_layers_per_chunk=[1],
            _reuse_graph_input_output_buffers=True,
            _graph_memory_slots=((0, 1, 0, 0, 0, 0, 1),),
        )
        arena_bytes[enabled] = sum(
            tensor.numel() * tensor.element_size()
            for tensor in graphs[0]._te_cuda_graph_saved_arenas.values()
        )
        try:
            for offset in (0, 1, 0):
                _clear_weight_grad(reference)
                _clear_weight_grad(owner)
                h, labels = _inputs(offset)
                ref_h = h.detach().clone().requires_grad_(True)
                expected = _head(reference, ref_h, labels)
                result = graphs[0](h, labels)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                result.mean().backward()
                expected.mean().backward()
                torch.testing.assert_close(h.grad, ref_h.grad, rtol=0, atol=0)
                torch.testing.assert_close(
                    _weight_grad(owner), _weight_grad(reference), rtol=0, atol=0
                )
                with torch.no_grad():
                    reference.output_layer.weight.add_(_weight_grad(reference), alpha=-0.1)
                    owner.output_layer.weight.add_(_weight_grad(owner), alpha=-0.1)
                torch.testing.assert_close(
                    owner.output_layer.weight, reference.output_layer.weight, rtol=0, atol=0
                )
        finally:
            for graph in graphs:
                graph.reset()
            del graphs, wrapper, owner, reference, sample
            gc.collect()
            torch.cuda.synchronize()
    softmax_bytes = 16 * 8192 * 4
    hidden_bytes = 16 * 64 * 2
    assert arena_bytes[False] >= softmax_bytes
    assert arena_bytes[False] - arena_bytes[True] >= softmax_bytes - hidden_bytes
