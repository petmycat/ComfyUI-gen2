from __future__ import annotations

import contextvars
import inspect
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .compatibility import (
    compute_freqs_cis,
    load_clip_model,
    require_comfy_runtime,
    resolve_qwen_transformer,
    validate_ideogram4_backend,
)
from .math_ops import (
    apply_conditioning_attention_mask,
    apply_masked_module_lora,
    combine_post_layer_captures,
    interpolate_embedding,
    remap_atomic_token_ids,
    replace_trigger_embeddings,
)
from .trigger_binding import bind_trigger_prompt, create_private_ideogram4_fast_tokenizer
from .types import (
    IDEOGRAM4_CAPTURE_LAYERS,
    IDEOGRAM4_LAYER_COUNT,
    ModuleLoRA,
    Ideogram4TriggerActivator,
    TriggerDiagnostics,
)

RUNTIME_MODES = (
    "semantic_only",
    "embedding_only",
    "internal_only",
    "activator_bypass",
    "stock_literal",
)


@dataclass(frozen=True, slots=True)
class StagedLoRA:
    down: torch.Tensor
    up: torch.Tensor
    strength: float
    alpha: float
    rank: int


@dataclass(frozen=True, slots=True)
class HookContext:
    trigger_mask: torch.Tensor
    layers: Mapping[int, StagedLoRA]


@dataclass(frozen=True, slots=True)
class TriggerEncodingConfig:
    atomic_token_id: int | None
    lookup_token_id: int | None
    trigger_mask: torch.Tensor
    virtual_token_indices: torch.Tensor
    embedding: torch.Tensor | None
    te_layers: Mapping[int, ModuleLoRA]
    te_strength: float
    capture_layers: tuple[int, ...] = IDEOGRAM4_CAPTURE_LAYERS
    apply_output_attention_mask: bool = True


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    conditioning: torch.Tensor
    attention_mask: torch.Tensor | None
    captures: tuple[torch.Tensor, ...]
    recurrent_hidden: torch.Tensor
    hooks_installed: int
    hooks_cleaned: bool

    def as_conditioning(self) -> list[list[Any]]:
        return [[self.conditioning, {"pooled_output": None, "attention_mask": self.attention_mask}]]


_HOOK_CONTEXT: contextvars.ContextVar[HookContext | None] = contextvars.ContextVar("ideogram4_v9_hook_context", default=None)
_MODEL_LOCKS_GUARD = threading.RLock()
_MODEL_LOCKS: weakref.WeakKeyDictionary[Any, threading.RLock] = weakref.WeakKeyDictionary()


def _model_lock(language_model: Any) -> threading.RLock:
    with _MODEL_LOCKS_GUARD:
        lock = _MODEL_LOCKS.get(language_model)
        if lock is None:
            lock = threading.RLock()
            _MODEL_LOCKS[language_model] = lock
        return lock


def _call_embedding(embedding: Any, token_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    try:
        return embedding(token_ids, out_dtype=dtype)
    except TypeError:
        return embedding(token_ids).to(dtype=dtype)


def _build_attention_bias(attention_mask: torch.Tensor | None, hidden: torch.Tensor) -> torch.Tensor | None:
    batch, sequence = hidden.shape[:2]
    mask = None
    if attention_mask is not None:
        if attention_mask.shape != (batch, sequence):
            raise ValueError("attention mask must be [B,L]")
        mask = 1.0 - attention_mask.to(hidden.dtype).reshape(batch, 1, 1, sequence)
        mask = mask.expand(batch, 1, sequence, sequence).masked_fill(mask.to(torch.bool), float("-inf"))
    causal = torch.full((sequence, sequence), float("-inf"), dtype=hidden.dtype, device=hidden.device).triu_(1)
    return causal if mask is None else mask + causal


def _call_decoder_layer(layer: Any, hidden: torch.Tensor, attention_bias: torch.Tensor | None, freqs_cis: Any, optimized_attention: Any) -> torch.Tensor:
    kwargs = {"x": hidden, "attention_mask": attention_bias, "freqs_cis": freqs_cis, "optimized_attention": optimized_attention}
    try:
        if "past_key_value" in inspect.signature(layer.forward).parameters:
            kwargs["past_key_value"] = None
    except (TypeError, ValueError, AttributeError):
        pass
    result = layer(**kwargs)
    return result[0] if isinstance(result, tuple) else result


def _validate_config(config: TriggerEncodingConfig, hidden_size: int, layer_count: int) -> None:
    if config.trigger_mask.dtype != torch.bool or config.trigger_mask.shape != config.virtual_token_indices.shape:
        raise ValueError("trigger_mask and virtual_token_indices must be aligned [B,L]")
    if config.embedding is not None and config.embedding.shape != (4, hidden_size):
        raise ValueError(f"V9 embedding must be [4,{hidden_size}]")
    if set(config.te_layers).difference(range(layer_count)):
        raise ValueError("TE module-LoRA contains invalid layers")
    if layer_count == IDEOGRAM4_LAYER_COUNT and config.te_layers and set(config.te_layers) != set(range(IDEOGRAM4_LAYER_COUNT)):
        raise ValueError("V9 TE module-LoRA must cover all 36 layers")


def _stage_loras(config: TriggerEncodingConfig, language_model: Any, device: torch.device, dtype: torch.dtype) -> dict[int, StagedLoRA]:
    staged: dict[int, StagedLoRA] = {}
    for index, source in config.te_layers.items():
        module = language_model.layers[index].mlp.down_proj
        weight = getattr(module, "weight", None)
        if weight is None or len(weight.shape) != 2:
            raise ValueError(f"Layer {index} mlp.down_proj does not expose a 2D weight")
        out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
        if tuple(source.down.shape) != (source.rank, in_features) or tuple(source.up.shape) != (out_features, source.rank):
            raise ValueError(
                f"Layer {index} artifact/module shape mismatch: down={tuple(source.down.shape)}, "
                f"up={tuple(source.up.shape)}, module=({out_features},{in_features})"
            )
        staged[index] = StagedLoRA(
            source.down.to(device=device, dtype=dtype), source.up.to(device=device, dtype=dtype),
            config.te_strength, source.alpha, source.rank,
        )
    return staged


def _hook_for_layer(layer_index: int):
    def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        del module
        context = _HOOK_CONTEXT.get()
        if context is None or layer_index not in context.layers:
            return output
        if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
            raise RuntimeError(f"Layer {layer_index} down_proj hook received unsupported input/output")
        staged = context.layers[layer_index]
        return apply_masked_module_lora(
            output, inputs[0], context.trigger_mask, staged.down, staged.up,
            staged.strength, staged.alpha, staged.rank,
        )
    return hook


def _install_hooks(language_model: Any, layer_indices: Mapping[int, StagedLoRA]) -> list[Any]:
    handles: list[Any] = []
    try:
        for index in sorted(layer_indices):
            handles.append(language_model.layers[index].mlp.down_proj.register_forward_hook(_hook_for_layer(index)))
    except Exception:
        for handle in reversed(handles):
            try:
                handle.remove()
            except Exception:
                pass
        raise
    return handles


def _remove_hooks(handles: list[Any]) -> None:
    errors: list[Exception] = []
    for handle in reversed(handles):
        try:
            handle.remove()
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise RuntimeError(f"Failed to remove {len(errors)} Ideogram4 V9 hooks") from errors[0]


@torch.inference_mode()
def encode_qwen_layers(language_model: Any, token_ids: torch.Tensor, attention_mask: torch.Tensor | None, config: TriggerEncodingConfig, dtype: torch.dtype = torch.float32, position_ids: torch.Tensor | None = None) -> EncoderOutput:
    layers = language_model.layers
    hidden_size = int(getattr(language_model.config, "hidden_size", 0))
    _validate_config(config, hidden_size, len(layers))
    if config.trigger_mask.shape != token_ids.shape:
        raise ValueError("trigger metadata shape differs from token IDs")
    if config.atomic_token_id is not None and config.lookup_token_id is not None:
        token_ids = remap_atomic_token_ids(token_ids, config.atomic_token_id, config.lookup_token_id)
    hidden = _call_embedding(language_model.embed_tokens, token_ids, dtype)
    trigger_mask = config.trigger_mask.to(hidden.device)
    virtual_indices = config.virtual_token_indices.to(hidden.device)
    if config.embedding is not None:
        hidden = replace_trigger_embeddings(hidden, trigger_mask, virtual_indices, config.embedding)
    if position_ids is None:
        position_ids = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
    freqs_cis = compute_freqs_cis(language_model, position_ids, hidden.device)
    attention_bias = _build_attention_bias(attention_mask, hidden)
    runtime = require_comfy_runtime()
    optimized_attention = runtime.optimized_attention_for_device(hidden.device, mask=attention_bias is not None, small_input=True)
    queue = runtime.make_prefetch_queue(list(layers), hidden.device, {"prefetch_dynamic_vbars": getattr(language_model, "prefetch_dynamic_vbars", False)})
    staged = _stage_loras(config, language_model, hidden.device, hidden.dtype)
    handles: list[Any] = []
    token = None
    primary_error: BaseException | None = None
    captures: list[torch.Tensor] = []
    cleaned = False
    try:
        token = _HOOK_CONTEXT.set(HookContext(trigger_mask, staged))
        handles = _install_hooks(language_model, staged)
        capture_set = set(config.capture_layers)
        for index, layer in enumerate(layers):
            def core() -> None:
                nonlocal hidden
                hidden = _call_decoder_layer(layer, hidden, attention_bias, freqs_cis, optimized_attention)
            runtime.prefetch_queue_pop(queue, hidden.device, layer, hidden.dtype, core=core, enable_graph=False)
            if index in capture_set:
                captures.append(hidden.clone())
        if queue is not None:
            runtime.prefetch_queue_pop(queue, hidden.device, None)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error: BaseException | None = None
        try:
            _remove_hooks(handles)
            cleaned = True
        except BaseException as exc:
            cleanup_error = exc
        if token is not None:
            _HOOK_CONTEXT.reset(token)
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error
    if len(captures) != len(config.capture_layers):
        raise RuntimeError(f"captured {len(captures)} post-layer tensors, expected {len(config.capture_layers)}")
    conditioning = combine_post_layer_captures(captures)
    output_mask = None if attention_mask is None else attention_mask.to(conditioning.device)
    if config.apply_output_attention_mask:
        conditioning = apply_conditioning_attention_mask(conditioning, output_mask)
    return EncoderOutput(conditioning, output_mask, tuple(captures), hidden, len(handles), cleaned)


def _mode_config(activator: Ideogram4TriggerActivator, mode: str) -> tuple[torch.Tensor | None, Mapping[int, ModuleLoRA], float]:
    if mode not in RUNTIME_MODES:
        raise ValueError(f"unsupported V9 trigger mode {mode!r}")
    if mode == "stock_literal":
        return None, {}, 0.0
    embedding = None
    if mode in ("semantic_only", "embedding_only"):
        embedding = interpolate_embedding(activator.embedding.frozen_initializer, activator.embedding.weight, activator.embedding_strength)
    elif mode in ("internal_only", "activator_bypass"):
        embedding = activator.embedding.frozen_initializer
    te_layers = activator.te_adapter.layers if mode in ("semantic_only", "internal_only") else {}
    return embedding, te_layers, activator.internal_strength


@torch.inference_mode()
def encode_ideogram4_trigger(clip: Any, activator: Ideogram4TriggerActivator, text: str, mode: str = "semantic_only", literal: str = "<r1X1dOn9mA2>", max_length: int | None = None) -> tuple[list[list[Any]], TriggerDiagnostics]:
    clip_model, language_model = resolve_qwen_transformer(clip)
    lock = _model_lock(language_model)
    private_tokenizer = create_private_ideogram4_fast_tokenizer(clip.tokenizer, literal=literal)
    vocab_limit = int(getattr(language_model, "vocab_size", language_model.embed_tokens.weight.shape[0]))
    binding = bind_trigger_prompt(
        private_tokenizer,
        text,
        literal,
        vocab_limit=vocab_limit,
        max_length=max_length,
        stock_literal=mode == "stock_literal",
    )
    token_ids = torch.tensor([binding.input_ids], dtype=torch.long)
    attention_mask = torch.tensor([binding.attention_mask], dtype=torch.long)
    trigger_mask = torch.tensor([binding.trigger_mask], dtype=torch.bool)
    virtual_indices = torch.full_like(token_ids, -1)
    for token_index, virtual_index in zip(binding.token_indices, binding.virtual_token_indices):
        virtual_indices[0, token_index] = virtual_index
    embedding, te_layers, te_strength = _mode_config(activator, mode)
    with lock:
        report = validate_ideogram4_backend(clip)
        patcher = load_clip_model(clip, token_ids)
        execution_device = getattr(patcher, "load_device", None)
        if execution_device is None:
            execution_device = getattr(getattr(clip, "patcher", None), "load_device", None)
        if execution_device is None:
            raise RuntimeError("Loaded Ideogram4 CLIP patcher does not expose an execution device")
        set_clip_options = getattr(getattr(clip, "cond_stage_model", None), "set_clip_options", None)
        if callable(set_clip_options):
            set_clip_options({"execution_device": execution_device})
        device = torch.device(execution_device)
        output = encode_qwen_layers(
            language_model, token_ids.to(device), attention_mask.to(device),
            TriggerEncodingConfig(
                binding.atomic_token_id, binding.lookup_token_id, trigger_mask.to(device),
                virtual_indices.to(device), embedding, te_layers, te_strength,
            ),
        )
    diagnostics = TriggerDiagnostics(
        mode, binding.rendered_text, literal, binding.occurrence_count, binding.slot_count,
        binding.token_spans, binding.token_indices, binding.virtual_token_indices, binding.occurrence_indices,
        binding.atomic_token_id, binding.lookup_token_id, len(binding.input_ids), report.runtime_source,
        report.backend_identity, str(output.conditioning.device), str(output.conditioning.dtype),
        output.hooks_installed, output.hooks_cleaned,
        ("V9 uses four virtual slots and 36 independent down_proj module-LoRA hooks",),
    )
    return output.as_conditioning(), diagnostics


__all__ = [
    "EncoderOutput", "HookContext", "RUNTIME_MODES", "TriggerEncodingConfig", "encode_ideogram4_trigger",
    "encode_qwen_layers",
]
