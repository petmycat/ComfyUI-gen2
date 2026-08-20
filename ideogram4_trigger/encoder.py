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
    interpolate_embedding,
    remap_atomic_token_ids,
    replace_trigger_embeddings,
)
from .trigger_binding import bind_trigger_prompt, create_private_ideogram4_fast_tokenizer
from .types import (
    IDEOGRAM4_LAYER_COUNT,
    V9_LORA_RANK,
    ModuleLoRA,
    Ideogram4TriggerActivator,
    PhaseCConditioningState,
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
    invocation_counts: dict[int, int]


@dataclass(frozen=True, slots=True)
class TriggerEncodingConfig:
    atomic_token_id: int | None
    lookup_token_id: int | None
    trigger_mask: torch.Tensor
    virtual_token_indices: torch.Tensor
    embedding: torch.Tensor | None
    te_layers: Mapping[int, ModuleLoRA]
    te_strength: float
    apply_output_attention_mask: bool = True


@dataclass(frozen=True, slots=True)
class EncoderOutput:
    conditioning: torch.Tensor
    attention_mask: torch.Tensor | None
    hooks_installed: int
    hooks_cleaned: bool

    def as_conditioning(self, phase_c_state: PhaseCConditioningState | None = None) -> list[list[Any]]:
        metadata: dict[str, Any] = {
            "pooled_output": None,
            "attention_mask": self.attention_mask,
        }
        if phase_c_state is not None:
            metadata["gen2_phase_c_v2"] = phase_c_state
        return [[self.conditioning, metadata]]


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
        output = embedding(token_ids, out_dtype=dtype)
    except TypeError:
        output = embedding(token_ids).to(dtype=dtype)
    if not isinstance(output, torch.Tensor) or output.ndim != 3:
        raise RuntimeError("Ideogram4 embedding lookup did not return a [B,L,H] tensor")
    if output.device != token_ids.device:
        raise RuntimeError(
            f"Ideogram4 embedding lookup returned {output.device}, expected {token_ids.device}"
        )
    return output


def _build_attention_bias(attention_mask: torch.Tensor | None, hidden: torch.Tensor) -> torch.Tensor | None:
    batch, sequence = hidden.shape[:2]
    mask = None
    if attention_mask is not None:
        if attention_mask.shape != (batch, sequence):
            raise ValueError("attention mask must be [B,L]")
        mask = 1.0 - attention_mask.to(hidden.dtype).reshape(batch, 1, 1, sequence)
        mask = mask.expand(batch, 1, sequence, sequence).masked_fill(
            mask.to(torch.bool), torch.finfo(hidden.dtype).min / 4
        )
    causal = torch.full(
        (sequence, sequence),
        torch.finfo(hidden.dtype).min / 4,
        dtype=hidden.dtype,
        device=hidden.device,
    ).triu_(1)
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
    active_slots = int(config.trigger_mask.sum().item())
    if config.atomic_token_id is None:
        if active_slots != 0 or config.lookup_token_id is not None or config.embedding is not None or config.te_layers:
            raise ValueError("stock literal mode must not carry V9 trigger metadata or components")
    else:
        if config.lookup_token_id is None or active_slots <= 0:
            raise ValueError("active V9 modes require atomic/lookup IDs and at least one trigger slot")
        active_indices = config.virtual_token_indices[config.trigger_mask]
        if not torch.all((active_indices >= 0) & (active_indices < 4)):
            raise ValueError("active V9 trigger slots must use virtual indices 0..3")
        if active_slots % 4 != 0:
            raise ValueError("active V9 trigger slots must occur in complete groups of four")
        expected_indices = torch.arange(4, device=active_indices.device).repeat(active_slots // 4)
        if not torch.equal(active_indices, expected_indices):
            raise ValueError("each V9 trigger occurrence must use ordered virtual indices 0,1,2,3")
    if config.embedding is not None and config.embedding.shape != (4, hidden_size):
        raise ValueError(f"V9 embedding must be [4,{hidden_size}]")
    if set(config.te_layers).difference(range(layer_count)):
        raise ValueError("TE module-LoRA contains invalid layers")
    if layer_count == IDEOGRAM4_LAYER_COUNT and config.te_layers and set(config.te_layers) != set(range(IDEOGRAM4_LAYER_COUNT)):
        raise ValueError("V9 TE module-LoRA must cover all 36 layers")


def _linear_dimensions(module: Any, fallback_out: int, fallback_in: int) -> tuple[int, int]:
    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if isinstance(in_features, int) and in_features > 0 and isinstance(out_features, int) and out_features > 0:
        return out_features, in_features
    original_shape = getattr(module, "_orig_shape", None)
    if isinstance(original_shape, (tuple, list)) and len(original_shape) == 2:
        return int(original_shape[0]), int(original_shape[1])
    weight = getattr(module, "weight", None)
    params = getattr(weight, "_params", None)
    quantized_shape = getattr(params, "orig_shape", None)
    if isinstance(quantized_shape, (tuple, list)) and len(quantized_shape) == 2:
        return int(quantized_shape[0]), int(quantized_shape[1])
    if weight is not None and len(weight.shape) == 2:
        return int(weight.shape[0]), int(weight.shape[1])
    if fallback_out > 0 and fallback_in > 0:
        return fallback_out, fallback_in
    raise ValueError("mlp.down_proj does not expose usable logical linear dimensions")


def _stage_loras(config: TriggerEncodingConfig, language_model: Any, device: torch.device, dtype: torch.dtype) -> dict[int, StagedLoRA]:
    staged: dict[int, StagedLoRA] = {}
    hidden_size = int(getattr(language_model.config, "hidden_size", 0))
    intermediate_size = int(getattr(language_model.config, "intermediate_size", 0))
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("Ideogram4 language model config lacks valid hidden/intermediate sizes")
    expected_down_shape = (V9_LORA_RANK, intermediate_size)
    expected_up_shape = (hidden_size, V9_LORA_RANK)
    for index, source in config.te_layers.items():
        module = language_model.layers[index].mlp.down_proj
        logical_shape = _linear_dimensions(module, hidden_size, intermediate_size)
        if logical_shape != (hidden_size, intermediate_size):
            raise ValueError(
                f"Layer {index} mlp.down_proj logical shape {logical_shape} conflicts with model config "
                f"({hidden_size}, {intermediate_size})"
            )
        if tuple(source.down.shape) != expected_down_shape or tuple(source.up.shape) != expected_up_shape:
            raise ValueError(
                f"Layer {index} artifact/module shape mismatch: down={tuple(source.down.shape)}, "
                f"up={tuple(source.up.shape)}, expected={expected_down_shape}/{expected_up_shape}"
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
        context.invocation_counts[layer_index] = context.invocation_counts.get(layer_index, 0) + 1
        updated = apply_masked_module_lora(
            output, inputs[0], context.trigger_mask, staged.down, staged.up,
            staged.strength, staged.alpha, staged.rank,
        )
        if not torch.isfinite(updated).all().item():
            raise RuntimeError(f"Layer {layer_index} V9 module-LoRA produced NaN or infinity")
        return updated
    return hook


def _install_hooks(language_model: Any, layer_indices: Mapping[int, StagedLoRA]) -> list[Any]:
    handles: list[Any] = []
    try:
        for index in sorted(layer_indices):
            handle = language_model.layers[index].mlp.down_proj.register_forward_hook(
                _hook_for_layer(index)
            )
            if handle is None or not callable(getattr(handle, "remove", None)):
                raise RuntimeError(f"Layer {index} returned an invalid forward-hook handle")
            handles.append(handle)
    except Exception as install_error:
        cleanup_errors: list[Exception] = []
        for handle in reversed(handles):
            try:
                handle.remove()
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise RuntimeError(
                f"Hook installation failed and {len(cleanup_errors)} installed hooks could not be removed"
            ) from install_error
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
    if hidden.shape[:2] != token_ids.shape or hidden.shape[-1] != hidden_size:
        raise RuntimeError(
            f"Ideogram4 embedding output shape {tuple(hidden.shape)} does not match "
            f"tokens {tuple(token_ids.shape)} and hidden size {hidden_size}"
        )
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
    invocation_counts: dict[int, int] = {}
    token = None
    primary_error: BaseException | None = None
    cleaned = False
    try:
        token = _HOOK_CONTEXT.set(HookContext(trigger_mask, staged, invocation_counts))
        handles = _install_hooks(language_model, staged)
        for layer in layers:
            def core() -> None:
                nonlocal hidden
                hidden = _call_decoder_layer(layer, hidden, attention_bias, freqs_cis, optimized_attention)
            if runtime.prefetch_executes_core:
                runtime.prefetch_queue_pop(
                    queue,
                    hidden.device,
                    layer,
                    hidden.dtype,
                    core=core,
                    enable_graph=False,
                )
            else:
                runtime.prefetch_queue_pop(queue, hidden.device, layer)
                core()
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if queue is not None:
            try:
                runtime.prefetch_queue_pop(queue, hidden.device, None)
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            _remove_hooks(handles)
            cleaned = True
        except BaseException as exc:
            cleanup_errors.append(exc)
        if token is not None:
            try:
                _HOOK_CONTEXT.reset(token)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            if primary_error is not None:
                raise RuntimeError(
                    f"Ideogram4 V9 encode failed and cleanup also reported {len(cleanup_errors)} error(s)"
                ) from primary_error
            raise RuntimeError(
                f"Ideogram4 V9 cleanup reported {len(cleanup_errors)} error(s)"
            ) from cleanup_errors[0]
    unexpected_counts = {
        index: invocation_counts.get(index, 0)
        for index in staged
        if invocation_counts.get(index, 0) != 1
    }
    if unexpected_counts:
        raise RuntimeError(
            f"V9 module-LoRA hooks did not execute exactly once per layer: {unexpected_counts}"
        )
    conditioning = hidden
    if not torch.isfinite(conditioning).all().item():
        raise RuntimeError("Ideogram4 V9 conditioning contains NaN or infinity")
    output_mask = None if attention_mask is None else attention_mask.to(conditioning.device)
    if config.apply_output_attention_mask:
        conditioning = apply_conditioning_attention_mask(conditioning, output_mask)
    return EncoderOutput(conditioning, output_mask, len(handles), cleaned)


def _phase_c_conditioning_state(
    output: EncoderOutput,
    binding: Any,
    activator: Ideogram4TriggerActivator,
    mode: str,
    literal: str,
) -> PhaseCConditioningState | None:
    if mode != "semantic_only" or binding.occurrence_count != 3:
        return None
    if len(binding.token_indices) != 12:
        raise RuntimeError("Phase C V2 requires exactly 12 expanded trigger slots")
    selected = output.conditioning[0, list(binding.token_indices)]
    virtual_indices = torch.tensor(binding.virtual_token_indices, device=selected.device, dtype=torch.long)
    occurrence_indices = torch.tensor(binding.occurrence_indices, device=selected.device, dtype=torch.long)
    occurrence_states = torch.empty(
        (3, 4, selected.shape[-1]),
        device=selected.device,
        dtype=selected.dtype,
    )
    assigned = torch.zeros((3, 4), device=selected.device, dtype=torch.bool)
    for state, occurrence_index, virtual_index in zip(selected, occurrence_indices, virtual_indices):
        occurrence = int(occurrence_index.item())
        virtual = int(virtual_index.item())
        if occurrence < 0 or occurrence >= 3 or virtual < 0 or virtual >= 4 or assigned[occurrence, virtual]:
            raise RuntimeError("Phase C trigger occurrence/slot metadata is malformed")
        occurrence_states[occurrence, virtual] = state
        assigned[occurrence, virtual] = True
    if not assigned.all().item():
        raise RuntimeError("Phase C trigger metadata does not contain a complete 3x4 occurrence grid")
    return PhaseCConditioningState(
        schema="gen2.ideogram4-phase-c-v2.conditioning",
        schema_version=1,
        mode=mode,
        literal=literal,
        occurrence_count=3,
        virtual_token_count=4,
        conditioning_width=int(selected.shape[-1]),
        occurrence_states=occurrence_states,
        embedding_file_sha256=activator.embedding.identity.file_sha256,
        te_adapter_file_sha256=activator.te_adapter.identity.file_sha256,
        compatibility_fingerprint=activator.embedding.manifest.compatibility_fingerprint,
    )


def _mode_config(activator: Ideogram4TriggerActivator, mode: str) -> tuple[torch.Tensor | None, Mapping[int, ModuleLoRA], float]:
    if mode not in RUNTIME_MODES:
        raise ValueError(f"unsupported V9 trigger mode {mode!r}")
    if not isinstance(activator, Ideogram4TriggerActivator):
        raise TypeError("activator input is not an Ideogram4 V9 trigger activator")
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
    vocab_limit_value = getattr(language_model, "vocab_size", None)
    if vocab_limit_value is None:
        config = getattr(language_model, "config", None)
        vocab_limit_value = getattr(config, "vocab_size", None)
    if vocab_limit_value is None:
        weight = getattr(getattr(language_model, "embed_tokens", None), "weight", None)
        vocab_limit_value = None if weight is None else weight.shape[0]
    if vocab_limit_value is None or int(vocab_limit_value) <= 0:
        raise RuntimeError("Ideogram4 language model does not expose a valid frozen vocabulary size")
    vocab_limit = int(vocab_limit_value)
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
        if not callable(set_clip_options):
            raise RuntimeError("Connected Ideogram4 CLIP cannot set its execution device")
        set_clip_options({"execution_device": execution_device})
        device = torch.device(execution_device)
        output = encode_qwen_layers(
            language_model, token_ids.to(device), attention_mask.to(device),
            TriggerEncodingConfig(
                binding.atomic_token_id, binding.lookup_token_id, trigger_mask.to(device),
                virtual_indices.to(device), embedding, te_layers, te_strength,
            ),
        )
    mode_notes = {
        "semantic_only": "four-slot embedding and 36-layer module-LoRA enabled",
        "embedding_only": "four-slot embedding enabled; module-LoRA disabled",
        "internal_only": "frozen four-slot initializer and 36-layer module-LoRA enabled",
        "activator_bypass": "frozen four-slot initializer enabled; module-LoRA disabled",
        "stock_literal": "literal encoded without virtual-slot expansion or module-LoRA",
    }
    expected_hooks = len(te_layers)
    if output.hooks_installed != expected_hooks:
        raise RuntimeError(
            f"V9 diagnostics mismatch: installed {output.hooks_installed} hooks, expected {expected_hooks}"
        )
    phase_c_state = _phase_c_conditioning_state(output, binding, activator, mode, literal)
    diagnostics = TriggerDiagnostics(
        mode, binding.rendered_text, literal, binding.occurrence_count, binding.slot_count,
        binding.token_spans, binding.token_indices, binding.virtual_token_indices, binding.occurrence_indices,
        binding.atomic_token_id, binding.lookup_token_id, len(binding.input_ids), report.runtime_source,
        report.backend_identity, str(output.conditioning.device), str(output.conditioning.dtype),
        output.hooks_installed, output.hooks_cleaned,
        (
            mode_notes[mode],
            "conditioning uses the final Qwen recurrent hidden state; no multi-layer tap concatenation",
            "Phase C V2 metadata attached" if phase_c_state is not None else "Phase C V2 metadata not routable",
        ),
    )
    return output.as_conditioning(phase_c_state), diagnostics


__all__ = [
    "EncoderOutput", "HookContext", "RUNTIME_MODES", "TriggerEncodingConfig", "encode_ideogram4_trigger",
    "_phase_c_conditioning_state",
    "encode_qwen_layers",
]
