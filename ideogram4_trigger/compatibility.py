from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

from .types import EXPECTED_HIDDEN_SIZE, IDEOGRAM4_CAPTURE_LAYERS, IDEOGRAM4_LAYER_COUNT


class UnsupportedComfyUIError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ComfyRuntime:
    optimized_attention_for_device: Callable[..., Any]
    make_prefetch_queue: Callable[..., Any]
    prefetch_queue_pop: Callable[..., Any]
    source: str


@dataclass(frozen=True, slots=True)
class Ideogram4CompatibilityReport:
    supported: bool
    backend_identity: str
    runtime_source: str
    hidden_size: int | None
    layer_count: int | None
    capture_layers: tuple[int, ...]
    model_class: str
    clip_model_class: str
    tokenizer_class: str
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "backend_identity": self.backend_identity,
            "runtime_source": self.runtime_source,
            "hidden_size": self.hidden_size,
            "layer_count": self.layer_count,
            "capture_layers": list(self.capture_layers),
            "model_class": self.model_class,
            "clip_model_class": self.clip_model_class,
            "tokenizer_class": self.tokenizer_class,
            "reason": self.reason,
        }


def _optional_import(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _direct_layer_call(queue, device, layer, dtype=None, core=None, enable_graph=False):
    del queue, device, layer, dtype, enable_graph
    return core() if core is not None else None


def _no_prefetch_queue(layers, device, model_options=None):
    del layers, device, model_options
    return None


def detect_comfy_runtime() -> ComfyRuntime | None:
    attention = _optional_import("comfy.ldm.modules.attention")
    if attention is None or not hasattr(attention, "optimized_attention_for_device"):
        return None
    prefetch = _optional_import("comfy.model_prefetch")
    if prefetch is not None and all(hasattr(prefetch, name) for name in ("make_prefetch_queue", "prefetch_queue_pop")):
        return ComfyRuntime(attention.optimized_attention_for_device, prefetch.make_prefetch_queue, prefetch.prefetch_queue_pop, "comfy-native-prefetch")
    return ComfyRuntime(attention.optimized_attention_for_device, _no_prefetch_queue, _direct_layer_call, "comfy-native-no-prefetch")


def require_comfy_runtime() -> ComfyRuntime:
    runtime = detect_comfy_runtime()
    if runtime is None:
        raise UnsupportedComfyUIError("ComfyUI lacks the native attention runtime required by Ideogram4/Qwen3-VL")
    return runtime


def resolve_clip_model(clip: Any) -> Any:
    try:
        cond_stage_model = clip.cond_stage_model
        return getattr(cond_stage_model, cond_stage_model.clip)
    except (AttributeError, TypeError) as exc:
        raise UnsupportedComfyUIError("Connected CLIP does not expose ComfyUI's text-encoder wrapper") from exc


def resolve_qwen_transformer(clip: Any) -> tuple[Any, Any]:
    clip_model = resolve_clip_model(clip)
    transformer = getattr(clip_model, "transformer", None)
    language_model = getattr(transformer, "model", None)
    if language_model is None:
        raise UnsupportedComfyUIError("Expected clip_model.transformer.model for native Ideogram4/Qwen3-VL")
    return clip_model, language_model


def _identity_text(*objects: Any) -> str:
    values: list[str] = []
    for value in objects:
        if value is None:
            continue
        values.extend((type(value).__module__, type(value).__name__))
        config = getattr(value, "config", None)
        for name in ("model_type", "architectures", "name_or_path", "_name_or_path"):
            item = getattr(config, name, None)
            if item is not None:
                values.append(str(item))
    return " ".join(values).lower()


def _validate_identity(clip_model: Any, language_model: Any, tokenizer: Any) -> str:
    identity = _identity_text(clip_model, getattr(clip_model, "transformer", None), language_model, tokenizer)
    if "klein" in identity or ("qwen3_8b" in identity and "vl" not in identity):
        raise UnsupportedComfyUIError("Flux/Klein Qwen3-8B is explicitly unsupported; native Ideogram4 Qwen3-VL is required")
    has_ideogram = "ideogram4" in identity or "ideogram_4" in identity
    has_qwen_vl = "qwen3" in identity and ("vl" in identity or "vision" in identity)
    if not (has_ideogram and has_qwen_vl):
        raise UnsupportedComfyUIError(
            "Backend identity is not the official native Ideogram4/Qwen3-VL implementation; current Klein/Qwen3-8B cores fail closed"
        )
    return identity


def validate_ideogram4_backend(clip: Any) -> Ideogram4CompatibilityReport:
    runtime = require_comfy_runtime()
    clip_model, language_model = resolve_qwen_transformer(clip)
    tokenizer = getattr(clip, "tokenizer", None)
    identity = _validate_identity(clip_model, language_model, tokenizer)
    layers = getattr(language_model, "layers", None)
    config = getattr(language_model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(getattr(language_model, "embed_tokens", None), "weight", None), "shape", (None, None))[-1]
    if hidden_size != EXPECTED_HIDDEN_SIZE or layers is None or len(layers) != IDEOGRAM4_LAYER_COUNT:
        raise UnsupportedComfyUIError(
            f"Ideogram4 V9 requires hidden size {EXPECTED_HIDDEN_SIZE} and {IDEOGRAM4_LAYER_COUNT} layers; got {hidden_size}/{None if layers is None else len(layers)}"
        )
    for index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        down_proj = getattr(mlp, "down_proj", None)
        if down_proj is None or not callable(getattr(down_proj, "register_forward_hook", None)):
            raise UnsupportedComfyUIError(f"Layer {index} lacks a hookable unique mlp.down_proj module")
    if not callable(getattr(language_model, "compute_freqs_cis", None)):
        config = getattr(language_model, "config", None)
        if config is None or not all(hasattr(config, name) for name in ("head_dim", "rope_theta", "rope_dims")):
            raise UnsupportedComfyUIError("Backend lacks native Qwen3-VL MRoPE/position interfaces")
    if tokenizer is None or not hasattr(tokenizer, "tokenizer"):
        raise UnsupportedComfyUIError("Connected CLIP lacks an identifiable native tokenizer wrapper")
    return Ideogram4CompatibilityReport(
        True, identity, runtime.source, int(hidden_size), len(layers), IDEOGRAM4_CAPTURE_LAYERS,
        type(language_model).__name__, type(clip_model).__name__, type(tokenizer).__name__, None,
    )


def inspect_ideogram4_clip(clip: Any) -> Ideogram4CompatibilityReport:
    try:
        return validate_ideogram4_backend(clip)
    except Exception as exc:
        runtime = detect_comfy_runtime()
        try:
            clip_model, language_model = resolve_qwen_transformer(clip)
        except Exception:
            clip_model, language_model = None, None
        layers = getattr(language_model, "layers", None)
        config = getattr(language_model, "config", None)
        return Ideogram4CompatibilityReport(
            False, _identity_text(clip_model, language_model), "unavailable" if runtime is None else runtime.source,
            getattr(config, "hidden_size", None), None if layers is None else len(layers), IDEOGRAM4_CAPTURE_LAYERS,
            type(language_model).__name__ if language_model is not None else "unresolved",
            type(clip_model).__name__ if clip_model is not None else "unresolved",
            type(getattr(clip, "tokenizer", None)).__name__, str(exc),
        )


def load_clip_model(clip: Any, tokens: Any = None) -> Any:
    load_model = getattr(clip, "load_model", None)
    if not callable(load_model):
        raise UnsupportedComfyUIError("Connected CLIP does not provide load_model()")
    try:
        return load_model(tokens) if tokens is not None else load_model()
    except TypeError:
        return load_model()


def compute_freqs_cis(language_model: Any, position_ids: Any, device: Any) -> Any:
    native = getattr(language_model, "compute_freqs_cis", None)
    if callable(native):
        return native(position_ids, device)
    llama = _optional_import("comfy.text_encoders.llama")
    fallback = None if llama is None else getattr(llama, "precompute_freqs_cis", None)
    config = getattr(language_model, "config", None)
    if not callable(fallback) or config is None or not getattr(config, "interleaved_mrope", False):
        raise UnsupportedComfyUIError("Native Ideogram4 Qwen3-VL interleaved MRoPE support is unavailable")
    try:
        return fallback(
            head_dim=config.head_dim, position_ids=position_ids, theta=config.rope_theta,
            rope_scale=getattr(config, "rope_scale", None), rope_dims=config.rope_dims,
            device=device, interleaved_mrope=True,
        )
    except TypeError as exc:
        raise UnsupportedComfyUIError("Installed ComfyUI RoPE helper lacks interleaved MRoPE") from exc


__all__ = [
    "ComfyRuntime", "Ideogram4CompatibilityReport", "UnsupportedComfyUIError", "compute_freqs_cis",
    "detect_comfy_runtime", "inspect_ideogram4_clip", "load_clip_model", "require_comfy_runtime",
    "resolve_clip_model", "resolve_qwen_transformer", "validate_ideogram4_backend",
]
