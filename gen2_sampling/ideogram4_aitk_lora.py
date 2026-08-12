from __future__ import annotations

import logging
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import comfy.lora
import comfy.sd
import comfy.utils
import folder_paths


LOGGER = logging.getLogger(__name__)
_AI_TOOLKIT_TE_PREFIX = "lora_te.language_model."
_LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight", ".alpha")
_VERIFIED_ATTENTION_MODULE = re.compile(
    r"^layers\.(?P<layer>\d+)\.self_attn\.(?P<projection>q_proj|k_proj|v_proj|o_proj)$"
)
_CLIP_MAPPER_LOCK = threading.RLock()


@dataclass(frozen=True)
class RemapDiagnostics:
    source_te_tensors: int
    aliased_te_tensors: int
    recognized_te_tensors: int
    unrecognized_te_keys: tuple[str, ...]
    unexpected_module_paths: tuple[str, ...]
    untouched_keys: int


def split_ai_toolkit_te_key(key: str) -> tuple[str, str] | None:
    if not key.startswith(_AI_TOOLKIT_TE_PREFIX):
        return None
    rest = key[len(_AI_TOOLKIT_TE_PREFIX):]
    for suffix in _LORA_SUFFIXES:
        if rest.endswith(suffix):
            module_path = rest[:-len(suffix)]
            if module_path:
                return module_path, suffix
    return None


def ai_toolkit_adapter_base(module_path: str) -> str:
    return _AI_TOOLKIT_TE_PREFIX + module_path


def _is_patchable_weight_module(module) -> bool:
    return hasattr(module, "weight") and getattr(module, "weight", None) is not None


def build_clip_runtime_targets(clip) -> dict[str, str]:
    if clip is None or not hasattr(clip, "cond_stage_model"):
        return {}
    targets: dict[str, str] = {}
    for module_name, module in clip.cond_stage_model.named_modules():
        if not module_name or not _is_patchable_weight_module(module):
            continue
        targets[module_name] = module_name + ".weight"
    return targets


def build_clip_mapped_targets(clip) -> dict[str, str]:
    if clip is None or not hasattr(clip, "cond_stage_model"):
        return {}
    key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
    targets: dict[str, str] = {}
    for alias, state_key in key_map.items():
        if not isinstance(state_key, str) or not state_key.endswith(".weight"):
            continue
        if alias.startswith("text_encoders."):
            targets.setdefault(alias[len("text_encoders."):], state_key)
        targets.setdefault(state_key.removesuffix(".weight"), state_key)
    return targets


def _resolve_unique_suffix(module_path: str, candidates: dict[str, str]) -> str | None:
    direct = candidates.get(module_path)
    if direct is not None:
        return direct
    suffix = "." + module_path
    matches = sorted({target for actual_path, target in candidates.items() if actual_path.endswith(suffix)})
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_clip_target(module_path: str, runtime_targets: dict[str, str]) -> str | None:
    return _resolve_unique_suffix(module_path, runtime_targets)


@contextmanager
def stock_loader_with_clip_aliases(clip, clip_aliases: dict[str, str]):
    if not clip_aliases:
        yield
        return

    original_mapper = comfy.lora.model_lora_keys_clip

    def mapper_with_runtime_targets(model, key_map={}):
        mapped = original_mapper(model, key_map)
        if clip is not None and model is clip.cond_stage_model:
            mapped.update(clip_aliases)
        return mapped

    with _CLIP_MAPPER_LOCK:
        comfy.lora.model_lora_keys_clip = mapper_with_runtime_targets
        try:
            yield
        finally:
            comfy.lora.model_lora_keys_clip = original_mapper


def build_ai_toolkit_clip_aliases(
    lora: dict,
    clip,
) -> tuple[RemapDiagnostics, dict[str, str]]:
    mapped_targets = build_clip_mapped_targets(clip)
    runtime_targets = build_clip_runtime_targets(clip)
    ai_toolkit_aliases: dict[str, str] = {}
    source_te_tensors = 0
    recognized_te_tensors = 0
    untouched_keys = 0
    unrecognized = []
    unexpected_modules = set()

    for key in lora:
        parsed = split_ai_toolkit_te_key(key)
        if parsed is None:
            untouched_keys += 1
            continue

        source_te_tensors += 1
        module_path, _suffix = parsed
        if _VERIFIED_ATTENTION_MODULE.fullmatch(module_path) is None:
            unexpected_modules.add(module_path)

        state_key = resolve_clip_target(module_path, mapped_targets)
        if state_key is None:
            state_key = resolve_clip_target(module_path, runtime_targets)
        if state_key is None:
            unrecognized.append(key)
            continue

        adapter_base = ai_toolkit_adapter_base(module_path)
        existing = ai_toolkit_aliases.get(adapter_base)
        if existing is not None and existing != state_key:
            raise RuntimeError(
                "Ideogram4 AI Toolkit LoRA Loader resolved one adapter base to multiple CLIP targets: "
                f"{adapter_base!r} -> {existing!r} and {state_key!r}."
            )
        ai_toolkit_aliases[adapter_base] = state_key
        recognized_te_tensors += 1

    diagnostics = RemapDiagnostics(
        source_te_tensors=source_te_tensors,
        aliased_te_tensors=recognized_te_tensors,
        recognized_te_tensors=recognized_te_tensors,
        unrecognized_te_keys=tuple(unrecognized),
        unexpected_module_paths=tuple(sorted(unexpected_modules)),
        untouched_keys=untouched_keys,
    )
    return diagnostics, ai_toolkit_aliases


class Gen2_Ideogram4AITKLoRALoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The Ideogram4 diffusion model to patch."}),
                "clip": ("CLIP", {"tooltip": "The Ideogram4 Qwen3-VL text encoder to patch."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "AI-Toolkit or standard ComfyUI LoRA."}),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "strength_clip": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "load_lora"
    CATEGORY = "loaders/ideogram4"
    DESCRIPTION = (
        "Loads a LoRA through stock ComfyUI after adding adapter-base aliases for AI-Toolkit "
        "lora_te.language_model.* Qwen text-encoder keys. LoRA tensors are passed through unchanged."
    )

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return model, clip

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        diagnostics, ai_toolkit_aliases = build_ai_toolkit_clip_aliases(lora, clip)
        LOGGER.info(
            "[Ideogram4 AITK LoRA] AI-Toolkit Qwen TE keys detected=%d aliased=%d recognized=%d "
            "unrecognized=%d other_keys_untouched=%d",
            diagnostics.source_te_tensors,
            diagnostics.aliased_te_tensors,
            diagnostics.recognized_te_tensors,
            len(diagnostics.unrecognized_te_keys),
            diagnostics.untouched_keys,
        )
        if diagnostics.unexpected_module_paths:
            LOGGER.warning(
                "[Ideogram4 AITK LoRA] Aliased Qwen TE module paths outside the verified attention set: %s",
                diagnostics.unexpected_module_paths,
            )
        if diagnostics.unrecognized_te_keys:
            preview = diagnostics.unrecognized_te_keys[:8]
            raise RuntimeError(
                "Ideogram4 AI Toolkit LoRA Loader could not match all AI-Toolkit Qwen TE keys to the "
                f"connected CLIP. Unrecognized {len(diagnostics.unrecognized_te_keys)} keys; first keys: {preview}. "
                "ComfyUI's Ideogram4 Qwen text-encoder structure may have changed."
            )

        with stock_loader_with_clip_aliases(clip, ai_toolkit_aliases):
            return comfy.sd.load_lora_for_models(
                model,
                clip,
                lora,
                strength_model,
                strength_clip,
            )
