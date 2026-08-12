from __future__ import annotations

import logging
import re
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


@dataclass(frozen=True)
class RemapDiagnostics:
    source_te_tensors: int
    remapped_te_tensors: int
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


def convert_ai_toolkit_qwen_te_key(key: str) -> str:
    parsed = split_ai_toolkit_te_key(key)
    if parsed is None:
        return key
    module_path, suffix = parsed
    return "lora_te_" + module_path.replace(".", "_") + suffix


def build_clip_module_aliases(clip) -> dict[str, str]:
    if clip is None or not hasattr(clip, "cond_stage_model"):
        return {}
    key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {})
    aliases: dict[str, str] = {}
    for alias, state_key in key_map.items():
        if not alias.startswith("text_encoders."):
            continue
        module_path = alias[len("text_encoders."):]
        aliases.setdefault(module_path, alias)
        aliases.setdefault(state_key.removesuffix(".weight"), alias)
    return aliases


def resolve_clip_alias(module_path: str, aliases: dict[str, str]) -> str | None:
    direct = aliases.get(module_path)
    if direct is not None:
        return direct

    suffix = "." + module_path
    matches = sorted({alias for actual_path, alias in aliases.items() if actual_path.endswith(suffix)})
    if len(matches) == 1:
        return matches[0]
    return None


def remap_ai_toolkit_ideogram_te_keys(lora: dict, clip) -> tuple[dict, RemapDiagnostics]:
    aliases = build_clip_module_aliases(clip)
    converted = {}
    source_te_tensors = 0
    remapped_te_tensors = 0
    recognized_te_tensors = 0
    untouched_keys = 0
    unrecognized = []
    unexpected_modules = set()

    for key, value in lora.items():
        parsed = split_ai_toolkit_te_key(key)
        if parsed is None:
            converted[key] = value
            untouched_keys += 1
            continue

        source_te_tensors += 1
        module_path, suffix = parsed
        if _VERIFIED_ATTENTION_MODULE.fullmatch(module_path) is None:
            unexpected_modules.add(module_path)

        alias = resolve_clip_alias(module_path, aliases)
        if alias is None:
            unrecognized.append(key)
            converted[key] = value
            continue

        new_key = alias + suffix
        if new_key in converted or (new_key in lora and new_key != key):
            raise RuntimeError(
                "Ideogram4 AI Toolkit LoRA Loader produced a duplicate key while remapping "
                f"{key!r} to {new_key!r}."
            )
        converted[new_key] = value
        remapped_te_tensors += 1
        recognized_te_tensors += 1

    diagnostics = RemapDiagnostics(
        source_te_tensors=source_te_tensors,
        remapped_te_tensors=remapped_te_tensors,
        recognized_te_tensors=recognized_te_tensors,
        unrecognized_te_keys=tuple(unrecognized),
        unexpected_module_paths=tuple(sorted(unexpected_modules)),
        untouched_keys=untouched_keys,
    )
    return converted, diagnostics


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
        "Loads a LoRA through stock ComfyUI after remapping only AI-Toolkit "
        "lora_te.language_model.* Qwen text-encoder keys to the connected CLIP's actual patch targets."
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

        converted_lora, diagnostics = remap_ai_toolkit_ideogram_te_keys(lora, clip)
        LOGGER.info(
            "[Ideogram4 AITK LoRA] AI-Toolkit Qwen TE keys detected=%d remapped=%d recognized=%d "
            "unrecognized=%d other_keys_untouched=%d",
            diagnostics.source_te_tensors,
            diagnostics.remapped_te_tensors,
            diagnostics.recognized_te_tensors,
            len(diagnostics.unrecognized_te_keys),
            diagnostics.untouched_keys,
        )
        if diagnostics.unexpected_module_paths:
            LOGGER.warning(
                "[Ideogram4 AITK LoRA] Remapped Qwen TE module paths outside the verified attention set: %s",
                diagnostics.unexpected_module_paths,
            )
        if diagnostics.unrecognized_te_keys:
            preview = diagnostics.unrecognized_te_keys[:8]
            raise RuntimeError(
                "Ideogram4 AI Toolkit LoRA Loader could not match all AI-Toolkit Qwen TE keys to the "
                f"connected CLIP. Unrecognized {len(diagnostics.unrecognized_te_keys)} keys; first keys: {preview}. "
                "ComfyUI's Ideogram4 Qwen text-encoder structure may have changed."
            )

        return comfy.sd.load_lora_for_models(
            model,
            clip,
            converted_lora,
            strength_model,
            strength_clip,
        )
