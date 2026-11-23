"""Look up the filename and trigger of an organized LoRA."""

from glob import escape
import json
from pathlib import Path

import folder_paths

from .lora_save import _check_folder_name, _number_component


class _ComboSocket(str):
    def __ne__(self, other):
        # Some core loaders still declare their COMBO as a list of filenames.
        # ComfyUI's link validator uses != when comparing socket types.
        if isinstance(other, list):
            return False
        return super().__ne__(other)


def _folder_parts(number, model, kind, use_name, name):
    if number is not None:
        number = _number_component(number)
    fields = [(model, "model"), (kind, "type")]
    if use_name:
        fields.append((name, "name"))
    for value, field in fields:
        if value is not None:
            _check_folder_name(value, field)
    return (number, *(value for value, _ in fields))


class Gen2_LoRALoad:
    CATEGORY = "Gen2/LoRA"
    FUNCTION = "load"
    RETURN_TYPES = (_ComboSocket("COMBO"), "STRING")
    RETURN_NAMES = ("lora_name", "trigger")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "number": ("STRING", {"forceInput": True, "display_name": "string",
                                      "tooltip": "Folder number as text, for example 3."}),
                "model": ("STRING", {"default": "", "dynamicPrompts": False}),
                "type": ("STRING", {"default": "", "dynamicPrompts": False}),
            },
            "optional": {
                "use_name": ("BOOLEAN", {"default": False, "socketless": True,
                                         "tooltip": "Look inside a named subfolder of type."}),
                "name": ("STRING", {"default": "", "dynamicPrompts": False, "socketless": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, number, model, type, use_name=False, name=""):
        # Linked values are unavailable during ComfyUI's cache check. Watch the
        # matching folders so unchanged files can still reuse downstream sampling.
        if use_name is None:
            # A prompt linked this switch, so even the lookup depth is unknown.
            return float("nan")
        parts = _folder_parts(number, model, type, use_name, name)
        pattern = "/".join(escape(part) if part is not None else "*" for part in parts)
        local = Path(folder_paths.models_dir) / "loras"
        roots = folder_paths.folder_names_and_paths["loras"][0]
        signature = []
        # A file added to a higher-priority root can also change which LoRA
        # the core loader opens. Later roots cannot shadow the saved file.
        for root in map(Path, [*roots, local]):
            for folder in sorted((root / "gen2").glob(pattern)):
                if not folder.is_dir():
                    continue
                signature.append((str(folder), folder.stat().st_mtime_ns))
                trigger_file = folder / "trigger.json"
                if root == local and trigger_file.is_file():
                    stat = trigger_file.stat()
                    signature.append((str(trigger_file), stat.st_mtime_ns, stat.st_size))
            if root == local:
                break
        return tuple(signature)

    def load(self, number: str, model: str, type: str, use_name: bool = False, name: str = "") -> tuple[str, str]:
        parts = _folder_parts(number, model, type, use_name, name)
        relative = Path("gen2", *parts)
        loras = Path(folder_paths.models_dir) / "loras"
        folder = loras / relative
        if not folder.is_dir():
            raise FileNotFoundError(f"LoRA folder does not exist: {folder}")

        extensions = folder_paths.folder_names_and_paths["loras"][1]
        matches = [
            path for path in folder.iterdir()
            if path.name.lower() != "trigger.json"
            and (not extensions or path.suffix.lower() in extensions)
            and path.is_file()
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one LoRA file in {folder}, found {len(matches)}.")
        path, = matches
        name = str(path.relative_to(loras))
        resolved = Path(folder_paths.get_full_path_or_raise("loras", name))
        if not resolved.samefile(path):
            raise ValueError(f"Another LoRA folder takes priority for {name!r}: {resolved}")

        trigger_file = folder / "trigger.json"
        try:
            trigger = json.loads(trigger_file.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid trigger in {trigger_file}: {error}") from error
        if not isinstance(trigger, str):
            raise ValueError(f"Expected a JSON string in {trigger_file}.")
        return name, trigger
