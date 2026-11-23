"""Organize a selected LoRA and its trigger in a numbered folder."""

import json
import os
from pathlib import Path, PureWindowsPath
import shutil

import folder_paths


def _check_folder_name(name: str, field: str) -> None:
    component = PureWindowsPath(name)
    if not name.strip() or name == ".." or component.name != name or "\x00" in name:
        raise ValueError(f"{field} must be a single folder name.")
    if os.name == "nt" and (
        name.endswith((" ", ".")) or component.is_reserved()
        or any(c in '<>:"|?*' or ord(c) < 32 for c in name)
    ):
        raise ValueError(f"{field} is not a valid Windows folder name: {name!r}.")


def _next_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    numbers = (int(p.name) for p in root.iterdir()
               if p.name.isascii() and p.name.isdecimal() and p.is_dir())
    number = max(numbers, default=0) + 1
    while True:
        directory = root / str(number)
        try:
            directory.mkdir()
            return directory
        except FileExistsError:
            # Another save or an existing file can occupy this number.
            number += 1


def _move_file(source: Path, destination: Path) -> None:
    if source.is_symlink() and not (target := source.readlink()).is_absolute():
        # A relative link moved verbatim would point to a different file.
        destination.symlink_to(source.absolute().parent / target)
        try:
            source.unlink()
        except OSError:
            destination.unlink()
            raise
    else:
        shutil.move(str(source), str(destination))


def _organize(source: Path, model: str, kind: str, metadata: bytes) -> Path:
    directory = _next_directory(Path(folder_paths.models_dir) / "loras" / "gen2")
    folder = directory / model / kind
    destination = folder / source.name
    try:
        folder.mkdir(parents=True)
        _move_file(source, destination)
    except OSError:
        # A cross-volume move can leave a partial copy in the new directory.
        shutil.rmtree(directory)
        raise
    try:
        (folder / "trigger.json").write_bytes(metadata)
    except OSError as error:
        # Restore the LoRA before cleaning up. Leave it at the destination
        # if restoration fails, and tell the user where to find it.
        try:
            if os.path.lexists(source):
                raise FileExistsError(f"Source location is occupied: {source}")
            _move_file(destination, source)
        except OSError as restore_error:
            raise OSError(f"{error}. LoRA kept at {destination}. "
                          f"Could not restore it: {restore_error}") from restore_error
        shutil.rmtree(directory)
        raise
    return destination


class Gen2_LoRASave:
    CATEGORY = "Gen2/LoRA"
    FUNCTION = "save"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)

    @classmethod
    def INPUT_TYPES(cls):
        text = {"default": "", "multiline": False, "dynamicPrompts": False, "socketless": True}
        return {"required": {
            "file": (folder_paths.get_filename_list("loras"), {"socketless": True}),
            "trigger": ("STRING", text.copy()),
            "model": ("STRING", text.copy()),
            "type": ("STRING", text.copy()),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        # Check the selection at execution so a stale filename reaches the status output.
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save(self, file: str, trigger: str, model: str, type: str) -> tuple[str]:
        try:
            _check_folder_name(model, "model")
            _check_folder_name(type, "type")
            trigger = trigger.strip()
            if not trigger:
                raise ValueError("Enter a trigger.")
            metadata = json.dumps(trigger, ensure_ascii=False).encode("utf-8")
            if file not in folder_paths.get_filename_list("loras"):
                raise ValueError(f"LoRA is no longer available in the file picker: {file!r}.")
            source = Path(folder_paths.get_full_path_or_raise("loras", file))
            if source.name.lower() == "trigger.json":
                raise ValueError("The selected filename is reserved for the trigger.")

            destination = _organize(source, model, type, metadata)
        except (OSError, ValueError) as error:
            return (f"Failed: {error}",)
        return (f"Successfully created {destination.parent}. Moved {source.name} and saved trigger.json.",)
