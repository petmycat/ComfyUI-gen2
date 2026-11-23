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


def _number_component(value: str) -> str:
    number = value.strip()
    if not (number.isascii() and number.isdecimal()):
        raise ValueError("number must be a string of digits, such as '3'.")
    return number


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


def _save_entry(source: Path, folder: Path, metadata: bytes) -> Path:
    destination = folder / source.name
    trigger_file = folder / "trigger.json"
    extensions = folder_paths.folder_names_and_paths["loras"][1]
    if os.path.lexists(destination) or any(
        (not extensions or p.suffix.lower() in extensions) and not p.is_dir()
        for p in folder.iterdir()
    ):
        raise FileExistsError(f"Destination is already occupied: {folder}")

    # Every entry has exactly one trigger. Claim it before moving anything so
    # two concurrent saves with different LoRA filenames cannot share a leaf.
    try:
        trigger_file.touch(exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"Destination already contains {trigger_file.name}: {folder}") from None
    try:
        _move_file(source, destination)
    except OSError:
        # A failed cross-volume copy may leave a partial destination file.
        destination.unlink(missing_ok=True)
        trigger_file.unlink()
        raise
    try:
        trigger_file.write_bytes(metadata)
    except OSError as error:
        try:
            if os.path.lexists(source):
                raise FileExistsError(f"Source location is occupied: {source}")
            _move_file(destination, source)
        except OSError as restore_error:
            raise OSError(f"{error}. LoRA kept at {destination}. "
                          f"Could not restore it: {restore_error}") from restore_error
        trigger_file.unlink()
        raise
    return destination


def _organize(source: Path, number: str | None, components: tuple[str, ...], metadata: bytes) -> Path:
    root = Path(folder_paths.models_dir) / "loras" / "gen2"
    created = []
    if number is None:
        directory = _next_directory(root)
        created.append(directory)
    else:
        directory = root / number
        if not directory.is_dir():
            raise FileNotFoundError(f"Number folder does not exist: {directory}")
        if directory.resolve() != root.resolve() / number:
            raise ValueError(f"Number folder points to a different location: {directory}")

    folder = directory
    try:
        for component in components:
            folder = folder / component
            try:
                folder.mkdir()
                created.append(folder)
            except FileExistsError:
                if not folder.is_dir() or not folder.resolve().is_relative_to(directory.resolve()):
                    raise ValueError(f"Expected a folder inside {directory}: {folder}") from None
        return _save_entry(source, folder, metadata)
    except (OSError, ValueError):
        # Other records may share any ancestor, even a newly allocated number.
        # Remove only directories this call created that are still empty.
        for path in reversed(created):
            try:
                path.rmdir()
            except OSError:
                pass
        raise


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
        }, "optional": {
            "new_number": ("BOOLEAN", {"default": True, "socketless": True,
                                       "tooltip": "Create the next numbered folder. Turn off to enter an existing number."}),
            "number": ("STRING", text.copy()),
            "use_name": ("BOOLEAN", {"default": False, "socketless": True,
                                     "tooltip": "Save the LoRA and trigger in a named subfolder of type."}),
            "name": ("STRING", text.copy()),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, file):
        # Check the selection at execution so a stale filename reaches the status output.
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save(self, file: str, trigger: str, model: str, type: str, new_number: bool = True,
             number: str = "", use_name: bool = False, name: str = "") -> tuple[str]:
        try:
            _check_folder_name(model, "model")
            _check_folder_name(type, "type")
            components = (model, type)
            if use_name:
                _check_folder_name(name, "name")
                components += (name,)
            selected_number = None if new_number else _number_component(number)
            trigger = trigger.strip()
            if not trigger:
                raise ValueError("Enter a trigger.")
            metadata = json.dumps(trigger, ensure_ascii=False).encode("utf-8")
            if file not in folder_paths.get_filename_list("loras"):
                raise ValueError(f"LoRA is no longer available in the file picker: {file!r}.")
            source = Path(folder_paths.get_full_path_or_raise("loras", file))
            if source.name.lower() == "trigger.json":
                raise ValueError("The selected filename is reserved for the trigger.")

            destination = _organize(source, selected_number, components, metadata)
        except (OSError, ValueError) as error:
            return (f"Failed: {error}",)
        return (f"Successfully created {destination.parent}. Moved {source.name} and saved trigger.json.",)
