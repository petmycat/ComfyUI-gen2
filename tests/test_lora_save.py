"""Exercise Gen2 LoRA Save using real files in isolated model folders."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from _sampling_support import load_implementation


class LoRASaveTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        models = Path(directory.name) / "models"
        self.loras = models / "loras"
        self.gen2 = self.loras / "gen2"
        self.loras.mkdir(parents=True)
        self.choices = []
        folders = SimpleNamespace(
            models_dir=str(models),
            folder_names_and_paths={"loras": ([str(self.loras)], {".pt", ".safetensors"})},
            get_filename_list=lambda category: self.choices.copy(),
            get_full_path_or_raise=self.resolve_choice,
        )
        module = load_implementation(
            "gen2/lora_nodes/lora_save.py", "_gen2_test_lora_save", {"folder_paths": folders},
        )
        self.node = module.Gen2_LoRASave()

    def resolve_choice(self, category, choice):
        self.assertEqual(category, "loras")
        path = self.loras / choice
        if not path.is_file():
            raise FileNotFoundError(choice)
        return str(path)

    def create_source(self, choice):
        path = self.loras / choice
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(choice.encode())
        self.choices.append(choice)
        return path

    def test_moves_nested_file_and_writes_one_json_string(self):
        choice = "incoming/cartoon.safetensors"
        source = self.create_source(choice)
        trigger = '猫,  "watercolor"'
        message, = self.node.save(choice, f"   {trigger}  ", "Q 21", "Style")
        self.assertTrue(message.startswith("Successfully created "), message)
        output = self.gen2 / "1" / "Q 21" / "Style"
        self.assertFalse(source.exists())
        self.assertEqual((output / source.name).read_bytes(), choice.encode())
        raw = (output / "trigger.json").read_bytes().decode("utf-8")
        self.assertEqual(raw.splitlines(), [raw])
        self.assertEqual(raw, raw.strip())
        self.assertEqual(json.loads(raw), trigger)

    def test_sequence_continues_past_highest_directory_and_skips_occupied_id(self):
        for name in ("1", "7", "notes"):
            (self.gen2 / name).mkdir(parents=True)
        for name in ("8", "50"):
            (self.gen2 / name).write_text("existing")
        self.create_source("model.pt")
        self.node.save("model.pt", "art", "q21", "style")
        self.assertTrue((self.gen2 / "9/q21/style/model.pt").is_file())
        self.assertEqual((self.gen2 / "8").read_text(), "existing")

    def test_invalid_folder_component_leaves_source_in_place(self):
        source = self.create_source("model.pt")
        for model, kind in (("", "style"), ("..", "style"), ("q21", "a/b"), ("q21", r"..\outside")):
            with self.subTest(model=model, kind=kind):
                message, = self.node.save(source.name, "art", model, kind)
                self.assertTrue(message.startswith("Failed: "), message)
        self.assertEqual(source.read_bytes(), b"model.pt")
        self.assertFalse(self.gen2.exists() and any(self.gen2.iterdir()))

    def test_stale_selection_does_not_consume_another_number(self):
        source = self.create_source("model.pt")
        self.node.save(source.name, "art", "q21", "style")
        message, = self.node.save(source.name, "art", "q21", "style")
        self.assertTrue(message.startswith("Failed: "), message)
        self.assertEqual([p.name for p in self.gen2.iterdir()], ["1"])

    def test_metadata_write_failure_restores_the_lora_and_releases_the_number(self):
        source = self.create_source("model.pt")
        with patch.object(Path, "write_bytes", side_effect=OSError("disk full")):
            message, = self.node.save(source.name, "art", "q21", "style")
        self.assertEqual(message, "Failed: disk full")
        self.assertEqual(source.read_bytes(), b"model.pt")
        self.assertEqual(list(self.gen2.iterdir()), [])

    def test_metadata_write_failure_keeps_both_files_if_source_was_replaced(self):
        source = self.create_source("model.pt")
        destination = self.gen2 / "1/q21/style" / source.name

        def replace_source_then_fail(*args):
            source.write_text("replacement")
            raise OSError("disk full")

        with patch.object(Path, "write_bytes", side_effect=replace_source_then_fail):
            message, = self.node.save(source.name, "art", "q21", "style")
        self.assertTrue(message.startswith("Failed: "), message)
        self.assertIn(str(destination), message)
        self.assertEqual(destination.read_bytes(), b"model.pt")
        self.assertEqual(source.read_text(), "replacement")

    def test_concurrent_saves_allocate_distinct_directories(self):
        choices = [f"adapter-{i}.pt" for i in range(4)]
        for name in choices:
            self.create_source(name)
        barrier = Barrier(len(choices))

        def save(name):
            barrier.wait(timeout=10)
            self.node.save(name, name, "q21", "style")

        with ThreadPoolExecutor(max_workers=len(choices)) as pool:
            list(pool.map(save, choices))
        self.assertEqual({p.name for p in self.gen2.iterdir()}, {"1", "2", "3", "4"})
        for index in self.gen2.iterdir():
            destination = index / "q21/style"
            name = json.loads((destination / "trigger.json").read_text(encoding="utf-8"))
            self.assertEqual((destination / name).read_bytes(), name.encode())
            self.assertFalse((self.loras / name).exists())

    def test_existing_number_supports_models_named_children_and_an_entry_at_the_type(self):
        def save(filename, model, **options):
            self.create_source(filename)
            message, = self.node.save(filename, filename, model, "characters", **options)
            self.assertTrue(message.startswith("Successfully created "), message)

        save("mouse.pt", "q21", use_name=True, name="Mouse")
        save("cat.pt", "q21", new_number=False, number="1", use_name=True, name="Cat")
        save("group.pt", "q21", new_number=False, number="1", name="../ignored")
        save("another.pt", "k2", new_number=False, number="1")
        self.assertEqual([p.name for p in self.gen2.iterdir()], ["1"])
        paths = {"q21/characters/Mouse": "mouse.pt", "q21/characters/Cat": "cat.pt",
                 "q21/characters": "group.pt", "k2/characters": "another.pt"}
        for path, expected in paths.items():
            folder = self.gen2 / "1" / path
            self.assertEqual((folder / expected).read_bytes(), expected.encode())
            self.assertEqual(json.loads((folder / "trigger.json").read_bytes()), expected)

    def test_manual_number_and_enabled_name_must_be_valid_before_move(self):
        source = self.create_source("model.pt")
        for options in ({"new_number": False}, {"new_number": False, "number": "../1"},
                        {"new_number": False, "number": "3"}, {"use_name": True},
                        {"use_name": True, "name": "../other"}):
            with self.subTest(options=options):
                message, = self.node.save(source.name, "art", "q21", "style", **options)
                self.assertTrue(message.startswith("Failed: "), message)
                self.assertTrue(source.is_file())
                self.assertFalse(self.gen2.exists())

    def test_manual_save_preserves_existing_files_even_in_partial_entries(self):
        incoming = self.create_source("incoming.pt")
        for occupied in ("incoming.pt", "another.safetensors", "trigger.json"):
            folder = self.gen2 / "3/q21/style" / occupied
            folder.mkdir(parents=True)
            (folder / occupied).write_bytes(b"keep")
            with self.subTest(occupied=occupied):
                message, = self.node.save(incoming.name, "art", "q21", "style",
                                          new_number=False, number="3", use_name=True, name=occupied)
                self.assertTrue(message.startswith("Failed: "), message)
                self.assertEqual((folder / occupied).read_bytes(), b"keep")
                self.assertEqual([p.name for p in folder.iterdir()], [occupied])
                self.assertTrue(incoming.is_file())

    def test_manual_save_failure_preserves_siblings_and_existing_parents(self):
        source = self.create_source("model.pt")
        kind = self.gen2 / "7/q21/style"
        sibling = kind / "existing"
        sibling.mkdir(parents=True)
        (sibling / "trigger.json").write_bytes(b'"old"')
        with patch.object(Path, "write_bytes", side_effect=OSError("disk full")):
            message, = self.node.save(source.name, "art", "q21", "style",
                                      new_number=False, number="7", use_name=True, name="new")
        self.assertEqual(message, "Failed: disk full")
        self.assertEqual(source.read_bytes(), b"model.pt")
        self.assertEqual(list(kind.iterdir()), [sibling])
        self.assertEqual((sibling / "trigger.json").read_bytes(), b'"old"')

    def test_linked_subfolder_cannot_redirect_a_manual_save_outside_the_number(self):
        source = self.create_source("model.pt")
        directory = self.gen2 / "3"
        directory.mkdir(parents=True)
        outside = self.loras / "outside"
        outside.mkdir()
        try:
            (directory / "q21").symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Directory symlinks unavailable: {error}")
        message, = self.node.save(source.name, "art", "q21", "style", new_number=False, number="3")
        self.assertTrue(message.startswith("Failed: "), message)
        self.assertTrue(source.is_file())
        self.assertEqual(list(outside.iterdir()), [])

    def test_same_destination_concurrent_saves_create_one_matching_pair(self):
        choices = ["first.pt", "second.pt"]
        (self.gen2 / "3").mkdir(parents=True)
        for choice in choices:
            self.create_source(choice)
        barrier = Barrier(len(choices))

        def save(choice):
            barrier.wait(timeout=10)
            message, = self.node.save(choice, choice, "q21", "style", new_number=False, number="3")
            return message

        with ThreadPoolExecutor(max_workers=len(choices)) as pool:
            results = dict(zip(choices, pool.map(save, choices)))
        winner, = (name for name in choices if results[name].startswith("Successfully created "))
        loser, = (name for name in choices if results[name].startswith("Failed: "))
        folder = self.gen2 / "3/q21/style"
        self.assertEqual({p.name for p in folder.iterdir()}, {"trigger.json", winner})
        self.assertEqual(json.loads((folder / "trigger.json").read_bytes()), winner)
        self.assertFalse((self.loras / winner).exists())
        self.assertTrue((self.loras / loser).is_file())


if __name__ == "__main__":
    unittest.main()
