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


if __name__ == "__main__":
    unittest.main()
