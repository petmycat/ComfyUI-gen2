"""Filesystem and cache behavior of the Gen2 LoRA lookup node."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class LoRALoadTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        models = Path(temporary.name) / "models"
        self.loras = models / "loras"
        self.extra = Path(temporary.name) / "extra_loras"
        self.folder = self.loras / "gen2/3/q21/style"
        self.folder.mkdir(parents=True)
        self.model = self.folder / "sketch.pt"
        self.model.touch()
        self.trigger = self.folder / "trigger.json"
        self.trigger.write_text('"sketch"', encoding="utf-8")

        roots = (self.extra, self.loras)

        def resolve_lora(category, name):
            self.assertEqual(category, "loras")
            return str(next(root / name for root in roots if (root / name).is_file()))

        folders = SimpleNamespace(models_dir=str(models), folder_names_and_paths={"loras": (roots, {".pt"})},
                                  get_full_path_or_raise=resolve_lora)
        with patch.dict(sys.modules, {"folder_paths": folders}):
            from gen2.lora_nodes import lora_load
        lookup = patch.object(lora_load, "folder_paths", folders)
        lookup.start()
        self.addCleanup(lookup.stop)
        self.node = lora_load.Gen2_LoRALoad()

    def load(self):
        return self.node.load("3", "q21", "style")

    def test_only_models_in_the_selected_folder_count(self):
        for directory in (self.folder / "nested", self.folder.parent / "character"):
            directory.mkdir()
            (directory / "other.pt").touch()
        (self.folder / "notes.txt").touch()
        trigger = '  猫 "ink"  art  '
        self.trigger.write_text(json.dumps(trigger), encoding="utf-8")
        self.assertEqual(self.load(), (str(self.model.relative_to(self.loras)), trigger))
        self.model.unlink()
        with self.assertRaisesRegex(ValueError, "found 0"):
            self.load()
        self.model.touch()
        (self.folder / "second.pt").touch()
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.load()

    def test_refuses_a_higher_priority_file_with_the_same_name(self):
        before = self.node.IS_CHANGED(None, "q21", "style")
        shadow = self.extra / self.model.relative_to(self.loras)
        shadow.parent.mkdir(parents=True)
        shadow.touch()
        self.assertNotEqual(before, self.node.IS_CHANGED(None, "q21", "style"))
        with self.assertRaisesRegex(ValueError, "priority"):
            self.load()

    def test_rejects_paths_in_each_input(self):
        for inputs in (("../3", "q21", "style"), ("three", "q21", "style"),
                       ("3", "q21/../q21", "style"), ("3", "q21", "../q21/style")):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                self.node.load(*inputs)

    def test_missing_malformed_and_nonstring_trigger(self):
        self.trigger.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()
        for contents in ("sketch", '{"trigger":"sketch"}', "5"):
            self.trigger.write_text(contents, encoding="utf-8")
            with self.subTest(contents=contents), self.assertRaisesRegex(ValueError, "trigger.json"):
                self.load()

    def test_cache_fingerprint_tracks_trigger_edits_with_wired_number(self):
        original = self.node.IS_CHANGED(None, "q21", "style")
        self.assertEqual(original, self.node.IS_CHANGED(None, "q21", "style"))
        timestamp = self.trigger.stat().st_mtime_ns + 2_000_000_000
        self.trigger.write_text('"revised"', encoding="utf-8")
        os.utime(self.trigger, ns=(timestamp, timestamp))
        self.assertNotEqual(original, self.node.IS_CHANGED(None, "q21", "style"))


if __name__ == "__main__":
    unittest.main()
