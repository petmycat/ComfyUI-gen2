"""Panel execution, JSON/history compatibility, images, and cache behavior."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from _panel_support import panels, load_panels


Input = panels.input_panel.Gen2InputPanel
Output = panels.output_panel.Gen2OutputPanel
EMPTY_LINK = {"version": 1, "inputs": {"schema": [], "latest_values": {}}}


def scalar_config():
    return [
        {"id": "text-id", "name": "text", "type": "STRING", "default": "猫"},
        {"id": "count-id", "name": "count", "type": "INT", "default": 4, "min": 0, "max": 10, "step": 2},
        {"id": "weight-id", "name": "weight", "type": "FLOAT", "default": 0.3, "min": 0, "max": 1, "step": 0.1},
        {"id": "enabled-id", "name": "enabled", "type": "BOOLEAN", "default": False},
        {"id": "choice-id", "name": "choice", "type": "COMBO", "default": "euler"},
        {"id": "seed-id", "name": "seed", "type": "SEED", "default": 7},
    ]


class PanelExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gen2-panel-tests-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.patches = [
            patch.object(panels.folders, "get_annotated_filepath", side_effect=lambda value: str(self.directory / value.removesuffix(" [input]"))),
            patch.object(panels.folders, "get_output_directory", return_value=str(self.directory)),
            patch.object(panels.folders, "get_save_image_path", create=True,
                         side_effect=lambda prefix, directory, width, height: (directory, prefix, 1, "", prefix)),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_schema_keeps_node_ids_and_fixed_slot_bank(self):
        schema = Input.define_schema()
        self.assertEqual(schema.node_id, "Gen2_InputPanel")
        self.assertEqual([port.id for port in schema.outputs], ["PANEL_LINK"] + [f"param_{i}" for i in range(32)])
        output = Output.define_schema()
        self.assertTrue(output.is_output_node)
        self.assertTrue(output.accept_all_inputs)
        self.assertEqual(output.node_id, "Gen2_OutputPanel")
        self.assertEqual([port.id for port in output.inputs], ["_config", "PANEL_LINK"] + [f"param_{i}" for i in range(32)])

    def test_scalar_roundtrip_preserves_version_one_and_history_fields(self):
        config = scalar_config()
        result = Input.execute(json.dumps(config), text="hello", count=6, enabled=True)
        self.assertEqual(len(result.args), 33)
        self.assertEqual(result.args[1:7], ("hello", 6, 0.3, True, "euler", 7))
        self.assertEqual(result.args[7:], (None,) * 26)
        self.assertEqual(result.ui, {"gen2_input_executed": [True]})
        outputs = [{"id": p["id"], "name": p["name"], "type": p["type"]} for p in config]
        payload = Output.execute(json.dumps(outputs), PANEL_LINK=result.args[0], **{f"param_{i}": result.args[i + 1] for i in range(6)}).ui
        document = payload["document"][0]
        self.assertEqual(set(payload), {"images", "params", "schema_json", "document", "document_json"})
        self.assertTrue(all(isinstance(value, list) for value in payload.values()))
        self.assertEqual(document["version"], 1)
        self.assertEqual(document["inputs"], result.args[0]["inputs"])
        self.assertEqual(document["outputs"]["schema"], outputs)
        self.assertEqual(document["outputs"]["latest_values"], document["inputs"]["latest_values"])
        self.assertEqual(json.loads(payload["document_json"][0]), document)
        self.assertEqual(payload["schema_json"], payload["document_json"])
        self.assertEqual(payload["params"], [document["outputs"]["latest_values"]])

    def test_input_rejects_lossy_or_ambiguous_values(self):
        for name, value in (("count", 4.9), ("count", True), ("count", "4"), ("count", 3),
                            ("enabled", "false"), ("text", 9), ("choice", ["euler"]),
                            ("seed", 2**53), ("weight", float("nan")), ("unknown", 1)):
            with self.subTest(name=name, value=value):
                self.assertIsInstance(Input.validate_inputs(scalar_config(), **{name: value}), str)
                with self.assertRaises(ValueError):
                    Input.execute(scalar_config(), **{name: value})

    def test_pending_links_are_checked_at_execution(self):
        self.assertIs(Input.validate_inputs(scalar_config(), text=None, count=None), True)
        with self.assertRaises(ValueError):
            Input.execute(scalar_config(), text=None)
        self.assertIsInstance(Input.validate_inputs("["), str)
        self.assertIsInstance(Output.validate_inputs("["), str)

    def test_image_reference_load_and_same_filename_cache_invalidation(self):
        path = self.directory / "image.png"
        Image.new("RGB", (4, 3), "red").save(path)
        config = [{"id": "source", "name": "image", "type": "IMAGE", "default": "image.png [input]"}]
        self.assertIs(Input.validate_inputs(config), True)
        first_fingerprint = Input.fingerprint_inputs(config)
        first = Input.execute(config)
        self.assertEqual(tuple(first.args[1].shape), (1, 3, 4, 3))
        self.assertEqual(first.args[1].dtype, torch.float32)
        self.assertEqual(first.args[0]["inputs"]["latest_values"]["image"], "image.png [input]")
        Image.new("RGB", (4, 3), "blue").save(path)
        self.assertNotEqual(first_fingerprint, Input.fingerprint_inputs(config))
        self.assertFalse(torch.equal(first.args[1], Input.execute(config).args[1]))
        self.assertTrue(math.isnan(Input.fingerprint_inputs(config, image=None)))
        path.unlink()
        self.assertIsInstance(Input.validate_inputs(config), str)
        self.assertTrue(math.isnan(Input.fingerprint_inputs(config)))

    def test_image_input_rejects_tensor_null_and_empty_refs(self):
        config = [{"name": "image", "type": "IMAGE", "default": "image.png"}]
        for value in (None, "", "  ", torch.zeros(1, 2, 2, 3), {"filename": "image.png"}):
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                Input.execute(config, image=value)

    def test_optional_outputs_and_legacy_named_fallback(self):
        config = [{"name": "result", "type": "INT"}, {"name": "preview", "type": "IMAGE"}]
        result = Output.execute(config, PANEL_LINK={"params": []}, result=5).ui["document"][0]
        self.assertEqual(result["outputs"]["latest_values"], {"result": 5, "preview": None})
        self.assertEqual(result["inputs"], {"schema": [], "latest_values": {}})
        positional = Output.execute(config, PANEL_LINK=EMPTY_LINK, result=5, param_0=8).ui
        self.assertEqual(positional["params"][0]["result"], 8)

    def test_output_declared_types_and_unused_slots_are_validated(self):
        for ptype, value in (("INT", 1.5), ("INT", True), ("FLOAT", float("inf")),
                             ("BOOLEAN", "false"), ("STRING", {}), ("SEED", -1), ("COMBO", 3)):
            config = [{"name": "result", "type": ptype}]
            with self.subTest(ptype=ptype):
                self.assertIsInstance(Output.validate_inputs(config, param_0=value), str)
                with self.assertRaises(ValueError):
                    Output.execute(config, PANEL_LINK=EMPTY_LINK, param_0=value)
        with self.assertRaisesRegex(ValueError, "not configured"):
            Output.execute([], PANEL_LINK=EMPTY_LINK, param_5=3)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            Output.execute([], PANEL_LINK=EMPTY_LINK, typo=True)

    def test_numpy_scalars_are_json_values(self):
        result = Output.execute([{"name": "result", "type": "INT"}], PANEL_LINK=EMPTY_LINK, param_0=np.int64(3)).ui
        self.assertEqual(json.loads(result["document_json"][0])["outputs"]["latest_values"], {"result": 3})

    def test_image_save_batch_channels_grad_and_bfloat16(self):
        for channels in (1, 3, 4):
            with self.subTest(channels=channels):
                config = [{"name": f"preview{channels}", "type": "IMAGE"}]
                tensor = torch.ones(2, 3, 4, channels, dtype=torch.bfloat16, requires_grad=True)
                payload = Output.execute(config, PANEL_LINK=EMPTY_LINK, param_0=tensor).ui
                self.assertEqual(len(payload["images"]), 2)
                self.assertEqual(payload["params"][0][f"preview{channels}"][0]["type"], "output")
                with Image.open(self.directory / payload["images"][0]["filename"]) as image:
                    self.assertEqual(image.size, (4, 3))
                    self.assertEqual(image.mode, {1: "L", 3: "RGB", 4: "RGBA"}[channels])

    def test_invalid_image_shape_dtype_and_pixels_fail_before_saving(self):
        config = [{"name": "preview", "type": "IMAGE"}]
        for tensor in ("file.png", torch.zeros(3, 4, 3), torch.zeros(0, 3, 4, 3),
                       torch.zeros(1, 3, 4, 2), torch.zeros(1, 3, 4, 3, dtype=torch.uint8),
                       torch.full((1, 3, 4, 3), float("nan"))):
            with self.subTest(shape=getattr(tensor, "shape", None)), self.assertRaises(ValueError):
                Output.execute(config, PANEL_LINK=EMPTY_LINK, param_0=tensor)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_bad_later_scalar_does_not_save_earlier_image(self):
        config = [{"name": "preview", "type": "IMAGE"}, {"name": "count", "type": "INT"}]
        with self.assertRaises(ValueError):
            Output.execute(config, PANEL_LINK=EMPTY_LINK, param_0=torch.ones(1, 3, 4, 3), param_1=1.5)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_bad_panel_link_cannot_publish_partial_documents_or_images(self):
        config = [{"name": "preview", "type": "IMAGE"}]
        bad_links = [None, {}, {"version": 2, "inputs": {}}, {"inputs": {"schema": [], "latest_values": {"extra": 1}}},
                     {"inputs": {"schema": [{"name": "x", "type": "STRING", "default": "", "extra": float("nan")}], "latest_values": {"x": ""}}}]
        for link in bad_links:
            with self.subTest(link=link), self.assertRaises(ValueError):
                Output.execute(config, PANEL_LINK=link, param_0=torch.ones(1, 3, 4, 3))
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_output_names_remain_json_keys_without_becoming_paths(self):
        name = "../../folder/猫%width%"
        result = Output.execute([{"name": name, "type": "IMAGE"}], PANEL_LINK=EMPTY_LINK, param_0=torch.ones(1, 3, 4, 3)).ui
        ref = result["params"][0][name][0]
        self.assertNotIn("/", ref["filename"])
        self.assertNotIn("%", ref["filename"])
        self.assertTrue((self.directory / ref["filename"]).is_file())

    def test_import_stubs_do_not_leak(self):
        before = {name: sys.modules.get(name) for name in ("comfy_api", "comfy_api.latest", "folder_paths", "node_helpers")}
        load_panels()
        self.assertEqual(before, {name: sys.modules.get(name) for name in before})


if __name__ == "__main__":
    unittest.main()
