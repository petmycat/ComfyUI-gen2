import importlib.util
import json
import unittest
from pathlib import Path

CONFIG_PATH = Path(__file__).parents[1] / "api_nodes" / "_config.py"
SPEC = importlib.util.spec_from_file_location("gen2_panel_config", CONFIG_PATH)
config = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(config)

MAX_PARAMS = config.MAX_PARAMS
SEED_MAX = config.SEED_MAX
parse_input_config = config.parse_input_config
parse_output_config = config.parse_output_config
schema_entries = config.schema_entries
validate_runtime_value = config.validate_runtime_value


class PanelConfigTests(unittest.TestCase):
    def test_migrates_legacy_input_and_applies_seed_defaults(self):
        params = parse_input_config(json.dumps([
            {"name": "prompt", "type": "STRING", "default": "cat"},
            {"name": "seed", "type": "SEED", "default": 7},
        ]))

        self.assertTrue(params[0]["id"].startswith("legacy-"))
        self.assertEqual(params[1]["min"], 0)
        self.assertEqual(params[1]["max"], SEED_MAX)
        self.assertEqual(params[1]["step"], 1)
        self.assertEqual(params[1]["controlMode"], "randomize")

    def test_output_contract_discards_legacy_input_metadata(self):
        params = parse_output_config([
            {"id": "image-id", "name": "image", "type": "IMAGE", "default": "ignored.png", "min": 1},
        ])
        self.assertEqual(params, [{"id": "image-id", "name": "image", "type": "IMAGE"}])
        self.assertEqual(schema_entries(params, "output"), params)

    def test_rejects_duplicate_names_and_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicated"):
            parse_output_config([
                {"id": "a", "name": "same", "type": "STRING"},
                {"id": "b", "name": "same", "type": "FLOAT"},
            ])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            parse_output_config([
                {"id": "same", "name": "a", "type": "STRING"},
                {"id": "same", "name": "b", "type": "FLOAT"},
            ])

    def test_rejects_invalid_numeric_contract(self):
        with self.assertRaisesRegex(ValueError, "step"):
            parse_input_config([
                {"name": "strength", "type": "FLOAT", "default": 0.5, "min": 0, "max": 1, "step": 0},
            ])
        with self.assertRaisesRegex(ValueError, "between"):
            parse_input_config([
                {"name": "steps", "type": "INT", "default": 30, "min": 1, "max": 20, "step": 1},
            ])

    def test_rejects_unknown_type_and_parameter_limit(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_output_config([{"name": "x", "type": "TENSOR"}])
        with self.assertRaisesRegex(ValueError, str(MAX_PARAMS)):
            parse_output_config([{"name": f"p{i}", "type": "STRING"} for i in range(MAX_PARAMS + 1)])

    def test_allows_empty_string_default_and_rejects_misaligned_numeric_default(self):
        params = parse_input_config([{"name": "prompt", "type": "STRING", "default": ""}])
        self.assertEqual(params[0]["default"], "")
        with self.assertRaisesRegex(ValueError, "step"):
            parse_input_config([{"name": "count", "type": "INT", "default": 3, "min": 0, "max": 10, "step": 2}])

    def test_runtime_validation_covers_all_types_and_step(self):
        cases = [
            ({"name": "text", "type": "STRING"}, ""),
            ({"name": "combo", "type": "COMBO"}, "option"),
            ({"name": "enabled", "type": "BOOLEAN"}, True),
            ({"name": "image", "type": "IMAGE"}, " folder/image.png "),
            ({"name": "count", "type": "INT", "min": 0, "max": 10, "step": 2}, 4),
            ({"name": "strength", "type": "FLOAT", "min": 0.0, "max": 1.0, "step": 0.1}, 0.3),
            ({"name": "seed", "type": "SEED", "min": 0, "max": SEED_MAX, "step": 1}, 12),
        ]
        results = [validate_runtime_value(param, value) for param, value in cases]
        self.assertEqual(results[0], "")
        self.assertEqual(results[3], "folder/image.png")
        with self.assertRaisesRegex(ValueError, "step"):
            validate_runtime_value({"name": "count", "type": "INT", "min": 0, "max": 10, "step": 2}, 3)
        with self.assertRaisesRegex(ValueError, "true or false"):
            validate_runtime_value({"name": "enabled", "type": "BOOLEAN"}, "false")

    def test_rejects_invalid_panel_mode(self):
        with self.assertRaisesRegex(ValueError, "mode"):
            config.parse_config([], "invalid")
        with self.assertRaisesRegex(ValueError, "mode"):
            config.schema_entries([], "invalid")


if __name__ == "__main__":
    unittest.main()
