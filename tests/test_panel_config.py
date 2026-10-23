"""Validation must agree with JSON clients without silently coercing bad calls."""

import math
import unittest

from _panel_support import panels


config = panels._config


class PanelConfigTests(unittest.TestCase):
    def test_all_types_normalize_and_roundtrip(self):
        entries = [
            {"name": "text", "type": "STRING", "default": "猫"},
            {"name": "choice", "type": "COMBO", "default": "value"},
            {"name": "enabled", "type": "BOOLEAN", "default": False},
            {"name": "image", "type": "IMAGE", "default": " a.png [input] "},
            {"name": "count", "type": "INT", "default": 4, "min": 0, "max": 8, "step": 2},
            {"name": "weight", "type": "FLOAT", "default": 0.3, "min": 0, "max": 1, "step": 0.1},
            {"name": "seed", "type": "SEED", "default": 0},
        ]
        result = config.parse_input_config(entries)
        self.assertEqual(config.parse_input_config(config.build_schema_json(result)), result)
        self.assertEqual(result[3]["default"], "a.png [input]")
        self.assertEqual(result[-1]["max"], 2**53 - 1)
        self.assertEqual(result[-1]["controlMode"], "randomize")

    def test_malformed_configs_fail_with_value_error(self):
        invalid = ["{", "{}", "null", [None], [{"name": "n", "default": None}],
                   [{"name": "", "default": ""}], [{"name": 4, "default": ""}],
                   [{"name": "n", "id": 3, "default": ""}],
                   [{"name": "n", "type": "MODEL", "default": ""}],
                   [{"name": str(i), "default": ""} for i in range(33)]]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                config.parse_input_config(raw)

    def test_reserved_names_and_ids_match_frontend(self):
        for parser in (config.parse_input_config, config.parse_output_config):
            for name in ("_config", "PANEL_LINK", "param_0", "param_100", "__proto__", "constructor", "prototype"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "reserved"):
                    parser([{"name": name, "default": ""}])
            for identity in ("__proto__", "constructor", "prototype"):
                with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, "reserved"):
                    parser([{"name": "safe", "id": identity, "default": ""}])

    def test_duplicate_names_and_ids_fail(self):
        for entries in ([{"name": "a", "default": ""}, {"name": " a ", "default": ""}],
                        [{"name": "a", "id": "same", "default": ""}, {"name": "b", "id": "same", "default": ""}]):
            with self.assertRaisesRegex(ValueError, "duplicated"):
                config.parse_input_config(entries)

    def test_legacy_numeric_config_normalizes_but_runtime_does_not(self):
        param = config.parse_input_config([{"name": "count", "type": "INT", "default": "4", "min": "0", "max": "8", "step": "2"}])[0]
        self.assertEqual(param["default"], 4)
        for value in ("4", 4.0, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config.validate_runtime_value(param, value)

    def test_numeric_constraints_and_step(self):
        for changes in ({"min": 10}, {"max": 1}, {"step": 0}, {"step": -1}, {"default": 3}, {"default": 2.5}, {"default": float("inf")}):
            entry = {"name": "n", "type": "INT", "default": 2, "min": 0, "max": 8, "step": 2, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config.parse_input_config([entry])
        param = config.parse_input_config([{"name": "f", "type": "FLOAT", "default": 0.3, "min": 0, "max": 1, "step": 0.1}])[0]
        self.assertAlmostEqual(config.validate_runtime_value(param, 0.1 + 0.2), 0.3)
        for value in (0.35, -0.1, 1.1, True, "0.3", float("nan"), float("inf"), 10**400):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config.validate_runtime_value(param, value)

    def test_seed_range_and_modes(self):
        for entry in ({"default": -1}, {"default": 2**53}, {"max": 2**64-1}, {"controlMode": "invalid"}):
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                config.parse_input_config([{"name": "seed", "type": "SEED", **entry}])
        for value in (-1, 2**53, True):
            with self.assertRaises(ValueError):
                config.validate_output_value({"name": "seed", "type": "SEED"}, value)

    def test_scalar_type_validation_and_optional_outputs(self):
        for ptype, bad in (("BOOLEAN", "false"), ("STRING", 42), ("COMBO", ["x"]), ("IMAGE", None)):
            with self.subTest(ptype=ptype), self.assertRaises(ValueError):
                config.validate_runtime_value({"name": "value", "type": ptype}, bad)
        for ptype in config.SUPPORTED_TYPES:
            self.assertIsNone(config.validate_output_value({"name": "optional", "type": ptype}, None))
        with self.assertRaisesRegex(ValueError, "tensors"):
            config.validate_output_value({"name": "image", "type": "IMAGE"}, "file.png")

    def test_schema_only_outputs_drop_input_metadata(self):
        parsed = config.parse_output_config([{"name": "result", "type": "FLOAT", "id": "stable", "default": "old", "min": "old"}])
        self.assertEqual(parsed, [{"id": "stable", "name": "result", "type": "FLOAT"}])


if __name__ == "__main__":
    unittest.main()
