import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE = types.ModuleType("gen2_contract_test")
PACKAGE.__path__ = [str(ROOT / "api_nodes")]
sys.modules[PACKAGE.__name__] = PACKAGE

for module_name in ("_config", "workflow_contract"):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE.__name__}.{module_name}", ROOT / "api_nodes" / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

contract = sys.modules[f"{PACKAGE.__name__}.workflow_contract"]


class WorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = json.loads((ROOT / "assets" / "test1.json").read_text(encoding="utf-8"))
        cls.api_prompt = json.loads((ROOT / "assets" / "test1_api.json").read_text(encoding="utf-8"))

    def test_discovers_same_panel_contract_from_both_exports(self):
        workflow_manifest = contract.discover_manifest(self.workflow)
        api_manifest = contract.discover_manifest(self.api_prompt)

        self.assertEqual(workflow_manifest["source_format"], "workflow")
        self.assertEqual(api_manifest["source_format"], "api_prompt")
        self.assertEqual(
            [(p["id"], p["name"], p["type"]) for p in workflow_manifest["input_panels"][0]["parameters"]],
            [(p["id"], p["name"], p["type"]) for p in api_manifest["input_panels"][0]["parameters"]],
        )
        self.assertFalse(workflow_manifest["input_panels"][0]["parameters"][0]["binding"]["patchable"])
        self.assertTrue(api_manifest["input_panels"][0]["parameters"][0]["binding"]["patchable"])
        self.assertEqual(api_manifest["output_panels"][0]["paired_input_node_id"], "1")

    def test_validates_and_patches_api_prompt_without_mutating_source(self):
        manifest = contract.discover_manifest(self.api_prompt)
        patched = contract.patch_api_prompt(
            self.api_prompt,
            manifest,
            {"image1Url": "requests/example.png", "seed": 42},
        )

        self.assertEqual(patched["1"]["inputs"]["image1Url"], "requests/example.png")
        self.assertEqual(patched["1"]["inputs"]["seed"], 42)
        self.assertEqual(self.api_prompt["1"]["inputs"]["seed"], 0)
        self.assertEqual(patched["1"]["inputs"]["_config"], self.api_prompt["1"]["inputs"]["_config"])

    def test_rejects_wrong_runtime_types_ranges_and_unknown_inputs(self):
        manifest = contract.discover_manifest(self.api_prompt)
        with self.assertRaisesRegex(ValueError, "integer"):
            contract.validate_call_inputs(manifest, {"image1Url": "a.png", "seed": True})
        with self.assertRaisesRegex(ValueError, "above max"):
            contract.validate_call_inputs(manifest, {"image1Url": "a.png", "seed": 9007199254740992})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            contract.validate_call_inputs(manifest, {"image1Url": "a.png", "seed": 1, "extra": 2})

    def test_extracts_latest_output_document_from_history(self):
        manifest = contract.discover_manifest(self.api_prompt)
        first = {"version": 1, "inputs": {"schema": [], "latest_values": {}}, "outputs": {"schema": [], "latest_values": {"imageOutput1Url": "first"}}}
        latest = {"version": 1, "inputs": {"schema": [], "latest_values": {}}, "outputs": {"schema": [], "latest_values": {"imageOutput1Url": "latest"}}}
        history = {
            "prompt-1": {
                "outputs": {
                    "2": {
                        "document": [first, latest],
                        "images": [{"filename": "result.png", "subfolder": "", "type": "output"}],
                    }
                }
            }
        }

        result = contract.extract_history_results(history, manifest, prompt_id="prompt-1")
        self.assertEqual(result["panels"]["2"]["latest"], latest)
        self.assertEqual(len(result["panels"]["2"]["runs"]), 2)
        self.assertEqual(result["panels"]["2"]["images"][0]["filename"], "result.png")

    def test_omitted_inputs_keep_api_export_current_values(self):
        prompt = json.loads(json.dumps(self.api_prompt))
        prompt["1"]["inputs"]["seed"] = 123
        manifest = contract.discover_manifest(prompt)
        patched = contract.patch_api_prompt(prompt, manifest, {"image1Url": "replacement.png"})
        self.assertEqual(patched["1"]["inputs"]["seed"], 123)

    def test_integer_node_ids_discover_pair_and_patch(self):
        prompt = {
            1: json.loads(json.dumps(self.api_prompt["1"])),
            2: json.loads(json.dumps(self.api_prompt["2"])),
        }
        prompt[2]["inputs"]["PANEL_LINK"] = [1, 0]
        prompt[2]["inputs"]["param_0"] = [1, 1]
        manifest = contract.discover_manifest(prompt)
        self.assertEqual(manifest["output_panels"][0]["paired_input_node_id"], "1")
        patched = contract.patch_api_prompt(prompt, manifest, {"image1Url": "integer.png", "seed": 7})
        self.assertEqual(patched[1]["inputs"]["image1Url"], "integer.png")
        self.assertEqual(patched[1]["inputs"]["seed"], 7)

    def test_rejects_invalid_panel_link_and_stale_manifest(self):
        prompt = json.loads(json.dumps(self.api_prompt))
        prompt["2"]["inputs"]["PANEL_LINK"] = ["1", 1]
        with self.assertRaisesRegex(ValueError, "invalid PANEL_LINK"):
            contract.discover_manifest(prompt)

        manifest = contract.discover_manifest(self.api_prompt)
        stale_prompt = json.loads(json.dumps(self.api_prompt))
        config = json.loads(stale_prompt["1"]["inputs"]["_config"])
        config[1]["controlMode"] = "increment"
        stale_prompt["1"]["inputs"]["_config"] = json.dumps(config)
        with self.assertRaisesRegex(ValueError, "panel contract"):
            contract.patch_api_prompt(stale_prompt, manifest, {})

    def test_history_falls_back_after_empty_or_invalid_document(self):
        manifest = contract.discover_manifest(self.api_prompt)
        latest = {"version": 1, "inputs": {"schema": [], "latest_values": {}}, "outputs": {"schema": [], "latest_values": {"imageOutput1Url": "fallback"}}}
        history = {"outputs": {"2": {"document": [], "document_json": ["invalid", json.dumps(latest)]}}}
        result = contract.extract_history_results(history, manifest)
        self.assertEqual(result["panels"]["2"]["latest"], latest)

    def test_history_can_build_legacy_document_from_params(self):
        manifest = contract.discover_manifest(self.api_prompt)
        history = {"outputs": {"2": {"params": [{"imageOutput1Url": "legacy.png"}], "images": []}}}
        result = contract.extract_history_results(history, manifest)
        self.assertEqual(
            result["panels"]["2"]["latest"]["outputs"]["latest_values"]["imageOutput1Url"],
            "legacy.png",
        )

    def test_duplicate_names_allow_empty_overrides_and_require_scoped_values(self):
        prompt = json.loads(json.dumps(self.api_prompt))
        second = json.loads(json.dumps(prompt["1"]))
        second_config = json.loads(second["inputs"]["_config"])
        second_config[0]["id"] = "other-image-id"
        second_config[1]["id"] = "other-seed-id"
        second["inputs"]["_config"] = json.dumps(second_config)
        prompt["3"] = second
        manifest = contract.discover_manifest(prompt)
        unchanged = contract.validate_call_inputs(manifest, {})
        self.assertEqual(unchanged["1"]["seed"], prompt["1"]["inputs"]["seed"])
        self.assertEqual(unchanged["3"]["seed"], prompt["3"]["inputs"]["seed"])
        with self.assertRaisesRegex(ValueError, "panel-scoped"):
            contract.validate_call_inputs(manifest, {"seed": 5, "image1Url": "x.png"})
        values = contract.validate_call_inputs(manifest, {
            "1": {"seed": 5},
            "3": {"image1Url": "second.png"},
        })
        self.assertEqual(values["1"]["seed"], 5)
        self.assertEqual(values["3"]["image1Url"], "second.png")

    def test_normal_workflow_cannot_be_patched_for_execution(self):
        manifest = contract.discover_manifest(self.workflow)
        with self.assertRaisesRegex(ValueError, "API prompt"):
            contract.patch_api_prompt(self.workflow, manifest, {})


if __name__ == "__main__":
    unittest.main()
