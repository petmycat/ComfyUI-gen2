"""Pure-JSON endpoint contracts; no ComfyUI runtime or fixture assets required."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


def _load_contract_module():
    directory = Path(__file__).resolve().parents[1] / "gen2" / "api_panel_nodes"
    package_name = "_gen2_workflow_contract_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(directory)]
    # Import only the pure helpers. Restore sys.modules immediately so these
    # tests cannot replace real ComfyUI modules or pollute other test loaders.
    with patch.dict(sys.modules, {package_name: package}):
        for name in ("_config", "workflow_contract"):
            full_name = f"{package_name}.{name}"
            spec = importlib.util.spec_from_file_location(full_name, directory / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
        return module


contract = _load_contract_module()


def api_fixture():
    return {
        "1": {
            "class_type": "Gen2_InputPanel",
            "inputs": {
                "_config": json.dumps([
                    {"id": "prompt-id", "name": "prompt", "type": "STRING", "default": "default prompt"},
                    {"id": "steps-id", "name": "steps", "type": "INT", "default": 10, "min": 1, "max": 20, "step": 1},
                ]),
                "prompt": "exported prompt",
                "steps": 12,
            },
        },
        "2": {
            "class_type": "Gen2_InputPanel",
            "inputs": {
                "_config": json.dumps([
                    {"id": "strength-id", "name": "strength", "type": "FLOAT", "default": 0.5, "min": 0, "max": 1, "step": 0.1},
                ]),
                "strength": 0.6,
            },
        },
        "3": {
            "class_type": "Gen2_OutputPanel",
            "inputs": {
                "_config": json.dumps([
                    {"id": "result-id", "name": "result", "type": "STRING"},
                    {"id": "image-id", "name": "preview", "type": "IMAGE"},
                ]),
                "PANEL_LINK": ["1", 0],
                "param_0": ["4", 0],
                "param_1": ["4", 1],
            },
        },
        "4": {"class_type": "ExampleWorker", "inputs": {"text": ["1", 1], "steps": ["1", 2]}},
    }


def workflow_fixture():
    prompt = api_fixture()
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1, "type": "Gen2_InputPanel", "inputs": [],
                "widgets_values": [prompt["1"]["inputs"]["_config"]],
                "properties": {"gen2RuntimeValues": {"prompt-id": "exported prompt", "steps-id": 12}},
            },
            {
                "id": 2, "type": "Gen2_InputPanel", "inputs": [],
                "widgets_values": [prompt["2"]["inputs"]["_config"]],
                "properties": {"gen2RuntimeValues": {"strength-id": 0.6}},
            },
            {
                "id": 3, "type": "Gen2_OutputPanel",
                "inputs": [{"name": "PANEL_LINK", "link": 10}, {"name": "param_0", "link": 11}],
                "widgets_values": [prompt["3"]["inputs"]["_config"]],
            },
            {"id": 4, "type": "ExampleWorker"},
        ],
        "links": [[10, 1, 0, 3, 0, "*"], [11, 4, 0, 3, 1, "STRING"]],
    }


def result_document(value):
    return {
        "version": 1,
        "inputs": {"schema": [], "latest_values": {"prompt": "exported prompt"}},
        "outputs": {"schema": [{"id": "result-id", "name": "result", "type": "STRING"}], "latest_values": {"result": value}},
    }


class ManifestTests(unittest.TestCase):
    def test_api_manifest_preserves_version_identity_and_positional_output_slots(self):
        source = api_fixture()
        original = copy.deepcopy(source)
        manifest = contract.discover_manifest(source)
        self.assertEqual(source, original)
        self.assertEqual(manifest["version"], 1)
        self.assertEqual(manifest["source_format"], "api_prompt")
        # Captured from the original HEAD:api_nodes/workflow_contract.py; this
        # protects existing endpoint clients against an accidental v1 change.
        self.assertEqual(manifest["contract_fingerprint"], "743b4e0d8a107fb641377de4ede8d86c812662ac9d30bf7f2daddaa652e72bc1")
        first = manifest["input_panels"][0]["parameters"][0]
        self.assertEqual(first["id"], "prompt-id")
        self.assertEqual(first["current_value"], "exported prompt")
        self.assertFalse(first["required"])
        self.assertEqual(first["binding"], {"node_id": "1", "input_key": "prompt", "patchable": True})
        output = manifest["output_panels"][0]
        self.assertEqual(output["paired_input_node_id"], "1")
        self.assertEqual([param["slot"] for param in output["parameters"]], ["param_0", "param_1"])
        self.assertEqual(output["history_binding"], {"node_id": "3", "preferred_field": "document", "fallback_fields": ["document_json", "schema_json", "params"]})

    def test_workflow_and_api_share_contract_but_have_different_bindings(self):
        workflow = contract.discover_manifest(workflow_fixture())
        api = contract.discover_manifest(api_fixture())
        self.assertEqual(workflow["contract_fingerprint"], api["contract_fingerprint"])
        self.assertEqual(workflow["input_panels"][0]["parameters"][0]["binding"], {"node_id": "1", "output_slot": 1, "output_name": "param_0", "patchable": False})
        self.assertEqual(workflow["output_panels"][0]["parameters"][1]["input_slot"], 2)

    def test_named_workflow_widgets_support_config_without_positional_values(self):
        source = workflow_fixture()
        for node in source["nodes"][:3]:
            node["widgets_values_named"] = {"_config": node.pop("widgets_values")[0]}
        source["nodes"][0]["widgets_values_named"].update(prompt="named prompt", steps=9)
        manifest = contract.discover_manifest(source)
        self.assertEqual(manifest["contract_fingerprint"], contract.discover_manifest(api_fixture())["contract_fingerprint"])
        self.assertEqual([param["current_value"] for param in manifest["input_panels"][0]["parameters"]], ["named prompt", 9])
        self.assertEqual(manifest["output_panels"][0]["parameters"][1]["slot"], "param_1")

    def test_named_config_and_values_win_over_stale_legacy_snapshots(self):
        source = workflow_fixture()
        node = source["nodes"][0]
        node["widgets_values_named"] = {"_config": node["widgets_values"][0], "prompt": ""}
        node["widgets_values"] = ["invalid stale config"]
        params = contract.discover_manifest(source)["input_panels"][0]["parameters"]
        self.assertEqual(params[0]["current_value"], "")
        self.assertEqual(params[1]["current_value"], 12)

    def test_invalid_named_values_do_not_silently_fall_back_to_valid_legacy_values(self):
        source = workflow_fixture()
        source["nodes"][0]["widgets_values_named"] = {"steps": "9"}
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            contract.discover_manifest(source)
        source["nodes"][0]["widgets_values_named"] = {"_config": "not JSON"}
        with self.assertRaises(ValueError):
            contract.discover_manifest(source)

    def test_linked_config_is_rejected_even_when_named_snapshot_exists(self):
        source = workflow_fixture()
        node = source["nodes"][0]
        node["widgets_values_named"] = {"_config": node["widgets_values"][0]}
        node["inputs"] = [{"name": "_config", "link": 22}]
        with self.assertRaisesRegex(ValueError, "connected _config"):
            contract.discover_manifest(source)

    def test_minimal_workflow_without_socket_metadata_remains_supported(self):
        source = workflow_fixture()
        del source["nodes"][2]["inputs"]
        self.assertEqual(contract.discover_manifest(source)["output_panels"][0]["paired_input_node_id"], "1")

    def test_workflow_pairing_uses_named_socket_and_ignores_other_wildcard_links(self):
        source = workflow_fixture()
        output = source["nodes"][2]
        output["inputs"] = [{"name": "param_0", "link": 11}, {"name": "PANEL_LINK", "link": 10}]
        source["links"] = [[10, 1, 0, 3, 1, "*"], [11, 4, 0, 3, 0, "*"], [12, 1, 0, 4, 0, "*"], [13, 2, 0, 4, 0, "*"]]
        self.assertEqual(contract.discover_manifest(source)["output_panels"][0]["paired_input_node_id"], "1")

    def test_connected_config_is_not_mistaken_for_a_static_schema(self):
        api = api_fixture()
        api["1"]["inputs"]["_config"] = ["4", 0]
        workflow = workflow_fixture()
        workflow["nodes"][0]["inputs"] = [{"name": "_config", "link": 22}]
        for document in (api, workflow):
            with self.subTest(document=document), self.assertRaisesRegex(ValueError, "connected _config"):
                contract.discover_manifest(document)

    def test_missing_config_and_exported_value_are_rejected(self):
        for key in ("_config", "prompt"):
            source = api_fixture()
            del source["1"]["inputs"][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.discover_manifest(source)

    def test_api_panel_link_rejects_wrong_source_and_noninteger_or_wrong_slot(self):
        for link in (["4", 0], ["missing", 0], ["1", 1], ["1", False], ["1", 0.0], "1", ["1"]):
            source = api_fixture()
            source["3"]["inputs"]["PANEL_LINK"] = link
            with self.subTest(link=link), self.assertRaisesRegex(ValueError, "PANEL_LINK"):
                contract.discover_manifest(source)

    def test_workflow_panel_link_rejects_wrong_source_slot_and_duplicates(self):
        for link in ([10, 4, 0, 3, 0, "*"], [10, 1, 1, 3, 0, "*"], [10, 1, 0, 3, 0, "STRING"]):
            source = workflow_fixture()
            source["links"][0] = link
            with self.subTest(link=link), self.assertRaisesRegex(ValueError, "PANEL_LINK"):
                contract.discover_manifest(source)
        source = workflow_fixture()
        source["links"].append([12, 2, 0, 3, 0, "*"])
        with self.assertRaisesRegex(ValueError, "multiple PANEL_LINK"):
            contract.discover_manifest(source)

    def test_duplicate_normalised_node_ids_fail(self):
        source = api_fixture()
        source[1] = copy.deepcopy(source["1"])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            contract.discover_manifest(source)
        workflow = workflow_fixture()
        workflow["nodes"].append(copy.deepcopy(workflow["nodes"][0]))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            contract.discover_manifest(workflow)

    def test_invalid_document_shapes_fail(self):
        for source in (None, [], {}, {"nodes": {}}, {"1": {"class_type": "Gen2_InputPanel"}}, {"nodes": [None]}):
            with self.subTest(source=source), self.assertRaises(ValueError):
                contract.discover_manifest(source)


class LegacyRuntimeMigrationTests(unittest.TestCase):
    def source(self):
        # These fixed legacy keys were produced by the old JavaScript function,
        # including its UTF-16 surrogate pair and unnormalised name/type input.
        return {
            "nodes": [{
                "id": 1, "type": "Gen2_InputPanel",
                "widgets_values": [json.dumps([
                    {"name": "prompt", "type": "STRING", "default": "first default"},
                    {"name": "提示😀", "type": "string", "default": "second default"},
                    {"name": " café ", "type": "STRING", "default": "third default"},
                ], ensure_ascii=False)],
                "properties": {"gen2RuntimeValues": {
                    "legacy-0-296d01c0": "saved ASCII",
                    "legacy-1-20c9c90c": "saved Unicode",
                    "legacy-2-e00d9389": "saved raw name",
                }},
            }],
        }

    def test_idless_workflow_recovers_old_frontend_runtime_values_and_keeps_canonical_ids(self):
        source = self.source()
        original = copy.deepcopy(source)
        manifest = contract.discover_manifest(source)
        params = manifest["input_panels"][0]["parameters"]
        self.assertEqual([param["current_value"] for param in params], ["saved ASCII", "saved Unicode", "saved raw name"])
        self.assertEqual([param["name"] for param in params], ["prompt", "提示😀", "café"])
        self.assertEqual([param["id"] for param in params], [param["id"] for param in contract.parse_input_config(source["nodes"][0]["widgets_values"][0])])
        self.assertEqual(source, original)

    def test_canonical_runtime_key_takes_precedence_over_old_key(self):
        source = self.source()
        canonical = contract.parse_input_config(source["nodes"][0]["widgets_values"][0])[0]["id"]
        source["nodes"][0]["properties"]["gen2RuntimeValues"][canonical] = "canonical value"
        self.assertEqual(contract.discover_manifest(source)["input_panels"][0]["parameters"][0]["current_value"], "canonical value")

    def test_explicit_ids_never_read_an_unrelated_old_runtime_key(self):
        source = self.source()
        entries = json.loads(source["nodes"][0]["widgets_values"][0])
        entries[0]["id"] = "explicit-id"
        source["nodes"][0]["widgets_values"][0] = json.dumps(entries)
        param = contract.discover_manifest(source)["input_panels"][0]["parameters"][0]
        self.assertEqual(param["id"], "explicit-id")
        self.assertEqual(param["current_value"], "first default")


class CallInputTests(unittest.TestCase):
    def setUp(self):
        self.manifest = contract.discover_manifest(api_fixture())

    def test_unique_flat_names_work_across_multiple_panels(self):
        values = contract.validate_call_inputs(self.manifest, {"prompt": "new", "strength": 0.9})
        self.assertEqual(values, {"1": {"prompt": "new", "steps": 12}, "2": {"strength": 0.9}})

    def test_scoped_calls_and_omitted_values_keep_exported_current_values(self):
        values = contract.validate_call_inputs(self.manifest, {"1": {"steps": 8}})
        self.assertEqual(values, {"1": {"prompt": "exported prompt", "steps": 8}, "2": {"strength": 0.6}})
        self.assertEqual(contract.validate_call_inputs(self.manifest, {})["1"]["steps"], 12)

    def test_integer_panel_ids_normalise_without_silently_discarding_values(self):
        self.assertEqual(contract.validate_call_inputs(self.manifest, {1: {"steps": 8}})["1"]["steps"], 8)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            contract.validate_call_inputs(self.manifest, {1: {"steps": 8}, "1": {"steps": 9}})

    def test_scoped_parameter_keys_normalise_and_reject_collisions(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["input_panels"][0]["parameters"][0]["name"] = "42"
        self.assertEqual(contract.validate_call_inputs(manifest, {"1": {42: "numeric key"}})["1"]["42"], "numeric key")
        with self.assertRaisesRegex(ValueError, "duplicated"):
            contract.validate_call_inputs(manifest, {"1": {42: "first", "42": "second"}})

    def test_duplicate_names_require_scoped_inputs(self):
        duplicate = copy.deepcopy(self.manifest)
        duplicate["input_panels"][1]["parameters"] = copy.deepcopy(duplicate["input_panels"][0]["parameters"])
        with self.assertRaisesRegex(ValueError, "panel-scoped"):
            contract.validate_call_inputs(duplicate, {"prompt": "ambiguous"})
        values = contract.validate_call_inputs(duplicate, {"1": {"prompt": "one"}, "2": {"prompt": "two"}})
        self.assertEqual(values["1"]["prompt"], "one")
        self.assertEqual(values["2"]["prompt"], "two")

    def test_unknown_keys_are_rejected_at_the_correct_scope(self):
        for values in ({"unknown": 1}, {"1": {"typo": 1}}, {"missing": {}}, {"1": {}, "missing": {}}, {"1": {}, "strength": 0.7}, {"1": {}, "2": 0.7}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                contract.validate_call_inputs(self.manifest, values)

    def test_reject_unknown_false_ignores_extras_in_flat_and_scoped_calls(self):
        flat = contract.validate_call_inputs(self.manifest, {"prompt": "new", "unused": 1}, reject_unknown=False)
        scoped = contract.validate_call_inputs(self.manifest, {"1": {"prompt": "new", "unused": 1}, "other": {}}, reject_unknown=False)
        self.assertEqual(flat, scoped)

    def test_values_keep_strict_types_ranges_and_steps(self):
        for values in ({"steps": "8"}, {"steps": True}, {"steps": 8.0}, {"steps": 21}, {"prompt": 10}, {"strength": 0.65}, {"strength": float("nan")}, {"strength": float("inf")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                contract.validate_call_inputs(self.manifest, values)


class PromptPatchingTests(unittest.TestCase):
    def test_patching_only_changes_input_values_and_does_not_mutate_source(self):
        source = api_fixture()
        original = copy.deepcopy(source)
        patched, manifest = contract.prepare_api_prompt(source, {"prompt": "new", "strength": 0.9})
        self.assertEqual(source, original)
        expected = copy.deepcopy(source)
        expected["1"]["inputs"]["prompt"] = "new"
        expected["2"]["inputs"]["strength"] = 0.9
        self.assertEqual(patched, expected)
        self.assertEqual(manifest["version"], 1)
        patched["3"]["inputs"]["PANEL_LINK"][0] = "changed"
        self.assertEqual(source["3"]["inputs"]["PANEL_LINK"], ["1", 0])

    def test_integer_node_keys_are_preserved(self):
        source = {int(key): node for key, node in api_fixture().items()}
        patched, _ = contract.prepare_api_prompt(source, {"1": {"steps": 8}})
        self.assertEqual(set(patched), {1, 2, 3, 4})
        self.assertEqual(patched[1]["inputs"]["steps"], 8)

    def test_runtime_changes_do_not_invalidate_fingerprint_and_current_prompt_wins(self):
        source = api_fixture()
        manifest = contract.discover_manifest(source)
        source["1"]["inputs"]["steps"] = 15
        manifest["input_panels"][0]["parameters"][1]["current_value"] = 2
        patched = contract.patch_api_prompt(source, manifest, {})
        self.assertEqual(patched["1"]["inputs"]["steps"], 15)

    def test_schema_pairing_or_fingerprint_changes_block_stale_calls(self):
        original = api_fixture()
        manifest = contract.discover_manifest(original)
        for kind in ("schema", "pairing", "fingerprint"):
            source, supplied_manifest = copy.deepcopy(original), copy.deepcopy(manifest)
            if kind == "schema":
                config = json.loads(source["1"]["inputs"]["_config"])
                config[1]["max"] = 30
                source["1"]["inputs"]["_config"] = json.dumps(config)
            elif kind == "pairing":
                source["3"]["inputs"]["PANEL_LINK"] = ["2", 0]
            else:
                supplied_manifest["contract_fingerprint"] = "invalid"
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "current panel contract"):
                contract.patch_api_prompt(source, supplied_manifest, {})

    def test_workflow_manifest_or_wrong_version_cannot_patch(self):
        manifest = contract.discover_manifest(api_fixture())
        manifest["version"] = 2
        for supplied in (manifest, contract.discover_manifest(workflow_fixture())):
            with self.assertRaisesRegex(ValueError, "version-1"):
                contract.patch_api_prompt(api_fixture(), supplied, {})
        with self.assertRaisesRegex(ValueError, "Only a ComfyUI API prompt"):
            contract.patch_api_prompt(workflow_fixture(), manifest, {})


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.manifest = contract.discover_manifest(api_fixture())

    def test_preferred_document_retains_all_runs_and_images(self):
        first, latest = result_document("first"), result_document("latest")
        images = [{"name": "preview", "filename": "image.png", "subfolder": "", "type": "output"}]
        history = {"job": {"outputs": {"3": {"document": [first, latest], "document_json": [json.dumps(first)], "params": [{"result": "older"}], "images": images}}}}
        result = contract.extract_history_results(history, self.manifest, prompt_id="job")
        self.assertEqual(result, {"prompt_id": "job", "panels": {"3": {"latest": latest, "runs": [first, latest], "images": images}}})
        result["panels"]["3"]["latest"]["outputs"]["latest_values"]["result"] = "mutated"
        self.assertEqual(history["job"]["outputs"]["3"]["document"][-1]["outputs"]["latest_values"]["result"], "latest")

    def test_json_compatibility_fields_and_direct_entry_are_supported(self):
        document = result_document("success")
        for field in ("document_json", "schema_json"):
            history = {"prompt_id": "job", "outputs": {3: {"document": ["invalid"], field: ["invalid", json.dumps(document)]}}}
            with self.subTest(field=field):
                result = contract.extract_history_results(history, self.manifest)
                self.assertEqual(result["prompt_id"], "job")
                self.assertEqual(result["panels"]["3"]["latest"], document)

    def test_legacy_params_fallback_keeps_document_shape(self):
        result = contract.extract_history_results({"job": {"outputs": {"3": {"params": [{"result": "old"}, {"result": "latest"}]}}}}, self.manifest)
        latest = result["panels"]["3"]["latest"]
        self.assertEqual(latest["version"], 1)
        self.assertEqual(latest["inputs"], {"schema": [], "latest_values": {}})
        self.assertEqual(latest["outputs"]["latest_values"], {"result": "latest"})
        self.assertEqual(latest["outputs"]["schema"], [{"id": "result-id", "name": "result", "type": "STRING"}, {"id": "image-id", "name": "preview", "type": "IMAGE"}])

    def test_images_only_history_preserves_absent_document_behavior(self):
        result = contract.extract_history_results({"outputs": {"3": {"images": []}}}, self.manifest)
        self.assertEqual(result["panels"]["3"], {"latest": None, "runs": [], "images": []})

    def test_ambiguous_missing_or_malformed_history_is_rejected(self):
        for history, kwargs in (({"a": {}, "b": {}}, {}), ({"job": {}}, {"prompt_id": "missing"}), ({"outputs": {}}, {}), ({"job": None}, {}), ({"outputs": []}, {})):
            with self.subTest(history=history), self.assertRaises(ValueError):
                contract.extract_history_results(history, self.manifest, **kwargs)


if __name__ == "__main__":
    unittest.main()
