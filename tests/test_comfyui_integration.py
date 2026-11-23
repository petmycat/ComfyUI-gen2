"""Opt-in execution against an actual ComfyUI checkout, in an isolated process.

Set COMFYUI_PATH to a checkout and optionally COMFYUI_TEST_DEPS to a temporary
pip --target directory. This test uses --cpu, no checkpoints and no HTTP server.
Websocket delivery uses an in-memory sink. ComfyUI's loader, validator,
scheduler, executor and node API run against temporary test inputs.
"""

import asyncio
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def run_worker(base_directory):
    core = Path(os.environ["COMFYUI_PATH"]).resolve()
    if not (core / "execution.py").is_file():
        raise ValueError("COMFYUI_PATH must point to a ComfyUI source checkout")
    paths = [str(core)]
    if os.environ.get("COMFYUI_TEST_DEPS"):
        paths.append(str(Path(os.environ["COMFYUI_TEST_DEPS"]).resolve()))
    sys.path[:0] = paths
    sys.argv = ["gen2-cpu-integration", "--cpu", "--disable-auto-launch", "--base-directory", base_directory]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfyui_version
    import execution
    import folder_paths
    import nodes
    import torch
    from PIL import Image
    from safetensors.torch import save as serialize_tensors

    loaded = asyncio.run(nodes.load_custom_node(str(ROOT)))
    if not loaded:
        raise AssertionError("ComfyUI could not load the node pack")
    expected_nodes = {
        "Gen2_InputPanel", "Gen2_OutputPanel", "Gen2_Checkerboard",
        "Gen2_StringReplace", "Gen2_LanPaintSoftDenoisePatch", "Gen2_Flux2KleinFix", "Gen2_LoRASave",
        "Gen2_LoRALoad",
    }
    missing = expected_nodes - nodes.NODE_CLASS_MAPPINGS.keys()
    if missing:
        raise AssertionError(f"ComfyUI failed to register nodes: {sorted(missing)}")
    input_class = nodes.NODE_CLASS_MAPPINGS["Gen2_InputPanel"]
    contract = importlib.import_module(input_class.__module__.rsplit(".", 1)[0] + ".workflow_contract")

    class ServerSink:
        client_id = None
        last_node_id = None
        sockets_metadata = {}

        def __init__(self):
            self.events = []

        def send_sync(self, event, data, sid=None):
            self.events.append((event, data, sid))

        def queue_updated(self):
            pass

    with tempfile.TemporaryDirectory(prefix="gen2-cpu-execution-") as temporary:
        scratch = Path(temporary)
        inputs = scratch / "input"
        outputs = scratch / "output"
        inputs.mkdir()
        outputs.mkdir()
        folder_paths.set_input_directory(str(inputs))
        folder_paths.set_output_directory(str(outputs))
        Image.new("RGB", (2, 2), color=(255, 0, 0)).save(inputs / "source.png")
        config = [
            {"id": "p-text", "name": "text", "type": "STRING", "default": "cat"},
            {"id": "p-count", "name": "count", "type": "INT", "default": 4, "min": 2, "max": 16, "step": 2},
            {"id": "p-ratio", "name": "ratio", "type": "FLOAT", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.25},
            {"id": "p-enabled", "name": "enabled", "type": "BOOLEAN", "default": False},
            {"id": "p-choice", "name": "choice", "type": "COMBO", "default": "base"},
            {"id": "p-seed", "name": "seed", "type": "SEED", "default": 0, "controlMode": "fixed"},
            {"id": "p-image", "name": "image_ref", "type": "IMAGE", "default": "source.png"},
        ]
        output_config = [{"id": "out-" + item["id"], "name": item["name"], "type": item["type"]} for item in config]
        output_config.extend([
            {"id": "out-checker", "name": "checker", "type": "IMAGE"},
            {"id": "out-replaced", "name": "replaced", "type": "STRING"},
        ])
        prompt = {
            "1": {"class_type": "Gen2_InputPanel", "inputs": {
                "_config": json.dumps(config), **{item["name"]: item["default"] for item in config},
            }},
            "2": {"class_type": "Gen2_Checkerboard", "inputs": {"width": ["1", 2], "height": 2, "block_size": 2}},
            "3": {"class_type": "Gen2_StringReplace", "inputs": {"input_string": ["1", 1], "search": "cat", "replace": "dog"}},
            "4": {"class_type": "Gen2_OutputPanel", "inputs": {
                "_config": json.dumps(output_config), "PANEL_LINK": ["1", 0],
                **{f"param_{index}": ["1", index + 1] for index in range(len(config))},
                "param_7": ["2", 0], "param_8": ["3", 0],
            }},
        }
        values = {"text": "cat café 猫", "count": 8, "ratio": 0.75, "enabled": True,
                  "choice": "fine", "seed": 9007199254740991, "image_ref": "source.png"}
        submitted, manifest = contract.prepare_api_prompt(prompt, values)
        assert prompt["1"]["inputs"]["text"] == "cat", "Preparing an API call mutated its source prompt"
        validation = asyncio.run(execution.validate_prompt("gen2-cpu-test", submitted, None))
        if not validation[0]:
            raise AssertionError(f"Actual ComfyUI rejected the valid prompt: {validation}")
        sink = ServerSink()
        executor = execution.PromptExecutor(sink, cache_type=execution.CacheType.CLASSIC,
                                            cache_args={"ram": 0, "ram_inactive": 0})
        # HTTP submissions are fresh JSON objects. ComfyUI adds internal
        # is_changed fields while executing, so never resubmit that mutated object.
        executor.execute(copy.deepcopy(submitted), "gen2-cpu-test", {"client_id": "test"}, validation[2])
        if not executor.success:
            raise AssertionError(f"Actual ComfyUI execution failed: {executor.status_messages}")
        extracted = contract.extract_history_results(
            {"gen2-cpu-test": executor.history_result}, manifest, prompt_id="gen2-cpu-test",
        )
        document = extracted["panels"]["4"]["latest"]
        assert document["inputs"]["latest_values"] == values, document
        result = document["outputs"]["latest_values"]
        for name in ("text", "count", "ratio", "enabled", "choice", "seed"):
            assert result[name] == values[name], (name, result[name])
        assert result["replaced"] == "dog café 猫"
        assert type(result["enabled"]) is bool and type(result["seed"]) is int
        assert len(result["image_ref"]) == len(result["checker"]) == 1
        checker_ref = result["checker"][0]
        checker_path = outputs / checker_ref["subfolder"] / checker_ref["filename"]
        with Image.open(checker_path) as checker:
            assert checker.size == (8, 2), checker.size
            assert [checker.getpixel((x, 0))[0] for x in range(8)] == [0, 0, 255, 255, 0, 0, 255, 255]
        source_ref = result["image_ref"][0]
        with Image.open(outputs / source_ref["subfolder"] / source_ref["filename"]) as source:
            assert source.getpixel((0, 0)) == (255, 0, 0)

        rejected = []
        for name, bad_value in (
            ("count", True), ("count", 3.5), ("count", 3), ("count", 20),
            ("ratio", float("nan")), ("ratio", "0.5"), ("enabled", "false"),
            ("seed", 9007199254740992), ("image_ref", "missing.png"), ("unexpected", 1),
        ):
            bad_prompt = copy.deepcopy(submitted)
            bad_prompt["1"]["inputs"][name] = bad_value
            outcome = asyncio.run(execution.validate_prompt("bad-input", bad_prompt, None))
            if outcome[0]:
                raise AssertionError(f"Actual ComfyUI accepted invalid {name}={bad_value!r}")
            rejected.append(name + ":" + repr(bad_value))

        # Re-uploading a new image under the same filename must invalidate cached data.
        Image.new("RGB", (2, 2), color=(0, 0, 255)).save(inputs / "source.png")
        executor.execute(copy.deepcopy(submitted), "gen2-cpu-rerun", {"client_id": "test"}, validation[2])
        if not executor.success:
            raise AssertionError(executor.status_messages)
        rerun = contract.extract_history_results(executor.history_result, manifest)["panels"]["4"]["latest"]
        rerun_ref = rerun["outputs"]["latest_values"]["image_ref"][0]
        with Image.open(outputs / rerun_ref["subfolder"] / rerun_ref["filename"]) as updated:
            assert updated.getpixel((0, 0)) == (0, 0, 255), "Cached old upload was reused"

        loras = Path(base_directory) / "models" / "loras"
        source = loras / "incoming" / "sample.safetensors"
        source.parent.mkdir(parents=True)
        trigger = '猫  "illustration" {literal}'
        organize_prompt = {"1": {"class_type": "Gen2_LoRASave", "inputs": {
            "file": str(source.relative_to(loras)), "trigger": trigger, "model": "q21", "type": "style",
        }}}

        def execute_organizer(prompt_id):
            submission = copy.deepcopy(organize_prompt)
            valid, error, output_nodes, node_errors = asyncio.run(execution.validate_prompt(prompt_id, submission, None))
            assert valid, (error, node_errors)
            assert set(output_nodes) == set(submission), output_nodes
            executor.execute(submission, prompt_id, {"client_id": "test"}, output_nodes)
            assert executor.success, executor.status_messages

        preview_node = core / "comfy_extras" / "nodes_preview_any.py"
        assert asyncio.run(nodes.load_custom_node(str(preview_node), module_parent="comfy_extras"))
        test_adapter = serialize_tensors({
            "lora_unet_proj.lora_down.weight": torch.ones(1, 2),
            "lora_unet_proj.lora_up.weight": torch.ones(2, 1),
        })
        for index, content in enumerate((b"first LoRA", test_adapter), start=1):
            source.write_bytes(content)
            if index == 2:
                organize_prompt["2"] = {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}}
            execute_organizer(f"gen2-lora-{index}")
            destination = loras / "gen2" / str(index) / "q21" / "style"
            assert not source.exists(), "Standalone node did not move the selected file"
            assert (destination / source.name).read_bytes() == content
            assert json.loads((destination / "trigger.json").read_text(encoding="utf-8")) == trigger
        message, = executor.history_result["outputs"]["2"]["text"]
        assert message.startswith("Successfully created"), message
        execute_organizer("gen2-lora-stale-selection")
        message, = executor.history_result["outputs"]["2"]["text"]
        assert message.startswith("Failed:"), message
        assert not (loras / "gen2" / "3").exists()

        # A two-by-two linear layer lets the real model-only loader apply the
        # saved adapter on CPU without downloading a diffusion checkpoint.
        class TinyTestModel:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}

            RETURN_TYPES = ("MODEL",)
            FUNCTION = "make_model"

            def make_model(self):
                model = torch.nn.Module()
                model.diffusion_model = torch.nn.ModuleDict({"proj": torch.nn.Linear(2, 2, bias=False)})
                model.model_config = SimpleNamespace(unet_config={})
                cpu = torch.device("cpu")
                return (comfy.model_patcher.ModelPatcher(model, cpu, cpu),)

        nodes.NODE_CLASS_MAPPINGS["Gen2TestTinyModel"] = TinyTestModel
        primitives = core / "comfy_extras" / "nodes_primitive.py"
        assert asyncio.run(nodes.load_custom_node(str(primitives), module_parent="comfy_extras"))
        lookup = {
            "1": {"class_type": "PrimitiveString", "inputs": {"value": "2"}},
            "2": {"class_type": "Gen2_LoRALoad", "inputs": {"number": ["1", 0], "model": "q21", "type": "style"}},
            "3": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}},
            "4": {"class_type": "PreviewAny", "inputs": {"source": ["2", 1]}},
            "5": {"class_type": "Gen2TestTinyModel", "inputs": {}},
            "6": {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": ["5", 0], "lora_name": ["2", 0], "strength_model": 1.0,
            }},
            "7": {"class_type": "PreviewAny", "inputs": {"source": ["6", 0]}},
        }
        valid, error, output_nodes, node_errors = asyncio.run(execution.validate_prompt("gen2-lora-load", lookup, None))
        assert valid and set(output_nodes) == {"3", "4", "7"}, (error, node_errors)
        executor.execute(copy.deepcopy(lookup), "gen2-lora-load", {"client_id": "test"}, output_nodes)
        assert executor.success, executor.status_messages
        lookup_outputs = executor.history_result["outputs"]
        lora_name, = lookup_outputs["3"]["text"]
        assert lora_name == str((destination / source.name).relative_to(loras)), lora_name
        assert Path(folder_paths.get_full_path_or_raise("loras", lora_name)) == destination / source.name
        loaded_trigger, = lookup_outputs["4"]["text"]
        assert loaded_trigger == trigger, loaded_trigger
        updated_trigger = "new  trigger"
        (destination / "trigger.json").write_text(json.dumps(updated_trigger), encoding="utf-8")
        executor.execute(copy.deepcopy(lookup), "gen2-lora-reread", {"client_id": "test"}, output_nodes)
        assert executor.success, executor.status_messages
        loaded_trigger, = executor.history_result["outputs"]["4"]["text"]
        assert loaded_trigger == updated_trigger, "Cached trigger was reused after the file changed"

        report = {
            "comfyui_version": comfyui_version.__version__, "device": "cpu",
            "registered_nodes": sorted(expected_nodes), "executed_nodes": [
                "Gen2_InputPanel", "Gen2_Checkerboard", "Gen2_StringReplace", "Gen2_OutputPanel", "Gen2_LoRASave",
                "Gen2_LoRALoad",
            ],
            "rejected_invalid_inputs": rejected, "image_cache_invalidation": True,
            "history_contract_roundtrip": True, "hardware_model_sampling_tested": False,
        }
        print("GEN2_INTEGRATION_RESULT=" + json.dumps(report, ensure_ascii=False))


@unittest.skipUnless(os.environ.get("COMFYUI_PATH"), "Set COMFYUI_PATH to opt into actual ComfyUI CPU integration")
class ComfyUIIntegrationTests(unittest.TestCase):
    def test_actual_prompt_validation_execution_history_and_image_cache(self):
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                   env=env, text=True, encoding="utf-8", errors="replace",
                                   capture_output=True, timeout=180)
        self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)
        reports = [line for line in completed.stdout.splitlines() if line.startswith("GEN2_INTEGRATION_RESULT=")]
        self.assertEqual(len(reports), 1, completed.stdout)
        report = json.loads(reports[0].split("=", 1)[1])
        self.assertTrue(report["history_contract_roundtrip"])
        self.assertTrue(report["image_cache_invalidation"])
        self.assertEqual(len(report["rejected_invalid_inputs"]), 10)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        with tempfile.TemporaryDirectory(prefix="gen2-comfy-base-") as base_directory:
            run_worker(base_directory)
    else:
        unittest.main()
