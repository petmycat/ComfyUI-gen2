"""Opt-in execution against an actual ComfyUI checkout, in an isolated process.

Set COMFYUI_PATH to a checkout and optionally COMFYUI_TEST_DEPS to a temporary
pip --target directory. This test uses --cpu, no checkpoints and no HTTP server.
Only websocket delivery is replaced with an in-memory sink; ComfyUI's loader,
validator, scheduler, executor and node API remain real.
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
import unittest


ROOT = Path(__file__).resolve().parents[1]


def run_worker():
    core = Path(os.environ["COMFYUI_PATH"]).resolve()
    if not (core / "execution.py").is_file():
        raise ValueError("COMFYUI_PATH must point to a ComfyUI source checkout")
    paths = [str(core)]
    if os.environ.get("COMFYUI_TEST_DEPS"):
        paths.append(str(Path(os.environ["COMFYUI_TEST_DEPS"]).resolve()))
    sys.path[:0] = paths
    sys.argv = ["gen2-cpu-integration", "--cpu", "--disable-auto-launch"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfyui_version
    import execution
    import folder_paths
    import nodes
    from PIL import Image

    loaded = asyncio.run(nodes.load_custom_node(str(ROOT)))
    if not loaded:
        raise AssertionError("ComfyUI could not load the node pack")
    expected_nodes = {
        "Gen2_InputPanel", "Gen2_OutputPanel", "Gen2_Checkerboard",
        "Gen2_StringReplace", "Gen2_LanPaintSoftDenoisePatch", "Gen2_Flux2KleinFix",
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

        report = {
            "comfyui_version": comfyui_version.__version__, "device": "cpu",
            "registered_nodes": sorted(expected_nodes), "executed_nodes": ["Gen2_InputPanel", "Gen2_Checkerboard", "Gen2_StringReplace", "Gen2_OutputPanel"],
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
        run_worker()
    else:
        unittest.main()
