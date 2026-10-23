"""Characterize the legacy Klein wrapper without claiming it is concurrency-safe."""

import contextlib
import io
import sys
import unittest

import torch

from _sampling_support import ModelPatcherDouble, klein_module


class KleinWrapperTests(unittest.TestCase):
    def setUp(self):
        self.module, self.flux = klein_module()
        self.native_orig = self.flux.forward_orig
        self.native_forward = self.flux._forward
        self.params = {"input": torch.ones(1), "timestep": torch.tensor([0.5]), "c": {"context": "prompt"}}

    def assert_restored(self):
        self.assertIs(self.flux.forward_orig, self.native_orig)
        self.assertIs(self.flux._forward, self.native_forward)

    def test_wrapper_installs_legacy_functions_only_during_call(self):
        calls = []

        def apply_model(x, timestep, **conditioning):
            self.assertIs(self.flux.forward_orig, self.module.good_forward_orig)
            self.assertIs(self.flux._forward, self.module.good_underscore_forward)
            calls.append((x, timestep, conditioning))
            return "sample result"

        result = self.module._KleinFixWrapper()(apply_model, self.params)
        self.assertEqual(result, "sample result")
        self.assertIs(calls[0][0], self.params["input"])
        self.assertIs(calls[0][1], self.params["timestep"])
        self.assertEqual(calls[0][2], {"context": "prompt"})
        self.assert_restored()

    def test_wrapper_restores_class_methods_when_sampling_raises(self):
        failure = RuntimeError("sampling failed")

        def fail(*args, **kwargs):
            raise failure

        with self.assertRaises(RuntimeError) as caught:
            self.module._KleinFixWrapper()(fail, self.params)
        self.assertIs(caught.exception, failure)
        self.assert_restored()

    def test_nested_sequential_calls_restore_outer_then_native_functions(self):
        wrapper = self.module._KleinFixWrapper()

        def outer(*args, **kwargs):
            result = wrapper(lambda *args, **kwargs: 42, self.params)
            self.assertIs(self.flux.forward_orig, self.module.good_forward_orig)
            self.assertIs(self.flux._forward, self.module.good_underscore_forward)
            return result

        self.assertEqual(wrapper(outer, self.params), 42)
        self.assert_restored()

    def test_patch_changes_clone_without_changing_source_or_global_methods(self):
        model = ModelPatcherDouble()
        original_wrapper = object()
        model.model_options = {"model_function_wrapper": original_wrapper, "other": "preserved"}
        with contextlib.redirect_stdout(io.StringIO()):
            clone, = self.module.Gen2_Flux2KleinFix().patch(model)
        self.assertIsNot(clone, model)
        self.assertIs(model.model_options["model_function_wrapper"], original_wrapper)
        self.assertIsInstance(clone.model_options["model_function_wrapper"], self.module._KleinFixWrapper)
        self.assertEqual(clone.model_options["other"], "preserved")
        self.assert_restored()

    def test_import_boundary_restores_existing_comfy_modules(self):
        before = {name: value for name, value in sys.modules.items() if name == "comfy" or name.startswith("comfy.")}
        klein_module()
        after = {name: value for name, value in sys.modules.items() if name == "comfy" or name.startswith("comfy.")}
        self.assertEqual(before, after)
        self.assertNotIn("_gen2_test_f2k", sys.modules)


if __name__ == "__main__":
    unittest.main()
