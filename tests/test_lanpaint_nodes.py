"""CPU contracts for soft masks, reference updates and LanPaint wrapper cleanup.

These tests use a deterministic sampler boundary, not a real LanPaint install or
model. They do not establish GPU parity or compatibility with future LanPaint.
"""

from dataclasses import replace
import sys
import types
import unittest

import torch

from _sampling_support import LanPaintRuntimeDouble, ModelPatcherDouble, lanpaint_module


LP = lanpaint_module()


def tensor(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)


def stage(state, candidate, region):
    ones = torch.ones_like(candidate)
    state.stage_update(candidate, region, torch.tensor([1.0]), 1.0, 1.0, ones, ones, ones)


class MaskAndScheduleTests(unittest.TestCase):
    def test_soft_mask_bilinear_resize_broadcasts_without_binarizing(self):
        source = torch.tensor([[0., 1.]])
        result = LP.prepare_soft_mask(source, torch.zeros(2, 3, 1, 4))
        self.assertEqual(result.shape, (2, 3, 1, 4))
        expected = tensor([0., 0.25, 0.75, 1.]).expand(2, 3, 1, 4)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(source, torch.tensor([[0., 1.]]))

    def test_soft_mask_clamps_and_preserves_per_batch_values(self):
        source = torch.tensor([[[-1., 0.2, 2.]], [[0.1, 0.6, 0.9]]])
        result = LP.prepare_soft_mask(source, torch.zeros(2, 1, 1, 3, dtype=torch.float64))
        self.assertEqual(result.dtype, torch.float64)
        torch.testing.assert_close(result[:, 0], source.clamp(0, 1).double())

    def test_invalid_masks_fail_before_sampling(self):
        for source in (torch.tensor([[float("nan")]]), torch.tensor([[float("inf")]])):
            with self.subTest(source=source), self.assertRaisesRegex(RuntimeError, "NaN or infinite"):
                LP.prepare_soft_mask(source, torch.zeros(1, 1, 1, 1))
        with self.assertRaisesRegex(RuntimeError, "batch"):
            LP.prepare_soft_mask(torch.zeros(2, 1, 1), torch.zeros(3, 1, 1, 1))
        with self.assertRaisesRegex(RuntimeError, "channels"):
            LP.prepare_soft_mask(torch.zeros(1, 2, 1, 1), torch.zeros(1, 3, 1, 1))
        with self.assertRaisesRegex(RuntimeError, "5D video"):
            LP.prepare_soft_mask(torch.ones(1, 1), torch.zeros(1, 1, 1, 1, 1))

    def test_hard_envelope_uses_strict_threshold_and_nearest_resize(self):
        source = torch.tensor([[0.5, 0.51]])
        result = LP.prepare_hard_envelope(source, torch.zeros(1, 1, 1, 4))
        torch.testing.assert_close(result, tensor([0., 0., 1., 1.]))

    def test_blend_schedules_obey_start_end_and_automatic_last_step(self):
        for curve in ("linear", "cosine"):
            with self.subTest(curve=curve):
                values = [LP.schedule_strength(1, 0, step, 5, curve, 0, 0) for step in (0, 2, 4, 8)]
                self.assertEqual(values[0], 1)
                self.assertAlmostEqual(values[1], 0.5)
                self.assertEqual(values[2:], [0, 0])
        self.assertEqual(LP.schedule_strength(0.8, 0.1, 99, 5, "constant", 0, 0), 0.8)
        self.assertEqual(LP.schedule_strength(1, 0, 1, 10, "linear", 2, 4), 1)
        self.assertEqual(LP.schedule_strength(1, 0, 4, 10, "linear", 2, 4), 0)

    def test_effective_mask_strength_and_disabled_blend(self):
        soft = tensor([0., 0.5, 1.])
        torch.testing.assert_close(LP.effective_edit_mask(soft, 0.5, True), tensor([0.5, 0.75, 1.]))
        torch.testing.assert_close(LP.effective_edit_mask(soft, 1, False), torch.ones_like(soft))

    def test_reference_selector_and_mixture_have_predictable_endpoints(self):
        soft = tensor([0., 0.5, 1.])
        selector = LP.build_reference_selector(soft, "smoothstep", 0, 1, 1, 0.2, 0.8)
        torch.testing.assert_close(selector, tensor([0.2, 0.5, 0.8]))
        inverted = LP.build_reference_selector(soft, "smoothstep", 0, 1, 1, 0.2, 0.8, True)
        torch.testing.assert_close(inverted, tensor([0.8, 0.5, 0.2]))
        mixed = LP.build_mixed_output_reference(torch.zeros_like(soft), torch.full_like(soft, 10), selector)
        torch.testing.assert_close(mixed, tensor([2., 5., 8.]))
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            LP.build_mixed_output_reference(soft, soft, torch.ones(1))

    def test_sigma_progress_uses_sigma_trajectory_with_degenerate_fallback(self):
        sigmas = torch.tensor([10., 9., 0.])
        self.assertAlmostEqual(LP.schedule_progress_from_sigma(torch.tensor([9.]), sigmas, 1, 3, "normalized_sigma"), 0.1)
        self.assertEqual(LP.schedule_progress_from_sigma(torch.tensor([9.]), sigmas, 1, 3, "normalized_step"), 0.5)
        self.assertEqual(LP.schedule_progress_from_sigma(torch.tensor([1.]), torch.ones(3), 1, 3, "normalized_sigma"), 0.5)

    def test_adaptive_regions_never_escape_hard_mask(self):
        soft = tensor([0., 0.5, 1., 0.5])
        hard = tensor([1., 1., 1., 0.])
        for mode, expected in (
            ("soft_band_only", [0., 1., 0., 0.]),
            ("soft_nonzero", [0., 1., 1., 0.]),
            ("hard_edit_region", [1., 1., 1., 0.]),
        ):
            with self.subTest(mode=mode):
                torch.testing.assert_close(LP.adaptive_update_region(mode, soft, hard), tensor(expected))

    def test_invalid_configuration_is_rejected(self):
        for changes in (
            {"reference_mode": "unknown"}, {"reference_ema_momentum": 1.0},
            {"reference_selector_low": 0.8, "reference_selector_high": 0.2},
            {"schedule_start_step": 4, "schedule_end_step": 2},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(LP.LanPaintSoftDenoiseConfig(), **changes).validate()


class AdaptiveReferenceTests(unittest.TestCase):
    def test_updates_are_deferred_and_ema_is_limited_to_selected_region(self):
        config = LP.LanPaintSoftDenoiseConfig(reference_mode="ema_generated", reference_ema_momentum=0.75)
        state = LP.AdaptiveReferenceState(config, 3)
        source = tensor([2., 2., 2.])
        state.initialize(source)
        candidate = tensor([10., 10., 10.])
        stage(state, candidate, tensor([0., 1., 0.]))
        candidate.fill_(100)
        source.fill_(200)
        torch.testing.assert_close(state.active_reference(), tensor([2., 2., 2.]))
        self.assertTrue(state.commit_outer_step(0))
        torch.testing.assert_close(state.active_reference(), tensor([2., 4., 2.]))
        self.assertIsNone(state.pending_candidate)
        self.assertEqual(state.current_step, 1)

    def test_warmup_and_first_generated_initialization(self):
        config = LP.LanPaintSoftDenoiseConfig(
            reference_mode="ema_generated", reference_warmup_steps=1,
            adaptive_reference_init="first_generated_after_warmup", reference_ema_momentum=0.9,
        )
        state = LP.AdaptiveReferenceState(config, 3)
        state.initialize(tensor([2.]))
        stage(state, tensor([10.]), tensor([1.]))
        self.assertFalse(state.commit_outer_step(0))
        torch.testing.assert_close(state.active_reference(), tensor([2.]))
        stage(state, tensor([8.]), tensor([1.]))
        self.assertTrue(state.commit_outer_step(1))
        # The first generated reference initializes directly, without EMA dilution.
        torch.testing.assert_close(state.active_reference(), tensor([8.]))

    def test_original_reference_mode_never_adapts(self):
        state = LP.AdaptiveReferenceState(LP.LanPaintSoftDenoiseConfig(reference_mode="original_source"), 2)
        state.initialize(tensor([2.]))
        stage(state, tensor([10.]), tensor([1.]))
        self.assertFalse(state.commit_outer_step(0))
        torch.testing.assert_close(state.active_reference(), tensor([2.]))


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.runtime = LanPaintRuntimeDouble()

    def test_soft_blend_preserves_exterior_and_forwards_runtime_arguments(self):
        proxy = LP.LanPaintSoftDenoiseProxy(self.runtime, tensor([0., 0.5, 1.]))
        x = tensor([8., 8., 8.])
        hard = tensor([1., 1., 0.])
        output = proxy(x, torch.tensor([1.]), denoise_mask=hard, seed=123, custom="kept")
        torch.testing.assert_close(output, tensor([2., 6., 2.]))
        torch.testing.assert_close(x, tensor([5., 6.5, 5.]))
        self.assertIs(self.runtime.calls[0][2]["denoise_mask"], hard)
        self.assertEqual(self.runtime.calls[0][2]["seed"], 123)
        self.assertEqual(self.runtime.calls[0][2]["custom"], "kept")
        torch.testing.assert_close(self.runtime.latent_image, tensor([2., 2., 2.]))

    def test_full_edit_matches_unpatched_runtime_output(self):
        proxy = LP.LanPaintSoftDenoiseProxy(self.runtime, torch.ones(1, 3))
        x = tensor([8., 8., 8.])
        output = proxy(x, torch.tensor([1.]), denoise_mask=torch.ones_like(x))
        torch.testing.assert_close(output, tensor([10., 10., 10.]))
        torch.testing.assert_close(x, tensor([8., 8., 8.]))

    def test_mask_function_runs_once_on_soft_path_without_mutating_options(self):
        calls = []

        def mask_function(sigma, mask, extra_options):
            calls.append((mask.clone(), extra_options))
            return torch.ones_like(mask)

        options = {"denoise_mask_function": mask_function, "nested": {"kept": 1}}
        proxy = LP.LanPaintSoftDenoiseProxy(self.runtime, tensor([0., 0.5, 1.]))
        output = proxy(tensor([8., 8., 8.]), torch.tensor([1.]), denoise_mask=tensor([1., 1., 0.]), model_options=options)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("denoise_mask_function", self.runtime.calls[0][2]["model_options"])
        self.assertIs(options["denoise_mask_function"], mask_function)
        torch.testing.assert_close(output, tensor([10., 10., 2.]))

    def test_missing_mask_and_incompatible_runtime_fail_closed(self):
        proxy = LP.LanPaintSoftDenoiseProxy(self.runtime, torch.ones(1, 3))
        with self.assertRaisesRegex(RuntimeError, "requires a noise mask"):
            proxy(tensor([8., 8., 8.]), torch.tensor([1.]))
        with self.assertRaisesRegex(RuntimeError, "compatible LanPaint"):
            LP.validate_lanpaint_runtime(lambda x: x)
        self.assertEqual(self.runtime.calls, [])

    def test_generated_shape_mismatch_is_rejected(self):
        class WrongShapeRuntime(LanPaintRuntimeDouble):
            def __call__(self, x, sigma, **kwargs):
                return torch.zeros(1)

        proxy = LP.LanPaintSoftDenoiseProxy(WrongShapeRuntime(), torch.ones(1, 3))
        with self.assertRaisesRegex(RuntimeError, "output shape"):
            proxy(tensor([8., 8., 8.]), torch.tensor([1.]), denoise_mask=tensor([1., 1., 1.]))


class WrapperLifecycleTests(unittest.TestCase):
    def test_patch_is_clone_local_and_reapplication_replaces_own_wrapper(self):
        model = ModelPatcherDouble()
        unrelated = object()
        model.wrappers = {"sampler_sample": {"unrelated": unrelated}}
        model.attachments = {"unrelated": unrelated}
        first_mask = torch.ones(1, 3)
        first = LP.apply_lanpaint_soft_denoise_patch(model, first_mask)
        second_mask = torch.zeros(1, 3)
        second = LP.apply_lanpaint_soft_denoise_patch(first, second_mask)
        self.assertNotIn(LP.WRAPPER_KEY, model.wrappers["sampler_sample"])
        self.assertNotIn(LP.ATTACHMENT_KEY, model.attachments)
        self.assertIs(first.attachments[LP.ATTACHMENT_KEY]["soft_mask"], first_mask)
        self.assertIs(second.attachments[LP.ATTACHMENT_KEY]["soft_mask"], second_mask)
        self.assertEqual(set(second.wrappers["sampler_sample"]), {"unrelated", LP.WRAPPER_KEY})
        self.assertIs(second.wrappers["sampler_sample"]["unrelated"], unrelated)

    def test_sampler_callable_is_restored_on_success_and_exception(self):
        original = lambda *args, **kwargs: None
        sampler = types.SimpleNamespace(sampler_function=original)
        wrapper = LP.make_sampler_sample_wrapper(torch.ones(1, 3))

        class Executor:
            class_obj = sampler
            fail = False

            def __call__(self, *args, **kwargs):
                if sampler.sampler_function is original:
                    raise AssertionError("wrapper was not installed during execution")
                if self.fail:
                    raise RuntimeError("sampler failed")
                return (args, kwargs)

        executor = Executor()
        self.assertEqual(wrapper(executor, "input", seed=4), (("input",), {"seed": 4}))
        self.assertIs(sampler.sampler_function, original)
        executor.fail = True
        with self.assertRaisesRegex(RuntimeError, "sampler failed"):
            wrapper(executor)
        self.assertIs(sampler.sampler_function, original)

    def test_each_run_gets_fresh_state_and_duplicate_callback_commits_once(self):
        states = []
        callback_events = []
        config = LP.LanPaintSoftDenoiseConfig(reference_mode="latest_generated")

        def sampler(proxy, noise, sigmas, callback):
            states.append(proxy.state)
            x = tensor([8., 8., 8.])
            proxy(x, torch.tensor([1.]), denoise_mask=torch.ones_like(x))
            torch.testing.assert_close(proxy.state.active_reference(), tensor([2., 2., 2.]))
            callback({"i": 0})
            callback({"i": 0})
            self.assertEqual(proxy.state.current_step, 1)
            torch.testing.assert_close(proxy.state.active_reference(), tensor([2., 10., 2.]))
            return "done"

        wrapped = LP.make_lanpaint_soft_sampler_function(sampler, tensor([0., 0.5, 1.]), config)
        runtime = LanPaintRuntimeDouble()
        for _ in range(2):
            self.assertEqual(wrapped(runtime, runtime.noise, runtime.sigmas, callback=callback_events.append), "done")
        self.assertIsNot(states[0], states[1])
        self.assertEqual(len(callback_events), 4)

    def test_import_does_not_leave_comfy_stubs_in_process(self):
        before = {name: value for name, value in sys.modules.items() if name == "comfy" or name.startswith("comfy.")}
        lanpaint_module()
        after = {name: value for name, value in sys.modules.items() if name == "comfy" or name.startswith("comfy.")}
        self.assertEqual(before, after)
        self.assertNotIn("_gen2_test_lanpaint", sys.modules)


if __name__ == "__main__":
    unittest.main()
