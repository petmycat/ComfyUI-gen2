import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "gen2_sampling" / "lanpaint_soft_denoise.py"


patcher_extension = types.ModuleType("comfy.patcher_extension")


class WrappersMP:
    SAMPLER_SAMPLE = "sampler_sample"


patcher_extension.WrappersMP = WrappersMP
model_patcher = types.ModuleType("comfy.model_patcher")
model_patcher.create_model_options_clone = lambda value: {
    key: (nested.copy() if isinstance(nested, dict) else nested.copy() if isinstance(nested, list) else nested)
    for key, nested in value.items()
}
comfy = types.ModuleType("comfy")
comfy.model_patcher = model_patcher
comfy.patcher_extension = patcher_extension
sys.modules.setdefault("comfy", comfy)
sys.modules.setdefault("comfy.model_patcher", model_patcher)
sys.modules.setdefault("comfy.patcher_extension", patcher_extension)

spec = importlib.util.spec_from_file_location("lanpaint_soft_denoise", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)


class FakeSampling:
    def noise_scaling(self, sigma, noise, latent_image):
        return latent_image + sigma * noise


class FakeInnerModel:
    def __init__(self):
        self.inner_model = types.SimpleNamespace(model_sampling=FakeSampling())


class FakeLanPaintRuntime:
    def __init__(self, latent_image=None, noise=None):
        self.PaintMethod = object()
        self.latent_image = torch.full((1, 2, 2, 2), 2.0) if latent_image is None else latent_image
        self.noise = torch.ones_like(self.latent_image) if noise is None else noise
        self.inner_model = FakeInnerModel()
        self.sigmas = torch.tensor([1.0, 0.0])
        self.LanPaint_early_stop = 1
        self.received = None

    def __call__(self, x, sigma, denoise_mask=None, model_options=None, seed=None, **kwargs):
        self.received = {
            "x": x.clone(),
            "denoise_mask": denoise_mask,
            "model_options": model_options,
            "seed": seed,
            "kwargs": kwargs,
        }
        x.copy_(torch.full_like(x, 7.0))
        return torch.full_like(x, 10.0)


class FakeModelPatcher:
    def __init__(self):
        self.wrappers = {}
        self.attachments = {}

    def clone(self):
        cloned = FakeModelPatcher()
        cloned.wrappers = {kind: {key: values.copy() for key, values in groups.items()} for kind, groups in self.wrappers.items()}
        cloned.attachments = self.attachments.copy()
        return cloned

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def set_attachments(self, key, value):
        self.attachments[key] = value

    def remove_attachments(self, key):
        self.attachments.pop(key, None)


class MaskPreparationTests(unittest.TestCase):
    def test_soft_mask_preserves_continuous_values_and_broadcasts(self):
        x = torch.zeros(2, 4, 4, 4)
        source = torch.tensor([[[0.0, 0.25], [0.75, 1.0]]])
        prepared = module.prepare_soft_mask(source, x)
        self.assertEqual(prepared.shape, x.shape)
        self.assertTrue(torch.all(prepared[:, 0] == prepared[:, 3]))
        self.assertGreater(torch.unique(prepared).numel(), 4)
        self.assertEqual(float(prepared.min()), 0.0)
        self.assertEqual(float(prepared.max()), 1.0)

    def test_soft_mask_matching_batch(self):
        x = torch.zeros(2, 3, 2, 2)
        source = torch.stack((torch.zeros(2, 2), torch.ones(2, 2)))
        prepared = module.prepare_soft_mask(source, x)
        self.assertTrue(torch.all(prepared[0] == 0))
        self.assertTrue(torch.all(prepared[1] == 1))

    def test_incompatible_mask_batch_fails(self):
        with self.assertRaisesRegex(RuntimeError, "cannot align soft mask batch 3"):
            module.prepare_soft_mask(torch.zeros(3, 4, 4), torch.zeros(2, 4, 4, 4))

    def test_video_latent_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "does not yet support 5D video latents"):
            module.prepare_soft_mask(torch.zeros(1, 4, 4), torch.zeros(1, 4, 2, 4, 4))

    def test_hard_envelope_uses_threshold_and_nearest_resize(self):
        x = torch.zeros(1, 2, 4, 4)
        hard = torch.tensor([[[0.49, 0.51], [1.0, 0.0]]])
        prepared = module.prepare_hard_envelope(hard, x)
        self.assertEqual(set(torch.unique(prepared).tolist()), {0.0, 1.0})
        self.assertTrue(torch.all(prepared[:, 0] == prepared[:, 1]))


class ProxyTests(unittest.TestCase):
    def test_full_edit_matches_lanpaint_output(self):
        runtime = FakeLanPaintRuntime()
        proxy = module.LanPaintSoftDenoiseProxy(runtime, torch.ones(1, 2, 2))
        x = torch.zeros_like(runtime.latent_image)
        hard = torch.ones(1, 1, 2, 2)
        out = proxy(x, torch.tensor([0.5]), denoise_mask=hard, model_options={}, seed=9)
        self.assertTrue(torch.all(out == 10))
        self.assertTrue(torch.all(runtime.received["x"] == 0))
        self.assertIs(runtime.received["denoise_mask"], hard)

    def test_protected_exterior_restores_source(self):
        runtime = FakeLanPaintRuntime()
        proxy = module.LanPaintSoftDenoiseProxy(runtime, torch.zeros(1, 2, 2))
        x = torch.zeros_like(runtime.latent_image)
        out = proxy(x, torch.tensor([0.5]), denoise_mask=torch.zeros(1, 1, 2, 2), model_options={})
        self.assertTrue(torch.all(out == runtime.latent_image))
        self.assertTrue(torch.all(runtime.received["x"] == 2.5))

    def test_feather_blends_input_output_and_propagates_state(self):
        runtime = FakeLanPaintRuntime()
        proxy = module.LanPaintSoftDenoiseProxy(runtime, torch.full((1, 2, 2), 0.5))
        x = torch.zeros_like(runtime.latent_image)
        out = proxy(x, torch.tensor([0.5]), denoise_mask=torch.ones(1, 1, 2, 2), model_options={})
        self.assertTrue(torch.allclose(runtime.received["x"], torch.full_like(x, 1.25)))
        self.assertTrue(torch.all(x == 7))
        self.assertTrue(torch.all(out == 6))

    def test_containment_removes_soft_values_outside_hard_mask(self):
        runtime = FakeLanPaintRuntime()
        proxy = module.LanPaintSoftDenoiseProxy(runtime, torch.ones(1, 2, 2))
        x = torch.zeros_like(runtime.latent_image)
        hard = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        out = proxy(x, torch.tensor([0.5]), denoise_mask=hard, model_options={})
        self.assertEqual(float(out[0, 0, 0, 0]), 10.0)
        self.assertEqual(float(out[0, 0, 0, 1]), 2.0)

    def test_mask_function_is_removed_from_lanpaint_options(self):
        runtime = FakeLanPaintRuntime()
        proxy = module.LanPaintSoftDenoiseProxy(runtime, torch.ones(1, 2, 2))
        original_options = {"keep": {"value": 1}, "denoise_mask_function": lambda sigma, mask, extra_options: mask * 0.25}
        out = proxy(
            torch.zeros_like(runtime.latent_image),
            torch.tensor([0.5]),
            denoise_mask=torch.ones(1, 1, 2, 2),
            model_options=original_options,
        )
        self.assertNotIn("denoise_mask_function", runtime.received["model_options"])
        self.assertIn("denoise_mask_function", original_options)
        self.assertTrue(torch.all(out == 4))

    def test_missing_hard_mask_fails(self):
        runtime = FakeLanPaintRuntime()
        proxy = module.LanPaintSoftDenoiseProxy(runtime, torch.ones(1, 2, 2))
        with self.assertRaisesRegex(RuntimeError, "requires a noise mask from InpaintModelConditioning"):
            proxy(torch.zeros_like(runtime.latent_image), torch.tensor([0.5]))

    def test_wrong_sampler_runtime_fails(self):
        wrapped = module.make_lanpaint_soft_sampler_function(lambda model, noise, sigmas: noise, torch.ones(1, 2, 2))
        with self.assertRaisesRegex(RuntimeError, "Missing attributes"):
            wrapped(object(), torch.zeros(1), torch.zeros(1))


class AdaptiveReferenceTests(unittest.TestCase):
    def make_config(self, **overrides):
        values = {
            "reference_mode": "ema_generated",
            "reference_warmup_steps": 0,
            "reference_ema_momentum": 0.5,
            "enable_input_blend": True,
            "input_blend_strength_start": 1.0,
            "input_blend_strength_end": 1.0,
            "enable_output_blend": True,
            "output_blend_strength_start": 1.0,
            "output_blend_strength_end": 0.0,
            "blend_schedule_type": "linear",
            "schedule_start_step": 0,
            "schedule_end_step": 2,
            "adaptive_region_mode": "hard_edit_region",
            "adaptive_update_source": "raw_generated_output",
            "lock_original_outside_adaptive_region": True,
            "adaptive_reference_init": "original_source",
        }
        values.update(overrides)
        return module.LanPaintSoftDenoiseConfig(**values)

    def test_schedule_constant_linear_and_cosine(self):
        self.assertEqual(module.schedule_strength(1.0, 0.0, 1, 3, "constant", 0, 2), 1.0)
        self.assertAlmostEqual(module.schedule_strength(1.0, 0.0, 1, 3, "linear", 0, 2), 0.5)
        self.assertAlmostEqual(module.schedule_strength(1.0, 0.0, 1, 3, "cosine", 0, 2), 0.5)
        self.assertEqual(module.schedule_strength(1.0, 0.0, 2, 3, "linear", 0, 0), 0.0)

    def test_mode_c_ema_updates_and_output_strength_decays(self):
        runtime = FakeLanPaintRuntime()
        config = self.make_config()
        state = module.AdaptiveReferenceState(config, total_steps=3)
        proxy = module.LanPaintSoftDenoiseProxy(runtime, torch.full((1, 2, 2), 0.5), config, state)
        hard = torch.ones(1, 1, 2, 2)
        out0 = proxy(torch.zeros_like(runtime.latent_image), torch.tensor([0.5]), denoise_mask=hard, model_options={})
        self.assertTrue(torch.all(out0 == 6))
        state.commit_outer_step(0)
        self.assertTrue(torch.all(state.adaptive_reference == 6))
        out1 = proxy(torch.zeros_like(runtime.latent_image), torch.tensor([0.5]), denoise_mask=hard, model_options={})
        self.assertTrue(torch.all(out1 == 9))
        self.assertAlmostEqual(state.pending_output_strength, 0.5)

    def test_mode_d_bypasses_output_blending_but_updates_input_reference(self):
        runtime = FakeLanPaintRuntime()
        config = self.make_config(enable_output_blend=False, output_blend_strength_start=1.0, output_blend_strength_end=1.0)
        state = module.AdaptiveReferenceState(config, total_steps=2)
        proxy = module.LanPaintSoftDenoiseProxy(runtime, torch.full((1, 2, 2), 0.5), config, state)
        hard = torch.ones(1, 1, 2, 2)
        out = proxy(torch.zeros_like(runtime.latent_image), torch.tensor([0.5]), denoise_mask=hard, model_options={})
        self.assertTrue(torch.all(out == 10))
        state.commit_outer_step(0)
        self.assertTrue(torch.all(state.adaptive_reference == 6))
        proxy(torch.zeros_like(runtime.latent_image), torch.tensor([0.5]), denoise_mask=hard, model_options={})
        self.assertTrue(torch.allclose(runtime.received["x"], torch.full_like(runtime.latent_image, 3.25)))

    def test_mode_e_warmup_and_first_generated_initialization(self):
        runtime = FakeLanPaintRuntime()
        config = self.make_config(
            reference_mode="latest_generated",
            reference_warmup_steps=2,
            adaptive_reference_init="first_generated_after_warmup",
        )
        state = module.AdaptiveReferenceState(config, total_steps=4)
        proxy = module.LanPaintSoftDenoiseProxy(runtime, torch.full((1, 2, 2), 0.5), config, state)
        hard = torch.ones(1, 1, 2, 2)
        for step in range(2):
            proxy(torch.zeros_like(runtime.latent_image), torch.tensor([0.5]), denoise_mask=hard, model_options={})
            state.commit_outer_step(step)
            self.assertIsNone(state.adaptive_reference)
        proxy(torch.zeros_like(runtime.latent_image), torch.tensor([0.5]), denoise_mask=hard, model_options={})
        state.commit_outer_step(2)
        self.assertTrue(torch.all(state.adaptive_reference == 10))

    def test_adaptive_region_modes(self):
        soft = torch.tensor([[[[0.0, 0.5], [1.0, 0.5]]]])
        hard = torch.ones_like(soft)
        band = module.adaptive_update_region("soft_band_only", soft, hard)
        nonzero = module.adaptive_update_region("soft_nonzero", soft, hard)
        full = module.adaptive_update_region("hard_edit_region", soft, hard)
        self.assertEqual(int(band.sum()), 2)
        self.assertEqual(int(nonzero.sum()), 3)
        self.assertEqual(int(full.sum()), 4)

    def test_update_source_raw_vs_blended(self):
        hard = torch.ones(1, 1, 2, 2)
        for source, expected in (("raw_generated_output", 10.0), ("blended_output", 6.0)):
            runtime = FakeLanPaintRuntime()
            config = self.make_config(
                reference_mode="latest_generated",
                reference_ema_momentum=0.0,
                adaptive_update_source=source,
                output_blend_strength_end=1.0,
                blend_schedule_type="constant",
            )
            state = module.AdaptiveReferenceState(config, total_steps=2)
            proxy = module.LanPaintSoftDenoiseProxy(runtime, torch.full((1, 2, 2), 0.5), config, state)
            proxy(torch.zeros_like(runtime.latent_image), torch.tensor([0.5]), denoise_mask=hard, model_options={})
            state.commit_outer_step(0)
            self.assertTrue(torch.all(state.adaptive_reference == expected))

    def test_lock_original_outside_region(self):
        config = self.make_config(reference_mode="latest_generated", adaptive_region_mode="soft_band_only")
        state = module.AdaptiveReferenceState(config, total_steps=2)
        source = torch.full((1, 1, 2, 2), 2.0)
        state.initialize(source)
        region = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        state.stage_update(torch.full_like(source, 10.0), region, torch.tensor([1.0]), 1.0, 1.0, region, region, region)
        state.commit_outer_step(0)
        self.assertEqual(float(state.adaptive_reference[0, 0, 0, 0]), 10.0)
        self.assertEqual(float(state.adaptive_reference[0, 0, 0, 1]), 2.0)

    def test_sampler_wrapper_creates_fresh_state_per_run_and_commits_on_callback(self):
        states = []

        def sampler_function(proxy, noise, sigmas, callback=None, **kwargs):
            states.append(proxy.state)
            proxy(
                torch.zeros_like(proxy.latent_image),
                torch.tensor([0.5]),
                denoise_mask=torch.ones(1, 1, 2, 2),
                model_options={},
            )
            callback(0, torch.zeros_like(noise), torch.zeros_like(noise), 1)
            return noise

        config = self.make_config(reference_mode="latest_generated")
        wrapped = module.make_lanpaint_soft_sampler_function(sampler_function, torch.full((1, 2, 2), 0.5), config)
        for _ in range(2):
            runtime = FakeLanPaintRuntime()
            wrapped(runtime, torch.zeros_like(runtime.latent_image), torch.tensor([1.0, 0.0]), callback=lambda *args: None)
        self.assertIsNot(states[0], states[1])
        self.assertEqual(states[0].current_step, 1)
        self.assertEqual(states[1].current_step, 1)


class MixedOutputReferenceTests(unittest.TestCase):
    def make_config(self, **overrides):
        values = {
            "reference_mode": "latest_generated",
            "enable_input_blend": True,
            "input_blend_strength_start": 1.0,
            "input_blend_strength_end": 1.0,
            "enable_output_blend": True,
            "output_blend_strength_start": 1.0,
            "output_blend_strength_end": 1.0,
            "blend_schedule_type": "constant",
            "adaptive_region_mode": "hard_edit_region",
            "adaptive_update_source": "blended_output",
            "output_reference_mode": "legacy",
        }
        values.update(overrides)
        return module.LanPaintSoftDenoiseConfig(**values)

    def make_initialized_proxy(self, config, soft=0.5):
        runtime = FakeLanPaintRuntime()
        state = module.AdaptiveReferenceState(config, total_steps=3)
        state.initialize(runtime.latent_image)
        state.adaptive_reference = torch.full_like(runtime.latent_image, 6.0)
        proxy = module.LanPaintSoftDenoiseProxy(runtime, torch.full((1, 2, 2), soft), config, state)
        return runtime, state, proxy

    def test_legacy_ignores_all_new_mixed_controls(self):
        baseline = self.make_config(output_reference_mode="legacy")
        changed = self.make_config(
            output_reference_mode="legacy",
            reference_selector_curve="power",
            reference_selector_low=0.2,
            reference_selector_high=0.8,
            reference_selector_gamma=4.0,
            latest_reference_ratio_at_soft_min=0.3,
            latest_reference_ratio_at_soft_max=0.7,
            invert_reference_selector=True,
            mixed_output_blend_strength_start=0.1,
            mixed_output_blend_strength_end=0.9,
            mixed_output_blend_schedule="smootherstep",
            mixed_output_blend_schedule_start=0.1,
            mixed_output_blend_schedule_end=0.9,
            output_blend_mask_curve="power",
            output_blend_mask_low=0.2,
            output_blend_mask_high=0.8,
            output_blend_mask_gamma=3.0,
        )
        outputs = []
        inputs = []
        for config in (baseline, changed):
            runtime, _, proxy = self.make_initialized_proxy(config)
            output = proxy(
                torch.zeros_like(runtime.latent_image),
                torch.tensor([0.5]),
                denoise_mask=torch.ones(1, 1, 2, 2),
                model_options={},
            )
            outputs.append(output)
            inputs.append(runtime.received["x"])
        self.assertTrue(torch.equal(outputs[0], outputs[1]))
        self.assertTrue(torch.equal(inputs[0], inputs[1]))

    def test_latest_only_matches_legacy(self):
        outputs = []
        for mode in ("legacy", "latest_only"):
            runtime, _, proxy = self.make_initialized_proxy(self.make_config(output_reference_mode=mode))
            outputs.append(
                proxy(
                    torch.zeros_like(runtime.latent_image),
                    torch.tensor([0.5]),
                    denoise_mask=torch.ones(1, 1, 2, 2),
                    model_options={},
                )
            )
        self.assertTrue(torch.equal(outputs[0], outputs[1]))

    def test_reference_selector_endpoints_ratios_and_inversion(self):
        soft = torch.tensor([[[[0.0, 0.5, 1.0]]]])
        selector = module.build_reference_selector(soft, "linear", 0.0, 1.0, 1.0, 0.0, 1.0)
        self.assertTrue(torch.equal(selector, soft))
        ranged = module.build_reference_selector(soft, "smoothstep", 0.0, 1.0, 1.0, 0.2, 0.9)
        self.assertAlmostEqual(float(ranged[0, 0, 0, 0]), 0.2)
        self.assertAlmostEqual(float(ranged[0, 0, 0, 2]), 0.9)
        inverted = module.build_reference_selector(soft, "linear", 0.0, 1.0, 1.0, 0.0, 1.0, True)
        self.assertAlmostEqual(float(inverted[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(inverted[0, 0, 0, 2]), 0.0)

    def test_reference_selector_curves_and_validation(self):
        soft = torch.tensor([[[[0.25]]]])
        self.assertAlmostEqual(float(module.build_reference_selector(soft, "linear", 0.0, 1.0, 1.0, 0.0, 1.0)), 0.25)
        self.assertAlmostEqual(float(module.build_reference_selector(soft, "smoothstep", 0.0, 1.0, 1.0, 0.0, 1.0)), 0.15625)
        self.assertAlmostEqual(float(module.build_reference_selector(soft, "smootherstep", 0.0, 1.0, 1.0, 0.0, 1.0)), 0.103515625)
        self.assertAlmostEqual(float(module.build_reference_selector(soft, "power", 0.0, 1.0, 2.0, 0.0, 1.0)), 0.0625)
        with self.assertRaisesRegex(ValueError, "low/high"):
            module.build_reference_selector(soft, "linear", 0.5, 0.5, 1.0, 0.0, 1.0)

    def test_temporal_schedule_curves_and_boundaries(self):
        expected_midpoints = {
            "constant": 0.0,
            "linear": 0.5,
            "cosine": 0.5,
            "smoothstep": 0.5,
            "smootherstep": 0.5,
        }
        for curve, midpoint in expected_midpoints.items():
            self.assertEqual(module.schedule_fraction(0.1, 0.2, 0.8, curve), 0.0)
            self.assertEqual(module.schedule_fraction(0.2, 0.2, 0.8, curve), 0.0)
            self.assertAlmostEqual(module.schedule_fraction(0.5, 0.2, 0.8, curve), midpoint)
            self.assertEqual(module.schedule_fraction(0.8, 0.2, 0.8, curve), 0.0 if curve == "constant" else 1.0)
            self.assertEqual(module.schedule_fraction(0.9, 0.2, 0.8, curve), 0.0 if curve == "constant" else 1.0)

    def test_mixed_reference_changes_only_output_path(self):
        legacy_config = self.make_config(output_reference_mode="legacy")
        mixed_config = self.make_config(
            output_reference_mode="mixed_original_latest",
            reference_selector_curve="linear",
            mixed_output_blend_schedule="constant",
            mixed_output_blend_strength_start=1.0,
            output_blend_mask_curve="legacy",
        )
        observations = []
        for config in (legacy_config, mixed_config):
            runtime, _, proxy = self.make_initialized_proxy(config)
            out = proxy(
                torch.zeros_like(runtime.latent_image),
                torch.tensor([0.5]),
                denoise_mask=torch.ones(1, 1, 2, 2),
                model_options={"keep": True},
                seed=9,
            )
            observations.append((runtime.received, out))
        legacy_received, legacy_out = observations[0]
        mixed_received, mixed_out = observations[1]
        self.assertTrue(torch.equal(legacy_received["x"], mixed_received["x"]))
        self.assertTrue(torch.equal(legacy_received["denoise_mask"], mixed_received["denoise_mask"]))
        self.assertEqual(legacy_received["model_options"], mixed_received["model_options"])
        self.assertEqual(legacy_received["seed"], mixed_received["seed"])
        self.assertTrue(torch.all(legacy_out == 8.0))
        self.assertTrue(torch.all(mixed_out == 7.0))

    def test_mixed_output_strength_zero_returns_raw_and_blended_updates_next_reference(self):
        zero_config = self.make_config(
            output_reference_mode="mixed_original_latest",
            mixed_output_blend_strength_start=0.0,
            mixed_output_blend_strength_end=0.0,
            mixed_output_blend_schedule="constant",
        )
        runtime, _, proxy = self.make_initialized_proxy(zero_config)
        out = proxy(
            torch.zeros_like(runtime.latent_image),
            torch.tensor([0.5]),
            denoise_mask=torch.ones(1, 1, 2, 2),
            model_options={},
        )
        self.assertTrue(torch.all(out == 10.0))

        blended_config = self.make_config(
            output_reference_mode="mixed_original_latest",
            mixed_output_blend_schedule="constant",
            mixed_output_blend_strength_start=1.0,
        )
        runtime, state, proxy = self.make_initialized_proxy(blended_config)
        out = proxy(
            torch.zeros_like(runtime.latent_image),
            torch.tensor([0.5]),
            denoise_mask=torch.ones(1, 1, 2, 2),
            model_options={},
        )
        state.commit_outer_step(0)
        self.assertTrue(torch.equal(state.adaptive_reference, out))
        self.assertTrue(torch.all(state.adaptive_reference == 7.0))

    def test_config_rejects_invalid_ranges(self):
        with self.assertRaisesRegex(ValueError, "reference_selector_low/high"):
            self.make_config(reference_selector_low=0.5, reference_selector_high=0.5).validate()
        with self.assertRaisesRegex(ValueError, "schedule start/end"):
            self.make_config(mixed_output_blend_schedule_start=0.8, mixed_output_blend_schedule_end=0.2).validate()
        with self.assertRaisesRegex(ValueError, "output_blend_mask_low/high"):
            self.make_config(output_blend_mask_low=1.0, output_blend_mask_high=1.0).validate()


class WrapperAndNodeTests(unittest.TestCase):
    def test_sampler_function_restored_on_error(self):
        original = lambda *args, **kwargs: None
        sampler = types.SimpleNamespace(sampler_function=original)

        class Executor:
            class_obj = sampler

            def __call__(self, *args, **kwargs):
                raise ValueError("boom")

        wrapper = module.make_sampler_sample_wrapper(torch.ones(1, 2, 2))
        with self.assertRaisesRegex(ValueError, "boom"):
            wrapper(Executor())
        self.assertIs(sampler.sampler_function, original)

    def test_duplicate_patch_replaces_key_without_mutating_original(self):
        original = FakeModelPatcher()
        first = module.apply_lanpaint_soft_denoise_patch(original, torch.zeros(1, 2, 2))
        second = module.apply_lanpaint_soft_denoise_patch(first, torch.ones(1, 2, 2))
        wrappers = second.wrappers[WrappersMP.SAMPLER_SAMPLE][module.WRAPPER_KEY]
        self.assertEqual(len(wrappers), 1)
        self.assertEqual(original.wrappers, {})
        self.assertTrue(torch.all(second.attachments[module.ATTACHMENT_KEY]["soft_mask"] == 1))


if __name__ == "__main__":
    unittest.main()
