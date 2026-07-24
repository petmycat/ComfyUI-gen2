import importlib.util
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "flux2_fun"


def load_module(name):
    full_name = f"flux2_fun.{name}"
    if "flux2_fun" not in sys.modules:
        package = types.ModuleType("flux2_fun")
        package.__path__ = [str(PACKAGE)]
        sys.modules["flux2_fun"] = package
    spec = importlib.util.spec_from_file_location(full_name, PACKAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


types_mod = load_module("types")
checkpoint = load_module("checkpoint")
preprocess = load_module("preprocess")
branch_mod = load_module("branch")
model_management_mod = load_module("model_management")
oracle = load_module("oracle")
runtime = load_module("runtime")


class FakeVAE:
    latent_channels = 128
    downscale_ratio = 16

    def encode(self, image):
        b, h, w, _ = image.shape
        pooled = torch.nn.functional.interpolate(image.movedim(-1, 1), size=(h // 16, w // 16), mode="area")
        base = pooled.mean(dim=1, keepdim=True)
        return base.repeat(1, 128, 1, 1)


@dataclass
class Mod:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


def make_vec(batch, hidden, regions=1):
    shape = (batch, regions, hidden)
    zero = torch.zeros(shape)
    one = torch.ones(shape)
    return ((Mod(zero, zero, one), Mod(zero, zero, one)), (Mod(zero, zero, one), Mod(zero, zero, one)))


class TorchOps:
    Linear = torch.nn.Linear
    LayerNorm = torch.nn.LayerNorm
    RMSNorm = torch.nn.RMSNorm


def tiny_profile():
    return types_mod.CheckpointProfile(
        name="tiny",
        tensor_count=0,
        hidden_size=8,
        control_dim=6,
        block_count=4,
        mlp_hidden_dim=12,
        num_heads=2,
        head_dim=4,
        block_layers=(0, 2, 4, 6),
        sha256="",
        snapshot="",
        shapes={},
    )


def simple_attention(q, k, v, pe=None, mask=None, transformer_options=None):
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / q.shape[-1] ** 0.5
    if mask is not None:
        scores = scores + mask
    return torch.matmul(scores.softmax(dim=-1), v.float()).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1).to(v.dtype)


class PreprocessTests(unittest.TestCase):
    def test_mask_polarity_inpaint_zeroing_and_channel_order(self):
        control = torch.ones(1, 32, 32, 3)
        inpaint = torch.ones(1, 32, 32, 3)
        mask = torch.zeros(1, 32, 32)
        mask[:, :16] = 1.0
        context = preprocess.prepare_control_context(FakeVAE(), control, mask, inpaint)
        self.assertEqual(context.packed.shape, (1, 4, 260))
        self.assertTrue(torch.all(context.packed[..., :128] == 1))
        self.assertTrue(torch.all(context.packed[:, :2, 128:132] == 0))
        self.assertTrue(torch.all(context.packed[:, 2:, 128:132] == 1))
        self.assertTrue(torch.all(context.packed[:, :2, 132:] == 0))
        self.assertTrue(torch.all(context.packed[:, 2:, 132:] == 1))

    def test_target_latent_controls_exact_canvas_and_batch(self):
        control = torch.ones(1, 32, 48, 3)
        target = {"samples": torch.zeros(2, 128, 3, 5)}
        context = preprocess.prepare_control_context(FakeVAE(), control_image=control, target_latent=target)
        self.assertEqual(context.packed.shape, (2, 15, 260))
        self.assertEqual((context.latent_height, context.latent_width, context.batch_size), (3, 5, 2))

    def test_target_latent_batch_is_authoritative(self):
        with self.assertRaisesRegex(ValueError, "Cannot align control_image batch 2 to target batch 1"):
            preprocess.prepare_control_context(
                FakeVAE(),
                control_image=torch.ones(2, 32, 32, 3),
                target_latent={"samples": torch.zeros(1, 128, 2, 2)},
            )

    def test_target_latent_requires_flux2_contract(self):
        with self.assertRaisesRegex(ValueError, "\[B,128,H,W\]"):
            preprocess.prepare_control_context(
                FakeVAE(),
                control_image=torch.ones(1, 32, 32, 3),
                target_latent={"samples": torch.zeros(1, 16, 2, 2)},
            )

    def test_missing_branch_is_direct_zero_latent(self):
        context = preprocess.prepare_control_context(FakeVAE(), inpaint_image=torch.ones(1, 32, 32, 3))
        self.assertTrue(torch.count_nonzero(context.packed[..., :128]) == 0)
        context = preprocess.prepare_control_context(FakeVAE(), control_image=torch.ones(1, 32, 32, 3))
        self.assertTrue(torch.count_nonzero(context.packed[..., 132:]) == 0)

    def test_batch_rules_and_reference_padding(self):
        value = torch.ones(2, 3, 4)
        self.assertEqual(preprocess.align_batch(value, 4, "value").shape[0], 4)
        with self.assertRaisesRegex(ValueError, "exact integer-multiple"):
            preprocess.align_batch(value, 3, "value")
        context = types_mod.PreparedFlux2FunContext(torch.ones(1, 2, 260), 2, 1, 2, 1)
        padded = preprocess.append_reference_zeros(context, 3, 2, device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(padded.shape, (2, 5, 260))
        self.assertEqual(torch.count_nonzero(padded[:, 2:]), 0)

    def test_token_mismatch_is_rejected_without_resize(self):
        control = torch.zeros(1, 128, 2, 2)
        inpaint = torch.zeros(1, 128, 2, 3)
        with self.assertRaisesRegex(ValueError, "match exactly"):
            preprocess.pack_control_context(control, torch.zeros(1, 1, 4, 4), inpaint)


class CheckpointTests(unittest.TestCase):
    def test_infers_official_profile_from_header_shapes(self):
        state = {}
        state["control_img_in.weight"] = torch.empty((6144, 260), device="meta")
        state["control_img_in.bias"] = torch.empty((6144,), device="meta")
        for block_id in range(4):
            prefix = f"control_transformer_blocks.{block_id}."
            for stream in ("", "add_"):
                for proj in ("q", "k", "v"):
                    state[prefix + f"attn.{stream}{proj}_proj.weight"] = torch.empty((6144, 6144), device="meta")
            for key in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
                state[prefix + f"attn.{key}.weight"] = torch.empty((128,), device="meta")
            state[prefix + "attn.to_out.0.weight"] = torch.empty((6144, 6144), device="meta")
            state[prefix + "attn.to_add_out.weight"] = torch.empty((6144, 6144), device="meta")
            for ff in ("ff", "ff_context"):
                state[prefix + f"{ff}.linear_in.weight"] = torch.empty((36864, 6144), device="meta")
                state[prefix + f"{ff}.linear_out.weight"] = torch.empty((6144, 18432), device="meta")
            state[prefix + "after_proj.weight"] = torch.empty((6144, 6144), device="meta")
            state[prefix + "after_proj.bias"] = torch.empty((6144,), device="meta")
        state["control_transformer_blocks.0.before_proj.weight"] = torch.empty((6144, 6144), device="meta")
        state["control_transformer_blocks.0.before_proj.bias"] = torch.empty((6144,), device="meta")
        profile = checkpoint.infer_checkpoint_profile(state)
        checkpoint.validate_official_2602_profile(profile)
        self.assertEqual(profile.tensor_count, 76)
        self.assertEqual((profile.num_heads, profile.head_dim), (48, 128))

    def test_prefix_collision_and_shape_mismatch_are_strict(self):
        tensor = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "collision"):
            checkpoint.normalize_state_dict_keys({"model.x": tensor, "x": tensor})
        profile = types_mod.CheckpointProfile("x", 1, 1, 1, 1, 1, 1, 1, (0,), "", "", {"x": (2,)})
        with self.assertRaisesRegex(ValueError, "wrong_shapes"):
            checkpoint.validate_state_dict_shapes({"x": tensor}, profile)


class BranchTests(unittest.TestCase):
    def test_four_block_branch_and_one_block_parity_interface(self):
        torch.manual_seed(2)
        model = branch_mod.Flux2FunControlBranch(tiny_profile(), TorchOps, dtype=torch.float32, attention_fn=simple_attention)
        image = torch.randn(1, 3, 8)
        context = torch.randn(1, 3, 6)
        text = torch.randn(1, 2, 8)
        vec = make_vec(1, 8)
        capture = {}
        hints = model.forward_control(image, context, text, vec, None, capture=capture)
        self.assertEqual(len(hints), 4)
        self.assertTrue(all(hint.shape == image.shape for hint in hints))
        initial = model.control_img_in(context)
        self.assertTrue(torch.allclose(capture["block_0_before_proj"], model.control_transformer_blocks[0].before_proj(initial)))
        direct = model.control_transformer_blocks[0](initial, image, text, vec, None, None, {}, simple_attention)
        parity = model.forward_one_block(0, initial, image, text, vec, None, None, {})
        for actual, expected in zip(parity, direct):
            self.assertTrue(torch.allclose(actual, expected))

    def test_reference_modulation_regions_use_distinct_batch_rows(self):
        tensor = torch.ones(1, 4, 2)
        scale = torch.tensor([[[2.0, 2.0], [3.0, 3.0]]])
        result = branch_mod._apply_modulation(tensor, scale, None, [(0, 2, 0), (2, 4, 1)])
        self.assertTrue(torch.all(result[:, :2] == 2))
        self.assertTrue(torch.all(result[:, 2:] == 3))

    def test_branch_compute_dtype_is_independent_from_storage_dtype(self):
        model = branch_mod.Flux2FunControlBranch(
            tiny_profile(),
            TorchOps,
            dtype=torch.float32,
            compute_dtype=torch.float64,
            attention_fn=simple_attention,
        )
        self.assertEqual(model.compute_dtype, torch.float64)


class TypeTests(unittest.TestCase):
    def test_group_is_immutable_and_preserves_descriptor_order(self):
        first = object()
        second = object()
        group = types_mod.Flux2FunControlGroup([first, second])
        self.assertEqual(group.descriptors, (first, second))
        with self.assertRaises(Exception):
            group.descriptors += (object(),)


class RuntimeTests(unittest.TestCase):
    def test_replacement_composes_then_injects_and_clears(self):
        dispatcher = runtime.Flux2FunDispatcher((), model_sampling=object())
        state = runtime.ForwardState((object(),), hints=(torch.ones(1, 2, 3),) * 4)
        args = {"img": torch.zeros(1, 2, 3), "txt": torch.zeros(1, 1, 3), "transformer_options": {runtime.STATE_KEY: state}}

        def existing(inner_args, extra):
            return {"img": inner_args["img"] + 2, "txt": inner_args["txt"]}

        out = dispatcher.replacement(6, existing)(args, {"original_block": None})
        self.assertTrue(torch.all(out["img"] == 3))
        self.assertIsNone(state.hints)

    def test_multi_control_hint_summation_order(self):
        class FakeBranch:
            def __init__(self, value):
                self.value = value

            def forward_control(self, image, *args, **kwargs):
                return (torch.full_like(image, self.value),) * 4

        class Handle:
            def __init__(self, value):
                self.model = FakeBranch(value)
                self.compute_dtype = torch.float32

        context = types_mod.PreparedFlux2FunContext(torch.zeros(1, 2, 260), 2, 1, 2, 1)
        descriptors = (
            types_mod.Flux2FunControlDescriptor(Handle(1), context, 0.5, 0, 1),
            types_mod.Flux2FunControlDescriptor(Handle(2), context, 2.0, 0, 1),
        )
        dispatcher = runtime.Flux2FunDispatcher(descriptors, object())
        args = {
            "img": torch.zeros(1, 2, 3),
            "txt": torch.zeros(1, 1, 3),
            "vec": make_vec(1, 3),
            "pe": None,
            "transformer_options": {},
        }
        hints = dispatcher._compute_hints(args, runtime.ForwardState(descriptors))
        self.assertTrue(all(torch.allclose(hint, torch.full_like(hint, 4.5)) for hint in hints))

    def test_fresh_forward_wrapper_state_does_not_leak(self):
        descriptor = types.SimpleNamespace(strength=0.0)
        dispatcher = runtime.Flux2FunDispatcher((descriptor,), object())
        seen = []

        def executor(x, timestep, context, y, guidance, refs, control, options, **kwargs):
            seen.append(options[runtime.STATE_KEY])
            return x

        original = {"kept": True}
        for _ in range(2):
            dispatcher.diffusion_wrapper(executor, torch.zeros(1), torch.zeros(1), None, transformer_options=original)
        self.assertIsNot(seen[0], seen[1])
        self.assertNotIn(runtime.STATE_KEY, original)

    def test_reapply_unwraps_prior_dispatcher_but_preserves_upstream(self):
        dispatcher = runtime.Flux2FunDispatcher((), object())

        def upstream(args, extra):
            return {"img": args["img"], "txt": args["txt"]}

        first = dispatcher.replacement(0, upstream)
        second = dispatcher.replacement(0, first.gen2_flux2_fun_upstream)
        self.assertIs(second.gen2_flux2_fun_upstream, upstream)

    def test_regular_reference_uses_global_modulation(self):
        vec = make_vec(1, 3, regions=1)
        self.assertIsNone(runtime._modulation_dims(4, 2, vec))
        zero_vec = make_vec(1, 3, regions=2)
        self.assertEqual(runtime._modulation_dims(4, 2, zero_vec), [(0, 2, 0), (2, 4, 1)])

    def test_multigpu_rebind_uses_cloned_control_patcher(self):
        original_patcher = types.SimpleNamespace(clone_base_uuid="control")
        cloned_module = object()
        cloned_patcher = types.SimpleNamespace(clone_base_uuid="control", model=cloned_module)
        handle = types_mod.Flux2FunModelHandle(
            model=object(),
            patcher=original_patcher,
            profile=tiny_profile(),
            storage_dtype=torch.float32,
            compute_dtype=torch.float32,
            checkpoint_path="test",
            checkpoint_sha256=None,
        )
        context = types_mod.PreparedFlux2FunContext(torch.zeros(1, 2, 260), 2, 1, 2, 1)
        descriptor = types_mod.Flux2FunControlDescriptor(handle, context, 1.0, 0.0, 1.0)
        dispatcher = runtime.Flux2FunDispatcher((descriptor,), object())

        class Clone:
            def __init__(self):
                self.attachment = dispatcher
                self.replacements = {}
                self.callbacks = {}

            def get_attachment(self, key):
                return self.attachment

            def set_attachments(self, key, value):
                self.attachment = value

            def get_additional_models_with_key(self, key):
                return [cloned_patcher]

            def get_model_object(self, key):
                return object()

            def remove_wrappers_with_key(self, *args):
                pass

            def add_wrapper_with_key(self, *args):
                pass

            def set_model_patch_replace(self, value, _name, _block_type, block_index):
                self.replacements[("double_block", block_index)] = value

            def remove_callbacks_with_key(self, *args):
                pass

            def add_callback_with_key(self, *args):
                pass

            model_options = {"transformer_options": {"patches_replace": {"dit": {}}}}

        cloned = Clone()
        runtime._rebind_multigpu_dispatcher(None, cloned)
        rebound = cloned.attachment.descriptors[0].model
        self.assertIs(rebound.patcher, cloned_patcher)
        self.assertIs(rebound.model, cloned_module)

    def test_strength_schedule_uses_model_sampling_percent_conversion(self):
        class Sampling:
            def percent_to_sigma(self, percent):
                return 1.0 - percent

            def timestep(self, sigma):
                return torch.as_tensor(sigma)

        descriptor = types_mod.Flux2FunControlDescriptor(types.SimpleNamespace(), types.SimpleNamespace(), 1.0, 0.25, 0.75)
        self.assertTrue(runtime._schedule_active(descriptor, torch.tensor([0.5]), Sampling()))
        self.assertFalse(runtime._schedule_active(descriptor, torch.tensor([0.9]), Sampling()))

    def test_reference_count_mismatch_is_descriptive(self):
        context = types_mod.PreparedFlux2FunContext(torch.zeros(1, 2, 260), 2, 1, 2, 1)
        descriptor = types_mod.Flux2FunControlDescriptor(types.SimpleNamespace(compute_dtype=torch.float32), context, 1, 0, 1)
        dispatcher = runtime.Flux2FunDispatcher((descriptor,), object())
        args = {
            "img": torch.zeros(1, 4, 3),
            "txt": torch.zeros(1, 1, 3),
            "vec": make_vec(1, 3),
            "pe": None,
            "transformer_options": {"reference_image_num_tokens": [1]},
        }
        with self.assertRaisesRegex(ValueError, "target_latent"):
            dispatcher._compute_hints(args, runtime.ForwardState((descriptor,)))


class ModelManagementTests(unittest.TestCase):
    def test_auto_precision_uses_comfy_dtype_policy(self):
        class Management:
            @staticmethod
            def unet_dtype(**kwargs):
                self.assertEqual(kwargs["weight_dtype"], torch.bfloat16)
                return torch.float16

        resolved = model_management_mod.resolve_precision(
            "auto",
            torch.bfloat16,
            model_management=Management,
            device=torch.device("cuda"),
            model_params=123,
        )
        self.assertEqual(resolved, torch.float16)

    def test_explicit_unsupported_precision_fails_early(self):
        class Management:
            @staticmethod
            def should_use_bf16(**kwargs):
                return False

            @staticmethod
            def should_use_fp16(**kwargs):
                return False

        with self.assertRaisesRegex(ValueError, "not supported"):
            model_management_mod.resolve_precision(
                "bf16",
                torch.float32,
                model_management=Management,
                device=torch.device("cpu"),
            )

    def test_explicit_supported_precision_is_selected(self):
        class Management:
            @staticmethod
            def should_use_bf16(**kwargs):
                return True

            @staticmethod
            def should_use_fp16(**kwargs):
                return True

        self.assertEqual(
            model_management_mod.resolve_precision(
                "fp16",
                torch.bfloat16,
                model_management=Management,
                device=torch.device("cuda"),
                model_params=123,
            ),
            torch.float16,
        )


class OracleTests(unittest.TestCase):
    def test_machine_readable_report(self):
        result = oracle.compare_tensors("packed_context", torch.ones(2), torch.ones(2), atol=1e-5, rtol=1e-5)
        self.assertTrue(result.passed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            oracle.write_oracle_report(path, [result], {"device": "cpu"})
            self.assertIn('"schema_version": 1', path.read_text(encoding="utf-8"))

    def test_tensor_bundle_requires_complete_oracle_set(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing required"):
                oracle.save_tensor_bundle(Path(directory) / "empty.pt", {})
            tensors = {name: torch.zeros(1) for name in oracle.ORACLE_TENSOR_NAMES}
            oracle.save_tensor_bundle(Path(directory) / "complete.pt", tensors)


if __name__ == "__main__":
    unittest.main()
