from __future__ import annotations

import contextvars
import logging
import math
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

try:
    from ..ideogram4_trigger.types import PhaseCConditioningState
except ImportError:
    from ideogram4_trigger.types import PhaseCConditioningState

from .router import PhaseCRouter, aggregate_projected_occurrences, effective_gates, normalize_canonical_timestep
from .types import PhaseCRegistryModule, PhaseCRouterBundle

LOGGER = logging.getLogger(__name__)
ATTACHMENT_KEY = "gen2.ideogram4_phase_c_v2"
WRAPPER_KEY = "gen2.ideogram4_phase_c_v2.diffusion"
INJECTION_KEY = "gen2.ideogram4_phase_c_v2.lora"
_CURRENT_GATES: contextvars.ContextVar[torch.Tensor | None] = contextvars.ContextVar(
    "gen2_phase_c_v2_gates", default=None
)
_CURRENT_OWNER: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "gen2_phase_c_v2_owner", default=None
)
_DISPATCHER_LOCK = threading.RLock()
_MODEL_LOCKS_GUARD = threading.RLock()
_MODEL_LOCKS: weakref.WeakKeyDictionary[Any, threading.RLock] = weakref.WeakKeyDictionary()


class PhaseCRuntimeError(RuntimeError):
    pass


def _model_lock(model: Any) -> threading.RLock:
    with _MODEL_LOCKS_GUARD:
        lock = _MODEL_LOCKS.get(model)
        if lock is None:
            lock = threading.RLock()
            _MODEL_LOCKS[model] = lock
        return lock


def normalize_module_name(name: str) -> str:
    value = str(name).removesuffix(".weight").replace("$$", ".")
    if value.startswith("diffusion_model."):
        value = "transformer." + value[len("diffusion_model."):]
    return value


def _adapter_weights(adapter: Any) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    weights = getattr(adapter, "weights", None)
    if not isinstance(weights, (list, tuple)) or len(weights) != 6:
        raise PhaseCRuntimeError("Phase C V3 patches must be ComfyUI LoRAAdapter instances")
    up, down, alpha, mid, dora_scale, reshape = weights
    if not isinstance(up, torch.Tensor) or not isinstance(down, torch.Tensor):
        raise PhaseCRuntimeError("Phase C V3 LoRA adapter lacks up/down tensors")
    if mid is not None or dora_scale is not None or reshape is not None:
        raise PhaseCRuntimeError("Phase C V2 supports only plain linear V3 LoRA adapters")
    return up, down, None if alpha is None else float(alpha)


def _fp32_norm(tensor: torch.Tensor) -> float:
    value = tensor.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(value).all().item():
        raise PhaseCRuntimeError("V3 LoRA tensor contains NaN or infinity")
    return float(torch.linalg.vector_norm(value).item())


def _norm_matches(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-5, abs_tol=1.0e-7)


@dataclass(frozen=True, slots=True)
class BoundV3Adapter:
    model_key: str
    module_name: str
    group_index: int
    adapter: Any
    multiplier: float


def _match_registry_row(key: str, patch: tuple[Any, ...], row: PhaseCRegistryModule) -> BoundV3Adapter | None:
    if len(patch) != 5:
        raise PhaseCRuntimeError(f"Unexpected ComfyUI patch tuple for {key!r}")
    strength_patch, adapter, strength_model, offset, function = patch
    if normalize_module_name(key) != row.module_name:
        return None
    weights = getattr(adapter, "weights", None)
    if not isinstance(weights, (list, tuple)) or len(weights) != 6:
        return None
    up, down, _ = _adapter_weights(adapter)
    if tuple(down.shape) != row.down_shape or tuple(up.shape) != row.up_shape:
        return None
    if down.shape[0] != row.rank or up.shape[1] != row.rank:
        return None
    if not _norm_matches(_fp32_norm(down), row.down_fp32_norm) or not _norm_matches(
        _fp32_norm(up), row.up_fp32_norm
    ):
        return None
    if offset is not None or function is not None:
        raise PhaseCRuntimeError(f"V3 LoRA {row.module_name!r} uses unsupported offset/function patching")
    if float(strength_model) != 1.0:
        raise PhaseCRuntimeError(
            f"V3 LoRA {row.module_name!r} uses strength_model={strength_model}; dynamic residual conversion requires 1.0"
        )
    multiplier = float(strength_patch)
    if not math.isfinite(multiplier):
        raise PhaseCRuntimeError(f"V3 LoRA multiplier is non-finite for {row.module_name!r}")
    return BoundV3Adapter(key, row.module_name, row.group_index, adapter, multiplier)


def extract_v3_adapters(model: Any, bundle: PhaseCRouterBundle) -> tuple[Any, tuple[BoundV3Adapter, ...]]:
    if not callable(getattr(model, "clone", None)) or not isinstance(getattr(model, "patches", None), dict):
        raise PhaseCRuntimeError("Phase C Strength requires a ComfyUI MODEL with inspectable LoRA patches")
    clone = model.clone()
    registry_by_name = {row.module_name: row for row in bundle.registry.modules}
    matched: dict[str, BoundV3Adapter] = {}
    updated_patches: dict[str, list[tuple[Any, ...]]] = {}
    for key, patch_list in clone.patches.items():
        retained: list[tuple[Any, ...]] = []
        normalized = normalize_module_name(key)
        row = registry_by_name.get(normalized)
        for patch in patch_list:
            bound = None if row is None else _match_registry_row(key, patch, row)
            if bound is None:
                retained.append(patch)
                continue
            if row.module_name in matched:
                raise PhaseCRuntimeError(f"Multiple V3 LoRA patches matched {row.module_name!r}")
            matched[row.module_name] = bound
        if retained:
            updated_patches[key] = retained
    missing = sorted(set(registry_by_name).difference(matched))
    if missing:
        raise PhaseCRuntimeError(
            f"MODEL does not contain the exact Phase C V3 registry; missing {len(missing)} module(s): {missing[:8]}"
        )
    clone.patches = updated_patches
    try:
        import uuid

        clone.patches_uuid = uuid.uuid4()
    except (ImportError, AttributeError):
        clone.patches_uuid = object()
    return clone, tuple(matched[name] for name in sorted(matched))


def _resolve_module(root: Any, model_key: str) -> Any:
    module_name = model_key.removesuffix(".weight")
    module = root
    for part in module_name.split("."):
        try:
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise PhaseCRuntimeError(f"Unable to resolve V3 module {module_name!r}") from exc
    return module


class SharedLoRADispatcher:
    def __init__(self, module: Any) -> None:
        self.module = module
        self.original_forward = module.forward
        self.controllers: dict[object, BoundV3Adapter] = {}
        module.forward = self.forward

    def add(self, owner: object, binding: BoundV3Adapter) -> None:
        if owner in self.controllers:
            raise PhaseCRuntimeError("Phase C dispatcher owner is already registered")
        self.controllers[owner] = binding

    def remove(self, owner: object) -> None:
        self.controllers.pop(owner, None)
        if not self.controllers:
            if self.module.forward != self.forward:
                raise PhaseCRuntimeError("V3 module forward changed while Phase C dispatcher was installed")
            self.module.forward = self.original_forward
            delattr(self.module, "_gen2_phase_c_dispatcher")

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        base = self.original_forward(inputs, *args, **kwargs)
        owner = _CURRENT_OWNER.get()
        if owner is None:
            return base
        binding = self.controllers.get(owner)
        if binding is None:
            raise PhaseCRuntimeError("Phase C dispatcher has no controller for the active MODEL")
        gates = _CURRENT_GATES.get()
        if gates is None:
            raise PhaseCRuntimeError("Phase C V3 residual executed without call-local gates")
        up, down, alpha = _adapter_weights(binding.adapter)
        rank = down.shape[0]
        scale = binding.multiplier * (1.0 if alpha is None else alpha / rank)
        down = down.to(device=inputs.device, dtype=inputs.dtype)
        up = up.to(device=inputs.device, dtype=inputs.dtype)
        residual = F.linear(F.linear(inputs, down), up) * scale
        gate = gates[:, binding.group_index]
        if residual.shape[0] != gate.shape[0]:
            if residual.shape[0] % gate.shape[0] != 0:
                raise PhaseCRuntimeError("V3 residual batch is incompatible with Phase C gate batch")
            gate = gate.repeat(residual.shape[0] // gate.shape[0])
        while gate.ndim < residual.ndim:
            gate = gate.unsqueeze(-1)
        return base + residual.to(base.dtype) * gate.to(device=base.device, dtype=base.dtype)


def _dispatcher(module: Any) -> SharedLoRADispatcher:
    existing = getattr(module, "_gen2_phase_c_dispatcher", None)
    if existing is not None:
        if not isinstance(existing, SharedLoRADispatcher):
            raise PhaseCRuntimeError("V3 module has an incompatible Phase C dispatcher")
        return existing
    dispatcher = SharedLoRADispatcher(module)
    module._gen2_phase_c_dispatcher = dispatcher
    return dispatcher


def _create_injection(model_root: Any, bindings: tuple[BoundV3Adapter, ...], owner: object) -> Any:
    try:
        from comfy.patcher_extension import PatcherInjection
    except ImportError as exc:
        raise PhaseCRuntimeError("Installed ComfyUI lacks PatcherInjection") from exc
    modules = [(_resolve_module(model_root, binding.model_key), binding) for binding in bindings]

    def inject(_model_patcher):
        installed: list[SharedLoRADispatcher] = []
        with _DISPATCHER_LOCK:
            try:
                for module, binding in modules:
                    dispatcher = _dispatcher(module)
                    dispatcher.add(owner, binding)
                    installed.append(dispatcher)
            except BaseException:
                for dispatcher in reversed(installed):
                    dispatcher.remove(owner)
                raise

    def eject(_model_patcher):
        errors: list[BaseException] = []
        with _DISPATCHER_LOCK:
            for module, _binding in reversed(modules):
                try:
                    dispatcher = getattr(module, "_gen2_phase_c_dispatcher", None)
                    if dispatcher is not None:
                        dispatcher.remove(owner)
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise PhaseCRuntimeError(f"Failed to eject {len(errors)} Phase C V3 controllers") from errors[0]

    return PatcherInjection(inject=inject, eject=eject)


def _extract_transformer_options(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    value = kwargs.get("transformer_options")
    if isinstance(value, Mapping):
        return value
    for item in reversed(args):
        if isinstance(item, Mapping) and ("cond_or_uncond" in item or "uuids" in item):
            return item
    return {}


def _project_activator_states(
    diffusion_model: Any,
    state: PhaseCConditioningState,
    device: torch.device,
    expected_projection_width: int,
) -> torch.Tensor:
    norm = getattr(diffusion_model, "llm_cond_norm", None)
    projection = getattr(diffusion_model, "llm_cond_proj", None)
    if not callable(norm) or not callable(projection):
        raise PhaseCRuntimeError("Ideogram4 diffusion model lacks llm_cond_norm/llm_cond_proj")
    selected = state.occurrence_states.to(device=device, dtype=torch.float32).reshape(-1, state.conditioning_width)
    norm_shape = getattr(norm, "normalized_shape", None)
    if norm_shape is not None:
        expected_width = int(norm_shape[-1] if isinstance(norm_shape, (tuple, list)) else norm_shape)
        if expected_width != state.conditioning_width:
            raise PhaseCRuntimeError(
                f"Ideogram llm_cond_norm expects width {expected_width}, Phase C conditioning state has {state.conditioning_width}"
            )
    weight = getattr(projection, "weight", None)
    in_features = getattr(projection, "in_features", None)
    if isinstance(in_features, int) and in_features != state.conditioning_width:
        raise PhaseCRuntimeError(
            f"Ideogram llm_cond_proj expects width {in_features}, Phase C conditioning state has {state.conditioning_width}"
        )
    projection_dtype = weight.dtype if isinstance(weight, torch.Tensor) else selected.dtype
    selected = selected.to(dtype=projection_dtype)
    projected = projection(norm(selected)).float()
    if not isinstance(projected, torch.Tensor) or projected.ndim != 2:
        raise PhaseCRuntimeError("Ideogram4 activator projection did not return [tokens,D]")
    if projected.shape[-1] != expected_projection_width:
        raise PhaseCRuntimeError(
            f"Ideogram4 activator projection returned width {projected.shape[-1]}, Router expects {expected_projection_width}"
        )
    if not torch.isfinite(projected).all().item():
        raise PhaseCRuntimeError("Projected Phase C activator states contain NaN or infinity")
    return aggregate_projected_occurrences(projected, state.occurrence_count, state.virtual_token_count)


def _expand_cfg_gates(
    routed: torch.Tensor,
    cond_or_uncond: list[int],
    input_batch: int,
) -> torch.Tensor:
    if not cond_or_uncond:
        cond_or_uncond = [0]
    chunks = len(cond_or_uncond)
    if input_batch % chunks != 0:
        raise PhaseCRuntimeError("ComfyUI CFG batch cannot be divided into conditioning chunks")
    rows_per_chunk = input_batch // chunks
    if routed.shape[0] not in (1, rows_per_chunk):
        raise PhaseCRuntimeError("Phase C routed batch does not match ComfyUI conditioning batch")
    routed_rows = routed.expand(rows_per_chunk, -1) if routed.shape[0] == 1 else routed
    ordinary = torch.ones_like(routed_rows)
    return torch.cat([routed_rows if int(index) == 0 else ordinary for index in cond_or_uncond], dim=0)


def apply_phase_c_strength(
    model: Any,
    conditioning: list[list[Any]],
    bundle: PhaseCRouterBundle,
    style_strength: float,
    debug_logging: bool = False,
) -> tuple[Any, list[list[Any]]]:
    strength = float(style_strength)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("style_strength must be finite and in [0,1]")
    if not isinstance(conditioning, list) or len(conditioning) != 1 or len(conditioning[0]) != 2:
        raise PhaseCRuntimeError("Phase C V2 currently requires exactly one CONDITIONING entry")
    metadata = conditioning[0][1]
    state = metadata.get("gen2_phase_c_v2") if isinstance(metadata, Mapping) else None
    if state is None:
        raise PhaseCRuntimeError(
            "CONDITIONING lacks Phase C V2 activator state; use V9 Trigger Text Encode in semantic_only mode "
            "with exactly three literal occurrences"
        )
    if not isinstance(state, PhaseCConditioningState) or not state.routable:
        raise PhaseCRuntimeError("CONDITIONING does not carry a routable Phase C V2 state")
    if bundle.config.activator_occurrence_count != state.occurrence_count:
        raise PhaseCRuntimeError("Router and CONDITIONING occurrence contracts differ")
    if state.embedding_file_sha256 != bundle.expected_embedding_sha256:
        raise PhaseCRuntimeError("CONDITIONING embedding artifact does not match the Router training source")
    if state.te_adapter_file_sha256 != bundle.expected_te_adapter_sha256:
        raise PhaseCRuntimeError("CONDITIONING TE adapter artifact does not match the Router training source")
    clone, bindings = extract_v3_adapters(model, bundle)
    base_model = getattr(clone, "model", None)
    diffusion_model = getattr(base_model, "diffusion_model", None)
    if diffusion_model is None or "ideogram" not in (
        type(base_model).__module__ + type(base_model).__name__ + type(diffusion_model).__module__ + type(diffusion_model).__name__
    ).lower():
        raise PhaseCRuntimeError("Phase C V2 requires a native Ideogram diffusion MODEL")
    router = PhaseCRouter(bundle)
    clone.attachments[ATTACHMENT_KEY] = {
        "bundle": bundle,
        "conditioning_state": state,
        "style_strength": strength,
        "binding_count": len(bindings),
    }

    try:
        from comfy.patcher_extension import CallbacksMP, WrappersMP
    except ImportError as exc:
        raise PhaseCRuntimeError("Installed ComfyUI lacks model wrapper support") from exc

    def install_runtime(target: Any) -> None:
        target_owner = object()
        target_injection = _create_injection(base_model, bindings, target_owner)

        def diffusion_wrapper(executor, *args, **kwargs):
            if len(args) < 2 or not isinstance(args[0], torch.Tensor) or not isinstance(args[1], torch.Tensor):
                raise PhaseCRuntimeError("Unexpected Ideogram4 diffusion wrapper signature")
            if executor.class_obj is not diffusion_model:
                raise PhaseCRuntimeError("Phase C wrapper was invoked by an unexpected diffusion model")
            input_tensor = args[0]
            canonical_timestep = args[1]
            options = _extract_transformer_options(args, kwargs)
            cond_or_uncond = list(options.get("cond_or_uncond", [0]))
            with _model_lock(executor.class_obj):
                projected = _project_activator_states(
                    executor.class_obj,
                    state,
                    input_tensor.device,
                    bundle.config.conditioning_dim,
                )
                code = router.encode_activator(projected)
                tau = normalize_canonical_timestep(canonical_timestep.float() / 1000.0)
                q = router(tau[:1], code)
                routed = effective_gates(q, strength)
                gates = _expand_cfg_gates(routed, cond_or_uncond, input_tensor.shape[0])
                if debug_logging:
                    LOGGER.info(
                        "[Gen2 Phase C debug] tau=%.6f strength=%.2f mean_abs_q=%.6f max_abs_q=%.6f gate_min=%.6f gate_max=%.6f",
                        float(tau[0].item()),
                        strength,
                        float(q.abs().mean().item()),
                        float(q.abs().max().item()),
                        float(gates.min().item()),
                        float(gates.max().item()),
                    )
                gate_token = _CURRENT_GATES.set(gates)
                owner_token = _CURRENT_OWNER.set(target_owner)
                try:
                    return executor(*args, **kwargs)
                finally:
                    _CURRENT_OWNER.reset(owner_token)
                    _CURRENT_GATES.reset(gate_token)

        def on_clone(source: Any, cloned: Any) -> None:
            if getattr(source, "is_injected", False) or getattr(cloned, "is_injected", False):
                raise PhaseCRuntimeError("Cannot clone a Phase C MODEL while its runtime injection is active")
            install_runtime(cloned)

        target.remove_injections(INJECTION_KEY)
        target.set_injections(INJECTION_KEY, [target_injection])
        target.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)
        target.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, diffusion_wrapper)
        target.remove_callbacks_with_key(CallbacksMP.ON_CLONE, WRAPPER_KEY)
        target.add_callback_with_key(CallbacksMP.ON_CLONE, WRAPPER_KEY, on_clone)

    install_runtime(clone)
    return clone, conditioning


__all__ = [
    "ATTACHMENT_KEY",
    "BoundV3Adapter",
    "PhaseCRuntimeError",
    "apply_phase_c_strength",
    "extract_v3_adapters",
    "normalize_module_name",
]
