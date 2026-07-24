from __future__ import annotations

import torch

from .branch import Flux2FunControlBranch
from .checkpoint import (
    OFFICIAL_2602_SHA256,
    infer_checkpoint_profile,
    sha256_file,
    strict_load_control_branch,
    validate_official_2602_profile,
)
from .types import Flux2FunModelHandle


def resolve_precision(
    precision: str,
    stored_dtype: torch.dtype,
    *,
    model_management=None,
    device: torch.device | None = None,
    model_params: int = 0,
) -> torch.dtype:
    if precision not in {"auto", "bf16", "fp16"}:
        raise ValueError(f"Unknown Flux2 Fun precision: {precision}")
    if model_management is None:
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        return stored_dtype if stored_dtype in (torch.bfloat16, torch.float16) else torch.bfloat16

    supported = [torch.bfloat16, torch.float16, torch.float32]
    if precision == "auto":
        return model_management.unet_dtype(
            device=device,
            model_params=model_params,
            supported_dtypes=supported,
            weight_dtype=stored_dtype,
        )

    requested = torch.bfloat16 if precision == "bf16" else torch.float16
    support_check = model_management.should_use_bf16 if requested == torch.bfloat16 else model_management.should_use_fp16
    if not support_check(device=device, model_params=model_params, manual_cast=True):
        raise ValueError(
            f"Flux2 Fun {precision} compute is not supported on device {device}; "
            "use auto or select a supported accelerator."
        )
    return requested


def load_control_branch_patcher(checkpoint_path: str, precision: str = "auto", disable_dynamic: bool = False):
    del disable_dynamic
    return load_managed_control_branch(checkpoint_path, precision).patcher


def load_managed_control_branch(checkpoint_path: str, precision: str = "auto") -> Flux2FunModelHandle:
    import comfy.model_management as model_management
    import comfy.ops
    import comfy.utils
    from comfy.model_patcher import ModelPatcher

    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != OFFICIAL_2602_SHA256:
        raise ValueError(
            "Flux2 Fun checkpoint SHA256 mismatch. Only the official 2602 file is supported: "
            f"expected {OFFICIAL_2602_SHA256}, got {checkpoint_hash}."
        )

    state_dict = comfy.utils.load_torch_file(checkpoint_path, safe_load=True)
    if not state_dict:
        raise ValueError(f"Flux2 Fun checkpoint is empty: {checkpoint_path}")
    profile = infer_checkpoint_profile(state_dict)
    validate_official_2602_profile(profile)

    first_tensor = next(iter(state_dict.values()))
    storage_dtype = first_tensor.dtype
    load_device = model_management.get_torch_device()
    offload_device = model_management.unet_offload_device()
    model_params = sum(int(tensor.numel()) for tensor in state_dict.values())
    compute_dtype = resolve_precision(
        precision,
        storage_dtype,
        model_management=model_management,
        device=load_device,
        model_params=model_params,
    )
    operations = comfy.ops.pick_operations(storage_dtype, compute_dtype, load_device=load_device)
    branch = Flux2FunControlBranch(
        profile,
        operations,
        dtype=storage_dtype,
        device=offload_device,
        compute_dtype=compute_dtype,
    )
    strict_load_control_branch(branch, state_dict, profile)
    branch.eval()

    patcher = ModelPatcher(
        branch,
        load_device=load_device,
        offload_device=offload_device,
        size=model_management.module_size(branch),
    )
    patcher.cached_patcher_init = (load_control_branch_patcher, (checkpoint_path, precision))
    return Flux2FunModelHandle(
        model=branch,
        patcher=patcher,
        profile=profile,
        storage_dtype=storage_dtype,
        compute_dtype=compute_dtype,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
    )
