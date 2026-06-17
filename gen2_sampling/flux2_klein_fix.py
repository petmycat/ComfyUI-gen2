"""
Gen2 Flux.2 [klein] Fix

Bypasses the ComfyUI regression introduced by commit 44f1246
("Support flux 2 klein kv cache model: Use the FluxKVCache node." #12905),
which was bisected as the exact commit that changed Flux.2 [klein] output for
normal `index` reference-latent workflows (e.g. masked inpaint) on ComfyUI
>= v0.17.0 — even though the new KV-cache / `index_timestep_zero` code paths
are gated off for those workflows.

This node restores the pre-44f1246 (v0.16.4) implementations of the flux
transformer's `forward_orig` and `_forward` *only during this model's
sampling*, by temporarily swapping them onto `comfy.ldm.flux.model.Flux` inside
a unet-function wrapper and restoring them immediately afterward. No ComfyUI
core files are modified, so it survives `git pull` upgrades.

The vendored functions below are copied verbatim from
comfy/ldm/flux/model.py at commit 44f1246~1 (the last good revision).
"""

import torch
from einops import rearrange

import comfy.ldm.flux.model as _flux_model
from comfy.ldm.flux.layers import timestep_embedding


# ---------------------------------------------------------------------------
# Vendored good (pre-44f1246) flux forward_orig
# ---------------------------------------------------------------------------
def good_forward_orig(
    self,
    img,
    img_ids,
    txt,
    txt_ids,
    timesteps,
    y,
    guidance=None,
    control=None,
    transformer_options={},
    attn_mask=None,
    **kwargs,
):
    transformer_options = transformer_options.copy()
    patches = transformer_options.get("patches", {})
    patches_replace = transformer_options.get("patches_replace", {})
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

    if self.vector_in is not None:
        if y is None:
            y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

    if self.txt_norm is not None:
        txt = self.txt_norm(txt)
    txt = self.txt_in(txt)

    # The crux of the fix: compute modulation here (before post_input / pe),
    # exactly like v0.16.4. 44f1246 moved this block to after pe.
    vec_orig = vec
    if self.params.global_modulation:
        vec = (self.double_stream_modulation_img(vec_orig), self.double_stream_modulation_txt(vec_orig))

    if "post_input" in patches:
        for p in patches["post_input"]:
            out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids, "transformer_options": transformer_options})
            img = out["img"]
            txt = out["txt"]
            img_ids = out["img_ids"]
            txt_ids = out["txt_ids"]

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None

    blocks_replace = patches_replace.get("dit", {})
    transformer_options["total_blocks"] = len(self.double_blocks)
    transformer_options["block_type"] = "double"
    for i, block in enumerate(self.double_blocks):
        transformer_options["block_index"] = i
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"], out["txt"] = block(img=args["img"],
                                               txt=args["txt"],
                                               vec=args["vec"],
                                               pe=args["pe"],
                                               attn_mask=args.get("attn_mask"),
                                               transformer_options=args.get("transformer_options"))
                return out

            out = blocks_replace[("double_block", i)]({"img": img,
                                                       "txt": txt,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask,
                                                       "transformer_options": transformer_options},
                                                      {"original_block": block_wrap})
            txt = out["txt"]
            img = out["img"]
        else:
            img, txt = block(img=img,
                             txt=txt,
                             vec=vec,
                             pe=pe,
                             attn_mask=attn_mask,
                             transformer_options=transformer_options)

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img[:, :add.shape[1]] += add

    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

    img = torch.cat((txt, img), 1)

    if self.params.global_modulation:
        vec, _ = self.single_stream_modulation(vec_orig)

    transformer_options["total_blocks"] = len(self.single_blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [txt.shape[1], img.shape[1]]
    for i, block in enumerate(self.single_blocks):
        transformer_options["block_index"] = i
        if ("single_block", i) in blocks_replace:
            def block_wrap(args):
                out = {}
                out["img"] = block(args["img"],
                                   vec=args["vec"],
                                   pe=args["pe"],
                                   attn_mask=args.get("attn_mask"),
                                   transformer_options=args.get("transformer_options"))
                return out

            out = blocks_replace[("single_block", i)]({"img": img,
                                                       "vec": vec,
                                                       "pe": pe,
                                                       "attn_mask": attn_mask,
                                                       "transformer_options": transformer_options},
                                                      {"original_block": block_wrap})
            img = out["img"]
        else:
            img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, transformer_options=transformer_options)

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1]: txt.shape[1] + add.shape[1], ...] += add

    img = img[:, txt.shape[1]:, ...]

    img = self.final_layer(img, vec_orig)  # (N, T, patch_size ** 2 * out_channels)
    return img


# ---------------------------------------------------------------------------
# Vendored good (pre-44f1246) flux _forward
# ---------------------------------------------------------------------------
def good_underscore_forward(
    self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None, transformer_options={}, **kwargs
):
    bs, c, h_orig, w_orig = x.shape
    patch_size = self.patch_size

    h_len = ((h_orig + (patch_size // 2)) // patch_size)
    w_len = ((w_orig + (patch_size // 2)) // patch_size)
    img, img_ids = self.process_img(x, transformer_options=transformer_options)
    img_tokens = img.shape[1]
    if ref_latents is not None:
        h = 0
        w = 0
        index = 0
        ref_latents_method = kwargs.get("ref_latents_method", self.params.default_ref_method)
        for ref in ref_latents:
            if ref_latents_method == "index":
                index += self.params.ref_index_scale
                h_offset = 0
                w_offset = 0
            elif ref_latents_method == "uxo":
                index = 0
                h_offset = h_len * patch_size + h
                w_offset = w_len * patch_size + w
                h += ref.shape[-2]
                w += ref.shape[-1]
            else:
                index = 1
                h_offset = 0
                w_offset = 0
                if ref.shape[-2] + h > ref.shape[-1] + w:
                    w_offset = w
                else:
                    h_offset = h
                h = max(h, ref.shape[-2] + h_offset)
                w = max(w, ref.shape[-1] + w_offset)

            kontext, kontext_ids = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
            img = torch.cat([img, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)

    txt_ids = torch.zeros((bs, context.shape[1], len(self.params.axes_dim)), device=x.device, dtype=torch.float32)

    if len(self.params.txt_ids_dims) > 0:
        for i in self.params.txt_ids_dims:
            txt_ids[:, :, i] = torch.linspace(0, context.shape[1] - 1, steps=context.shape[1], device=x.device, dtype=torch.float32)

    out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control, transformer_options, attn_mask=kwargs.get("attention_mask", None))
    out = out[:, :img_tokens]
    return rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=self.patch_size, pw=self.patch_size)[:, :, :h_orig, :w_orig]


class _KleinFixWrapper:
    """unet_function_wrapper that swaps the good flux forward in for the
    duration of one apply_model call, then restores the originals."""

    def __call__(self, apply_model, params):
        cls = _flux_model.Flux
        saved_forward_orig = cls.forward_orig
        saved_forward = cls._forward
        cls.forward_orig = good_forward_orig
        cls._forward = good_underscore_forward
        try:
            return apply_model(params["input"], params["timestep"], **params["c"])
        finally:
            cls.forward_orig = saved_forward_orig
            cls._forward = saved_forward


class Gen2_Flux2KleinFix:
    """
    Restore pre-#12905 (commit 44f1246) Flux.2 [klein] sampling behavior on
    ComfyUI >= v0.17.0 without editing any core files.

    Wire it between your "Load Diffusion Model" output and the guider/sampler.
    Use it for Flux.2 klein workflows (especially masked inpaint with
    ReferenceLatent) that changed after upgrading ComfyUI past v0.16.4.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Gen2/Sampling"
    DESCRIPTION = "Restore pre-44f1246 (#12905) Flux.2 klein forward; fixes index reference-latent / inpaint regression."

    def patch(self, model):
        m = model.clone()
        m.set_model_unet_function_wrapper(_KleinFixWrapper())
        print("[Gen2] Flux.2 klein fix applied (reverting #12905 forward_orig/_forward for this model).")
        return (m,)
