# QwenImage ControlNet Fun - All ComfyUI Node Definitions
# Thin interface layer: node definitions import logic from core/

"""
Gen2 QwenImage Nodes - ComfyUI Node Classes

All 6 QwenImage node classes. These are thin wrappers around the core logic modules.
Each node defines INPUT_TYPES, RETURN_TYPES, etc. and delegates to core/ for actual work.
"""

import os
import sys
import gc
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import comfy.model_management as mm
import comfy.utils
import comfy.latent_formats
import folder_paths

from .core import (
    DIFFUSERS_AVAILABLE,
    QUANTIZED_DTYPES, get_compute_dtype, get_autocast_dtype,
    FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE, SAGE_ATTENTION_AVAILABLE,
    gen2_merge_lora, gen2_unmerge_lora,
    QwenImageControlModel,
    Gen2QwenImageModelWrapper,
    filter_kwargs, get_qwen_scheduler,
    QWEN_VAE_CONFIG, calculate_shift, retrieve_timesteps_v2, pack_latents_v2, unpack_latents_v2,
    get_gen2_tokenizer, VIDEOX_PROMPT_TEMPLATE, VIDEOX_DROP_IDX, VIDEOX_TOKENIZER_MAX_LENGTH,
)

from .core.imports import VaeImageProcessor


# =============================================================================
# Node 1: Load ControlNet
# =============================================================================

class Gen2_LoadQwenControlNetFun:
    """
    Loads VideoX Fun's QwenImage ControlNet weights into our standalone model.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Support both controlnet and model_patches folders
        controlnet_files = []
        
        # Get files from controlnet folder
        try:
            controlnet_files.extend(folder_paths.get_filename_list("controlnet"))
        except:
            pass
        
        # Get files from model_patches folder (where VideoX Fun ControlNet is often placed)
        try:
            model_patches = folder_paths.get_filename_list("model_patches")
            for f in model_patches:
                if "controlnet" in f.lower() or "qwen" in f.lower():
                    if f not in controlnet_files:
                        controlnet_files.append(f)
        except:
            pass
        
        if not controlnet_files:
            controlnet_files = ["No ControlNet files found"]
        
        return {
            "required": {
                "controlnet_name": (controlnet_files, ),
            }
        }
    
    RETURN_TYPES = ("GEN2_CONTROLNET",)
    RETURN_NAMES = ("controlnet",)
    FUNCTION = "load"
    CATEGORY = "Gen2/QwenImage/ControlNet"
    
    def load(self, controlnet_name):
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers is required for QwenImage ControlNet")
        
        # Try multiple folders to find the file
        controlnet_path = None
        for folder_type in ["controlnet", "model_patches"]:
            try:
                path = folder_paths.get_full_path(folder_type, controlnet_name)
                if path and os.path.exists(path):
                    controlnet_path = path
                    break
            except:
                pass
        
        if controlnet_path is None:
            raise FileNotFoundError(f"ControlNet file not found: {controlnet_name}")
        
        print(f"[Gen2] Loading QwenImage ControlNet Fun: {controlnet_name}")
        
        # Load state dict
        state_dict = comfy.utils.load_torch_file(controlnet_path)
        
        # VideoX control config
        control_layers = [0, 12, 24, 36, 48]
        control_in_dim = 132
        inner_dim = 3072
        num_attention_heads = 24
        attention_head_dim = 128
        
        # Create control model
        control_model = QwenImageControlModel(
            control_layers=control_layers,
            control_in_dim=control_in_dim,
            inner_dim=inner_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
        )
        
        # Filter control-specific weights
        control_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('control_'):
                control_state_dict[key] = value
        
        # Load into model
        missing, unexpected = control_model.load_state_dict(control_state_dict, strict=False)
        
        print(f"[Gen2] ControlNet loaded: {len(control_layers)} blocks at layers {control_layers}")
        if missing:
            print(f"[Gen2] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[Gen2] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        
        return ({
            'model': control_model,
            'control_layers': control_layers,
            'control_in_dim': control_in_dim,
        },)


# =============================================================================
# Node 2: Apply ControlNet
# =============================================================================

class Gen2_ApplyQwenControlNetFun:
    """
    Applies QwenImage ControlNet to the model.
    Uses GEN2_VAE for proper VideoX-compatible encoding.
    Outputs GEN2_WRAPPED_MODEL for use with Gen2_QwenImageControlSampler.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", ),
                "controlnet": ("GEN2_CONTROLNET", ),
                "vae": ("GEN2_VAE", ),
                "control_image": ("IMAGE", ),
                "control_context_scale": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "inpaint_image": ("IMAGE", ),
                "mask": ("MASK", ),
            }
        }
    
    RETURN_TYPES = ("GEN2_WRAPPED_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Gen2/QwenImage"
    
    def prepare_control_context(self, vae, control_image, inpaint_image, mask, device, dtype, height, width):
        """
        Prepare the 132-feature control context matching VideoX's QwenImageControlPipeline exactly.
        """
        # Extract VAE model and config
        vae_model = vae['model']
        vae_config = vae['config']
        vae_dtype = vae['dtype']
        vae_offload = vae['device']
        
        # Move VAE to compute device
        vae_model = vae_model.to(device)
        
        vae_scale_factor = 8
        batch_size = control_image.shape[0]
        _, img_h, img_w, _ = control_image.shape
        
        latent_height = 2 * (int(height) // (vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (vae_scale_factor * 2))
        num_channels_latents = 16
        
        # Create latent normalization tensors from VAE config
        latents_mean = torch.tensor(vae_config['latents_mean']).view(1, num_channels_latents, 1, 1, 1).to(device)
        latents_std = 1.0 / torch.tensor(vae_config['latents_std']).view(1, num_channels_latents, 1, 1, 1).to(device)
        
        # VideoX-style processors
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)
        mask_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        
        # --- Process mask ---
        if mask is not None:
            mask_condition = mask_processor.preprocess(mask, height=height, width=width)
            mask_condition = torch.where(mask_condition >= 0.5, torch.ones_like(mask_condition), torch.zeros_like(mask_condition))
            mask_condition = torch.tile(mask_condition, [1, 3, 1, 1]).to(dtype=dtype, device=device)
        else:
            mask_condition = torch.ones(batch_size, 3, height, width, dtype=dtype, device=device)
        
        def _to_bchw(image_tensor):
            if isinstance(image_tensor, torch.Tensor) and image_tensor.ndim == 4:
                if image_tensor.shape[-1] in (1, 3, 4):
                    return image_tensor.permute(0, 3, 1, 2)
            return image_tensor
        
        # --- Process inpaint image ---
        if inpaint_image is not None:
            inpaint_image_bchw = _to_bchw(inpaint_image)
            init_image = image_processor.preprocess(inpaint_image_bchw, height=height, width=width)
            init_image = init_image.to(dtype=dtype, device=device) * (mask_condition < 0.5)
            init_image = init_image.unsqueeze(2)
            
            with torch.no_grad():
                inpaint_latent = vae_model.encode(init_image)[0].mode()
            inpaint_latent = ((inpaint_latent - latents_mean) * latents_std).to(dtype=dtype)
        else:
            inpaint_latent = torch.zeros(
                batch_size, num_channels_latents, 1, latent_height, latent_width,
                dtype=dtype, device=device
            )
        
        # --- Process control image ---
        if control_image is not None:
            control_image_bchw = _to_bchw(control_image)
            control_image = image_processor.preprocess(control_image_bchw, height=height, width=width)
            control_image = control_image.to(dtype=dtype, device=device)
            control_image = control_image.unsqueeze(2)
            
            with torch.no_grad():
                control_latents = vae_model.encode(control_image)[0].mode()
            control_latents = ((control_latents - latents_mean) * latents_std).to(dtype=dtype)
        else:
            control_latents = torch.zeros_like(inpaint_latent)
        
        # --- Prepare mask for latent space ---
        mask_latent = F.interpolate(
            1 - mask_condition[:, :1],
            size=inpaint_latent.size()[-2:],
            mode='nearest'
        ).to(dtype=dtype, device=device)
        mask_latent = mask_latent.unsqueeze(2)
        
        # --- Concatenate control context ---
        # VideoX order: [control_latents(16), mask(1), inpaint_latent(16)] = 33 channels
        control_context = torch.cat([control_latents, mask_latent, inpaint_latent], dim=1)
        
        # Get dimensions for packing
        ctrl_batch, ctrl_channels, ctrl_frames, ctrl_h, ctrl_w = control_context.shape
        
        # Pack to sequence format using pack_latents_v2
        control_context = pack_latents_v2(
            control_context, ctrl_batch, ctrl_channels, ctrl_h, ctrl_w, num_frame=ctrl_frames
        )
        
        # Move VAE back to offload device
        vae_model = vae_model.to(vae_offload)
        
        print(f"[Gen2] Control context (VideoX style): image={height}x{width}, "
              f"latent={latent_height}x{latent_width}, packed_seq={control_context.shape[1]}, "
              f"features={control_context.shape[2]}")
        
        return control_context, latent_height, latent_width
    
    def apply(self, model, controlnet, vae, control_image, control_context_scale,
              inpaint_image=None, mask=None):
        
        device = mm.get_torch_device()
        
        # Get the underlying diffusion model from ComfyUI's ModelPatcher
        comfyui_diffusion_model = model.model.diffusion_model
        
        # Get model storage dtype (may be quantized: fp8, int8, etc.)
        model_storage_dtype = next(comfyui_diffusion_model.parameters()).dtype
        
        # Get compute dtype using VideoX-style detection
        vae_storage_dtype = vae.get('dtype', model_storage_dtype)
        compute_dtype = get_compute_dtype(vae_storage_dtype, fallback_dtype=torch.bfloat16)
        
        dtype = compute_dtype
        
        print(f"[Gen2] Model storage dtype: {model_storage_dtype}")
        print(f"[Gen2] VAE storage dtype: {vae_storage_dtype}")
        print(f"[Gen2] Compute dtype: {compute_dtype}")
        
        # Get image dimensions from control_image
        _, img_h, img_w, _ = control_image.shape
        
        # Round to divisible by 16
        height = (img_h // 16) * 16
        width = (img_w // 16) * 16
        
        # Prepare control context (VideoX style)
        print(f"[Gen2] Preparing control context (VideoX style)...")
        control_context, lh, lw = self.prepare_control_context(
            vae, control_image, inpaint_image, mask, device, dtype, height, width
        )
        
        # Create VideoX-compatible wrapper
        wrapped_transformer = Gen2QwenImageModelWrapper(
            comfyui_model=comfyui_diffusion_model,
            control_model=controlnet['model'],
            latent_height=lh,
            latent_width=lw,
            control_layers=controlnet['control_layers'],
        )
        
        # Package wrapped model with all necessary components
        wrapped_model = {
            'wrapped_model': wrapped_transformer,
            'control_model': controlnet['model'],
            'control_context': control_context,
            'control_context_scale': control_context_scale,
            'control_layers': controlnet['control_layers'],
            'latent_height': lh,
            'latent_width': lw,
            'image_height': height,
            'image_width': width,
            'model_storage_dtype': model_storage_dtype,
            'compute_dtype': compute_dtype,
            'dtype': dtype,
            'vae': vae,
        }
        
        print(f"[Gen2] ControlNet prepared: scale={control_context_scale}, layers={controlnet['control_layers']}")
        print(f"[Gen2] Image size: {width}x{height}, latent size: {lw}x{lh}")
        
        return (wrapped_model,)


# =============================================================================
# Node 3: Load VAE
# =============================================================================

class Gen2_LoadQwenVAE:
    """
    Load QwenImage VAE with proper VideoX configuration.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"default": "qwen_image_vae.safetensors"}),
                "precision": (["bf16", "fp16"], {"default": "bf16"}),
            }
        }
    
    RETURN_TYPES = ("GEN2_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "Gen2/QwenImage"
    
    def load(self, model_name, precision):
        # Import VideoX's VAE class
        try:
            from videox_fun.models.qwenimage_vae import AutoencoderKLQwenImage
        except ImportError as e:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            custom_nodes_dir = os.path.dirname(os.path.dirname(current_dir))
            videox_path = os.path.join(custom_nodes_dir, "videox-fun")
            raise ImportError(
                f"Cannot import AutoencoderKLQwenImage from videox_fun.\n"
                f"  Looking for videox-fun at: {videox_path}\n"
                f"  Exists: {os.path.exists(videox_path)}\n"
                f"  sys.path includes videox-fun: {videox_path in sys.path}\n"
                f"  Original error: {e}\n"
                f"Make sure videox-fun is installed in custom_nodes folder."
            )
        
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        
        # Load state dict
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
        
        # Check for Wan compiled VAE format
        if "conv1.weight" in vae_state_dict:
            use_wan_compiled_vae = True
            if not any(k.startswith("model.") for k in vae_state_dict.keys()):
                vae_state_dict = {f"model.{k}": v for k, v in vae_state_dict.items()}
        else:
            use_wan_compiled_vae = False
        
        # Filter kwargs to match class signature
        kwargs = dict(QWEN_VAE_CONFIG)
        sig = inspect.signature(AutoencoderKLQwenImage)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        
        # Create VAE model
        if use_wan_compiled_vae:
            try:
                from videox_fun.models.wan_vae import AutoencoderKLWanCompileQwenImage
                vae = AutoencoderKLWanCompileQwenImage(**accepted)
            except ImportError:
                vae = AutoencoderKLQwenImage(**accepted)
        else:
            vae = AutoencoderKLQwenImage(**accepted)
        
        vae.load_state_dict(vae_state_dict)
        vae = vae.eval().to(device=offload_device, dtype=weight_dtype)
        
        print(f"[Gen2] Loaded QwenImage VAE: {model_name}")
        print(f"  z_dim={vae.z_dim}, spatial_compression={vae.spatial_compression_ratio}")
                        
        return ({
            'model': vae,
            'config': QWEN_VAE_CONFIG,
            'dtype': weight_dtype,
            'device': offload_device,
        },)


# =============================================================================
# Node 4: Text Encode
# =============================================================================

class Gen2_QwenClipTextEncode:
    """
    Encode text prompts for QwenImage using VideoX's EXACT encoding process.
    
    KEY DIFFERENCES from ComfyUI's CLIPTextEncode:
    1. Uses our own HuggingFace tokenizer (not ComfyUI's wrapped version)
    2. Applies VideoX's exact template with FIXED drop_idx=34
    3. Extracts valid tokens and drops template prefix exactly like VideoX
    4. Returns embeddings with actual token length (no fixed padding)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, 
                         "tooltip": "The text prompt to encode"}),
                "max_sequence_length": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64,
                                        "tooltip": "Maximum sequence length (truncate if longer, VideoX default: 1024)"}),
                "embeds_dtype": (["auto", "fp16", "bf16"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("GEN2_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Gen2/QwenImage"
    DESCRIPTION = "Encode text for QwenImage using VideoX's exact encoding process"
    
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """Extract valid (non-padded) hidden states. Exact copy of VideoX's method."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result
    
    def encode(self, clip, text, max_sequence_length, embeds_dtype):
        """
        Encode text using VideoX's EXACT process.
        """
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None")
        
        # Get our custom tokenizer
        tokenizer = get_gen2_tokenizer()
        device = mm.get_torch_device()
        
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
        
        # Apply VideoX template
        txt = [VIDEOX_PROMPT_TEMPLATE.format(t) for t in text]
        
        # Tokenize using OUR tokenizer with VideoX's exact parameters
        txt_tokens = tokenizer(
            txt, 
            max_length=VIDEOX_TOKENIZER_MAX_LENGTH + VIDEOX_DROP_IDX,
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids = txt_tokens.input_ids.to(device)
        attention_mask = txt_tokens.attention_mask.to(device)
        
        # Load CLIP model to GPU
        clip.load_model()
        
        # Access the underlying text encoder model
        cond_stage = clip.cond_stage_model
        
        hidden_states = None
        text_encoder_dtype = None
        
        with torch.no_grad():
            try:
                text_encoder_dtype = getattr(cond_stage, "dtype", None)
                encoder_out = cond_stage(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = encoder_out.hidden_states[-1]
            except Exception:
                if hasattr(cond_stage, 'qwen25_7b'):
                    clip_model = cond_stage.qwen25_7b
                elif hasattr(cond_stage, 'clip'):
                    clip_model = getattr(cond_stage, cond_stage.clip)
                else:
                    clip_model = cond_stage
                
                transformer = clip_model.transformer if hasattr(clip_model, 'transformer') else clip_model.model
                text_encoder_dtype = getattr(transformer, "dtype", None)
                
                embeddings = transformer.get_input_embeddings()(input_ids, out_dtype=transformer.dtype if hasattr(transformer, "dtype") else torch.float16)
                hidden_states = transformer(
                    None,
                    attention_mask=attention_mask.float(),
                    embeds=embeddings,
                    num_tokens=None,
                    intermediate_output=None,
                    final_layer_norm_intermediate=True,
                    dtype=embeddings.dtype,
                    embeds_info=[]
                )
                
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
        
        # Apply VideoX's exact post-processing
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[VIDEOX_DROP_IDX:] for e in split_hidden_states]
        
        # Create attention mask (all 1s for actual tokens after template removal)
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=device) for e in split_hidden_states]
        
        # Get max sequence length in batch
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        
        # Truncate to max_sequence_length if needed
        if max_seq_len > max_sequence_length:
            split_hidden_states = [e[:max_sequence_length] for e in split_hidden_states]
            attn_mask_list = [m[:max_sequence_length] for m in attn_mask_list]
            max_seq_len = max_sequence_length
        
        # Pad to max_seq_len in batch (VideoX style - pad with zeros)
        prompt_embeds = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) 
            for u in split_hidden_states
        ])
        encoder_attention_mask = torch.stack([
            torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) 
            for u in attn_mask_list
        ])
        
        # Get actual token count
        actual_seq_lens = encoder_attention_mask.sum(dim=1).tolist()
        
        # Convert to requested dtype
        if embeds_dtype == "bf16":
            target_dtype = torch.bfloat16
        elif embeds_dtype == "fp16":
            target_dtype = torch.float16
        else:
            base_dtype = text_encoder_dtype if text_encoder_dtype is not None else prompt_embeds.dtype
            target_dtype = get_compute_dtype(base_dtype, fallback_dtype=torch.bfloat16)
        prompt_embeds = prompt_embeds.to(dtype=target_dtype)
        
        # Create GEN2_CONDITIONING format
        conditioning = {
            "embeds": prompt_embeds,
            "attention_mask": encoder_attention_mask,
            "txt_seq_len": actual_seq_lens,
            "pooled_output": None,
        }
        
        # Diagnostic output
        print(f"[Gen2 TextEncode] VideoX-style: seq_len={prompt_embeds.shape[1]}, actual_tokens={actual_seq_lens}")
        print(f"[Gen2 TextEncode] embeds dtype: {prompt_embeds.dtype}")
        print(f"[Gen2 TextEncode] embeds: mean={prompt_embeds.mean().item():.6f}, std={prompt_embeds.std().item():.6f}")
        
        return (conditioning,)


# =============================================================================
# Node 5: Load LoRA
# =============================================================================

class Gen2_LoadQwenLora:
    """
    Load LoRA for QwenImage ControlNet (VideoX style).
    Stores LoRA path and strength for use by the sampler.
    Multiple LoRAs can be chained.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {"default": None}),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "lora": ("GEN2_LORA",),
            }
        }
    
    RETURN_TYPES = ("GEN2_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "load_lora"
    CATEGORY = "Gen2/QwenImage"
    
    def load_lora(self, lora_name, strength, lora=None):
        # Start with previous LoRAs if provided
        if lora is not None:
            lora_paths = list(lora.get('lora_paths', []))
            lora_strengths = list(lora.get('lora_strengths', []))
        else:
            lora_paths = []
            lora_strengths = []
        
        # Add this LoRA
        if lora_name is not None:
            full_path = folder_paths.get_full_path("loras", lora_name)
            if full_path:
                lora_paths.append(full_path)
                lora_strengths.append(strength)
                print(f"[Gen2 LoRA] Added LoRA: {lora_name} (strength: {strength})")
            else:
                print(f"[Gen2 LoRA] Warning: LoRA not found: {lora_name}")
        
        lora_info = {
            'lora_paths': lora_paths,
            'lora_strengths': lora_strengths,
        }
        
        return (lora_info,)


# =============================================================================
# Node 6: Sampler
# =============================================================================

class Gen2_QwenImageControlSampler:
    """
    QwenImage ControlNet Sampler using VideoX's EXACT denoising loop.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("GEN2_WRAPPED_MODEL",),
                "positive": ("GEN2_CONDITIONING",),
                "negative": ("GEN2_CONDITIONING",),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shift": ("INT", {"default": 3, "min": 1, "max": 100, "step": 1}),
                "sampler": (["Flow", "Flow_Unipc", "Flow_DPM++"], {"default": "Flow"}),
            },
            "optional": {
                "lora": ("GEN2_LORA",),
                "attention_backend": (["AUTO", "FLASH_ATTENTION", "SAGE_ATTENTION", "SDPA"], {"default": "AUTO"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"
    CATEGORY = "Gen2/QwenImage"
    
    def sample(self, model, positive, negative, width, height, seed, steps, cfg, shift, sampler, lora=None, attention_backend="AUTO"):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # Get model components
        transformer = model['wrapped_model']
        
        model_storage_dtype = model.get('model_storage_dtype', model.get('dtype'))
        compute_dtype = model.get('compute_dtype', model.get('dtype'))
        
        # Verify dtype
        base_transformer = transformer.model if hasattr(transformer, "model") else transformer
        actual_model_dtype = next(base_transformer.parameters()).dtype
        
        if actual_model_dtype in QUANTIZED_DTYPES:
            print(f"[Gen2] Quantized model detected: storage={actual_model_dtype}, compute={compute_dtype}")
        elif actual_model_dtype != compute_dtype:
            print(f"[Gen2 WARNING] dtype mismatch! model={actual_model_dtype}, expected={compute_dtype}")
        
        # Get control context
        control_context_raw = model['control_context']
        control_context_scale = model['control_context_scale']
        
        vae_scale_factor = 8
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        
        # Get VAE
        vae = model['vae']
        vae_model = vae['model']
        vae_config = vae['config']
        vae_storage_dtype = vae['dtype']
        vae_compute_dtype = get_compute_dtype(vae_storage_dtype, fallback_dtype=compute_dtype)
        vae_model = vae_model.to(device)
        
        # =================================================================
        # 0. Apply LoRA (if provided)
        # =================================================================
        lora_applied = False
        lora_paths_applied = []
        lora_strengths_applied = []
        
        if lora is not None and len(lora.get('lora_paths', [])) > 0:
            lora_paths_applied = lora['lora_paths']
            lora_strengths_applied = lora['lora_strengths']
            
            actual_transformer = transformer.model if hasattr(transformer, "model") else transformer
            lora_dtype = compute_dtype
            
            print(f"[Gen2] Merging {len(lora_paths_applied)} LoRA(s) (dtype={lora_dtype})...")
            for lora_path, lora_strength in zip(lora_paths_applied, lora_strengths_applied):
                gen2_merge_lora(actual_transformer, lora_path, lora_strength, device=device, dtype=lora_dtype)
            
            lora_applied = True
        
        # =================================================================
        # 1. Extract prompt embeddings
        # =================================================================
        prompt_embeds = positive["embeds"].to(device=device)
        prompt_embeds_mask = positive["attention_mask"].to(device=device)
        pos_txt_seq_len = positive["txt_seq_len"]
        
        negative_prompt_embeds = negative["embeds"].to(device=device)
        negative_prompt_embeds_mask = negative["attention_mask"].to(device=device)
        neg_txt_seq_len = negative["txt_seq_len"]
        
        compute_dtype = prompt_embeds.dtype
        prompt_embeds = prompt_embeds.to(dtype=compute_dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(dtype=compute_dtype)

        control_context = control_context_raw.to(device=device, dtype=compute_dtype)
        
        # Pad to same length for CFG
        max_seq_len = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
        
        def pad_to_length(embeds, mask, target_len):
            seq_len = embeds.shape[1]
            if seq_len < target_len:
                batch = embeds.shape[0]
                hidden = embeds.shape[2]
                pad_len = target_len - seq_len
                embeds = torch.cat([embeds, torch.zeros(batch, pad_len, hidden, device=embeds.device, dtype=embeds.dtype)], dim=1)
                mask = torch.cat([mask, torch.zeros(batch, pad_len, device=mask.device, dtype=mask.dtype)], dim=1)
            return embeds, mask
        
        prompt_embeds, prompt_embeds_mask = pad_to_length(prompt_embeds, prompt_embeds_mask, max_seq_len)
        negative_prompt_embeds, negative_prompt_embeds_mask = pad_to_length(negative_prompt_embeds, negative_prompt_embeds_mask, max_seq_len)
        
        batch_size = prompt_embeds.shape[0]
        
        guidance_scale_input = cfg
        true_cfg_scale = 4.0

        has_neg_prompt = negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        
        txt_seq_lens = pos_txt_seq_len
        negative_txt_seq_lens = neg_txt_seq_len if do_true_cfg else None
        
        # Determine attention backend
        import os
        if attention_backend is not None and attention_backend != "AUTO":
            os.environ["VIDEOX_ATTENTION_TYPE"] = attention_backend
        attn_type = os.environ.get("VIDEOX_ATTENTION_TYPE", "FLASH_ATTENTION")
        if attn_type == "SAGE_ATTENTION" and SAGE_ATTENTION_AVAILABLE:
            active_backend = "SageAttention"
        elif attn_type == "FLASH_ATTENTION" and (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
            active_backend = "FlashAttn3" if FLASH_ATTN_3_AVAILABLE else "FlashAttn2"
        else:
            active_backend = "SDPA"
        
        print(f"[Gen2] Sampling: {width}x{height}, steps={steps}, cfg={cfg}, sampler={sampler}, backend={active_backend}")
        
        # =================================================================
        # 2. Prepare latents
        # =================================================================
        num_channels_latents = 16
        
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(
            (batch_size, 1, num_channels_latents, latent_height, latent_width),
            generator=generator, device=device, dtype=compute_dtype
        )
        latents = pack_latents_v2(latents, batch_size, num_channels_latents, latent_height, latent_width)
        
        # =================================================================
        # 3. Prepare img_shapes for RoPE
        # =================================================================
        packed_h = latent_height // 2
        packed_w = latent_width // 2
        img_shapes = [
            [(1, packed_h, packed_w)]
        ] * batch_size
        
        # =================================================================
        # 4. Setup scheduler
        # =================================================================
        scheduler = get_qwen_scheduler(sampler, shift)
        
        sigmas = np.linspace(1.0, 1 / steps, steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        
        timesteps, num_inference_steps = retrieve_timesteps_v2(
            scheduler, steps, device, sigmas=sigmas, mu=mu
        )
        
        # =================================================================
        # 5. Denoising loop (Optimized)
        # =================================================================
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        
        scheduler.set_begin_index(0)
        
        # Pre-allocate CFG inputs outside the loop
        if do_true_cfg:
            prompt_embeds_mask_input = [
                m for m in negative_prompt_embeds_mask
            ] + [m for m in prompt_embeds_mask] if prompt_embeds_mask.dim() > 1 else [
                negative_prompt_embeds_mask, prompt_embeds_mask
            ]
            prompt_embeds_input = [
                e for e in negative_prompt_embeds
            ] + [e for e in prompt_embeds] if prompt_embeds.dim() > 2 else [
                negative_prompt_embeds, prompt_embeds
            ]
            img_shapes_input = img_shapes * 2
            txt_seq_lens_input = (negative_txt_seq_lens or txt_seq_lens) + txt_seq_lens
            control_context_doubled = torch.cat([control_context] * 2)
        else:
            prompt_embeds_mask_input = prompt_embeds_mask
            prompt_embeds_input = prompt_embeds
            img_shapes_input = img_shapes
            txt_seq_lens_input = txt_seq_lens
            control_context_doubled = control_context
        
        for i, t in enumerate(timesteps):
            if do_true_cfg:
                latent_model_input = torch.cat([latents] * 2)
                control_context_input = control_context_doubled
            else:
                latent_model_input = latents
                control_context_input = control_context_doubled
            
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)
            
            with torch.cuda.amp.autocast(dtype=compute_dtype), torch.cuda.device(device=device), torch.no_grad():
                noise_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=(
                        torch.full([1], guidance_scale_input, device=device, dtype=torch.float32).expand(latent_model_input.shape[0])
                        if getattr(transformer, "config", None) is not None and transformer.config.guidance_embeds
                        else None
                    ),
                    encoder_hidden_states_mask=prompt_embeds_mask_input,
                    encoder_hidden_states=prompt_embeds_input,
                    img_shapes=img_shapes_input,
                    txt_seq_lens=txt_seq_lens_input,
                    attention_kwargs=None,
                    control_context=control_context_input,
                    control_context_scale=control_context_scale,
                    return_dict=False,
                )
            
            # Apply CFG with norm rescaling
            if do_true_cfg:
                neg_noise_pred, pos_noise_pred = noise_pred.chunk(2)
                comb_pred = neg_noise_pred + true_cfg_scale * (pos_noise_pred - neg_noise_pred)
                
                cond_norm = torch.norm(pos_noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                
                noise_pred = comb_pred * (cond_norm / noise_norm)
            
            latents_dtype = latents.dtype
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)
            
            pbar.update(1)
        
        # =================================================================
        # 6. Decode latents to image
        # =================================================================
        actual_height = latent_height * vae_scale_factor
        actual_width = latent_width * vae_scale_factor
        
        num_patches_to_keep = packed_h * packed_w
        
        latents = unpack_latents_v2(
            latents[:, :num_patches_to_keep],
            actual_height, actual_width, vae_scale_factor, num_frame=1
        )
        latents = latents.to(vae_model.dtype)
        latents = latents[:, :, :1]
        
        # Denormalize
        latents_mean_dec = torch.tensor(vae_config['latents_mean']).view(1, vae_config['z_dim'], 1, 1, 1).to(latents.device, latents.dtype)
        latents_std_dec = 1.0 / torch.tensor(vae_config['latents_std']).view(1, vae_config['z_dim'], 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents / latents_std_dec + latents_mean_dec
        
        # Decode
        with torch.no_grad():
            decoded = vae_model.decode(latents, return_dict=False)[0]
        
        if decoded.ndim == 5:
            decoded = decoded[:, :, 0]
        
        decoded = (decoded + 1.0) / 2.0
        decoded = decoded.clamp(0, 1)
        
        image = decoded.permute(0, 2, 3, 1).cpu().float()
        
        # Move VAE back
        vae_model = vae_model.to(vae['device'])
        
        # =================================================================
        # 7. Unmerge LoRA (if applied)
        # =================================================================
        if lora_applied:
            print(f"[Gen2] Unmerging {len(lora_paths_applied)} LoRA(s) (dtype={lora_dtype})...")
            actual_transformer = transformer.model if hasattr(transformer, "model") else transformer
            for lora_path, lora_strength in zip(lora_paths_applied, lora_strengths_applied):
                gen2_unmerge_lora(actual_transformer, lora_path, lora_strength, device=device, dtype=lora_dtype)
        
        print(f"[Gen2 V2] Generated image: {tuple(image.shape)} ({actual_width}x{actual_height})")
        
        return (image,)

