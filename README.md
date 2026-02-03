# ComfyUI-Gen2 Custom Nodes

Custom ComfyUI nodes for QwenImage ControlNet and some other QoL nodes, designed to achieve **100% output compatibility with VideoX-Fun's diffusers pipeline** while leveraging ComfyUI's efficient model loading system.

## Why This Implementation?

We integrate with **ComfyUI's model loading nodes** (Load Diffusion Model, Load CLIP, Load VAE) but use our **own sampler and conditioning nodes**. This approach was chosen because:

1. **ComfyUI's model loading is highly optimized** - fast loading, memory efficient, supports quantized models (fp8, GGUF)
2. **VideoX's sampling pipeline has specific requirements** - custom RoPE calculation, True CFG with norm rescaling, and packed 3D latent format that differ from ComfyUI's standard sampler
3. **Exact output matching** - by replicating VideoX's exact forward logic while using ComfyUI's loaded weights, we achieve near identical outputs with the same seed

Our nodes act as a bridge: ComfyUI handles the heavy lifting of model management, while we ensure the inference process matches VideoX exactly.

## Credits

- **[VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun/tree/main/comfyui/qwenimage)** - The original QwenImage ControlNet implementation. Our pipeline logic is derived from their excellent work.
- **[ComfyUI](https://github.com/Comfy-Org/ComfyUI)** - The powerful and modular diffusion model GUI that makes this integration possible.

## Installation

1. **Prerequisites** - Install these custom node packs first:
   - [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) - Required for model components and utilities
   - [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) - Required if using GGUF quantized models

2. **Install ComfyUI-Gen2**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/petmycat/ComfyUI-gen2.git
   ```

3. **Tokenizer** - Download from [Qwen-Image-2512 on HuggingFace](https://huggingface.co/alibaba-pai/Qwen-Image-2512):
   - Navigate to the model's files and download all files from the `tokenizer/` folder
   - Place them in:
   ```
   ComfyUI/models/gen2/qwen_2512_tokenizer/
   ```

## Example Workflow

Example workflow and reference images are located in:
- `workflows/qwen_control_example_workflow.json` - Example ComfyUI workflow
- `assets/` - Reference images for testing (example (1).png, example (2).png)

## Nodes

### QwenImage ControlNet

| Node | Description |
|------|-------------|
| **Gen2 Load QwenImage ControlNet** | Load ControlNet weights |
| **Gen2 Load QwenImage VAE** | Load VAE with VideoX-compatible config |
| **Gen2 Apply QwenImage ControlNet** | Prepare control context and wrap model |
| **Gen2 QwenImage Text Encode** | VideoX-style text encoding (use instead of CLIPTextEncode) |
| **Gen2 Load QwenImage LoRA** | Load LoRA for VideoX-style merging |
| **Gen2 QwenImage Control Sampler** | VideoX-compatible sampling with True CFG |

### Utilities

| Node | Description |
|------|-------------|
| **Gen2 DWpose with Threshold** | DWpose detector with configurable confidence thresholds for body/hand/face keypoints |

## Dtype Support

Supports multiple precision modes:
- **bf16/fp16** - Full precision models
- **fp8** - Quantized models (automatic compute dtype detection)
- **GGUF** - Quantized models via ComfyUI-GGUF

## License

This project is licensed under the Apache License 2.0. It also follows the licensing requirements of its dependencies (VideoX-Fun, ComfyUI).

