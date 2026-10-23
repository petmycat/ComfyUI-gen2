# ComfyUI-gen2

Custom nodes for API-driven workflows, LanPaint inpainting, and everyday image and text tasks.

## API panels

### Gen2 Input Panel

Bring a workflow's controls into one node. Define up to 32 named inputs: text, integers, decimals, booleans, images, dropdown choices, and seeds.

Set defaults, numeric ranges, and step sizes in **Configure**. Upload images from the panel and choose whether seeds stay fixed or change between runs. Connect each parameter to the part of your workflow it controls.

### Gen2 Output Panel

Collect named results from your workflow in one place. Preview and save images, inspect text and numeric results, and copy the combined inputs and outputs as JSON.

Connect the Input Panel's **PANEL_LINK** output to this node to keep each run's parameters alongside its results. Together, the panels define the inputs and outputs for workflows you call through an API.

## LanPaint inpainting

### Gen2 LanPaint Soft Denoise Patch

Use a soft mask to control how strongly LanPaint preserves the source image across an edit. Adjust blending strength over the sampling steps, and choose whether the reference follows the original image, the latest generated result, or a moving average.

Requires the separate **LanPaint** custom node pack. Connect this node's patched model to your LanPaint sampler and supply a soft mask for the preservation blend.

## Image and text utilities

### Gen2 Checkerboard

Generate a black-and-white checkerboard image with your own width, height, and block size. Useful for test images, backgrounds, and checking image alignment.

### Gen2 StringReplace

Replace every occurrence of a piece of text in a string. Useful for prompt templates and placeholder substitution. Matching is literal and case-sensitive.

## Experimental

### Gen2 Flux.2 Klein Fix

An optional compatibility patch that restores older Flux.2 Klein sampling behavior for reference-image and masked-inpainting workflows affected by ComfyUI's #12905 change.

Use it only when you need that earlier behavior. It is experimental and should not be combined with Flux KV-cache features.

## Installation

Find **ComfyUI-gen2** in ComfyUI Manager, or run this command from your `ComfyUI/custom_nodes` directory:

```bash
git clone https://github.com/petmycat/ComfyUI-gen2.git
```

Restart ComfyUI, then search for **Gen2** in the node menu. Requires ComfyUI **v0.28.0** or newer.

[View on Comfy Registry](https://registry.comfy.org/nodes/gen2)

## Upgrading from 2.x

Version 3 ships the six nodes listed above. Older Qwen Image, Flux.2 Fun ControlNet, Ideogram, AITK loader, pose, and tiling nodes are no longer included. Keep a 2.x installation for workflows that still depend on them.
