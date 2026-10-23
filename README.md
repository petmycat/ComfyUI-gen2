# ComfyUI-gen2

Custom nodes for configurable API workflows, image and string utilities, and LanPaint integration, with a separate experimental Flux.2 Klein compatibility fix.

## Directory layout

```text
ComfyUI-gen2/
├── gen2/
│   ├── api_panel_nodes/
│   │   ├── _config.py
│   │   ├── input_panel.py
│   │   ├── output_panel.py
│   │   ├── workflow_contract.py
│   │   └── web/js/gen2Panels.js
│   ├── misc_nodes/
│   │   ├── checkerboard.py
│   │   └── string_replace.py
│   └── lanpaint_nodes/
│       └── lanpaint_soft_denoise.py
├── legacy/                         # Local archive; ignored by Git
│   ├── qwenimage2512/
│   └── f2k_fun_controlnet/
├── experimental/
│   └── f2k/
│       └── flux2_klein_fix.py
├── tests/                          # Python, frontend, and opt-in ComfyUI tests
├── docs/                           # Local documentation; ignored by Git
└── workflows/
    └── default_comfyui_v0.42.1.json # Official placeholder workflow
```

Python package entrypoints are omitted from the diagram. Root packaging files and Git metadata remain alongside these six directories.

The root entrypoint registers the five maintained nodes and the experimental Klein fix independently. The legacy archive is not imported or registered. Its directories exist locally and are not included in a normal Git clone.

## Maintained nodes

### API panels

- **Gen2 Input Panel** (`Gen2_InputPanel`): defines named, typed workflow parameters, including strings, numbers, booleans, images, combo values, and seeds.
- **Gen2 Output Panel** (`Gen2_OutputPanel`): collects named results and publishes a structured document through ComfyUI execution history.

The **Configure** interface is served from `gen2/api_panel_nodes/web/js`. The public extension URL is unchanged by its filesystem move.

`gen2/api_panel_nodes/workflow_contract.py` provides helpers for discovering panel contracts, validating call inputs, patching API prompts, and extracting history results.

The public workflow contract remains **version 1**:

- Node IDs are `Gen2_InputPanel` and `Gen2_OutputPanel`.
- Input values are named keys beside `_config`; the first output is `PANEL_LINK`, followed by 32 positional parameter outputs.
- Output values bind to `param_0` through `param_31`. Existing named-key fallback remains supported.
- Result documents contain `version`, `inputs`, and `outputs`; both sections contain `schema` and `latest_values`.
- History retains `images`, `params`, `schema_json`, `document`, and `document_json`. Structured document fields remain list-valued as required by ComfyUI.
- Existing explicit parameter IDs and the contract fingerprint algorithm are unchanged. Older ID-less workflows recover FNV-keyed frontend values using the backend's canonical SHA1 IDs.

Runtime validation is strict. INT/SEED values must be integers, BOOLEAN values must be JSON booleans, and FLOAT values must be finite numbers. Input numbers must satisfy their configured range and step. Numeric strings in old *configuration* fields still normalize; numeric strings supplied as runtime values are rejected. IMAGE inputs are uploaded filename references, while IMAGE outputs are non-empty floating-point BHWC tensors with 1, 3, or 4 channels. Unconnected optional outputs remain `null`.

Names `_config`, `PANEL_LINK`, `param_<number>`, `__proto__`, `prototype`, and `constructor` are reserved. Malformed configuration produces an error without clearing the existing panel. Configuring a panel and applying changes intentionally recreates its parameter sockets; reconnect those sockets afterward. The `PANEL_LINK` connection is retained.

The frontend supports current named-widget workflow exports. Seed controls advance after each successful queue submission, including queued batches; exporting a workflow does not advance a seed. Uploaded images are fingerprinted by content so overwriting the same filename invalidates cached image data.

For multiple Input Panels, flat API calls are supported when parameter names are unique. Use panel-scoped calls when names repeat. Dynamically connected `_config` values cannot be discovered as a static API contract and are rejected by the contract helper.

### Miscellaneous utilities

- **Gen2 Checkerboard** (`Gen2_Checkerboard`): generates a black-and-white RGB checkerboard with configurable dimensions and block size.
- **Gen2 StringReplace** (`Gen2_StringReplace`): case-sensitive literal string replacement; an empty search string leaves the input unchanged.

### LanPaint

- **Gen2 LanPaint Soft Denoise Patch** (`Gen2_LanPaintSoftDenoisePatch`): adds soft-mask preservation, adaptive references, and scheduled blending around an installed LanPaint sampler.

The implementation and its existing experimental behavior are unchanged by the directory reorganization. Use it with the separate LanPaint node pack and its expected sampler interface.

## Experimental nodes

- **Gen2 Flux.2 Klein Fix (#12905 revert)** (`Gen2_Flux2KleinFix`): a historical compatibility workaround that temporarily substitutes older Flux forward methods during sampling.

The workaround changes process-wide Flux class methods while its wrapper runs. It should not be combined with Flux KV-cache or `index_timestep_zero` features. Its necessity and compatibility on newer ComfyUI versions require workflow validation.

## Installation

Install this repository under `ComfyUI/custom_nodes/ComfyUI-gen2` and restart ComfyUI. The declared minimum is **ComfyUI v0.28.0**. The API panels have been checked against **ComfyUI v0.42.1** and its pinned **frontend v1.55.18**. The CPU integration test runs the actual loader, validator, scheduler, and executor without model checkpoints.

The registered nodes use Python packages supplied by ComfyUI. Install the separate LanPaint custom node pack when using the LanPaint integration. Archived code may have additional dependencies, but it is not loaded by this pack.

## Tests

Run from this repository using a Python environment with PyTorch, NumPy, Pillow, and typing_extensions installed. Frontend tests use Node.js 18 or newer and need no npm packages.

```text
python -m unittest discover -s tests -v
node tests/test_panels_frontend.mjs
```

The suite covers API configuration, panel execution and image saving, version-1 workflow/history contracts, frontend serialization and lifecycle, checkerboard/string replacement, LanPaint mask/reference/wrapper behavior, and the experimental Klein wrapper's installation/restoration behavior. CPU unit tests use isolated doubles only at ComfyUI and LanPaint boundaries; real tensor operations and image files are exercised.

To additionally run against an installed ComfyUI checkout, set `COMFYUI_PATH`. On PowerShell:

```powershell
$env:COMFYUI_PATH = 'C:\path\to\ComfyUI'
python -m unittest discover -s tests -p test_comfyui_integration.py -v
```

Use that checkout's Python environment and dependencies. `COMFYUI_TEST_DEPS` optionally points to an isolated `pip --target` directory. The integration test runs in a separate CPU process, registers all six nodes, executes both panels and both misc nodes, rejects invalid API inputs, verifies saved images and history results, and tests cache invalidation after replacing an upload. Without `COMFYUI_PATH`, this one test is explicitly skipped. These checks do not establish GPU sampling quality or full external LanPaint compatibility.

The v0.42.1 verification passed 94 Python tests with integration enabled and 18 frontend regression tests. A live frontend v1.55.18 browser smoke test also configured and connected both panels, displayed a reserved-name validation error, queued a string roundtrip, saved the workflow, and reopened and executed it in a fresh page. The saved workflow's named widgets were successfully rediscovered by the contract helper.

## Placeholder workflow

`workflows/default_comfyui_v0.42.1.json` is the unmodified [official default workflow from ComfyUI v0.42.1](https://github.com/Comfy-Org/ComfyUI/blob/v0.42.1/default_workflow.json), copied from that tagged source archive. It is an editable UI workflow, not an API prompt or a Gen2 panel example.

It contains the release's Z-Image Turbo graph and references `z_image_turbo_bf16.safetensors`, `qwen_3_4b.safetensors`, and `ae.safetensors`. Supply the corresponding models before running it. Its SHA256 is `1b648133eb9ec1e5f44aadb8b7226cdccb3c350b83fd37409f9eee869c94a8cb`.

## Repository policy

- `gen2/` contains maintained features.
- `experimental/` contains experimental integrations.
- `legacy/` and `docs/` are ignored by Git.
- `tests/` and `workflows/` are included in version control.
- Existing retained node IDs and display names are preserved.

The local `docs/` directory retains only:

- `CONTRIBUTING_EN.md`
- `CONTRIBUTING_CN.md`
- `workflow_api_en.md`
- `workflow_api_cn.md`
- `LANPAINT_SOFT_DENOISE.md`

