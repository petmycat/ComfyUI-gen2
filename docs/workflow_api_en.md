# Gen2 Workflow API Integration Guide

This guide is for backend developers who expose ComfyUI workflows through an API. It explains:

- how the traditional manually maintained node-ID strategy works;
- how `Gen2_InputPanel` and `Gen2_OutputPanel` differ from that strategy;
- where `api_nodes/workflow_contract.py` can cut into an existing API service to automate it gradually;
- how to discover parameters, validate requests, patch API prompts, and extract execution results.

Chinese version: [`workflow_api_cn.md`](workflow_api_cn.md)

## 1. First understand the traditional strategy

Before Gen2 API Panels, a server developer normally opened the ComfyUI API-format export and manually found every node whose inputs should be replaced.

For example:

```json
{
  "15": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "default prompt"
    }
  },
  "27": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 12345,
      "steps": 20
    }
  },
  "31": {
    "class_type": "LoadImage",
    "inputs": {
      "image": "default.png"
    }
  }
}
```

The server then maintained a workflow-specific binding table:

```python
WORKFLOW_BINDINGS = {
    "prompt": {"node_id": "15", "input_key": "text"},
    "seed": {"node_id": "27", "input_key": "seed"},
    "steps": {"node_id": "27", "input_key": "steps"},
    "inputImage": {"node_id": "31", "input_key": "image"},
}
```

For every run, it copied the template and replaced those locations:

```python
prompt["15"]["inputs"]["text"] = request_body["prompt"]
prompt["27"]["inputs"]["seed"] = request_body["seed"]
prompt["27"]["inputs"]["steps"] = request_body["steps"]
prompt["31"]["inputs"]["image"] = request_body["inputImage"]
```

Outputs required another manual binding, for example reading images from node `44`:

```python
images = history[prompt_id]["outputs"]["44"]["images"]
```

This is a valid and still usable strategy. It can be sufficiently direct for a small number of workflows that rarely change.

## 2. The traditional strategy is not broken; it lacks a public contract

The complete interface definition is distributed across several places:

```text
workflow_api.json
+ manually maintained input node IDs
+ manually maintained input_key values
+ manually written type and range checks
+ manually maintained output node IDs
+ a separate frontend form definition
```

A ComfyUI API export can say:

> The current `seed` input of node 27 is 12345.

It does not automatically say:

> `seed` is a public parameter that external users may change. It is an integer with a configured default, minimum, maximum, and step.

ComfyUI node inputs are implementation details of a graph. Product API fields are an external interface. The traditional strategy binds these two layers directly.

Node IDs are also not stable business identities. Deleting and recreating nodes, copying graph sections, merging workflows, or replacing implementation nodes may change node IDs and input keys.

An old binding can then:

1. point to a missing node and fail immediately;
2. point to another node and modify the wrong field;
3. still modify a valid field whose meaning has changed, producing a successful run with the wrong behavior.

The third case is particularly difficult to detect.

## 3. Gen2 Panels declare the workflow's public boundary

Gen2 introduces explicit input and output boundaries inside the graph:

```text
External API input
    ↓
Gen2_InputPanel
    ↓
Internal workflow nodes and links
    ↓
Gen2_OutputPanel
    ↓
External API output
```

The workflow author declares public parameters in `Gen2_InputPanel`, for example:

```text
prompt       STRING
seed         SEED
steps        INT
inputImage   IMAGE
```

The author then connects those outputs to the real internal controls.

The server modifies only the public Input Panel fields. It does not need to know whether the internal implementation is:

```text
InputPanel.seed → KSampler.seed
```

or:

```text
InputPanel.seed → SeedProcessor → NoiseGenerator → Sampler
```

The workflow author similarly declares outputs in `Gen2_OutputPanel`:

```text
resultImage  IMAGE
maskImage    IMAGE
score        FLOAT
caption      STRING
```

and connects internal results to those inputs.

The central difference is:

```text
Traditional: the server modifies internal workflow nodes directly.
Gen2: the server modifies declared public inputs and reads declared public outputs.
```

## 4. Gen2 still patches native API JSON, but discovers addresses automatically

Gen2 does not replace ComfyUI's native API or introduce a new execution format.

The API export remains a normal ComfyUI prompt:

```json
{
  "1": {
    "class_type": "Gen2_InputPanel",
    "inputs": {
      "_config": "[...]",
      "prompt": "default prompt",
      "seed": 12345,
      "steps": 20,
      "inputImage": "default.png"
    }
  }
}
```

`workflow_contract.py` finds `Gen2_InputPanel` and `Gen2_OutputPanel`, parses `_config`, and derives:

- public parameter names and stable IDs;
- types, configured defaults, and exported current values;
- `min`, `max`, and `step` metadata;
- the default SEED control mode;
- patchable API-prompt locations;
- Output Panel locations in ComfyUI history;
- Input/Output pairing through `PANEL_LINK`.

The manifest still contains node IDs because JSON patching and history lookup eventually require concrete addresses.

The difference is that a node ID is now an **automatically discovered execution address**, not a manually maintained business configuration.

## 5. Specific automation insertion points

`workflow_contract.py` does not require replacing the entire API service. It can replace the three most manual parts of the execution chain:

```text
Manual input bindings → discover_manifest() + patch_api_prompt()
Manual input validation → validate_call_inputs()
Manual output bindings → extract_history_results()
```

Existing authentication, rate limits, uploads, queues, WebSocket handling, storage, and CDN logic can remain unchanged.

### 5.1 Replace manual node-ID configuration

Traditional code:

```python
WORKFLOW_INPUTS = {
    "prompt": ("15", "text"),
    "seed": ("27", "seed"),
    "image": ("31", "image"),
}
```

Gen2 registration code:

```python
manifest = discover_manifest(api_prompt)
```

The manifest can be generated and cached at service startup or workflow registration time.

### 5.2 Replace workflow-specific validation code

Traditional services often contain many workflow-specific checks:

```python
if not isinstance(seed, int):
    raise ValueError("seed must be an integer")
if seed < 0 or seed > max_seed:
    raise ValueError("seed is outside the allowed range")
```

Gen2 uses one common call:

```python
validated = validate_call_inputs(manifest, request_body)
```

It checks unknown fields, value types, numeric ranges, step alignment, the safe SEED range, and IMAGE references.

### 5.3 Replace manual JSON mutation

Traditional code:

```python
prompt["15"]["inputs"]["text"] = body["prompt"]
prompt["27"]["inputs"]["seed"] = body["seed"]
```

Gen2 code:

```python
patched_prompt = patch_api_prompt(
    api_prompt,
    manifest,
    request_body,
)
```

The helper deep-copies the template, changes only public Input Panel runtime fields, and preserves `_config`, node IDs, links, metadata, and internal nodes.

### 5.4 Generate frontend and OpenAPI schemas

A manifest supplies semantics in addition to patch locations:

```json
{
  "id": "stable-steps-id",
  "name": "steps",
  "type": "INT",
  "default": 20,
  "current_value": 28,
  "min": 1,
  "max": 100,
  "step": 1
}
```

A service can derive:

- Pydantic or other request models;
- JSON Schema;
- OpenAPI field descriptions;
- WebUI controls;
- IMAGE upload requirements;
- default request examples.

### 5.5 Replace manual output-node configuration

Traditional code:

```python
WORKFLOW_OUTPUTS = {
    "resultImage": {"node_id": "44", "field": "images"}
}
```

Gen2 code:

```python
result = extract_history_results(
    history,
    manifest,
    prompt_id=prompt_id,
)
```

The server no longer needs to know which SaveImage, custom node, or intermediate node produced a public result. It reads the fields declared by the Output Panel.

### 5.6 Automatically register workflow endpoints

Recommended layout:

```text
workflows/
├── product-image/
│   ├── workflow.json
│   └── workflow_api.json
├── background-removal/
│   ├── workflow.json
│   └── workflow_api.json
└── portrait-enhancement/
    ├── workflow.json
    └── workflow_api.json
```

At startup, the service can scan directories, discover manifests, verify that both exports agree, and register common routes:

```text
GET  /api/v1/workflows/{workflow_id}
POST /api/v1/workflows/{workflow_id}/runs
GET  /api/v1/runs/{run_id}
```

Ideally, adding a workflow only requires exporting it and placing it in a directory, without writing another node-ID map or patch function.

## 6. Gen2 and legacy workflows can coexist

There is no need to migrate every workflow at once.

```python
registry = {
    "legacy-product": {
        "mode": "legacy",
        "bindings": {
            "prompt": ("15", "text"),
            "seed": ("27", "seed"),
        },
    },
    "gen2-product": {
        "mode": "gen2_contract",
        "api_prompt": api_prompt,
        "manifest": discover_manifest(api_prompt),
    },
}
```

Dispatch by mode:

```python
if workflow["mode"] == "legacy":
    patched = patch_legacy_prompt(
        workflow["api_prompt"],
        workflow["bindings"],
        request_body,
    )
else:
    patched = patch_api_prompt(
        workflow["api_prompt"],
        workflow["manifest"],
        request_body,
    )
```

Recommended migration order:

1. use Gen2 Panels for new workflows by default;
2. migrate frequently edited workflows whose node IDs change often;
3. leave long-term stable legacy workflows on manual bindings when appropriate;
4. keep the same ComfyUI submission, queue, and result-storage infrastructure.

## 7. Keep both ComfyUI exports

Every published workflow should contain:

```text
workflows/product-image/
├── workflow.json
└── workflow_api.json
```

- `workflow.json`: UI reconstruction, editing, sharing, and exported current widget values.
- `workflow_api.json`: executable prompt submitted to ComfyUI `/prompt`.

Do not submit the normal workflow to `/prompt`, and do not replace the editable normal workflow with the API-format export.

Export both files from the same tested graph state.

## 8. Import the public functions

```python
from api_nodes.workflow_contract import (
    detect_format,
    discover_manifest,
    extract_history_results,
    patch_api_prompt,
    prepare_api_prompt,
    validate_call_inputs,
)
```

The module only performs JSON and contract operations. Keep `workflow_contract.py` and `_config.py` from the same plugin version.

## 9. Load exports and discover manifests

```python
import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


normal_workflow = load_json("workflows/product-image/workflow.json")
api_prompt = load_json("workflows/product-image/workflow_api.json")

assert detect_format(normal_workflow) == "workflow"
assert detect_format(api_prompt) == "api_prompt"

normal_manifest = discover_manifest(normal_workflow)
api_manifest = discover_manifest(api_prompt)
```

`detect_format()` rejects malformed, mixed, or unsupported structures.

`discover_manifest()` does not modify the source document.

## 10. Important manifest fields

Simplified example:

```json
{
  "version": 1,
  "source_format": "api_prompt",
  "contract_fingerprint": "sha256...",
  "input_panels": [
    {
      "node_id": "1",
      "parameters": [
        {
          "id": "stable-parameter-id",
          "name": "seed",
          "type": "SEED",
          "default": 0,
          "current_value": 12345,
          "min": 0,
          "max": 9007199254740991,
          "step": 1,
          "controlMode": "fixed",
          "binding": {
            "node_id": "1",
            "input_key": "seed",
            "patchable": true
          }
        }
      ]
    }
  ],
  "output_panels": [
    {
      "node_id": "2",
      "paired_input_node_id": "1",
      "parameters": [
        {
          "id": "stable-output-id",
          "name": "resultImage",
          "type": "IMAGE",
          "slot": "param_0"
        }
      ]
    }
  ]
}
```

`default` is the configured reset value. `current_value` is the runtime value stored in the export. Fields omitted by a call keep `current_value`; they are not automatically reset to `default`.

Normal-workflow bindings have `patchable: false`. They can be inspected but cannot be patched as executable API prompts.

## 11. Input types and validation

- `STRING`: JSON string; an empty string is allowed.
- `COMBO`: non-empty JSON string; the contract does not yet contain an option list.
- `BOOLEAN`: JSON `true` or `false` only.
- `INT`: integer within range and aligned to `step`.
- `FLOAT`: finite number within range and aligned to `step`.
- `SEED`: integer within the configured safe range and aligned to `step`.
- `IMAGE`: non-empty ComfyUI image reference string.

A flat request is allowed for one panel or globally unique parameter names:

```json
{
  "prompt": "a product photo",
  "seed": 12345,
  "inputImage": "requests/request-123/source.png"
}
```

If multiple panels contain duplicate names, scope the request by panel:

```json
{
  "1": {"seed": 12345},
  "7": {"seed": 67890}
}
```

An empty request is valid:

```json
{}
```

It executes using all current values stored in the API export.

## 12. Prepare and submit an API prompt

```python
patched_prompt, manifest = prepare_api_prompt(
    api_prompt,
    {
        "prompt": "a red running shoe, studio lighting",
        "seed": 42,
    },
)
```

Submit it to ComfyUI:

```python
submission = post_json(
    f"{COMFY_URL}/prompt",
    {
        "prompt": patched_prompt,
        "client_id": "your-api-service-instance-id",
    },
)

prompt_id = submission["prompt_id"]
```

Always associate `prompt_id` with the API request. Do not use a global "latest history" lookup when runs can execute concurrently.

## 13. IMAGE inputs

An IMAGE parameter is a ComfyUI file reference, not raw bytes or a public URL.

Upload first:

```text
POST /upload/image
```

Then build the reference:

```python
def image_reference(upload_response: dict) -> str:
    name = upload_response["name"]
    subfolder = upload_response.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name
```

Download policy, MIME checks, size limits, authentication, and storage isolation remain API-service responsibilities.

## 14. Extract Output Panel results

After execution completes:

```python
result = extract_history_results(
    history,
    manifest,
    prompt_id=prompt_id,
)
```

Read configured values through:

```python
latest = result["panels"]["2"]["latest"]
output_values = latest["outputs"]["latest_values"]
result_images = output_values["resultImage"]
```

The extractor checks these compatibility fields in order:

1. `document`
2. `document_json`
3. `schema_json`
4. legacy `params`

It raises `ValueError` if a configured Output Panel is absent from history.

## 15. Workflow registry example

```python
class RegisteredWorkflow:
    def __init__(self, workflow_id, api_prompt):
        self.workflow_id = workflow_id
        self.api_prompt = api_prompt
        self.manifest = discover_manifest(api_prompt)

    def build_prompt(self, values):
        return patch_api_prompt(
            self.api_prompt,
            self.manifest,
            values,
        )

    def extract_result(self, history, prompt_id):
        return extract_history_results(
            history,
            self.manifest,
            prompt_id=prompt_id,
        )
```

At registration time, discover manifests from both exports and compare public parameter IDs, names, types, order, metadata, outputs, and `PANEL_LINK` pairing. Do not expose an endpoint when they disagree.

## 16. Suggested error mapping

- Invalid request type or range: `422 workflow_input_invalid`
- Unknown workflow: `404 workflow_not_found`
- Export contract mismatch: registration failure
- Stale contract fingerprint: `409 workflow_contract_changed`
- ComfyUI unavailable: `503 comfyui_unavailable`
- Prompt rejected: `502 comfyui_prompt_rejected`
- Execution timeout: `504 workflow_timeout`
- Output Panel missing: `502 workflow_output_missing`

Do not expose Python tracebacks, local file paths, or unrestricted internal ComfyUI errors to public clients.

## 17. Security and operations checklist

- Register only matching normal/API export pairs.
- Validate files with `discover_manifest()` at startup.
- Keep contract-fingerprint checks enabled.
- Limit JSON size, node count, request size, and string length.
- Authenticate workflow runs and image uploads.
- Isolate uploads by request or user.
- Reject arbitrary server filesystem paths as IMAGE values.
- Enforce queue limits and execution timeouts.
- Keep each request's own `prompt_id`.
- Read outputs only through configured Output Panel IDs.
- Apply input/output retention and cleanup policies.

## 18. Publishing workflow updates

Re-export both files after changing:

- parameter order, stable ID, name, or type;
- default, range, or step;
- the default SEED mode;
- Output Panel fields;
- `PANEL_LINK` pairing.

Recommended process:

1. test the graph in ComfyUI;
2. set the current values that should be stored in the export;
3. save the normal workflow;
4. export API format from the same graph state;
5. replace both files in the package;
6. discover and compare both manifests;
7. reload the workflow registry;
8. run one smoke execution.

## 19. Module boundaries

`workflow_contract.py` does not:

- convert a normal workflow into an API prompt;
- upload or download images;
- send HTTP requests;
- manage the ComfyUI queue;
- wait for execution;
- authenticate users;
- generate framework-specific routes;
- delete temporary files;
- validate COMBO membership against an option list.

The recommended complete call chain is:

```text
Frontend request
  ↓
Existing authentication and rate limits
  ↓
validate_call_inputs()
  ↓
Existing image upload logic
  ↓
patch_api_prompt()
  ↓
Existing /prompt submission logic
  ↓
Existing WebSocket or history waiting logic
  ↓
extract_history_results()
  ↓
Existing response, storage, and CDN logic
```

The Gen2 contract layer is not intended to replace the whole API service. It lets workflow authors declare a business boundary inside ComfyUI and lets the server use one common implementation to discover, validate, patch, and extract values, removing the most repetitive and fragile manually maintained node-ID configuration.
