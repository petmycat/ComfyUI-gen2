# Gen2 Workflow API Integration

This guide is for API server developers integrating exported ComfyUI workflows that use `Gen2_InputPanel` and `Gen2_OutputPanel`.

`api_nodes/workflow_contract.py` provides the JSON contract layer needed to discover parameters, validate requests, patch API prompts, and extract results from ComfyUI history.

Chinese version: [`workflow_api_cn.md`](workflow_api_cn.md)

## 1. Keep both ComfyUI exports

ComfyUI has two export formats with different responsibilities:

| File | Purpose |
|---|---|
| Normal workflow | Reopen, document, edit, and reproduce the graph in ComfyUI UI |
| API-format workflow | Submit an executable prompt to ComfyUI `/prompt` |

Store both files for every published workflow:

```text
workflows/
└── product-image/
    ├── workflow.json
    └── workflow_api.json
```

Do not submit the normal workflow to `/prompt`. Do not use the API-format file as a replacement for the editable normal workflow.

A published graph should contain:

1. One or more `Gen2_InputPanel` nodes.
2. One or more `Gen2_OutputPanel` nodes.
3. A `PANEL_LINK` from the relevant Input Panel to each Output Panel.
4. Input parameters connected to the workflow controls they represent.
5. Workflow results connected to the corresponding Output Panel inputs.

Export both files from the same tested graph state.

## 2. Public functions

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

The module is a pure JSON adapter. It does not execute nodes, upload images, call HTTP endpoints, or manage the ComfyUI queue.

Keep `workflow_contract.py` and `_config.py` from the same plugin version because they share validation rules.

## 3. Load and inspect exports

```python
import json
from pathlib import Path

from api_nodes.workflow_contract import detect_format, discover_manifest


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

`detect_format()` rejects malformed or mixed documents instead of silently guessing.

`discover_manifest()` validates panel configuration and returns a derived manifest without modifying the export.

## 4. Manifest structure

A manifest contains:

```json
{
  "version": 1,
  "source_format": "api_prompt",
  "contract_fingerprint": "sha256...",
  "input_panels": [
    {
      "node_id": "1",
      "class_type": "Gen2_InputPanel",
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
          "required": false,
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

Important fields:

- `id`: stable parameter identity.
- `name`: public API field name.
- `type`: request and response type.
- `default`: value configured in the Input Panel dialog.
- `current_value`: value stored in the exported workflow.
- `min`, `max`, `step`: numeric validation metadata.
- `binding`: API-prompt input location.
- `output_panels[].node_id`: result location in ComfyUI history.
- `contract_fingerprint`: stale-contract protection.

Normal-workflow bindings have `patchable: false`. They can be inspected but cannot be submitted as API prompts.

## 5. Check that both exports match

The two files should expose the same stable parameter IDs, names, types, order, metadata, outputs, and panel pairing.

```python
def contract_view(manifest: dict) -> dict:
    return {
        "inputs": [
            {
                "parameters": [
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "type": p["type"],
                        "default": p.get("default"),
                        "min": p.get("min"),
                        "max": p.get("max"),
                        "step": p.get("step"),
                        "controlMode": p.get("controlMode"),
                    }
                    for p in panel["parameters"]
                ]
            }
            for panel in manifest["input_panels"]
        ],
        "outputs": [
            {
                "parameters": [
                    {"id": p["id"], "name": p["name"], "type": p["type"]}
                    for p in panel["parameters"]
                ]
            }
            for panel in manifest["output_panels"]
        ],
    }


if contract_view(normal_manifest) != contract_view(api_manifest):
    raise ValueError("Normal and API exports expose different panel contracts")
```

Fail workflow registration if the files do not match. Re-export both files from the same graph rather than editing their JSON manually.

## 6. Input types

| Type | Accepted API value |
|---|---|
| `STRING` | JSON string; empty string is allowed |
| `COMBO` | Non-empty JSON string |
| `BOOLEAN` | JSON `true` or `false` only |
| `INT` | Integer within range and aligned to `step` |
| `FLOAT` | Finite number within range and aligned to `step` |
| `SEED` | Integer within configured safe range and aligned to `step` |
| `IMAGE` | Non-empty ComfyUI image reference string |

COMBO currently validates as a string because its schema does not yet contain an options list.

## 7. Validate request inputs

For one Input Panel, or multiple panels with globally unique parameter names, a flat request is allowed:

```json
{
  "prompt": "a product photo on a white background",
  "seed": 12345,
  "inputImage": "requests/request-123/source.png"
}
```

```python
validated = validate_call_inputs(api_manifest, request_body)
```

The returned mapping is panel-scoped:

```json
{
  "1": {
    "prompt": "a product photo on a white background",
    "seed": 12345,
    "inputImage": "requests/request-123/source.png"
  }
}
```

If multiple panels use the same parameter name, the request must be scoped explicitly:

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

It executes with all `current_value` fields stored in the API export. If only one value is supplied, omitted values keep their exported current values and are not reset to configuration defaults.

Translate validation `ValueError` exceptions to HTTP `400` or `422` responses.

## 8. Patch the API prompt

For the usual one-shot path:

```python
patched_prompt, manifest = prepare_api_prompt(
    api_prompt,
    {
        "prompt": "a red running shoe, studio lighting",
        "seed": 42,
        "inputImage": "requests/request-123/source.png",
    },
)
```

When a manifest was cached during registration:

```python
patched_prompt = patch_api_prompt(
    api_prompt,
    cached_manifest,
    request_body,
)
```

`patch_api_prompt()`:

1. requires an API-format export;
2. re-discovers the current contract;
3. checks the contract fingerprint;
4. validates request values;
5. deep-copies the prompt;
6. changes only Input Panel runtime fields;
7. preserves `_config`, links, node IDs, metadata, and all other nodes.

The source object is not modified. A stale fingerprint must trigger workflow reload or re-registration.

## 9. Upload IMAGE inputs

An IMAGE parameter is a ComfyUI file reference, not raw bytes or a public URL.

Upload the file first:

```text
POST /upload/image
```

Build the reference from the response:

```python
def image_reference(upload_response: dict) -> str:
    name = upload_response["name"]
    subfolder = upload_response.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name
```

The API server remains responsible for download policy, MIME checking, size limits, authentication, and upload storage isolation.

## 10. Submit to ComfyUI

```python
import json
from urllib.request import Request, urlopen


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


COMFY_URL = "http://127.0.0.1:8188"

submission = post_json(
    f"{COMFY_URL}/prompt",
    {
        "prompt": patched_prompt,
        "client_id": "your-api-service-instance-id",
    },
)

prompt_id = submission["prompt_id"]
```

Always associate the returned `prompt_id` with the API request. Never use global "latest history" when jobs can run concurrently.

## 11. Wait for completion

Polling example:

```python
import json
import time
from urllib.request import urlopen


def get_json(url: str) -> dict:
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def wait_for_history(comfy_url: str, prompt_id: str, timeout: float = 300) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = get_json(f"{comfy_url}/history/{prompt_id}")
        entry = history.get(prompt_id)
        if entry and entry.get("outputs") is not None:
            return history
        time.sleep(0.5)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out")
```

For higher throughput, use ComfyUI WebSocket events and fetch history after completion.

## 12. Extract configured outputs

```python
history = wait_for_history(COMFY_URL, prompt_id)

result = extract_history_results(
    history,
    manifest,
    prompt_id=prompt_id,
)
```

Result shape:

```json
{
  "prompt_id": "prompt-id",
  "panels": {
    "2": {
      "latest": {
        "version": 1,
        "inputs": {
          "schema": [],
          "latest_values": {}
        },
        "outputs": {
          "schema": [],
          "latest_values": {
            "resultImage": [
              {
                "filename": "result.png",
                "subfolder": "",
                "type": "output"
              }
            ]
          }
        }
      },
      "runs": [],
      "images": []
    }
  }
}
```

Read configured values through:

```python
latest = result["panels"]["2"]["latest"]
output_values = latest["outputs"]["latest_values"]
```

The extractor reads `document`, then `document_json`, then `schema_json`, then legacy `params`. Missing configured Output Panels raise `ValueError`.

## 13. Build image download URLs

```python
from urllib.parse import urlencode


def comfy_view_url(comfy_url: str, image: dict) -> str:
    query = urlencode(
        {
            "filename": image["filename"],
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        }
    )
    return f"{comfy_url}/view?{query}"
```

Do not store environment-specific ComfyUI URLs in the workflow or manifest.

## 14. Minimal workflow registry

```python
class RegisteredWorkflow:
    def __init__(self, workflow_id, api_prompt):
        self.workflow_id = workflow_id
        self.api_prompt = api_prompt
        self.manifest = discover_manifest(api_prompt)

    def build_prompt(self, values):
        return patch_api_prompt(self.api_prompt, self.manifest, values)

    def extract_result(self, history, prompt_id):
        return extract_history_results(
            history,
            self.manifest,
            prompt_id=prompt_id,
        )
```

Directory registration can load `workflow.json` and `workflow_api.json`, compare their contract views, and expose only matching, valid packages.

Suggested routes:

```text
GET  /api/v1/workflows/{workflow_id}
POST /api/v1/workflows/{workflow_id}/runs
GET  /api/v1/runs/{run_id}
```

Use the manifest to generate request schemas, OpenAPI metadata, WebUI controls, upload requirements, and output descriptions.

## 15. Error mapping

| Situation | Suggested response |
|---|---|
| Invalid request type/range | `422 workflow_input_invalid` |
| Unknown workflow | `404 workflow_not_found` |
| Export contract mismatch | Registration failure |
| Stale fingerprint | `409 workflow_contract_changed` |
| ComfyUI unavailable | `503 comfyui_unavailable` |
| Prompt rejected | `502 comfyui_prompt_rejected` |
| Execution timeout | `504 workflow_timeout` |
| Output Panel missing | `502 workflow_output_missing` |

Do not expose Python tracebacks, local paths, or unrestricted internal ComfyUI errors to public clients.

## 16. Security and operations checklist

- Register only matching normal/API export pairs.
- Validate both files at startup.
- Keep fingerprint checks enabled.
- Limit JSON size, node count, request size, and string length.
- Authenticate workflow runs and uploads.
- Isolate uploaded files by request or user.
- Reject arbitrary server filesystem paths.
- Enforce queue limits and execution timeouts.
- Keep each request's `prompt_id`.
- Read outputs only from configured Output Panel IDs.
- Apply input/output file retention rules.

## 17. Publishing updates

Re-export both files when changing:

- parameter order or stable ID;
- name or type;
- default, range, or step;
- SEED default mode;
- Output Panel fields;
- `PANEL_LINK` pairing.

Recommended sequence:

1. Test the graph in ComfyUI.
2. Set the desired current values.
3. save the normal workflow;
4. export API format from the same graph state;
5. replace both package files;
6. discover and compare both manifests;
7. reload the workflow registry;
8. run a smoke execution.

## 18. Out of scope

`workflow_contract.py` intentionally does not:

- convert normal workflows to API prompts;
- upload or download images;
- submit HTTP requests;
- manage the ComfyUI queue;
- wait for executions;
- authenticate users;
- generate framework-specific routes;
- delete temporary files;
- validate COMBO membership against an options list.

These belong to the surrounding API service. The module provides the stable bridge between Gen2 panel declarations, native ComfyUI exports, API request values, and history results.
