"""Pure-JSON helpers for discovering and invoking Gen2 panel workflows.

ComfyUI's normal workflow and API-prompt exports remain untouched. This module
creates a derived manifest, validates API calls, patches a copied API prompt,
and extracts Gen2 Output Panel documents from history responses.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal

from ._config import parse_input_config, parse_output_config, validate_runtime_value

DocumentFormat = Literal["workflow", "api_prompt"]


def detect_format(document: Mapping[str, Any]) -> DocumentFormat:
    if not isinstance(document, Mapping):
        raise ValueError("ComfyUI document must be a JSON object.")
    has_nodes = "nodes" in document
    prompt_like = bool(document) and all(
        isinstance(node_id, (str, int))
        and isinstance(node, Mapping)
        and isinstance(node.get("class_type"), str)
        and bool(node.get("class_type"))
        and isinstance(node.get("inputs"), Mapping)
        for node_id, node in document.items()
    )
    if has_nodes and prompt_like:
        raise ValueError("Document ambiguously mixes workflow and API-prompt structures.")
    if has_nodes:
        if not isinstance(document.get("nodes"), list):
            raise ValueError("Normal workflow field 'nodes' must be a list.")
        return "workflow"
    if prompt_like:
        return "api_prompt"
    raise ValueError("Document is neither a normal ComfyUI workflow nor an API prompt.")


def _normalised_node_view(document: Mapping[Any, Any]) -> dict[str, tuple[Any, Mapping[str, Any]]]:
    view: dict[str, tuple[Any, Mapping[str, Any]]] = {}
    for raw_id, node in document.items():
        node_id = str(raw_id)
        if node_id in view:
            raise ValueError(f"Node id {node_id!r} is duplicated after JSON normalisation.")
        if not isinstance(node, Mapping):
            raise ValueError(f"Node {node_id!r} must be an object.")
        view[node_id] = (raw_id, node)
    return view


def _config_from_workflow_node(node: Mapping[str, Any]) -> Any:
    widget_values = node.get("widgets_values")
    if not isinstance(widget_values, list) or not widget_values:
        raise ValueError(f"Panel node {node.get('id')!r} has no serialized _config widget.")
    return widget_values[0]


def _required_api_config(node_id: str, inputs: Mapping[str, Any]) -> Any:
    if "_config" not in inputs:
        raise ValueError(f"Panel node {node_id} is missing its exported _config input.")
    return inputs["_config"]


def _input_parameter_manifest(
    node_id: str,
    params: list[dict[str, Any]],
    source_format: DocumentFormat,
    node: Mapping[str, Any],
) -> list[dict[str, Any]]:
    properties = node.get("properties", {})
    runtime = properties.get("gen2RuntimeValues", {}) if isinstance(properties, Mapping) else {}
    inputs = node.get("inputs", {}) if source_format == "api_prompt" else {}
    result: list[dict[str, Any]] = []
    for index, param in enumerate(params):
        item = copy.deepcopy(param)
        item["required"] = False
        if source_format == "api_prompt":
            if param["name"] not in inputs:
                raise ValueError(f"Input Panel {node_id} is missing exported input {param['name']!r}.")
            current = validate_runtime_value(param, inputs[param["name"]])
            item["current_value"] = copy.deepcopy(current)
            item["binding"] = {"node_id": node_id, "input_key": param["name"], "patchable": True}
        else:
            current = runtime.get(param["id"], param.get("default")) if isinstance(runtime, Mapping) else param.get("default")
            item["current_value"] = copy.deepcopy(validate_runtime_value(param, current))
            item["binding"] = {
                "node_id": node_id,
                "output_slot": index + 1,
                "output_name": f"param_{index}",
                "patchable": False,
            }
        result.append(item)
    return result


def _history_binding(node_id: str) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "preferred_field": "document",
        "fallback_fields": ["document_json", "schema_json", "params"],
    }


def _manifest_contract(input_panels: list[dict[str, Any]], output_panels: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_panels": [
            {
                "node_id": panel["node_id"],
                "parameters": [
                    {key: copy.deepcopy(value) for key, value in param.items() if key not in ("current_value", "binding", "required")}
                    for param in panel["parameters"]
                ],
            }
            for panel in input_panels
        ],
        "output_panels": [
            {
                "node_id": panel["node_id"],
                "paired_input_node_id": panel.get("paired_input_node_id"),
                "parameters": [
                    {key: copy.deepcopy(value) for key, value in param.items() if key not in ("slot", "input_slot")}
                    for param in panel["parameters"]
                ],
            }
            for panel in output_panels
        ],
    }


def _contract_fingerprint(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def discover_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    source_format = detect_format(document)
    input_panels: list[dict[str, Any]] = []
    output_panels: list[dict[str, Any]] = []

    if source_format == "workflow":
        raw_nodes = document["nodes"]
        node_by_id: dict[str, Mapping[str, Any]] = {}
        for index, node in enumerate(raw_nodes):
            if not isinstance(node, Mapping):
                raise ValueError(f"Workflow node {index} must be an object.")
            if "id" not in node:
                raise ValueError(f"Workflow node {index} is missing an id.")
            node_id = str(node["id"])
            if node_id in node_by_id:
                raise ValueError(f"Workflow node id {node_id!r} is duplicated.")
            node_by_id[node_id] = node

        links = document.get("links", [])
        if not isinstance(links, list):
            raise ValueError("Normal workflow field 'links' must be a list when present.")
        panel_links: dict[str, str] = {}
        for index, link in enumerate(links):
            if not isinstance(link, list) or len(link) < 6:
                raise ValueError(f"Workflow link {index} must contain at least six fields.")
            origin_id, origin_slot, target_id, target_slot, link_type = str(link[1]), link[2], str(link[3]), link[4], link[5]
            if origin_slot == 0 and target_slot == 0 and link_type == "*":
                if target_id in panel_links:
                    raise ValueError(f"Output node {target_id} has multiple PANEL_LINK connections.")
                panel_links[target_id] = origin_id

        for node_id, node in node_by_id.items():
            node_type = node.get("type")
            if node_type == "Gen2_InputPanel":
                params = parse_input_config(_config_from_workflow_node(node))
                input_panels.append({
                    "node_id": node_id,
                    "class_type": node_type,
                    "parameters": _input_parameter_manifest(node_id, params, source_format, node),
                })
            elif node_type == "Gen2_OutputPanel":
                params = parse_output_config(_config_from_workflow_node(node))
                paired = panel_links.get(node_id)
                if paired is not None and node_by_id.get(paired, {}).get("type") != "Gen2_InputPanel":
                    raise ValueError(f"Output Panel {node_id} PANEL_LINK does not originate from an Input Panel.")
                output_panels.append({
                    "node_id": node_id,
                    "class_type": node_type,
                    "paired_input_node_id": paired,
                    "parameters": [
                        {**copy.deepcopy(param), "slot": f"param_{index}", "input_slot": index + 1}
                        for index, param in enumerate(params)
                    ],
                    "history_binding": _history_binding(node_id),
                })
    else:
        node_view = _normalised_node_view(document)
        for node_id, (_, node) in node_view.items():
            class_type = node["class_type"]
            inputs = node["inputs"]
            if class_type == "Gen2_InputPanel":
                params = parse_input_config(_required_api_config(node_id, inputs))
                input_panels.append({
                    "node_id": node_id,
                    "class_type": class_type,
                    "parameters": _input_parameter_manifest(node_id, params, source_format, node),
                })
            elif class_type == "Gen2_OutputPanel":
                params = parse_output_config(_required_api_config(node_id, inputs))
                paired = None
                if "PANEL_LINK" in inputs and inputs["PANEL_LINK"] is not None:
                    link = inputs["PANEL_LINK"]
                    if not isinstance(link, (list, tuple)) or len(link) != 2 or link[1] != 0:
                        raise ValueError(f"Output Panel {node_id} has an invalid PANEL_LINK connection.")
                    paired = str(link[0])
                    source = node_view.get(paired)
                    if source is None or source[1].get("class_type") != "Gen2_InputPanel":
                        raise ValueError(f"Output Panel {node_id} PANEL_LINK does not originate from an Input Panel.")
                output_panels.append({
                    "node_id": node_id,
                    "class_type": class_type,
                    "paired_input_node_id": paired,
                    "parameters": [
                        {**copy.deepcopy(param), "slot": f"param_{index}"}
                        for index, param in enumerate(params)
                    ],
                    "history_binding": _history_binding(node_id),
                })

    contract = _manifest_contract(input_panels, output_panels)
    return {
        "version": 1,
        "source_format": source_format,
        "contract_fingerprint": _contract_fingerprint(contract),
        "input_panels": input_panels,
        "output_panels": output_panels,
    }


def validate_call_inputs(
    manifest: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    reject_unknown: bool = True,
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping):
        raise ValueError("Call inputs must be a JSON object.")
    panels = manifest.get("input_panels", [])
    panel_ids = {str(panel["node_id"]) for panel in panels}
    names = [param["name"] for panel in panels for param in panel.get("parameters", [])]
    scoped = not values or all(str(key) in panel_ids and isinstance(value, Mapping) for key, value in values.items())
    if not scoped and len(names) != len(set(names)):
        raise ValueError("Duplicate parameter names require panel-scoped call inputs.")

    normalised: dict[str, dict[str, Any]] = {}
    consumed_top: set[str] = set()
    for panel in panels:
        panel_id = str(panel["node_id"])
        supplied_panel = values.get(panel_id, {}) if scoped else values
        if not isinstance(supplied_panel, Mapping):
            raise ValueError(f"Call inputs for panel {panel_id} must be an object.")
        if scoped and panel_id in values:
            consumed_top.add(panel_id)
        known_names = {param["name"] for param in panel.get("parameters", [])}
        panel_values: dict[str, Any] = {}
        for param in panel.get("parameters", []):
            name = param["name"]
            if name in supplied_panel:
                supplied = supplied_panel[name]
                if not scoped:
                    consumed_top.add(name)
            else:
                supplied = param.get("current_value", param.get("default"))
            panel_values[name] = validate_runtime_value(param, supplied)
        if reject_unknown:
            unknown = set(map(str, supplied_panel.keys())) - known_names
            if unknown:
                raise ValueError(f"Unknown inputs for panel {panel_id}: {sorted(unknown)}")
        normalised[panel_id] = panel_values

    if reject_unknown:
        unknown_top = set(map(str, values.keys())) - consumed_top
        if unknown_top:
            raise ValueError(f"Unknown panel or input keys: {sorted(unknown_top)}")
    return normalised


def patch_api_prompt(
    api_prompt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    if detect_format(api_prompt) != "api_prompt":
        raise ValueError("Only a ComfyUI API prompt can be patched for execution.")
    if manifest.get("version") != 1 or manifest.get("source_format") != "api_prompt":
        raise ValueError("Patching requires a version-1 manifest discovered from an API prompt.")
    current = discover_manifest(api_prompt)
    if manifest.get("contract_fingerprint") != current.get("contract_fingerprint"):
        raise ValueError("Manifest does not match the API prompt's current panel contract.")

    normalised = validate_call_inputs(current, values)
    patched = copy.deepcopy(dict(api_prompt))
    patched_view = _normalised_node_view(patched)
    for panel in current["input_panels"]:
        node_id = str(panel["node_id"])
        raw_node_id = patched_view[node_id][0]
        for param in panel["parameters"]:
            name = param["name"]
            patched[raw_node_id]["inputs"][name] = normalised[node_id][name]
    return patched


def prepare_api_prompt(
    api_prompt: Mapping[str, Any], values: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = discover_manifest(api_prompt)
    return patch_api_prompt(api_prompt, manifest, values), manifest


def _parse_document_candidates(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        candidates = [candidates]
    runs: list[dict[str, Any]] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except json.JSONDecodeError:
                continue
        if isinstance(candidate, Mapping) and candidate.get("version") == 1:
            runs.append(copy.deepcopy(dict(candidate)))
    return runs


def _latest_document(
    node_output: Mapping[str, Any], output_schema: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    for field in ("document", "document_json", "schema_json"):
        runs = _parse_document_candidates(node_output.get(field, []))
        if runs:
            return runs[-1], runs

    params = node_output.get("params")
    if isinstance(params, list) and params:
        params = params[-1]
    if isinstance(params, Mapping):
        legacy = {
            "version": 1,
            "inputs": {"schema": [], "latest_values": {}},
            "outputs": {"schema": copy.deepcopy(output_schema), "latest_values": copy.deepcopy(dict(params))},
        }
        return legacy, [legacy]
    return None, []


def extract_history_results(
    history: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    prompt_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(history, Mapping):
        raise ValueError("ComfyUI history response must be an object.")
    if prompt_id is not None:
        if prompt_id not in history:
            raise ValueError(f"Prompt {prompt_id!r} is not present in history.")
        entry = history[prompt_id]
        resolved_prompt_id = prompt_id
    elif "outputs" in history:
        entry = history
        resolved_prompt_id = str(history.get("prompt_id", "")) or None
    elif len(history) == 1:
        resolved_prompt_id, entry = next(iter(history.items()))
    else:
        raise ValueError("Specify prompt_id when the history response contains multiple prompts.")
    if not isinstance(entry, Mapping):
        raise ValueError("History entry must be an object.")
    outputs = entry.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise ValueError("History entry has no outputs object.")
    output_view = {str(node_id): value for node_id, value in outputs.items()}

    panels: dict[str, Any] = {}
    for panel in manifest.get("output_panels", []):
        node_id = str(panel["node_id"])
        node_output = output_view.get(node_id)
        if not isinstance(node_output, Mapping):
            raise ValueError(f"Output Panel {node_id} did not execute or is missing from history.")
        output_schema = [
            {key: copy.deepcopy(value) for key, value in param.items() if key not in ("slot", "input_slot")}
            for param in panel.get("parameters", [])
        ]
        latest, runs = _latest_document(node_output, output_schema)
        panels[node_id] = {
            "latest": latest,
            "runs": runs,
            "images": copy.deepcopy(node_output.get("images", [])),
        }
    return {"prompt_id": resolved_prompt_id, "panels": panels}
