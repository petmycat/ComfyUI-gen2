// Gen2 Input/Output Panel frontend extension.
//
// Adds a "Configure" button to Gen2_InputPanel and Gen2_OutputPanel. Clicking
// opens a popup dialog where the user defines parameter rows. The config is
// stored in the node's _config widget (JSON, always serialized).
//
// Parameter config entry:
//   { name, type, default, min, max, step, controlMode, options }
//   - min/max/step, controlMode: INT/FLOAT (controlMode INT only)
//   - options: COMBO only (list of dropdown choices)
//   - default can be null
//
// Each param becomes:
//   - a standard widget named exactly <name> (number/text/toggle/combo) OR a
//     hidden value widget + DOM upload widget (IMAGE). The value widget
//     serializes its current value under <name>, so the API export / prompt
//     contains `"<name>": value` directly.
//   - a typed output slot (input panel) / input slot (output panel) whose type
//     matches the param type (IMAGE→IMAGE, INT→INT, ...).
//
// Uses only ComfyUI standard widget APIs (addWidget / addDOMWidget) so it works
// on both the legacy LiteGraph canvas and the Nodes 2.0 (Vue) renderer.

import { app } from "../../scripts/app.js";

const NODE_TYPES = {
  Gen2_InputPanel: { mode: "input" },
  Gen2_OutputPanel: { mode: "output" },
};
const MAX_PARAMS = 32;
const SUPPORTED_TYPES = ["STRING", "INT", "FLOAT", "BOOLEAN", "IMAGE", "COMBO", "SEED"];
const NUMERIC_TYPES = ["INT", "FLOAT", "SEED"];   // carry min/max/step
const INT_TYPES = ["INT", "SEED"];                // integer precision
const SLOT_PREFIX = "param_";
const PANEL_LINK_NAME = "PANEL_LINK";

const SEED_DEFAULT = 0;
const SEED_MIN = 0;
// JavaScript Number cannot exactly represent ComfyUI's full uint64 seed range.
// Cap interactive values at the largest exact integer instead of silently rounding.
const SEED_MAX = Number.MAX_SAFE_INTEGER;
const SEED_STEP = 1;

function applySeedDefaults(entry, reset = false) {
  if (entry.type !== "SEED") return entry;
  if (reset || entry.default == null) entry.default = SEED_DEFAULT;
  if (reset || entry.min == null) entry.min = SEED_MIN;
  if (reset || entry.max == null) entry.max = SEED_MAX;
  if (reset || entry.step == null) entry.step = SEED_STEP;
  if (reset || !entry.controlMode) entry.controlMode = "randomize";
  return entry;
}

// Slot (socket) type shown for each param type. COMBO outputs a chosen option
// string typed as COMBO so it connects to combo inputs (sampler_name, etc.).
function slotTypeFor(ptype) {
  switch (ptype) {
    case "IMAGE": return "IMAGE";
    case "INT": return "INT";
    case "SEED": return "INT";
    case "FLOAT": return "FLOAT";
    case "BOOLEAN": return "BOOLEAN";
    case "COMBO": return "COMBO";
    default: return "STRING";
  }
}

// ---- Config and runtime state ----
function newParamId() {
  return globalThis.crypto?.randomUUID?.() || `gen2-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function legacyParamId(entry, index) {
  const source = `${index}:${entry.name || ""}:${entry.type || "STRING"}`;
  let hash = 2166136261;
  for (let i = 0; i < source.length; i++) {
    hash ^= source.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return `legacy-${index}-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function parseConfig(raw) {
  if (!raw) return [];
  let entries;
  if (Array.isArray(raw)) entries = raw;
  else {
    try { entries = JSON.parse(raw); } catch (e) { return []; }
  }
  if (!Array.isArray(entries)) return [];
  return entries.map((entry, index) => ({ ...entry, id: entry.id || legacyParamId(entry, index) }));
}

function serializeConfig(entries, mode) {
  return JSON.stringify(entries.map((e) => {
    const o = { id: e.id || newParamId(), name: e.name || "", type: e.type || "STRING" };
    if (mode === "output") return o;
    o.default = e.default ?? null;
    if (NUMERIC_TYPES.includes(e.type)) { o.min = e.min ?? null; o.max = e.max ?? null; o.step = e.step ?? null; }
    if (e.type === "SEED") o.controlMode = e.controlMode || "randomize";
    return o;
  }));
}

function getConfigWidget(node) {
  return node.widgets?.find((w) => w.name === "_config");
}

function setConfig(node, entries, mode) {
  const w = getConfigWidget(node);
  if (w) w.value = serializeConfig(entries, mode);
  return w;
}

function runtimeValues(node) {
  node.properties = node.properties || {};
  node.properties.gen2RuntimeValues = node.properties.gen2RuntimeValues || {};
  return node.properties.gen2RuntimeValues;
}

function readRuntimeValue(node, entry) {
  const values = runtimeValues(node);
  return Object.prototype.hasOwnProperty.call(values, entry.id) ? values[entry.id] : entry.default;
}

function writeRuntimeValue(node, entry, value) {
  runtimeValues(node)[entry.id] = value;
}

function pruneRuntimeValues(node, entries) {
  const values = runtimeValues(node);
  const valid = new Set(entries.map((entry) => entry.id));
  for (const id of Object.keys(values)) if (!valid.has(id)) delete values[id];
}

function convertRuntimeValue(value, oldType, entry) {
  if (value == null || oldType === entry.type) return value;
  if (entry.type === "BOOLEAN") {
    if (value === true || value === false) return value;
    if (value === "true" || value === 1 || value === "1") return true;
    if (value === "false" || value === 0 || value === "0") return false;
    return entry.default;
  }
  if (NUMERIC_TYPES.includes(entry.type)) {
    const number = Number(value);
    if (!Number.isFinite(number)) return entry.default;
    const converted = INT_TYPES.includes(entry.type) ? Math.trunc(number) : number;
    if (entry.min != null && converted < entry.min) return entry.default;
    if (entry.max != null && converted > entry.max) return entry.default;
    return converted;
  }
  if (entry.type === "STRING" || entry.type === "COMBO") return String(value);
  if (entry.type === "IMAGE") return typeof value === "string" && value ? value : entry.default;
  return entry.default;
}

function migrateRuntimeValues(node, oldEntries, newEntries) {
  const values = runtimeValues(node);
  const oldById = new Map(oldEntries.map((entry) => [entry.id, entry]));
  for (const entry of newEntries) {
    const old = oldById.get(entry.id);
    if (!old || !Object.prototype.hasOwnProperty.call(values, entry.id)) {
      values[entry.id] = entry.default;
      continue;
    }
    values[entry.id] = convertRuntimeValue(values[entry.id], old.type, entry);
  }
  pruneRuntimeValues(node, newEntries);
}

// ---- Image helpers ----
async function uploadImage(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("type", "input");
  body.append("overwrite", "true");
  const resp = await fetch("/upload/image", { method: "POST", body });
  if (!resp.ok) throw new Error("Upload failed: " + resp.status);
  return resp.json();
}

function viewUrl(name, subfolder, type) {
  const p = new URLSearchParams({ filename: name, type: type || "input" });
  if (subfolder) p.set("subfolder", subfolder);
  return "/view?" + p.toString();
}

function filenameFromUpload(up) { return up.subfolder ? up.subfolder + "/" + up.name : up.name; }

function imageRefToViewUrl(ref) {
  if (!ref || typeof ref !== "string") return null;
  const slash = ref.lastIndexOf("/");
  const subfolder = slash >= 0 ? ref.slice(0, slash) : "";
  const name = slash >= 0 ? ref.slice(slash + 1) : ref;
  return viewUrl(name, subfolder, "input");
}

// ---- Slot rebuild (fully remove param slots, keep PANEL_LINK at index 0) ----
function rebuildSlots(node, mode, entries) {
  if (mode === "input") {
    if (!node.outputs) node.outputs = [];
    while (node.outputs.length > 1) node.removeOutput(node.outputs.length - 1);
    if (node.outputs.length === 0) node.addOutput(PANEL_LINK_NAME, "*");
    node.outputs[0].name = PANEL_LINK_NAME; node.outputs[0].label = "panel_link"; node.outputs[0].type = "*";
    for (let i = 0; i < entries.length; i++) {
      node.addOutput(SLOT_PREFIX + i, slotTypeFor(entries[i].type));
      const s = node.outputs[node.outputs.length - 1];
      s.label = entries[i].name || SLOT_PREFIX + i;
      s.__gen2ParamId = entries[i].id;
    }
  } else {
    if (!node.inputs) node.inputs = [];
    while (node.inputs.length > 1) node.removeInput(node.inputs.length - 1);
    if (node.inputs.length === 0) node.addInput(PANEL_LINK_NAME, "*");
    node.inputs[0].name = PANEL_LINK_NAME; node.inputs[0].label = "panel_link"; node.inputs[0].type = "*";
    for (let i = 0; i < entries.length; i++) {
      node.addInput(SLOT_PREFIX + i, slotTypeFor(entries[i].type));
      const s = node.inputs[node.inputs.length - 1];
      s.label = entries[i].name || SLOT_PREFIX + i;
      s.__gen2ParamId = entries[i].id;
    }
  }
}

// ---- Link capture / restore by param name (fixes wrong-slot-on-delete) ----
function captureLinks(node, mode) {
  const map = {};
  if (!node.graph) return map;
  if (mode === "input") {
    for (const out of node.outputs || []) {
      if (out.name === PANEL_LINK_NAME || !out.links) continue;
      const key = out.__gen2ParamId || out.label || out.name;
      for (const lid of out.links) {
        const link = node.graph.links[lid];
        if (link) (map[key] = map[key] || []).push({ target_id: link.target_id, target_slot: link.target_slot });
      }
    }
  } else {
    for (const inp of node.inputs || []) {
      if (inp.name === PANEL_LINK_NAME || inp.link == null) continue;
      const key = inp.__gen2ParamId || inp.label || inp.name;
      const link = node.graph.links[inp.link];
      if (link) map[key] = { origin_id: link.origin_id, origin_slot: link.origin_slot };
    }
  }
  return map;
}

function restoreLinks(node, mode, map) {
  if (!node.graph) return;
  if (mode === "input") {
    for (let i = 0; i < node.outputs.length; i++) {
      const out = node.outputs[i];
      if (out.name === PANEL_LINK_NAME) continue;
      const targets = map[out.__gen2ParamId || out.label || out.name];
      if (!targets) continue;
      for (const t of targets) {
        const tnode = node.graph.getNodeById(t.target_id);
        if (tnode) { try { node.connect(i, tnode, t.target_slot); } catch (e) {} }
      }
    }
  } else {
    for (let i = 0; i < node.inputs.length; i++) {
      const inp = node.inputs[i];
      if (inp.name === PANEL_LINK_NAME) continue;
      const src = map[inp.__gen2ParamId || inp.label || inp.name];
      if (!src) continue;
      const snode = node.graph.getNodeById(src.origin_id);
      if (snode) { try { snode.connect(src.origin_slot, node, i); } catch (e) {} }
    }
  }
}

// ---- Managed widget helpers ----
// Uses ComfyUI's node.removeWidget/ensureWidgetRemoved so DOM widgets are
// properly unregistered from the Vue domWidget store. A manual splice leaves
// ghost DOM widgets that break re-adding on reconfigure — the image bug.
//
// IMPORTANT: do NOT manually removeChild the element here. removeWidget() calls
// the widget's onRemove (which unregisters from the Vue store), and Vue's
// WidgetDOM component owns the element's parent — when the widget leaves
// node.widgets, Vue unmounts the WidgetDOM and removes the element itself.
// Manually removeChild races with that and leaves stale references, causing the
// image/schema box to disappear or render in the wrong place after reconfigure.
function clearManagedWidgets(node) {
  if (!node.widgets) return;
  // Snapshot first — removeWidget splices node.widgets while we iterate.
  const managed = node.widgets.filter((w) => w.__gen2Managed);
  for (const w of managed) {
    // Vue removes DOM elements on its next render pass. Hide the retiring
    // element immediately so it cannot remain above the newly registered
    // widget and intercept clicks during that short overlap window.
    try {
      if (w.element) {
        w.element.style.visibility = "hidden";
        w.element.style.pointerEvents = "none";
      }
    } catch (e) {}
    try {
      if (typeof node.ensureWidgetRemoved === "function") node.ensureWidgetRemoved(w);
      else if (typeof node.removeWidget === "function") node.removeWidget(w);
      else {
        try { w.onRemove?.(); } catch (e) {}
        const idx = node.widgets.indexOf(w);
        if (idx >= 0) node.widgets.splice(idx, 1);
      }
    } catch (e) { console.error("[gen2] widget remove error", e); }
  }
  node.gen2ParamWidgets = [];
  node.gen2ControlHandlers = [];
  node.gen2SchemaWidget = null;
}

function addButtonWidget(node, label, cb, deferUntilPointerReleased = false) {
  const callback = deferUntilPointerReleased ? (...args) => {
    // Older LiteGraph builds may invoke a button callback on pointerdown, while
    // newer builds invoke it from pointerup processing. Wait until the canvas
    // pointer is actually released before opening a native modal; otherwise the
    // modal can steal pointerup and leave node_widget stuck.
    const startedAt = performance.now();
    const invokeWhenReleased = () => {
      const pointerDown = app.canvas?.pointer?.isDown;
      if (pointerDown && performance.now() - startedAt < 2000) {
        requestAnimationFrame(invokeWhenReleased);
        return;
      }
      window.setTimeout(() => cb(...args), 0);
    };
    invokeWhenReleased();
  } : cb;
  const w = node.addWidget("button", label, "", callback, { serialize: false });
  w.__gen2Managed = true;
  return w;
}

function addManagedDOMWidget(node, name, element, opts) {
  let w;
  if (typeof node.addDOMWidget === "function") {
    w = node.addDOMWidget(name, "gen2", element, Object.assign({ serialize: false, hideOnZoom: false }, opts || {}));
  } else {
    w = { name, type: "gen2_dom", element, serializeValue: () => undefined };
    node.widgets = node.widgets || [];
    node.widgets.push(w);
  }
  w.__gen2Managed = true;
  w.element = element;
  return w;
}

// ---- Per-parameter value widget(s) ----
// Returns { entry, paramIndex, getValue, setValue }.
function makeParamWidget(node, paramIndex, entry) {
  const t = entry.type;
  const initialValue = readRuntimeValue(node, entry);
  const remember = (value) => writeRuntimeValue(node, entry, value);

  if (t === "INT" || t === "FLOAT" || t === "SEED") {
    const isInt = INT_TYPES.includes(t);
    let mn = entry.min, mx = entry.max, st = entry.step;
    if (t === "SEED") { if (mn == null) mn = SEED_MIN; if (mx == null) mx = SEED_MAX; }
    if (mn == null) mn = isInt ? -Number.MAX_SAFE_INTEGER : -1e12;
    if (mx == null) mx = isInt ? Number.MAX_SAFE_INTEGER : 1e12;
    if (st == null) st = isInt ? 1 : 0.01;
    // step2 is the real drag/step increment; step is the legacy 10x value.
    const opts = { min: mn, max: mx, step: st * 10, step2: st };
    if (isInt) opts.precision = 0;
    const dflt = initialValue != null ? initialValue : (entry.default != null ? entry.default : 0);
    const w = node.addWidget("number", entry.name, dflt, (value) => remember(value), opts);
    w.__gen2Managed = true;
    remember(w.value);
    w.serializeValue = () => { remember(w.value); return w.value; };
    const pw = { entry, paramIndex, getValue: () => w.value, setValue: (v) => { w.value = v; remember(v); } };

    // Only SEED gets control_after_generate (fixed/randomize/increment/decrement).
    if (t === "SEED") {
      const ctrl = node.addWidget("combo", entry.name + " · after run", entry.controlMode || "randomize",
        (v) => { entry.controlMode = v; persistConfig(node, paramIndex, entry); },
        { values: ["fixed", "randomize", "increment", "decrement"], serialize: false });
      ctrl.__gen2Managed = true;
      const applyControlMode = () => {
        const mode = ctrl.value;
        if (mode === "fixed") return;
        let v = w.value; if (v == null || isNaN(v)) v = 0; v = Math.floor(v);
        if (mode === "randomize") { const range = Math.min(mx - mn + 1, Number.MAX_SAFE_INTEGER); v = mn + Math.floor(Math.random() * range); }
        else if (mode === "increment") { v = v + 1; if (v > mx) v = mn; }
        else if (mode === "decrement") { v = v - 1; if (v < mn) v = mx; }
        w.value = v;
        remember(v);
      };
      node.gen2ControlHandlers = node.gen2ControlHandlers || [];
      node.gen2ControlHandlers.push(applyControlMode);
    }
    return pw;
  }

  if (t === "BOOLEAN") {
    const w = node.addWidget("toggle", entry.name, !!initialValue, (value) => remember(!!value));
    w.__gen2Managed = true;
    remember(w.value);
    w.serializeValue = () => { remember(!!w.value); return !!w.value; };
    return { entry, paramIndex, getValue: () => w.value, setValue: (v) => { w.value = !!v; remember(w.value); } };
  }

  if (t === "STRING" || t === "COMBO") {
    const w = node.addWidget("text", entry.name, initialValue ?? "", (value) => remember(value ?? ""));
    w.__gen2Managed = true;
    remember(w.value);
    w.serializeValue = () => { remember(w.value ?? ""); return w.value ?? ""; };
    return { entry, paramIndex, getValue: () => w.value, setValue: (v) => { w.value = v ?? ""; remember(w.value); } };
  }

  // IMAGE: hidden value widget (serializes filename by name) + DOM upload UI.
  // The value widget is a plain object (not a BaseWidget) so it draws nothing
  // and reports zero height; the DOM widget next to it owns the visible UI.
  let imgVal = initialValue ?? entry.default ?? null;
  remember(imgVal);
  const valueWidget = {
    name: entry.name,
    type: "hidden",          // hidden → computeLayoutSize returns 0 (no height)
    get value() { return imgVal; },
    set value(v) { imgVal = v; },
    serializeValue: () => imgVal,
    computeSize: () => [0, -4],
    draw: () => {},
    options: { hideOnZoom: false },
    y: 0,
    __gen2Managed: true,
  };
  node.widgets = node.widgets || [];
  node.widgets.push(valueWidget);

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.style.display = "none";
  // Keep the native picker outside the DOM widget renderer. The standard canvas
  // button remains interactive even when the frontend deactivates DOM widgets.
  document.body.appendChild(fileInput);

  let disposed = false;
  let uploadGeneration = 0;
  const uploadWidget = addButtonWidget(node, "Choose / Upload · " + entry.name, () => fileInput.click(), true);
  const removeUploadWidget = uploadWidget.onRemove?.bind(uploadWidget);
  uploadWidget.onRemove = () => {
    disposed = true;
    try { fileInput.remove(); } catch (e) {}
    removeUploadWidget?.();
  };

  const el = document.createElement("div");
  el.style.display = "flex";
  el.style.flexDirection = "column";
  el.style.gap = "4px";
  el.style.padding = "2px 0";
  // Preview only; interaction is handled by the standard LiteGraph button.
  el.style.pointerEvents = "none";
  const lbl = document.createElement("div");
  lbl.textContent = entry.name; lbl.style.fontSize = "11px"; lbl.style.opacity = "0.85";
  el.appendChild(lbl);
  const thumb = document.createElement("img");
  thumb.style.display = "none"; thumb.style.maxWidth = "100%"; thumb.style.maxHeight = "180px";
  thumb.style.objectFit = "contain"; thumb.style.borderRadius = "3px";
  el.appendChild(thumb);
  const fnameLbl = document.createElement("div");
  fnameLbl.style.fontSize = "10px"; fnameLbl.style.opacity = "0.65";
  fnameLbl.style.overflow = "hidden"; fnameLbl.style.textOverflow = "ellipsis"; fnameLbl.style.whiteSpace = "nowrap";
  el.appendChild(fnameLbl);

  const updateThumb = () => {
    const url = imageRefToViewUrl(imgVal);
    if (url) {
      thumb.src = url + "&gen2_preview=" + Date.now();
      thumb.style.display = "block";
      fnameLbl.textContent = String(imgVal).split("/").pop();
      fnameLbl.title = imgVal;
    } else {
      thumb.removeAttribute("src");
      thumb.style.display = "none";
      fnameLbl.textContent = "no image";
    }
  };
  const doUpload = async (file) => {
    if (!file) return;
    const generation = ++uploadGeneration;
    try {
      const up = await uploadImage(file);
      if (disposed || generation !== uploadGeneration) return;
      imgVal = filenameFromUpload(up);
      remember(imgVal);
      updateThumb();
      refreshPanelLayout(node);
    } catch (err) {
      console.error("[gen2] upload failed", err);
    } finally {
      fileInput.value = "";
      node.setDirtyCanvas?.(true, true);
    }
  };
  fileInput.addEventListener("change", () => doUpload(fileInput.files && fileInput.files[0]));

  addManagedDOMWidget(node, "__imgui_" + entry.name, el, { getMinHeight: () => (imgVal ? 240 : 48), margin: 0 });
  updateThumb();

  return {
    entry, paramIndex,
    getValue: () => imgVal,
    setValue: (v) => { imgVal = v; remember(v); updateThumb(); },
  };
}

function persistConfig(node, paramIndex, entry) {
  const w = getConfigWidget(node);
  if (!w) return;
  const all = parseConfig(w.value || "[]");
  if (all[paramIndex]) all[paramIndex] = Object.assign({}, all[paramIndex], entry);
  w.value = serializeConfig(all, "input");
}

// ---- Result document (output panel) ----
function emptyResultDocument(entries) {
  return {
    version: 1,
    inputs: { schema: [], latest_values: {} },
    outputs: {
      schema: entries.map((e) => ({ id: e.id, name: e.name, type: e.type })),
      latest_values: {},
    },
  };
}

function buildSchemaJsonLocal(entries) {
  return JSON.stringify(emptyResultDocument(entries), null, 2);
}

function addSchemaWidget(node) {
  const el = document.createElement("div");
  el.style.display = "flex"; el.style.flexDirection = "column"; el.style.gap = "3px"; el.style.padding = "2px 0"; el.style.minHeight = "140px";
  const lbl = document.createElement("div");
  lbl.textContent = "Latest input/output document"; lbl.style.fontSize = "11px"; lbl.style.opacity = "0.7";
  el.appendChild(lbl);
  const ta = document.createElement("textarea");
  ta.readOnly = true; ta.rows = 8;
  ta.style.width = "100%"; ta.style.boxSizing = "border-box"; ta.style.minHeight = "120px";
  ta.style.fontSize = "11px"; ta.style.fontFamily = "monospace";
  ta.style.background = "var(--comfy-input-bg, #1a1a1a)"; ta.style.color = "var(--fg-color, #ddd)";
  ta.style.border = "1px solid var(--border-color, #444)"; ta.style.borderRadius = "3px"; ta.style.resize = "vertical";
  ta.placeholder = "Run the workflow to display the latest input/output JSON.";
  ta.addEventListener("focus", () => ta.select());
  el.appendChild(ta);
  const copyBtn = document.createElement("button");
  copyBtn.textContent = "Copy"; copyBtn.style.cursor = "pointer"; copyBtn.style.fontSize = "11px"; copyBtn.style.padding = "2px 8px";
  copyBtn.addEventListener("click", async (e) => {
    e.preventDefault(); e.stopPropagation();
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(ta.value);
      else { ta.select(); if (!document.execCommand("copy")) throw new Error("copy rejected"); }
      copyBtn.textContent = "Copied!";
    } catch (err) {
      copyBtn.textContent = "Copy failed";
      console.error("[gen2] copy failed", err);
    }
    setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200);
  });
  el.appendChild(copyBtn);
  addManagedDOMWidget(node, "__schema", el, { getMinHeight: () => 210 });
  node.gen2SchemaWidget = { element: el, textarea: ta };
}

function refreshSchemaBox(node) {
  if (!node.gen2SchemaWidget) return;
  node.gen2SchemaWidget.textarea.value = node.gen2LatestDocument
    ? JSON.stringify(node.gen2LatestDocument, null, 2)
    : "";
}

function panelRenderSignature(entries) {
  return JSON.stringify(entries.map((e) => {
    const signature = { id: e.id, name: e.name, type: e.type };
    if (NUMERIC_TYPES.includes(e.type)) {
      signature.min = e.min ?? null;
      signature.max = e.max ?? null;
      signature.step = e.step ?? null;
    }
    if (e.type === "SEED") signature.controlMode = e.controlMode || "randomize";
    return signature;
  }));
}

function refreshPanelLayout(node) {
  const applySize = () => {
    try {
      if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
        const cs = node.computeSize();
        const w = Math.max((node.size && node.size[0]) || 0, cs[0]);
        node.setSize([w, cs[1]]);
      }
    } catch (e) {}
    try { node.setDirtyCanvas?.(true, true); } catch (e) {}
    try { app.canvas?.setDirty?.(true, true); } catch (e) {}
  };
  applySize();
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(applySize);
}

function updatePanelDefaultsInPlace(node, mode, entries) {
  const current = parseConfig(getConfigWidget(node)?.value || "[]");
  if (panelRenderSignature(current) !== panelRenderSignature(entries)) return false;
  if (mode === "input" && (!node.gen2ParamWidgets || node.gen2ParamWidgets.length !== entries.length)) return false;

  setConfig(node, entries, mode);
  if (mode === "input") {
    for (let i = 0; i < entries.length; i++) Object.assign(node.gen2ParamWidgets[i].entry, entries[i]);
  } else {
    node.gen2LatestDocument = null;
    refreshSchemaBox(node);
  }
  refreshPanelLayout(node);
  return true;
}

// ---- Rebuild whole panel (buttons + params + slots + schema), preserving links ----
function rebuildPanel(node, mode) {
  const generation = (node.__gen2WidgetGeneration || 0) + 1;
  node.__gen2WidgetGeneration = generation;
  const links = captureLinks(node, mode);
  clearManagedWidgets(node);

  const build = () => {
    if (node.__gen2WidgetGeneration !== generation) return;
    node.gen2ParamWidgets = [];
    node.gen2ControlHandlers = [];
    addButtonWidget(node, "Configure", () => openConfigDialog(node, mode), true);

    const entries = parseConfig(getConfigWidget(node)?.value || "[]");
    setConfig(node, entries, mode);
    pruneRuntimeValues(node, entries);
    rebuildSlots(node, mode, entries);

    if (mode === "input") {
      for (let i = 0; i < entries.length; i++) node.gen2ParamWidgets.push(makeParamWidget(node, i, entries[i]));
      addButtonWidget(node, "Reset to defaults", () => {
        for (const pw of node.gen2ParamWidgets) pw.setValue(pw.entry.default ?? null);
        if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
      });
    } else {
      addSchemaWidget(node);
      refreshSchemaBox(node);
    }

    restoreLinks(node, mode, links);
    refreshPanelLayout(node);
  };

  // Vue owns DOM-widget unmounting. Re-registering an IMAGE preview in the same
  // render turn can let the retiring component remove the new element. Wait for
  // one completed paint before rebuilding the managed widgets.
  if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => requestAnimationFrame(build));
  else window.setTimeout(build, 0);
}

// ---- Configure popup dialog ----
function releaseCanvasInteraction(node) {
  const canvas = app.canvas;
  if (!canvas) return;

  // Clear stale references to managed widgets before/after a native dialog.
  // Support both the current CanvasPointer API and older LiteGraph builds.
  if (canvas.node_widget?.[0] === node) canvas.node_widget = null;
  try {
    if (canvas.pointer?.reset) canvas.pointer.reset();
    else if (canvas.pointer) canvas.pointer.isDown = false;
  } catch (e) {}
  try { canvas.dragging_canvas = false; } catch (e) {}
  try { canvas.last_mouse_dragging = false; } catch (e) {}
  try { canvas.block_click = false; } catch (e) {}
}

function afterDialogClosed(dlg, callback) {
  dlg.addEventListener("close", () => {
    // Let the browser finish removing the top-layer modal before mutating the
    // node's widget list. This prevents Vue DOM-widget unmount/register races.
    window.setTimeout(callback, 0);
  }, { once: true });
  dlg.close();
}

function openConfigDialog(node, mode) {
  releaseCanvasInteraction(node);
  const entries = parseConfig(getConfigWidget(node)?.value || "[]");
  const draft = entries.map((e) => mode === "input" ? applySeedDefaults({ ...e }) : { id: e.id, name: e.name, type: e.type });

  const dlg = document.createElement("dialog");
  dlg.style.minWidth = "680px";
  dlg.style.color = "var(--fg-color, #fff)";
  dlg.style.background = "var(--bg-color, #222)";
  dlg.style.border = "1px solid var(--border-color, #555)";
  dlg.style.padding = "16px"; dlg.style.borderRadius = "8px";

  const title = document.createElement("h3");
  title.textContent = mode === "input" ? "Configure Input Panel" : "Configure Output Panel";
  title.style.margin = "0 0 12px 0";
  dlg.appendChild(title);

  const help = document.createElement("p");
  help.textContent = mode === "input"
    ? "Parameters keep this exact order. Name and default are required for every type; INT/FLOAT/SEED also require min, max, and step."
    : "Parameters keep this exact order. Each output requires only a unique name and type; values come from the connected workflow slots.";
  help.style.opacity = "0.8"; help.style.fontSize = "12px"; help.style.margin = "0 0 12px 0";
  dlg.appendChild(help);

  const safetyBar = document.createElement("div");
  safetyBar.style.display = "none";
  safetyBar.style.padding = "8px 10px";
  safetyBar.style.marginBottom = "10px";
  safetyBar.style.border = "1px solid var(--error-color, #d66)";
  safetyBar.style.borderRadius = "4px";
  safetyBar.style.background = "var(--comfy-input-bg, #332222)";
  safetyBar.style.color = "var(--error-color, #ffaaaa)";
  safetyBar.style.fontSize = "12px";
  safetyBar.setAttribute("role", "alert");
  dlg.appendChild(safetyBar);

  const rowsEl = document.createElement("div");
  rowsEl.style.display = "flex"; rowsEl.style.flexDirection = "column"; rowsEl.style.gap = "8px";
  dlg.appendChild(rowsEl);
  let pendingUploads = 0;

  const showSafetyMessage = (messages) => {
    if (!messages.length) {
      safetyBar.style.display = "none";
      safetyBar.textContent = "";
      return;
    }
    safetyBar.textContent = messages.join(" ");
    safetyBar.style.display = "block";
    safetyBar.scrollIntoView({ block: "nearest" });
  };

  function renderRows() {
    rowsEl.innerHTML = "";
    draft.forEach((entry, idx) => {
      const row = document.createElement("div");
      row.style.display = "flex"; row.style.gap = "6px"; row.style.alignItems = "center"; row.style.flexWrap = "wrap";

      const typeSel = document.createElement("select");
      for (const tp of SUPPORTED_TYPES) { const o = document.createElement("option"); o.value = tp; o.textContent = tp; if (entry.type === tp) o.selected = true; typeSel.appendChild(o); }
      typeSel.onchange = () => {
        entry.type = typeSel.value;
        if (entry.type === "SEED") applySeedDefaults(entry, true);
        renderRows();
      };
      typeSel.style.width = "90px";

      const nameIn = document.createElement("input");
      nameIn.type = "text"; nameIn.value = entry.name || ""; nameIn.placeholder = "name * (API key)";
      nameIn.style.flex = "1"; nameIn.style.minWidth = "110px";
      nameIn.oninput = () => { entry.name = nameIn.value; };

      row.appendChild(typeSel); row.appendChild(nameIn);

      // Input defaults are configuration; Output rows contain name and type only.
      if (mode === "output") {
        // no default/range controls
      } else if (entry.type === "IMAGE") {
        const pickBtn = document.createElement("button");
        pickBtn.textContent = "Upload"; pickBtn.style.fontSize = "11px"; pickBtn.style.padding = "2px 8px";
        const fileInput = document.createElement("input");
        fileInput.type = "file"; fileInput.accept = "image/*"; fileInput.style.display = "none";
        pickBtn.onclick = (e) => { e.preventDefault(); fileInput.click(); };
        const thumb = document.createElement("img");
        thumb.style.maxWidth = "48px"; thumb.style.maxHeight = "48px"; thumb.style.objectFit = "contain"; thumb.style.borderRadius = "3px"; thumb.style.display = "none"; thumb.style.cursor = "pointer";
        thumb.onclick = (e) => { e.preventDefault(); const u = imageRefToViewUrl(entry.default); if (u) window.open(u, "_blank"); };
        const fnameLbl = document.createElement("span");
        fnameLbl.style.fontSize = "10px"; fnameLbl.style.opacity = "0.7"; fnameLbl.style.maxWidth = "80px";
        fnameLbl.style.overflow = "hidden"; fnameLbl.style.textOverflow = "ellipsis"; fnameLbl.style.whiteSpace = "nowrap";
        const refresh = () => {
          const cur = entry.default;
          const u = imageRefToViewUrl(cur);
          if (u) { thumb.src = u + "&gen2_preview=" + Date.now(); thumb.style.display = "inline-block"; }
          else { thumb.removeAttribute("src"); thumb.style.display = "none"; }
          fnameLbl.textContent = (typeof cur === "string" && cur) ? cur.split("/").pop() : "";
          fnameLbl.title = cur || "";
        };
        refresh();
        fileInput.onchange = async () => {
          const file = fileInput.files && fileInput.files[0];
          if (!file) return;
          pendingUploads++;
          pickBtn.disabled = true;
          pickBtn.textContent = "...";
          try {
            const up = await uploadImage(file);
            if (!dlg.isConnected) return;
            entry.default = filenameFromUpload(up);
            refresh();
          } catch (err) {
            if (dlg.isConnected) fnameLbl.textContent = "failed";
          } finally {
            pendingUploads--;
            fileInput.value = "";
            if (dlg.isConnected) {
              pickBtn.disabled = false;
              pickBtn.textContent = "Upload";
            }
          }
        };
        row.appendChild(pickBtn); row.appendChild(fileInput); row.appendChild(thumb); row.appendChild(fnameLbl);
      } else if (entry.type === "BOOLEAN") {
        const boolSel = document.createElement("select");
        for (const bv of ["false", "true"]) { const o = document.createElement("option"); o.value = bv; o.textContent = bv; if (String(!!entry.default) === bv) o.selected = true; boolSel.appendChild(o); }
        boolSel.style.width = "100px"; boolSel.disabled = mode === "output";
        boolSel.onchange = () => { entry.default = boolSel.value === "true"; };
        if (entry.default == null) entry.default = false;
        row.appendChild(boolSel);
      } else {
        const defIn = document.createElement("input");
        const isNum = NUMERIC_TYPES.includes(entry.type);
        defIn.type = isNum ? "number" : "text";
        defIn.value = entry.default ?? "";
        defIn.placeholder = entry.type === "COMBO" ? "default option *" : "default *";
        defIn.style.width = "110px";
        defIn.disabled = mode === "output";
        defIn.oninput = () => { const v = defIn.value; if (v === "") { entry.default = null; return; } if (NUMERIC_TYPES.includes(entry.type)) entry.default = Number(v); else entry.default = v; };
        row.appendChild(defIn);
      }

      // Range/step for numeric types (INT/FLOAT/SEED). SEED also gets after-run.
      if (mode === "input" && NUMERIC_TYPES.includes(entry.type)) {
        const mkNum = (key, ph) => { const inp = document.createElement("input"); inp.type = "number"; inp.value = entry[key] != null ? entry[key] : ""; inp.placeholder = mode === "input" ? ph + " *" : ph; inp.style.width = "62px"; inp.disabled = mode === "output"; inp.oninput = () => { const v = inp.value; entry[key] = v === "" ? null : Number(v); }; return inp; };
        for (const [k, label] of [["min", "min"], ["max", "max"], ["step", "step"]]) { const l = document.createElement("span"); l.textContent = label; l.style.fontSize = "10px"; l.style.opacity = "0.6"; row.appendChild(l); row.appendChild(mkNum(k, label)); }
        if (entry.type === "SEED") {
          const l = document.createElement("span"); l.textContent = `Exact JavaScript seed range: 0 to ${Number.MAX_SAFE_INTEGER}, step 1`; l.style.fontSize = "9px"; l.style.opacity = "0.5"; l.style.flexBasis = "100%";
          row.appendChild(l);
          if (mode === "input") {
            const l2 = document.createElement("span"); l2.textContent = "after run"; l2.style.fontSize = "10px"; l2.style.opacity = "0.6";
            const ctrlSel = document.createElement("select"); ctrlSel.style.width = "85px"; ctrlSel.style.fontSize = "11px";
            for (const cm of ["fixed", "randomize", "increment", "decrement"]) { const o = document.createElement("option"); o.value = cm; o.textContent = cm; if ((entry.controlMode || "randomize") === cm) o.selected = true; ctrlSel.appendChild(o); }
            ctrlSel.onchange = () => { entry.controlMode = ctrlSel.value; };
            row.appendChild(l2); row.appendChild(ctrlSel);
          }
        }
      }

      const delBtn = document.createElement("button");
      delBtn.textContent = "✕"; delBtn.title = "Remove";
      delBtn.onclick = () => { draft.splice(idx, 1); renderRows(); updateCount(); };
      row.appendChild(delBtn);
      rowsEl.appendChild(row);
    });
  }

  const addBtn = document.createElement("button");
  addBtn.textContent = "+ Add parameter"; addBtn.style.marginTop = "8px";
  addBtn.onclick = () => {
    if (draft.length >= MAX_PARAMS) return;
    draft.push(mode === "input" ? { id: newParamId(), name: "", type: "STRING", default: null } : { id: newParamId(), name: "", type: "STRING" });
    renderRows(); updateCount();
  };

  const countLabel = document.createElement("div");
  countLabel.style.fontSize = "11px"; countLabel.style.opacity = "0.6"; countLabel.style.marginTop = "4px";
  function updateCount() { countLabel.textContent = draft.length + " / " + MAX_PARAMS + " parameters"; addBtn.disabled = draft.length >= MAX_PARAMS; }
  renderRows(); updateCount();
  dlg.appendChild(addBtn); dlg.appendChild(countLabel);

  const btnRow = document.createElement("div");
  btnRow.style.display = "flex"; btnRow.style.gap = "8px"; btnRow.style.justifyContent = "flex-end"; btnRow.style.marginTop = "16px";
  let finishing = false;
  const finish = (applyChanges) => {
    if (finishing) return;

    let clean = null;
    if (applyChanges) {
      if (pendingUploads > 0) {
        showSafetyMessage(["Wait for all image uploads to finish before applying."]);
        return;
      }
      const errors = [];
      const seen = new Set(); clean = [];
      for (let i = 0; i < draft.length; i++) {
        const e = draft[i];
        const rowLabel = `Parameter ${i + 1}`;
        const n = (e.name || "").trim();
        if (!SUPPORTED_TYPES.includes(e.type)) {
          errors.push(`${rowLabel} has an unsupported type.`);
          continue;
        }
        if (!n) {
          errors.push(`${rowLabel} needs a name.`);
          continue;
        }
        if (seen.has(n)) {
          errors.push(`Parameter name '${n}' is duplicated.`);
          continue;
        }
        const missingDefault = e.default == null || (typeof e.default === "string" && e.default.trim() === "");
        if (mode === "input" && missingDefault && !NUMERIC_TYPES.includes(e.type)) {
          errors.push(`${rowLabel} ('${n}') needs a default value.`);
          continue;
        }

        if (mode === "input" && NUMERIC_TYPES.includes(e.type)) {
          const numericFields = [
            ["default", e.default],
            ["min", e.min],
            ["max", e.max],
            ["step", e.step],
          ];
          let numericInvalid = false;
          for (const [field, value] of numericFields) {
            if (value == null || value === "" || !Number.isFinite(Number(value))) {
              errors.push(`${rowLabel} ('${n}') needs a valid ${field}.`);
              numericInvalid = true;
            } else if (INT_TYPES.includes(e.type) && !Number.isSafeInteger(Number(value))) {
              errors.push(`${rowLabel} ('${n}') ${field} must be an exactly representable integer.`);
              numericInvalid = true;
            }
          }
          if (numericInvalid) continue;

          const defaultValue = Number(e.default);
          const minValue = Number(e.min);
          const maxValue = Number(e.max);
          const stepValue = Number(e.step);
          if (e.type === "SEED" && (minValue < SEED_MIN || defaultValue < SEED_MIN || maxValue > SEED_MAX)) {
            errors.push(`${rowLabel} ('${n}') seed values must be between ${SEED_MIN} and ${SEED_MAX}.`);
            continue;
          }
          if (minValue > maxValue) {
            errors.push(`${rowLabel} ('${n}') min must not exceed max.`);
            continue;
          }
          if (defaultValue < minValue || defaultValue > maxValue) {
            errors.push(`${rowLabel} ('${n}') default must be between min and max.`);
            continue;
          }
          if (stepValue <= 0) {
            errors.push(`${rowLabel} ('${n}') step must be greater than 0.`);
            continue;
          }
          e.default = INT_TYPES.includes(e.type) ? Math.trunc(defaultValue) : defaultValue;
          e.min = INT_TYPES.includes(e.type) ? Math.trunc(minValue) : minValue;
          e.max = INT_TYPES.includes(e.type) ? Math.trunc(maxValue) : maxValue;
          e.step = INT_TYPES.includes(e.type) ? Math.trunc(stepValue) : stepValue;
        }

        if (mode === "input" && e.type === "SEED") {
          const controlModes = ["fixed", "randomize", "increment", "decrement"];
          if (!controlModes.includes(e.controlMode || "randomize")) {
            errors.push(`${rowLabel} ('${n}') has an invalid after-run mode.`);
            continue;
          }
        }

        const c = { id: e.id || newParamId(), name: n, type: e.type };
        if (mode === "input") c.default = e.default;
        if (mode === "input" && NUMERIC_TYPES.includes(e.type)) { c.min = e.min ?? null; c.max = e.max ?? null; c.step = e.step ?? null; }
        if (mode === "input" && e.type === "SEED") c.controlMode = e.controlMode || "randomize";
        clean.push(c);
        seen.add(n);
      }
      if (errors.length) {
        showSafetyMessage(errors);
        return;
      }
      showSafetyMessage([]);
    }

    finishing = true;
    cancelBtn.disabled = true;
    okBtn.disabled = true;
    afterDialogClosed(dlg, () => {
      dlg.remove();
      releaseCanvasInteraction(node);
      if (clean) {
        if (mode === "input") migrateRuntimeValues(node, entries, clean);
        if (!updatePanelDefaultsInPlace(node, mode, clean)) {
          setConfig(node, clean, mode);
          rebuildPanel(node, mode);
        }
      } else {
        // Cancel must not rebuild or replace working widgets. It only dismisses
        // the draft dialog and restores canvas interaction.
        node.setDirtyCanvas?.(true, true);
        app.canvas?.setDirty?.(true, true);
      }
    });
  };

  const cancelBtn = document.createElement("button"); cancelBtn.textContent = "Cancel"; cancelBtn.onclick = () => finish(false);
  const okBtn = document.createElement("button"); okBtn.textContent = "Apply"; okBtn.onclick = () => finish(true);
  btnRow.appendChild(cancelBtn); btnRow.appendChild(okBtn); dlg.appendChild(btnRow);
  document.body.appendChild(dlg);
  dlg.addEventListener("cancel", (event) => { event.preventDefault(); finish(false); });
  dlg.showModal();
}

// ---- Extension registration ----
app.registerExtension({
  name: "gen2.api_panels",

  async setup() {},

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_TYPES[nodeData.name]) return;
    const mode = NODE_TYPES[nodeData.name].mode;

    const scheduleRebuild = (node) => {
      const generation = (node.__gen2RebuildGeneration || 0) + 1;
      node.__gen2RebuildGeneration = generation;
      setTimeout(() => {
        if (node.__gen2RebuildGeneration !== generation) return;
        try { rebuildPanel(node, mode); } catch (err) { console.error("[gen2] rebuild error", err); }
      }, 0);
    };

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      try { origOnNodeCreated?.apply(this, arguments); } catch (err) {}
      try {
        const cw = getConfigWidget(this);
        if (cw) { cw.type = "hidden"; cw.computeSize = () => [0, -4]; cw.hidden = true; }
        scheduleRebuild(this);
      } catch (err) { console.error("[gen2] onNodeCreated error", err); }
    };

    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      try { origOnConfigure?.apply(this, arguments); } catch (err) {}
      scheduleRebuild(this);
    };

    if (mode === "input") {
      const origOnExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        try { origOnExecuted?.apply(this, arguments); } catch (err) {}
        const promptId = message?.prompt_id ?? message?.promptId ?? message?.detail?.prompt_id ?? null;
        if (promptId != null && this.__gen2LastAdvancedPrompt === promptId) return;
        this.__gen2LastAdvancedPrompt = promptId ?? Symbol("execution");
        for (const handler of this.gen2ControlHandlers || []) {
          try { handler(); } catch (err) { console.error("[gen2] control handler error", err); }
        }
        this.setDirtyCanvas?.(true, true);
      };
    }

    // Output panel keeps its own slots; PANEL_LINK only enriches execution documents.
    if (mode === "output") {
      const origOnConn = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function () {
        try { origOnConn?.apply(this, arguments); } catch (err) {}
        try { refreshSchemaBox(this); } catch (err) {}
      };
      const origOnExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        try { origOnExecuted?.apply(this, arguments); } catch (err) {}
        try {
          const ui = message?.ui || message?.output?.ui || message?.output || message || {};
          let documentValue = ui.document ?? ui.document_json ?? ui.schema_json;
          if (Array.isArray(documentValue)) documentValue = documentValue.at(-1);
          if (typeof documentValue === "string") documentValue = JSON.parse(documentValue);
          if (!documentValue || typeof documentValue !== "object") return;
          const promptId = message?.prompt_id ?? message?.promptId ?? message?.detail?.prompt_id ?? null;
          if (promptId != null && this.__gen2LatestPromptId === promptId) return;
          this.__gen2LatestPromptId = promptId;
          this.gen2LatestDocument = documentValue;
          refreshSchemaBox(this);
        } catch (err) { console.error("[gen2] output document parse failed", err); }
      };
    }
  },
});
