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
const SUPPORTED_TYPES = ["STRING", "INT", "FLOAT", "BOOLEAN", "IMAGE", "COMBO"];
const NUMERIC_TYPES = ["INT", "FLOAT"];
const SLOT_PREFIX = "param_";
const PANEL_LINK_NAME = "PANEL_LINK";

// Slot (socket) type shown for each param type. COMBO outputs a chosen string.
function slotTypeFor(ptype) {
  switch (ptype) {
    case "IMAGE": return "IMAGE";
    case "INT": return "INT";
    case "FLOAT": return "FLOAT";
    case "BOOLEAN": return "BOOLEAN";
    case "COMBO": return "STRING";
    default: return "STRING";
  }
}

// ---- Config (de)serialization ----
function parseConfig(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  try { const v = JSON.parse(raw); return Array.isArray(v) ? v : []; } catch (e) { return []; }
}

function serializeConfig(entries) {
  return JSON.stringify(entries.map((e) => {
    const o = { name: e.name || "", type: e.type || "STRING", default: e.default ?? null };
    if (NUMERIC_TYPES.includes(e.type)) { o.min = e.min ?? null; o.max = e.max ?? null; o.step = e.step ?? null; }
    if (e.type === "INT" && e.controlMode && e.controlMode !== "fixed") o.controlMode = e.controlMode;
    if (e.type === "COMBO") o.options = Array.isArray(e.options) ? e.options : [];
    return o;
  }));
}

function getConfigWidget(node) {
  return node.widgets?.find((w) => w.name === "_config");
}

function setConfig(node, entries) {
  const w = getConfigWidget(node);
  if (w) w.value = serializeConfig(entries);
  return w;
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
      const key = out.label || out.name;
      for (const lid of out.links) {
        const link = node.graph.links[lid];
        if (link) (map[key] = map[key] || []).push({ target_id: link.target_id, target_slot: link.target_slot });
      }
    }
  } else {
    for (const inp of node.inputs || []) {
      if (inp.name === PANEL_LINK_NAME || inp.link == null) continue;
      const key = inp.label || inp.name;
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
      const targets = map[out.label || out.name];
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
      const src = map[inp.label || inp.name];
      if (!src) continue;
      const snode = node.graph.getNodeById(src.origin_id);
      if (snode) { try { snode.connect(src.origin_slot, node, i); } catch (e) {} }
    }
  }
}

// ---- Managed widget helpers ----
function clearManagedWidgets(node) {
  if (!node.widgets) return;
  const keep = [];
  for (const w of node.widgets) {
    if (w.__gen2Managed) {
      try { if (w.element && w.element.parentNode) w.element.parentNode.removeChild(w.element); } catch (e) {}
      try { w.onRemove?.(); } catch (e) {}
      continue;
    }
    keep.push(w);
  }
  node.widgets = keep;
  node.gen2ParamWidgets = [];
  node.gen2ControlHandlers = [];
  node.gen2SchemaWidget = null;
}

function addButtonWidget(node, label, cb) {
  const w = node.addWidget("button", label, "", cb, { serialize: false });
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

  if (t === "INT" || t === "FLOAT") {
    const opts = { min: entry.min ?? undefined, max: entry.max ?? undefined, step: entry.step ?? undefined };
    if (t === "INT") opts.precision = 0;
    const w = node.addWidget("number", entry.name, entry.default ?? 0, () => {}, opts);
    w.__gen2Managed = true;
    const pw = { entry, paramIndex, getValue: () => w.value, setValue: (v) => { w.value = v; } };

    if (t === "INT") {
      const ctrl = node.addWidget("combo", entry.name + " · after run", entry.controlMode || "fixed",
        (v) => { entry.controlMode = v; persistConfig(node, paramIndex, entry); },
        { values: ["fixed", "randomize", "increment", "decrement"], serialize: false });
      ctrl.__gen2Managed = true;
      const applyControlMode = () => {
        const mode = ctrl.value;
        if (mode === "fixed") return;
        let v = w.value; if (v == null || isNaN(v)) v = 0; v = parseInt(v, 10);
        const lo = entry.min != null ? entry.min : 0;
        const hi = entry.max != null ? entry.max : 0xffffffffffffffff;
        if (mode === "randomize") v = Math.floor(Math.random() * (hi - lo + 1)) + lo;
        else if (mode === "increment") { v = v + 1; if (v > hi) v = lo; }
        else if (mode === "decrement") { v = v - 1; if (v < lo) v = hi; }
        w.value = v;
      };
      node.gen2ControlHandlers = node.gen2ControlHandlers || [];
      node.gen2ControlHandlers.push(applyControlMode);
    }
    return pw;
  }

  if (t === "BOOLEAN") {
    const w = node.addWidget("toggle", entry.name, !!entry.default, () => {});
    w.__gen2Managed = true;
    return { entry, paramIndex, getValue: () => w.value, setValue: (v) => { w.value = !!v; } };
  }

  if (t === "STRING") {
    const w = node.addWidget("text", entry.name, entry.default ?? "", () => {});
    w.__gen2Managed = true;
    return { entry, paramIndex, getValue: () => w.value, setValue: (v) => { w.value = v ?? ""; } };
  }

  if (t === "COMBO") {
    const values = Array.isArray(entry.options) && entry.options.length ? entry.options : [""];
    const def = (entry.default != null && values.includes(entry.default)) ? entry.default : values[0];
    const w = node.addWidget("combo", entry.name, def, () => {}, { values });
    w.__gen2Managed = true;
    return { entry, paramIndex, getValue: () => w.value, setValue: (v) => { if (values.includes(v)) w.value = v; } };
  }

  // IMAGE: hidden value widget (serializes filename by name) + DOM upload UI.
  let imgVal = entry.default ?? null;
  const valueWidget = {
    name: entry.name,
    type: "gen2_image_value",
    get value() { return imgVal; },
    set value(v) { imgVal = v; },
    serializeValue: () => imgVal,
    computeSize: () => [0, -4],
    draw: () => {},
    __gen2Managed: true,
  };
  node.widgets = node.widgets || [];
  node.widgets.push(valueWidget);

  const el = document.createElement("div");
  el.style.display = "flex"; el.style.flexDirection = "column"; el.style.gap = "4px"; el.style.padding = "2px 0";
  const lbl = document.createElement("div");
  lbl.textContent = entry.name; lbl.style.fontSize = "11px"; lbl.style.opacity = "0.85";
  el.appendChild(lbl);
  const btn = document.createElement("button");
  btn.textContent = "Choose / Upload image";
  btn.style.cursor = "pointer"; btn.style.padding = "4px 8px"; btn.style.fontSize = "12px"; btn.style.borderRadius = "4px";
  btn.style.background = "var(--comfy-input-bg, #333)"; btn.style.color = "var(--fg-color, #fff)"; btn.style.border = "1px solid var(--border-color, #555)";
  el.appendChild(btn);
  const fileInput = document.createElement("input");
  fileInput.type = "file"; fileInput.accept = "image/*"; fileInput.style.display = "none";
  el.appendChild(fileInput);
  const thumb = document.createElement("img");
  thumb.style.display = "none"; thumb.style.maxWidth = "100%"; thumb.style.maxHeight = "180px";
  thumb.style.objectFit = "contain"; thumb.style.borderRadius = "3px"; thumb.style.cursor = "pointer";
  thumb.title = "Click to view full size";
  el.appendChild(thumb);
  const fnameLbl = document.createElement("div");
  fnameLbl.style.fontSize = "10px"; fnameLbl.style.opacity = "0.65";
  fnameLbl.style.overflow = "hidden"; fnameLbl.style.textOverflow = "ellipsis"; fnameLbl.style.whiteSpace = "nowrap";
  el.appendChild(fnameLbl);

  const updateThumb = () => {
    const url = imageRefToViewUrl(imgVal);
    if (url) { thumb.src = url; thumb.style.display = "block"; fnameLbl.textContent = String(imgVal).split("/").pop(); fnameLbl.title = imgVal; }
    else { thumb.style.display = "none"; fnameLbl.textContent = "no image"; }
  };
  const doUpload = async (file) => {
    if (!file) return;
    btn.textContent = "Uploading...";
    try {
      const up = await uploadImage(file);
      imgVal = filenameFromUpload(up);
      entry.default = imgVal;
      persistConfig(node, paramIndex, entry);
      updateThumb();
    } catch (err) { console.error("[gen2] upload failed", err); }
    btn.textContent = "Choose / Upload image";
  };
  fileInput.addEventListener("change", () => doUpload(fileInput.files && fileInput.files[0]));
  btn.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); fileInput.click(); });
  thumb.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); const u = imageRefToViewUrl(imgVal); if (u) window.open(u, "_blank"); });
  el.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); el.style.outline = "2px solid var(--p-primary-color, #4af)"; });
  el.addEventListener("dragleave", (e) => { e.preventDefault(); e.stopPropagation(); el.style.outline = "none"; });
  el.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); el.style.outline = "none"; const f = e.dataTransfer?.files && e.dataTransfer.files[0]; if (f) doUpload(f); });

  addManagedDOMWidget(node, "__imgui_" + entry.name, el);
  updateThumb();

  return {
    entry, paramIndex,
    getValue: () => imgVal,
    setValue: (v) => { imgVal = v; entry.default = v; updateThumb(); },
  };
}

function persistConfig(node, paramIndex, entry) {
  const w = getConfigWidget(node);
  if (!w) return;
  const all = parseConfig(w.value || "[]");
  if (all[paramIndex]) all[paramIndex] = Object.assign({}, all[paramIndex], entry);
  w.value = serializeConfig(all);
}

// ---- JSON schema (output panel) ----
function buildSchemaJsonLocal(entries) {
  const out = entries.map((e) => {
    const o = { name: e.name, type: e.type, default: e.default ?? null };
    if (NUMERIC_TYPES.includes(e.type)) { o.min = e.min ?? null; o.max = e.max ?? null; o.step = e.step ?? null; }
    if (e.type === "INT" && e.controlMode && e.controlMode !== "fixed") o.controlMode = e.controlMode;
    if (e.type === "COMBO") o.options = Array.isArray(e.options) ? e.options : [];
    return o;
  });
  return JSON.stringify(out, null, 2);
}

function addSchemaWidget(node) {
  const el = document.createElement("div");
  el.style.display = "flex"; el.style.flexDirection = "column"; el.style.gap = "3px"; el.style.padding = "2px 0"; el.style.minHeight = "140px";
  const lbl = document.createElement("div");
  lbl.textContent = "JSON schema"; lbl.style.fontSize = "11px"; lbl.style.opacity = "0.7";
  el.appendChild(lbl);
  const ta = document.createElement("textarea");
  ta.readOnly = true; ta.rows = 8;
  ta.style.width = "100%"; ta.style.boxSizing = "border-box"; ta.style.minHeight = "120px";
  ta.style.fontSize = "11px"; ta.style.fontFamily = "monospace";
  ta.style.background = "var(--comfy-input-bg, #1a1a1a)"; ta.style.color = "var(--fg-color, #ddd)";
  ta.style.border = "1px solid var(--border-color, #444)"; ta.style.borderRadius = "3px"; ta.style.resize = "vertical";
  ta.placeholder = "Connect an Input Panel's PANEL_LINK (or configure this node) to see the schema.";
  ta.addEventListener("focus", () => ta.select());
  el.appendChild(ta);
  const copyBtn = document.createElement("button");
  copyBtn.textContent = "Copy"; copyBtn.style.cursor = "pointer"; copyBtn.style.fontSize = "11px"; copyBtn.style.padding = "2px 8px";
  copyBtn.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); ta.select(); try { document.execCommand("copy"); copyBtn.textContent = "Copied!"; setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200); } catch (err) {} });
  el.appendChild(copyBtn);
  addManagedDOMWidget(node, "__schema", el);
  node.gen2SchemaWidget = { element: el, textarea: ta };
}

// Prefer the config from a connected Input Panel (via PANEL_LINK); else own config.
function refreshSchemaBox(node) {
  if (!node.gen2SchemaWidget) return;
  let cfgRaw = getConfigWidget(node)?.value || "[]";
  try {
    const pin = node.inputs?.find((i) => i.name === PANEL_LINK_NAME);
    if (pin && pin.link != null && node.graph) {
      const link = node.graph.links[pin.link];
      const src = link && node.graph.getNodeById(link.origin_id);
      if (src && src.comfyClass === "Gen2_InputPanel") {
        const scw = src.widgets?.find((w) => w.name === "_config");
        if (scw && scw.value) cfgRaw = scw.value;
      }
    }
  } catch (e) {}
  node.gen2SchemaWidget.textarea.value = buildSchemaJsonLocal(parseConfig(cfgRaw));
}

// ---- Rebuild whole panel (buttons + params + slots + schema), preserving links ----
function rebuildPanel(node, mode) {
  const links = captureLinks(node, mode);
  clearManagedWidgets(node);
  node.gen2ParamWidgets = [];
  node.gen2ControlHandlers = [];

  addButtonWidget(node, "Configure", () => openConfigDialog(node, mode));

  const entries = parseConfig(getConfigWidget(node)?.value || "[]");
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
  if (typeof node.setSize === "function") node.setSize(node.computeSize());
  if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
}

// ---- Configure popup dialog ----
function openConfigDialog(node, mode) {
  const entries = parseConfig(getConfigWidget(node)?.value || "[]");
  const draft = entries.map((e) => ({ ...e }));

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
    ? "Each parameter becomes a typed output slot; its Name is the API-export key and appears as a widget on the node."
    : "Each parameter becomes a typed input slot. IMAGE inputs are saved to the output folder; a JSON schema is shown on the node.";
  help.style.opacity = "0.8"; help.style.fontSize = "12px"; help.style.margin = "0 0 12px 0";
  dlg.appendChild(help);

  const rowsEl = document.createElement("div");
  rowsEl.style.display = "flex"; rowsEl.style.flexDirection = "column"; rowsEl.style.gap = "8px";
  dlg.appendChild(rowsEl);

  function renderRows() {
    rowsEl.innerHTML = "";
    draft.forEach((entry, idx) => {
      const row = document.createElement("div");
      row.style.display = "flex"; row.style.gap = "6px"; row.style.alignItems = "center"; row.style.flexWrap = "wrap";

      const typeSel = document.createElement("select");
      for (const tp of SUPPORTED_TYPES) { const o = document.createElement("option"); o.value = tp; o.textContent = tp; if (entry.type === tp) o.selected = true; typeSel.appendChild(o); }
      typeSel.onchange = () => { entry.type = typeSel.value; renderRows(); };
      typeSel.style.width = "90px";

      const nameIn = document.createElement("input");
      nameIn.type = "text"; nameIn.value = entry.name || ""; nameIn.placeholder = "param name (API key)";
      nameIn.style.flex = "1"; nameIn.style.minWidth = "110px";
      nameIn.oninput = () => { entry.name = nameIn.value; };

      row.appendChild(typeSel); row.appendChild(nameIn);

      // Default cell
      if (mode === "input" && entry.type === "IMAGE") {
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
        const refresh = () => { const cur = entry.default; const u = imageRefToViewUrl(cur); if (u) { thumb.src = u; thumb.style.display = "inline-block"; } else { thumb.style.display = "none"; } fnameLbl.textContent = (typeof cur === "string" && cur) ? cur.split("/").pop() : ""; fnameLbl.title = cur || ""; };
        refresh();
        fileInput.onchange = async () => { const file = fileInput.files && fileInput.files[0]; if (!file) return; pickBtn.textContent = "..."; try { const up = await uploadImage(file); entry.default = filenameFromUpload(up); refresh(); } catch (err) { fnameLbl.textContent = "failed"; } finally { pickBtn.textContent = "Upload"; } };
        row.appendChild(pickBtn); row.appendChild(fileInput); row.appendChild(thumb); row.appendChild(fnameLbl);
      } else {
        const defIn = document.createElement("input");
        defIn.type = NUMERIC_TYPES.includes(entry.type) ? "number" : "text";
        defIn.value = entry.default ?? ""; defIn.placeholder = "default"; defIn.style.width = "100px";
        defIn.disabled = mode === "output";
        defIn.oninput = () => { const v = defIn.value; if (v === "") { entry.default = null; return; } if (entry.type === "INT") entry.default = parseInt(v, 10); else if (entry.type === "FLOAT") entry.default = parseFloat(v); else if (entry.type === "BOOLEAN") entry.default = v.toLowerCase() === "true" || v === "1"; else entry.default = v; };
        row.appendChild(defIn);
      }

      // COMBO options
      if (entry.type === "COMBO") {
        const optIn = document.createElement("input");
        optIn.type = "text";
        optIn.value = Array.isArray(entry.options) ? entry.options.join(", ") : "";
        optIn.placeholder = "options (comma separated)"; optIn.style.flexBasis = "100%"; optIn.style.minWidth = "200px";
        optIn.disabled = mode === "output";
        optIn.oninput = () => { entry.options = optIn.value.split(",").map((s) => s.trim()).filter((s) => s); };
        row.appendChild(optIn);
      }

      // Range/step + after-run for numeric
      if (NUMERIC_TYPES.includes(entry.type)) {
        const mkNum = (key, ph) => { const inp = document.createElement("input"); inp.type = "number"; inp.value = entry[key] != null ? entry[key] : ""; inp.placeholder = ph; inp.style.width = "58px"; inp.disabled = mode === "output"; inp.oninput = () => { const v = inp.value; if (v === "") { entry[key] = null; return; } entry[key] = entry.type === "INT" ? parseInt(v, 10) : parseFloat(v); }; return inp; };
        for (const [k, label] of [["min", "min"], ["max", "max"], ["step", "step"]]) { const l = document.createElement("span"); l.textContent = label; l.style.fontSize = "10px"; l.style.opacity = "0.6"; row.appendChild(l); row.appendChild(mkNum(k, label)); }
        if (entry.type === "INT" && mode === "input") {
          const l = document.createElement("span"); l.textContent = "after run"; l.style.fontSize = "10px"; l.style.opacity = "0.6";
          const ctrlSel = document.createElement("select"); ctrlSel.style.width = "85px"; ctrlSel.style.fontSize = "11px";
          for (const cm of ["fixed", "randomize", "increment", "decrement"]) { const o = document.createElement("option"); o.value = cm; o.textContent = cm; if ((entry.controlMode || "fixed") === cm) o.selected = true; ctrlSel.appendChild(o); }
          ctrlSel.onchange = () => { entry.controlMode = ctrlSel.value; };
          row.appendChild(l); row.appendChild(ctrlSel);
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
  addBtn.onclick = () => { if (draft.length >= MAX_PARAMS) return; draft.push({ name: "", type: "STRING", default: null }); renderRows(); updateCount(); };

  const countLabel = document.createElement("div");
  countLabel.style.fontSize = "11px"; countLabel.style.opacity = "0.6"; countLabel.style.marginTop = "4px";
  function updateCount() { countLabel.textContent = draft.length + " / " + MAX_PARAMS + " parameters"; addBtn.disabled = draft.length >= MAX_PARAMS; }
  renderRows(); updateCount();
  dlg.appendChild(addBtn); dlg.appendChild(countLabel);

  const btnRow = document.createElement("div");
  btnRow.style.display = "flex"; btnRow.style.gap = "8px"; btnRow.style.justifyContent = "flex-end"; btnRow.style.marginTop = "16px";
  const cancelBtn = document.createElement("button"); cancelBtn.textContent = "Cancel"; cancelBtn.onclick = () => { dlg.close(); };
  const okBtn = document.createElement("button"); okBtn.textContent = "Apply";
  okBtn.onclick = () => {
    const seen = new Set(); const clean = [];
    for (const e of draft) {
      const n = (e.name || "").trim();
      if (!n || seen.has(n)) continue;
      seen.add(n);
      const c = { name: n, type: e.type, default: e.default ?? null };
      if (NUMERIC_TYPES.includes(e.type)) { c.min = e.min ?? null; c.max = e.max ?? null; c.step = e.step ?? null; }
      if (e.type === "INT" && e.controlMode && e.controlMode !== "fixed") c.controlMode = e.controlMode;
      if (e.type === "COMBO") c.options = Array.isArray(e.options) ? e.options : [];
      clean.push(c);
    }
    setConfig(node, clean);
    rebuildPanel(node, mode);
    dlg.close();
  };
  btnRow.appendChild(cancelBtn); btnRow.appendChild(okBtn); dlg.appendChild(btnRow);
  document.body.appendChild(dlg); dlg.showModal();
  dlg.addEventListener("close", () => { dlg.remove(); });
}

// ---- Extension registration ----
app.registerExtension({
  name: "gen2.api_panels",

  async setup() {
    try {
      if (!app.api) return;
      app.api.addEventListener("executed", () => {
        try {
          const graph = app.graph; if (!graph || !graph.nodes) return;
          for (const n of graph.nodes) {
            if (n.comfyClass === "Gen2_InputPanel" && n.gen2ControlHandlers) {
              for (const h of n.gen2ControlHandlers) { try { h(); } catch (err) { console.error("[gen2] control handler error", err); } }
            }
          }
        } catch (err) { console.error("[gen2] executed handler error", err); }
      });
    } catch (err) { console.error("[gen2] setup error", err); }
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_TYPES[nodeData.name]) return;
    const mode = NODE_TYPES[nodeData.name].mode;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      try { origOnNodeCreated?.apply(this, arguments); } catch (err) {}
      try {
        const cw = getConfigWidget(this);
        if (cw) { cw.type = "hidden"; cw.computeSize = () => [0, -4]; cw.hidden = true; }
        const self = this;
        setTimeout(() => { try { rebuildPanel(self, mode); } catch (err) { console.error("[gen2] rebuild error", err); } }, 0);
      } catch (err) { console.error("[gen2] onNodeCreated error", err); }
    };

    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      try { origOnConfigure?.apply(this, arguments); } catch (err) {}
      const self = this;
      setTimeout(() => { try { rebuildPanel(self, mode); } catch (err) { console.error("[gen2] onConfigure rebuild error", err); } }, 0);
    };

    // Output panel: refresh schema preview when PANEL_LINK connects/disconnects.
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
          const schema = message?.ui?.schema_json;
          if (schema && this.gen2SchemaWidget) this.gen2SchemaWidget.textarea.value = Array.isArray(schema) ? schema[0] : schema;
        } catch (err) {}
      };
    }
  },
});
