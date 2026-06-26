// Gen2 Input/Output Panel frontend extension.
//
// Adds a "Configure" button to Gen2_InputPanel and Gen2_OutputPanel. Clicking
// opens a popup dialog where the user defines parameter rows. The config is
// serialized into the node's _config widget (JSON) so it round-trips through
// save/load and API export.
//
// Parameter config entry:
//   { name, type, default, min, max, step, controlMode }
//   - min/max/step only for INT/FLOAT
//   - default can be null (no default)
//   - controlMode (INT only): fixed/randomize/increment/decrement
//
// Key behaviors:
//  - The per-parameter widgets on the node body always show the DEFAULT value,
//    and their serializeValue() returns the default. So exporting the workflow
//    / API always yields defaults, not whatever was set during a run. The live
//    runtime value is stored in a side widget "__val_<name>" which the backend
//    reads first (falling back to default).
//  - INT/FLOAT widgets use min/max/step for snapping.
//  - IMAGE params get an upload widget (file picker + thumbnail).
//  - The Output Panel shows a read-only JSON schema textbox (from PANEL_LINK).
//  - A "Reset to defaults" button clears live values back to defaults.
//
// Designed to work on both the legacy LiteGraph canvas renderer and the Nodes
// 2.0 (Vue) frontend. Uses onDrawForeground for button rendering (a standard
// LiteGraph hook) to avoid fragility around custom DOM widget APIs.

import { app } from "../../scripts/app.js";

const NODE_TYPES = {
  Gen2_InputPanel: { mode: "input" },
  Gen2_OutputPanel: { mode: "output" },
};
const MAX_PARAMS = 32;
const SUPPORTED_TYPES = ["STRING", "INT", "FLOAT", "BOOLEAN", "IMAGE"];
const NUMERIC_TYPES = ["INT", "FLOAT"];
const SLOT_PREFIX = "param_";
const PANEL_LINK_NAME = "PANEL_LINK";

// ---- Config (de)serialization ----
function parseConfig(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  try { const v = JSON.parse(raw); return Array.isArray(v) ? v : []; } catch (e) { return []; }
}

function serializeConfig(entries) {
  return JSON.stringify(entries.map((e) => {
    const o = { name: e.name || "", type: e.type || "STRING", default: e.default ?? null, min: e.min ?? null, max: e.max ?? null, step: e.step ?? null };
    if (e.type === "INT" && e.controlMode && e.controlMode !== "fixed") o.controlMode = e.controlMode;
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

// ---- Image upload ----
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

function filenameFromUpload(up) {
  return up.subfolder ? up.subfolder + "/" + up.name : up.name;
}

// ---- Slot management ----
function applyInputPanelSlots(node, entries) {
  const desired = [PANEL_LINK_NAME], displayNames = ["panel_link"], types = ["*"];
  for (let i = 0; i < entries.length; i++) { desired.push(SLOT_PREFIX + i); displayNames.push(entries[i].name || SLOT_PREFIX + i); types.push("*"); }
  rebuildOutputs(node, desired, displayNames, types);
}

function applyOutputPanelSlots(node, entries) {
  const desired = [PANEL_LINK_NAME], displayNames = ["panel_link"], types = ["*"];
  for (let i = 0; i < entries.length; i++) { desired.push(SLOT_PREFIX + i); displayNames.push(entries[i].name || SLOT_PREFIX + i); types.push("*"); }
  rebuildInputs(node, desired, displayNames, types);
}

function rebuildOutputs(node, names, displayNames, types) {
  if (!node.outputs) node.outputs = [];
  while (node.outputs.length > names.length) node.removeOutput(node.outputs.length - 1);
  while (node.outputs.length < names.length) node.addOutput(names[node.outputs.length], types[node.outputs.length] || "*");
  for (let i = 0; i < node.outputs.length; i++) { node.outputs[i].name = names[i]; node.outputs[i].type = types[i] || "*"; node.outputs[i].label = displayNames[i]; }
  if (typeof node.setSize === "function") node.setSize(node.computeSize());
}

function rebuildInputs(node, names, displayNames, types) {
  if (!node.inputs) node.inputs = [];
  while (node.inputs.length > names.length) node.removeInput(node.inputs.length - 1);
  while (node.inputs.length < names.length) node.addInput(names[node.inputs.length], types[node.inputs.length] || "*");
  for (let i = 0; i < node.inputs.length; i++) { node.inputs[i].name = names[i]; node.inputs[i].type = types[i] || "*"; node.inputs[i].label = displayNames[i]; }
  if (typeof node.setSize === "function") node.setSize(node.computeSize());
}

// ---- Per-parameter widgets ----
// We use node.addWidget (standard LiteGraph, universally available) for the
// parameter values, then override their serializeValue to return defaults.
// For IMAGE params and the Configure/Reset buttons, we use onDrawForeground
// to render HTML elements positioned over the node — a standard LiteGraph hook
// that doesn't depend on custom widget APIs.

function clearParamWidgets(node) {
  if (!node.gen2ParamWidgets) return;
  for (const pw of node.gen2ParamWidgets) {
    // Remove standard widgets from node.widgets
    if (pw.widget && node.widgets) {
      const idx = node.widgets.indexOf(pw.widget);
      if (idx >= 0) node.widgets.splice(idx, 1);
    }
    if (pw.liveWidget && node.widgets) {
      const idx = node.widgets.indexOf(pw.liveWidget);
      if (idx >= 0) node.widgets.splice(idx, 1);
    }
  }
  node.gen2ParamWidgets = [];
  node.gen2ControlHandlers = [];
}

function buildParamWidgets(node, entries) {
  clearParamWidgets(node);
  node.gen2ParamWidgets = [];
  for (let i = 0; i < entries.length; i++) {
    node.gen2ParamWidgets.push(makeParamWidget(node, i, entries[i]));
  }
  if (typeof node.setSize === "function") node.setSize(node.computeSize());
}

function makeParamWidget(node, paramIndex, entry) {
  let curVal = entry.default ?? null;
  let widget = null;

  if (entry.type === "INT") {
    widget = node.addWidget("number", entry.name, entry.default ?? 0, () => {}, { min: entry.min, max: entry.max, step: entry.step, precision: 0 });
    widget.serializeValue = () => entry.default ?? null;
    const origSetValue = widget.setValue;
    widget.setValue = function(v) { curVal = v; if (origSetValue) origSetValue.call(this, v); else this.value = v; };
    widget.getLiveValue = () => curVal;
    widget.setLiveValue = (v) => { curVal = v; widget.value = v; };

    // control_after_generate dropdown
    const ctrlWidget = node.addWidget("combo", entry.name + "_ctrl", entry.controlMode || "fixed", (v) => { entry.controlMode = v; persistConfig(node, paramIndex, entry); }, { values: ["fixed", "randomize", "increment", "decrement"] });
    ctrlWidget.serializeValue = () => undefined; // not serialized into prompt

    const applyControlMode = () => {
      const mode = ctrlWidget.value;
      if (mode === "fixed") return;
      let v = curVal; if (v == null || isNaN(v)) v = 0; v = parseInt(v, 10);
      const lo = entry.min != null ? entry.min : 0;
      const hi = entry.max != null ? entry.max : 0xffffffffffffffff;
      if (mode === "randomize") v = Math.floor(Math.random() * (hi - lo + 1)) + lo;
      else if (mode === "increment") { v = v + 1; if (v > hi) v = lo; }
      else if (mode === "decrement") { v = v - 1; if (v < lo) v = hi; }
      widget.setLiveValue(v);
    };
    node.gen2ControlHandlers = node.gen2ControlHandlers || [];
    node.gen2ControlHandlers.push(applyControlMode);

  } else if (entry.type === "FLOAT") {
    widget = node.addWidget("number", entry.name, entry.default ?? 0.0, () => {}, { min: entry.min, max: entry.max, step: entry.step });
    widget.serializeValue = () => entry.default ?? null;
    widget.getLiveValue = () => widget.value;
    widget.setLiveValue = (v) => { widget.value = v; };

  } else if (entry.type === "BOOLEAN") {
    widget = node.addWidget("toggle", entry.name, !!entry.default, () => {});
    widget.serializeValue = () => entry.default ?? null;
    widget.getLiveValue = () => widget.value;
    widget.setLiveValue = (v) => { widget.value = !!v; };

  } else if (entry.type === "STRING") {
    widget = node.addWidget("text", entry.name, entry.default ?? "", () => {});
    widget.serializeValue = () => entry.default ?? null;
    widget.getLiveValue = () => widget.value;
    widget.setLiveValue = (v) => { widget.value = v ?? ""; };

  } else if (entry.type === "IMAGE") {
    // For IMAGE, we store the filename as a hidden string widget. The upload
    // UI is handled via onDrawForeground (see makeImageOverlay).
    widget = node.addWidget("text", entry.name, entry.default ?? "", () => {});
    widget.serializeValue = () => entry.default ?? null;
    widget.getLiveValue = () => widget.value;
    widget.setLiveValue = (v) => { widget.value = v ?? ""; };
    // Mark it so onDrawForeground knows to draw an upload button for it.
    widget.__gen2Image = true;
    widget.__gen2Entry = entry;
    widget.__gen2ParamIndex = paramIndex;
  }

  // Hidden side widget for the live value under __val_<name>
  const liveName = "__val_" + entry.name;
  const liveWidget = node.addWidget("text", liveName, entry.default ?? null, () => {});
  liveWidget.serializeValue = () => widget.getLiveValue();
  liveWidget.computeSize = () => [0, 0];
  // Hide it from the node body by making it not render (ComfyUI skips zero-size)

  return { widget, liveWidget, paramIndex, entry };
}

function persistConfig(node, paramIndex, entry) {
  const w = getConfigWidget(node);
  if (!w) return;
  const all = parseConfig(w.value || "[]");
  if (all[paramIndex]) all[paramIndex] = Object.assign({}, all[paramIndex], entry);
  w.value = serializeConfig(all);
}

// ---- onDrawForeground: renders Configure/Reset buttons + IMAGE upload overlays ----
// This is a standard LiteGraph hook called every frame when the node is visible.
// We draw HTML elements positioned over the node. This avoids relying on custom
// DOM widget APIs which vary across ComfyUI versions.

function ensureOverlay(node) {
  if (node.__gen2Overlay) return node.__gen2Overlay;
  const overlay = document.createElement("div");
  overlay.style.position = "absolute";
  overlay.style.pointerEvents = "none";
  overlay.style.zIndex = "1000";
  document.body.appendChild(overlay);
  node.__gen2Overlay = overlay;

  // Configure button
  const cfgBtn = document.createElement("button");
  cfgBtn.textContent = "Configure";
  cfgBtn.style.pointerEvents = "auto";
  cfgBtn.style.cursor = "pointer";
  cfgBtn.style.padding = "3px 10px";
  cfgBtn.style.fontSize = "12px";
  cfgBtn.style.borderRadius = "4px";
  cfgBtn.style.background = "var(--comfy-input-bg, #333)";
  cfgBtn.style.color = "var(--fg-color, #fff)";
  cfgBtn.style.border = "1px solid var(--border-color, #555)";
  cfgBtn.style.display = "block";
  cfgBtn.style.marginBottom = "4px";
  cfgBtn.addEventListener("click", (e) => {
    e.preventDefault(); e.stopPropagation();
    const mode = NODE_TYPES[node.comfyClass]?.mode || "input";
    openConfigDialog(node, mode);
  });
  overlay.appendChild(cfgBtn);
  node.__gen2CfgBtn = cfgBtn;

  // Reset button (input panel only)
  const mode = NODE_TYPES[node.comfyClass]?.mode;
  if (mode === "input") {
    const resetBtn = document.createElement("button");
    resetBtn.textContent = "Reset to defaults";
    resetBtn.style.pointerEvents = "auto";
    resetBtn.style.cursor = "pointer";
    resetBtn.style.padding = "3px 10px";
    resetBtn.style.fontSize = "11px";
    resetBtn.style.borderRadius = "4px";
    resetBtn.style.background = "var(--comfy-input-bg, #333)";
    resetBtn.style.color = "var(--fg-color, #fff)";
    resetBtn.style.border = "1px solid var(--border-color, #555)";
    resetBtn.style.display = "block";
    resetBtn.style.marginBottom = "4px";
    resetBtn.addEventListener("click", (e) => {
      e.preventDefault(); e.stopPropagation();
      if (!node.gen2ParamWidgets) return;
      for (const pw of node.gen2ParamWidgets) {
        pw.widget.setLiveValue(pw.entry.default ?? null);
      }
      if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
      resetBtn.textContent = "Reset ✓";
      setTimeout(() => { resetBtn.textContent = "Reset to defaults"; }, 1000);
    });
    overlay.appendChild(resetBtn);
    node.__gen2ResetBtn = resetBtn;
  }

  // Schema textbox (output panel only)
  if (mode === "output") {
    const lbl = document.createElement("div");
    lbl.textContent = "JSON schema";
    lbl.style.fontSize = "11px"; lbl.style.opacity = "0.7"; lbl.style.marginBottom = "2px";
    lbl.style.pointerEvents = "none";
    overlay.appendChild(lbl);

    const ta = document.createElement("textarea");
    ta.readOnly = true; ta.rows = 8;
    ta.style.width = "300px"; ta.style.fontSize = "11px"; ta.style.fontFamily = "monospace";
    ta.style.background = "var(--comfy-input-bg, #1a1a1a)"; ta.style.color = "var(--fg-color, #ddd)";
    ta.style.border = "1px solid var(--border-color, #444)"; ta.style.borderRadius = "3px";
    ta.style.resize = "vertical"; ta.style.pointerEvents = "auto"; ta.style.display = "block";
    ta.placeholder = "Connect an Input Panel's PANEL_LINK and run to see the schema.";
    ta.addEventListener("focus", () => ta.select());
    overlay.appendChild(ta);
    node.__gen2SchemaTa = ta;

    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy"; copyBtn.style.pointerEvents = "auto"; copyBtn.style.cursor = "pointer";
    copyBtn.style.fontSize = "11px"; copyBtn.style.padding = "2px 8px"; copyBtn.style.marginTop = "2px";
    copyBtn.addEventListener("click", (e) => {
      e.preventDefault(); e.stopPropagation(); ta.select();
      try { document.execCommand("copy"); copyBtn.textContent = "Copied!"; setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200); } catch (err) {}
    });
    overlay.appendChild(copyBtn);
  }

  // IMAGE upload buttons container
  node.__gen2ImageBtns = {};
  return overlay;
}

function positionOverlay(node, ctx) {
  const overlay = node.__gen2Overlay;
  if (!overlay) return;
  // Position the overlay at the node's top-left, offset below the title bar.
  const pos = node.pos;
  const scale = ctx.canvas?.style?.zoom ? parseFloat(ctx.canvas.style.zoom) : 1;
  // Use the node's absolute position on the canvas (LiteGraph keeps pos in canvas coords)
  // The canvas element's transform handles the rest; we use absolute screen pos via getBoundingClientRect
  const canvas = ctx.canvas;
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  // Convert canvas-space to screen-space using the canvas DS (delta scale)
  const ds = canvas.style.transform ? null : (window.LiteGraph?.Canvas?.current_ds || app.canvas?.ds);
  let offsetX = pos[0], offsetY = pos[1];
  if (ds) { offsetX = offsetX * ds.scale + ds.offset[0]; offsetY = offsetY * ds.scale + ds.offset[1]; }
  else { offsetX = pos[0]; offsetY = pos[1]; }
  overlay.style.left = (rect.left + offsetX + 8) + "px";
  overlay.style.top = (rect.top + offsetY + 30) + "px";
}

function refreshImageButtons(node) {
  if (!node.gen2ParamWidgets) return;
  for (const pw of node.gen2ParamWidgets) {
    if (!pw.widget.__gen2Image) continue;
    const name = pw.entry.name;
    let btn = node.__gen2ImageBtns[name];
    if (!btn) {
      btn = document.createElement("button");
      btn.textContent = "Upload " + name;
      btn.style.pointerEvents = "auto"; btn.style.cursor = "pointer";
      btn.style.padding = "2px 8px"; btn.style.fontSize = "11px";
      btn.style.borderRadius = "3px"; btn.style.marginBottom = "2px"; btn.style.display = "block";
      btn.style.background = "var(--comfy-input-bg, #333)"; btn.style.color = "var(--fg-color, #fff)";
      btn.style.border = "1px solid var(--border-color, #555)";
      const fileInput = document.createElement("input");
      fileInput.type = "file"; fileInput.accept = "image/*"; fileInput.style.display = "none";
      fileInput.addEventListener("change", async () => {
        const file = fileInput.files && fileInput.files[0]; if (!file) return;
        btn.textContent = "...";
        try {
          const up = await uploadImage(file);
          const fname = filenameFromUpload(up);
          pw.entry.default = fname;
          pw.widget.setLiveValue(fname);
          persistConfig(node, pw.paramIndex, pw.entry);
        } catch (err) { btn.textContent = "failed"; }
        finally { btn.textContent = "Upload " + name; }
      });
      btn.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); fileInput.click(); });
      btn.appendChild(fileInput);
      node.__gen2Overlay.appendChild(btn);
      node.__gen2ImageBtns[name] = btn;
    }
  }
}

function refreshSchemaBox(node) {
  if (!node.__gen2SchemaTa) return;
  const entries = parseConfig(getConfigWidget(node)?.value || "[]");
  const out = entries.map((e) => {
    const o = { name: e.name, type: e.type, default: e.default ?? null };
    if (NUMERIC_TYPES.includes(e.type)) { o.min = e.min ?? null; o.max = e.max ?? null; o.step = e.step ?? null; }
    if (e.type === "INT" && e.controlMode && e.controlMode !== "fixed") o.controlMode = e.controlMode;
    return o;
  });
  node.__gen2SchemaTa.value = JSON.stringify(out, null, 2);
}

// ---- Configure popup dialog ----
function openConfigDialog(node, mode) {
  const entries = parseConfig(getConfigWidget(node)?.value || "[]");
  const draft = entries.map((e) => ({ ...e }));

  const dlg = document.createElement("dialog");
  dlg.style.minWidth = "640px";
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
    ? "Each parameter becomes a typed output slot. Its Name is the API-export key. INT/FLOAT accept min/max/step. Default can be empty (null). IMAGE params get an upload button. Export always yields defaults, not runtime values."
    : "Each parameter becomes a typed input slot. IMAGE inputs are saved to the output folder and their URL returned via /history.";
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
      for (const t of SUPPORTED_TYPES) { const o = document.createElement("option"); o.value = t; o.textContent = t; if (entry.type === t) o.selected = true; typeSel.appendChild(o); }
      typeSel.onchange = () => { entry.type = typeSel.value; renderRows(); };
      typeSel.style.width = "90px";

      const nameIn = document.createElement("input");
      nameIn.type = "text"; nameIn.value = entry.name || ""; nameIn.placeholder = "param name (API key)";
      nameIn.style.flex = "1"; nameIn.style.minWidth = "120px";
      nameIn.oninput = () => { entry.name = nameIn.value; };

      row.appendChild(typeSel); row.appendChild(nameIn);

      // Default cell
      if (mode === "input" && entry.type === "IMAGE") {
        const pickBtn = document.createElement("button");
        pickBtn.textContent = "Upload"; pickBtn.style.fontSize = "11px"; pickBtn.style.padding = "2px 8px";
        const fileInput = document.createElement("input");
        fileInput.type = "file"; fileInput.accept = "image/*"; fileInput.style.display = "none";
        pickBtn.onclick = (e) => { e.preventDefault(); fileInput.click(); };
        const fnameLbl = document.createElement("span");
        fnameLbl.style.fontSize = "10px"; fnameLbl.style.opacity = "0.7"; fnameLbl.style.maxWidth = "70px";
        fnameLbl.style.overflow = "hidden"; fnameLbl.style.textOverflow = "ellipsis"; fnameLbl.style.whiteSpace = "nowrap";
        function refresh() { const cur = entry.default; fnameLbl.textContent = (typeof cur === "string" && cur) ? cur.split("/").pop() : ""; fnameLbl.title = cur || ""; }
        refresh();
        fileInput.onchange = async () => { const file = fileInput.files && fileInput.files[0]; if (!file) return; pickBtn.textContent = "..."; try { const up = await uploadImage(file); entry.default = filenameFromUpload(up); refresh(); } catch (err) { fnameLbl.textContent = "failed"; } finally { pickBtn.textContent = "Upload"; } };
        row.appendChild(pickBtn); row.appendChild(fileInput); row.appendChild(fnameLbl);
      } else {
        const defIn = document.createElement("input");
        defIn.type = NUMERIC_TYPES.includes(entry.type) ? "number" : "text";
        defIn.value = entry.default ?? ""; defIn.placeholder = "default (empty = null)"; defIn.style.width = "110px";
        defIn.disabled = mode === "output";
        defIn.oninput = () => { const v = defIn.value; if (v === "") { entry.default = null; return; } if (entry.type === "INT") entry.default = parseInt(v, 10); else if (entry.type === "FLOAT") entry.default = parseFloat(v); else if (entry.type === "BOOLEAN") entry.default = v.toLowerCase() === "true" || v === "1"; else entry.default = v; };
        row.appendChild(defIn);
      }

      // Range/step for INT/FLOAT
      if (NUMERIC_TYPES.includes(entry.type)) {
        const mkNum = (key, ph) => { const inp = document.createElement("input"); inp.type = "number"; inp.value = entry[key] != null ? entry[key] : ""; inp.placeholder = ph; inp.style.width = "60px"; inp.disabled = mode === "output"; inp.oninput = () => { const v = inp.value; if (v === "") { entry[key] = null; return; } entry[key] = entry.type === "INT" ? parseInt(v, 10) : parseFloat(v); }; return inp; };
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
  addBtn.onclick = () => { if (draft.length >= MAX_PARAMS) return; draft.push({ name: "", type: "STRING", default: null, min: null, max: null, step: null }); renderRows(); updateCount(); };

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
    for (const e of draft) { const n = (e.name || "").trim(); if (!n || seen.has(n)) continue; seen.add(n); const c = { name: n, type: e.type, default: e.default ?? null, min: e.min ?? null, max: e.max ?? null, step: e.step ?? null }; if (e.type === "INT" && e.controlMode && e.controlMode !== "fixed") c.controlMode = e.controlMode; clean.push(c); }
    setConfig(node, clean);
    if (mode === "input") { applyInputPanelSlots(node, clean); buildParamWidgets(node, clean); }
    else { applyOutputPanelSlots(node, clean); refreshSchemaBox(node); }
    if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
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
          const graph = app.graph;
          if (!graph || !graph.nodes) return;
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
        ensureOverlay(this);
        // Hide the raw _config text widget if present
        const cw = getConfigWidget(this);
        if (cw) { cw.type = "hidden"; cw.computeSize = () => [0, -4]; }
        // Defer slot/widget build until widgets exist
        const self = this;
        setTimeout(() => {
          try {
            const w = getConfigWidget(self);
            if (w) {
              const entries = parseConfig(w.value || "[]");
              if (mode === "input") { applyInputPanelSlots(self, entries); buildParamWidgets(self, entries); refreshImageButtons(self); }
              else { applyOutputPanelSlots(self, entries); refreshSchemaBox(self); }
            }
          } catch (err) { console.error("[gen2] deferred build error", err); }
        }, 50);
      } catch (err) { console.error("[gen2] onNodeCreated error", err); }
    };

    // onDrawForeground: position the overlay each frame
    const origOnDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      try { origOnDrawForeground?.apply(this, arguments); } catch (err) {}
      try {
        if (!this.__gen2Overlay) return;
        if (this.flags?.collapsed) { this.__gen2Overlay.style.display = "none"; return; }
        this.__gen2Overlay.style.display = "block";
        positionOverlay(this, ctx);
        if (mode === "input") refreshImageButtons(this);
      } catch (err) {}
    };

    // Clean up overlay on removal
    const origOnRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      try { if (this.__gen2Overlay) { this.__gen2Overlay.remove(); this.__gen2Overlay = null; } } catch (err) {}
      try { clearParamWidgets(this); } catch (err) {}
      try { origOnRemoved?.apply(this, arguments); } catch (err) {}
    };

    // Output panel: update schema textbox after execution
    if (mode === "output") {
      const origOnExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        try { origOnExecuted?.apply(this, arguments); } catch (err) {}
        try {
          const schema = message?.ui?.schema_json;
          if (schema && this.__gen2SchemaTa) {
            this.__gen2SchemaTa.value = Array.isArray(schema) ? schema[0] : schema;
          }
        } catch (err) {}
      };
    }
  },
});
