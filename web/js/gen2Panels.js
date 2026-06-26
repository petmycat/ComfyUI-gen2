// Gen2 Input/Output Panel frontend extension.
//
// Adds a "Configure" button to Gen2_InputPanel and Gen2_OutputPanel. Clicking
// opens a popup dialog where the user defines parameter rows. The config is
// serialized into the node's _config widget (JSON) so it round-trips through
// save/load and API export.
//
// Parameter config entry:
//   { name, type, default, min, max, step }
//   - min/max/step only for INT/FLOAT
//   - default can be null (no default)
//
// Key behaviors:
//  - The per-parameter widgets on the node body always show the DEFAULT value,
//    and their serializeValue() returns the default. So exporting the workflow
//    / API always yields defaults, not whatever was set during a run. The live
//    runtime value is written to a side widget "__val_<name>" which the backend
//    reads first (falling back to default).
//  - INT/FLOAT widgets use min/max/step for snapping.
//  - IMAGE params get an upload widget (file picker + thumbnail).
//  - The Output Panel shows a read-only JSON schema textbox (from PANEL_LINK).
//
// Designed to work on both the legacy LiteGraph canvas renderer and the Nodes
// 2.0 (Vue) frontend.

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

// ---- Config (de)serialization, shared with the Python backend ----
function parseConfig(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  try {
    const v = JSON.parse(raw);
    return Array.isArray(v) ? v : [];
  } catch (e) {
    return [];
  }
}

function serializeConfig(entries) {
  return JSON.stringify(entries.map((e) => {
    const o = {
      name: e.name || "",
      type: e.type || "STRING",
      default: e.default ?? null,
      min: e.min ?? null,
      max: e.max ?? null,
      step: e.step ?? null,
    };
    if (e.type === "INT" && e.controlMode && e.controlMode !== "fixed") {
      o.controlMode = e.controlMode;
    }
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

// ---- Image upload helpers (use ComfyUI's built-in endpoints) ----
async function uploadImage(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("type", "input");
  body.append("overwrite", "true");
  const resp = await fetch("/upload/image", { method: "POST", body });
  if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
  return resp.json();
}

function viewUrl(name, subfolder, type) {
  const p = new URLSearchParams({ filename: name, type: type || "input" });
  if (subfolder) p.set("subfolder", subfolder);
  return `/view?${p.toString()}`;
}

function filenameFromUpload(up) {
  return up.subfolder ? `${up.subfolder}/${up.name}` : up.name;
}

// ---- Slot management ----
function applyInputPanelSlots(node, entries) {
  const desired = [PANEL_LINK_NAME];
  const displayNames = ["panel_link"];
  const types = ["*"];
  for (let i = 0; i < entries.length; i++) {
    desired.push(SLOT_PREFIX + i);
    displayNames.push(entries[i].name || SLOT_PREFIX + i);
    types.push("*");
  }
  rebuildOutputs(node, desired, displayNames, types);
}

function applyOutputPanelSlots(node, entries) {
  const desired = [PANEL_LINK_NAME];
  const displayNames = ["panel_link"];
  const types = ["*"];
  for (let i = 0; i < entries.length; i++) {
    desired.push(SLOT_PREFIX + i);
    displayNames.push(entries[i].name || SLOT_PREFIX + i);
    types.push("*");
  }
  rebuildInputs(node, desired, displayNames, types);
}

function rebuildOutputs(node, names, displayNames, types) {
  if (!node.outputs) node.outputs = [];
  while (node.outputs.length > names.length) node.removeOutput(node.outputs.length - 1);
  while (node.outputs.length < names.length) node.addOutput(names[node.outputs.length], types[node.outputs.length] || "*");
  for (let i = 0; i < node.outputs.length; i++) {
    node.outputs[i].name = names[i];
    node.outputs[i].type = types[i] || "*";
    node.outputs[i].label = displayNames[i];
  }
  if (typeof node.setSize === "function") node.setSize(node.computeSize());
}

function rebuildInputs(node, names, displayNames, types) {
  if (!node.inputs) node.inputs = [];
  while (node.inputs.length > names.length) node.removeInput(node.inputs.length - 1);
  while (node.inputs.length < names.length) node.addInput(names[node.inputs.length], types[node.inputs.length] || "*");
  for (let i = 0; i < node.inputs.length; i++) {
    node.inputs[i].name = names[i];
    node.inputs[i].type = types[i] || "*";
    node.inputs[i].label = displayNames[i];
  }
  if (typeof node.setSize === "function") node.setSize(node.computeSize());
}

// ---- Widget removal helper ----
// addCustomWidget pushes into node.widgets; to reconfigure cleanly we must
// also splice the widget objects out of that array, or stale draw() callbacks
// keep firing and re-append removed DOM elements. We also mark widgets dead so
// their draw() short-circuits even if a stale reference lingers for a frame.
function removeWidgetFromNode(node, widget) {
  widget.__dead = true;
  if (widget.element && widget.element.parentNode) {
    widget.element.parentNode.removeChild(widget.element);
  }
  if (node.widgets) {
    const idx = node.widgets.indexOf(widget);
    if (idx >= 0) node.widgets.splice(idx, 1);
  }
  // Some ComfyUI versions keep DOM widgets on a separate list.
  if (node.domWidgets) {
    const idx2 = node.domWidgets.indexOf(widget);
    if (idx2 >= 0) node.domWidgets.splice(idx2, 1);
  }
}

// ---- Per-parameter widgets on the node body (Input Panel) ----
// For each configured param we add a widget showing its DEFAULT value. The
// widget's serializeValue() returns the default, so export always yields
// defaults. The live runtime value is stored in a side channel which the
// backend reads first.
function clearParamWidgets(node) {
  if (!node.gen2ParamWidgets) return;
  for (const pw of node.gen2ParamWidgets) {
    removeWidgetFromNode(node, pw.widget);
    if (pw.liveWidget) removeWidgetFromNode(node, pw.liveWidget);
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
  const wrapper = document.createElement("div");
  wrapper.style.display = "flex";
  wrapper.style.alignItems = "center";
  wrapper.style.gap = "6px";
  wrapper.style.padding = "2px 4px";

  const label = document.createElement("span");
  label.textContent = entry.name;
  label.style.fontSize = "12px";
  label.style.opacity = "0.85";
  label.style.minWidth = "70px";
  wrapper.appendChild(label);

  let valueEl;   // the editable control showing the live value
  let curVal = entry.default ?? "";  // live runtime value (initialized to default)

  if (entry.type === "IMAGE") {
    // Upload widget: button + thumbnail + filename. The "value" is the filename.
    const fileInput = document.createElement("input");
    fileInput.type = "file"; fileInput.accept = "image/*"; fileInput.style.display = "none";
    const pickBtn = document.createElement("button");
    pickBtn.textContent = "Choose image";
    pickBtn.style.fontSize = "11px"; pickBtn.style.padding = "2px 8px"; pickBtn.style.cursor = "pointer";
    pickBtn.onclick = (e) => { e.preventDefault(); e.stopPropagation(); fileInput.click(); };
    const thumb = document.createElement("img");
    Object.assign(thumb.style, { maxHeight: "32px", maxWidth: "32px", objectFit: "cover", borderRadius: "3px", display: "none" });
    const fnameLbl = document.createElement("span");
    Object.assign(fnameLbl.style, { fontSize: "10px", opacity: "0.7", maxWidth: "90px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" });

    function refresh() {
      const cur = entry.default;
      if (typeof cur === "string" && cur) {
        thumb.src = viewUrl(cur, "", "input"); thumb.style.display = "block";
        fnameLbl.textContent = cur.split("/").pop(); fnameLbl.title = cur;
      } else { thumb.style.display = "none"; fnameLbl.textContent = "no image"; }
    }
    refresh();

    fileInput.onchange = async () => {
      const file = fileInput.files && fileInput.files[0];
      if (!file) return;
      pickBtn.textContent = "...";
      try {
        const up = await uploadImage(file);
        const fname = filenameFromUpload(up);
        entry.default = fname;  // upload sets the DEFAULT (persisted)
        curVal = fname;          // live value too
        persistConfig(node, paramIndex, entry);
        refresh();
      } catch (err) { fnameLbl.textContent = "failed"; }
      finally { pickBtn.textContent = "Choose image"; }
    };
    valueEl = { get: () => curVal, set: (v) => { curVal = v; }, fileInput, pickBtn, thumb, fnameLbl, refresh };
    wrapper.appendChild(pickBtn); wrapper.appendChild(fileInput); wrapper.appendChild(thumb); wrapper.appendChild(fnameLbl);

  } else if (entry.type === "BOOLEAN") {
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !!entry.default;
    cb.onchange = () => { curVal = cb.checked; };
    valueEl = { get: () => curVal, set: (v) => { cb.checked = !!v; curVal = !!v; }, checkbox: cb };
    wrapper.appendChild(cb);

  } else {
    // INT / FLOAT / STRING: a text/number input.
    const inp = document.createElement("input");
    inp.type = NUMERIC_TYPES.includes(entry.type) ? "number" : "text";
    if (NUMERIC_TYPES.includes(entry.type)) {
      if (entry.min != null) inp.min = entry.min;
      if (entry.max != null) inp.max = entry.max;
      if (entry.step != null) inp.step = entry.step;
    }
    inp.value = entry.default ?? "";
    inp.style.flex = "1";
    inp.oninput = () => {
      if (entry.type === "INT") curVal = parseInt(inp.value, 10); if (isNaN(curVal)) curVal = null;
      if (entry.type === "FLOAT") curVal = parseFloat(inp.value); if (isNaN(curVal)) curVal = null;
      if (entry.type === "STRING") curVal = inp.value;
    };
    valueEl = { get: () => curVal, set: (v) => { inp.value = v ?? ""; curVal = v; }, input: inp };
    wrapper.appendChild(inp);

    // control_after_generate: for INT params, a dropdown that auto-changes
    // the seed after each run (fixed / randomize / increment / decrement),
    // mirroring ComfyUI core's built-in seed control.
    if (entry.type === "INT") {
      const ctrlSel = document.createElement("select");
      for (const m of ["fixed", "randomize", "increment", "decrement"]) {
        const o = document.createElement("option");
        o.value = m; o.textContent = m;
        if ((entry.controlMode || "fixed") === m) o.selected = true;
        ctrlSel.appendChild(o);
      }
      ctrlSel.style.fontSize = "11px";
      ctrlSel.style.width = "90px";
      ctrlSel.onchange = () => {
        entry.controlMode = ctrlSel.value;
        persistConfig(node, paramIndex, entry);
      };
      wrapper.appendChild(ctrlSel);

      // Apply the control mode after a generation completes. We hook into the
      // "executed" API event (fires per-node after execute). The handler reads
      // the current live value, transforms it, and writes it back.
      const applyControlMode = () => {
        const mode = ctrlSel.value;
        if (mode === "fixed") return;
        let v = valueEl.get();
        if (v == null || isNaN(v)) v = 0;
        v = parseInt(v, 10);
        const lo = entry.min != null ? entry.min : 0;
        const hi = entry.max != null ? entry.max : 0xffffffffffffffff;
        if (mode === "randomize") {
          v = Math.floor(Math.random() * (hi - lo + 1)) + lo;
        } else if (mode === "increment") {
          v = v + 1;
          if (v > hi) v = lo;
        } else if (mode === "decrement") {
          v = v - 1;
          if (v < lo) v = hi;
        }
        valueEl.set(v);
      };
      // Store the handler so we can wire/unwire it on reconfigure.
      node.gen2ControlHandlers = node.gen2ControlHandlers || [];
      node.gen2ControlHandlers.push(applyControlMode);
    }
  }

  // The widget object: serializeValue returns the DEFAULT so export yields
  // defaults; the live value is surfaced via the __val_<name> side channel.
  const widget = {
    type: "gen2_param",
    name: `param_${paramIndex}`,
    element: wrapper,
    draw(ctx, n, wWidth, y) {
      if (widget.__dead) return;
      if (!wrapper.parentNode) document.body.appendChild(wrapper);
      Object.assign(wrapper.style, {
        position: "absolute",
        left: (n.pos[0] + 8) + "px",
        top: (n.pos[1] + y) + "px",
        width: (wWidth - 16) + "px",
      });
    },
    computeSize() { return [260, 40]; },
    // Export path: always the default (not the live value).
    serializeValue() { return entry.default ?? null; },
    // Live runtime value, written into the __val_<name> side channel.
    getLiveValue() { return valueEl.get(); },
    setLiveValue(v) { valueEl.set(v); },
  };
  if (typeof node.addCustomWidget === "function") node.addCustomWidget(widget);
  else { node.customWidgets = node.customWidgets || []; node.customWidgets.push(widget); }

  // Hidden side widget that serializes the LIVE value under "__val_<name>".
  // The backend reads this first (falling back to default). It has no DOM
  // element and zero size, so it's invisible on the node. It IS serialized
  // into the prompt/export, so execution gets the live value even though the
  // main param widget above serializes the default.
  const liveName = `__val_${entry.name}`;
  const liveWidget = {
    type: "gen2_live_value",
    name: liveName,
    // No element / draw → invisible.
    draw() {},
    computeSize() { return [0, 0]; },
    serializeValue() { return valueEl.get(); },
  };
  if (typeof node.addCustomWidget === "function") node.addCustomWidget(liveWidget);
  else { node.customWidgets = node.customWidgets || []; node.customWidgets.push(liveWidget); }

  return { wrapper, widget, liveWidget, paramIndex, entry, valueEl };
}

function persistConfig(node, paramIndex, entry) {
  const w = getConfigWidget(node);
  if (!w) return;
  const all = parseConfig(w.value || "[]");
  if (all[paramIndex]) all[paramIndex] = { ...all[paramIndex], ...entry };
  w.value = serializeConfig(all);
}

// ---- JSON schema textbox on the Output Panel ----
function clearSchemaBox(node) {
  if (node.gen2SchemaBox) {
    removeWidgetFromNode(node, node.gen2SchemaBox.widget);
    node.gen2SchemaBox = null;
  }
}

function buildSchemaBox(node) {
  clearSchemaBox(node);
  const wrapper = document.createElement("div");
  wrapper.style.padding = "4px";

  const lbl = document.createElement("div");
  lbl.textContent = "JSON schema";
  lbl.style.fontSize = "11px"; lbl.style.opacity = "0.7"; lbl.style.marginBottom = "2px";
  wrapper.appendChild(lbl);

  const ta = document.createElement("textarea");
  ta.readOnly = true;
  ta.rows = 8;
  ta.style.width = "100%";
  ta.style.fontSize = "11px";
  ta.style.fontFamily = "monospace";
  ta.style.background = "var(--comfy-input-bg, #1a1a1a)";
  ta.style.color = "var(--fg-color, #ddd)";
  ta.style.border = "1px solid var(--border-color, #444)";
  ta.style.borderRadius = "3px";
  ta.style.resize = "vertical";
  ta.placeholder = "Connect an Input Panel's PANEL_LINK and run to see the schema.";
  // Click-to-select-all for easy copy.
  ta.addEventListener("focus", () => ta.select());
  wrapper.appendChild(ta);

  const copyBtn = document.createElement("button");
  copyBtn.textContent = "Copy";
  copyBtn.style.fontSize = "11px"; copyBtn.style.padding = "2px 8px"; copyBtn.style.marginTop = "2px"; copyBtn.style.cursor = "pointer";
  copyBtn.onclick = (e) => {
    e.preventDefault(); e.stopPropagation();
    ta.select();
    document.execCommand("copy");
    copyBtn.textContent = "Copied!";
    setTimeout(() => { copyBtn.textContent = "Copy"; }, 1200);
  };
  wrapper.appendChild(copyBtn);

  const widget = {
    type: "gen2_schema_box",
    name: "__schema",
    element: wrapper,
    draw(ctx, n, wWidth, y) {
      if (widget.__dead) return;
      if (!wrapper.parentNode) document.body.appendChild(wrapper);
      Object.assign(wrapper.style, {
        position: "absolute",
        left: (n.pos[0] + 8) + "px",
        top: (n.pos[1] + y) + "px",
        width: (wWidth - 16) + "px",
      });
    },
    computeSize() { return [320, 200]; },
    serializeValue() { return undefined; },
  };
  if (typeof node.addCustomWidget === "function") node.addCustomWidget(widget);
  else { node.customWidgets = node.customWidgets || []; node.customWidgets.push(widget); }
  node.gen2SchemaBox = { wrapper, widget, textarea: ta };
}

// Update the schema textbox from the node's own _config (preview before run).
function refreshSchemaBoxFromConfig(node) {
  if (!node.gen2SchemaBox) return;
  const entries = parseConfig(getConfigWidget(node)?.value || "[]");
  node.gen2SchemaBox.textarea.value = buildSchemaJsonLocal(entries);
}

function buildSchemaJsonLocal(entries) {
  const out = entries.map((e) => {
    const o = { name: e.name, type: e.type, default: e.default ?? null };
    if (NUMERIC_TYPES.includes(e.type)) {
      o.min = e.min ?? null; o.max = e.max ?? null; o.step = e.step ?? null;
    }
    if (e.type === "INT" && e.controlMode && e.controlMode !== "fixed") {
      o.controlMode = e.controlMode;
    }
    return o;
  });
  return JSON.stringify(out, null, 2);
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
  dlg.style.padding = "16px";
  dlg.style.borderRadius = "8px";

  const title = document.createElement("h3");
  title.textContent = mode === "input" ? "Configure Input Panel" : "Configure Output Panel";
  title.style.margin = "0 0 12px 0";
  dlg.appendChild(title);

  const help = document.createElement("p");
  help.textContent = mode === "input"
    ? "Each parameter becomes a typed output slot. Its Name is the API-export key. INT/FLOAT accept min/max/step. Default can be empty (null). IMAGE params get an upload widget on the node. Export always yields defaults, not runtime values."
    : "Each parameter becomes a typed input slot. IMAGE inputs are saved to the output folder and their URL returned via /history. A JSON schema of the parameters is shown on the node.";
  help.style.opacity = "0.8";
  help.style.fontSize = "12px";
  help.style.margin = "0 0 12px 0";
  dlg.appendChild(help);

  const rowsEl = document.createElement("div");
  rowsEl.style.display = "flex";
  rowsEl.style.flexDirection = "column";
  rowsEl.style.gap = "8px";
  dlg.appendChild(rowsEl);

  function renderRows() {
    rowsEl.innerHTML = "";
    draft.forEach((entry, idx) => {
      const row = document.createElement("div");
      row.style.display = "flex";
      row.style.gap = "6px";
      row.style.alignItems = "center";
      row.style.flexWrap = "wrap";

      const typeSel = document.createElement("select");
      for (const t of SUPPORTED_TYPES) {
        const o = document.createElement("option");
        o.value = t; o.textContent = t;
        if (entry.type === t) o.selected = true;
        typeSel.appendChild(o);
      }
      typeSel.onchange = () => { entry.type = typeSel.value; renderRows(); };
      typeSel.style.width = "90px";

      const nameIn = document.createElement("input");
      nameIn.type = "text";
      nameIn.value = entry.name || "";
      nameIn.placeholder = "param name (API key)";
      nameIn.style.flex = "1";
      nameIn.style.minWidth = "120px";
      nameIn.oninput = () => { entry.name = nameIn.value; };

      row.appendChild(typeSel);
      row.appendChild(nameIn);

      // Default cell
      const defCell = document.createElement("div");
      defCell.style.display = "flex";
      defCell.style.alignItems = "center";
      defCell.style.gap = "4px";

      if (mode === "input" && entry.type === "IMAGE") {
        const fileInput = document.createElement("input");
        fileInput.type = "file"; fileInput.accept = "image/*"; fileInput.style.display = "none";
        const pickBtn = document.createElement("button");
        pickBtn.textContent = "Upload"; pickBtn.style.fontSize = "11px"; pickBtn.style.padding = "2px 8px";
        pickBtn.onclick = (e) => { e.preventDefault(); fileInput.click(); };
        const thumb = document.createElement("img");
        Object.assign(thumb.style, { maxHeight: "28px", maxWidth: "28px", objectFit: "cover", borderRadius: "3px", display: "none" });
        const fnameLbl = document.createElement("span");
        Object.assign(fnameLbl.style, { fontSize: "10px", opacity: "0.7", maxWidth: "70px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" });
        function refresh() {
          const cur = entry.default;
          if (typeof cur === "string" && cur) {
            thumb.src = viewUrl(cur, "", "input"); thumb.style.display = "block";
            fnameLbl.textContent = cur.split("/").pop(); fnameLbl.title = cur;
          } else { thumb.style.display = "none"; fnameLbl.textContent = ""; }
        }
        refresh();
        fileInput.onchange = async () => {
          const file = fileInput.files && fileInput.files[0];
          if (!file) return;
          pickBtn.textContent = "...";
          try { const up = await uploadImage(file); entry.default = filenameFromUpload(up); refresh(); }
          catch (err) { fnameLbl.textContent = "failed"; }
          finally { pickBtn.textContent = "Upload"; }
        };
        defCell.appendChild(pickBtn); defCell.appendChild(fileInput); defCell.appendChild(thumb); defCell.appendChild(fnameLbl);
      } else {
        const defIn = document.createElement("input");
        defIn.type = NUMERIC_TYPES.includes(entry.type) ? "number" : "text";
        defIn.value = entry.default ?? "";
        defIn.placeholder = "default (empty = null)";
        defIn.style.width = "110px";
        defIn.disabled = mode === "output";
        defIn.oninput = () => {
          const v = defIn.value;
          if (v === "") { entry.default = null; return; }
          if (entry.type === "INT") entry.default = parseInt(v, 10);
          else if (entry.type === "FLOAT") entry.default = parseFloat(v);
          else if (entry.type === "BOOLEAN") entry.default = v.toLowerCase() === "true" || v === "1";
          else entry.default = v;
        };
        defCell.appendChild(defIn);
      }
      row.appendChild(defCell);

      // Range/step cells for INT/FLOAT
      if (NUMERIC_TYPES.includes(entry.type)) {
        const mkNum = (key, ph) => {
          const inp = document.createElement("input");
          inp.type = "number";
          inp.value = entry[key] != null ? entry[key] : "";
          inp.placeholder = ph;
          inp.style.width = "60px";
          inp.disabled = mode === "output";
          inp.oninput = () => {
            const v = inp.value;
            if (v === "") { entry[key] = null; return; }
            entry[key] = entry.type === "INT" ? parseInt(v, 10) : parseFloat(v);
          };
          return inp;
        };
        const minL = document.createElement("span"); minL.textContent = "min"; minL.style.fontSize = "10px"; minL.style.opacity = "0.6";
        const maxL = document.createElement("span"); maxL.textContent = "max"; maxL.style.fontSize = "10px"; maxL.style.opacity = "0.6";
        const stepL = document.createElement("span"); stepL.textContent = "step"; stepL.style.fontSize = "10px"; stepL.style.opacity = "0.6";
        row.appendChild(minL); row.appendChild(mkNum("min", "min"));
        row.appendChild(maxL); row.appendChild(mkNum("max", "max"));
        row.appendChild(stepL); row.appendChild(mkNum("step", "step"));

        // control_after_generate: INT-only dropdown (fixed/randomize/increment/decrement)
        if (entry.type === "INT" && mode === "input") {
          const ctrlL = document.createElement("span"); ctrlL.textContent = "after run"; ctrlL.style.fontSize = "10px"; ctrlL.style.opacity = "0.6";
          const ctrlSel = document.createElement("select");
          ctrlSel.style.width = "85px"; ctrlSel.style.fontSize = "11px";
          for (const cm of ["fixed", "randomize", "increment", "decrement"]) {
            const o = document.createElement("option");
            o.value = cm; o.textContent = cm;
            if ((entry.controlMode || "fixed") === cm) o.selected = true;
            ctrlSel.appendChild(o);
          }
          ctrlSel.onchange = () => { entry.controlMode = ctrlSel.value; };
          row.appendChild(ctrlL); row.appendChild(ctrlSel);
        }
      }

      const delBtn = document.createElement("button");
      delBtn.textContent = "✕"; delBtn.title = "Remove";
      delBtn.onclick = () => { draft.splice(idx, 1); renderRows(); updateCount(); };
      row.appendChild(delBtn);

      rowsEl.appendChild(row);
    });
  }
  renderRows();

  const addBtn = document.createElement("button");
  addBtn.textContent = "+ Add parameter";
  addBtn.style.marginTop = "8px";
  addBtn.disabled = draft.length >= MAX_PARAMS;
  addBtn.onclick = () => {
    if (draft.length >= MAX_PARAMS) return;
    draft.push({ name: "", type: "STRING", default: null, min: null, max: null, step: null });
    renderRows();
    updateCount();
  };
  dlg.appendChild(addBtn);

  const countLabel = document.createElement("div");
  countLabel.style.fontSize = "11px"; countLabel.style.opacity = "0.6"; countLabel.style.marginTop = "4px";
  function updateCount() {
    countLabel.textContent = `${draft.length} / ${MAX_PARAMS} parameters`;
    addBtn.disabled = draft.length >= MAX_PARAMS;
  }
  updateCount();
  dlg.appendChild(countLabel);

  const btnRow = document.createElement("div");
  btnRow.style.display = "flex"; btnRow.style.gap = "8px"; btnRow.style.justifyContent = "flex-end"; btnRow.style.marginTop = "16px";
  const cancelBtn = document.createElement("button");
  cancelBtn.textContent = "Cancel";
  cancelBtn.onclick = () => { dlg.close(); };
  const okBtn = document.createElement("button");
  okBtn.textContent = "Apply";
  okBtn.onclick = () => {
    const seen = new Set();
    const clean = [];
    for (const e of draft) {
      const n = (e.name || "").trim();
      if (!n) continue;
      if (seen.has(n)) continue;
      seen.add(n);
      const c = { name: n, type: e.type, default: e.default ?? null, min: e.min ?? null, max: e.max ?? null, step: e.step ?? null };
      if (e.type === "INT" && e.controlMode && e.controlMode !== "fixed") c.controlMode = e.controlMode;
      clean.push(c);
    }
    setConfig(node, clean);
    if (mode === "input") {
      applyInputPanelSlots(node, clean);
      buildParamWidgets(node, clean);
    } else {
      applyOutputPanelSlots(node, clean);
      refreshSchemaBoxFromConfig(node);
    }
    if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
    dlg.close();
  };
  btnRow.appendChild(cancelBtn); btnRow.appendChild(okBtn);
  dlg.appendChild(btnRow);

  document.body.appendChild(dlg);
  dlg.showModal();
  dlg.addEventListener("close", () => { dlg.remove(); });
}

// ---- Reset to defaults button on the node (Input Panel only) ----
// Clears all __val_* live values by writing each param's default back into its
// live value. After clicking, export yields clean defaults (no leaked live
// values from prior runs).
function addResetButton(node) {
  const btn = document.createElement("button");
  btn.textContent = "Reset to defaults";
  btn.style.cursor = "pointer"; btn.style.padding = "4px 10px"; btn.style.borderRadius = "4px";
  btn.style.background = "var(--comfy-input-bg, #333)"; btn.style.color = "var(--fg-color, #fff)";
  btn.style.border = "1px solid var(--border-color, #555)";
  btn.style.fontSize = "11px";
  btn.addEventListener("click", (e) => {
    e.preventDefault(); e.stopPropagation();
    if (!node.gen2ParamWidgets) return;
    for (const pw of node.gen2ParamWidgets) {
      const def = pw.entry.default ?? null;
      pw.widget.setLiveValue(def);
    }
    if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
    btn.textContent = "Reset ✓";
    setTimeout(() => { btn.textContent = "Reset to defaults"; }, 1000);
  });
  const widget = {
    type: "gen2_reset_button", name: "reset_defaults", element: btn,
    draw(ctx, n, wWidth, y) {
      if (widget.__dead) return;
      if (!btn.parentNode) document.body.appendChild(btn);
      Object.assign(btn.style, { position: "absolute", left: (n.pos[0] + 8) + "px", top: (n.pos[1] + y) + "px", width: (wWidth - 16) + "px" });
    },
    computeSize() { return [200, 24]; },
    serializeValue() { return undefined; },
  };
  if (typeof node.addCustomWidget === "function") node.addCustomWidget(widget);
  else { node.customWidgets = node.customWidgets || []; node.customWidgets.push(widget); }
  node.gen2ResetWidget = widget;
}

// ---- Configure button on the node ----
function addConfigureButton(node) {
  const btn = document.createElement("button");
  btn.textContent = "Configure";
  btn.style.cursor = "pointer"; btn.style.padding = "4px 10px"; btn.style.borderRadius = "4px";
  btn.style.background = "var(--comfy-input-bg, #333)"; btn.style.color = "var(--fg-color, #fff)";
  btn.style.border = "1px solid var(--border-color, #555)";
  btn.addEventListener("click", (e) => {
    e.preventDefault(); e.stopPropagation();
    const mode = NODE_TYPES[node.comfyClass]?.mode || "input";
    openConfigDialog(node, mode);
  });
  const widget = {
    type: "gen2_configure_button", name: "configure", element: btn,
    draw(ctx, n, wWidth, y) {
      if (widget.__dead) return;
      if (!btn.parentNode) document.body.appendChild(btn);
      Object.assign(btn.style, { position: "absolute", left: (n.pos[0] + 8) + "px", top: (n.pos[1] + y) + "px", width: (wWidth - 16) + "px" });
    },
    computeSize() { return [200, 28]; },
    serializeValue() { return undefined; },
  };
  if (typeof node.addCustomWidget === "function") node.addCustomWidget(widget);
  else { node.customWidgets = node.customWidgets || []; node.customWidgets.push(widget); }
  const origOnRemoved = node.onRemoved;
  node.onRemoved = function () {
    btn.remove();
    clearParamWidgets(this);
    clearSchemaBox(this);
    if (this.gen2ResetWidget) removeWidgetFromNode(this, this.gen2ResetWidget);
    origOnRemoved?.apply(this, arguments);
  };
}

// ---- Extension registration ----
app.registerExtension({
  name: "gen2.api_panels",

  async setup() {
    // After any generation completes, run control_after_generate handlers on
    // every InputPanel node (randomize/increment/decrement its INT params).
    // The "executed" event fires per-node with {node_id, ...}.
    app.api.addEventListener("executed", (e) => {
      const detail = e.detail;
      if (!detail || !detail.node_id) return;
      const node = app.graph?.getNodeById?.(detail.node_id);
      // Also run on ALL input panels, since the executed node might be the
      // OutputPanel but the seed lives on the InputPanel. Running on every
      // executed event is cheap (handlers are a no-op when mode is "fixed").
      const graph = app.graph || (window?.LiteGraph?.get?.()?.graph);
      if (!graph) return;
      for (const n of graph.nodes) {
        if (n.comfyClass === "Gen2_InputPanel" && n.gen2ControlHandlers) {
          for (const h of n.gen2ControlHandlers) {
            try { h(); } catch (err) { console.error("[gen2] control handler error", err); }
          }
        }
      }
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (!NODE_TYPES[nodeData.name]) return;
    const mode = NODE_TYPES[nodeData.name].mode;

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      origOnNodeCreated?.apply(this, arguments);
      addConfigureButton(this);
      if (mode === "input") addResetButton(this);
      if (mode === "output") buildSchemaBox(this);
      queueMicrotask(() => {
        const w = getConfigWidget(this);
        if (w) {
          const entries = parseConfig(w.value || "[]");
          if (mode === "input") {
            applyInputPanelSlots(this, entries);
            buildParamWidgets(this, entries);
          } else {
            applyOutputPanelSlots(this, entries);
            refreshSchemaBoxFromConfig(this);
          }
        }
      });
    };

    // After execution, the OutputPanel's UI payload carries schema_json from the
    // backend (built from the InputPanel's config via PANEL_LINK). Update the
    // textbox so it reflects the actually-run config.
    if (mode === "output") {
      const origOnExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        origOnExecuted?.apply(this, arguments);
        const schema = message?.ui?.schema_json;
        if (schema && this.gen2SchemaBox) {
          this.gen2SchemaBox.textarea.value = Array.isArray(schema) ? schema[0] : schema;
        }
      };
    }
  },
});
