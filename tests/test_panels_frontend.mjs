import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../gen2/api_panel_nodes/web/js/gen2Panels.js", import.meta.url), "utf8")
  .replace(/^import \{ app \} from .*;$/m, "");
const plain = (value) => JSON.parse(JSON.stringify(value));

// Model the documented extension boundary, not a GPU or a browser renderer.
// Frontend 1.55 distinguishes workflow widget.serialize from API
// widget.options.serialize and restores widgets_values_named during configure.
function harness() {
  const pending = [], errors = [], domWidgets = new Set();
  let extension;
  class Element {
    constructor(tag) { this.tagName = tag; this.style = {}; this.children = []; this.listeners = new Map(); this.value = ""; }
    appendChild(child) { this.children.push(child); child.parentNode = this; return child; }
    remove() { if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1); this.parentNode = null; }
    get isConnected() { return this === body || !!this.parentNode?.isConnected; }
    set innerHTML(value) { for (const child of this.children) child.parentNode = null; this.children = []; }
    addEventListener(name, listener) { this.listeners.set(name, [...(this.listeners.get(name) || []), listener]); }
    dispatch(name, event = {}) { for (const callback of this.listeners.get(name) || []) callback(event); }
    setAttribute(name, value) { this[name] = value; }
    removeAttribute(name) { delete this[name]; }
    close() { this.open = false; this.dispatch("close"); }
    showModal() { this.open = true; }
    scrollIntoView() {}
    select() {}
    click() { this.onclick?.({ preventDefault() {}, stopPropagation() {} }); }
  }
  const body = new Element("body");
  const app = { canvas: { pointer: { isDown: false, reset() {} }, setDirty() {} }, registerExtension(value) { extension = value; } };
  const context = vm.createContext({
    app, TextEncoder, URLSearchParams, performance: { now: () => 0 },
    setTimeout: (fn) => pending.push(fn), requestAnimationFrame: (fn) => pending.push(fn),
    window: { setTimeout: (fn) => pending.push(fn), open() {} },
    document: { body, createElement: (tag) => new Element(tag) },
    navigator: {}, console: { error: (...args) => errors.push(args) },
  });
  vm.runInContext(source, context, { filename: "gen2Panels.js" });
  const api = vm.runInContext("({parseConfig, serializeConfig, legacyParamId, oldLegacyParamId, validateRuntimeValue, migrateRuntimeValues, imageRefToViewUrl, syncSlots, rebuildPanel, makeParamWidget, openConfigDialog})", context);
  const flush = () => { let count = 0; while (pending.length) { assert.ok(count++ < 1000, "deferred callbacks must settle"); pending.shift()(); } };
  class Node {
    constructor(config = []) {
      this.widgets = [{ name: "_config", type: "text", value: JSON.stringify(config), options: {} }];
      this.properties = {}; this.outputs = [{ name: "PANEL_LINK", type: "*", links: [90] }];
      this.inputs = [{ name: "PANEL_LINK", type: "*", link: 90 }]; this.size = [300, 100];
    }
    addWidget(type, name, value, callback, options = {}) { const widget = { type, name, value, callback, options }; this.widgets.push(widget); return widget; }
    addDOMWidget(name, type, element, options) {
      const widget = this.addWidget(type, name, undefined, undefined, options);
      widget.element = element; domWidgets.add(widget); widget.onRemove = () => domWidgets.delete(widget); return widget;
    }
    ensureWidgetRemoved(widget) { widget.onRemove?.(); this.widgets.splice(this.widgets.indexOf(widget), 1); }
    addOutput(name, type) { this.outputs.push({ name, type, links: [] }); }
    addInput(name, type) { this.inputs.push({ name, type, link: null }); }
    removeOutput(index) { this.outputs.splice(index, 1); }
    removeInput(index) { this.inputs.splice(index, 1); }
    computeSize() { return [300, 100]; }
    setSize(value) { this.size = value; }
    setDirtyCanvas() {}
  }
  const register = async (type = "Gen2_InputPanel") => { class RegisteredNode extends Node {} await extension.beforeRegisterNodeDef(RegisteredNode, { name: type }); return RegisteredNode; };
  const exportApi = (node) => Object.fromEntries(node.widgets.filter((w) => w.options?.serialize !== false).map((w) => [w.name, w.serializeValue ? w.serializeValue(node) : w.value]));
  const exportWorkflow = (node) => ({ properties: plain(node.properties), widgets_values_named: Object.fromEntries(node.widgets.filter((w) => w.serialize !== false).map((w) => [w.name, w.value])) });
  return { api, context, Node, register, flush, body, errors, domWidgets, exportApi, exportWorkflow };
}

const textEntry = (overrides = {}) => ({ id: "text-id", name: "prompt", type: "STRING", default: "default prompt", ...overrides });
const numberEntry = (overrides = {}) => ({ id: "number-id", name: "count", type: "INT", default: 2, min: 0, max: 10, step: 2, ...overrides });
const seedEntry = (overrides = {}) => numberEntry({ id: "seed-id", name: "seed", type: "SEED", default: 3, min: 0, max: 10, step: 3, controlMode: "increment", ...overrides });

test("configuration rejects malformed documents, reserved names, duplicate IDs, and incorrect defaults", () => {
  const { api } = harness();
  for (const raw of ["{broken", "{}", "null", [null], ["str"], Array.from({ length: 33 }, (_, i) => textEntry({ id: `${i}`, name: `p${i}` }))]) {
    assert.throws(() => api.parseConfig(raw));
  }
  for (const name of ["_config", "PANEL_LINK", "param_0", "param_31", "__proto__", "constructor", "prototype"]) {
    assert.throws(() => api.parseConfig([textEntry({ name })]), /reserved/);
    assert.throws(() => api.parseConfig([{ name, type: "STRING" }], "output"), /reserved/);
  }
  assert.throws(() => api.parseConfig([textEntry(), textEntry({ name: "other" })]), /id.*duplicated/);
  for (const entry of [textEntry({ default: false }), textEntry({ type: "BOOLEAN", default: "false" }), numberEntry({ default: "2.0" }), numberEntry({ default: 1.5 }), numberEntry({ default: 3 }), numberEntry({ step: 0 }), numberEntry({ max: Infinity }), textEntry({ type: "IMAGE", default: " " }), textEntry({ type: "COMBO", default: "" }), textEntry({ id: "__proto__" }), textEntry({ id: null }), textEntry({ name: 12 })]) {
    assert.throws(() => api.parseConfig([entry]));
  }
  assert.equal(api.parseConfig([textEntry({ default: "" })])[0].default, "");
  assert.equal(api.parseConfig([{ name: "result", type: "IMAGE" }], "output")[0].name, "result");
});

test("legacy numeric config strings normalize while runtime values stay strictly typed", () => {
  const { api } = harness();
  const entry = api.parseConfig([numberEntry({ default: "2", min: "0", max: "10", step: "2" })])[0];
  assert.equal(entry.default, 2); assert.equal(entry.step, 2);
  assert.throws(() => api.validateRuntimeValue("2", entry), /finite number/);
  const float = api.parseConfig([numberEntry({ type: "FLOAT", default: "1e-1", min: "0", max: "1", step: ".1" })])[0];
  assert.equal(float.default, 0.1);
  for (const value of ["", " ", "0x10", "Infinity", "NaN", "1e0", "1.0"]) assert.throws(() => api.parseConfig([numberEntry({ default: value })]));
});

test("seed control modes normalize text while only absent modes receive a default", () => {
  const { api } = harness();
  assert.equal(api.parseConfig([seedEntry({ controlMode: " FIXED " })])[0].controlMode, "fixed");
  const entry = seedEntry(); delete entry.controlMode;
  assert.equal(api.parseConfig([entry])[0].controlMode, "randomize");
  for (const controlMode of ["", " ", null, false, 0]) {
    assert.throws(() => api.parseConfig([seedEntry({ controlMode })]), /invalid after-run mode/);
  }
});

test("legacy IDs use the backend SHA1 UTF-8 contract and preserve explicit IDs", () => {
  const { api } = harness();
  for (const name of [" prompt ", "猫 😀", "a".repeat(100), "foo/bar"]) {
    const normalized = name.trim();
    const slug = normalized.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 24) || "param";
    const hash = createHash("sha1").update(`0:${normalized}:STRING`).digest("hex").slice(0, 12);
    const entry = api.parseConfig([{ name, type: " string ", default: "" }])[0];
    assert.equal(entry.id, `legacy-${slug}-${hash}`);
    assert.equal(entry._legacyRuntimeId, api.oldLegacyParamId({ name, type: " string " }, 0));
  }
  assert.equal(api.parseConfig([textEntry({ id: "existing" })])[0].id, "existing");
});

test("loading no-id legacy configs migrates FNV runtime values and control modes without serializing aliases", () => {
  const { api, Node, flush } = harness();
  const raw = seedEntry(); delete raw.id;
  const node = new Node([raw]);
  const oldId = api.oldLegacyParamId(raw, 0);
  node.properties.gen2RuntimeValues = { [oldId]: 6 };
  node.properties.gen2RuntimeControlModes = { [oldId]: "decrement" };
  api.rebuildPanel(node, "input"); flush();
  const config = JSON.parse(node.widgets[0].value);
  assert.equal(config[0]._legacyRuntimeId, undefined);
  assert.equal(node.properties.gen2RuntimeValues[config[0].id], 6);
  assert.equal(node.properties.gen2RuntimeControlModes[config[0].id], "decrement");
  assert.equal(Object.hasOwn(node.properties.gen2RuntimeValues, oldId), false);
});

test("corrupt config fails before existing widgets, graph links, or raw configuration are changed", () => {
  const { api, Node, flush } = harness();
  const node = new Node([textEntry()]); api.rebuildPanel(node, "input"); flush();
  node.outputs[1].links = [41];
  const widget = node.widgets.find((w) => w.name === "prompt");
  node.widgets[0].value = "{broken";
  assert.throws(() => api.rebuildPanel(node, "input"), /valid JSON/);
  assert.equal(node.widgets[0].value, "{broken");
  assert.ok(node.widgets.includes(widget)); assert.deepEqual(node.outputs[1].links, [41]);
});

test("ordinary rebuild preserves slot identity and links; explicit refresh preserves only PANEL_LINK", () => {
  const { api, Node, flush } = harness();
  const node = new Node([textEntry()]); api.rebuildPanel(node, "input"); flush();
  const panel = node.outputs[0], output = node.outputs[1]; output.links = [42];
  api.rebuildPanel(node, "input"); flush();
  assert.equal(node.outputs[0], panel); assert.equal(node.outputs[1], output);
  assert.deepEqual(output.links, [42]); assert.equal(output.name, "param_0");
  api.syncSlots(node, "input", [textEntry()], true);
  assert.equal(node.outputs[0], panel); assert.deepEqual(panel.links, [90]); assert.notEqual(node.outputs[1], output);
});

test("seed controls advance once per successful queue, respecting step and wrap in a queued batch", async () => {
  const h = harness(), Type = await h.register();
  const node = new Type([seedEntry()]); node.onNodeCreated(); h.flush();
  const widget = node.widgets.find((w) => w.name === "seed");
  const submitted = [];
  for (let i = 0; i < 4; i++) { submitted.push(h.exportApi(node).seed); widget.afterQueued(); }
  assert.deepEqual(submitted, [3, 6, 9, 0]);
  assert.equal(h.exportApi(node).seed, 3); assert.equal(h.exportApi(node).seed, 3, "export must not advance");
  node.onExecuted?.({ gen2_input_executed: [true] });
  assert.equal(widget.value, 3, "execution completion must not advance a second time");
  const ctrl = node.widgets.find((w) => w.type === "combo");
  ctrl.value = "decrement"; widget.value = 0; widget.afterQueued(); assert.equal(widget.value, 9);
  ctrl.value = "fixed"; widget.afterQueued(); assert.equal(widget.value, 9);
  ctrl.value = "randomize";
  for (let i = 0; i < 50; i++) { widget.afterQueued(); assert.ok([0, 3, 6, 9].includes(widget.value)); }
});

test("seed increment remains exact near MAX_SAFE_INTEGER", () => {
  const h = harness(), max = Number.MAX_SAFE_INTEGER;
  const node = new h.Node([seedEntry({ default: max, min: max - 4, max, step: 2 })]);
  h.api.rebuildPanel(node, "input"); h.flush();
  const widget = node.widgets.find((w) => w.name === "seed"); widget.afterQueued();
  assert.equal(widget.value, max - 4); widget.afterQueued(); assert.equal(widget.value, max - 2);
});

test("API export serializes current typed values and filenames while excluding UI-only controls", () => {
  const h = harness();
  const entries = [textEntry(), numberEntry(), textEntry({ id: "bool", name: "enabled", type: "BOOLEAN", default: false }), textEntry({ id: "image", name: "source", type: "IMAGE", default: "old.png" })];
  const node = new h.Node(entries); h.api.rebuildPanel(node, "input"); h.flush();
  node.widgets.find((w) => w.name === "prompt").value = "current prompt";
  node.widgets.find((w) => w.name === "count").value = 4;
  const image = node.widgets.find((w) => w.name === "source"); image.value = "folder/new.png [input]";
  const values = h.exportApi(node);
  assert.deepEqual(Object.keys(values).sort(), ["_config", "count", "enabled", "prompt", "source"]);
  assert.equal(values.prompt, "current prompt"); assert.equal(values.count, 4); assert.equal(values.source, image.value);
  assert.equal(node.properties.gen2RuntimeValues.image, image.value);
  assert.equal(JSON.parse(values._config)[0].default, "default prompt");
  assert.deepEqual(Object.keys(h.exportWorkflow(node).widgets_values_named).sort(), Object.keys(values).sort());
  node.widgets.find((w) => w.name === "count").value = 3.5; assert.throws(() => h.exportApi(node), /integer/);
  node.widgets.find((w) => w.name === "count").value = 4;
  node.widgets.find((w) => w.name === "enabled").value = "false"; assert.throws(() => h.exportApi(node), /true or false/);
  node.widgets.find((w) => w.name === "enabled").value = false;
  image.value = { tensor: true }; assert.throws(() => h.exportApi(node), /string/);
});

test("named workflow restoration survives deferred creation and overrides stale runtime properties", async () => {
  const h = harness(), Type = await h.register();
  const config = [textEntry(), numberEntry()];
  const node = new Type(config); node.properties.gen2RuntimeValues = { "text-id": "stale", "number-id": 2 };
  node.onNodeCreated();
  node.onConfigure({ widgets_values_named: { _config: JSON.stringify(config), prompt: "saved edit", count: 8 } });
  h.flush();
  assert.equal(h.exportApi(node).prompt, "saved edit"); assert.equal(h.exportApi(node).count, 8);
});

test("reconfiguring a range or type resets only values that no longer satisfy the contract", () => {
  const h = harness(), old = [numberEntry()], node = new h.Node(old);
  node.properties.gen2RuntimeValues = { "number-id": 8 };
  h.api.migrateRuntimeValues(node, old, [numberEntry({ max: 6 })]); assert.equal(node.properties.gen2RuntimeValues["number-id"], 2);
  node.properties.gen2RuntimeValues["number-id"] = 4;
  h.api.migrateRuntimeValues(node, old, [numberEntry({ default: 3, step: 3 })]); assert.equal(node.properties.gen2RuntimeValues["number-id"], 3);
  h.api.migrateRuntimeValues(node, old, [textEntry({ id: "number-id", default: "new default" })]); assert.equal(node.properties.gen2RuntimeValues["number-id"], "new default");
});

test("Configure rejects reserved names and misaligned defaults before touching the live graph", () => {
  const h = harness(), node = new h.Node([numberEntry()]);
  h.api.rebuildPanel(node, "input"); h.flush();
  const originalConfig = node.widgets[0].value, slot = node.outputs[1]; slot.links = [44];
  h.api.openConfigDialog(node, "input");
  const dialog = h.body.children.find((el) => el.tagName === "dialog");
  const warning = dialog.children.find((el) => el.role === "alert");
  const row = dialog.children[3].children[0], name = row.children[1], defaultValue = row.children[2];
  const apply = dialog.children.at(-1).children.find((el) => el.textContent === "Apply");
  name.value = "_config"; name.oninput(); apply.click();
  assert.match(warning.textContent, /reserved/); assert.equal(dialog.open, true);
  name.value = "count"; name.oninput(); defaultValue.value = "3"; defaultValue.oninput(); apply.click();
  assert.match(warning.textContent, /align to step/); assert.equal(node.widgets[0].value, originalConfig);
  assert.equal(node.outputs[1], slot); assert.deepEqual(slot.links, [44]);
  dialog.children.at(-1).children.find((el) => el.textContent === "Cancel").click(); h.flush();
  assert.equal(node.outputs[1], slot); assert.equal(node.widgets[0].value, originalConfig);
  assert.equal(h.body.children.length, 0);
});

test("Configure Apply accepts an empty STRING default, preserves current value and PANEL_LINK", () => {
  const h = harness(), node = new h.Node([textEntry()]);
  h.api.rebuildPanel(node, "input"); h.flush();
  const panel = node.outputs[0], previousSlot = node.outputs[1]; previousSlot.links = [45];
  node.gen2ParamWidgets[0].setValue("current text");
  h.api.openConfigDialog(node, "input");
  const dialog = h.body.children.find((el) => el.tagName === "dialog");
  const defaultValue = dialog.children[3].children[0].children[2];
  defaultValue.value = ""; defaultValue.oninput();
  dialog.children.at(-1).children.find((el) => el.textContent === "Apply").click(); h.flush();
  assert.equal(JSON.parse(node.widgets[0].value)[0].default, "");
  assert.equal(h.exportApi(node).prompt, "current text");
  assert.equal(node.outputs[0], panel); assert.deepEqual(panel.links, [90]);
  assert.notEqual(node.outputs[1], previousSlot); assert.deepEqual(node.outputs[1].links, []);
  assert.equal(h.body.children.length, 0);
});

test("node removal cancels delayed rebuilds and cleans image upload elements and DOM registrations", async () => {
  const h = harness(), Type = await h.register();
  const entry = textEntry({ type: "IMAGE", default: "image.png" });
  const node = new Type([entry]); node.onNodeCreated(); node.onRemoved(); h.flush();
  assert.equal(h.body.children.length, 0); assert.equal(h.domWidgets.size, 0);
  node.onAdded(); h.flush(); assert.equal(h.body.children.length, 1); assert.equal(h.domWidgets.size, 1);
  node.onConfigure({}); node.onRemoved(); h.flush();
  assert.equal(h.body.children.length, 0); assert.equal(h.domWidgets.size, 0);
});

test("repeated IMAGE rebuilds dispose old DOM widgets without removing the replacement", () => {
  const h = harness(), node = new h.Node([textEntry({ type: "IMAGE", default: "image.png" })]);
  for (let i = 0; i < 4; i++) { h.api.rebuildPanel(node, "input"); h.flush(); assert.equal(h.body.children.length, 1); assert.equal(h.domWidgets.size, 1); }
});

test("image previews honor ComfyUI annotated folder types", () => {
  const { api } = harness();
  const result = new URL(api.imageRefToViewUrl("folder/image x.png [output]"), "http://localhost");
  assert.equal(result.searchParams.get("filename"), "image x.png");
  assert.equal(result.searchParams.get("subfolder"), "folder"); assert.equal(result.searchParams.get("type"), "output");
});

test("output panel accepts the frontend's raw onExecuted payload and preserves version-1 documents", async () => {
  const h = harness(), Type = await h.register("Gen2_OutputPanel"), node = new Type([{ id: "out", name: "result", type: "STRING" }]);
  node.onNodeCreated(); h.flush();
  const doc = { version: 1, inputs: { schema: [], latest_values: {} }, outputs: { schema: [{ id: "out", name: "result", type: "STRING" }], latest_values: { result: "first" } } };
  node.onExecuted({ document_json: [JSON.stringify(doc)] });
  assert.deepEqual(JSON.parse(node.gen2SchemaWidget.textarea.value), doc);
  doc.outputs.latest_values.result = "second";
  node.onExecuted({ prompt_id: "same", output: { document: [doc] } });
  doc.outputs.latest_values.result = "third";
  node.onExecuted({ prompt_id: "same", output: { document_json: [JSON.stringify(doc)] } });
  assert.equal(JSON.parse(node.gen2SchemaWidget.textarea.value).outputs.latest_values.result, "third");
});
