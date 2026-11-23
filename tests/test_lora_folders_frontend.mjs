import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../gen2/api_panel_nodes/web/js/gen2LoraFolders.js", import.meta.url), "utf8");
let extension;
vm.runInNewContext(source.replace(/^import .*;$/m, ""), {
  app: { registerExtension(value) { extension = value; } },
});

function makeNode(comfyClass, defaults) {
  const node = {
    comfyClass,
    inputs: comfyClass === "Gen2_LoRALoad" ? [{ name: "number", type: "STRING", link: 17 }] : [],
    widgets: Object.entries(defaults).map(([name, value]) => ({ name, value })),
    size: [400, 300],
    computeSize() { return [200, 40 + this.widgets.filter((w) => !w.hidden).length * 24]; },
    setSize(size) { this.size = size; },
    setDirtyCanvas() {},
    configure(values) { this.widgets.forEach((w, i) => { if (i < values.length) w.value = values[i]; }); },
  };
  extension.nodeCreated(node);
  return node;
}

function assertVisible(field, visible) {
  assert.equal(field.hidden, !visible);
  assert.equal(field.options.hidden, !visible);
}

test("Save toggles hide independent fields while preserving values, order, and manual sizing", () => {
  const node = makeNode("Gen2_LoRASave", {
    file: "model.pt", trigger: "ink", model: "q21", type: "character",
    new_number: true, number: "", use_name: false, name: "",
  });
  const [, , , , newNumber, number, useName, name] = node.widgets;
  const initialHeight = node.size[1];
  assertVisible(number, false);
  assertVisible(name, false);
  const order = node.widgets.slice();
  newNumber.value = false;
  newNumber.callback(false);
  assertVisible(number, true);
  assertVisible(name, false);
  number.value = "3";
  useName.value = true;
  useName.callback(true);
  assertVisible(name, true);
  name.value = "my_character";
  assert.equal(node.size[1], initialHeight + 48);
  assert.equal(node.size[0], 400);

  const saved = node.widgets.map((w) => w.value);
  newNumber.value = true;
  newNumber.callback(true);
  useName.value = false;
  useName.callback(false);
  assert.equal(node.size[1], initialHeight);
  assert.equal(number.value, "3");
  assert.equal(name.value, "my_character");
  assert.deepEqual(node.widgets, order);
  node.size[1] = 500;
  node.configure(saved);
  assertVisible(number, true);
  assertVisible(name, true);
  assert.equal(node.size[1], 500);
});

test("Load shows the string label on new and restored number sockets and restores name visibility", () => {
  const node = makeNode("Gen2_LoRALoad", { model: "", type: "", use_name: false, name: "" });
  const name = node.widgets[3];
  const numberSocket = node.inputs[0];
  assert.equal(numberSocket.label, "string");
  delete numberSocket.label;
  node.configure(["q21", "style"]);
  assert.equal(node.inputs[0], numberSocket);
  assert.deepEqual(numberSocket, { name: "number", type: "STRING", link: 17, label: "string" });
  assertVisible(name, false);
  node.configure(["q21", "character", true, "my_character"]);
  assertVisible(name, true);
  assert.equal(name.value, "my_character");
  assert.deepEqual(node.widgets.map((w) => w.name), ["model", "type", "use_name", "name"]);
});

test("other nodes and older definitions keep their widget callbacks", () => {
  const callback = () => "original";
  for (const comfyClass of ["Gen2_LoRASave", "OtherLoRA"]) {
    const node = { comfyClass, widgets: [{ name: "number", value: "3", callback }] };
    extension.nodeCreated(node);
    assert.equal(node.widgets[0].callback, callback);
    assert.equal(node.widgets[0].hidden, undefined);
  }
});
