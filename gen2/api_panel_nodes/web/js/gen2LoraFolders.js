import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "gen2.lora_folders",
  nodeCreated(node) {
    const save = node.comfyClass === "Gen2_LoRASave";
    if (!save && node.comfyClass !== "Gen2_LoRALoad") return;

    const widget = (name) => node.widgets.find((item) => item.name === name);
    const fields = [[widget("use_name"), widget("name"), true]];
    if (save) fields.push([widget("new_number"), widget("number"), false]);
    if (fields.some(([toggle, field]) => !toggle || !field)) return;

    function updateVisibility(restoring = false) {
      // Loading an older graph can restore the slot without its new display label.
      if (!save) {
        const input = node.inputs.find((slot) => slot.name === "number");
        if (input) input.label = "string";
      }
      const previousHeight = node.computeSize()[1];
      for (const [toggle, field, visibleWhen] of fields) {
        const hidden = toggle.value !== visibleWhen;
        // LiteGraph and the Vue renderer read separate visibility flags.
        // Keep the widget in place so its workflow and API values still serialize.
        field.hidden = hidden;
        field.options = { ...field.options, hidden };
      }
      const minHeight = node.computeSize()[1];
      const height = restoring ? node.size[1] : node.size[1] + minHeight - previousHeight;
      node.setSize([node.size[0], Math.max(minHeight, height)]);
      node.setDirtyCanvas(true, true);
    }

    for (const [toggle] of fields) {
      const callback = toggle.callback;
      toggle.callback = function (...args) {
        const result = callback?.apply(this, args);
        updateVisibility();
        return result;
      };
    }
    const configure = node.configure;
    node.configure = function (...args) {
      const result = configure.apply(this, args);
      updateVisibility(true);
      return result;
    };
    updateVisibility();
  },
});
