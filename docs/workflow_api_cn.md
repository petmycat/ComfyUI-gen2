# Gen2 工作流 API 集成指南

本文面向需要将 ComfyUI 工作流接入 API 服务的后端开发者，重点说明：

- 传统的“手工维护节点 ID”方案如何工作；
- `Gen2_InputPanel` 和 `Gen2_OutputPanel` 与传统方案有什么区别；
- `api_nodes/workflow_contract.py` 如何切入现有 API 服务并逐步实现自动化；
- 如何发现参数、校验请求、替换 API prompt 和提取执行结果。

英文版本：[`workflow_api_en.md`](workflow_api_en.md)

## 1. 先理解传统方案

在没有 Gen2 API Panels 时，服务端程序员通常会打开 ComfyUI 的 API 格式导出，人工查找需要修改的节点。

例如：

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

服务端随后维护一份工作流专用绑定表：

```python
WORKFLOW_BINDINGS = {
    "prompt": {"node_id": "15", "input_key": "text"},
    "seed": {"node_id": "27", "input_key": "seed"},
    "steps": {"node_id": "27", "input_key": "steps"},
    "inputImage": {"node_id": "31", "input_key": "image"},
}
```

每次执行时，复制模板并替换指定位置：

```python
prompt["15"]["inputs"]["text"] = request_body["prompt"]
prompt["27"]["inputs"]["seed"] = request_body["seed"]
prompt["27"]["inputs"]["steps"] = request_body["steps"]
prompt["31"]["inputs"]["image"] = request_body["inputImage"]
```

输出也需要人工绑定，例如从节点 `44` 读取图片：

```python
images = history[prompt_id]["outputs"]["44"]["images"]
```

这是一种有效且仍然可用的方案。对于少量、长期不变的工作流，它足够直接。

## 2. 传统方案的限制不是“不能运行”，而是缺少公开契约

传统方案的完整接口定义被拆散在多个地方：

```text
workflow_api.json
+ 服务端手写的输入节点 ID
+ 服务端手写的 input_key
+ 服务端手写的类型与范围校验
+ 服务端手写的输出节点 ID
+ 前端单独维护的表单定义
```

ComfyUI API 导出可以表达：

> 节点 27 的 `seed` 当前值是 12345。

但它不会自动表达：

> `seed` 是一个允许外部用户修改的公开参数，类型为整数，默认值为 0，具有指定的最小值、最大值和步长。

ComfyUI 节点输入是工作流的内部实现；产品 API 参数则是对外接口。传统方案直接把二者绑定在一起。

节点 ID 也不是稳定的业务身份。删除并重建节点、复制节点、合并工作流或替换实现节点，都可能改变节点 ID 和输入字段。

旧绑定可能因此：

1. 指向已经不存在的节点并立即报错；
2. 指向另一个节点并修改错误字段；
3. 字段仍可修改，但含义已经变化，造成“执行成功但结果错误”。

第三种情况尤其难以发现。

## 3. Gen2 Panels 定义的是工作流的公开边界

Gen2 方案在工作流内部增加明确的输入与输出边界：

```text
外部 API 输入
    ↓
Gen2_InputPanel
    ↓
工作流内部节点和连线
    ↓
Gen2_OutputPanel
    ↓
外部 API 输出
```

工作流作者在 `Gen2_InputPanel` 中声明公开参数，例如：

```text
prompt       STRING
seed         SEED
steps        INT
inputImage   IMAGE
```

然后将这些参数连接到真正控制工作流的内部节点。

服务端只修改 Input Panel 的公开字段，不需要知道内部是：

```text
InputPanel.seed → KSampler.seed
```

还是：

```text
InputPanel.seed → SeedProcessor → NoiseGenerator → Sampler
```

同样，工作流作者在 `Gen2_OutputPanel` 中声明：

```text
resultImage  IMAGE
maskImage    IMAGE
score        FLOAT
caption      STRING
```

并把内部计算结果连接到对应输入。

核心区别是：

```text
传统方案：服务端直接修改工作流内部节点。
Gen2 方案：服务端修改工作流作者显式声明的公开入口，并从公开出口读取结果。
```

## 4. 我们仍然修改原生 API JSON，但地址由代码自动发现

Gen2 没有替换 ComfyUI 的原生 API，也没有发明新的执行格式。

API 格式导出仍然是普通 ComfyUI prompt：

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

`workflow_contract.py` 会查找 `Gen2_InputPanel` 和 `Gen2_OutputPanel`，解析 `_config`，自动得到：

- 公开参数名称与稳定 ID；
- 参数类型、默认值和导出时当前值；
- `min`、`max` 和 `step`；
- SEED 默认控制模式；
- API prompt 中的可替换位置；
- Output Panel 在 history 中的位置；
- Input/Output Panel 的 `PANEL_LINK` 配对关系。

Manifest 中仍然包含节点 ID，因为修改 JSON 和读取 history 最终必须使用实际地址。

区别在于：节点 ID 现在是**自动发现的执行地址**，而不是程序员手工维护的业务配置。

## 5. 自动化可以具体切入哪些位置

`workflow_contract.py` 不要求推翻现有 API 服务。它适合替换执行链中最依赖人工的三个部分：

```text
手工输入绑定 → discover_manifest() + patch_api_prompt()
手工参数校验 → validate_call_inputs()
手工输出绑定 → extract_history_results()
```

现有认证、限流、上传、队列、WebSocket、存储和 CDN 逻辑可以继续使用。

### 5.1 替代人工节点 ID 配置

传统代码：

```python
WORKFLOW_INPUTS = {
    "prompt": ("15", "text"),
    "seed": ("27", "seed"),
    "image": ("31", "image"),
}
```

Gen2 注册代码：

```python
manifest = discover_manifest(api_prompt)
```

Manifest 可以在服务启动或工作流注册时生成并缓存。

### 5.2 替代工作流专用校验代码

传统服务需要为每个工作流编写大量 `if`：

```python
if not isinstance(seed, int):
    raise ValueError("seed must be an integer")
if seed < 0 or seed > max_seed:
    raise ValueError("seed is outside the allowed range")
```

Gen2 统一使用：

```python
validated = validate_call_inputs(manifest, request_body)
```

它会检查未知字段、类型、数值范围、步长对齐、SEED 安全范围和 IMAGE 引用。

### 5.3 替代手工 JSON 修改

传统代码：

```python
prompt["15"]["inputs"]["text"] = body["prompt"]
prompt["27"]["inputs"]["seed"] = body["seed"]
```

Gen2 代码：

```python
patched_prompt = patch_api_prompt(
    api_prompt,
    manifest,
    request_body,
)
```

模块会深拷贝模板，只修改 Input Panel 公开字段，并保留 `_config`、节点 ID、连线、元数据和内部节点。

### 5.4 自动生成前端与 OpenAPI Schema

Manifest 不仅提供修改地址，还提供参数语义：

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

服务端可以据此生成：

- Pydantic 或其他请求模型；
- JSON Schema；
- OpenAPI 字段说明；
- WebUI 输入控件；
- IMAGE 上传要求；
- 默认请求示例。

### 5.5 替代人工输出节点配置

传统代码：

```python
WORKFLOW_OUTPUTS = {
    "resultImage": {"node_id": "44", "field": "images"}
}
```

Gen2 代码：

```python
result = extract_history_results(
    history,
    manifest,
    prompt_id=prompt_id,
)
```

服务端不再需要知道结果来自哪个 SaveImage、自定义节点或中间节点，只需读取 Output Panel 声明的字段。

### 5.6 自动注册工作流 Endpoint

推荐目录：

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

服务端启动时扫描目录，发现 manifest，检查两种导出是否一致，再注册统一路由：

```text
GET  /api/v1/workflows/{workflow_id}
POST /api/v1/workflows/{workflow_id}/runs
GET  /api/v1/runs/{run_id}
```

新增工作流时，理想情况下只需导出并放入目录，不再为每个工作流编写新的节点 ID 配置和替换函数。

## 6. Gen2 与传统工作流可以共存

不需要一次迁移所有旧工作流。

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

执行时按模式分流：

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

建议迁移顺序：

1. 新工作流默认使用 Gen2 Panels；
2. 频繁修改、节点 ID 经常变化的旧工作流优先迁移；
3. 已长期稳定的旧工作流可继续使用传统绑定；
4. 保留同一套 ComfyUI 提交、队列与结果存储基础设施。

## 7. 同时保留两种 ComfyUI 导出

每个发布的工作流应保存：

```text
workflows/product-image/
├── workflow.json
└── workflow_api.json
```

- `workflow.json`：用于 UI 重建、编辑、分享和记录导出时的当前控件值。
- `workflow_api.json`：用于提交到 ComfyUI `/prompt`。

不要将普通 workflow 提交给 `/prompt`，也不要用 API 格式文件替代可编辑的普通 workflow。

两个文件必须从同一个已经测试通过的工作流图状态导出。

## 8. 导入公共函数

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

该模块只负责 JSON 和契约处理。`workflow_contract.py` 与 `_config.py` 必须来自同一插件版本。

## 9. 加载导出并发现 Manifest

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

`detect_format()` 会拒绝损坏、混合或不支持的结构。

`discover_manifest()` 不会修改源文件。

## 10. Manifest 的关键字段

简化示例：

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

`default` 是配置默认值，用于面板 Reset；`current_value` 是导出文件中的当前运行值。调用时未覆盖的字段继续使用 `current_value`，不会自动重置到 `default`。

普通 workflow 的 binding 为 `patchable: false`，只能检查，不能作为 API prompt 修改执行。

## 11. 输入类型与校验

- `STRING`：JSON 字符串，允许空字符串。
- `COMBO`：非空 JSON 字符串；当前契约尚未包含选项列表。
- `BOOLEAN`：只能是 JSON `true` 或 `false`。
- `INT`：范围内且符合 `step` 的整数。
- `FLOAT`：范围内且符合 `step` 的有限数值。
- `SEED`：配置安全范围内且符合 `step` 的整数。
- `IMAGE`：非空 ComfyUI 图片引用字符串。

单个面板或参数名全局唯一时，可以发送扁平请求：

```json
{
  "prompt": "a product photo",
  "seed": 12345,
  "inputImage": "requests/request-123/source.png"
}
```

多个面板存在同名参数时，必须按面板分组：

```json
{
  "1": {"seed": 12345},
  "7": {"seed": 67890}
}
```

空请求也合法：

```json
{}
```

它表示使用 API 导出中保存的全部当前值。

## 12. 准备并提交 API Prompt

```python
patched_prompt, manifest = prepare_api_prompt(
    api_prompt,
    {
        "prompt": "a red running shoe, studio lighting",
        "seed": 42,
    },
)
```

提交到 ComfyUI：

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

必须将 `prompt_id` 与 API 请求关联。并发执行时不要使用全局“最新 history”。

## 13. IMAGE 输入

IMAGE 参数是 ComfyUI 文件引用，不是原始字节或公网 URL。

先调用：

```text
POST /upload/image
```

然后生成引用：

```python
def image_reference(upload_response: dict) -> str:
    name = upload_response["name"]
    subfolder = upload_response.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name
```

下载策略、MIME 检查、大小限制、认证和存储隔离属于 API 服务职责。

## 14. 提取 Output Panel 结果

执行完成后：

```python
result = extract_history_results(
    history,
    manifest,
    prompt_id=prompt_id,
)
```

读取结果：

```python
latest = result["panels"]["2"]["latest"]
output_values = latest["outputs"]["latest_values"]
result_images = output_values["resultImage"]
```

提取器会按以下顺序兼容读取：

1. `document`
2. `document_json`
3. `schema_json`
4. 旧版 `params`

如果配置的 Output Panel 没有出现在 history 中，会抛出 `ValueError`。

## 15. 工作流注册器示例

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

注册时应同时发现普通与 API 导出的 manifest，并比较公开参数 ID、名称、类型、顺序、元数据、输出字段和 `PANEL_LINK` 配对关系。不一致时拒绝公开 endpoint。

## 16. 错误映射建议

- 请求类型或范围错误：`422 workflow_input_invalid`
- 工作流不存在：`404 workflow_not_found`
- 两种导出不一致：注册失败
- 契约指纹过期：`409 workflow_contract_changed`
- ComfyUI 不可用：`503 comfyui_unavailable`
- Prompt 被拒绝：`502 comfyui_prompt_rejected`
- 执行超时：`504 workflow_timeout`
- Output Panel 缺失：`502 workflow_output_missing`

不要向公网客户端返回 Python traceback、本地文件路径或未处理的内部 ComfyUI 错误。

## 17. 安全与运维检查

- 只注册相互匹配的普通/API 导出对。
- 启动时调用 `discover_manifest()` 校验文件。
- 保持契约指纹检查开启。
- 限制 JSON 大小、节点数、请求体和字符串长度。
- 对工作流执行和图片上传进行认证。
- 按请求或用户隔离上传文件。
- 拒绝任意服务器文件系统路径作为 IMAGE 值。
- 设置队列限制和执行超时。
- 保存每个请求自己的 `prompt_id`。
- 只从配置的 Output Panel ID 读取输出。
- 对输入输出文件执行保留和清理策略。

## 18. 发布工作流更新

修改以下内容后应重新导出两个文件：

- 参数顺序、稳定 ID、名称或类型；
- 默认值、范围或步长；
- SEED 默认模式；
- Output Panel 字段；
- `PANEL_LINK` 配对关系。

推荐流程：

1. 在 ComfyUI 中测试工作流。
2. 设置希望保存在导出中的当前值。
3. 保存普通 workflow。
4. 从同一图状态导出 API 格式。
5. 同时替换工作流包中的两个文件。
6. 发现并比较两个 manifest。
7. 重新加载工作流注册表。
8. 执行一次冒烟测试。

## 19. 模块职责边界

`workflow_contract.py` 不负责：

- 将普通 workflow 转换为 API prompt；
- 上传或下载图片；
- 发送 HTTP 请求；
- 管理 ComfyUI 队列；
- 等待执行完成；
- 认证用户；
- 生成特定框架路由；
- 删除临时文件；
- 根据选项列表验证 COMBO。

推荐的完整调用链是：

```text
前端请求
  ↓
现有认证和限流
  ↓
validate_call_inputs()
  ↓
现有图片上传逻辑
  ↓
patch_api_prompt()
  ↓
现有 /prompt 提交逻辑
  ↓
现有 WebSocket 或 history 等待逻辑
  ↓
extract_history_results()
  ↓
现有响应、存储和 CDN 逻辑
```

Gen2 契约层的目标不是替代整个 API 服务，而是让工作流作者在 ComfyUI 内声明业务边界，让服务端用一套通用代码自动发现、校验、替换和提取，从而移除最容易重复和失效的手工节点 ID 配置。
