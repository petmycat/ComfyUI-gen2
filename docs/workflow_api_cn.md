# Gen2 工作流 API 集成指南

本文面向需要接入含有 `Gen2_InputPanel` 和 `Gen2_OutputPanel` 的 ComfyUI 导出工作流的 API 服务端开发者。

`api_nodes/workflow_contract.py` 提供一层稳定的 JSON 契约，用于发现参数、校验请求、替换 API prompt 中的运行值，以及从 ComfyUI history 中提取结果。

英文版本：[`workflow_api_en.md`](workflow_api_en.md)

## 1. 同时保留两种 ComfyUI 导出

ComfyUI 有两种职责不同的导出格式：

| 文件 | 用途 |
|---|---|
| 普通 workflow | 在 ComfyUI UI 中重新打开、记录、编辑和复现工作流图 |
| API 格式 workflow | 作为可执行 prompt 提交给 ComfyUI `/prompt` |

每个发布的工作流都应同时保存两个文件：

```text
workflows/
└── product-image/
    ├── workflow.json
    └── workflow_api.json
```

不要把普通 workflow 提交给 `/prompt`，也不要用 API 格式文件替代可编辑的普通 workflow。

一个可发布的工作流图应包含：

1. 一个或多个 `Gen2_InputPanel` 节点。
2. 一个或多个 `Gen2_OutputPanel` 节点。
3. 从对应 Input Panel 到每个 Output Panel 的 `PANEL_LINK`。
4. Input Panel 参数连接到它们所控制的工作流输入。
5. 工作流结果连接到对应的 Output Panel 输入。

两个文件必须从同一个已经测试通过的工作流图状态导出。

## 2. 公共函数

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

该模块只负责 JSON 与参数契约处理，不会执行节点、上传图片、发送 HTTP 请求或管理 ComfyUI 队列。

`workflow_contract.py` 与 `_config.py` 共用参数校验规则，因此服务端必须使用来自同一插件版本的这两个文件。

## 3. 加载并检查导出文件

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

`detect_format()` 遇到损坏、混合或不支持的结构时会报错，而不是静默猜测格式。

`discover_manifest()` 会校验面板配置并生成派生 manifest，不会修改原始导出对象。

## 4. Manifest 结构

Manifest 的基本结构如下：

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

重要字段：

- `id`：参数的稳定身份标识。
- `name`：对外公开的 API 字段名。
- `type`：请求或响应值类型。
- `default`：在 Input Panel 配置对话框中设置的默认值。
- `current_value`：导出文件中保存的当前值，也就是导出时的节点值。
- `min`、`max`、`step`：数值校验元数据。
- `binding`：该参数在 API prompt 中对应的可替换输入位置。
- `output_panels[].node_id`：从 ComfyUI history 读取结果的位置。
- `contract_fingerprint`：防止使用过期 manifest 的契约指纹。

普通 workflow 中的 binding 会标记为 `patchable: false`。普通 workflow 可以被检查，但不能直接作为 API prompt 执行。

## 5. 检查两种导出是否一致

两个文件应公开相同的稳定参数 ID、名称、类型、顺序、元数据、输出字段和面板配对关系。

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
    raise ValueError("普通导出与 API 导出公开了不同的面板契约")
```

如果二者不一致，应使工作流注册失败。正确做法是从同一个工作流图重新导出两个文件，而不是手工编辑 JSON 来凑齐字段。

## 6. 输入类型

| 类型 | API 可接受的值 |
|---|---|
| `STRING` | JSON 字符串；允许空字符串 |
| `COMBO` | 非空 JSON 字符串 |
| `BOOLEAN` | 只能是 JSON `true` 或 `false` |
| `INT` | 范围内且符合 `step` 对齐规则的整数 |
| `FLOAT` | 范围内且符合 `step` 对齐规则的有限数值 |
| `SEED` | 配置的安全范围内且符合 `step` 的整数 |
| `IMAGE` | 非空的 ComfyUI 图片引用字符串 |

当前 COMBO 契约只将其校验为字符串，因为 schema 中暂时没有包含可选项列表。

## 7. 校验请求输入

只有一个 Input Panel，或者多个面板中的参数名全局唯一时，可以使用扁平请求：

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

返回值会被标准化为按面板分组的映射：

```json
{
  "1": {
    "prompt": "a product photo on a white background",
    "seed": 12345,
    "inputImage": "requests/request-123/source.png"
  }
}
```

如果多个面板使用了相同参数名，请求必须显式按面板分组：

```json
{
  "1": {"seed": 12345},
  "7": {"seed": 67890}
}
```

空请求也是合法的：

```json
{}
```

它表示直接使用 API 导出中保存的全部 `current_value` 执行。若只提供一个覆盖值，其他未提供字段仍保留导出时的当前值，而不会重置为配置默认值。

服务端应将校验产生的 `ValueError` 转换为 HTTP `400` 或 `422` 响应。

## 8. 替换 API Prompt 中的运行值

常规的一次性处理可使用：

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

如果注册工作流时已经缓存 manifest：

```python
patched_prompt = patch_api_prompt(
    api_prompt,
    cached_manifest,
    request_body,
)
```

`patch_api_prompt()` 会：

1. 要求源文件必须是 API 格式导出；
2. 重新发现其当前契约；
3. 检查契约指纹；
4. 校验请求值；
5. 深拷贝 prompt；
6. 只修改 Input Panel 的运行字段；
7. 保留 `_config`、连接、节点 ID、元数据和其他全部节点。

原始对象不会被修改。如果指纹过期，必须重新加载或注册工作流，不能忽略该错误。

## 9. 上传 IMAGE 输入

IMAGE 参数是 ComfyUI 文件引用，不是图片原始字节，也不是公网 URL。

应先上传文件：

```text
POST /upload/image
```

然后根据响应生成引用：

```python
def image_reference(upload_response: dict) -> str:
    name = upload_response["name"]
    subfolder = upload_response.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name
```

API 服务端仍需负责 URL 下载策略、MIME 检查、文件大小限制、身份认证和上传目录隔离。

## 10. 提交到 ComfyUI

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

必须将返回的 `prompt_id` 与当前 API 请求关联。存在并发任务时，绝对不要使用全局“最新 history”来猜测任务结果。

## 11. 等待执行完成

轮询示例：

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

吞吐量较高时，建议使用 ComfyUI WebSocket 事件判断完成状态，再读取最终 history。

## 12. 提取已配置的输出

```python
history = wait_for_history(COMFY_URL, prompt_id)

result = extract_history_results(
    history,
    manifest,
    prompt_id=prompt_id,
)
```

结果结构：

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

读取已配置输出值：

```python
latest = result["panels"]["2"]["latest"]
output_values = latest["outputs"]["latest_values"]
```

提取器会依次读取 `document`、`document_json`、`schema_json` 和旧版 `params`。如果 history 中缺少 manifest 指定的 Output Panel，则会抛出 `ValueError`。

## 13. 生成图片下载 URL

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

不要将与部署环境绑定的 ComfyUI URL 写回工作流或 manifest。

## 14. 最小工作流注册器

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

目录注册逻辑可以加载 `workflow.json` 与 `workflow_api.json`、比较二者的契约视图，并且只公开匹配且有效的工作流包。

建议路由：

```text
GET  /api/v1/workflows/{workflow_id}
POST /api/v1/workflows/{workflow_id}/runs
GET  /api/v1/runs/{run_id}
```

服务端可以根据 manifest 自动生成请求 schema、OpenAPI 元数据、WebUI 控件、上传要求和输出说明。

## 15. 错误映射

| 情况 | 建议响应 |
|---|---|
| 请求类型或范围错误 | `422 workflow_input_invalid` |
| 工作流不存在 | `404 workflow_not_found` |
| 两种导出的契约不一致 | 注册失败，不公开接口 |
| 契约指纹过期 | `409 workflow_contract_changed` |
| ComfyUI 不可用 | `503 comfyui_unavailable` |
| Prompt 被拒绝 | `502 comfyui_prompt_rejected` |
| 执行超时 | `504 workflow_timeout` |
| Output Panel 缺失 | `502 workflow_output_missing` |

不要向公网客户端返回 Python traceback、本地文件路径或未处理的内部 ComfyUI 错误。

## 16. 安全与运维检查表

- 只注册相互匹配的普通/API 导出对。
- 启动时校验两个文件。
- 保持契约指纹检查开启。
- 限制 JSON 大小、节点数、请求大小和字符串长度。
- 对工作流执行和图片上传进行身份认证。
- 按请求或用户隔离上传文件。
- 拒绝任意服务器文件系统路径。
- 设置队列限制和执行超时。
- 保存每个请求自己的 `prompt_id`。
- 只从配置的 Output Panel ID 读取输出。
- 对输入输出文件应用保留和清理策略。

## 17. 发布工作流更新

修改以下内容时必须重新导出两个文件：

- 参数顺序或稳定 ID；
- 名称或类型；
- 默认值、范围或步长；
- SEED 默认模式；
- Output Panel 字段；
- `PANEL_LINK` 配对关系。

建议发布流程：

1. 在 ComfyUI 中测试工作流图。
2. 设置期望的当前值。
3. 保存普通 workflow。
4. 从相同图状态导出 API 格式。
5. 同时替换工作流包中的两个文件。
6. 发现并比较两个 manifest。
7. 重新加载工作流注册表。
8. 执行一次冒烟测试。

## 18. 模块职责边界

`workflow_contract.py` 不负责：

- 将普通 workflow 转换成 API prompt；
- 上传或下载图片；
- 发送 HTTP 请求；
- 管理 ComfyUI 队列；
- 等待工作流执行；
- 认证用户；
- 生成某个服务端框架专用的路由；
- 删除临时文件；
- 根据选项列表校验 COMBO 值。

这些工作属于外围 API 服务。该模块只提供 Gen2 面板声明、ComfyUI 原生导出、API 请求值和 history 结果之间的稳定桥接层。
