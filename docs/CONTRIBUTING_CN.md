# ComfyUI 自定义节点维护指南

本文档面向项目维护者和开发者，重点讲解如何在 GitHub 上进行协作开发，以及如何正确发布版本让 ComfyUI Manager 能够及时获取更新。

---

## 目录

1. [开发场景的推荐实践](#1-开发场景的推荐实践)
2. [版本发布](#2-版本发布)
3. [处理外部贡献 (PR)](#3-处理外部贡献-pr)
4. [多人协作规范](#4-多人协作规范)

---

## 1. 开发场景的推荐实践

### 场景一：非代码改动（README、文档、LICENSE 等）

**特点**：不涉及 Python 代码逻辑，100% 不会影响节点功能。

**推荐做法**：直接在 main 分支修改并推送。

```bash
# 直接修改文件
git add README.md
git commit -m "docs: 更新安装说明"
git push origin main
```

**说明**：
- 这种改动风险极低，不需要走 PR 流程
- 由于没有修改 `pyproject.toml`，不会触发版本发布
- 用户如果想获取这些更新，需要手动 `git pull` 或重新安装

---

### 场景二：代码改动（新功能、Bug 修复、重构）

**特点**：涉及 `.py` 文件的修改，有引入问题的风险。

**核心原则**：
> ⚠️ **无论多有自信，都不建议直接在 main 分支上开发代码。**
> 
> 个人项目可以随意，但多人协作时，直接在 main 上改动会导致：
> - 其他人 pull 时可能拉到半成品代码
> - 出问题时难以回滚
> - 无法追溯"谁改了什么、为什么改"

**推荐流程：分支开发 → 推送分支 → 通过 PR 合并**

#### 第一步：创建功能分支

```bash
git checkout -b feature/new-sampler
```

**分支命名建议**：

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feature/` | 新功能开发 | `feature/add-lora-loader` |
| `fix/` | Bug 修复 | `fix/memory-leak` |
| `refactor/` | 代码重构 | `refactor/sampler-logic` |

#### 第二步：在分支上开发

```bash
# 正常开发，多次提交
git add .
git commit -m "feat: 添加新采样器节点"

# 继续开发...
git add .
git commit -m "feat: 完善参数配置"
```

#### 第三步：推送分支到 GitHub

```bash
git push origin feature/new-sampler
```

**注意**：这里是推送**分支**，不是 main。此时 main 分支完全不受影响。

#### 第四步：在 GitHub 上创建 Pull Request

1. 打开 GitHub 仓库页面
2. 通常会看到黄色提示条："feature/new-sampler had recent pushes"
3. 点击 **Compare & pull request**
4. 确认：`base: main` ← `compare: feature/new-sampler`
5. 填写 PR 标题和描述
6. 点击 **Create pull request**

#### 第五步：合并 PR

**为什么不在本地 `git merge` 后直接推送？**

| 方式 | 优点 | 缺点 |
|------|------|------|
| 本地 merge 后 push | 快 | 没有 PR 记录，无法追溯讨论 |
| **GitHub PR 合并** | 留下完整记录、可触发 CI | 需要多一步网页操作 |

**推荐**：通过 GitHub 网页（或 IDE 工具）完成合并。这样 PR 记录会永久保存在仓库的 Pull requests 标签页中。

合并时选择 **Squash and merge**（推荐），将多个提交压缩成一个干净的记录。

#### 第六步：清理分支（可选但强烈建议）

当 PR 合并到 main 后，该开发分支的使命就完成了。

1. **GitHub 网页删除**：合并 PR 后，页面通常会提示 "Pull request successfully merged and closed. You can now safely delete the branch."，直接点击 **Delete branch** 即可删除远程分支。
2. **本地删除**：
   ```bash
   git checkout main
   git pull origin main           # 同步最新的 main
   git branch -d feature/new-sampler  # 删除本地分支
   ```

**如果不删除会怎样？**
- 仓库会堆积大量无效分支，显得混乱。
- **注意**：如果下次开发想复用旧分支（不推荐），必须先在旧分支上执行 `git merge main` 将主线的最新改动平移过来（之后在这个旧分支上开发，最后再来一次上面的第一到第五步），否则会产生严重的冲突。

---

## 2. 版本发布

### 版本发布流程

当你完成开发并决定发布新版本时：

#### 第一步：更新版本号

修改 `pyproject.toml` 中的 `version` 字段：

```toml
[project]
name = "gen2"
version = "1.1.0"  # 从 1.0.0 升级到 1.1.0
```

**版本号规范（语义化版本 SemVer）**：

格式：`主版本.次版本.修订号`（如 `1.2.3`）

| 升级类型 | 何时使用 | 示例 |
|----------|----------|------|
| **主版本** | 有破坏性变更（API 不兼容） | `1.0.0` → `2.0.0` |
| **次版本** | 新增功能，向后兼容 | `1.0.0` → `1.1.0` |
| **修订号** | 仅修复 Bug | `1.0.0` → `1.0.1` |

#### 第二步：提交并推送

```bash
git add pyproject.toml
git commit -m "chore: release v1.1.0"
git push origin main
```

#### 第三步：自动发布触发

推送后，以下流程自动执行：

1. **GitHub 检测变动**：`pyproject.toml` 被推送到 main 分支
2. **Action 启动**：`.github/workflows/publish.yml` 定义的任务开始运行
3. **登录 Registry**：使用预先配置的 `REGISTRY_ACCESS_TOKEN`
4. **发布快照**：当前代码（此刻的 Commit）被打包发布到 Comfy Registry
5. **Registry 记录**：版本号与 Commit Hash 绑定

#### 第四步：验证发布结果

1. **查看 Action 状态**：
   - 进入仓库 → Actions 标签页
   - 找到 "Publish to Comfy registry" 任务
   - 确认显示绿色勾 ✅

2. **确认 Registry 更新**：
   - 访问 `https://registry.comfy.org/nodes/你的节点名`
   - 检查版本号是否正确显示

3. **等待 Manager 同步**：
   - ComfyUI Manager 有缓存机制，通常需要数小时才能看到更新
   - 如果 Registry 显示正确，就说明发布成功

---

### 版本发布的核心机制

#### 数据流向

```
GitHub 仓库  ──(Action 触发)──>  Comfy Registry  ──(同步)──>  ComfyUI Manager
```

#### 直接相关的文件

| 文件 | 作用 |
|------|------|
| `pyproject.toml` | 节点的元信息，**包含版本号**，是触发发布的核心 |
| `.github/workflows/publish.yml` | GitHub Action 脚本，**监听 `pyproject.toml` 的变动** |

#### ⚠️ 关键理解：版本号与 Commit Hash 的对应关系

这是很多开发者容易混淆的地方。请务必理解以下机制：

**当你修改 `pyproject.toml` 中的版本号并推送到 main 分支时：**

1. GitHub Action 检测到 `pyproject.toml` 发生变动
2. Action 执行发布脚本，将**当前这一刻的代码快照**发布到 Comfy Registry
3. Registry 记录下这个版本号对应的**具体 Commit Hash**

**这意味着什么？**

假设你执行了以下操作：

```
时间线：
├── Commit A: 新增功能代码
├── Commit B: 修复 Bug
├── Commit C: 修改 pyproject.toml，版本号改为 1.0.0  ← 触发发布
├── Commit D: 又改了一些代码
├── Commit E: 继续开发...
└── (没有再改 pyproject.toml)
```

**结果**：
- ComfyUI Manager 下载的 `1.0.0` 版本，对应的是 **Commit C** 的代码快照
- Commit D、E 的改动**不会**被用户获取到，除非你再次修改 `pyproject.toml` 发布新版本

**验证方法**：
- 在 GitHub 仓库的 Actions 标签页，找到对应的发布任务
- 点击进入可以看到该任务运行时的 Commit Hash
- 这个 Hash 就是用户通过 Manager 安装时实际获取的代码版本

**结论**：
> **发布 = 修改 `pyproject.toml` 版本号 + 推送到 main**
> 
> 只有在你修改版本号并推送的那一刻，代码才会被"定格"并发布给用户。在此之后的任何提交，都需要再次发布才能让用户获取。

---

## 3. 处理外部贡献 (PR)

### 审查流程

1. **理解改动目的**：阅读 PR 描述
2. **查看代码**：点击 `Files changed`，逐行检查
3. **验证功能**：如有必要，拉取到本地测试
4. **沟通反馈**：在评论区讨论改进建议
5. **做出决定**：
   - 合并：点击 `Merge pull request`
   - 拒绝：说明原因后点击 `Close pull request`

### 常见 PR 类型处理建议

| PR 类型 | 处理建议 |
|---------|----------|
| **Comfy-Org 官方**（如添加 pyproject.toml） | 仔细阅读说明，通常是标准化邀请 |
| **Bug 修复** | 验证问题是否确实存在，测试修复效果 |
| **新功能** | 评估是否符合项目方向，代码质量如何 |
| **文档改进** | 检查准确性后可快速合并 |

### 合并后的礼貌回复

合并完成后，可以简单回复一句感谢：
> "Thanks for the contribution!"

---

## 4. 多人协作规范

### 角色划分

| 角色 | 权限 | 职责 |
|------|------|------|
| **Owner** | 完全控制 | 管理仓库设置、Secrets、成员权限 |
| **Maintainer** | 合并 PR、推送代码 | 审查代码、合并贡献、发布版本 |
| **Contributor** | 提交 PR | 贡献代码、修复 Bug |

### 分支管理策略

```
main (主分支) ─────────────────────────────────────────>
     │                    │                    │
     ├── feature/xxx ─────┤                    │
     │   (功能开发)        │                    │
     │                    PR 合并               │
     │                                         │
     └── fix/xxx ──────────────────────────────┤
         (Bug 修复)                            PR 合并
```

- **main 分支**：保持稳定，随时可发布
- **功能/修复分支**：从 main 检出，通过 PR 合并回 main
- **严禁**：在 main 上直接进行大规模开发

### 代码审查 (Code Review)

当收到 PR 时：

1. **查看改动**：点击 `Files changed` 标签页
2. **逐行审查**：在有疑问的代码行点击 `+` 添加评论
3. **提交审查结果**：点击 `Review changes`，选择：
   - **Approve**：代码没问题，可以合并
   - **Request changes**：需要修改后才能合并
   - **Comment**：仅讨论，不表态

### 合并策略选择

点击 `Merge pull request` 旁边的下拉箭头：

| 策略 | 效果 | 推荐场景 |
|------|------|----------|
| **Squash and merge** | 多个提交压缩成一个 | ⭐ 推荐！保持历史清晰 |
| Create a merge commit | 保留所有原始提交 | 需要完整历史记录时 |
| Rebase and merge | 线性历史 | 高级用户 |

---

## 附录：关键文件示例

### pyproject.toml

```toml
[project]
name = "gen2"
description = "Custom ComfyUI nodes for QwenImage ControlNet..."
version = "1.0.0"
license = {file = "LICENSE"}

[project.urls]
Repository = "https://github.com/petmycat/ComfyUI-gen2"

[tool.comfy]
PublisherId = "petmycat"
DisplayName = "ComfyUI-gen2"
```

### .github/workflows/publish.yml

```yaml
name: Publish to Comfy registry
on:
  push:
    branches:
      - main
    paths:
      - "pyproject.toml"  # 只有这个文件变动才触发

jobs:
  publish-node:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: Comfy-Org/publish-node-action@v1
        with:
          personal_access_token: ${{ secrets.REGISTRY_ACCESS_TOKEN }}
```

---

## 快速参考卡片

| 我要做什么 | 怎么做 | 是否触发发布 |
|------------|--------|-------------|
| 改 README/文档 | 直接 commit & push 到 main | ❌ |
| 开发新功能 | 创建分支 → 开发 → 推送分支 → PR → 合并 | ❌ |
| **发布新版本** | **修改 `pyproject.toml` 版本号 → push 到 main** | ✅ |

---

*最后更新：2026.2*
