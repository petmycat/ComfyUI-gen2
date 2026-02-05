# ComfyUI Custom Node Maintainer Guide

This document is intended for project maintainers and developers. It focuses on how to collaborate on GitHub and how to properly publish versions so that ComfyUI Manager can promptly receive updates.

---

## Table of Contents

1. [Recommended Development Practices](#1-recommended-development-practices)
2. [Version Publishing](#2-version-publishing)
3. [Handling External Contributions (PRs)](#3-handling-external-contributions-prs)
4. [Team Collaboration Guidelines](#4-team-collaboration-guidelines)

---

## 1. Recommended Development Practices

### Scenario 1: Non-Code Changes (README, Documentation, LICENSE, etc.)

**Characteristics**: Does not involve Python code logic; 100% safe with no impact on node functionality.

**Recommended Approach**: Edit directly on the main branch and push.

```bash
# Direct modification
git add README.md
git commit -m "docs: update installation instructions"
git push origin main
```

**Notes**:
- These changes are extremely low-risk and do not require a PR workflow
- Since `pyproject.toml` is not modified, no version release will be triggered
- Users who want these updates need to manually run `git pull` or reinstall

---

### Scenario 2: Code Changes (New Features, Bug Fixes, Refactoring)

**Characteristics**: Involves modifications to `.py` files; carries risk of introducing issues.

**Core Principle**:
> ⚠️ **No matter how confident you are, it is not recommended to develop code directly on the main branch.**
> 
> For personal projects, you can do as you wish. But in team collaboration, working directly on main can cause:
> - Other team members may pull incomplete code
> - Difficult to rollback when issues arise
> - Unable to trace "who changed what and why"

**Recommended Workflow: Branch Development → Push Branch → Merge via PR**

#### Step 1: Create a Feature Branch

```bash
git checkout -b feature/new-sampler
```

**Branch Naming Conventions**:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New feature development | `feature/add-lora-loader` |
| `fix/` | Bug fixes | `fix/memory-leak` |
| `refactor/` | Code refactoring | `refactor/sampler-logic` |

#### Step 2: Develop on the Branch

```bash
# Normal development with multiple commits
git add .
git commit -m "feat: add new sampler node"

# Continue development...
git add .
git commit -m "feat: improve parameter configuration"
```

#### Step 3: Push the Branch to GitHub

```bash
git push origin feature/new-sampler
```

**Note**: You are pushing the **branch**, not main. The main branch remains completely unaffected.

#### Step 4: Create a Pull Request on GitHub

1. Open the GitHub repository page
2. You will typically see a yellow banner: "feature/new-sampler had recent pushes"
3. Click **Compare & pull request**
4. Confirm: `base: main` ← `compare: feature/new-sampler`
5. Fill in the PR title and description
6. Click **Create pull request**

#### Step 5: Merge the PR

**Why not merge locally with `git merge` and push directly?**

| Method | Pros | Cons |
|--------|------|------|
| Local merge then push | Fast | No PR record, cannot trace discussions |
| **GitHub PR merge** | Complete record preserved, can trigger CI | Requires one extra step on the web |

**Recommended**: Complete the merge through GitHub web interface (or IDE tools). This way, the PR record will be permanently saved in the repository's Pull requests tab.

When merging, select **Squash and merge** (recommended) to compress multiple commits into one clean record.

#### Step 6: Clean Up Branches (Optional but Highly Recommended)

After the PR is merged into main, the development branch has completed its mission.

1. **Delete on GitHub web**: After merging the PR, the page usually prompts "Pull request successfully merged and closed. You can now safely delete the branch." Simply click **Delete branch** to remove the remote branch.
2. **Delete locally**:
   ```bash
   git checkout main
   git pull origin main           # Sync the latest main
   git branch -d feature/new-sampler  # Delete local branch
   ```

**What happens if you don't delete?**
- The repository will accumulate many invalid branches, appearing cluttered.
- **Note**: If you want to reuse an old branch for future development (not recommended), you must first run `git merge main` on the old branch to bring in the latest changes from main (then develop on this old branch, and finally go through Steps 1-5 again). Otherwise, serious conflicts will occur.

---

## 2. Version Publishing

### Publishing Workflow

When you have completed development and decide to release a new version:

#### Step 1: Update the Version Number

Modify the `version` field in `pyproject.toml`:

```toml
[project]
name = "gen2"
version = "1.1.0"  # Upgraded from 1.0.0 to 1.1.0
```

**Version Number Convention (Semantic Versioning - SemVer)**:

Format: `MAJOR.MINOR.PATCH` (e.g., `1.2.3`)

| Upgrade Type | When to Use | Example |
|--------------|-------------|---------|
| **MAJOR** | Breaking changes (incompatible API) | `1.0.0` → `2.0.0` |
| **MINOR** | New features, backward compatible | `1.0.0` → `1.1.0` |
| **PATCH** | Bug fixes only | `1.0.0` → `1.0.1` |

#### Step 2: Commit and Push

```bash
git add pyproject.toml
git commit -m "chore: release v1.1.0"
git push origin main
```

#### Step 3: Automatic Publishing Trigger

After pushing, the following process executes automatically:

1. **GitHub detects changes**: `pyproject.toml` is pushed to the main branch
2. **Action starts**: The task defined in `.github/workflows/publish.yml` begins running
3. **Login to Registry**: Uses the pre-configured `REGISTRY_ACCESS_TOKEN`
4. **Publish snapshot**: The current code (at this commit) is packaged and published to Comfy Registry
5. **Registry records**: The version number is bound to the Commit Hash

#### Step 4: Verify Publishing Results

1. **Check Action status**:
   - Go to the repository → Actions tab
   - Find the "Publish to Comfy registry" task
   - Confirm it shows a green checkmark ✅

2. **Confirm Registry update**:
   - Visit `https://registry.comfy.org/nodes/your-node-name`
   - Check if the version number is displayed correctly

3. **Wait for Manager sync**:
   - ComfyUI Manager has a caching mechanism; it usually takes several hours to see updates
   - If Registry shows correctly, the publishing was successful

---

### Core Publishing Mechanism

#### Data Flow

```
GitHub Repository  ──(Action Trigger)──>  Comfy Registry  ──(Sync)──>  ComfyUI Manager
```

#### Relevant Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Node metadata, **contains version number**, core trigger for publishing |
| `.github/workflows/publish.yml` | GitHub Action script, **monitors changes to `pyproject.toml`** |

#### ⚠️ Key Understanding: Correspondence Between Version Number and Commit Hash

This is where many developers get confused. Please understand the following mechanism:

**When you modify the version number in `pyproject.toml` and push to the main branch:**

1. GitHub Action detects that `pyproject.toml` has changed
2. Action executes the publishing script, publishing **the code snapshot at that exact moment** to Comfy Registry
3. Registry records the **specific Commit Hash** corresponding to that version number

**What does this mean?**

Suppose you performed the following operations:

```
Timeline:
├── Commit A: Added new feature code
├── Commit B: Fixed a bug
├── Commit C: Modified pyproject.toml, version changed to 1.0.0  ← Triggers publishing
├── Commit D: Made more code changes
├── Commit E: Continued development...
└── (Did not modify pyproject.toml again)
```

**Result**:
- The `1.0.0` version downloaded by ComfyUI Manager corresponds to the code snapshot at **Commit C**
- Changes from Commit D and E **will not** be received by users unless you modify `pyproject.toml` again to release a new version

**Verification Method**:
- In the GitHub repository's Actions tab, find the corresponding publishing task
- Click to see the Commit Hash when that task ran
- This Hash is the exact code version that users receive when installing through Manager

**Conclusion**:
> **Publishing = Modify `pyproject.toml` version number + Push to main**
> 
> Only at the moment you modify the version number and push, the code is "frozen" and released to users. Any commits after that require another release for users to receive them.

---

## 3. Handling External Contributions (PRs)

### Review Process

1. **Understand the purpose**: Read the PR description
2. **Review code**: Click `Files changed`, check line by line
3. **Verify functionality**: If necessary, pull to local and test
4. **Communicate feedback**: Discuss suggestions in the comments
5. **Make a decision**:
   - Merge: Click `Merge pull request`
   - Reject: Explain the reason then click `Close pull request`

### Common PR Types and Handling Suggestions

| PR Type | Handling Suggestion |
|---------|---------------------|
| **Comfy-Org official** (e.g., adding pyproject.toml) | Read carefully; usually a standardization invitation |
| **Bug fix** | Verify the issue exists, test the fix |
| **New feature** | Evaluate if it fits project direction and code quality |
| **Documentation improvement** | Check accuracy then merge quickly |

### Polite Response After Merging

After merging, a simple thank you is appreciated:
> "Thanks for the contribution!"

---

## 4. Team Collaboration Guidelines

### Role Division

| Role | Permissions | Responsibilities |
|------|-------------|------------------|
| **Owner** | Full control | Manage repository settings, Secrets, member permissions |
| **Maintainer** | Merge PRs, push code | Review code, merge contributions, publish versions |
| **Contributor** | Submit PRs | Contribute code, fix bugs |

### Branch Management Strategy

```
main (main branch) ─────────────────────────────────────────>
     │                    │                    │
     ├── feature/xxx ─────┤                    │
     │   (feature dev)    │                    │
     │                    PR merge             │
     │                                         │
     └── fix/xxx ──────────────────────────────┤
         (bug fix)                             PR merge
```

- **main branch**: Keep stable, ready to release at any time
- **feature/fix branches**: Checkout from main, merge back to main via PR
- **Forbidden**: Large-scale development directly on main

### Code Review

When receiving a PR:

1. **View changes**: Click the `Files changed` tab
2. **Review line by line**: Click `+` on lines with questions to add comments
3. **Submit review result**: Click `Review changes`, select:
   - **Approve**: Code is fine, can be merged
   - **Request changes**: Needs modification before merging
   - **Comment**: Discussion only, no stance

### Merge Strategy Selection

Click the dropdown arrow next to `Merge pull request`:

| Strategy | Effect | Recommended Scenario |
|----------|--------|---------------------|
| **Squash and merge** | Compress multiple commits into one | ⭐ Recommended! Keeps history clean |
| Create a merge commit | Preserve all original commits | When complete history is needed |
| Rebase and merge | Linear history | Advanced users |

---

## Appendix: Key File Examples

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
      - "pyproject.toml"  # Only triggers when this file changes

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

## Quick Reference Card

| What I Want to Do | How to Do It | Triggers Publishing? |
|-------------------|--------------|---------------------|
| Edit README/docs | Commit & push directly to main | ❌ |
| Develop new feature | Create branch → Develop → Push branch → PR → Merge | ❌ |
| **Release new version** | **Modify `pyproject.toml` version → push to main** | ✅ |

---

*Last updated: Feb 2026*
