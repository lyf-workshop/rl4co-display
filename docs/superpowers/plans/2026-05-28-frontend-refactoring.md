# Frontend Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 HTML 重复代码，将内联 CSS/JS 提取到独立文件，建立 Jinja2 模板继承，使所有页面共用统一骨架。

**Architecture:** 新建 `templates/base.html` 作为公共骨架，所有页面通过 `{% extends "base.html" %}` + `{% block %}` 继承它。内联 `<style>` 提取为 `static/css/<page>.css`，内联 `<script>` 提取为 `static/js/<page>.js`。删除废弃文件。

**Tech Stack:** Flask/Jinja2 模板继承、HTML/CSS/JS（无构建工具，纯静态）

---

## 阶段概览

三个阶段各自独立，每阶段完成后 app 完全可用，可随时停止。

| 阶段 | 内容 | 收益 |
|---|---|---|
| 1 | 创建 base.html，所有页面改为继承 | 消除 9 个页面重复的 `<head>` + 导航栏 |
| 2 | 提取内联 CSS 到 .css 文件 | index.html 从 2867 行缩减约 500 行 |
| 3 | 提取内联 JS 到 .js 文件 + 清理废弃文件 | 消除剩余内联脚本，删除无用文件 |

---

## 文件清单

### 阶段 1 新增/修改
- **新增：** `templates/base.html`
- **修改：** `templates/includes/navbar.html` — 修正 class 名与 navigation.css 一致
- **修改：** `templates/index.html`
- **修改：** `templates/benchmark.html`
- **修改：** `templates/model_info.html`
- **修改：** `templates/file_manager.html`
- **修改：** `templates/profile.html`
- **修改：** `templates/history.html`
- **修改：** `templates/res.html`
- **修改：** `templates/login.html`
- **修改：** `templates/register.html`

### 阶段 2 新增/修改
- **新增：** `static/css/base.css` — 全局 reset + CSS 变量
- **新增：** `static/css/index.css` — 从 index.html 提取
- **新增：** `static/css/benchmark.css` — 从 benchmark.html 提取
- **新增：** `static/css/model_info.css` — 从 model_info.html 提取
- **新增：** `static/css/file_manager.css` — 从 file_manager.html 提取
- **新增：** `static/css/profile.css` — 从 profile.html 提取
- **新增：** `static/css/auth.css` — 从 login.html / register.html 提取
- **修改：** 上述所有 HTML 模板 — 删 `<style>` 块，在 `{% block head %}` 加 `<link>`

### 阶段 3 新增/修改/删除
- **新增：** `static/js/index.js` — 训练表单交互、参数动态显示、兼容性校验
- **新增：** `static/js/training-progress.js` — SSE 进度流逻辑
- **新增：** `static/js/benchmark.js` — 图表渲染 + 筛选逻辑
- **修改：** `templates/index.html`、`templates/benchmark.html` — 删内联 `<script>`
- **删除：** `templates/FRONTEND_ALGORITHM_SELECTOR.html`
- **删除：** `static/css/ollama-chat-embedded.css`（已被 enhanced 版取代）
- **删除：** 根目录下 `validation_results_*.csv/json`（共 4 个文件）

---

## 阶段 1：模板继承

### Task 1：修复 navbar include + 创建 base.html

**Files:**
- Modify: `templates/includes/navbar.html`
- Create: `templates/base.html`

- [ ] **Step 1：更新 navbar include 的 class 名**

当前 `includes/navbar.html` 用 `.page-header`/`.page-nav`，而 `navigation.css` 期望 `.header`/`.nav`。将文件改为：

```html
<header class="header">
    <nav class="nav">
        <div class="logo-section">
            <div class="university-badge">🎓 山西大学 · RL4CO</div>
        </div>
        <div class="nav-links">
            <a href="/" {% if active_page == 'home' %}class="active"{% endif %}>🏠 训练首页</a>
            <a href="/benchmark" {% if active_page == 'benchmark' %}class="active"{% endif %}>📊 算法对比</a>
            <a href="/model_info" {% if active_page == 'model_info' %}class="active"{% endif %}>📚 模型知识库</a>
            <a href="/file_manager" {% if active_page == 'file_manager' %}class="active"{% endif %}>🗂️ 文件管理</a>
            <a href="/profile" {% if active_page == 'profile' %}class="active"{% endif %}>👤 我的账户</a>
        </div>
    </nav>
</header>
```

- [ ] **Step 2：创建 base.html**

```html
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}RL4CO 平台 | 山西大学{% endblock %}</title>
    <link rel="stylesheet" href="/static/css/navigation.css">
    {% block head %}{% endblock %}
</head>
<body>
    {% block nav %}
        {% include "includes/navbar.html" %}
    {% endblock %}
    {% block content %}{% endblock %}
    <script src="/static/js/navigation.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
```

设计说明：
- `{% block nav %}` 默认包含导航栏；login/register 页面覆写为空即可
- `{% block head %}` 放页面独有的 CSS link 和临时内联 style（阶段 2 再提取）
- `{% block scripts %}` 放页面独有 JS 和临时内联 script（阶段 3 再提取）
- `navigation.js` 放在 base.html 的 `</body>` 前，所有页面都自动获得

- [ ] **Step 3：启动 app 验证 base.html 语法**

```bash
python app.py
```

预期：Flask 启动无报错。此时 base.html 尚未被任何页面使用，不影响现有功能。

---

### Task 2：转换 index.html

**Files:**
- Modify: `templates/index.html`

- [ ] **Step 1：找出 index.html 的结构边界**

读取 index.html，确认：
- `<head>` 内容范围：第 3 行到第 495 行（含 `</head>-->` 结构 bug）
- `<body>` 开始：第 496 行
- 导航栏：第 498–511 行（`<header class="header">...</header>`）
- 页面主体：第 513 行起
- 底部脚本：最后几行（navigation.js + ollama-chat-embedded.js）
- `</body></html>`：最后两行

- [ ] **Step 2：改写 index.html**

将整个文件替换为继承结构。保留所有内联 `<style>` 和 `<script>` 内容（阶段 2/3 再提取），只做结构迁移：

```html
{% extends "base.html" %}

{% block title %}RL4CO - 强化学习优化平台 | 山西大学{% endblock %}

{% block head %}
<link rel="stylesheet" href="/static/css/ollama-chat-embedded-enhanced.css">
<style>
  /* ← 将原 index.html 第 12–493 行的完整 <style> 内容粘贴到这里 */
</style>
{% endblock %}

{% block content %}
<!-- ← 将原 index.html 第 513 行（导航栏之后）到倒数第 4 行之前的全部内容粘贴到这里 -->
<!-- 即 <div class="main-container-with-chat"> ... </div> 整段 -->
{% endblock %}

{% block scripts %}
<script>
  /* ← 将原 index.html 最后一个 </script> 前的所有内联 JS 粘贴到这里 */
</script>
<script src="/static/js/ollama-chat-embedded.js"></script>
{% endblock %}
```

操作步骤：
1. 打开原 index.html
2. 复制第 12–493 行内容，粘贴到 `{% block head %}` 的 `<style>` 中
3. 复制第 513 行到结尾前导航栏之后、底部脚本之前的内容，粘贴到 `{% block content %}`
4. 复制最后的内联 `<script>` 内容，粘贴到 `{% block scripts %}`
5. **不要保留** 原文件的 `<!DOCTYPE>`、`<html>`、`<head>`、`<body>`、导航栏 HTML——这些由 base.html 提供

- [ ] **Step 3：验证 index.html 转换**

```bash
python app.py
```

访问 `http://localhost:5000/`，检查：
- [ ] 页面正常渲染，导航栏可见
- [ ] 训练表单可以操作
- [ ] 浏览器控制台无 JS 错误（F12 → Console）
- [ ] AI 助手面板正常显示

---

### Task 3：转换 benchmark.html

**Files:**
- Modify: `templates/benchmark.html`

- [ ] **Step 1：改写 benchmark.html**

benchmark.html 额外引入了 echarts CDN，放到 `{% block head %}` 里：

```html
{% extends "base.html" %}

{% block title %}算法性能对比 - RL4CO 平台 | 山西大学{% endblock %}

{% block head %}
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
<style>
  /* ← 原 benchmark.html <style>...</style> 内容 */
</style>
{% endblock %}

{% block content %}
<!-- ← 原 benchmark.html <body> 内、导航栏之后到底部脚本之前的全部内容 -->
{% endblock %}

{% block scripts %}
<script>
  /* ← 原 benchmark.html 内联 <script> 内容 */
</script>
{% endblock %}
```

- [ ] **Step 2：验证**

访问 `http://localhost:5000/benchmark`，检查：
- [ ] 图表正常渲染
- [ ] 算法类别筛选下拉生效
- [ ] 浏览器控制台无错误

---

### Task 4：转换其余页面（model_info / file_manager / profile / history / res）

**Files:**
- Modify: `templates/model_info.html`
- Modify: `templates/file_manager.html`
- Modify: `templates/profile.html`
- Modify: `templates/history.html`
- Modify: `templates/res.html`

对每个文件执行相同操作，模板如下（以 model_info.html 为例）：

- [ ] **Step 1：转换 model_info.html**

```html
{% extends "base.html" %}

{% block title %}模型知识库 - RL4CO 平台 | 山西大学{% endblock %}

{% block head %}
<style>
  /* ← 原 model_info.html <style> 内容 */
</style>
{% endblock %}

{% block content %}
<!-- ← 原 model_info.html <body> 内导航栏之后的全部内容 -->
{% endblock %}

{% block scripts %}
<script>
  /* ← 原 model_info.html 内联 <script> 内容（若有） */
</script>
{% endblock %}
```

- [ ] **Step 2：按相同模式转换 file_manager.html、profile.html、history.html、res.html**

- [ ] **Step 3：逐页验证**

```bash
python app.py
```

访问以下路由，每页检查渲染正常 + 控制台无错误：
- `http://localhost:5000/model_info`
- `http://localhost:5000/file_manager`
- `http://localhost:5000/profile`
- `http://localhost:5000/history`（若有路由）
- `http://localhost:5000/res`（若有路由）

---

### Task 5：转换 login / register（无导航栏页面）

**Files:**
- Modify: `templates/login.html`
- Modify: `templates/register.html`

login/register 没有导航栏，覆写 `{% block nav %}` 为空：

- [ ] **Step 1：转换 login.html**

```html
{% extends "base.html" %}

{% block title %}登录 - RL4CO 强化学习平台 | 山西大学{% endblock %}

{% block nav %}{% endblock %}

{% block head %}
<style>
  /* ← 原 login.html <style> 内容 */
</style>
{% endblock %}

{% block content %}
<!-- ← 原 login.html <body> 内的全部内容 -->
{% endblock %}

{% block scripts %}
<script>
  /* ← 原 login.html 内联 <script> 内容（若有） */
</script>
{% endblock %}
```

- [ ] **Step 2：按相同模式转换 register.html**

- [ ] **Step 3：验证**

访问 `http://localhost:5000/login`、`http://localhost:5000/register`，检查：
- [ ] 页面居中登录框正常显示（无导航栏）
- [ ] 表单提交功能正常

- [ ] **Step 4：阶段 1 提交**

```bash
git add templates/
git commit -m "refactor: introduce base.html template inheritance for all pages"
```

---

## 阶段 2：提取内联 CSS

### Task 6：创建 base.css（全局 reset + 变量）

**Files:**
- Create: `static/css/base.css`
- Modify: `templates/base.html`

每个页面的 `<style>` 都以相同的 reset 和 body 样式开头：

```css
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: "Microsoft YaHei", "Segoe UI", sans-serif; ... }
```

- [ ] **Step 1：创建 static/css/base.css**

```css
/* RL4CO Display — 全局基础样式 */

:root {
    --color-primary: #667eea;
    --color-secondary: #764ba2;
    --gradient-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --font-main: "Microsoft YaHei", "Segoe UI", sans-serif;
    --radius-card: 20px;
    --shadow-card: 0 10px 40px rgba(0,0,0,0.1);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: var(--font-main);
    background: var(--gradient-bg);
    color: #333;
    min-height: 100vh;
}
```

- [ ] **Step 2：在 base.html 的 `<head>` 中引入 base.css**

在 `<link rel="stylesheet" href="/static/css/navigation.css">` 之后加一行：

```html
<link rel="stylesheet" href="/static/css/base.css">
```

- [ ] **Step 3：验证所有页面背景渐变和字体正常**

---

### Task 7：提取 index.html 的内联 CSS

**Files:**
- Create: `static/css/index.css`
- Modify: `templates/index.html`

- [ ] **Step 1：创建 static/css/index.css**

读取 index.html 的 `{% block head %}` 中 `<style>` 标签内的全部内容（阶段 1 迁移后约在 block head 里），剔除 `* { reset }` 和 `body { }` 这两段（已在 base.css 里），将剩余内容写入 `static/css/index.css`。

- [ ] **Step 2：修改 templates/index.html 的 {% block head %}**

将：
```html
{% block head %}
<link rel="stylesheet" href="/static/css/ollama-chat-embedded-enhanced.css">
<style>
  /* 大段内联样式 */
</style>
{% endblock %}
```

改为：
```html
{% block head %}
<link rel="stylesheet" href="/static/css/ollama-chat-embedded-enhanced.css">
<link rel="stylesheet" href="/static/css/index.css">
{% endblock %}
```

- [ ] **Step 3：验证**

访问 `http://localhost:5000/`，检查训练表单、参数面板、AI 助手面板样式正常。

---

### Task 8：提取其余页面的内联 CSS

**Files:**
- Create: `static/css/benchmark.css`
- Create: `static/css/model_info.css`
- Create: `static/css/file_manager.css`
- Create: `static/css/profile.css`
- Create: `static/css/auth.css`
- Modify: 对应 HTML 模板

对每个页面重复 Task 7 的流程：
1. 读 `{% block head %}` 中的 `<style>` 内容
2. 剔除与 base.css 重复的 `* {}` 和 `body {}` 段
3. 写入 `static/css/<page>.css`
4. HTML 模板改为 `<link>` 引用
5. 验证页面样式无变化

注意：login.html 和 register.html 共用 `static/css/auth.css`（它们的 body 样式有差异，放到各自文件也可以）。

- [ ] **Step 1：提取 benchmark.css + 验证 /benchmark**
- [ ] **Step 2：提取 model_info.css + 验证 /model_info**
- [ ] **Step 3：提取 file_manager.css + 验证 /file_manager**
- [ ] **Step 4：提取 profile.css + 验证 /profile**
- [ ] **Step 5：提取 auth.css，login.html 和 register.html 各自引用 + 验证**

- [ ] **Step 6：阶段 2 提交**

```bash
git add static/css/ templates/
git commit -m "refactor: extract all inline CSS to dedicated static files"
```

---

## 阶段 3：提取内联 JS + 清理废弃文件

### Task 9：提取 index.html 的内联 JS

**Files:**
- Create: `static/js/index.js`
- Create: `static/js/training-progress.js`
- Modify: `templates/index.html`

index.html 的内联 JS 主要包含两块逻辑，分别提取到独立文件：

- [ ] **Step 1：识别两块 JS 的边界**

读取 index.html 的 `{% block scripts %}` 内容，找出：
- **训练表单逻辑**：问题类型切换、参数面板显示/隐藏、兼容性校验、表单提交前验证
- **SSE 进度流逻辑**：`EventSource` 连接、进度消息处理、训练结果展示

- [ ] **Step 2：创建 static/js/training-progress.js**

将 SSE 相关代码（`EventSource`、进度条更新、结果渲染函数）提取到此文件。此文件只依赖 DOM，无外部库依赖。

- [ ] **Step 3：创建 static/js/index.js**

将训练表单交互代码（问题切换、参数联动、兼容性校验调用）提取到此文件。

- [ ] **Step 4：修改 templates/index.html 的 {% block scripts %}**

```html
{% block scripts %}
<script src="/static/js/training-progress.js"></script>
<script src="/static/js/index.js"></script>
<script src="/static/js/ollama-chat-embedded.js"></script>
{% endblock %}
```

- [ ] **Step 5：验证 index.html 全功能**

```bash
python app.py
```
- [ ] 训练表单：切换问题类型，对应参数面板正确出现/消失
- [ ] 开始训练：点击训练按钮，进度条正常更新
- [ ] AI 助手：发送消息，回复正常流式显示

---

### Task 10：提取 benchmark.html 的内联 JS

**Files:**
- Create: `static/js/benchmark.js`
- Modify: `templates/benchmark.html`

- [ ] **Step 1：创建 static/js/benchmark.js**

将 benchmark.html 中 echarts 图表初始化、数据筛选、`updateCharts()` 等函数提取到此文件。

注意：`benchmarkData` 数据对象通常较大，一并移入文件。

- [ ] **Step 2：修改 templates/benchmark.html 的 {% block scripts %}**

```html
{% block scripts %}
<script src="/static/js/benchmark.js"></script>
{% endblock %}
```

- [ ] **Step 3：验证 /benchmark**

- [ ] 图表正常渲染
- [ ] 问题类型、指标、算法类别筛选均生效
- [ ] 控制台无 `benchmarkData is not defined` 等错误

---

### Task 11：清理废弃文件

**Files:**
- Delete: `templates/FRONTEND_ALGORITHM_SELECTOR.html`
- Delete: `static/css/ollama-chat-embedded.css`
- Delete: `validation_results_20260527_184038.csv`
- Delete: `validation_results_20260527_184038.json`
- Delete: `validation_results_20260527_223424.csv`
- Delete: `validation_results_20260527_223424.json`

- [ ] **Step 1：确认 ollama-chat-embedded.css 未被引用**

```bash
grep -r "ollama-chat-embedded.css" templates/ static/
```

预期：只出现 `ollama-chat-embedded-enhanced.css`，不再有对非 enhanced 版本的引用。

- [ ] **Step 2：删除废弃文件**

```bash
rm templates/FRONTEND_ALGORITHM_SELECTOR.html
rm static/css/ollama-chat-embedded.css
rm validation_results_20260527_184038.csv
rm validation_results_20260527_184038.json
rm validation_results_20260527_223424.csv
rm validation_results_20260527_223424.json
```

- [ ] **Step 3：阶段 3 提交**

```bash
git add -A
git commit -m "refactor: extract inline JS to static files, remove stale assets"
```

---

## 最终验证

- [ ] 全站页面逐一访问，无样式丢失、无 JS 报错
- [ ] 完整执行一次训练流程（选问题 → 配置参数 → 开始训练 → 查看进度）
- [ ] 登录/注册流程正常
- [ ] AI 助手对话正常

完成后：`index.html` 预计从 2867 行缩减至 ~200 行，所有页面共享同一个 `<head>` 骨架，CSS/JS 均可浏览器缓存。
