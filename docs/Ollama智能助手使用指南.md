# Ollama智能助手使用指南

## 📚 目录

- [功能概述](#功能概述)
- [快速开始](#快速开始)
- [Ollama安装配置](#ollama安装配置)
- [功能详解](#功能详解)
- [界面说明](#界面说明)
- [常见问题](#常见问题)
- [高级配置](#高级配置)

---

## 功能概述

Ollama智能助手是集成在RL4CO平台首页的AI对话模块，支持与本地运行的Ollama大模型进行实时对话。

### ✨ 核心特性

1. **悬浮式设计**
   - 默认显示为右下角的机器人图标 🤖
   - 点击展开完整对话界面
   - 不遮挡主要功能区域

2. **智能对话**
   - 支持多轮对话，保留上下文
   - 流式响应，实时显示AI回复
   - 打字机效果，提升交互体验

3. **模型管理**
   - 自动检测本地可用模型
   - 支持快速切换模型
   - 实时显示连接状态

4. **历史记录**
   - 本地存储对话历史
   - 支持查看和清空历史
   - 最多保存50条消息

5. **消息操作**
   - 一键复制消息内容
   - 重新生成AI回复
   - 支持代码块高亮显示

6. **响应式设计**
   - 桌面端：右侧悬浮面板（400×600px）
   - 移动端：底部展开式（占屏幕60%）
   - 自适应深色/浅色模式

---

## 快速开始

### 1. 启动Ollama服务

在使用智能助手前，需要确保Ollama服务正在运行：

```bash
# 启动Ollama服务
ollama serve
```

默认服务地址：`http://localhost:11434`

### 2. 下载模型

如果还没有下载任何模型，需要先下载：

```bash
# 下载llama2模型（推荐）
ollama pull llama2

# 或下载其他模型
ollama pull mistral
ollama pull codellama
ollama pull qwen
```

### 3. 使用智能助手

1. 打开RL4CO平台首页
2. 点击右下角的机器人图标 🤖
3. 在模型选择器中选择已下载的模型
4. 在输入框中输入问题并发送
5. 享受AI对话！

---

## Ollama安装配置

### Windows系统

#### 方法1：使用安装程序（推荐）

1. 访问 [Ollama官网](https://ollama.ai/)
2. 下载Windows安装程序
3. 运行安装程序，按提示完成安装
4. 安装完成后，Ollama会自动启动

#### 方法2：使用命令行

```powershell
# 使用winget安装
winget install Ollama.Ollama

# 启动服务
ollama serve
```

### Linux系统

```bash
# 一键安装脚本
curl -fsSL https://ollama.ai/install.sh | sh

# 启动服务
ollama serve

# 或使用systemd管理
sudo systemctl start ollama
sudo systemctl enable ollama
```

### macOS系统

```bash
# 使用Homebrew安装
brew install ollama

# 启动服务
ollama serve
```

### Docker部署

```bash
# 拉取镜像
docker pull ollama/ollama

# 运行容器
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# 在容器中下载模型
docker exec -it ollama ollama pull llama2
```

### 验证安装

```bash
# 检查服务状态
curl http://localhost:11434/api/tags

# 列出已安装的模型
ollama list

# 测试对话
ollama run llama2
```

---

## 功能详解

### 1. 对话功能

#### 发送消息
- **键盘快捷键**：
  - `Enter`：发送消息
  - `Shift + Enter`：换行
- **按钮操作**：点击发送按钮 ➤

#### 消息格式支持
- **普通文本**：直接显示
- **代码块**：
  ```python
  def hello():
      print("Hello, World!")
  ```
- **行内代码**：使用 `反引号` 包裹
- **粗体**：使用 **双星号** 包裹
- **斜体**：使用 *单星号* 包裹

### 2. 模型选择

#### 切换模型
1. 点击顶部的模型选择下拉框
2. 从列表中选择目标模型
3. 系统会显示切换成功提示
4. 新对话将使用新模型

#### 模型推荐

| 模型 | 大小 | 特点 | 适用场景 |
|------|------|------|----------|
| llama2 | 7GB | 通用对话，速度快 | 日常问答、技术咨询 |
| mistral | 4GB | 轻量级，响应快 | 快速问答 |
| codellama | 7GB | 代码专用 | 代码解释、生成 |
| qwen | 7GB | 中文优化 | 中文对话 |
| deepseek-coder | 6GB | 代码能力强 | 编程辅助 |

### 3. 历史记录

#### 保存机制
- 自动保存到浏览器本地存储（localStorage）
- 最多保存最近50条消息
- 刷新页面后自动恢复

#### 清空历史
1. 点击顶部的垃圾桶图标 🗑️
2. 确认清空操作
3. 所有对话记录将被删除

### 4. 消息操作

#### 复制消息
1. 鼠标悬停在消息上
2. 点击 "📋 复制" 按钮
3. 内容已复制到剪贴板

#### 重新生成
1. 仅对AI回复有效
2. 鼠标悬停在AI消息上
3. 点击 "🔄 重新生成" 按钮
4. AI将基于相同问题重新回答

---

## 界面说明

### 布局结构

```
┌─────────────────────────────────────┐
│ 🤖 AI智能助手          🗑️ ✖       │  ← 头部
├─────────────────────────────────────┤
│ 选择模型: [llama2 ▼]               │  ← 模型选择器
│ 🟢 已连接到Ollama                   │
├─────────────────────────────────────┤
│                                     │
│ 👤 用户消息                         │  ← 消息区域
│     [消息内容]                      │
│                                     │
│ 🤖 AI回复                           │
│     [回复内容]                      │
│                                     │
├─────────────────────────────────────┤
│ [输入您的问题...            ] ➤    │  ← 输入区域
│ 按 Enter 发送，Shift+Enter 换行     │
└─────────────────────────────────────┘
```

### 视觉设计

#### 颜色方案
- **主题色**：绿色渐变 (#188754 → #156b43)
- **用户消息**：浅蓝色渐变 (#e3f2fd → #bbdefb)
- **AI消息**：浅紫色渐变 (#f3e5f5 → #e1bee7)

#### 动画效果
- **呼吸灯**：机器人图标缓慢跳动
- **打字指示器**：三个小点上下跳动
- **消息滑入**：新消息从下方滑入
- **按钮悬停**：轻微放大效果

---

## 常见问题

### Q1: 点击机器人图标没有反应？

**原因**：可能是JavaScript未正确加载

**解决方法**：
1. 检查浏览器控制台是否有错误
2. 刷新页面（Ctrl+F5 强制刷新）
3. 确认已正确引入CSS和JS文件

### Q2: 显示"Ollama未运行"？

**原因**：Ollama服务未启动或端口不正确

**解决方法**：
```bash
# 启动Ollama服务
ollama serve

# 检查服务状态
curl http://localhost:11434/api/tags

# 检查端口占用
netstat -an | grep 11434
```

### Q3: 显示"未找到可用模型"？

**原因**：还没有下载任何模型

**解决方法**：
```bash
# 下载推荐模型
ollama pull llama2

# 查看已有模型
ollama list
```

### Q4: AI回复很慢或卡顿？

**原因**：模型太大或硬件性能不足

**解决方法**：
1. 切换到更小的模型（如mistral）
2. 关闭其他占用资源的程序
3. 考虑使用GPU加速

### Q5: 代码块没有高亮显示？

**原因**：需要指定代码语言

**解决方法**：
在对话中使用以下格式：
````markdown
```python
def hello():
    print("Hello")
```
````

### Q6: 历史记录丢失了？

**原因**：清理了浏览器缓存或使用了无痕模式

**解决方法**：
- 避免清理localStorage
- 不要使用无痕/隐私模式
- 可以考虑实现服务端存储（需要后端支持）

### Q7: 移动端显示异常？

**原因**：屏幕尺寸不兼容

**解决方法**：
1. 更新到最新版本
2. 检查是否有CSS冲突
3. 尝试横屏显示

### Q8: 如何更改API地址？

**场景**：Ollama运行在其他机器或端口

**解决方法**：
修改 `static/js/ollama-chat.js` 文件：

```javascript
this.config = {
    apiUrl: 'http://your-server:11434/api',  // 修改这里
    // ...
};
```

---

## 高级配置

### 1. 自定义样式

#### 修改主题色

编辑 `static/css/ollama-chat.css`：

```css
/* 修改主题渐变色 */
.ai-chat-trigger {
    background: linear-gradient(135deg, #YOUR_COLOR_1 0%, #YOUR_COLOR_2 100%);
}

.ai-chat-header {
    background: linear-gradient(135deg, #YOUR_COLOR_1 0%, #YOUR_COLOR_2 100%);
}
```

#### 调整面板大小

```css
.ai-chat-panel {
    width: 450px;  /* 默认400px */
    height: 650px; /* 默认600px */
}
```

### 2. 配置选项

编辑 `static/js/ollama-chat.js` 中的 `config` 对象：

```javascript
this.config = {
    apiUrl: 'http://localhost:11434/api',  // API地址
    defaultModel: 'llama2',                 // 默认模型
    storageKey: 'ollama_chat_history',      // 存储键名
    guideKey: 'ollama_guide_shown',         // 引导键名
    maxHistoryLength: 50                    // 最大历史记录数
};
```

### 3. 添加预设提示词

在发送消息前自动添加系统提示词：

```javascript
// 在 sendMessage() 方法中修改
const systemPrompt = {
    role: 'system',
    content: '你是RL4CO平台的AI助手，专门回答强化学习和组合优化相关问题。'
};

messages.unshift(systemPrompt);
```

### 4. 启用CORS（跨域）

如果Ollama运行在其他机器上，可能需要配置CORS：

```bash
# 设置环境变量
export OLLAMA_ORIGINS="*"

# 或在启动时指定
OLLAMA_ORIGINS="*" ollama serve
```

### 5. 性能优化

#### 减少历史消息数量

```javascript
this.config = {
    maxHistoryLength: 20  // 从50减少到20
};
```

#### 禁用动画（提升性能）

```css
/* 添加到CSS末尾 */
* {
    animation-duration: 0ms !important;
    transition-duration: 0ms !important;
}
```

### 6. 集成其他LLM服务

智能助手架构支持替换为其他兼容OpenAI API的服务：

```javascript
// 修改API调用部分
const response = await fetch('https://your-api-endpoint/v1/chat/completions', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: JSON.stringify({
        model: 'gpt-3.5-turbo',
        messages: messages,
        stream: true
    })
});
```

---

## 使用技巧

### 1. 有效提问

**好的问题**：
- "如何在RL4CO中使用POMO算法训练TSP模型？"
- "解释一下Attention Model的工作原理"
- "这段代码有什么问题？[粘贴代码]"

**不太好的问题**：
- "你好"（过于简单）
- "帮我做作业"（范围太广）
- "最好的算法是什么"（主观问题）

### 2. 上下文管理

- AI会记住当前对话的所有内容
- 可以进行多轮追问
- 如果切换话题，考虑清空历史

### 3. 代码交互

**询问代码**：
```
请生成一个Python函数，用于读取TSP数据集
```

**调试代码**：
```
这段代码报错了，帮我看看：
[粘贴代码]
```

**优化代码**：
```
如何优化这段代码的性能？
[粘贴代码]
```

### 4. 学习辅助

- 询问概念解释
- 要求举例说明
- 让AI生成练习题
- 请AI推荐学习资源

---

## 开发指南

### 文件结构

```
rl4co-display/
├── static/
│   ├── css/
│   │   └── ollama-chat.css      # 样式文件
│   └── js/
│       └── ollama-chat.js       # 功能脚本
├── templates/
│   └── index.html               # 首页（已集成）
└── docs/
    └── Ollama智能助手使用指南.md  # 本文档
```

### 核心类结构

```javascript
class OllamaChat {
    constructor()           // 初始化
    init()                  // 异步初始化
    createElements()        // 创建DOM
    bindEvents()            // 绑定事件
    toggle()                // 切换显示
    checkConnection()       // 检查连接
    loadModels()            // 加载模型列表
    sendMessage()           // 发送消息
    handleStreamResponse()  // 处理流式响应
    addMessage()            // 添加消息
    formatMessage()         // 格式化消息
    // ...更多方法
}
```

### 扩展开发

#### 添加新功能按钮

```javascript
// 在创建面板时添加
<button class="ai-chat-btn" id="ai-export-btn" title="导出对话">
    📥
</button>

// 绑定事件
this.elements.exportBtn = document.getElementById('ai-export-btn');
this.elements.exportBtn.addEventListener('click', () => this.exportChat());

// 实现功能
exportChat() {
    const data = JSON.stringify(this.state.messages, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'chat-history.json';
    a.click();
}
```

#### 添加快捷指令

```javascript
// 在输入框中检测特殊命令
if (text.startsWith('/')) {
    this.handleCommand(text);
    return;
}

handleCommand(command) {
    switch(command) {
        case '/clear':
            this.clearHistory();
            break;
        case '/help':
            this.showHelp();
            break;
        // 添加更多命令
    }
}
```

---

## 更新日志

### v1.0.0 (2025-11-27)

**首次发布**
- ✅ 悬浮式对话界面
- ✅ Ollama集成
- ✅ 流式响应支持
- ✅ 模型管理
- ✅ 历史记录
- ✅ 消息操作（复制、重新生成）
- ✅ 响应式设计
- ✅ 深色模式支持
- ✅ 代码块高亮
- ✅ 首次使用引导

---

## 技术支持

### 问题反馈

如果遇到问题，请提供以下信息：

1. 操作系统版本
2. 浏览器类型和版本
3. Ollama版本（`ollama --version`）
4. 错误截图或控制台日志
5. 复现步骤

### 联系方式

- **项目仓库**：[GitHub](https://github.com/your-repo)
- **问题跟踪**：[Issues](https://github.com/your-repo/issues)
- **开发团队**：山西大学 计算机科学与技术学院

---

## 致谢

本智能助手模块基于以下开源技术：

- [Ollama](https://ollama.ai/) - 本地大模型运行环境
- [Llama 2](https://ai.meta.com/llama/) - Meta开源大语言模型
- CSS Grid & Flexbox - 现代布局技术
- LocalStorage API - 浏览器本地存储

---

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

---

© 2025 山西大学 计算机科学与技术学院. All Rights Reserved.

