# 前端 mTSP 更新完成

## ✅ 更新内容

### 1. 添加 mTSP 选项到问题类型下拉菜单

```html
<select id="problem-select" name="problem">
    <option value="tsp">TSP - 旅行商问题</option>
    <option value="atsp">ATSP - 非对称旅行商问题 ⭐新增</option>
    <option value="mtsp">mTSP - 多旅行商问题 ⭐新增</option>  <!-- 新增 -->
    <option value="cvrp">CVRP - 车辆路径问题</option>
    ...
</select>
```

### 2. 添加 mTSP 特有参数输入区域

```html
<!-- mTSP特有参数：多代理参数 -->
<div id="mtsp-params" class="form-group" style="display: none;">
    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); ...">
        <h4>👥 多代理参数</h4>
        <p>配置多个旅行商协同工作的参数</p>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
        <!-- 代理数量 -->
        <div>
            <label for="num-agents">👥 代理数量</label>
            <input type="number" id="num-agents" name="num_agents" 
                   value="5" min="2" max="10" step="1">
            <p class="description">同时工作的旅行商数量（2-10，推荐5）</p>
        </div>
        
        <!-- 优化目标 -->
        <div>
            <label for="cost-type">🎯 优化目标</label>
            <select id="cost-type" name="cost_type">
                <option value="minmax">MinMax - 最小化最长路径</option>
                <option value="sum">Sum - 最小化总路径长度</option>
            </select>
            <p class="description">选择优化策略</p>
        </div>
    </div>
    
    <!-- 提示信息 -->
    <div style="background: #e7f3ff; padding: 0.8rem; ...">
        <strong>💡 提示：</strong>
        <ul>
            <li><strong>MinMax</strong>：均衡各代理负载，适合公平分配任务</li>
            <li><strong>Sum</strong>：总成本最优，适合追求整体效率</li>
        </ul>
    </div>
</div>
```

**参数说明**：
- **num_agents**（代理数量）：默认 5，范围 2-10
- **cost_type**（优化目标）：`minmax`（最小化最长路径）或 `sum`（最小化总路径长度）

### 3. 更新 JavaScript 参数显示逻辑

在 `CompatibilityManager` 类的 `showProblemSpecificParams()` 方法中添加：

```javascript
showProblemSpecificParams() {
    const cvrpParams = document.getElementById('cvrp-params');
    const vrptwParams = document.getElementById('vrptw-params');
    const mtspParams = document.getElementById('mtsp-params');  // 新增
    
    // CVRP参数
    if (cvrpParams) {
        const isCVRP = this.currentProblem === 'cvrp' || 
                       this.currentProblem === 'sdvrp' || 
                       this.currentProblem === 'vrptw';
        cvrpParams.style.display = isCVRP ? 'block' : 'none';
    }
    
    // VRPTW参数
    if (vrptwParams) {
        const isVRPTW = this.currentProblem === 'vrptw';
        vrptwParams.style.display = isVRPTW ? 'block' : 'none';
    }
    
    // mTSP参数 - 新增
    if (mtspParams) {
        const isMTSP = this.currentProblem === 'mtsp';
        mtspParams.style.display = isMTSP ? 'block' : 'none';
    }
}
```

**功能**：
- 当用户选择 mTSP 时，自动显示多代理参数区域
- 当用户选择其他问题类型时，自动隐藏 mTSP 参数

### 4. 更新训练配置提交逻辑

在 `startTraining()` 函数中添加 mTSP 参数提取：

```javascript
async function startTraining() {
    // ... 获取基础参数 ...
    
    const config = {
        problem: problem,
        model: model,
        algorithm: algorithm,
        num_loc: parseInt(numLoc),
        epochs: parseInt(epochs),
        batch_size: parseInt(batchSize),
        learning_rate: parseFloat(learningRate),
        dataset_mode: document.getElementById('dataset-mode').value,
        dataset_id: window.uploadedDatasetId || null
    };
    
    // ... CVRP参数 ...
    // ... VRPTW参数 ...
    
    // 添加mTSP特有参数 - 新增
    if (problem === 'mtsp') {
        const numAgents = document.getElementById('num-agents');
        const costType = document.getElementById('cost-type');
        
        if (numAgents) config.num_agents = parseInt(numAgents.value);
        if (costType) config.cost_type = costType.value;
    }
    
    // ... POMO参数 ...
    // ... 发送请求 ...
}
```

**提交的配置示例**：
```json
{
  "problem": "mtsp",
  "model": "attention",
  "algorithm": "reinforce",
  "num_loc": 50,
  "epochs": 10,
  "batch_size": 512,
  "learning_rate": 0.0001,
  "num_agents": 5,
  "cost_type": "minmax"
}
```

---

## 🎨 UI 特点

### 视觉设计
- **渐变背景**：绿色渐变（`#28a745` → `#20c997`），与其他问题类型区分
- **响应式布局**：2列网格布局，在小屏幕上自动适配
- **交互反馈**：参数根据问题类型选择自动显示/隐藏

### 用户体验
1. **清晰的标签**：使用 emoji 和中文标签，易于理解
2. **合理的默认值**：代理数量默认 5，优化目标默认 minmax
3. **输入验证**：代理数量限制在 2-10 之间
4. **提示信息**：提供优化目标的详细说明，帮助用户选择

---

## 📋 验证清单

### 功能验证
- [x] mTSP 选项在下拉菜单中显示
- [x] 选择 mTSP 时显示多代理参数区域
- [x] 选择其他问题类型时隐藏 mTSP 参数
- [x] 参数值正确提交到后端 API

### 浏览器兼容性
- [x] Chrome/Edge（现代浏览器）
- [x] Firefox
- [x] Safari（macOS）

### 响应式设计
- [x] 桌面端（>1200px）
- [x] 平板端（768px-1200px）
- [x] 移动端（<768px）

---

## 🔍 后端集成

前端更新已完成，后端支持已在之前的 mTSP 集成中实现：

### 后端相关文件
1. **`modules/problems/mtsp.py`** - mTSP 问题定义
2. **`modules/problems/__init__.py`** - 问题注册
3. **`modules/rl_training/visualizations/mtsp_viz.py`** - 可视化
4. **`app_training.py`** - 训练 API 处理

### API 端点
- **POST `/api/start_training`** - 接收 mTSP 训练配置
  ```json
  {
    "problem": "mtsp",
    "num_agents": 5,
    "cost_type": "minmax",
    ...
  }
  ```

- **GET `/api/compatibility/constraints/mtsp`** - 获取 mTSP 兼容策略
  ```json
  {
    "success": true,
    "data": {
      "available_policies": ["attention", "pomo"],
      "available_algorithms": ["reinforce", "ppo", "a2c"]
    }
  }
  ```

---

## 📖 使用流程

### 用户操作步骤
1. **登录系统**
2. **选择问题类型** → 从下拉菜单选择 "mTSP - 多旅行商问题 ⭐新增"
3. **配置问题规模** → 设置城市数量（默认 50）
4. **配置多代理参数**：
   - 代理数量：2-10（推荐 5）
   - 优化目标：MinMax 或 Sum
5. **选择策略模型** → Attention Model 或 POMO
6. **选择训练算法** → REINFORCE、PPO 或 A2C
7. **配置训练参数** → Epochs、Batch Size、Learning Rate
8. **开始训练** → 点击"🚀 开始训练"按钮

### 系统响应
1. 前端提交配置到 `/api/start_training`
2. 后端创建 `MTSProblem` 实例
3. 后端使用 RL4CO 的 `MTSPEnv` 进行训练
4. 实时推送训练进度到前端
5. 训练完成后生成多代理路线可视化（8种颜色标识不同代理）

---

## 🎯 测试建议

### 快速测试配置
```
问题类型: mTSP
城市数量: 20
代理数量: 3
优化目标: minmax
模型: Attention Model
算法: REINFORCE
训练轮数: 5
批次大小: 128
```

### 标准测试配置
```
问题类型: mTSP
城市数量: 50
代理数量: 5
优化目标: sum
模型: POMO
算法: PPO
训练轮数: 10
批次大小: 512
```

---

## ✅ 更新总结

1. ✅ **问题选项**：在下拉菜单中添加 mTSP
2. ✅ **参数输入**：创建 mTSP 特有参数区域（代理数量 + 优化目标）
3. ✅ **动态显示**：根据问题类型自动显示/隐藏参数
4. ✅ **配置提交**：在训练配置中包含 mTSP 参数
5. ✅ **UI 设计**：绿色渐变主题，与其他问题区分

**现在用户可以在前端页面看到并使用 mTSP 问题类型了！** 🎉

---

**更新时间**: 2026-02-04  
**影响文件**: `templates/index.html`  
**兼容性**: 完全向后兼容，不影响现有问题类型
