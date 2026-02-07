# API 使用示例

## 📋 文档说明

本文档提供 RL4CO Display API 的实际使用示例代码。

**配套文档**: [API_PROTOCOL.md](./API_PROTOCOL.md)

---

## 🚀 完整训练流程示例

### 场景：从登录到训练完成

```javascript
// ========== 第1步：用户登录 ==========
async function loginAndTrain() {
  // 登录
  const loginResponse = await fetch('/api/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      username: 'demo_user',
      password: 'demo_password'
    })
  });
  
  const loginResult = await loginResponse.json();
  
  if (!loginResult.success) {
    console.error('登录失败:', loginResult.error);
    return;
  }
  
  console.log('✅ 登录成功，用户ID:', loginResult.user.user_id);
  
  // ========== 第2步：检查兼容性 ==========
  const problem = 'mtsp';
  
  // 获取该问题支持的策略和算法
  const constraintsResponse = await fetch(`/api/compatibility/constraints/${problem}`);
  const constraintsResult = await constraintsResponse.json();
  
  console.log('可用策略:', constraintsResult.data.available_policies);
  console.log('可用算法:', constraintsResult.data.available_algorithms);
  
  // ========== 第3步：获取推荐配置 ==========
  const recommendResponse = await fetch(`/api/compatibility/recommend?problem=${problem}&preference=best`);
  const recommendResult = await recommendResponse.json();
  
  console.log('推荐配置:', recommendResult.recommended);
  // 输出: { policy: 'pomo', algorithm: 'ppo' }
  
  // ========== 第4步：验证配置 ==========
  const validateResponse = await fetch('/api/compatibility/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      problem: 'mtsp',
      model: 'pomo',
      algorithm: 'ppo'
    })
  });
  
  const validateResult = await validateResponse.json();
  
  if (!validateResult.valid) {
    console.error('配置无效:', validateResult.message);
    return;
  }
  
  console.log('✅ 配置有效');
  
  // ========== 第5步：启动训练 ==========
  const trainingConfig = {
    problem: 'mtsp',
    model: 'pomo',
    algorithm: 'ppo',
    num_loc: 50,
    num_agents: 5,
    cost_type: 'minmax',
    epochs: 10,
    batch_size: 512,
    learning_rate: 0.0001,
    num_starts: 50
  };
  
  const startResponse = await fetch('/api/start_training', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(trainingConfig)
  });
  
  const startResult = await startResponse.json();
  
  if (!startResult.success) {
    console.error('启动失败:', startResult.message);
    return;
  }
  
  const sessionId = startResult.session_id;
  console.log('✅ 训练已启动，Session ID:', sessionId);
  
  // ========== 第6步：订阅训练进度 ==========
  const eventSource = new EventSource(`/api/training_progress/${sessionId}`);
  
  eventSource.onopen = () => {
    console.log('✅ SSE 连接已建立');
  };
  
  eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch (data.type) {
      case 'info':
        console.log('[信息]', data.message);
        break;
        
      case 'warning':
        console.warn('[警告]', data.message);
        break;
        
      case 'error':
        console.error('[错误]', data.message);
        eventSource.close();
        break;
        
      case 'progress':
        console.log(`[进度] Epoch ${data.epoch}/${data.total_epochs} (${data.progress}%)`);
        console.log(`  Loss: ${data.loss}, Reward: ${data.reward}, Best: ${data.best_reward}`);
        break;
        
      case 'plot':
        console.log('[图表]', data.message);
        console.log('  训练曲线:', data.plot_url);
        break;
        
      case 'complete':
        console.log('✅ 训练完成!');
        console.log('最终结果:', data.results);
        
        // 显示可视化图片
        if (data.results.plot_paths) {
          console.log('静态对比图:', data.results.plot_paths);
        }
        if (data.results.animation_paths) {
          console.log('动画:', data.results.animation_paths);
        }
        if (data.results.training_curve) {
          console.log('训练曲线:', data.results.training_curve);
        }
        
        eventSource.close();
        break;
        
      case 'heartbeat':
        // 心跳消息，忽略
        break;
    }
  };
  
  eventSource.onerror = (error) => {
    console.error('❌ SSE 连接错误:', error);
    eventSource.close();
  };
}

// 执行
loginAndTrain();
```

**预期输出**:
```
✅ 登录成功，用户ID: 1
可用策略: ["attention", "pomo"]
可用算法: ["reinforce", "ppo", "a2c"]
推荐配置: { policy: 'pomo', algorithm: 'ppo' }
✅ 配置有效
✅ 训练已启动，Session ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
✅ SSE 连接已建立
[信息] 开始训练 POMO 模型，问题类型: MTSP
[信息] 配置: Epochs=10, Batch=512, LR=0.0001, 问题规模=50
[信息] 📋 mTSP配置: 5个代理, 优化目标=minmax
[信息] 使用设备: cpu
[信息] ✅ mTSP环境创建成功: 50个城市, 5个代理
[进度] Epoch 1/10 (10%)
  Loss: 1.5432, Reward: -6.7890, Best: -6.7890
[图表] Epoch 1 训练曲线已更新
  训练曲线: /static/model_plots/user_1/training_curves_abc123.png
...
[进度] Epoch 10/10 (100%)
  Loss: 0.8765, Reward: -5.1234, Best: -5.1234
[信息] 训练完成，开始生成可视化结果...
[信息] 🎨 开始生成mTSP可视化（多代理路径）...
[信息] ✅ 动画 1 生成成功
[信息] ✅ 对比图 1 生成成功
[信息] 🎉 mTSP可视化完成: 3个动画, 3个对比图
✅ 训练完成!
最终结果: {
  model: 'pomo',
  problem: 'mtsp',
  strategy: 'PPO',
  total_epochs: 10,
  final_loss: 0.8765,
  final_reward: -5.1234,
  best_reward: -5.1234,
  plot_paths: [
    '/static/model_plots/user_1/mtsp_comparison_1.png',
    '/static/model_plots/user_1/mtsp_comparison_2.png',
    '/static/model_plots/user_1/mtsp_comparison_3.png'
  ],
  animation_paths: [
    '/static/model_plots/user_1/mtsp_animation_1.gif',
    '/static/model_plots/user_1/mtsp_animation_2.gif',
    '/static/model_plots/user_1/mtsp_animation_3.gif'
  ],
  training_curve: '/static/model_plots/user_1/training_curves_abc123.png',
  checkpoint_path: '/path/to/mtsp-pomo.ckpt'
}
```

---

## 🎨 不同问题类型的配置示例

### 示例 1: TSP + Attention + REINFORCE

```javascript
const config = {
  problem: 'tsp',
  model: 'attention',
  algorithm: 'reinforce',
  num_loc: 50,
  epochs: 10,
  batch_size: 512,
  learning_rate: 0.0001
};

// 预期结果
// - 训练时间: 较短
// - 质量: 中等
// - 适合: 学习入门
```

### 示例 2: mTSP + POMO + PPO

```javascript
const config = {
  problem: 'mtsp',
  model: 'pomo',
  algorithm: 'ppo',
  num_loc: 50,
  num_agents: 5,
  cost_type: 'minmax',
  epochs: 10,
  batch_size: 512,
  learning_rate: 0.0001,
  num_starts: 50  // POMO 特有参数
};

// 预期结果
// - 训练时间: 较长
// - 质量: 最高
// - 适合: 生产使用
```

### 示例 3: CVRP + POMO + PPO

```javascript
const config = {
  problem: 'cvrp',
  model: 'pomo',
  algorithm: 'ppo',
  num_loc: 100,
  vehicle_capacity: 1.0,
  epochs: 20,
  batch_size: 512,
  learning_rate: 0.0001,
  num_starts: 50
};
```

### 示例 4: VRPTW + Attention + PPO

```javascript
const config = {
  problem: 'vrptw',
  model: 'attention',
  algorithm: 'ppo',
  num_loc: 50,
  vehicle_capacity: 1.0,
  time_window_size: 10,
  epochs: 15,
  batch_size: 512,
  learning_rate: 0.0001
};
```

---

## 🔄 错误处理示例

### 1. 处理配置验证失败

```javascript
async function startTrainingWithValidation(config) {
  // 先验证配置
  const validateResponse = await fetch('/api/compatibility/validate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      problem: config.problem,
      model: config.model,
      algorithm: config.algorithm
    })
  });
  
  const validateResult = await validateResponse.json();
  
  if (!validateResult.valid) {
    // 配置无效，显示错误
    alert('配置无效: ' + validateResult.message);
    return;
  }
  
  if (validateResult.warnings && validateResult.warnings.length > 0) {
    // 有警告，询问用户是否继续
    const warnings = validateResult.warnings
      .map(w => w.message)
      .join('\n');
    
    if (!confirm('检测到以下警告:\n' + warnings + '\n\n是否继续?')) {
      return;
    }
  }
  
  // 启动训练
  const startResponse = await fetch('/api/start_training', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  });
  
  const startResult = await startResponse.json();
  
  if (!startResult.success) {
    alert('启动失败: ' + startResult.error);
    return;
  }
  
  console.log('✅ 训练已启动');
  subscribeProgress(startResult.session_id);
}
```

---

### 2. 处理 SSE 重连

```javascript
function subscribeProgressWithRetry(sessionId, maxRetries = 3) {
  let retryCount = 0;
  
  function connect() {
    const eventSource = new EventSource(`/api/training_progress/${sessionId}`);
    
    eventSource.onopen = () => {
      console.log('✅ SSE 连接已建立');
      retryCount = 0;  // 重置重试计数
    };
    
    eventSource.onerror = (error) => {
      console.error('❌ SSE 连接错误:', error);
      eventSource.close();
      
      // 重试逻辑
      if (retryCount < maxRetries) {
        retryCount++;
        console.log(`尝试重连... (${retryCount}/${maxRetries})`);
        setTimeout(() => connect(), 2000);  // 2秒后重试
      } else {
        console.error('重连失败，达到最大重试次数');
        alert('连接断开，请刷新页面');
      }
    };
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleMessage(data);
      
      if (data.type === 'complete' || data.type === 'error') {
        eventSource.close();
      }
    };
  }
  
  connect();
}
```

---

### 3. 处理网络错误

```javascript
async function fetchWithRetry(url, options, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url, options);
      return response;
    } catch (error) {
      console.error(`请求失败 (${i + 1}/${maxRetries}):`, error);
      
      if (i === maxRetries - 1) {
        throw error;  // 最后一次重试失败，抛出错误
      }
      
      // 等待后重试（指数退避）
      await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, i)));
    }
  }
}

// 使用示例
async function startTrainingWithRetry(config) {
  try {
    const response = await fetchWithRetry('/api/start_training', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    
    const result = await response.json();
    console.log('训练已启动:', result.session_id);
  } catch (error) {
    alert('网络错误，请检查连接');
  }
}
```

---

## 📊 数据集管理示例

### 上传数据集

```javascript
async function uploadDataset() {
  const fileInput = document.getElementById('dataset-file');
  const file = fileInput.files[0];
  
  if (!file) {
    alert('请选择文件');
    return;
  }
  
  const formData = new FormData();
  formData.append('file', file);
  formData.append('problem_type', 'tsp');
  
  try {
    const response = await fetch('/api/upload_dataset', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('✅ 上传成功');
      console.log('数据集ID:', result.dataset_id);
      console.log('实例数量:', result.num_instances);
      
      // 刷新数据集列表
      await loadDatasets();
    } else {
      alert('上传失败: ' + result.error);
    }
  } catch (error) {
    console.error('上传失败:', error);
    alert('网络错误');
  }
}
```

---

### 列出和删除数据集

```javascript
async function loadDatasets() {
  try {
    const response = await fetch('/api/list_datasets');
    const result = await response.json();
    
    if (result.success) {
      const datasets = result.datasets;
      console.log('数据集列表:', datasets);
      
      // 渲染列表
      const listHTML = datasets.map(ds => `
        <div class="dataset-item">
          <span>${ds.filename}</span>
          <span>${ds.num_instances} 个实例</span>
          <span>${ds.upload_time}</span>
          <button onclick="deleteDataset('${ds.dataset_id}')">删除</button>
        </div>
      `).join('');
      
      document.getElementById('dataset-list').innerHTML = listHTML;
    }
  } catch (error) {
    console.error('加载失败:', error);
  }
}

async function deleteDataset(datasetId) {
  if (!confirm('确定要删除这个数据集吗？')) {
    return;
  }
  
  try {
    const response = await fetch('/api/delete_dataset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id: datasetId })
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('✅ 删除成功');
      await loadDatasets();  // 刷新列表
    } else {
      alert('删除失败: ' + result.error);
    }
  } catch (error) {
    console.error('删除失败:', error);
  }
}
```

---

## 🎯 高级功能示例

### 1. 实时更新训练曲线

```javascript
class TrainingMonitor {
  constructor(sessionId) {
    this.sessionId = sessionId;
    this.eventSource = null;
    this.currentEpoch = 0;
    this.trainingCurveUrl = null;
  }
  
  start() {
    this.eventSource = new EventSource(`/api/training_progress/${this.sessionId}`);
    
    this.eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'progress') {
        this.updateProgress(data);
      } else if (data.type === 'plot') {
        this.updateTrainingCurve(data.plot_url);
      } else if (data.type === 'complete') {
        this.onComplete(data.results);
      }
    };
  }
  
  updateProgress(data) {
    // 更新进度条
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = data.progress + '%';
    
    // 更新数字显示
    document.getElementById('epoch-display').textContent = 
      `${data.epoch}/${data.total_epochs}`;
    document.getElementById('loss-display').textContent = 
      data.loss.toFixed(4);
    document.getElementById('reward-display').textContent = 
      data.reward.toFixed(4);
    document.getElementById('best-reward-display').textContent = 
      data.best_reward.toFixed(4);
    
    // 更新训练曲线（如果有）
    if (data.plot_url && data.plot_url !== this.trainingCurveUrl) {
      this.updateTrainingCurve(data.plot_url);
    }
  }
  
  updateTrainingCurve(url) {
    this.trainingCurveUrl = url;
    const img = document.getElementById('training-curve-img');
    // 添加时间戳避免缓存
    img.src = url + '?t=' + new Date().getTime();
  }
  
  onComplete(results) {
    console.log('训练完成:', results);
    
    // 显示结果面板
    document.getElementById('results-panel').style.display = 'block';
    
    // 显示所有可视化
    this.displayVisualizations(results);
    
    // 关闭连接
    this.eventSource.close();
  }
  
  displayVisualizations(results) {
    const container = document.getElementById('visualizations-container');
    container.innerHTML = '';
    
    // 显示静态对比图和对应的动画
    if (results.plot_paths && results.plot_paths.length > 0) {
      results.plot_paths.forEach((plotPath, index) => {
        const card = document.createElement('div');
        card.className = 'viz-card';
        
        // 静态对比图
        const plotImg = document.createElement('img');
        plotImg.src = plotPath;
        plotImg.alt = `对比图 ${index + 1}`;
        card.appendChild(plotImg);
        
        // 对应的动画（如果有）
        if (results.animation_paths && results.animation_paths[index]) {
          const animImg = document.createElement('img');
          animImg.src = results.animation_paths[index];
          animImg.alt = `动画 ${index + 1}`;
          card.appendChild(animImg);
        }
        
        container.appendChild(card);
      });
    }
  }
  
  stop() {
    if (this.eventSource) {
      this.eventSource.close();
    }
  }
}

// 使用
const monitor = new TrainingMonitor(sessionId);
monitor.start();

// 停止监控
// monitor.stop();
```

---

### 2. 批量训练任务

```javascript
async function batchTraining(configs) {
  const sessions = [];
  
  // 启动所有训练任务
  for (const config of configs) {
    try {
      const response = await fetch('/api/start_training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      
      const result = await response.json();
      
      if (result.success) {
        sessions.push({
          config: config,
          sessionId: result.session_id,
          status: 'running'
        });
        console.log(`✅ 任务 ${sessions.length} 已启动`);
      } else {
        console.error(`❌ 任务 ${sessions.length + 1} 启动失败:`, result.error);
      }
      
      // 延迟避免并发冲突
      await new Promise(resolve => setTimeout(resolve, 1000));
      
    } catch (error) {
      console.error('启动失败:', error);
    }
  }
  
  console.log(`总共启动 ${sessions.length}/${configs.length} 个任务`);
  
  // 监控所有任务
  sessions.forEach((session, index) => {
    const eventSource = new EventSource(`/api/training_progress/${session.sessionId}`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'complete') {
        session.status = 'completed';
        session.results = data.results;
        console.log(`✅ 任务 ${index + 1} 完成`);
        eventSource.close();
        
        // 检查是否所有任务都完成
        if (sessions.every(s => s.status === 'completed')) {
          console.log('🎉 所有任务完成!');
          compareResults(sessions);
        }
      }
    };
  });
  
  return sessions;
}

function compareResults(sessions) {
  console.log('=== 批量训练结果对比 ===');
  sessions.forEach((session, index) => {
    console.log(`任务 ${index + 1}:`, {
      problem: session.config.problem,
      model: session.config.model,
      algorithm: session.config.algorithm,
      best_reward: session.results.best_reward
    });
  });
}

// 使用示例
const configs = [
  { problem: 'tsp', model: 'attention', algorithm: 'reinforce', num_loc: 50, epochs: 5 },
  { problem: 'tsp', model: 'pomo', algorithm: 'ppo', num_loc: 50, epochs: 5 },
  { problem: 'tsp', model: 'attention', algorithm: 'a2c', num_loc: 50, epochs: 5 }
];

batchTraining(configs);
```

---

## 🧪 测试接口

### 使用 curl 测试

#### 1. 登录
```bash
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{"username": "demo", "password": "demo123"}' \
  -c cookies.txt

# 响应示例
# {"success": true, "message": "登录成功", "user": {...}}
```

#### 2. 启动训练
```bash
curl -X POST http://localhost:5000/api/start_training \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{
    "problem": "tsp",
    "model": "attention",
    "algorithm": "reinforce",
    "num_loc": 20,
    "epochs": 3,
    "batch_size": 128,
    "learning_rate": 0.0001
  }'

# 响应示例
# {"success": true, "session_id": "a1b2c3d4-...", "message": "训练已启动"}
```

#### 3. 获取兼容性信息
```bash
curl http://localhost:5000/api/compatibility/info \
  -b cookies.txt

# 响应示例
# {"success": true, "data": {...}}
```

#### 4. 验证配置
```bash
curl -X POST http://localhost:5000/api/compatibility/validate \
  -H "Content-Type: application/json" \
  -b cookies.txt \
  -d '{"problem": "atsp", "model": "pomo", "algorithm": "ppo"}'

# 响应示例
# {"success": true, "valid": false, "message": "POMO不支持ATSP"}
```

---

### 使用 Python requests 测试

```python
import requests
import json
from sseclient import SSEClient  # pip install sseclient-py

# 基础 URL
BASE_URL = 'http://localhost:5000'

# 创建 Session（保存 Cookie）
session = requests.Session()

# 1. 登录
login_response = session.post(
    f'{BASE_URL}/api/login',
    json={'username': 'demo', 'password': 'demo123'}
)
print('登录:', login_response.json())

# 2. 启动训练
config = {
    'problem': 'tsp',
    'model': 'attention',
    'algorithm': 'reinforce',
    'num_loc': 20,
    'epochs': 3,
    'batch_size': 128,
    'learning_rate': 0.0001
}

start_response = session.post(
    f'{BASE_URL}/api/start_training',
    json=config
)
result = start_response.json()
print('启动训练:', result)

session_id = result['session_id']

# 3. 订阅进度（SSE）
url = f'{BASE_URL}/api/training_progress/{session_id}'
messages = SSEClient(url, session=session)

for msg in messages:
    if msg.data:
        data = json.loads(msg.data)
        print(f'[{data["type"]}]', data.get('message', ''))
        
        if data['type'] == 'progress':
            print(f'  Epoch {data["epoch"]}/{data["total_epochs"]}, Loss: {data["loss"]}, Reward: {data["reward"]}')
        
        if data['type'] in ['complete', 'error']:
            if data['type'] == 'complete':
                print('最终结果:', data['results'])
            break

print('训练监控结束')
```

---

## 📝 注意事项

### 1. Session Cookie 管理

前端使用 Cookie 存储 Session，后端自动验证。注意：
- Cookie 有效期：7天
- 过期后需要重新登录
- 跨域请求需要设置 `credentials: 'include'`

### 2. SSE 连接限制

浏览器对同一域名的 SSE 连接有限制（通常6个）：
- 避免同时打开过多训练任务
- 任务完成后及时关闭 EventSource

### 3. 文件路径格式

前端访问文件时使用 URL 路径：
- ✅ 正确：`/static/model_plots/user_1/file.png`
- ❌ 错误：`/full/local/path/to/file.png`

### 4. 图片缓存

训练曲线会不断更新，需要添加时间戳避免浏览器缓存：
```javascript
img.src = url + '?t=' + new Date().getTime();
```

### 5. 并发训练

目前系统支持单用户多任务并发训练，但建议：
- 同时运行的任务不超过 3 个
- 大规模训练（epochs > 50）建议单独运行

---

**文档版本**: v1.0  
**最后更新**: 2026-02-04  
**配套文档**: [API_PROTOCOL.md](./API_PROTOCOL.md)
