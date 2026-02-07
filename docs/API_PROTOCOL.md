# RL4CO Display 前后端接口协议文档

## 📋 文档概述

**版本**: v1.0  
**最后更新**: 2026-02-04  
**适用范围**: RL4CO Display 平台前后端数据交互

本文档详细说明了 RL4CO Display 平台前端（HTML/JavaScript）与后端（Flask/Python）之间的所有接口协议，包括 REST API、SSE（Server-Sent Events）实时推送等。

---

## 📖 目录

1. [接口总览](#接口总览)
2. [认证接口](#认证接口)
3. [训练管理接口](#训练管理接口)
4. [兼容性检查接口](#兼容性检查接口)
5. [数据集管理接口](#数据集管理接口)
6. [文件管理接口](#文件管理接口)
7. [SSE 实时消息](#sse-实时消息)
8. [数据结构定义](#数据结构定义)
9. [错误处理](#错误处理)
10. [示例代码](#示例代码)

---

## 接口总览

### API 端点列表

| 分类 | 端点 | 方法 | 认证 | 说明 |
|------|------|------|------|------|
| **认证** | `/api/register` | POST | ❌ | 用户注册 |
| **认证** | `/api/login` | POST | ❌ | 用户登录 |
| **认证** | `/api/logout` | POST | ✅ | 用户登出 |
| **认证** | `/api/check_auth` | GET | ❌ | 检查认证状态 |
| **训练** | `/api/start_training` | POST | ✅ | 启动训练任务 |
| **训练** | `/api/training_progress/<session_id>` | GET (SSE) | ✅ | 订阅训练进度 |
| **训练** | `/api/training_status/<session_id>` | GET | ✅ | 获取训练状态 |
| **训练** | `/api/stop_training/<session_id>` | POST | ✅ | 停止训练任务 |
| **兼容性** | `/api/compatibility/info` | GET | ✅ | 获取兼容性信息 |
| **兼容性** | `/api/compatibility/constraints/<problem>` | GET | ✅ | 获取问题约束 |
| **兼容性** | `/api/compatibility/validate` | POST | ✅ | 验证配置组合 |
| **兼容性** | `/api/compatibility/recommend` | GET | ✅ | 获取推荐配置 |
| **数据集** | `/api/upload_dataset` | POST | ✅ | 上传数据集 |
| **数据集** | `/api/list_datasets` | GET | ✅ | 列出数据集 |
| **数据集** | `/api/delete_dataset` | POST | ✅ | 删除数据集 |
| **文件** | `/api/user_files` | GET | ✅ | 获取用户文件 |
| **文件** | `/api/session_files/<session_id>` | GET | ✅ | 获取会话文件 |

### 基础 URL

```
开发环境: http://localhost:5000
生产环境: https://your-domain.com
```

### 通用请求头

```http
Content-Type: application/json
Cookie: session=<session_cookie>
```

### 通用响应格式

```json
{
  "success": true,           // 布尔值，表示请求是否成功
  "message": "操作成功",      // 字符串，描述操作结果
  "data": { ... },           // 对象，具体的返回数据（可选）
  "error": "错误信息"         // 字符串，错误描述（仅在 success=false 时）
}
```

---

## 认证接口

### 1. 用户注册

**端点**: `POST /api/register`

**请求体**:
```json
{
  "username": "user123",
  "email": "user@example.com",
  "password": "secure_password"
}
```

**响应** (成功):
```json
{
  "success": true,
  "message": "注册成功",
  "user_id": 1
}
```

**响应** (失败):
```json
{
  "success": false,
  "error": "用户名已存在"
}
```

**字段说明**:
- `username`: 用户名，3-50字符，唯一
- `email`: 邮箱地址，需符合邮箱格式，唯一
- `password`: 密码，最少6个字符

---

### 2. 用户登录

**端点**: `POST /api/login`

**请求体**:
```json
{
  "username": "user123",
  "password": "secure_password"
}
```

**响应** (成功):
```json
{
  "success": true,
  "message": "登录成功",
  "user": {
    "user_id": 1,
    "username": "user123",
    "email": "user@example.com",
    "created_at": "2026-02-04 10:30:00"
  }
}
```

**响应** (失败):
```json
{
  "success": false,
  "error": "用户名或密码错误"
}
```

**Side Effects**:
- 成功登录后会设置 Session Cookie
- Session 有效期：7天

---

### 3. 用户登出

**端点**: `POST /api/logout`

**请求**: 无需请求体

**响应**:
```json
{
  "success": true,
  "message": "登出成功"
}
```

**Side Effects**:
- 清除 Session Cookie
- 清除服务器端 Session 数据

---

### 4. 检查认证状态

**端点**: `GET /api/check_auth`

**请求**: 无参数

**响应** (已认证):
```json
{
  "authenticated": true,
  "user": {
    "user_id": 1,
    "username": "user123"
  }
}
```

**响应** (未认证):
```json
{
  "authenticated": false
}
```

---

## 训练管理接口

### 1. 启动训练任务

**端点**: `POST /api/start_training`

**请求体**:
```json
{
  "problem": "tsp",
  "model": "pomo",
  "algorithm": "ppo",
  "num_loc": 50,
  "epochs": 10,
  "batch_size": 512,
  "learning_rate": 0.0001,
  
  // 问题特有参数（根据问题类型不同）
  // TSP: 无额外参数
  // mTSP: 
  "num_agents": 5,
  "cost_type": "minmax",
  
  // CVRP:
  "vehicle_capacity": 1.0,
  
  // VRPTW:
  "time_window_size": 10,
  
  // 策略特有参数
  // POMO:
  "num_starts": 50,
  
  // 算法特有参数
  // PPO:
  "clip_ratio": 0.2,
  
  // A2C:
  "value_loss_coef": 0.5,
  "entropy_coef": 0.01
}
```

**响应** (成功):
```json
{
  "success": true,
  "message": "训练任务已启动",
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**响应** (失败):
```json
{
  "success": false,
  "error": "配置验证失败: POMO不支持ATSP问题"
}
```

**字段说明**:

**通用参数**:
- `problem`: 问题类型，可选值：`tsp`, `atsp`, `mtsp`, `cvrp`, `sdvrp`, `vrptw`, `pctsp`, `op`
- `model`: 策略模型，可选值：`attention`, `am`, `pomo`
- `algorithm`: RL算法，可选值：`reinforce`, `ppo`, `a2c`
- `num_loc`: 节点数量，整数，10-200
- `epochs`: 训练轮数，整数，1-1000
- `batch_size`: 批次大小，整数，64-2048
- `learning_rate`: 学习率，浮点数，0.00001-0.01

**mTSP 特有参数**:
- `num_agents`: 代理数量，整数，2-10
- `cost_type`: 优化目标，可选值：`minmax`, `sum`

**CVRP 特有参数**:
- `vehicle_capacity`: 车辆容量，浮点数，0.5-2.0

**VRPTW 特有参数**:
- `time_window_size`: 时间窗大小，整数，5-20

**POMO 策略参数**:
- `num_starts`: 多起点数量，整数，10-100

**PPO 算法参数**:
- `clip_ratio`: 裁剪比率，浮点数，0.1-0.3

**A2C 算法参数**:
- `value_loss_coef`: 价值损失系数，浮点数，0.1-1.0
- `entropy_coef`: 熵系数，浮点数，0.001-0.1

---

### 2. 订阅训练进度 (SSE)

**端点**: `GET /api/training_progress/<session_id>`

**请求**: 
```
GET /api/training_progress/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**响应类型**: `text/event-stream`

**SSE 消息格式**:

每条消息以 `data:` 开头，内容为 JSON 字符串：

```
data: {"type": "info", "message": "训练开始"}

data: {"type": "progress", "epoch": 5, "total_epochs": 10, "progress": 50, "loss": 1.2345, "reward": -5.6789}

data: {"type": "complete", "message": "训练完成", "results": {...}}
```

**详细消息类型见下文 [SSE 实时消息](#sse-实时消息) 章节**

---

### 3. 获取训练状态

**端点**: `GET /api/training_status/<session_id>`

**请求**:
```
GET /api/training_status/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**响应**:
```json
{
  "success": true,
  "status": {
    "status": "running",
    "progress": 50,
    "epoch": 5,
    "loss": 1.2345,
    "reward": -5.6789,
    "best_reward": -5.6789
  }
}
```

**状态值**:
- `running`: 训练进行中
- `completed`: 训练完成
- `error`: 训练出错
- `stopped`: 训练已停止

---

### 4. 停止训练任务

**端点**: `POST /api/stop_training/<session_id>`

**请求**:
```
POST /api/stop_training/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**响应**:
```json
{
  "success": true,
  "message": "训练任务已停止"
}
```

---

## 兼容性检查接口

### 1. 获取兼容性信息

**端点**: `GET /api/compatibility/info`

**请求**: 无参数

**响应**:
```json
{
  "success": true,
  "data": {
    "policy_problem_compatibility": {
      "attention": ["tsp", "atsp", "mtsp", "cvrp", "sdvrp", "vrptw"],
      "pomo": ["tsp", "mtsp", "cvrp"]
    },
    "algorithm_problem_compatibility": {
      "reinforce": ["tsp", "atsp", "mtsp", "cvrp", "sdvrp", "vrptw"],
      "ppo": ["tsp", "atsp", "mtsp", "cvrp", "sdvrp", "vrptw"],
      "a2c": ["tsp", "atsp", "mtsp", "cvrp", "sdvrp", "vrptw"]
    },
    "policy_algorithm_compatibility": {
      "attention": ["reinforce", "ppo", "a2c"],
      "pomo": ["reinforce", "ppo", "a2c"]
    }
  }
}
```

---

### 2. 获取问题约束

**端点**: `GET /api/compatibility/constraints/<problem>`

**请求**:
```
GET /api/compatibility/constraints/mtsp
```

**响应**:
```json
{
  "success": true,
  "data": {
    "problem": "mtsp",
    "available_policies": ["attention", "pomo"],
    "available_algorithms": ["reinforce", "ppo", "a2c"]
  }
}
```

---

### 3. 验证配置组合

**端点**: `POST /api/compatibility/validate`

**请求体**:
```json
{
  "problem": "atsp",
  "model": "pomo",
  "algorithm": "ppo"
}
```

**响应** (有效配置):
```json
{
  "success": true,
  "valid": true,
  "warnings": [
    {
      "message": "POMO不支持ATSP（非对称距离）。请使用Attention Model",
      "severity": "error"
    }
  ]
}
```

**响应** (无效配置):
```json
{
  "success": true,
  "valid": false,
  "warnings": [...]
}
```

---

### 4. 获取推荐配置

**端点**: `GET /api/compatibility/recommend`

**查询参数**:
- `problem`: 问题类型（必需）
- `preference`: 偏好，可选值：`best`, `fast`, `simple`（可选，默认 `best`）

**请求**:
```
GET /api/compatibility/recommend?problem=mtsp&preference=best
```

**响应**:
```json
{
  "success": true,
  "recommended": {
    "policy": "pomo",
    "algorithm": "ppo"
  },
  "description": "最佳质量配置"
}
```

---

## 数据集管理接口

### 1. 上传数据集

**端点**: `POST /api/upload_dataset`

**请求类型**: `multipart/form-data`

**请求体**:
```
FormData {
  file: <File Object>,
  problem_type: "tsp"
}
```

**响应** (成功):
```json
{
  "success": true,
  "message": "数据集上传成功",
  "dataset_id": "d123abc",
  "filename": "cities_50.json",
  "num_instances": 50
}
```

**支持的文件格式**:
- JSON: `{ "coordinates": [[x1, y1], [x2, y2], ...] }`
- TSP: TSPLIB 格式文件
- TXT: 每行一个坐标点

---

### 2. 列出数据集

**端点**: `GET /api/list_datasets`

**请求**: 无参数

**响应**:
```json
{
  "success": true,
  "datasets": [
    {
      "dataset_id": "d123abc",
      "filename": "cities_50.json",
      "problem_type": "tsp",
      "num_instances": 50,
      "upload_time": "2026-02-04 10:30:00"
    }
  ]
}
```

---

### 3. 删除数据集

**端点**: `POST /api/delete_dataset`

**请求体**:
```json
{
  "dataset_id": "d123abc"
}
```

**响应**:
```json
{
  "success": true,
  "message": "数据集已删除"
}
```

---

## 文件管理接口

### 1. 获取用户文件

**端点**: `GET /api/user_files`

**查询参数**:
- `file_type`: 文件类型（可选），可选值：`plot`, `animation`, `checkpoint`, `curve`
- `limit`: 返回数量限制（可选），默认 50

**请求**:
```
GET /api/user_files?file_type=plot&limit=20
```

**响应**:
```json
{
  "success": true,
  "files": [
    {
      "file_id": 123,
      "filename": "tsp_comparison_1.png",
      "file_type": "plot",
      "file_path": "/static/model_plots/user_1/tsp_comparison_1.png",
      "created_at": "2026-02-04 10:30:00",
      "session_id": "a1b2c3d4..."
    }
  ]
}
```

---

### 2. 获取会话文件

**端点**: `GET /api/session_files/<session_id>`

**请求**:
```
GET /api/session_files/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**响应**:
```json
{
  "success": true,
  "files": [
    {
      "file_id": 123,
      "filename": "tsp_comparison_1.png",
      "file_type": "plot",
      "file_path": "/static/model_plots/user_1/tsp_comparison_1.png"
    }
  ]
}
```

---

## SSE 实时消息

### 消息类型

SSE（Server-Sent Events）用于实时推送训练进度。前端通过 `EventSource` 订阅消息。

**连接方式**:
```javascript
const eventSource = new EventSource(`/api/training_progress/${sessionId}`);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
```

### 消息格式

所有消息都是 JSON 格式，包含 `type` 字段标识消息类型。

---

### 1. info - 信息消息

**用途**: 显示训练过程中的信息

**格式**:
```json
{
  "type": "info",
  "message": "开始训练 POMO 模型，问题类型: MTSP"
}
```

**前端处理**: 显示在训练日志中

---

### 2. warning - 警告消息

**用途**: 显示非致命性警告

**格式**:
```json
{
  "type": "warning",
  "message": "无法从trainer获取loss和reward，将使用累积的batch指标"
}
```

**前端处理**: 以警告样式显示在训练日志中

---

### 3. error - 错误消息

**用途**: 显示错误信息

**格式**:
```json
{
  "type": "error",
  "message": "训练出错: 'MTSPTrainer' object has no attribute 'plots_dir'"
}
```

**前端处理**: 
- 以错误样式显示在训练日志中
- 停止训练进度条
- 关闭 EventSource 连接

---

### 4. progress - 训练进度

**用途**: 更新训练进度

**格式**:
```json
{
  "type": "progress",
  "epoch": 5,
  "total_epochs": 10,
  "progress": 50,
  "loss": 1.2345,
  "reward": -5.6789,
  "best_reward": -5.6789,
  "plot_url": "/static/model_plots/user_1/training_curves_abc123.png"
}
```

**字段说明**:
- `epoch`: 当前训练轮次
- `total_epochs`: 总训练轮次
- `progress`: 进度百分比（0-100）
- `loss`: 当前 Loss 值
- `reward`: 当前 Reward 值
- `best_reward`: 历史最佳 Reward 值
- `plot_url`: 训练曲线图片 URL（可选）

**前端处理**:
- 更新进度条
- 更新 Loss/Reward 显示
- 更新训练曲线图片

---

### 5. plot - 图表更新

**用途**: 通知前端有新的图表生成

**格式**:
```json
{
  "type": "plot",
  "plot_url": "/static/model_plots/user_1/training_curves_abc123.png",
  "message": "Epoch 5 训练曲线已更新"
}
```

**前端处理**: 更新训练曲线图片（添加时间戳避免缓存）

---

### 6. complete - 训练完成

**用途**: 通知训练完成并返回结果

**格式**:
```json
{
  "type": "complete",
  "message": "训练完成！",
  "results": {
    "model": "pomo",
    "problem": "mtsp",
    "strategy": "PPO",
    "total_epochs": 10,
    "final_loss": 1.2345,
    "final_reward": -5.6789,
    "best_reward": -5.6789,
    "plot_paths": [
      "/static/model_plots/user_1/mtsp_comparison_1.png",
      "/static/model_plots/user_1/mtsp_comparison_2.png"
    ],
    "animation_paths": [
      "/static/model_plots/user_1/mtsp_animation_1.gif",
      "/static/model_plots/user_1/mtsp_animation_2.gif"
    ],
    "training_curve": "/static/model_plots/user_1/training_curves_abc123.png",
    "checkpoint_path": "/full/path/to/checkpoint.ckpt"
  }
}
```

**前端处理**:
- 停止进度条，显示为 100%
- 显示训练结果摘要
- 显示所有可视化图片
- 关闭 EventSource 连接

---

## 数据结构定义

### TrainingConfig - 训练配置

```typescript
interface TrainingConfig {
  // 必需参数
  problem: 'tsp' | 'atsp' | 'mtsp' | 'cvrp' | 'sdvrp' | 'vrptw' | 'pctsp' | 'op';
  model: 'attention' | 'am' | 'pomo';
  algorithm: 'reinforce' | 'ppo' | 'a2c';
  num_loc: number;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  
  // 问题特有参数（可选，根据问题类型）
  num_agents?: number;        // mTSP
  cost_type?: 'minmax' | 'sum';  // mTSP
  vehicle_capacity?: number;  // CVRP
  time_window_size?: number;  // VRPTW
  
  // 策略特有参数（可选）
  num_starts?: number;        // POMO
  embed_dim?: number;         // Attention Model
  num_encoder_layers?: number;  // Attention Model
  num_heads?: number;         // Attention Model
  
  // 算法特有参数（可选）
  clip_ratio?: number;        // PPO
  value_loss_coef?: number;   // A2C
  entropy_coef?: number;      // A2C
}
```

---

### TrainingResults - 训练结果

```typescript
interface TrainingResults {
  model: string;              // 策略模型名称
  problem: string;            // 问题类型
  strategy: string;           // RL算法名称
  total_epochs: number;       // 总训练轮次
  final_loss: number;         // 最终 Loss
  final_reward: number;       // 最终 Reward
  best_reward: number;        // 最佳 Reward
  
  // 可视化资源（可选）
  plot_paths?: string[];      // 静态对比图路径列表
  animation_paths?: string[]; // 动画路径列表
  training_curve?: string;    // 训练曲线路径
  checkpoint_path?: string;   // 检查点文件路径
}
```

---

### User - 用户信息

```typescript
interface User {
  user_id: number;
  username: string;
  email: string;
  created_at: string;  // ISO 8601 格式
}
```

---

### File - 文件信息

```typescript
interface File {
  file_id: number;
  filename: string;
  file_type: 'plot' | 'animation' | 'checkpoint' | 'curve';
  file_path: string;
  created_at: string;  // ISO 8601 格式
  session_id?: string;
}
```

---

### Dataset - 数据集信息

```typescript
interface Dataset {
  dataset_id: string;
  filename: string;
  problem_type: string;
  num_instances: number;
  upload_time: string;  // ISO 8601 格式
}
```

---

## 错误处理

### HTTP 状态码

| 状态码 | 说明 | 常见原因 |
|-------|------|---------|
| 200 | 成功 | 请求成功处理 |
| 400 | 错误请求 | 参数验证失败、格式错误 |
| 401 | 未认证 | 未登录或 Session 过期 |
| 403 | 禁止访问 | 权限不足 |
| 404 | 未找到 | 资源不存在 |
| 500 | 服务器错误 | 服务器内部错误 |

---

### 错误响应格式

**标准错误响应**:
```json
{
  "success": false,
  "error": "错误描述信息"
}
```

**带详细信息的错误响应**:
```json
{
  "success": false,
  "error": "配置验证失败",
  "details": {
    "field": "model",
    "value": "pomo",
    "reason": "POMO不支持ATSP问题"
  }
}
```

---

### 常见错误

#### 1. 认证错误

**错误**: 401 Unauthorized

**原因**: 
- Session 已过期
- 未登录就访问需要认证的接口

**处理**: 重定向到登录页面

```javascript
if (response.status === 401) {
  window.location.href = '/login';
}
```

---

#### 2. 配置验证失败

**错误**: 400 Bad Request

**响应**:
```json
{
  "success": false,
  "error": "配置验证失败: POMO不支持ATSP问题"
}
```

**处理**: 显示错误信息，让用户修改配置

---

#### 3. 训练任务冲突

**错误**: 400 Bad Request

**响应**:
```json
{
  "success": false,
  "error": "您已有正在运行的训练任务"
}
```

**处理**: 提示用户等待当前任务完成

---

#### 4. 文件上传失败

**错误**: 400 Bad Request

**响应**:
```json
{
  "success": false,
  "error": "不支持的文件格式"
}
```

**处理**: 显示错误信息，提示用户选择正确格式

---

## 示例代码

### 前端完整流程示例

#### 1. 用户登录

```javascript
async function login(username, password) {
  try {
    const response = await fetch('/api/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ username, password })
    });
    
    const result = await response.json();
    
    if (result.success) {
      console.log('登录成功:', result.user);
      // 跳转到主页
      window.location.href = '/';
    } else {
      alert('登录失败: ' + result.error);
    }
  } catch (error) {
    console.error('请求失败:', error);
    alert('网络错误');
  }
}
```

---

#### 2. 启动训练并订阅进度

```javascript
async function startTraining() {
  // 准备训练配置
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
    num_starts: 50
  };
  
  try {
    // 启动训练
    const response = await fetch('/api/start_training', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(config)
    });
    
    const result = await response.json();
    
    if (!result.success) {
      alert('启动失败: ' + result.error);
      return;
    }
    
    const sessionId = result.session_id;
    console.log('训练已启动，Session ID:', sessionId);
    
    // 订阅训练进度
    const eventSource = new EventSource(`/api/training_progress/${sessionId}`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'info':
          console.log('[INFO]', data.message);
          break;
          
        case 'warning':
          console.warn('[WARNING]', data.message);
          break;
          
        case 'error':
          console.error('[ERROR]', data.message);
          eventSource.close();
          break;
          
        case 'progress':
          updateProgress(data);
          break;
          
        case 'plot':
          updatePlot(data.plot_url);
          break;
          
        case 'complete':
          console.log('训练完成!');
          displayResults(data.results);
          eventSource.close();
          break;
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('SSE连接错误:', error);
      eventSource.close();
    };
    
  } catch (error) {
    console.error('请求失败:', error);
    alert('网络错误');
  }
}

function updateProgress(data) {
  // 更新进度条
  const progressBar = document.getElementById('progress-bar');
  progressBar.style.width = data.progress + '%';
  progressBar.textContent = `${data.progress}%`;
  
  // 更新指标
  document.getElementById('current-epoch').textContent = `${data.epoch}/${data.total_epochs}`;
  document.getElementById('current-loss').textContent = data.loss.toFixed(4);
  document.getElementById('current-reward').textContent = data.reward.toFixed(4);
  document.getElementById('best-reward').textContent = data.best_reward.toFixed(4);
}

function updatePlot(plotUrl) {
  // 更新训练曲线图片（添加时间戳避免缓存）
  const img = document.getElementById('training-curve');
  img.src = plotUrl + '?t=' + new Date().getTime();
}

function displayResults(results) {
  // 显示训练结果
  console.log('最终结果:', results);
  
  // 显示可视化图片
  if (results.plot_paths && results.plot_paths.length > 0) {
    results.plot_paths.forEach((path, index) => {
      const img = document.createElement('img');
      img.src = path;
      img.alt = `对比图 ${index + 1}`;
      document.getElementById('results-container').appendChild(img);
    });
  }
  
  // 显示动画
  if (results.animation_paths && results.animation_paths.length > 0) {
    results.animation_paths.forEach((path, index) => {
      const img = document.createElement('img');
      img.src = path;
      img.alt = `动画 ${index + 1}`;
      document.getElementById('animations-container').appendChild(img);
    });
  }
}
```

---

#### 3. 兼容性检查

```javascript
async function checkCompatibility(problem, model, algorithm) {
  try {
    const response = await fetch('/api/compatibility/validate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ problem, model, algorithm })
    });
    
    const result = await response.json();
    
    if (result.valid) {
      console.log('配置有效');
      if (result.warnings && result.warnings.length > 0) {
        result.warnings.forEach(warning => {
          if (warning.severity === 'warning') {
            console.warn(warning.message);
          } else if (warning.severity === 'info') {
            console.info(warning.message);
          }
        });
      }
      return true;
    } else {
      console.error('配置无效');
      result.warnings.forEach(warning => {
        console.error(warning.message);
      });
      return false;
    }
  } catch (error) {
    console.error('验证失败:', error);
    return false;
  }
}
```

---

### 后端示例

#### 1. API 端点实现

```python
from flask import Blueprint, request, jsonify, Response
from auth_module import login_required, get_current_user_id
import json

training_bp = Blueprint('training', __name__)

@training_bp.route('/api/start_training', methods=['POST'])
@login_required
def start_training():
    """启动训练任务"""
    user_id = get_current_user_id()
    config = request.get_json()
    
    # 验证配置
    valid, error_msg = validate_config(config)
    if not valid:
        return jsonify({
            'success': False,
            'error': error_msg
        }), 400
    
    # 生成 session_id
    session_id = str(uuid.uuid4())
    
    # 启动训练线程
    training_thread = Thread(
        target=train_model,
        args=(config, session_id, user_id)
    )
    training_thread.start()
    
    return jsonify({
        'success': True,
        'message': '训练任务已启动',
        'session_id': session_id
    })


@training_bp.route('/api/training_progress/<session_id>')
@login_required
def training_progress(session_id):
    """SSE 推送训练进度"""
    def generate():
        queue = get_training_queue(session_id)
        
        while True:
            try:
                # 从队列获取消息（阻塞）
                message = queue.get(timeout=30)
                yield f"data: {message}\n\n"
                
                # 如果是完成或错误消息，结束推送
                data = json.loads(message)
                if data['type'] in ['complete', 'error']:
                    break
                    
            except Empty:
                # 超时，发送心跳
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')
```

---

#### 2. 发送 SSE 消息

```python
import json
from queue import Queue

def send_message(queue: Queue, msg_type: str, message: str, **kwargs):
    """发送 SSE 消息到队列"""
    msg = {
        'type': msg_type,
        'message': message,
        **kwargs
    }
    queue.put(json.dumps(msg))

# 使用示例
send_message(queue, 'info', '训练开始')
send_message(queue, 'progress', '', epoch=5, total_epochs=10, progress=50, loss=1.23, reward=-5.67)
send_message(queue, 'complete', '训练完成', results={...})
```

---

## 最佳实践

### 前端

1. **错误处理**: 总是检查响应的 `success` 字段
2. **超时处理**: 为长时间运行的请求设置合理的超时
3. **重试机制**: 网络错误时实现重试逻辑
4. **缓存管理**: 图片 URL 添加时间戳避免缓存
5. **连接管理**: SSE 连接在完成或错误时及时关闭

### 后端

1. **参数验证**: 总是验证输入参数
2. **错误捕获**: 捕获异常并返回友好的错误信息
3. **日志记录**: 记录重要操作和错误
4. **资源清理**: 及时关闭数据库连接、队列等资源
5. **并发控制**: 使用锁或队列管理并发训练任务

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2026-02-04 | 初始版本，完整的接口文档 |

---

## 附录

### A. 问题类型参数对照表

| 问题类型 | 必需参数 | 可选参数 |
|---------|---------|---------|
| TSP | num_loc | - |
| ATSP | num_loc | - |
| mTSP | num_loc, num_agents, cost_type | - |
| CVRP | num_loc, vehicle_capacity | - |
| SDVRP | num_loc, vehicle_capacity | allow_split |
| VRPTW | num_loc, vehicle_capacity, time_window_size | - |

### B. 策略模型参数对照表

| 策略 | 参数名 | 类型 | 默认值 | 说明 |
|------|-------|------|--------|------|
| Attention | embed_dim | int | 128 | 嵌入维度 |
| Attention | num_encoder_layers | int | 3 | 编码器层数 |
| Attention | num_heads | int | 8 | 注意力头数 |
| POMO | num_starts | int | 50 | 多起点数量 |

### C. 算法参数对照表

| 算法 | 参数名 | 类型 | 默认值 | 说明 |
|------|-------|------|--------|------|
| REINFORCE | baseline | str | 'rollout' | 基线方法 |
| PPO | clip_ratio | float | 0.2 | 裁剪比率 |
| A2C | value_loss_coef | float | 0.5 | 价值损失系数 |
| A2C | entropy_coef | float | 0.01 | 熵系数 |

---

### D. SSE 消息类型完整列表

| 消息类型 | 必需字段 | 可选字段 | 触发时机 |
|---------|---------|---------|---------|
| info | message | - | 训练过程中的信息输出 |
| warning | message | - | 非致命性警告 |
| error | message | - | 训练出错 |
| progress | epoch, total_epochs, progress, loss, reward, best_reward | plot_url | 每个 epoch 结束 |
| plot | plot_url, message | - | 训练曲线更新 |
| complete | message, results | - | 训练成功完成 |
| heartbeat | - | - | 保持连接活跃（30秒） |

### E. 返回字段名规范

**TrainingResults 返回字典必须包含的字段**:

```python
# ✅ 正确的返回格式（所有训练器必须遵循）
return {
    'plot_paths': [...],        # 静态对比图 URL 列表
    'animation_paths': [...],   # 动画 GIF URL 列表
    'training_curve': '...',    # 训练曲线 URL
    'checkpoint_path': '...'    # 检查点文件路径
}

# ❌ 错误的字段名
return {
    'comparison_paths': [...],  # 错误！前端期望 plot_paths
    'plots': [...],             # 错误！前端期望 plot_paths
    'animations': [...],        # 错误！前端期望 animation_paths
}
```

**为什么字段名必须一致**:

前端代码（`templates/index.html`）硬编码了字段名：

```javascript
// 前端期望的字段
if (results.plot_paths && results.plot_paths.length > 0) {
    results.plot_paths.forEach((path, index) => {
        // 显示静态对比图
    });
}

if (results.animation_paths && results.animation_paths[index]) {
    // 显示对应的动画
}
```

如果后端返回的字段名不是 `plot_paths`，前端将无法显示可视化图片！

### F. 文件路径规范

**本地文件路径 vs URL 路径**:

```python
# 1. 生成文件时使用本地路径
filename = 'mtsp_comparison_1.png'
local_path = os.path.join(self.user_plots_dir, filename)
# local_path 示例: /full/path/to/static/model_plots/user_1/mtsp_comparison_1.png

# 2. 生成文件
create_visualization(data, local_path)

# 3. 保存文件记录时使用本地路径
self.bg_file_manager.save_file_record(
    user_id=self.user_id,
    filename=filename,
    file_path=local_path  # ← 本地路径
)

# 4. 返回给前端时使用 URL 路径
plot_paths.append(f'/static/model_plots/user_{self.user_id}/{filename}')
# URL 路径示例: /static/model_plots/user_1/mtsp_comparison_1.png

# ✅ 正确
return {
    'plot_paths': ['/static/model_plots/user_1/mtsp_comparison_1.png'],  # URL
}

# ❌ 错误
return {
    'plot_paths': ['/full/path/to/static/model_plots/user_1/mtsp_comparison_1.png'],  # 本地路径
}
```

---

## 调试指南

### 1. 检查训练是否启动

```javascript
// 前端：发送请求后检查响应
const response = await fetch('/api/start_training', { ... });
const result = await response.json();
console.log('Session ID:', result.session_id);  // 应该有 session_id
```

### 2. 检查 SSE 连接

```javascript
// 前端：监听连接状态
eventSource.onopen = () => {
  console.log('✅ SSE 连接已建立');
};

eventSource.onerror = (error) => {
  console.error('❌ SSE 连接错误:', error);
};
```

### 3. 检查消息接收

```javascript
// 前端：打印所有接收到的消息
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('[SSE消息]', data.type, data);
};
```

### 4. 检查文件生成

```bash
# 后端：检查文件是否生成
ls -la static/model_plots/user_*/

# 应该看到 .png 和 .gif 文件
```

### 5. 检查返回字段名

```javascript
// 前端：训练完成后打印结果对象
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'complete') {
    console.log('训练结果:', data.results);
    console.log('字段名:', Object.keys(data.results));
    // 应该包含: plot_paths, animation_paths, training_curve
  }
};
```

### 6. 常见问题排查

| 问题 | 可能原因 | 排查方法 |
|------|---------|---------|
| 前端无法显示图片 | 字段名不匹配 | 检查返回的字段名是否为 `plot_paths` |
| SSE 连接失败 | Session 过期 | 检查是否已登录 |
| 训练无响应 | 线程卡死 | 检查后端日志 |
| 图片404 | 路径错误 | 检查 URL 路径格式 |
| 训练立即完成 | 参数错误 | 检查 epochs 是否 > 0 |

---

**文档维护者**: RL4CO Display Team  
**联系方式**: 请通过项目 Issues 提问

**最后更新**: 2026-02-04  
**文档版本**: v1.0  
**状态**: ✅ **完整可用**
