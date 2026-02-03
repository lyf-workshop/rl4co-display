# CVRP (带容量约束的车辆路径问题) 使用指南

## ✅ **CVRP 已完全支持！**

CVRP（Capacitated Vehicle Routing Problem）是TSP的扩展，考虑了车辆容量限制和客户需求。

---

## 🎯 **CVRP 问题描述**

### 基本概念
- **仓库（Depot）**：车辆的起点和终点
- **客户（Customers）**：需要配送的节点，每个客户有需求量
- **车辆容量**：每辆车的最大载重量
- **目标**：在满足容量约束的前提下，最小化总行驶距离

### 与TSP的区别
| 特性 | TSP | CVRP |
|------|-----|------|
| **起点** | 任意城市 | 固定仓库 |
| **返程** | 回到起点 | 必须回仓库补货 |
| **容量约束** | 无 | 有车辆容量限制 |
| **需求** | 无 | 每个客户有需求量 |
| **路径数** | 1条 | 可能多条（多次返回仓库） |

---

## 🚀 **如何使用CVRP训练**

### 1. **前端界面配置**

在训练页面：

```
问题类型：选择 "CVRP (Capacitated Vehicle Routing Problem)"
模型选择：Attention Model / POMO / 等
训练轮数：10-50 epochs（推荐）
批次大小：512（默认）
```

### 2. **后端API调用**

```python
from modules.rl_training import train_cvrp

config = {
    'problem': 'cvrp',
    'model': 'attention',
    'epochs': 10,
    'batch_size': 512,
    'learning_rate': 1e-4,
    'num_loc': 20,              # 客户数量（不含仓库）
    'vehicle_capacity': 1.0,    # 车辆容量
}

train_cvrp(
    config=config,
    session_id=session_id,
    user_id=user_id,
    queue=message_queue,
    training_status=status_dict,
    get_background_db_func=get_db
)
```

### 3. **使用训练器类（高级）**

```python
from modules.rl_training import CVRPTrainer

trainer = CVRPTrainer(
    config=config,
    session_id=session_id,
    user_id=user_id,
    queue=queue,
    training_status=training_status,
    get_background_db_func=get_db
)

# 执行训练
trainer.train()
```

---

## 📊 **CVRP特有可视化**

### 1. **路线动画（GIF）**

CVRP动画会显示：
- 🏭 **仓库**：红色方块标识
- 👥 **客户**：蓝色圆点，旁边显示需求量
- 🚚 **配送路径**：蓝色实线
- 🔄 **返回仓库**：绿色虚线
- 📦 **当前载重**：实时显示车辆当前载重
- 📏 **累计距离**：总行驶距离

**代码示例：**
```python
from modules.rl_training.visualizations.cvrp_viz import create_cvrp_route_animation

create_cvrp_route_animation(
    td=tensor_dict,              # 包含坐标、需求等信息
    actions=action_sequence,     # 访问顺序（0为仓库）
    save_path='cvrp_route.gif',
    title='CVRP配送路线',
    fps=2                        # 帧率
)
```

### 2. **对比图（PNG）**

训练前后路线对比：
- 左图：随机策略（Random）
- 右图：训练后策略（Trained）
- 显示总成本（Cost）

**代码示例：**
```python
from modules.rl_training.visualizations.cvrp_viz import create_cvrp_comparison_plot

create_cvrp_comparison_plot(
    env=cvrp_env,
    td=tensor_dict,
    actions_untrained=random_actions,
    rewards_untrained=random_rewards,
    actions_trained=trained_actions,
    rewards_trained=trained_rewards,
    save_path='cvrp_comparison.png'
)
```

---

## ⚙️ **CVRP配置参数**

### 必需参数
```python
config = {
    'problem': 'cvrp',           # 必须设置为'cvrp'
    'model': 'attention',        # 模型类型
    'epochs': 10,                # 训练轮数
}
```

### 可选参数（CVRP特有）
```python
config = {
    # ... 必需参数 ...
    
    'num_loc': 20,               # 客户数量（默认50，不含仓库）
    'vehicle_capacity': 1.0,     # 车辆容量（默认1.0）
    'num_vehicles': 1,           # 车辆数量（默认1，暂时只支持单车）
    'batch_size': 512,           # 批次大小
    'learning_rate': 1e-4,       # 学习率
}
```

### 参数说明
- **num_loc**：客户数量，实际节点数 = num_loc + 1（仓库）
- **vehicle_capacity**：归一化容量，通常设为1.0
- **需求生成**：自动随机生成，总需求 < 容量 * 车辆数

---

## 🎨 **可视化效果说明**

### 动画特征
```
帧1: 车辆从仓库出发 [载重: 0.00]
帧2: 访问客户1 [载重: 0.15, 需求: 0.15]
帧3: 访问客户2 [载重: 0.32, 需求: 0.17]
...
帧N: 载重接近容量，返回仓库 [虚线]
帧N+1: 重新从仓库出发 [载重: 0.00]
```

### 颜色编码
- 🔴 **红色方块** = 仓库（Depot）
- 🔵 **蓝色圆点** = 客户（Customers）
- 🟠 **橙色星标** = 当前访问节点
- 🔵 **蓝色实线** = 配送路径
- 🟢 **绿色虚线** = 返回仓库

---

## 📝 **完整示例**

### 示例1：基础CVRP训练
```python
from modules.rl_training import train_cvrp
from queue import Queue

# 配置
config = {
    'problem': 'cvrp',
    'model': 'attention',
    'epochs': 20,
    'num_loc': 30,
    'vehicle_capacity': 1.0,
}

# 创建消息队列
queue = Queue()

# 训练状态
training_status = {}

# 启动训练
train_cvrp(
    config=config,
    session_id='cvrp-test-001',
    user_id=1,
    queue=queue,
    training_status=training_status,
    get_background_db_func=lambda: your_db_connection
)

# 监听进度
while True:
    msg = queue.get()
    data = json.loads(msg)
    
    if data['type'] == 'progress':
        print(f"Epoch {data['epoch']}/{data['total_epochs']}")
        print(f"Loss: {data['loss']}, Reward: {data['reward']}")
    
    elif data['type'] == 'complete':
        print("Training completed!")
        results = data['results']
        print(f"Best reward: {results['best_reward']}")
        break
```

### 示例2：Flask API集成
```python
from flask import Flask, request, jsonify, Response
from modules.rl_training import real_rl4co_training
from queue import Queue
import threading
import json

app = Flask(__name__)

@app.route('/api/start_cvrp_training', methods=['POST'])
def start_cvrp_training():
    config = request.json
    config['problem'] = 'cvrp'  # 强制设置为CVRP
    
    session_id = str(uuid.uuid4())
    user_id = get_current_user_id()
    
    queue = Queue()
    training_queues[session_id] = queue
    
    # 启动训练线程
    thread = threading.Thread(
        target=real_rl4co_training,
        args=(config, session_id, user_id, queue, 
              training_status, get_background_db),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'message': 'CVRP training started'
    })

@app.route('/api/cvrp_progress/<session_id>')
def cvrp_progress(session_id):
    """SSE进度推送"""
    def generate():
        queue = training_queues[session_id]
        while True:
            msg = queue.get(timeout=1)
            yield f"data: {msg}\n\n"
            
            data = json.loads(msg)
            if data['type'] in ['complete', 'error']:
                break
    
    return Response(generate(), mimetype='text/event-stream')
```

---

## 🔍 **与TSP的实现对比**

| 组件 | TSP | CVRP |
|------|-----|------|
| **环境** | TSPEnv | CVRPEnv |
| **节点类型** | 城市 | 仓库 + 客户 |
| **约束** | 访问所有节点 | 容量约束 |
| **动作空间** | 选择下一个城市 | 选择客户/返回仓库 |
| **状态** | 当前位置 | 位置 + 当前载重 |
| **奖励** | 负路径长度 | 负路径长度（含多次往返） |
| **可视化** | 单条路径 | 多条路径（虚线分隔） |

---

## 🧪 **测试CVRP功能**

### 快速测试脚本
```python
# test_cvrp.py
from modules.rl_training import CVRPTrainer
from queue import Queue

def test_cvrp():
    config = {
        'problem': 'cvrp',
        'model': 'attention',
        'epochs': 2,  # 快速测试
        'num_loc': 10,
        'vehicle_capacity': 1.0,
    }
    
    queue = Queue()
    training_status = {}
    
    trainer = CVRPTrainer(
        config=config,
        session_id='test',
        user_id=1,
        queue=queue,
        training_status=training_status,
        get_background_db_func=lambda: None
    )
    
    print("Starting CVRP training test...")
    trainer.train()
    print("Test completed!")

if __name__ == '__main__':
    test_cvrp()
```

---

## 💡 **常见问题**

### Q: CVRP训练比TSP慢吗？
**A:** 是的，因为CVRP问题更复杂：
- 需要考虑容量约束
- 可能需要多次返回仓库
- 动作空间更大
- 建议：减少epoch数或使用更小的问题规模

### Q: 如何调整容量约束？
**A:** 
```python
config = {
    'vehicle_capacity': 0.5,  # 更严格的容量限制
    # 或
    'vehicle_capacity': 2.0,  # 更宽松的容量限制
}
```

### Q: 支持多车辆吗？
**A:** 目前主要支持单车辆多次往返。多车辆并行配送需要更复杂的建模。

### Q: 客户需求如何生成？
**A:** 自动随机生成，确保总需求不超过车辆总容量。

### Q: 可以使用自定义数据集吗？
**A:** 目前CVRP仅支持随机生成。自定义数据集需要额外开发（需要包含需求信息）。

---

## 📚 **相关文档**

- [模块架构](./README.md)
- [使用指南](./USAGE_GUIDE.md)
- [TSP vs CVRP对比](./QUICK_REFERENCE.md)
- [可视化文档](./visualizations/README.md)

---

## 🎊 **总结**

✅ **CVRP已完全支持：**
- 完整的训练流程
- 专业的可视化（仓库、载重、需求）
- 容量约束自动处理
- 多次往返路径规划

✅ **使用简单：**
```python
# 只需设置 problem='cvrp'
config = {'problem': 'cvrp', 'epochs': 10}
train_cvrp(config, ...)
```

✅ **前端已启用：**
- 问题类型下拉框可直接选择CVRP
- 不再显示"敬请期待"

**开始训练CVRP吧！** 🚀

---

**最后更新**: 2024年
**维护者**: RL4CO平台开发团队



