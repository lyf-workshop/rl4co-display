# RL4CO训练模块使用指南

## 📚 快速开始

### 方式1：使用统一入口（最简单，推荐）

```python
from modules.rl_training import real_rl4co_training

# 配置训练参数
config = {
    'problem': 'tsp',           # 问题类型：'tsp', 'cvrp'
    'model': 'attention',       # 模型类型
    'epochs': 10,               # 训练轮数
    'batch_size': 512,          # 批次大小
    'learning_rate': 1e-4,      # 学习率
}

# 启动训练（自动路由到对应的训练器）
real_rl4co_training(
    config=config,
    session_id='training-session-123',
    user_id=1,
    queue=message_queue,
    training_status=global_status_dict,
    get_background_db_func=get_db_connection
)
```

### 方式2：直接使用具体训练器

```python
from modules.rl_training import train_tsp, train_cvrp

# TSP训练
train_tsp(config, session_id, user_id, queue, training_status, get_db)

# CVRP训练
train_cvrp(config, session_id, user_id, queue, training_status, get_db)
```

### 方式3：使用训练器类（高级用法）

```python
from modules.rl_training import TSPTrainer, CVRPTrainer

# 创建训练器实例
trainer = TSPTrainer(
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

## 🎨 可视化使用

### TSP路线动画

```python
from modules.rl_training.visualizations.tsp_viz import create_tsp_route_animation
import torch

# 假设你有TSP环境和训练好的模型
td = env.reset(batch_size=[1])
actions = model.policy(td, phase="test", decode_type="greedy", return_actions=True)['actions']

# 创建动画
create_tsp_route_animation(
    td=td,
    actions=actions[0].cpu().numpy(),
    save_path='output/tsp_route.gif',
    title='TSP路线生成过程',
    fps=2  # 每秒2帧
)
```

### CVRP路线动画

```python
from modules.rl_training.visualizations.cvrp_viz import create_cvrp_route_animation

# CVRP动画会显示仓库、客户、需求等信息
create_cvrp_route_animation(
    td=td,
    actions=actions[0].cpu().numpy(),
    save_path='output/cvrp_route.gif',
    title='CVRP配送路线',
    fps=2
)
```

### 训练曲线绘制

```python
from modules.rl_training.visualizations.common import create_training_curve_plot

create_training_curve_plot(
    history_epochs=[1, 2, 3, 4, 5],
    history_losses=[2.5, 2.0, 1.5, 1.2, 1.0],
    history_rewards=[-15, -12, -10, -8, -7],
    save_path='output/training_curves.png',
    best_reward=-7
)
```

### 对比图生成

```python
from modules.rl_training.visualizations.tsp_viz import create_tsp_comparison_plot

# 生成训练前后对比图
create_tsp_comparison_plot(
    env=tsp_env,
    td=td,
    actions_untrained=random_actions,
    rewards_untrained=random_rewards,
    actions_trained=trained_actions,
    rewards_trained=trained_rewards,
    save_path='output/comparison.png',
    index=1
)
```

## 🔧 在Flask应用中使用

### app.py 中的集成示例

```python
from flask import Flask, request, jsonify
from modules.rl_training import real_rl4co_training
from queue import Queue
import threading
import uuid

app = Flask(__name__)

# 全局变量
training_status = {}
training_queues = {}

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """启动训练API"""
    config = request.json
    session_id = str(uuid.uuid4())
    user_id = get_current_user_id()  # 从session获取
    
    # 创建消息队列
    training_queues[session_id] = Queue()
    
    # 启动训练线程
    training_thread = threading.Thread(
        target=real_rl4co_training,
        args=(config, session_id, user_id, training_queues[session_id], 
              training_status, get_background_db),
        daemon=True
    )
    training_thread.start()
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'message': '训练已启动'
    })

@app.route('/api/training_progress/<session_id>')
def training_progress(session_id):
    """获取训练进度（SSE）"""
    def generate():
        queue = training_queues[session_id]
        while True:
            try:
                message = queue.get(timeout=1)
                yield f"data: {message}\n\n"
                
                data = json.loads(message)
                if data['type'] in ['complete', 'error']:
                    break
            except:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')
```

## 📖 配置参数说明

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `problem` | str | 'tsp' | 问题类型：'tsp', 'cvrp' |
| `model` | str | 'attention' | 模型类型：'attention', 'pomo', 'symncco' 等 |
| `epochs` | int | 3 | 训练轮数 |
| `batch_size` | int | 512 | 批次大小 |
| `learning_rate` | float | 1e-4 | 学习率 |

### TSP特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_mode` | str | 'random' | 数据集模式：'random'（随机生成）或'upload'（上传数据集） |
| `dataset_id` | str | None | 上传数据集的ID（当dataset_mode='upload'时） |
| `num_loc` | int | 50 | 城市数量（随机生成时） |

### CVRP特有参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vehicle_capacity` | float | 1.0 | 车辆容量 |
| `num_vehicles` | int | 1 | 车辆数量 |
| `num_loc` | int | 50 | 客户数量 |

## 🎯 训练流程说明

### 1. 初始化阶段
- 创建用户专属目录
- 初始化数据库连接
- 设置设备（GPU/CPU）

### 2. 环境创建
- 根据问题类型创建对应环境（TSPEnv/CVRPEnv）
- 加载自定义数据集（如果有）

### 3. 模型构建
- 创建策略网络（AttentionModelPolicy）
- 创建RL模型（REINFORCE）
- 加载checkpoint（如果存在）

### 4. 训练执行
- 使用RL4COTrainer进行训练
- ProgressCallback实时推送进度
- 每个epoch结束时生成训练曲线

### 5. 可视化生成
- 生成路线对比图（训练前vs训练后）
- 生成路线动画GIF
- 保存checkpoint

### 6. 结果存储
- 保存到用户专属目录
- 记录到数据库（文件表、会话表）
- 返回结果路径

## 🔍 进度消息类型

训练过程中通过队列发送的消息类型：

### info
```json
{
    "type": "info",
    "message": "开始训练..."
}
```

### progress
```json
{
    "type": "progress",
    "epoch": 5,
    "total_epochs": 10,
    "progress": 50.0,
    "loss": 1.234,
    "reward": -8.56,
    "best_reward": -7.89,
    "plot_url": "/static/model_plots/user_1/training_curves_abc123.png"
}
```

### plot
```json
{
    "type": "plot",
    "plot_url": "/static/model_plots/user_1/tsp_comparison_abc123_1.png",
    "message": "对比图已生成"
}
```

### complete
```json
{
    "type": "complete",
    "message": "训练完成！",
    "results": {
        "model": "attention",
        "problem": "tsp",
        "strategy": "REINFORCE",
        "total_epochs": 10,
        "final_loss": 1.0,
        "final_reward": -7.5,
        "best_reward": -7.2,
        "plot_paths": ["..."],
        "animation_paths": ["..."],
        "checkpoint_path": "..."
    }
}
```

### error
```json
{
    "type": "error",
    "message": "训练出错: ..."
}
```

## 🚀 扩展新问题类型

### 步骤1: 创建训练器文件

```bash
# 复制模板
cp modules/rl_training/templates/problem_trainer_template.py \
   modules/rl_training/op_trainer.py
```

编辑 `op_trainer.py`，替换所有 `[PROBLEM_NAME]` 为 `OP`。

### 步骤2: 创建可视化文件

```bash
# 复制模板
cp modules/rl_training/templates/visualization_template.py \
   modules/rl_training/visualizations/op_viz.py
```

编辑 `op_viz.py`，实现OP特有的可视化逻辑。

### 步骤3: 更新导出

在 `modules/rl_training/__init__.py` 中添加：

```python
from .op_trainer import OPTrainer, train_op

def real_rl4co_training(...):
    problem_type = config.get('problem', 'tsp').lower()
    
    if problem_type == 'op':
        train_op(config, session_id, user_id, queue, training_status, get_background_db_func)
    # ... 其他问题类型
```

在 `modules/rl_training/visualizations/__init__.py` 中添加：

```python
from .op_viz import create_op_route_animation, create_op_comparison_plot

__all__ = [
    # ... 现有导出
    'create_op_route_animation',
    'create_op_comparison_plot',
]
```

## 🧪 测试模块

运行测试脚本验证模块正确性：

```bash
python modules/rl_training/test_module.py
```

测试内容包括：
- ✅ 文件结构检查
- ✅ 模块导入测试
- ✅ 向后兼容性验证
- ✅ 使用示例展示

## 💡 常见问题

### Q: 如何切换问题类型？
A: 只需修改配置中的 `problem` 参数：
```python
config = {'problem': 'tsp'}  # 或 'cvrp'
```

### Q: 如何使用自定义TSP数据集？
A: 设置 `dataset_mode` 和 `dataset_id`：
```python
config = {
    'problem': 'tsp',
    'dataset_mode': 'upload',
    'dataset_id': 'your-dataset-id'
}
```

### Q: 可视化文件保存在哪里？
A: 保存在用户专属目录：
- 图片/动画：`static/model_plots/user_{user_id}/`
- Checkpoint：`checkpoints/user_{user_id}/`

### Q: 如何调整动画帧率？
A: 修改 `fps` 参数：
```python
create_tsp_route_animation(..., fps=5)  # 5帧/秒，更快
```

### Q: 训练过程中如何获取实时进度？
A: 通过消息队列接收进度更新，或使用SSE流式传输到前端。

## 📝 更多资源

- [模块README](./README.md) - 详细的架构说明
- [可视化README](./visualizations/README.md) - 可视化函数文档
- [问题训练器模板](./templates/problem_trainer_template.py)
- [可视化函数模板](./templates/visualization_template.py)

## 🤝 贡献指南

欢迎贡献新的问题类型训练器！请遵循以下步骤：

1. 使用模板创建新训练器
2. 实现必要的方法
3. 添加测试用例
4. 更新文档
5. 提交PR

---

**最后更新**: 2024年
**维护者**: RL4CO平台开发团队



