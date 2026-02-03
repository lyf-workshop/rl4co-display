# RL4CO 训练模块 - 模块化架构

## 📁 目录结构

```
modules/rl_training/
├── __init__.py                 # 统一导出接口
├── base_trainer.py             # 通用训练基类和ProgressCallback
├── tsp_trainer.py              # TSP专用训练器
├── cvrp_trainer.py             # CVRP专用训练器
├── training_functions.py       # 🔒 旧版本（保留用于向后兼容）
└── visualizations/             # 可视化模块
    ├── __init__.py
    ├── common.py               # 通用可视化函数
    ├── tsp_viz.py              # TSP可视化
    └── cvrp_viz.py             # CVRP可视化
```

## 🎯 设计思想

### 1. **按问题类型组织**
每种强化学习问题（TSP、CVRP等）都有独立的训练器模块和可视化函数。

### 2. **面向对象的训练器**
- `BaseTrainer`: 提供通用训练逻辑
- `TSPTrainer`: 继承BaseTrainer，实现TSP特有逻辑
- `CVRPTrainer`: 继承BaseTrainer，实现CVRP特有逻辑

### 3. **可视化功能分离**
每种问题类型都有专门的可视化模块：
- `tsp_viz.py`: TSP路线动画、对比图
- `cvrp_viz.py`: CVRP路线动画（显示仓库、载重等）
- `common.py`: 训练曲线等通用可视化

## 📖 使用方法

### 方法1: 使用统一入口（推荐）

```python
from modules.rl_training import real_rl4co_training

# 自动根据问题类型路由到对应训练器
real_rl4co_training(
    config={'problem': 'tsp', 'epochs': 10, ...},
    session_id='xxx',
    user_id=123,
    queue=queue,
    training_status=status_dict,
    get_background_db_func=get_db
)
```

### 方法2: 直接使用具体训练器

```python
from modules.rl_training import train_tsp, train_cvrp

# TSP训练
train_tsp(config, session_id, user_id, queue, training_status, get_db)

# CVRP训练
train_cvrp(config, session_id, user_id, queue, training_status, get_db)
```

### 方法3: 使用训练器类（高级用法）

```python
from modules.rl_training import TSPTrainer

trainer = TSPTrainer(config, session_id, user_id, queue, training_status, get_db)
trainer.train()
```

## 🔧 扩展新问题类型

添加新问题类型（如OP、PCTSP等）的步骤：

### 1. 创建训练器文件
```python
# modules/rl_training/op_trainer.py

from .base_trainer import BaseTrainer
from .visualizations.op_viz import create_op_route_animation, create_op_comparison_plot

class OPTrainer(BaseTrainer):
    def initialize_environment(self):
        from rl4co.envs import OPEnv
        return OPEnv(generator_params={'num_loc': self.num_loc})
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        # 实现OP特有的可视化逻辑
        ...

def train_op(config, session_id, user_id, queue, training_status, get_background_db_func):
    trainer = OPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func)
    trainer.train()
```

### 2. 创建可视化文件
```python
# modules/rl_training/visualizations/op_viz.py

def create_op_route_animation(td, actions, save_path, title="OP路线生成过程", fps=2):
    # 实现OP动画生成逻辑
    ...

def create_op_comparison_plot(env, td, actions_untrained, rewards_untrained, 
                              actions_trained, rewards_trained, save_path, index=1):
    # 实现OP对比图逻辑
    ...
```

### 3. 更新 __init__.py
```python
from .op_trainer import OPTrainer, train_op

def real_rl4co_training(config, session_id, user_id, queue, training_status, get_background_db_func):
    problem_type = config.get('problem', 'tsp').lower()
    
    if problem_type == 'tsp':
        train_tsp(...)
    elif problem_type == 'cvrp':
        train_cvrp(...)
    elif problem_type == 'op':  # 新增
        train_op(...)
    ...
```

## 🎨 可视化函数API

### TSP可视化

```python
from modules.rl_training.visualizations.tsp_viz import (
    create_tsp_route_animation,
    create_tsp_comparison_plot
)

# 创建TSP路线动画
create_tsp_route_animation(
    td=tensor_dict,           # TensorDict包含城市坐标
    actions=action_sequence,  # 访问城市的顺序
    save_path='path.gif',
    title='TSP路线生成过程',
    fps=2
)

# 创建对比图
create_tsp_comparison_plot(
    env=tsp_env,
    td=tensor_dict,
    actions_untrained=random_actions,
    rewards_untrained=random_rewards,
    actions_trained=trained_actions,
    rewards_trained=trained_rewards,
    save_path='comparison.png',
    index=1
)
```

### CVRP可视化

```python
from modules.rl_training.visualizations.cvrp_viz import (
    create_cvrp_route_animation,
    create_cvrp_comparison_plot
)

# CVRP动画（显示仓库、载重等信息）
create_cvrp_route_animation(
    td=tensor_dict,           # 包含客户、仓库、需求等
    actions=action_sequence,
    save_path='path.gif',
    title='CVRP路线生成过程',
    fps=2
)
```

## 🔄 向后兼容

旧代码无需修改，以下导入方式仍然有效：

```python
# 旧版导入（仍然可用）
from modules.rl_training import real_rl4co_training, ProgressCallback, create_route_animation
```

## 📝 类继承关系

```
BaseTrainer (base_trainer.py)
    ├── TSPTrainer (tsp_trainer.py)
    ├── CVRPTrainer (cvrp_trainer.py)
    └── [Future] OPTrainer, PCTSPTrainer, ...
```

每个子类必须实现：
- `initialize_environment()`: 初始化问题环境
- `generate_visualizations()`: 生成可视化结果

可选重写：
- `create_model()`: 自定义模型创建逻辑
- `load_custom_dataset()`: 加载自定义数据集

## 🧪 测试

确保新模块正常工作：

```bash
# 测试TSP训练
python -c "from modules.rl_training import train_tsp; print('TSP模块加载成功')"

# 测试CVRP训练
python -c "from modules.rl_training import train_cvrp; print('CVRP模块加载成功')"

# 测试统一入口
python -c "from modules.rl_training import real_rl4co_training; print('统一入口加载成功')"
```

## 💡 优势

1. **模块化**: 每种问题独立管理，互不干扰
2. **可扩展**: 新增问题类型只需添加新文件
3. **可维护**: 代码结构清晰，易于理解和修改
4. **向后兼容**: 不影响现有代码
5. **专业可视化**: 每种问题有针对性的可视化

## 📚 相关文件

- `app.py`: Flask应用主文件（导入training模块）
- `model_database.py`: 模型知识库
- `auth_module.py`: 用户认证和文件管理

## 🔮 未来计划

- [ ] 添加 OP (Orienteering Problem) 训练器
- [ ] 添加 PCTSP (Prize Collecting TSP) 训练器
- [ ] 添加 SDVRP (Split Delivery VRP) 训练器
- [ ] 添加 MDPP (Multiple Depot Pickup Problem) 训练器
- [ ] 支持更多可视化选项（3D路线、热力图等）
- [ ] 添加训练过程可视化（梯度流、注意力权重等）



