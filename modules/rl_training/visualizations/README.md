# 可视化函数模块

本目录包含不同强化学习问题类型的可视化函数。

## 📁 文件说明

### common.py - 通用可视化
- `create_training_curve_plot()`: 创建训练Loss和Reward曲线图
- 适用于所有问题类型的通用可视化

### tsp_viz.py - TSP可视化
- `create_tsp_route_animation()`: 创建TSP路线逐步构建的GIF动画
- `create_tsp_comparison_plot()`: 创建训练前后路线对比图

**特点**：
- 显示城市节点和编号
- 高亮起点和当前访问节点
- 实时显示累计成本
- 带箭头的路径方向指示

### cvrp_viz.py - CVRP可视化
- `create_cvrp_route_animation()`: 创建CVRP路线逐步构建的GIF动画
- `create_cvrp_comparison_plot()`: 创建训练前后路线对比图

**特点**：
- 区分仓库（depot）和客户（customers）
- 显示每个客户的需求量
- 实时显示当前车辆载重
- 返回仓库使用虚线标识

## 🎨 使用示例

### TSP动画
```python
from modules.rl_training.visualizations.tsp_viz import create_tsp_route_animation

create_tsp_route_animation(
    td=tensor_dict,           # TensorDict包含城市坐标
    actions=[0, 3, 1, 4, 2],  # 访问顺序
    save_path='tsp_route.gif',
    title='TSP路线生成过程',
    fps=2                     # 每秒2帧
)
```

### CVRP动画
```python
from modules.rl_training.visualizations.cvrp_viz import create_cvrp_route_animation

create_cvrp_route_animation(
    td=tensor_dict,           # 包含坐标、需求等
    actions=[0, 3, 1, 0, 4, 2, 0],  # 访问顺序（0为仓库）
    save_path='cvrp_route.gif',
    title='CVRP配送路线',
    fps=2
)
```

### 训练曲线
```python
from modules.rl_training.visualizations.common import create_training_curve_plot

create_training_curve_plot(
    history_epochs=[1, 2, 3, 4, 5],
    history_losses=[2.5, 2.0, 1.5, 1.2, 1.0],
    history_rewards=[-15, -12, -10, -8, -7],
    save_path='training_curves.png',
    best_reward=-7
)
```

## 🔧 添加新问题类型可视化

创建新的可视化文件，例如 `op_viz.py`:

```python
"""
OP (Orienteering Problem) 可视化函数
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def create_op_route_animation(td, actions, save_path, title="OP路线", fps=2):
    """创建OP路线动画"""
    # 提取节点坐标和奖励
    locs = td['locs'].cpu().numpy()
    prizes = td.get('prizes', None)
    if prizes is not None:
        prizes = prizes.cpu().numpy()
    
    # 实现动画逻辑...
    frames = []
    
    for step in range(len(actions) + 1):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制节点（大小根据奖励值缩放）
        if prizes is not None:
            sizes = prizes * 300
            ax.scatter(locs[:, 0], locs[:, 1], s=sizes, c='lightblue', alpha=0.6)
        
        # 绘制路径...
        # 保存帧...
        
    # 生成GIF...
    
def create_op_comparison_plot(env, td, actions_untrained, rewards_untrained, 
                              actions_trained, rewards_trained, save_path, index=1):
    """创建OP对比图"""
    # 实现对比图逻辑...
```

然后在 `__init__.py` 中导出：

```python
from .op_viz import create_op_route_animation, create_op_comparison_plot

__all__ = [
    # ... 现有导出
    'create_op_route_animation',
    'create_op_comparison_plot',
]
```

## 📊 可视化效果说明

### TSP动画特征
- ✅ 城市节点用浅蓝色圆点表示
- ✅ 起点用绿色方块标识
- ✅ 当前访问节点用红色星标高亮
- ✅ 路径用蓝色实线和箭头连接
- ✅ 实时显示累计距离
- ✅ 进度条显示完成百分比

### CVRP动画特征
- ✅ 仓库用红色方块表示
- ✅ 客户用浅蓝色圆点表示
- ✅ 节点旁显示需求量
- ✅ 配送路径用蓝色实线
- ✅ 返回仓库用绿色虚线
- ✅ 实时显示车辆载重和总距离

## 🎯 未来扩展

计划添加的可视化类型：

- [ ] `op_viz.py` - Orienteering Problem
- [ ] `pctsp_viz.py` - Prize Collecting TSP
- [ ] `sdvrp_viz.py` - Split Delivery VRP
- [ ] `mdpp_viz.py` - Multiple Depot Pickup Problem
- [ ] 3D路线可视化
- [ ] 注意力权重热力图
- [ ] 训练过程动画（梯度流、学习曲线等）



