# 新增问题类型完整指南

## 📋 文档概述

本文档详细说明如何在 RL4CO Display 平台中添加新的问题类型。基于 mTSP 的实际实现经验，包含所有必需步骤、常见错误及其解决方案。

**建议阅读时间**: 30分钟  
**实施时间**: 2-4小时（取决于问题复杂度）

---

## 📖 目录

1. [概述](#概述)
2. [准备工作](#准备工作)
3. [实施步骤](#实施步骤)
   - [步骤1: 创建问题定义](#步骤1-创建问题定义)
   - [步骤2: 创建训练器](#步骤2-创建训练器)
   - [步骤3: 创建可视化函数](#步骤3-创建可视化函数)
   - [步骤4: 更新兼容性配置](#步骤4-更新兼容性配置)
   - [步骤5: 注册到系统](#步骤5-注册到系统)
   - [步骤6: 前端集成](#步骤6-前端集成)
   - [步骤7: 测试验证](#步骤7-测试验证)
4. [常见错误及解决方案](#常见错误及解决方案)
5. [完整示例: mTSP](#完整示例-mtsp)
6. [检查清单](#检查清单)

---

## 概述

### 系统架构

```
用户输入 → 前端 → 后端路由 → 训练器 → 环境/模型/算法 → 可视化 → 前端显示
```

**添加新问题类型需要修改的位置**:
1. `modules/problems/` - 问题定义（可选，如果使用 RL4CO 内置环境）
2. `modules/rl_training/` - 训练器实现（必需）
3. `modules/rl_training/visualizations/` - 可视化函数（必需）
4. `modules/compatibility.py` - 兼容性配置（必需）
5. `app.py` 或 `app_training.py` - 路由注册（必需）
6. `templates/index.html` - 前端界面（必需）

### 涉及的核心类

```python
BaseTrainer           # 所有训练器的基类
  ├── initialize_environment()     # 初始化环境（必须实现）
  ├── generate_visualizations()   # 生成可视化（必须实现）
  └── train()                      # 训练流程（已实现，无需修改）
```

---

## 准备工作

### 1. 确认 RL4CO 支持

检查问题类型是否在 RL4CO 中实现：

```python
# 查看 RL4CO 支持的环境
from rl4co.envs import *

# 常见环境：
# - TSPEnv, ATSPEnv, MTSPEnv
# - CVRPEnv, SDVRPEnv
# - VRPTWEnv
# - PCTSPEnv, OPEnv
```

### 2. 了解问题特性

- **问题特有参数**: 如 mTSP 的 `num_agents`, `cost_type`
- **环境初始化方式**: 参数名称和取值范围
- **动作空间**: 离散还是连续
- **奖励计算**: 最小化成本还是最大化奖励

### 3. 确定可视化需求

- 静态对比图（必需）
- 动态路线动画（推荐）
- 训练曲线（自动生成）

---

## 实施步骤

## 步骤1: 创建问题定义

**路径**: `modules/problems/your_problem.py`

### 1.1 创建文件（如果需要自定义）

如果使用 RL4CO 内置环境，这一步**可以跳过**。

```python
"""
YOUR_PROBLEM 问题定义
"""
from .base_problem import BaseProblem

class YourProblem(BaseProblem):
    """YOUR_PROBLEM 问题类"""
    
    def __init__(self, **kwargs):
        super().__init__(
            name="your_problem",
            description="问题描述",
            **kwargs
        )
    
    def validate_config(self, config):
        """验证配置参数"""
        # 验证必需参数
        required = ['param1', 'param2']
        for param in required:
            if param not in config:
                raise ValueError(f"缺少必需参数: {param}")
        
        return True
```

### 1.2 注册到问题注册表

**文件**: `modules/problems/__init__.py`

```python
from .your_problem import YourProblem

PROBLEM_REGISTRY = {
    'tsp': TSPProblem,
    'mtsp': MTSPProblem,
    'your_problem': YourProblem,  # ✅ 添加这里
}

def get_problem_class(problem_name):
    """获取问题类"""
    problem_name = problem_name.lower()
    if problem_name not in PROBLEM_REGISTRY:
        raise ValueError(f"未知问题类型: {problem_name}")
    return PROBLEM_REGISTRY[problem_name]
```

---

## 步骤2: 创建训练器

**路径**: `modules/rl_training/your_problem_trainer.py`

这是**最关键**的步骤！

### 2.1 创建训练器类

```python
"""
YOUR_PROBLEM 问题专用训练器
"""
import os
import torch
import numpy as np
from datetime import datetime

try:
    from rl4co.envs import YourProblemEnv  # ✅ 导入对应的环境
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    print("警告: RL4CO 库未安装或不支持 YourProblemEnv")

from .base_trainer import BaseTrainer
from .visualizations.your_problem_viz import (
    create_your_problem_animation, 
    create_your_problem_comparison_plot
)


class YourProblemTrainer(BaseTrainer):
    """YOUR_PROBLEM 问题训练器"""
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func):
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func)
        
        # ✅ 提取问题特有参数
        self.param1 = config.get('param1', default_value1)
        self.param2 = config.get('param2', default_value2)
        
        # 发送配置信息
        self.send_message('info', f'📋 YOUR_PROBLEM配置: param1={self.param1}, param2={self.param2}')
    
    def initialize_environment(self):
        """
        初始化 YOUR_PROBLEM 环境
        
        ⚠️ 关键点：
        1. 方法名必须是 initialize_environment（不是 create_environment）
        2. 必须返回环境对象
        3. 参数名必须与 RL4CO 的环境类一致
        """
        if not RL4CO_AVAILABLE:
            raise ImportError("RL4CO 库未安装")
        
        # ✅ 创建环境，传入所有必需参数
        env = YourProblemEnv(
            num_loc=self.num_loc,
            param1=self.param1,
            param2=self.param2,
            # ... 其他参数
        )
        
        self.send_message('info', f'✅ YOUR_PROBLEM环境创建成功: {self.num_loc}个节点')
        return env
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """
        生成 YOUR_PROBLEM 可视化
        
        ⚠️ 关键点：
        1. 必须返回字典，不能返回元组
        2. 字段名必须是: plot_paths, animation_paths, training_curve, checkpoint_path
        3. 路径必须是 URL 路径（/static/...），不是本地路径
        
        参数:
            env: 环境对象
            model: 训练好的模型
            trainer: PyTorch Lightning 训练器
            checkpoint_path: 检查点保存路径
        
        返回:
            dict: {
                'plot_paths': [...],        # 静态对比图 URL 列表
                'animation_paths': [...],   # 动画 URL 列表
                'training_curve': '...',    # 训练曲线 URL
                'checkpoint_path': '...'    # 检查点文件路径
            }
        """
        try:
            self.send_message('info', '🎨 开始生成 YOUR_PROBLEM 可视化...')
            
            device = next(model.parameters()).device
            model.eval()
            
            # ========== 生成测试数据 ==========
            num_test_instances = min(3, self.batch_size)
            td = env.reset(batch_size=[num_test_instances])
            td = td.to(device)
            
            # ========== 使用模型生成路径 ==========
            with torch.no_grad():
                out = model(td.clone(), phase='test', decode_type='greedy')
            
            # ========== 提取结果 ==========
            actions = out['actions'].cpu().numpy()
            rewards = out['reward'].cpu().numpy()
            locs = td['locs'].cpu().numpy()
            
            # ========== 创建可视化 ==========
            animation_paths = []
            comparison_paths = []
            
            for i in range(num_test_instances):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                anim_filename = f'your_problem_animation_{i+1}_{timestamp}.gif'
                comp_filename = f'your_problem_comparison_{i+1}_{timestamp}.png'
                
                anim_path = os.path.join(self.user_plots_dir, anim_filename)
                comp_path = os.path.join(self.user_plots_dir, comp_filename)
                
                # ⚠️ 关键：正确处理数据维度
                instance_locs = locs[i]         # shape: (num_loc, 2) ✅
                instance_actions = actions[i]   # shape: (action_length,) ✅
                
                # 创建动画
                try:
                    create_your_problem_animation(
                        td={'locs': torch.from_numpy(instance_locs)},  # ✅ 2D 数组
                        actions=instance_actions,
                        save_path=anim_path,
                        title=f'YOUR_PROBLEM 路线生成 - 问题{i+1}',
                        fps=2
                    )
                    # ✅ 添加 URL 路径，不是本地路径
                    animation_paths.append(f"/static/model_plots/user_{self.user_id}/{anim_filename}")
                    self.send_message('info', f'✅ 动画 {i+1} 生成成功')
                    
                    # 保存文件记录到数据库
                    if self.bg_file_manager:
                        try:
                            self.bg_file_manager.save_file_record(
                                user_id=self.user_id,
                                session_id=self.session_id,
                                filename=anim_filename,
                                file_type='animation',
                                file_path=anim_path  # 数据库记录用本地路径
                            )
                        except Exception as e:
                            print(f"保存动画记录失败: {str(e)}")
                except Exception as e:
                    self.send_message('info', f'⚠️ 动画 {i+1} 生成失败: {str(e)}')
                
                # 创建对比图
                try:
                    create_your_problem_comparison_plot(
                        td={'locs': torch.from_numpy(instance_locs)},  # ✅ 2D 数组
                        actions=instance_actions,
                        save_path=comp_path,
                        cost=-rewards[i],
                        title=f'YOUR_PROBLEM 路线对比 - 问题{i+1}'
                    )
                    # ✅ 添加 URL 路径
                    comparison_paths.append(f"/static/model_plots/user_{self.user_id}/{comp_filename}")
                    self.send_message('info', f'✅ 对比图 {i+1} 生成成功')
                    
                    # 保存文件记录
                    if self.bg_file_manager:
                        try:
                            self.bg_file_manager.save_file_record(
                                user_id=self.user_id,
                                session_id=self.session_id,
                                filename=comp_filename,
                                file_type='plot',
                                file_path=comp_path
                            )
                        except Exception as e:
                            print(f"保存对比图记录失败: {str(e)}")
                except Exception as e:
                    self.send_message('info', f'⚠️ 对比图 {i+1} 生成失败: {str(e)}')
            
            self.send_message('info', f'🎉 可视化完成: {len(animation_paths)}个动画, {len(comparison_paths)}个对比图')
            
            # 保存检查点
            trainer.save_checkpoint(checkpoint_path)
            
            # 保存 checkpoint 记录
            if self.bg_file_manager:
                try:
                    checkpoint_filename = os.path.basename(checkpoint_path)
                    self.bg_file_manager.save_file_record(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        filename=checkpoint_filename,
                        file_type='checkpoint',
                        file_path=checkpoint_path
                    )
                except Exception as e:
                    print(f"保存checkpoint记录失败: {str(e)}")
            
            self.send_message('info', f'检查点已保存: {checkpoint_path}')
            
            # ✅ 返回字典格式，字段名必须匹配前端期望
            return {
                'plot_paths': comparison_paths,      # ✅ 前端期望的字段名
                'animation_paths': animation_paths,
                'training_curve': self.training_status[self.session_id].get('plot_url', ''),
                'checkpoint_path': checkpoint_path
            }
            
        except Exception as e:
            self.send_message('info', f'❌ 可视化失败: {str(e)}')
            import traceback
            traceback.print_exc()
            
            # ✅ 失败时也要返回字典
            return {
                'plot_paths': [],
                'animation_paths': [],
                'training_curve': '',
                'checkpoint_path': checkpoint_path
            }
    
    def get_training_summary(self):
        """获取训练总结（可选）"""
        summary = super().get_training_summary()
        summary.update({
            'param1': self.param1,
            'param2': self.param2,
        })
        return summary


# ✅ 定义训练入口函数
def train_your_problem(config, session_id, user_id, queue, training_status, get_background_db_func):
    """YOUR_PROBLEM 训练入口函数"""
    trainer = YourProblemTrainer(config, session_id, user_id, queue, training_status, get_background_db_func)
    trainer.train()


__all__ = ['YourProblemTrainer', 'train_your_problem']
```

### 2.2 关键点总结

✅ **必须做**:
1. 方法名 `initialize_environment`（不是 `create_environment`）
2. 返回字典，字段名: `plot_paths`, `animation_paths`, `training_curve`, `checkpoint_path`
3. URL 路径 `/static/...`，不是本地路径
4. 数据维度处理：`locs[i]` 而不是 `locs[i:i+1]`

❌ **不要做**:
1. 返回元组
2. 使用本地文件路径
3. 使用错误的字段名（如 `comparison_paths`）
4. 保留 batch 维度

---

## 步骤3: 创建可视化函数

**路径**: `modules/rl_training/visualizations/your_problem_viz.py`

### 3.1 导入和配置

```python
"""
YOUR_PROBLEM 问题专用可视化函数
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
```

### 3.2 创建动画函数

```python
def create_your_problem_animation(td, actions, save_path, title="路线生成过程", fps=2):
    """
    创建动态 GIF 动画
    
    ⚠️ 关键点：
    1. 检查并处理数据维度
    2. 使用 PIL.Image 生成 GIF
    3. 正确处理 matplotlib buffer
    
    参数:
        td: dict，包含 'locs' 等信息
        actions: numpy 数组，动作序列
        save_path: 保存路径（本地文件路径）
        title: 图表标题
        fps: 帧率
    """
    # ✅ 提取坐标，处理维度
    if hasattr(td, 'get'):
        locs = td.get('locs', td['locs'])
    else:
        locs = td['locs']
    
    # 转换为 numpy
    if hasattr(locs, 'cpu'):
        locs = locs.cpu().numpy()
    
    # ✅ 确保是 2D 数组 (num_cities, 2)
    if locs.ndim == 3:
        locs = locs[0]  # 去掉 batch 维度
    
    frames = []
    
    # 生成每一帧
    for step in range(len(actions)):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制当前状态
        # ... 你的绘图逻辑 ...
        
        ax.set_title(f"{title} - Step {step+1}/{len(actions)}")
        
        # 保存当前帧
        fig.tight_layout()
        fig.canvas.draw()
        
        # ✅ 正确获取图像数据
        try:
            buf = fig.canvas.buffer_rgba()
            image = np.frombuffer(buf, dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:, :, :3]  # 去掉 alpha 通道
        except AttributeError:
            # 旧版本 matplotlib
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        frames.append(Image.fromarray(image))
        plt.close(fig)
    
    # ✅ 保存为 GIF
    if frames:
        duration = int(1000 / fps)  # 毫秒
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                      duration=duration, loop=0, optimize=False)
        print(f"✓ 动画已保存: {save_path}")
    else:
        print("✗ 没有生成任何帧")
```

### 3.3 创建对比图函数

```python
def create_your_problem_comparison_plot(td, actions, save_path, cost=None, title="路线对比图"):
    """
    创建静态对比图
    
    参数:
        td: dict，包含 'locs' 等信息
        actions: numpy 数组，动作序列
        save_path: 保存路径（本地文件路径）
        cost: 可选，成本值
        title: 图表标题
    """
    # ✅ 提取坐标，处理维度
    if hasattr(td, 'get'):
        locs = td.get('locs', td['locs'])
    else:
        locs = td['locs']
    
    if hasattr(locs, 'cpu'):
        locs = locs.cpu().numpy()
    
    if locs.ndim == 3:
        locs = locs[0]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制路径
    # ... 你的绘图逻辑 ...
    
    ax.set_title(title)
    
    # 保存图片
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 对比图已保存: {save_path}")
```

---

## 步骤4: 更新兼容性配置

**文件**: `modules/compatibility.py`

### 4.1 添加策略兼容性

```python
# 策略-问题兼容性矩阵
POLICY_PROBLEM_COMPATIBILITY = {
    'attention': ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'your_problem'],  # ✅ 添加
    'pomo': ['tsp', 'mtsp', 'cvrp', 'your_problem'],  # ✅ 添加（如果支持）
}
```

### 4.2 添加算法兼容性

```python
# 算法-问题兼容性矩阵
ALGORITHM_PROBLEM_COMPATIBILITY = {
    'reinforce': ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'your_problem'],  # ✅ 添加
    'ppo': ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'your_problem'],       # ✅ 添加
    'a2c': ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'your_problem'],       # ✅ 添加
}
```

### 4.3 添加推荐配置

```python
# 推荐的策略-算法组合
RECOMMENDED_COMBINATIONS = {
    'tsp': {...},
    'mtsp': {...},
    'your_problem': {  # ✅ 添加推荐配置
        'best': {
            'policy': 'pomo',
            'algorithm': 'ppo',
            'description': '最佳质量配置'
        },
        'fast': {
            'policy': 'attention',
            'algorithm': 'reinforce',
            'description': '快速训练配置'
        },
        'simple': {
            'policy': 'attention',
            'algorithm': 'reinforce',
            'description': '简单易用配置'
        }
    }
}
```

---

## 步骤5: 注册到系统

### 5.1 更新训练路由

**文件**: `app_training.py` 或 `app.py`

```python
# 导入新的训练函数
from modules.rl_training.your_problem_trainer import train_your_problem

# 在 real_rl4co_training 函数中添加分支
def real_rl4co_training(config, session_id, user_id, queue, training_status, get_background_db_func):
    """真实的 RL4CO 训练逻辑"""
    problem_type = config.get('problem', 'tsp').lower()
    
    # 根据问题类型选择训练函数
    if problem_type == 'tsp':
        from modules.rl_training.tsp_trainer import train_tsp
        train_tsp(config, session_id, user_id, queue, training_status, get_background_db_func)
    elif problem_type == 'mtsp':
        from modules.rl_training.mtsp_trainer import train_mtsp
        train_mtsp(config, session_id, user_id, queue, training_status, get_background_db_func)
    elif problem_type == 'your_problem':  # ✅ 添加这里
        train_your_problem(config, session_id, user_id, queue, training_status, get_background_db_func)
    else:
        raise ValueError(f"不支持的问题类型: {problem_type}")
```

---

## 步骤6: 前端集成

### 6.1 更新问题选择下拉框

**文件**: `templates/index.html`

找到问题类型选择部分：

```html
<select id="problem-select" onchange="problemChanged()">
    <option value="tsp">TSP - 旅行商问题</option>
    <option value="atsp">ATSP - 非对称旅行商</option>
    <option value="mtsp">mTSP - 多旅行商问题</option>
    <option value="cvrp">CVRP - 带容量约束的车辆路径问题</option>
    <option value="sdvrp">SDVRP - 分割配送车辆路径问题</option>
    <option value="vrptw">VRPTW - 带时间窗的车辆路径问题</option>
    <option value="your_problem">YOUR_PROBLEM - 你的问题描述</option> <!-- ✅ 添加这里 -->
</select>
```

### 6.2 添加问题特有参数输入

```html
<div id="your-problem-params" class="problem-specific-params" style="display: none;">
    <div class="form-group">
        <label for="param1-input">参数1:</label>
        <input type="number" id="param1-input" value="默认值" min="最小值" max="最大值">
    </div>
    <div class="form-group">
        <label for="param2-input">参数2:</label>
        <select id="param2-input">
            <option value="option1">选项1</option>
            <option value="option2">选项2</option>
        </select>
    </div>
</div>
```

### 6.3 更新 JavaScript 逻辑

```javascript
// 问题类型切换
function problemChanged() {
    const problem = document.getElementById('problem-select').value;
    
    // 隐藏所有特定参数
    document.querySelectorAll('.problem-specific-params').forEach(el => {
        el.style.display = 'none';
    });
    
    // 显示当前问题的参数
    if (problem === 'your_problem') {
        document.getElementById('your-problem-params').style.display = 'block';
    }
    
    // 更新兼容性检查...
}

// 启动训练时收集参数
async function startTraining() {
    const problem = document.getElementById('problem-select').value;
    const config = {
        problem: problem,
        model: document.getElementById('model-select').value,
        algorithm: document.getElementById('algorithm-select').value,
        num_loc: parseInt(document.getElementById('num-loc-input').value),
        epochs: parseInt(document.getElementById('epochs-input').value),
        batch_size: parseInt(document.getElementById('batch-size-input').value),
        learning_rate: parseFloat(document.getElementById('lr-input').value)
    };
    
    // ✅ 添加问题特有参数
    if (problem === 'your_problem') {
        const param1 = document.getElementById('param1-input');
        const param2 = document.getElementById('param2-input');
        
        if (param1) {
            config.param1 = parseInt(param1.value);
        }
        if (param2) {
            config.param2 = param2.value;
        }
    }
    
    // 发送训练请求...
}
```

---

## 步骤7: 测试验证

### 7.1 单元测试

创建 `modules/rl_training/test_your_problem.py`:

```python
"""测试 YOUR_PROBLEM 训练器"""
import sys
sys.path.append('/path/to/project')

from modules.rl_training.your_problem_trainer import YourProblemTrainer
from queue import Queue

def test_trainer():
    config = {
        'problem': 'your_problem',
        'model': 'attention',
        'algorithm': 'reinforce',
        'num_loc': 20,
        'epochs': 2,
        'batch_size': 64,
        'learning_rate': 0.0001,
        'param1': value1,
        'param2': value2,
    }
    
    queue = Queue()
    training_status = {}
    session_id = 'test_session'
    user_id = 1
    
    trainer = YourProblemTrainer(
        config, session_id, user_id, queue, 
        training_status, lambda: None
    )
    
    # 测试环境初始化
    env = trainer.initialize_environment()
    print(f"✅ 环境创建成功: {env.name}")
    
    # 测试完整训练
    trainer.train()
    print("✅ 训练完成")

if __name__ == '__main__':
    test_trainer()
```

### 7.2 集成测试

1. **启动应用**:
   ```bash
   python app.py
   ```

2. **访问前端**: http://localhost:5000

3. **配置训练**（小规模快速测试）:
   - 问题类型: YOUR_PROBLEM
   - 节点数量: 20
   - 训练轮数: 3
   - 其他参数: 使用默认值

4. **检查点**:
   - ✅ 训练能否启动
   - ✅ SSE 连接正常
   - ✅ 进度更新正常
   - ✅ 可视化生成成功
   - ✅ 前端显示正确

### 7.3 验证清单

- [ ] 环境初始化成功
- [ ] 训练进度正常更新
- [ ] 生成 3 个动画文件
- [ ] 生成 3 个对比图文件
- [ ] 前端显示所有可视化
- [ ] 检查点文件保存成功
- [ ] 数据库记录正确
- [ ] 无 JavaScript 错误
- [ ] 无 Python 异常

---

## 常见错误及解决方案

基于 mTSP 实际实现经验总结。

### 错误 1: NotImplementedError - 方法名错误

**错误信息**:
```
NotImplementedError: 子类必须实现 initialize_environment 方法
```

**原因**: 方法名拼写错误

**错误代码**:
```python
def create_environment(self):  # ❌ 错误
    ...
```

**正确代码**:
```python
def initialize_environment(self):  # ✅ 正确
    ...
```

---

### 错误 2: 前端无法选择策略/算法

**现象**: 下拉框为空或禁用

**原因**: 未在 `modules/compatibility.py` 中添加兼容性配置

**解决方法**:
```python
# modules/compatibility.py
POLICY_PROBLEM_COMPATIBILITY = {
    'attention': [..., 'your_problem'],  # ✅ 添加
}

ALGORITHM_PROBLEM_COMPATIBILITY = {
    'reinforce': [..., 'your_problem'],  # ✅ 添加
}
```

---

### 错误 3: AttributeError - 'plots_dir' 不存在

**错误信息**:
```
AttributeError: 'YourProblemTrainer' object has no attribute 'plots_dir'
```

**原因**: 使用了错误的属性名

**错误代码**:
```python
save_path = os.path.join(self.plots_dir, filename)  # ❌
```

**正确代码**:
```python
save_path = os.path.join(self.user_plots_dir, filename)  # ✅
```

**BaseTrainer 提供的正确属性**:
- `self.user_plots_dir` - 用户图表目录
- `self.user_checkpoints_dir` - 用户检查点目录
- `self.bg_file_manager` - 文件管理器
- `self.bg_session_manager` - 会话管理器

---

### 错误 4: TypeError - 'tuple' object is not a mapping

**错误信息**:
```
TypeError: 'tuple' object is not a mapping
```

**原因**: `generate_visualizations` 返回了元组而不是字典

**错误代码**:
```python
def generate_visualizations(self, env, model, trainer, checkpoint_path):
    ...
    return (comparison_paths, animation_paths, training_curve)  # ❌ 元组
```

**正确代码**:
```python
def generate_visualizations(self, env, model, trainer, checkpoint_path):
    ...
    return {  # ✅ 字典
        'plot_paths': comparison_paths,
        'animation_paths': animation_paths,
        'training_curve': training_curve,
        'checkpoint_path': checkpoint_path
    }
```

---

### 错误 5: 前端不显示可视化图片

**现象**: 训练完成但前端空白

**可能原因 1**: 字段名不匹配

**错误代码**:
```python
return {
    'comparison_paths': [...],  # ❌ 错误的字段名
    'animations': [...],        # ❌ 错误的字段名
}
```

**正确代码**:
```python
return {
    'plot_paths': [...],        # ✅ 前端期望的字段名
    'animation_paths': [...],   # ✅ 前端期望的字段名
}
```

**可能原因 2**: 使用了本地路径而不是 URL 路径

**错误代码**:
```python
comparison_paths.append('/full/local/path/to/file.png')  # ❌
```

**正确代码**:
```python
comparison_paths.append(f'/static/model_plots/user_{self.user_id}/file.png')  # ✅
```

---

### 错误 6: 可视化生成失败 - 维度错误

**错误信息**:
```
index 33 is out of bounds for axis 0 with size 1
only length-1 arrays can be converted to Python scalars
```

**原因**: 数据维度不匹配

**错误代码**:
```python
for i in range(num_test_instances):
    create_animation(
        td={'locs': torch.from_numpy(locs[i:i+1])},  # ❌ 3D数组 (1, 50, 2)
        ...
    )
```

**正确代码**:
```python
for i in range(num_test_instances):
    instance_locs = locs[i]  # ✅ 2D数组 (50, 2)
    create_animation(
        td={'locs': torch.from_numpy(instance_locs)},
        ...
    )
```

**可视化函数中添加防御性检查**:
```python
def create_animation(td, actions, save_path, ...):
    locs = td['locs']
    if hasattr(locs, 'cpu'):
        locs = locs.cpu().numpy()
    
    # ✅ 确保是 2D 数组
    if locs.ndim == 3:
        locs = locs[0]  # 去掉 batch 维度
```

---

### 错误 7: 文件未保存到数据库

**现象**: 文件生成了但在文件管理器中看不到

**原因**: 忘记调用 `bg_file_manager.save_file_record`

**解决方法**:
```python
# 保存文件记录到数据库
if self.bg_file_manager:
    try:
        self.bg_file_manager.save_file_record(
            user_id=self.user_id,
            session_id=self.session_id,
            filename=filename,
            file_type='plot',  # 或 'animation', 'checkpoint'
            file_path=local_path  # 本地完整路径
        )
    except Exception as e:
        print(f"保存文件记录失败: {str(e)}")
```

---

### 错误 8: ImportError - 模块未找到

**错误信息**:
```
ModuleNotFoundError: No module named 'rl4co.envs.YourProblemEnv'
```

**解决方法**:
1. 检查 RL4CO 版本是否支持该环境
2. 使用 try-except 优雅处理
3. 设置 `RL4CO_AVAILABLE` 标志

```python
try:
    from rl4co.envs import YourProblemEnv
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    print("警告: RL4CO 不支持 YourProblemEnv")
```

---

### 错误 9: strategy 字段始终显示 REINFORCE

**原因**: `BaseTrainer` 中硬编码了算法名称

**已修复**: 现在使用 `self.algorithm_name.upper()`

如果你的版本仍有问题，检查 `modules/rl_training/base_trainer.py`:
```python
final_results = {
    ...
    'strategy': self.algorithm_name.upper(),  # ✅ 使用实际算法
    ...
}
```

---

## 完整示例: mTSP

这是一个完整的实际案例，展示了所有步骤。

### 文件结构

```
modules/
├── problems/
│   ├── mtsp.py                    # ✅ 问题定义（使用 RL4CO 内置，可选）
│   └── MTSP_GUIDE.md             # 文档
├── rl_training/
│   ├── mtsp_trainer.py           # ✅ 训练器实现
│   └── visualizations/
│       └── mtsp_viz.py           # ✅ 可视化函数
└── compatibility.py              # ✅ 兼容性配置
```

### 关键代码片段

**1. 训练器 - initialize_environment**:
```python
def initialize_environment(self):
    from rl4co.envs import MTSPEnv
    
    env = MTSPEnv(
        num_loc=self.num_loc,
        min_loc=0,
        max_loc=1,
        num_agents=self.num_agents,
        cost_type=self.cost_type
    )
    
    self.send_message('info', f'✅ mTSP环境创建成功')
    return env
```

**2. 训练器 - generate_visualizations**:
```python
def generate_visualizations(self, env, model, trainer, checkpoint_path):
    # 生成测试数据
    td = env.reset(batch_size=[3])
    out = model(td.clone(), phase='test', decode_type='greedy')
    
    # 提取结果
    actions = out['actions'].cpu().numpy()
    locs = td['locs'].cpu().numpy()
    
    # 创建可视化
    for i in range(3):
        instance_locs = locs[i]  # ✅ 去掉 batch 维度
        instance_actions = actions[i]
        
        create_mtsp_route_animation(
            td={'locs': torch.from_numpy(instance_locs)},
            actions=instance_actions,
            save_path=anim_path
        )
        
        animation_paths.append(f'/static/model_plots/user_{self.user_id}/{filename}')
    
    return {
        'plot_paths': comparison_paths,
        'animation_paths': animation_paths,
        'training_curve': self.training_status[self.session_id].get('plot_url', ''),
        'checkpoint_path': checkpoint_path
    }
```

**3. 兼容性配置**:
```python
POLICY_PROBLEM_COMPATIBILITY = {
    'attention': ['tsp', 'atsp', 'mtsp', ...],
    'pomo': ['tsp', 'mtsp', 'cvrp'],
}

ALGORITHM_PROBLEM_COMPATIBILITY = {
    'reinforce': ['tsp', 'atsp', 'mtsp', ...],
    'ppo': ['tsp', 'atsp', 'mtsp', ...],
    'a2c': ['tsp', 'atsp', 'mtsp', ...],
}

RECOMMENDED_COMBINATIONS = {
    'mtsp': {
        'best': {'policy': 'pomo', 'algorithm': 'ppo'},
        'fast': {'policy': 'attention', 'algorithm': 'reinforce'},
        'simple': {'policy': 'attention', 'algorithm': 'reinforce'},
    }
}
```

**4. 前端集成**:
```html
<option value="mtsp">mTSP - 多旅行商问题</option>

<div id="mtsp-params" class="problem-specific-params" style="display: none;">
    <div class="form-group">
        <label for="num-agents-input">代理数量:</label>
        <input type="number" id="num-agents-input" value="5" min="2" max="10">
    </div>
    <div class="form-group">
        <label for="cost-type-input">优化目标:</label>
        <select id="cost-type-input">
            <option value="minmax">MinMax - 最小化最长路径</option>
            <option value="sum">Sum - 最小化总距离</option>
        </select>
    </div>
</div>
```

```javascript
if (problem === 'mtsp') {
    const numAgents = document.getElementById('num-agents-input');
    const costType = document.getElementById('cost-type-input');
    
    if (numAgents) {
        config.num_agents = parseInt(numAgents.value);
    }
    if (costType) {
        config.cost_type = costType.value;
    }
}
```

---

## 检查清单

在提交代码前，使用此清单确保所有步骤完成：

### 后端实现

- [ ] **问题定义** (如需要)
  - [ ] 创建 `modules/problems/your_problem.py`
  - [ ] 实现 `YourProblem` 类
  - [ ] 注册到 `problems/__init__.py`

- [ ] **训练器**
  - [ ] 创建 `modules/rl_training/your_problem_trainer.py`
  - [ ] 继承 `BaseTrainer`
  - [ ] 实现 `initialize_environment()` 方法（名称正确）
  - [ ] 实现 `generate_visualizations()` 方法
  - [ ] 返回正确的字典格式
  - [ ] 使用正确的字段名 (`plot_paths`, `animation_paths`)
  - [ ] 使用 URL 路径而非本地路径
  - [ ] 正确处理数据维度
  - [ ] 保存文件记录到数据库
  - [ ] 定义 `train_your_problem()` 入口函数

- [ ] **可视化**
  - [ ] 创建 `modules/rl_training/visualizations/your_problem_viz.py`
  - [ ] 实现 `create_your_problem_animation()`
  - [ ] 实现 `create_your_problem_comparison_plot()`
  - [ ] 处理数据维度（去掉 batch 维度）
  - [ ] 正确生成 GIF 和 PNG

- [ ] **兼容性配置**
  - [ ] 更新 `POLICY_PROBLEM_COMPATIBILITY`
  - [ ] 更新 `ALGORITHM_PROBLEM_COMPATIBILITY`
  - [ ] 添加 `RECOMMENDED_COMBINATIONS`

- [ ] **路由注册**
  - [ ] 在 `app_training.py` 中添加分支
  - [ ] 导入训练函数

### 前端集成

- [ ] **界面更新**
  - [ ] 添加问题选项到下拉框
  - [ ] 创建问题特有参数输入区域
  - [ ] 实现参数显示/隐藏逻辑

- [ ] **JavaScript**
  - [ ] 更新 `problemChanged()` 函数
  - [ ] 更新 `startTraining()` 函数
  - [ ] 收集问题特有参数
  - [ ] 测试兼容性检查逻辑

### 文档

- [ ] **创建问题指南**
  - [ ] 创建 `modules/problems/YOUR_PROBLEM_GUIDE.md`
  - [ ] 包含问题描述
  - [ ] 包含参数说明
  - [ ] 包含使用示例

- [ ] **更新主文档**
  - [ ] 更新 `modules/README.md`
  - [ ] 更新 `modules/PROBLEM_COMPATIBILITY.md`

### 测试

- [ ] **单元测试**
  - [ ] 测试环境初始化
  - [ ] 测试训练流程
  - [ ] 测试可视化生成

- [ ] **集成测试**
  - [ ] 小规模快速训练（3 轮）
  - [ ] 检查前端显示
  - [ ] 检查文件生成
  - [ ] 检查数据库记录

- [ ] **错误处理**
  - [ ] 测试参数验证
  - [ ] 测试异常处理
  - [ ] 测试边界情况

---

## 附录

### A. BaseTrainer 可用属性

```python
# 路径
self.user_plots_dir          # 用户图表目录
self.user_checkpoints_dir    # 用户检查点目录

# 管理器
self.bg_file_manager         # 文件管理器
self.bg_session_manager      # 会话管理器
self.bg_db                   # 数据库连接

# 配置
self.problem_type            # 问题类型
self.model_type             # 策略模型
self.algorithm_name         # 算法名称
self.num_loc                # 节点数量
self.epochs                 # 训练轮数
self.batch_size             # 批次大小
self.learning_rate          # 学习率
self.device                 # 设备 (cpu/cuda)

# 训练状态
self.training_status        # 全局训练状态字典
self.session_id             # 当前会话ID
self.user_id                # 当前用户ID
self.queue                  # 消息队列
```

### B. 前端期望的字段名

```javascript
// 训练结果对象
results = {
  model: 'pomo',
  problem: 'mtsp',
  strategy: 'PPO',
  total_epochs: 10,
  final_loss: 1.2345,
  final_reward: -5.6789,
  best_reward: -5.4321,
  
  // ⚠️ 关键字段，必须精确匹配
  plot_paths: [...],        // 不是 comparison_paths
  animation_paths: [...],   // 不是 animations
  training_curve: '...',    // 不是 training_curves
  checkpoint_path: '...'    // 不是 checkpoint
}
```

### C. 文件路径规范

```python
# 本地文件路径（用于文件操作和数据库记录）
local_path = '/full/path/to/static/model_plots/user_1/file.png'
local_path = os.path.join(self.user_plots_dir, filename)

# URL 路径（返回给前端显示）
url_path = '/static/model_plots/user_1/file.png'
url_path = f'/static/model_plots/user_{self.user_id}/{filename}'

# 使用场景
# 1. 生成文件时 → 使用本地路径
plt.savefig(local_path)

# 2. 保存数据库记录时 → 使用本地路径
self.bg_file_manager.save_file_record(..., file_path=local_path)

# 3. 返回给前端时 → 使用 URL 路径
plot_paths.append(url_path)
```

### D. 数据维度处理

```python
# 模型输出
actions.shape = (batch_size, action_length)    # 例如: (3, 100)
locs.shape = (batch_size, num_cities, 2)       # 例如: (3, 50, 2)

# 提取单个实例
instance_actions = actions[i]    # shape: (action_length,) = (100,)
instance_locs = locs[i]          # shape: (num_cities, 2) = (50, 2)

# ❌ 错误：保留 batch 维度
locs[i:i+1]                      # shape: (1, 50, 2)

# ✅ 正确：去掉 batch 维度
locs[i]                          # shape: (50, 2)
```

---

## 总结

新增问题类型的核心步骤：

1. **创建训练器** - 实现 `initialize_environment` 和 `generate_visualizations`
2. **创建可视化** - 实现动画和对比图生成函数
3. **更新兼容性** - 配置策略和算法支持
4. **注册到系统** - 添加路由分支
5. **前端集成** - 添加界面和参数收集
6. **测试验证** - 确保所有功能正常

**关键注意事项**:
- ✅ 方法名: `initialize_environment`
- ✅ 返回格式: 字典，不是元组
- ✅ 字段名: `plot_paths`, `animation_paths`
- ✅ 路径类型: URL 路径 `/static/...`
- ✅ 数据维度: 使用 `locs[i]` 而不是 `locs[i:i+1]`

遵循本指南，你可以顺利地向系统添加新的问题类型，避免常见的陷阱！

---

**文档维护者**: RL4CO Display Team  
**最后更新**: 2026-02-05  
**基于实际案例**: mTSP 完整实现
