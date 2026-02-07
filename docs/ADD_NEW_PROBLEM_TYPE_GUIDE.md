# 添加新问题类型完整指南

## 📋 目录

1. [概述](#概述)
2. [前置要求](#前置要求)
3. [添加步骤](#添加步骤)
4. [文件清单](#文件清单)
5. [测试验证](#测试验证)
6. [常见错误](#常见错误)
7. [示例：添加 mTSP](#示例添加-mtsp)

---

## 概述

本指南详细说明如何向 RL4CO Display 平台添加新的组合优化问题类型。

**时间估算**：完整添加一个新问题类型需要 **2-4 小时**

**难度等级**：⭐⭐⭐ 中等

---

## 前置要求

### 1. 技术知识
- ✅ 熟悉 Python 面向对象编程
- ✅ 了解 RL4CO 库的基本使用
- ✅ 熟悉 Flask 框架
- ✅ 了解 HTML/JavaScript（前端集成）

### 2. 环境准备
- ✅ 确保 RL4CO 支持该问题类型
- ✅ 了解问题的数学定义和约束
- ✅ 准备测试数据集

### 3. 文档准备
- ✅ 阅读 `modules/README.md` 了解架构
- ✅ 阅读 `modules/PROBLEM_COMPATIBILITY.md` 了解兼容性
- ✅ 查看现有问题类型的实现（如 TSP、CVRP）

---

## 添加步骤

### 📝 第一步：问题定义层（Problem Definition）

#### 1.1 创建问题类文件

**位置**：`modules/problems/your_problem.py`

**内容**：
```python
"""
YOUR_PROBLEM 问题定义
描述问题的特点、约束等
"""

from .base_problem import BaseProblem

class YourProblem(BaseProblem):
    """YOUR_PROBLEM 问题类"""
    
    def __init__(self):
        super().__init__()
        self.problem_name = "your_problem"
        self.display_name = "您的问题显示名称"
        self.description = "问题的详细描述"
    
    def get_default_params(self):
        """返回问题的默认参数"""
        return {
            'num_loc': 50,
            'your_param_1': default_value_1,
            'your_param_2': default_value_2,
        }
    
    def validate_params(self, params):
        """
        验证参数有效性
        
        返回:
            tuple: (is_valid, error_message)
        """
        # 添加参数验证逻辑
        if params.get('num_loc', 0) < 2:
            return False, "节点数量必须 >= 2"
        
        # 更多验证...
        
        return True, None
    
    def get_constraints(self):
        """返回问题的约束描述"""
        return [
            "约束1描述",
            "约束2描述",
        ]
    
    def get_features(self):
        """返回问题的特征"""
        return {
            'is_symmetric': True,  # 是否对称
            'has_capacity': False,  # 是否有容量约束
            'has_time_windows': False,  # 是否有时间窗
            'multi_agent': False,  # 是否多代理
        }
```

#### 1.2 注册问题类

**文件**：`modules/problems/__init__.py`

**修改**：
```python
# 1. 添加导入
from .your_problem import YourProblem

# 2. 注册到字典
PROBLEM_REGISTRY = {
    'tsp': TSProblem,
    'cvrp': CVRProblem,
    'your_problem': YourProblem,  # ← 添加这行
}

# 3. 添加到 __all__
__all__ = [
    'BaseProblem',
    'TSProblem',
    'CVRProblem',
    'YourProblem',  # ← 添加这行
    'get_problem_class',
    'list_available_problems',
]
```

---

### 🔧 第二步：训练器层（Trainer）

#### 2.1 创建训练器文件

**位置**：`modules/rl_training/your_problem_trainer.py`

**内容**：
```python
"""
YOUR_PROBLEM 问题专用训练器
包含训练逻辑和可视化生成
"""

import os
import torch
from datetime import datetime

try:
    from rl4co.envs import YourProblemEnv
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    print("警告: RL4CO 库未安装或不支持 YourProblemEnv")

from .base_trainer import BaseTrainer
from .visualizations.your_problem_viz import (
    create_your_problem_route_animation,
    create_your_problem_comparison_plot
)


class YourProblemTrainer(BaseTrainer):
    """YOUR_PROBLEM 问题训练器"""
    
    def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func):
        super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func)
        
        # 获取问题特有参数
        self.your_param_1 = config.get('your_param_1', default_value)
        self.your_param_2 = config.get('your_param_2', default_value)
        
        self.send_message('info', f'📋 YOUR_PROBLEM配置: param1={self.your_param_1}, param2={self.your_param_2}')
    
    def get_problem_type(self):
        """返回问题类型标识"""
        return 'your_problem'
    
    def initialize_environment(self):
        """
        初始化环境（必须实现！）
        
        ⚠️ 重要：方法名必须是 initialize_environment
        BaseTrainer.train() 会调用此方法
        """
        if not RL4CO_AVAILABLE:
            raise ImportError("RL4CO库未安装或不支持YourProblemEnv")
        
        # 创建环境
        env = YourProblemEnv(
            generator_params={
                'num_loc': self.num_loc,
                'your_param_1': self.your_param_1,
                'your_param_2': self.your_param_2,
            }
        )
        
        self.send_message('info', f'✅ YOUR_PROBLEM环境创建成功')
        return env
    
    def generate_visualizations(self, env, model, trainer, checkpoint_path):
        """
        生成可视化（可选）
        
        参数:
            env: 环境对象
            model: 训练好的模型
            trainer: PyTorch Lightning训练器
            checkpoint_path: 检查点路径
        
        返回:
            tuple: (animation_paths, comparison_paths)
        """
        try:
            self.send_message('info', '🎨 开始生成可视化...')
            
            # 生成测试数据
            device = next(model.parameters()).device
            model.eval()
            
            num_test_instances = min(3, self.batch_size)
            td = env.reset(batch_size=[num_test_instances])
            td = td.to(device)
            
            # 使用模型生成解
            with torch.no_grad():
                out = model(td.clone(), phase='test', decode_type='greedy')
            
            # 提取结果
            actions = out['actions'].cpu().numpy()
            rewards = out['reward'].cpu().numpy()
            locs = td['locs'].cpu().numpy()
            
            # 创建可视化
            animation_paths = []
            comparison_paths = []
            
            for i in range(num_test_instances):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                anim_path = os.path.join(self.plots_dir, f'your_problem_anim_{i+1}_{timestamp}.gif')
                comp_path = os.path.join(self.plots_dir, f'your_problem_comp_{i+1}_{timestamp}.png')
                
                # 调用可视化函数
                try:
                    create_your_problem_route_animation(
                        td={'locs': torch.from_numpy(locs[i:i+1])},
                        actions=actions[i],
                        save_path=anim_path,
                        title=f'问题 {i+1}',
                        fps=2
                    )
                    animation_paths.append(anim_path)
                except Exception as e:
                    self.send_message('info', f'⚠️ 动画 {i+1} 生成失败: {str(e)}')
                
                try:
                    create_your_problem_comparison_plot(
                        td={'locs': torch.from_numpy(locs[i:i+1])},
                        actions=actions[i],
                        save_path=comp_path,
                        cost=-rewards[i],
                        title=f'问题 {i+1}'
                    )
                    comparison_paths.append(comp_path)
                except Exception as e:
                    self.send_message('info', f'⚠️ 对比图 {i+1} 生成失败: {str(e)}')
            
            self.send_message('info', f'🎉 可视化完成')
            return animation_paths, comparison_paths
            
        except Exception as e:
            self.send_message('info', f'❌ 可视化失败: {str(e)}')
            import traceback
            traceback.print_exc()
            return [], []
    
    def get_training_summary(self):
        """获取训练总结（可选）"""
        summary = super().get_training_summary()
        summary.update({
            'your_param_1': self.your_param_1,
            'your_param_2': self.your_param_2,
        })
        return summary


def train_your_problem(config, session_id, user_id, queue, training_status, get_background_db_func):
    """
    训练入口函数
    """
    trainer = YourProblemTrainer(config, session_id, user_id, queue, training_status, get_background_db_func)
    trainer.train()


__all__ = ['YourProblemTrainer', 'train_your_problem']
```

#### 2.2 注册训练器

**文件**：`modules/rl_training/__init__.py`

**修改位置 1 - 导入**（约第 15-25 行）：
```python
# 导入所有训练器
from .tsp_trainer import TSPTrainer, train_tsp
from .cvrp_trainer import CVRPTrainer, train_cvrp
from .your_problem_trainer import YourProblemTrainer, train_your_problem  # ← 添加
```

**修改位置 2 - 路由**（约第 40-60 行）：
```python
def real_rl4co_training(config, session_id, user_id, queue, training_status, get_background_db_func):
    problem_type = config.get('problem', 'tsp').lower()

    if problem_type == 'tsp':
        train_tsp(...)
    elif problem_type == 'cvrp':
        train_cvrp(...)
    elif problem_type == 'your_problem':  # ← 添加这个分支
        train_your_problem(config, session_id, user_id, queue, training_status, get_background_db_func)
    else:
        # 更新错误信息
        raise ValueError(f"不支持的问题类型: {problem_type}，支持的类型: tsp, cvrp, your_problem")
```

**修改位置 3 - 导出**（约第 75-90 行）：
```python
__all__ = [
    'TSPTrainer',
    'CVRPTrainer',
    'YourProblemTrainer',  # ← 添加
    'train_tsp',
    'train_cvrp',
    'train_your_problem',  # ← 添加
    'real_rl4co_training',
]
```

---

### 🎨 第三步：可视化层（Visualization）

#### 3.1 创建可视化文件

**位置**：`modules/rl_training/visualizations/your_problem_viz.py`

**内容**：
```python
"""
YOUR_PROBLEM 问题可视化函数
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def create_your_problem_route_animation(td, actions, save_path, title='', fps=2):
    """
    创建路线生成动画
    
    参数:
        td: 包含位置信息的TensorDict
        actions: 动作序列
        save_path: 保存路径
        title: 图表标题
        fps: 帧率
    """
    locs = td['locs'][0].cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        # 绘制节点
        ax.scatter(locs[:, 0], locs[:, 1], c='blue', s=100, zorder=3)
        
        # 绘制已访问路径
        if frame > 0:
            for i in range(frame):
                start_idx = actions[i]
                end_idx = actions[i+1] if i+1 < len(actions) else actions[0]
                ax.plot([locs[start_idx, 0], locs[end_idx, 0]],
                       [locs[start_idx, 1], locs[end_idx, 1]],
                       'r-', linewidth=2)
        
        ax.set_title(f'{title} - 步骤 {frame}/{len(actions)}')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
    
    anim = animation.FuncAnimation(fig, update, frames=len(actions)+1, interval=1000//fps)
    anim.save(save_path, writer='pillow')
    plt.close(fig)


def create_your_problem_comparison_plot(td, actions, save_path, cost, title=''):
    """
    创建对比图
    
    参数:
        td: 包含位置信息的TensorDict
        actions: 动作序列
        save_path: 保存路径
        cost: 路径成本
        title: 图表标题
    """
    locs = td['locs'][0].cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制节点
    ax.scatter(locs[:, 0], locs[:, 1], c='blue', s=100, zorder=3, label='节点')
    
    # 绘制完整路径
    for i in range(len(actions)):
        start_idx = actions[i]
        end_idx = actions[i+1] if i+1 < len(actions) else actions[0]
        ax.plot([locs[start_idx, 0], locs[end_idx, 0]],
               [locs[start_idx, 1], locs[end_idx, 1]],
               'r-', linewidth=2, alpha=0.7)
    
    ax.set_title(f'{title}\n成本: {cost:.2f}')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


__all__ = ['create_your_problem_route_animation', 'create_your_problem_comparison_plot']
```

---

### ⚙️ 第四步：兼容性配置

#### 4.1 添加兼容性规则

**文件**：`modules/compatibility.py`

**修改位置 1 - 策略兼容性**（约第 13-17 行）：
```python
POLICY_PROBLEM_COMPATIBILITY = {
    'attention': ['tsp', 'cvrp', 'your_problem'],  # ← 添加 your_problem
    'pomo': ['tsp', 'cvrp', 'your_problem'],       # ← 如果支持POMO
}
```

**说明**：
- ✅ 如果问题是**对称**的，可以添加到 `pomo` 列表
- ❌ 如果问题是**非对称**的，只添加到 `attention` 列表

**修改位置 2 - 算法兼容性**（约第 20-24 行）：
```python
ALGORITHM_PROBLEM_COMPATIBILITY = {
    'reinforce': ['tsp', 'cvrp', 'your_problem'],  # ← 添加
    'ppo': ['tsp', 'cvrp', 'your_problem'],        # ← 添加
    'a2c': ['tsp', 'cvrp', 'your_problem'],        # ← 添加
}
```

**修改位置 3 - 推荐配置**（约第 68-94 行）：
```python
RECOMMENDED_COMBINATIONS = {
    'tsp': {...},
    'cvrp': {...},
    'your_problem': {  # ← 添加推荐配置
        'best': {'policy': 'pomo', 'algorithm': 'ppo'},        # 最佳质量
        'fast': {'policy': 'attention', 'algorithm': 'ppo'},    # 快速原型
        'simple': {'policy': 'attention', 'algorithm': 'reinforce'},  # 学习入门
    },
}
```

**修改位置 4 - 警告信息**（约第 34-65 行，可选）：
```python
WARNING_COMBINATIONS = [
    {
        'problem': 'your_problem',
        'policy': 'pomo',
        'message': '如果POMO效果未验证，添加此警告',
        'severity': 'warning'
    },
    # 更多警告...
]
```

---

### 🌐 第五步：前端集成

#### 5.1 添加问题选项

**文件**：`templates/index.html`

**修改位置**（约第 504-520 行，问题选择下拉菜单）：
```html
<select id="problem-select" name="problem">
    <option value="tsp">TSP - 旅行商问题</option>
    <option value="cvrp">CVRP - 车辆路径问题</option>
    <option value="your_problem">YOUR_PROBLEM - 您的问题名称 ⭐新增</option>  <!-- 添加这行 -->
</select>
```

#### 5.2 添加参数输入区域

**文件**：`templates/index.html`

**位置**：在其他问题参数区域之后添加

**内容**：
```html
<!-- YOUR_PROBLEM特有参数 -->
<div id="your-problem-params" class="form-group" style="display: none;">
    <div style="background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%); 
                padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4>🔧 YOUR_PROBLEM 参数</h4>
        <p>配置问题特有参数</p>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
        <!-- 参数1 -->
        <div>
            <label for="your-param-1">参数1名称</label>
            <input type="number" 
                   id="your-param-1" 
                   name="your_param_1" 
                   value="默认值" 
                   min="最小值" 
                   max="最大值" 
                   step="步长">
            <small style="display: block; color: #666; margin-top: 0.25rem;">
                参数说明文字
            </small>
        </div>
        
        <!-- 参数2 -->
        <div>
            <label for="your-param-2">参数2名称</label>
            <select id="your-param-2" name="your_param_2">
                <option value="option1">选项1</option>
                <option value="option2">选项2</option>
            </select>
            <small style="display: block; color: #666; margin-top: 0.25rem;">
                参数说明文字
            </small>
        </div>
    </div>
    
    <!-- 参数提示 -->
    <div style="margin-top: 1rem; padding: 0.75rem; background: #f0f7ff; 
                border-left: 4px solid #4CAF50; border-radius: 4px;">
        <strong>💡 提示：</strong>
        <ul style="margin: 0.5rem 0 0 1.5rem; padding: 0;">
            <li>提示1</li>
            <li>提示2</li>
        </ul>
    </div>
</div>
```

#### 5.3 更新 JavaScript 显示逻辑

**文件**：`templates/index.html`

**修改位置 1 - showProblemSpecificParams 函数**（约第 1200-1250 行）：
```javascript
showProblemSpecificParams() {
    // 隐藏所有参数区域
    document.getElementById('tsp-params').style.display = 'none';
    document.getElementById('cvrp-params').style.display = 'none';
    document.getElementById('your-problem-params').style.display = 'none';  // ← 添加
    
    // 显示当前问题的参数区域
    if (this.currentProblem === 'tsp') {
        document.getElementById('tsp-params').style.display = 'block';
    } else if (this.currentProblem === 'cvrp') {
        document.getElementById('cvrp-params').style.display = 'block';
    } else if (this.currentProblem === 'your_problem') {  // ← 添加
        document.getElementById('your-problem-params').style.display = 'block';
    }
    // ... 其他问题 ...
}
```

#### 5.4 更新 JavaScript 提交逻辑

**文件**：`templates/index.html`

**修改位置 - startTraining 函数**（约第 1400-1500 行）：
```javascript
startTraining() {
    // ... 前面的代码 ...
    
    // 根据问题类型添加特有参数
    if (problem === 'tsp') {
        // TSP 参数...
    } else if (problem === 'cvrp') {
        // CVRP 参数...
    } else if (problem === 'your_problem') {  // ← 添加
        config.your_param_1 = parseInt(document.getElementById('your-param-1').value);
        config.your_param_2 = document.getElementById('your-param-2').value;
    }
    
    // ... 后面的代码 ...
}
```

---

### 📚 第六步：文档更新

#### 6.1 更新架构文档

**文件**：`modules/README.md`

**修改位置 1 - 文件结构**（约第 7-36 行）：
```markdown
modules/
├── problems/
│   ├── tsp.py
│   ├── cvrp.py
│   ├── your_problem.py          # ← 添加
│
├── rl_training/
│   ├── tsp_trainer.py
│   ├── cvrp_trainer.py
│   ├── your_problem_trainer.py  # ← 添加
│   └── visualizations/
│       ├── tsp_viz.py
│       ├── cvrp_viz.py
│       └── your_problem_viz.py  # ← 添加
```

**修改位置 2 - 使用示例**（添加新章节）：
```markdown
### 示例X: YOUR_PROBLEM + POMO + PPO

\```python
config = {
    'problem': 'your_problem',
    'model': 'pomo',
    'algorithm': 'ppo',
    'num_loc': 50,
    'your_param_1': value1,
    'your_param_2': value2,
    'epochs': 10,
}
\```
```

**修改位置 3 - 问题列表**（约第 380-385 行）：
```markdown
### 问题层
- [x] TSP ✅
- [x] CVRP ✅
- [x] YOUR_PROBLEM ✅  # ← 添加
- [ ] 其他问题 - 计划中
```

#### 6.2 更新兼容性文档

**文件**：`modules/PROBLEM_COMPATIBILITY.md`

**添加新章节**：
```markdown
### X. YOUR_PROBLEM - 您的问题名称

**问题特点**：描述问题特点

#### 支持的策略
| 策略 | 支持 | 说明 |
|------|------|------|
| Attention Model | ✅ 完全支持 | 说明 |
| POMO | ✅/❌ | 说明 |

#### 支持的算法
| 算法 | 支持 | 说明 |
|------|------|------|
| REINFORCE | ✅ 完全支持 | 说明 |
| PPO | ✅ 完全支持 | 说明 |
| A2C | ✅ 完全支持 | 说明 |

#### 推荐组合
- **最佳质量**：策略 + 算法
- **快速原型**：策略 + 算法
- **学习入门**：策略 + 算法

#### 特有参数
\```python
config = {
    'problem': 'your_problem',
    'your_param_1': value1,  # 参数说明
    'your_param_2': value2,  # 参数说明
}
\```
```

**更新兼容性矩阵表格**：
```markdown
| 问题类型 | 策略支持 | 算法支持 | 特殊说明 |
|---------|---------|---------|---------|
| TSP | ... | ... | ... |
| CVRP | ... | ... | ... |
| **YOUR_PROBLEM** | Attention, POMO | REINFORCE, PPO, A2C | 您的说明 |
```

#### 6.3 创建问题专用文档

**文件**：`modules/problems/YOUR_PROBLEM_GUIDE.md`

**内容**：
```markdown
# YOUR_PROBLEM 完整使用指南

## 问题定义
详细描述问题...

## 数学模型
数学公式和约束...

## 使用方法
步骤说明...

## 参数说明
详细参数说明...

## 示例
完整示例...

## 常见问题
FAQ...
```

---

## 文件清单

### ✅ 必须创建/修改的文件

| 序号 | 文件路径 | 操作 | 说明 |
|------|---------|------|------|
| 1 | `modules/problems/your_problem.py` | 新建 | 问题定义类 |
| 2 | `modules/problems/__init__.py` | 修改 | 注册问题类 |
| 3 | `modules/rl_training/your_problem_trainer.py` | 新建 | 训练器实现 |
| 4 | `modules/rl_training/__init__.py` | 修改 | 注册训练器和路由 |
| 5 | `modules/rl_training/visualizations/your_problem_viz.py` | 新建 | 可视化函数 |
| 6 | `modules/compatibility.py` | 修改 | 添加兼容性配置 |
| 7 | `templates/index.html` | 修改 | 前端集成 |

### 📖 必须更新的文档

| 序号 | 文件路径 | 更新内容 |
|------|---------|---------|
| 8 | `modules/README.md` | 添加问题到架构图、示例 |
| 9 | `modules/PROBLEM_COMPATIBILITY.md` | 添加兼容性章节和矩阵 |
| 10 | `modules/problems/YOUR_PROBLEM_GUIDE.md` | 新建问题使用指南 |

### 📝 可选文档

| 序号 | 文件路径 | 说明 |
|------|---------|------|
| 11 | `YOUR_PROBLEM_QUICKSTART.md` | 快速开始指南 |
| 12 | `YOUR_PROBLEM_INTEGRATION_COMPLETE.md` | 集成完成报告 |

---

## 测试验证

### 第一步：Python 语法验证

```bash
# 编译检查
python3 -m py_compile modules/problems/your_problem.py
python3 -m py_compile modules/rl_training/your_problem_trainer.py
python3 -m py_compile modules/rl_training/visualizations/your_problem_viz.py

# 导入测试
python3 -c "from modules.problems import YourProblem; print('✅ 问题类导入成功')"
python3 -c "from modules.rl_training import YourProblemTrainer; print('✅ 训练器导入成功')"
```

### 第二步：兼容性验证

```bash
python3 -c "
from modules.compatibility import POLICY_PROBLEM_COMPATIBILITY, ALGORITHM_PROBLEM_COMPATIBILITY
print('策略支持:', [p for p, probs in POLICY_PROBLEM_COMPATIBILITY.items() if 'your_problem' in probs])
print('算法支持:', [a for a, probs in ALGORITHM_PROBLEM_COMPATIBILITY.items() if 'your_problem' in probs])
"
```

### 第三步：前端测试

1. 启动应用：`bash scripts/start.sh`
2. 访问：`http://localhost:5000`
3. 检查：
   - ✅ 问题类型下拉菜单中有新选项
   - ✅ 选择问题后显示参数区域
   - ✅ 选择问题后策略/算法选项正确显示
   - ✅ 点击"开始训练"不报错

### 第四步：完整训练测试

配置小规模测试：
```python
config = {
    'problem': 'your_problem',
    'model': 'attention',
    'algorithm': 'reinforce',
    'num_loc': 10,          # 小规模
    'epochs': 2,            # 少轮数
    'batch_size': 64,       # 小批次
}
```

检查：
- ✅ 训练成功启动
- ✅ 实时进度显示正常
- ✅ 训练完成无报错
- ✅ 生成可视化结果

---

## 常见错误

### ❌ 错误 1：子类必须实现 initialize_environment 方法

**原因**：训练器类没有实现 `initialize_environment()` 方法

**解决**：
```python
class YourProblemTrainer(BaseTrainer):
    def initialize_environment(self):  # ← 方法名必须是这个
        env = YourProblemEnv(...)
        return env
```

### ❌ 错误 2：前端策略/算法选项为空

**原因**：`modules/compatibility.py` 中没有添加兼容性配置

**解决**：确保在 `POLICY_PROBLEM_COMPATIBILITY` 和 `ALGORITHM_PROBLEM_COMPATIBILITY` 中添加了问题类型

### ❌ 错误 3：不支持的问题类型

**原因**：训练路由中没有添加分支

**解决**：在 `modules/rl_training/__init__.py` 的 `real_rl4co_training()` 函数中添加 `elif` 分支

### ❌ 错误 4：ModuleNotFoundError

**原因**：忘记在 `__init__.py` 中导入或添加到 `__all__`

**解决**：检查所有 `__init__.py` 文件的导入和导出

### ❌ 错误 5：前端参数不显示

**原因**：JavaScript 中的 `showProblemSpecificParams()` 函数没有更新

**解决**：确保添加了问题类型的判断分支和参数区域的显示/隐藏逻辑

### ❌ 错误 6：参数未传递到后端

**原因**：`startTraining()` 函数中没有提取参数

**解决**：在配置对象中添加参数提取逻辑

### ❌ 错误 7：AttributeError - 'plots_dir' 不存在

**原因**：使用了错误的属性名，`BaseTrainer` 提供的是 `user_plots_dir` 而不是 `plots_dir`

**错误代码**：
```python
def generate_visualizations(self, env, model, trainer, checkpoint_path):
    anim_path = os.path.join(self.plots_dir, anim_filename)  # ❌ 错误
    comp_path = os.path.join(self.plots_dir, comp_filename)  # ❌ 错误
```

**正确代码**：
```python
def generate_visualizations(self, env, model, trainer, checkpoint_path):
    anim_path = os.path.join(self.user_plots_dir, anim_filename)  # ✅ 正确
    comp_path = os.path.join(self.user_plots_dir, comp_filename)  # ✅ 正确
```

**BaseTrainer 提供的属性**：
- ✅ `self.user_plots_dir` - 用户图表目录
- ✅ `self.user_checkpoints_dir` - 用户检查点目录
- ✅ `self.bg_file_manager` - 文件管理器
- ✅ `self.bg_session_manager` - 会话管理器
- ❌ `self.plots_dir` - 不存在
- ❌ `self.checkpoints_dir` - 不存在

### ❌ 错误 8：TypeError - 'tuple' object is not a mapping

**原因**：`generate_visualizations()` 返回了 tuple，但 `BaseTrainer.train()` 期望返回 dict

**错误代码**：
```python
def generate_visualizations(self, env, model, trainer, checkpoint_path):
    animation_paths = []
    comparison_paths = []
    # ... 生成可视化 ...
    return animation_paths, comparison_paths  # ❌ 返回 tuple
```

**正确代码**：
```python
def generate_visualizations(self, env, model, trainer, checkpoint_path):
    animation_paths = []
    comparison_paths = []
    # ... 生成可视化 ...
    
    # 保存检查点
    trainer.save_checkpoint(checkpoint_path)
    
    # 返回字典
    return {  # ✅ 返回 dict
        'animation_paths': animation_paths,
        'comparison_paths': comparison_paths,
        'training_curve': self.training_status[self.session_id].get('plot_url', ''),
        'checkpoint_path': checkpoint_path
    }
```

**为什么必须返回字典**：
```python
# BaseTrainer.train() 第517行
final_results = {
    'model': self.model_type,
    'problem': self.problem_type,
    **results  # ← 这里使用 ** 展开字典，如果是 tuple 会报错
}
```

### ❌ 错误 9：路径类型混淆

**原因**：将本地文件路径添加到返回列表，而不是 URL 路径

**错误代码**：
```python
anim_path = os.path.join(self.user_plots_dir, anim_filename)
animation_paths.append(anim_path)  # ❌ 添加本地路径 /full/path/to/file.gif
```

**正确代码**：
```python
anim_path = os.path.join(self.user_plots_dir, anim_filename)
# 生成文件到 anim_path
create_animation(data, anim_path)
# 添加 URL 路径（前端访问）
animation_paths.append(f'/static/model_plots/user_{self.user_id}/{anim_filename}')  # ✅
```

---

### ❌ 错误 10：忘记保存文件记录到数据库

**原因**：生成了可视化文件，但没有保存文件记录到数据库

**错误代码**：
```python
# 生成文件
create_visualization(data, file_path)
animation_paths.append(f'/static/model_plots/user_{self.user_id}/{filename}')
# ❌ 没有保存到数据库
```

**正确代码**：
```python
# 生成文件
create_visualization(data, file_path)

# 保存到数据库
if self.bg_file_manager:
    try:
        self.bg_file_manager.save_file_record(
            user_id=self.user_id,
            session_id=self.session_id,
            filename=filename,
            file_type='animation',  # 或 'plot', 'checkpoint'
            file_path=file_path
        )
    except Exception as e:
        print(f"保存文件记录失败: {str(e)}")

animation_paths.append(f'/static/model_plots/user_{self.user_id}/{filename}')
```

### ❌ 错误 11：返回字段名与前端不一致

**原因**：返回字典的字段名与前端期望的字段名不匹配

**错误代码**：
```python
def generate_visualizations(self, env, model, trainer, checkpoint_path):
    comparison_paths = []
    animation_paths = []
    # ... 生成可视化 ...
    
    return {
        'comparison_paths': comparison_paths,  # ❌ 前端不认识这个字段
        'animation_paths': animation_paths,
        'training_curve': '...',
        'checkpoint_path': '...'
    }
```

**正确代码**：
```python
def generate_visualizations(self, env, model, trainer, checkpoint_path):
    comparison_paths = []
    animation_paths = []
    # ... 生成可视化 ...
    
    return {
        'plot_paths': comparison_paths,  # ✅ 前端期望的字段名
        'animation_paths': animation_paths,
        'training_curve': '...',
        'checkpoint_path': '...'
    }
```

**前端期望的字段**（templates/index.html）：
```javascript
// 前端检查是否有可视化图片
if (results.plot_paths && results.plot_paths.length > 0) {
    // 显示静态对比图
    results.plot_paths.forEach((path, index) => {
        // 显示图片
    });
}
```

**标准返回格式**（所有训练器必须一致）：
```python
return {
    'plot_paths': [...],        # ✅ 静态对比图路径列表
    'animation_paths': [...],   # ✅ 动画路径列表
    'training_curve': '...',    # ✅ 训练曲线路径
    'checkpoint_path': '...'    # ✅ 检查点路径
}
```

**症状**：
- 文件已生成，但前端不显示
- 浏览器控制台没有错误
- 后端日志显示成功
- 但是前端页面空白（没有可视化图片）

---

## 示例：添加 mTSP

以下是添加 mTSP（多旅行商问题）的完整示例。

### 文件修改记录

#### 1. `modules/problems/mtsp.py` (新建)
```python
class MTSProblem(BaseProblem):
    def __init__(self):
        super().__init__()
        self.problem_name = "mtsp"
        self.display_name = "多旅行商问题"
    # ... 完整代码见实际文件
```

#### 2. `modules/problems/__init__.py` (修改)
```python
from .mtsp import MTSProblem  # 添加导入
PROBLEM_REGISTRY = {
    'mtsp': MTSProblem,  # 添加注册
}
```

#### 3. `modules/rl_training/mtsp_trainer.py` (新建)
```python
class MTSPTrainer(BaseTrainer):
    def initialize_environment(self):  # ← 必须实现
        env = MTSPEnv(...)
        return env
```

#### 4. `modules/rl_training/__init__.py` (修改)
```python
from .mtsp_trainer import MTSPTrainer, train_mtsp  # 导入

elif problem_type == 'mtsp':  # 路由
    train_mtsp(...)
```

#### 5. `modules/compatibility.py` (修改)
```python
POLICY_PROBLEM_COMPATIBILITY = {
    'attention': [..., 'mtsp'],  # 添加
    'pomo': [..., 'mtsp'],       # 添加
}
```

#### 6. `templates/index.html` (修改)
```html
<option value="mtsp">mTSP - 多旅行商问题 ⭐新增</option>
<!-- 添加参数区域 -->
<div id="mtsp-params">...</div>
```

#### 7. 文档更新
- ✅ `modules/README.md` - 添加 mTSP 章节
- ✅ `modules/PROBLEM_COMPATIBILITY.md` - 添加兼容性说明
- ✅ `modules/problems/MTSP_GUIDE.md` - 创建使用指南

### 测试结果
```
✅ 问题类导入成功
✅ 训练器导入成功
✅ 兼容性配置正确
✅ 前端显示正常
✅ 训练成功完成
✅ 可视化生成成功
```

---

## 总结

### ✅ 添加新问题的核心要点

1. **必须实现 `initialize_environment()` 方法**
   - 这是最常见的错误！
   - BaseTrainer 会调用此方法创建环境

2. **完整的注册流程**
   - 问题类 → `problems/__init__.py`
   - 训练器 → `rl_training/__init__.py`
   - 路由 → `real_rl4co_training()` 函数

3. **兼容性配置不能漏**
   - 策略兼容性
   - 算法兼容性
   - 推荐配置

4. **前端三步走**
   - 添加选项
   - 添加参数区域
   - 更新 JavaScript 逻辑

5. **文档要完善**
   - 架构文档
   - 兼容性文档
   - 使用指南

### 📊 工作量估算

| 任务 | 预计时间 |
|------|---------|
| 问题定义 | 30 分钟 |
| 训练器实现 | 60 分钟 |
| 可视化函数 | 45 分钟 |
| 前端集成 | 30 分钟 |
| 兼容性配置 | 15 分钟 |
| 文档编写 | 45 分钟 |
| 测试验证 | 30 分钟 |
| **总计** | **4-5 小时** |

### 🎯 检查清单

开始前：
- [ ] 确认 RL4CO 支持该问题
- [ ] 了解问题的数学定义
- [ ] 准备测试数据

实现中：
- [ ] 创建问题定义类
- [ ] 创建训练器类（包含 `initialize_environment()`）
- [ ] 创建可视化函数
- [ ] 更新所有 `__init__.py`
- [ ] 添加兼容性配置
- [ ] 前端集成（选项、参数、JS）
- [ ] 更新文档

完成后：
- [ ] Python 语法验证通过
- [ ] 兼容性验证通过
- [ ] 前端显示正常
- [ ] 训练测试成功
- [ ] 可视化正常生成
- [ ] 文档完整准确

---

**文档版本**：v1.0  
**最后更新**：2026-02-04  
**维护者**：RL4CO Display Team

---

## 附录

### A. BaseTrainer 抽象方法

子类**必须实现**的方法：
```python
def initialize_environment(self):
    """初始化环境（必须实现）"""
    raise NotImplementedError("子类必须实现 initialize_environment 方法")
```

子类**可以重写**的方法：
```python
def create_policy(self, env):
    """创建策略（可重写，默认实现）"""
    pass

def create_model(self, env, policy):
    """创建模型（可重写，默认实现）"""
    pass

def generate_visualizations(self, env, model, trainer, checkpoint_path):
    """生成可视化（可选实现）"""
    pass

def get_training_summary(self):
    """获取训练总结（可选实现）"""
    pass
```

### B. 参考实现

- **简单对称问题**：参考 `TSPTrainer`
- **带容量约束**：参考 `CVRPTrainer`
- **多代理问题**：参考 `MTSPTrainer`
- **复杂约束**：参考 `SDVRPTrainer`、`VRPTWTrainer`

### C. 常用配置参数

```python
# 通用参数
config = {
    'problem': 'your_problem',      # 问题类型
    'model': 'pomo',                # 策略模型
    'algorithm': 'ppo',             # RL算法
    'num_loc': 50,                  # 问题规模
    'epochs': 10,                   # 训练轮数
    'batch_size': 512,              # 批次大小
    'learning_rate': 1e-4,          # 学习率
    
    # POMO 特有
    'num_starts': 50,               # 多起点数量
    
    # PPO 特有
    'clip_ratio': 0.2,              # 裁剪比率
    
    # A2C 特有
    'value_loss_coef': 0.5,         # 价值损失系数
    'entropy_coef': 0.01,           # 熵系数
}
```

### D. 有用的链接

- [RL4CO 官方文档](https://github.com/ai4co/rl4co)
- [PyTorch Lightning 文档](https://lightning.ai/docs/pytorch/stable/)
- [项目架构说明](../modules/README.md)
- [兼容性矩阵](../modules/COMPATIBILITY_MATRIX.md)
