# 组合优化问题类型模块

## 📦 **模块说明**

这个模块提供了统一的问题类型管理接口，方便添加新的组合优化问题。

---

## 📂 **目录结构**

```
modules/problems/
│
├── __init__.py              # 模块入口，问题注册表
├── base_problem.py          # 问题基类
│
├── tsp.py                   # TSP问题类
├── cvrp.py                  # CVRP问题类
│
├── templates/               # 新问题模板
│   └── problem_template.py  # 问题类模板
│
└── README.md                # 本文档
```

---

## 🎯 **核心概念**

### 问题类 (Problem Class)

每个问题类型都是 `BaseProblem` 的子类，提供：
- **环境创建**: `create_environment()` - 创建RL4CO环境
- **可视化函数**: `get_visualization_functions()` - 返回可视化函数集合
- **参数管理**: 问题特定的参数和验证
- **元信息**: 问题描述、特征等

### 问题注册表 (PROBLEM_REGISTRY)

所有问题类型都注册在 `PROBLEM_REGISTRY` 中，通过统一接口调用。

---

## 🚀 **使用方式**

### 1. 获取问题类

```python
from modules.problems import get_problem_class

# 获取TSP问题类
TSProblem = get_problem_class('tsp')

# 创建问题实例
config = {
    'num_loc': 50,
    'batch_size': 512,
}
tsp = TSProblem(config)
```

### 2. 创建环境

```python
# 创建RL4CO环境
env = tsp.create_environment()

# 使用环境训练
# ...
```

### 3. 获取可视化函数

```python
# 获取可视化函数
viz_funcs = tsp.get_visualization_functions()

# 创建动画
viz_funcs['animation'](td, actions, save_path='tsp_route.gif')

# 创建对比图
viz_funcs['comparison'](env, td, actions_untrained, rewards_untrained,
                        actions_trained, rewards_trained, save_path='comparison.png')
```

### 4. 列出所有问题

```python
from modules.problems import list_available_problems, get_problem_info

# 列出所有可用问题
problems = list_available_problems()
for problem_type, status, cn_name in problems:
    print(f"{problem_type}: {cn_name} ({status})")

# 获取问题详细信息
info = get_problem_info('tsp')
print(info)
```

---

## 📝 **在app.py中使用**

### 原有方式（耦合）

```python
# app.py - 旧版本
from modules.rl_training import train_tsp, train_cvrp

if problem_type == 'tsp':
    train_tsp(config, ...)
elif problem_type == 'cvrp':
    train_cvrp(config, ...)
```

### 新方式（解耦）

```python
# app.py - 新版本
from modules.problems import get_problem_class, get_problem_info

# 获取问题类
try:
    ProblemClass = get_problem_class(problem_type)
    problem = ProblemClass(config)
except ValueError as e:
    return jsonify({'error': str(e)}), 400

# 验证配置
valid, error_msg = problem.validate_config()
if not valid:
    return jsonify({'error': error_msg}), 400

# 创建环境
env = problem.create_environment()

# 获取可视化函数
viz_funcs = problem.get_visualization_functions()

# 传递给训练器
trainer = UnifiedTrainer(
    config=config,
    problem=problem,
    env=env,
    viz_funcs=viz_funcs,
    ...
)
```

---

## ➕ **添加新问题类型**

### 步骤1：创建问题类文件

```bash
# 复制模板
cp modules/problems/templates/problem_template.py modules/problems/pctsp.py
```

### 步骤2：实现问题类

```python
# modules/problems/pctsp.py
from typing import Dict, Any, Callable
from .base_problem import BaseProblem


class PCTSProblem(BaseProblem):
    """奖励收集旅行商问题"""
    
    def _init_problem_params(self):
        """初始化PCTSP特定参数"""
        self.prize_required = float(self.config.get('prize_required', 1.0))
    
    def get_problem_type(self) -> str:
        return 'pctsp'
    
    def get_problem_name(self) -> str:
        return 'Prize Collecting TSP'
    
    def create_environment(self):
        from rl4co.envs import PCTSPEnv
        return PCTSPEnv(generator_params=self.get_env_params())
    
    def get_visualization_functions(self) -> Dict[str, Callable]:
        from modules.rl_training.visualizations.pctsp_viz import (
            create_pctsp_route_animation,
            create_pctsp_comparison_plot
        )
        return {
            'animation': create_pctsp_route_animation,
            'comparison': create_pctsp_comparison_plot,
        }
    
    def get_env_params(self) -> Dict[str, Any]:
        return {
            'num_loc': self.num_loc,
            'prize_required': self.prize_required,
        }
```

### 步骤3：注册到系统

```python
# modules/problems/__init__.py

from .pctsp import PCTSProblem  # 新增

PROBLEM_REGISTRY = {
    'tsp': TSProblem,
    'cvrp': CVRProblem,
    'pctsp': PCTSProblem,  # 新增
}

PROBLEM_INFO = {
    # ... 其他问题 ...
    'pctsp': {  # 新增
        'name': 'PCTSP',
        'full_name': 'Prize Collecting TSP',
        'cn_name': '奖励收集旅行商问题',
        'category': 'routing',
        'difficulty': 'medium',
        'status': 'active',
        'description': '收集足够奖励的前提下最小化路径',
        'params': ['num_loc', 'prize_required'],
        'features': ['奖励约束', '部分访问', '灵活规划'],
    },
}
```

### 步骤4：创建可视化函数

```python
# modules/rl_training/visualizations/pctsp_viz.py

def create_pctsp_route_animation(td, actions, save_path, ...):
    """创建PCTSP路线动画"""
    # 类似TSP，但要显示奖励值
    pass

def create_pctsp_comparison_plot(env, td, ...):
    """创建PCTSP对比图"""
    # 类似TSP，但要显示收集的奖励
    pass
```

### 步骤5：更新前端

```html
<!-- templates/index.html -->
<option value="pctsp">PCTSP (Prize Collecting TSP)</option>
```

**完成！** 🎉 新问题类型即可使用！

---

## 🎨 **问题类型分类**

### 路径规划问题 (Routing)
- TSP - 旅行商问题 ✅
- CVRP - 车辆路径问题 ✅
- PCTSP - 奖励收集TSP 📦
- OP - 定向问题 📦
- SDVRP - 分割配送VRP 📦

### 调度问题 (Scheduling)
- JSSP - 作业车间调度 📦
- FJSP - 柔性作业车间调度 📦
- FFSP - 柔性流水车间 📦

### 装箱问题 (Packing)
- BPP - 装箱问题 📦
- KP - 背包问题 📦
- MKP - 多背包问题 📦

### 图问题 (Graph)
- MCT - 最大割问题 📦
- MIS - 最大独立集 📦
- MVC - 最小顶点覆盖 📦

---

## 📊 **问题元信息**

每个问题都包含以下元信息：

```python
{
    'name': 'TSP',                          # 缩写
    'full_name': 'Traveling Salesman Problem',  # 完整英文名
    'cn_name': '旅行商问题',                 # 中文名
    'category': 'routing',                  # 分类
    'difficulty': 'medium',                 # 难度
    'status': 'active',                     # 状态 (active/planned/experimental)
    'description': '寻找访问所有城市的最短路径',  # 描述
    'params': ['num_loc'],                  # 必需参数
    'features': ['无容量约束', '单条路径'],    # 特征
}
```

---

## 🔧 **API参考**

### get_problem_class(problem_type)

获取问题类。

**参数**:
- `problem_type` (str): 问题类型名称

**返回**: 问题类（BaseProblem的子类）

**异常**: `ValueError` - 如果问题类型不支持

---

### get_problem_info(problem_type)

获取问题元信息。

**参数**:
- `problem_type` (str, optional): 问题类型，None返回所有

**返回**: dict - 问题信息字典

---

### list_available_problems()

列出所有可用问题。

**返回**: list - [(问题类型, 状态, 中文名)]

---

### list_problems_by_category()

按类别列出问题。

**返回**: dict - {类别: [问题列表]}

---

## ✅ **最佳实践**

### 1. 统一接口
所有问题类型都使用相同的接口，便于统一处理。

### 2. 解耦设计
问题定义与训练逻辑分离，易于维护和扩展。

### 3. 自描述
每个问题类都包含完整的元信息和文档。

### 4. 参数验证
提供参数验证机制，防止配置错误。

### 5. 可扩展性
新问题类型只需实现几个方法即可集成。

---

## 📚 **相关文档**

- [问题模板](./templates/problem_template.py)
- [RL4CO完整参考](../../RL4CO_COMPLETE_REFERENCE.md)
- [训练模块文档](../rl_training/README.md)

---

**创建日期**: 2024年12月16日  
**维护者**: RL4CO Display Team  
**状态**: ✅ **活跃维护**


