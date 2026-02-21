# RL4CO Display 问题类型集成历史

> 记录项目中各种问题类型和策略模型的集成过程

**最后更新**: 2026-02-17

---

## 概述

本文档记录了 RL4CO Display 平台集成各种组合优化问题和强化学习策略的历史。每个集成都包含实现细节、遇到的挑战和解决方案。

---

## 已集成问题类型

### 路由问题 (Routing)

| 问题 | 状态 | 集成时间 | 支持策略 |
|------|------|----------|----------|
| TSP | ✅ 已完成 | 2024-01 | AM, POMO, PtrNet |
| ATSP | ✅ 已完成 | 2026-01 | AM, MatNet |
| mTSP | ✅ 已完成 | 2026-02 | AM, POMO |
| CVRP | ✅ 已完成 | 2024-02 | AM, POMO |
| SDVRP | ✅ 已完成 | 2024-03 | AM |
| VRPTW | ✅ 已完成 | 2024-04 | AM |
| PDP | ✅ 已完成 | 2026-02 | AM |
| OP | ✅ 已完成 | 2026-02 | AM |

### 调度问题 (Scheduling)

| 问题 | 状态 | 集成时间 | 支持策略 |
|------|------|----------|----------|
| FFSP | ✅ 已完成 | 2026-02 | MatNet |

---

## 策略模型集成

### 1. Attention Model (AM)

**集成时间**: 2024-01  
**状态**: ✅ 已完成

**支持问题**:
- TSP, ATSP, CVRP, SDVRP, VRPTW, PDP, OP, mTSP

**特点**:
- 基于 Transformer 的注意力机制
- 通用性强，适用于大多数路由问题
- 性能中等，速度快

---

### 2. POMO

**集成时间**: 2024-02  
**状态**: ✅ 已完成

**支持问题**:
- TSP, CVRP, mTSP

**特点**:
- 多起点优化，利用问题对称性
- 质量高，但计算量大
- 只适用于对称问题

---

### 3. PtrNet (Pointer Network)

**集成时间**: 2026-01  
**状态**: ✅ 已完成

**支持问题**:
- TSP, CVRP

**特点**:
- 2015年的经典方法
- 基于 LSTM 的序列到序列模型
- 性能不如现代方法，主要用于学习和对比

**集成挑战**:
- RL4CO 无独立 PtrNet 实现
- 使用简化的 AM 模拟

**相关文档**: `PTRNET_INTEGRATION_COMPLETE.md`

---

### 4. MatNet

**集成时间**: 2026-02  
**状态**: ✅ 已完成

**支持问题**:
- ATSP, FFSP

**特点**:
- 矩阵注意力机制
- 专为非对称和调度问题设计
- FFSP 必须使用此策略

**集成挑战**:
1. 环境适配：需要 `cost_matrix` 键
2. 参数不匹配：`out_bias` vs `out_bias_pointer_attn`
3. 方法缺失：`IndexTables.augment_machine_tables`

**解决方案**:
- 环境适配器 (`FFSPEnvWithCostMatrix`)
- Monkey Patch 修复参数和方法

**相关文档**: `FFSP_INTEGRATION_ERRORS.md`

---

## 问题类型详细集成记录

### TSP (Traveling Salesman Problem)

**集成时间**: 2024-01  
**状态**: ✅ 项目初始支持

**实现**:
- 环境: `rl4co.envs.TSPEnv`
- 训练器: `modules/rl_training/tsp_trainer.py`
- 可视化: `modules/rl_training/visualizations/tsp_viz.py`

**特色功能**:
- 路径生成动画 (GIF)
- 训练前后对比图
- 实时训练曲线

---

### ATSP (Asymmetric TSP)

**集成时间**: 2026-01  
**状态**: ✅ 已完成

**挑战**:
- 非对称距离矩阵
- POMO 不支持（需要对称性）

**解决方案**:
- 使用 Attention Model 或 MatNet
- 前端显示距离矩阵非对称特性

---

### mTSP (Multiple TSP)

**集成时间**: 2026-02  
**状态**: ✅ 已完成

**挑战**:
1. 可视化维度不匹配
2. 前端显示问题
3. 多代理路径分配

**解决方案**:
- 修复张量维度处理
- 优化可视化生成逻辑
- 支持 minmax 和 sum 两种优化目标

**相关文档**: `mTSP完整修复报告-最终版.md`

---

### PDP (Pickup and Delivery Problem)

**集成时间**: 2026-02  
**状态**: ✅ 已完成

**特点**:
- 取货和送货配对约束
- 路径可行性验证

**实现**:
- 环境: `rl4co.envs.PDPEnv`
- 训练器: `modules/rl_training/pdp_trainer.py`
- 可视化: `modules/rl_training/visualizations/pdp_viz.py`

**相关文档**: `PDP_INTEGRATION_COMPLETE.md`

---

### OP (Orienteering Problem)

**集成时间**: 2026-02  
**状态**: ✅ 已完成

**特点**:
- 时间预算约束
- 最大化收集奖励
- 不需要访问所有节点

**实现**:
- 环境: `rl4co.envs.OPEnv`
- 训练器: `modules/rl_training/op_trainer.py`
- 可视化: `modules/rl_training/visualizations/op_viz.py`

**相关文档**: `OP_INTEGRATION_COMPLETE.md`

---

### FFSP (Flexible Flow Shop Problem)

**集成时间**: 2026-02  
**状态**: ✅ 已完成

**这是最复杂的集成**，遇到了多个技术挑战：

**挑战列表**:
1. ❌ 策略参数属性缺失
2. ❌ 解码器参数名称不匹配
3. ❌ TensorDict 缺少 `cost_matrix` 键
4. ❌ 环境包装器导致无限递归
5. ❌ Lightning 训练缺少 `dataset` 方法
6. ❌ `IndexTables` 缺少方法

**解决方案**:
- 创建环境适配器
- Monkey Patch 修复 RL4CO bug
- 显式属性代理避免递归
- 全局修补类方法

**详细文档**: 
- `FFSP_INTEGRATION_ERRORS.md` (21KB) - 完整错误排查
- `FFSP_INTEGRATION_BEST_PRACTICES.md` (14KB) - 最佳实践

---

## 通用集成模式

基于多个问题类型的集成经验，我们总结了通用模式：

### 1. 模块化结构

```
modules/
├── problems/
│   ├── base_problem.py
│   └── {problem}_problem.py
├── rl_training/
│   ├── base_trainer.py
│   ├── {problem}_trainer.py
│   └── visualizations/
│       └── {problem}_viz.py
└── policies/
    └── {policy}_policy.py
```

### 2. 训练器基类

所有训练器继承 `BaseTrainer`：
- `initialize_environment()`: 创建环境
- `create_policy()`: 创建策略
- `create_model()`: 创建 RL 模型
- `generate_visualizations()`: 生成可视化

### 3. 环境适配器模式

当环境和策略接口不匹配时：
```python
class CustomEnvAdapter:
    def __init__(self, base_env):
        # 显式代理所有属性
        
    def reset(self, *args, **kwargs):
        td = self._base_env.reset(*args, **kwargs)
        # 添加/转换键
        return td
```

### 4. 兼容性检查

使用 `modules/compatibility.py` 验证：
- 策略-问题兼容性
- 算法-问题兼容性
- 推荐配置

---

## 集成清单

新问题类型集成需要完成：

### 代码实现
- [ ] 问题类 (`modules/problems/`)
- [ ] 训练器 (`modules/rl_training/`)
- [ ] 可视化函数 (`modules/rl_training/visualizations/`)
- [ ] 注册到 `PROBLEM_REGISTRY`
- [ ] 更新兼容性矩阵

### 前端集成
- [ ] 添加问题类型选项
- [ ] 问题特定参数面板
- [ ] 可视化展示逻辑

### 文档
- [ ] 快速开始指南
- [ ] API 文档
- [ ] 故障排查指南

### 测试
- [ ] 单元测试
- [ ] 集成测试
- [ ] 可视化验证

---

## 经验教训

### 1. 环境兼容性

**教训**: 不同环境的 TensorDict 键名可能不一致

**解决**: 使用适配器统一接口

### 2. 避免 `__getattr__`

**教训**: 动态属性代理会被框架内省触发，导致递归

**解决**: 显式代理所有必要属性

### 3. Monkey Patch 的使用

**教训**: RL4CO 库可能有 bug，直接修改源码不利于版本控制

**解决**: 运行时修补，文档化修补原因

### 4. 策略-问题匹配

**教训**: 不是所有策略都支持所有问题

**解决**: 建立兼容性矩阵，前端自动切换

### 5. 可视化的重要性

**教训**: 用户需要直观看到训练效果

**解决**: 为每个问题类型设计专门的可视化

---

## 待集成问题类型

### 计划中
- PCTSP (Prize Collecting TSP)
- SPCTSP (Stochastic PCTSP)
- CVRPTW (CVRP with Time Windows)
- JSSP (Job Shop Scheduling)
- FJSP (Flexible Job Shop)

### 评估中
- SVRP (Skill VRP)
- DPP (Dial-a-Ride Problem)
- MTVRP (Multi-Task VRP)

---

## 相关资源

### 内部文档
- [添加新问题类型指南](../ADD_NEW_PROBLEM_TYPE_GUIDE.md)
- [FFSP完整指南](../guides/FFSP_COMPLETE_GUIDE.md)
- [mTSP完整指南](../guides/MTSP_COMPLETE_GUIDE.md)

### 外部资源
- [RL4CO GitHub](https://github.com/ai4co/rl4co)
- [RL4CO Documentation](https://rl4co.readthedocs.io/)

---

## 贡献者

本项目的问题类型集成由以下人员完成：
- 项目核心团队
- 山西大学计算机科学与技术学院

---

**本文档持续更新中...**
