# FFSP (Flexible Flow Shop Problem) 完整指南

> 柔性流水车间调度问题 - 从快速开始到问题排查的完整手册

**最后更新**: 2026-02-17  
**版本**: v2.0

---

## 📖 目录

1. [快速开始](#快速开始)
2. [问题介绍](#问题介绍)
3. [参数配置](#参数配置)
4. [训练建议](#训练建议)
5. [可视化说明](#可视化说明)
6. [常见问题](#常见问题)
7. [故障排查](#故障排查)
8. [最佳实践](#最佳实践)

---

## 快速开始

### 3步开始训练

**第 1 步：选择问题类型**
- 在训练页面选择 **FFSP - 柔性流水车间调度**
- 系统会自动显示 FFSP 专用参数面板

**第 2 步：配置参数（推荐配置）**
```
阶段数: 2-3
机器数/阶段: 2-4
作业数: 10-20
最小/最大加工时间: 2/10
阶段展平模式: 启用
```

**第 3 步：选择训练算法**
- 策略模型: **MatNet**（自动选择）
- 训练算法: **PPO**（推荐）或 A2C
- 训练轮次: 10-30

---

## 问题介绍

### 什么是 FFSP？

FFSP（Flexible Flow Shop Problem）是经典的生产调度问题：
- 多个生产**阶段**按顺序执行
- 每个阶段有多台**并行机器**
- 多个**作业**需要按阶段依次加工
- 目标：最小化**完工时间** (Makespan)

### 实际应用场景

1. **制造业**：流水线生产调度
2. **半导体**：芯片制造流程
3. **纺织业**：纺纱、织布、染色工序
4. **食品加工**：清洗、切割、包装流程

### 为什么要用 MatNet？

FFSP 是**调度问题**，与路由问题（TSP/CVRP）本质不同：
- ✅ **MatNet**：专为调度设计，使用矩阵注意力处理作业-机器关系
- ❌ **Attention Model / POMO**：为路由问题优化，不支持 FFSP

---

## 参数配置

### 核心参数说明

| 参数 | 含义 | 范围 | 推荐值 |
|------|------|------|--------|
| **阶段数** (num_stage) | 生产工序数 | 2-10 | 3 |
| **机器数/阶段** (num_machine) | 每阶段并行机器 | 1-10 | 4 |
| **作业数** (num_job) | 需调度工件数 | 5-100 | 20 |
| **最小加工时间** (min_time) | 最短加工时间 | 1-10 | 2 |
| **最大加工时间** (max_time) | 最长加工时间 | 2-50 | 10 |
| **阶段展平** (flatten_stages) | 独立编码模式 | True/False | True |

### 规模计算

```
总机器数 = 阶段数 × 机器数/阶段
总操作数 = 作业数 × 阶段数
状态空间 = 作业数^总机器数（极其巨大）
```

**示例**：3阶段 × 4机器 × 20作业 = 12台机器，60个操作

### 配置模板

#### 入门配置（快速验证）
```yaml
阶段数: 2
机器数/阶段: 2  
作业数: 10
训练轮次: 10
算法: REINFORCE
```
⏱️ 训练时间: ~5分钟

#### 标准配置（推荐）
```yaml
阶段数: 3
机器数/阶段: 4
作业数: 20
训练轮次: 30
算法: PPO
```
⏱️ 训练时间: ~20分钟

#### 生产配置（高质量）
```yaml
阶段数: 5
机器数/阶段: 6
作业数: 50
训练轮次: 50
算法: PPO
Batch Size: 512
```
⏱️ 训练时间: ~90分钟

---

## 训练建议

### 算法选择

| 算法 | 特点 | 适用场景 | 收敛速度 |
|------|------|----------|----------|
| **REINFORCE** | 简单易懂 | 小规模测试 | 慢（20-30轮） |
| **PPO** | 稳定性好 | 生产环境 | 快（15-25轮） |
| **A2C** | 收敛快 | 快速验证 | 最快（10-20轮） |

**FFSP 推荐**：PPO 或 A2C（因为问题复杂，REINFORCE 不够稳定）

### 超参数调优

**Learning Rate**:
- 小问题: 1e-4（默认）
- 大问题: 5e-5（更稳定）

**Batch Size**:
- GPU充足: 512-1024
- GPU有限: 256
- CPU only: 128

**Embed Dim** (MatNet):
- 标准: 256（默认）
- 高质量: 512（训练更慢）

---

## 可视化说明

### 甘特图解读

FFSP 训练完成后会生成**甘特图**（Gantt Chart）：

```
时间 →
┌─────────────────────────────────────┐
│ M0 │ J0 │  J2  │ J1 │              │ Stage 0
├─────────────────────────────────────┤
│ M1 │   J3   │ J0 │  J1  │          │ Stage 0
├─────────────────────────────────────┤
│ M2 │ J0 │    J1    │  J2  │        │ Stage 1
├─────────────────────────────────────┤
│ M3 │  J2  │ J3 │   J0   │          │ Stage 1
└─────────────────────────────────────┘
```

**图例**：
- **横轴**: 时间进度（0 → Makespan）
- **纵轴**: 机器编号（按阶段分组）
- **彩色块**: 不同作业（J0, J1, J2...）
- **块宽度**: 加工时间长度
- **阶段分隔线**: 黑色横线

### 性能指标

**Makespan（完工时间）**：
- 定义：所有作业完成的总时间
- 目标：越小越好
- 改进：通常 15-40%

**训练曲线**：
- Loss：逐渐下降
- Reward：逐渐上升（接近 0）
- 最佳 Reward：标记为红星

---

## 常见问题

### Q1: 为什么训练失败提示 "MatNet required"？

**原因**: FFSP 必须使用 MatNet 策略

**解决**: 系统会自动切换，如果手动修改请改回 MatNet

---

### Q2: 训练速度很慢怎么办？

**优化方案**：

1. **减小问题规模**
   ```
   作业数: 50 → 20
   阶段数: 5 → 3
   ```

2. **使用 GPU 加速**
   ```bash
   # 检查 GPU 是否可用
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **切换到 A2C 算法**（比 PPO 快 30%）

4. **减少训练轮次**
   ```
   轮次: 50 → 20
   ```

---

### Q3: GPU 内存不足 (OOM) 怎么办？

**错误信息**: `CUDA out of memory`

**解决方案**：
```python
# 方案1: 减小 batch size
Batch Size: 512 → 256 → 128

# 方案2: 减小问题规模
作业数: 50 → 30 → 20

# 方案3: 使用 CPU（较慢）
在训练配置中禁用 GPU
```

---

### Q4: 如何提高训练质量？

**质量优化清单**：

1. ✅ 增加训练轮次（50 epochs）
2. ✅ 使用 PPO 算法（更稳定）
3. ✅ 增大 batch size（512-1024）
4. ✅ 降低学习率（1e-4 → 5e-5）
5. ✅ 多次训练取最优结果

---

### Q5: flatten_stages 参数如何选择？

**启用 (True) - 推荐**:
- 每个阶段的机器独立编码
- 性能更好，收敛更快
- 适用于大多数场景

**禁用 (False)**:
- 使用多阶段 FFSP 策略
- 不同阶段共享 embedding
- 适用于特殊需求

**建议**: 保持默认（True）

---

## 故障排查

### 错误 #1: RL4CO 库未安装 MatNet

**错误信息**:
```
ModuleNotFoundError: No module named 'rl4co.models.zoo.matnet'
```

**解决方案**:
```bash
pip install rl4co --upgrade
# 或者
pip install rl4co>=0.6.0
```

---

### 错误 #2: TensorDict 缺少 cost_matrix 键

**错误信息**:
```
KeyError: 'key "cost_matrix" not found in TensorDict'
```

**原因**: MatNet 需要 `cost_matrix`，但 FFSP 环境只提供 `run_time`

**解决**: 系统已自动修复（使用环境适配器）

**验证**:
```python
# 检查适配器是否生效
from modules.rl_training.ffsp_trainer import FFSPEnvWithCostMatrix
```

---

### 错误 #3: IndexTables 缺少 augment_machine_tables 方法

**错误信息**:
```
AttributeError: 'IndexTables' object has no attribute 'augment_machine_tables'
```

**原因**: RL4CO 库的 bug

**解决**: 系统已自动修补（在 `ffsp_trainer.py` 中）

**验证**:
```python
from rl4co.envs.scheduling.ffsp.env import IndexTables
print(hasattr(IndexTables, 'augment_machine_tables'))  # 应该为 True
```

---

### 错误 #4: 解码器参数不匹配

**错误信息**:
```
TypeError: __init__() got an unexpected keyword argument 'out_bias'
```

**原因**: `MatNetPolicy` 传递 `out_bias`，但解码器期望 `out_bias_pointer_attn`

**解决**: 系统已自动修补（Monkey Patch）

---

### 警告: REINFORCE 收敛慢

**警告信息**: `训练进度慢，Loss 不下降`

**原因**: REINFORCE 方差大，不适合复杂的 FFSP 问题

**解决**: 切换到 PPO 或 A2C 算法

---

## 最佳实践

### 开发流程

1. **小规模测试**（2×2×10，10轮）- 验证配置正确
2. **标准规模验证**（3×4×20，30轮）- 评估性能
3. **生产规模训练**（5×6×50，50轮）- 获得最优模型

### 性能基准

| 规模 | Makespan (训练前) | Makespan (训练后) | 改进 |
|------|-------------------|-------------------|------|
| 2×2×10 | ~15-20 | ~12-16 | 15-20% |
| 3×4×20 | ~30-40 | ~22-30 | 20-30% |
| 5×6×50 | ~80-100 | ~55-70 | 25-35% |

### 调试技巧

**启用详细日志**:
```python
# 在训练配置中
logger_level = "DEBUG"
```

**检查中间结果**:
```python
# 每 5 轮保存 checkpoint
checkpoint_interval = 5
```

**对比多次训练**:
```bash
# 运行 3 次取最优
for i in {1..3}; do
    python train_ffsp.py --seed $i
done
```

### 代码集成

如果你要在代码中使用 FFSP 训练：

```python
from modules.rl_training.ffsp_trainer import FFSPTrainer

config = {
    'problem': 'ffsp',
    'model': 'matnet',
    'algorithm': 'ppo',
    'num_stage': 3,
    'num_machine': 4,
    'num_job': 20,
    'epochs': 30,
    'batch_size': 512,
}

trainer = FFSPTrainer(config, session_id, user_id, queue, training_status, get_db)
trainer.train()
```

---

## 技术细节

### 环境适配器

为了兼容 MatNet，项目实现了 `FFSPEnvWithCostMatrix` 适配器：

```python
class FFSPEnvWithCostMatrix:
    """FFSP 环境适配器，为 MatNet 添加 cost_matrix"""
    
    def reset(self, *args, **kwargs):
        td = self._base_env.reset(*args, **kwargs)
        # 关键：添加 MatNet 需要的键
        if 'run_time' in td.keys():
            td['cost_matrix'] = td['run_time']
        return td
```

### Monkey Patch

为了修复 RL4CO 的 bug，项目使用了运行时修补：

```python
# 修补 IndexTables.augment_machine_tables
def augment_machine_tables(self, num_starts):
    if hasattr(self, 'bs'):
        self.bs = self.bs * num_starts
    else:
        self.set_bs(num_starts)

IndexTables.augment_machine_tables = augment_machine_tables
```

---

## 进阶阅读

### 学术资源

1. **MatNet 论文**: Kwon et al., "Matrix Encoding Networks for Neural Combinatorial Optimization" (2021)
2. **FFSP 综述**: Ruiz & Vázquez-Rodríguez, "The hybrid flow shop scheduling problem" (2010)
3. **RL4CO 框架**: https://github.com/ai4co/rl4co

### 相关问题

- **JSSP** (Job Shop Scheduling): 工件路径不固定
- **FJSP** (Flexible Job Shop): JSSP + 机器选择
- **SMTWTP** (Single Machine Tardiness): 单机延迟优化

---

## 更新日志

**v2.0 (2026-02-17)**:
- 整合所有 FFSP 文档
- 添加可视化美化说明
- 完善故障排查章节

**v1.0 (2026-02-13)**:
- 初始版本
- 基本快速开始指南

---

## 相关文档

- [项目主文档](../../README.md)
- [API文档](../api/API_PROTOCOL.md)
- [添加新问题类型指南](../ADD_NEW_PROBLEM_TYPE_GUIDE.md)

---

**如有问题，请查看项目 Issues 或联系开发团队。**
