# mTSP (Multiple Traveling Salesman Problem) 使用指南

## 📝 问题简介

**mTSP（多旅行商问题）**是TSP的多代理推广版本。在这个问题中，有多个旅行商（代理/车辆）需要协同工作，访问所有城市并返回起点（depot）。

### 问题特点

- ✅ **多代理协同**：多个旅行商同时工作
- ✅ **共享起点**：所有代理从同一depot出发和返回
- ✅ **无重复访问**：每个城市只被一个代理访问一次
- ✅ **灵活优化**：支持两种优化目标（minmax或sum）
- ✅ **TSP推广**：是经典TSP的自然扩展

### 应用场景

1. **多车辆配送**：快递公司的配送车辆调度
2. **无人机巡检**：多架无人机协同完成区域巡检任务
3. **机器人任务分配**：仓库中多个机器人拣货路径规划
4. **团队任务调度**：多个服务团队的工作路线安排

---

## 🎯 优化目标

mTSP支持两种优化目标：

### 1. Minmax（最小化最大路径）

**目标**：最小化所有代理中最长的路径长度

```python
cost_type = 'minmax'
cost = max(path_length_agent_1, path_length_agent_2, ..., path_length_agent_n)
```

**适用场景**：
- 需要平衡各代理工作负载
- 关注完成任务的总时间（由最慢的代理决定）
- 团队协作场景

**优点**：
- 工作负载相对均衡
- 总体完成时间最短
- 避免单个代理过载

### 2. Sum（最小化总路径）

**目标**：最小化所有代理路径长度的总和

```python
cost_type = 'sum'
cost = path_length_agent_1 + path_length_agent_2 + ... + path_length_agent_n
```

**适用场景**：
- 关注总体成本（如燃油消耗）
- 代理能力无差异
- 追求整体效率

**优点**：
- 总体成本最低
- 资源利用最优
- 适合成本敏感场景

---

## 🔧 配置参数

### 基本参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `problem_type` | str | 'mtsp' | 问题类型标识 |
| `num_loc` | int | 50 | 城市数量（不含depot） |
| `num_agents` | int | 5 | 代理（旅行商）数量 |
| `cost_type` | str | 'minmax' | 优化目标类型 |
| `batch_size` | int | 512 | 训练批次大小 |
| `epochs` | int | 100 | 训练轮数 |

### 配置示例

#### 示例 1：均衡负载的多车辆配送

```python
config = {
    'problem_type': 'mtsp',
    'num_loc': 100,           # 100个配送点
    'num_agents': 5,          # 5辆配送车
    'cost_type': 'minmax',    # 均衡各车辆工作量
    'batch_size': 256,
    'epochs': 200,
}
```

#### 示例 2：最小化总成本的无人机巡检

```python
config = {
    'problem_type': 'mtsp',
    'num_loc': 50,            # 50个巡检点
    'num_agents': 3,          # 3架无人机
    'cost_type': 'sum',       # 最小化总飞行距离
    'batch_size': 512,
    'epochs': 150,
}
```

#### 示例 3：小规模任务分配

```python
config = {
    'problem_type': 'mtsp',
    'num_loc': 20,            # 20个任务点
    'num_agents': 2,          # 2个代理
    'cost_type': 'minmax',    # 尽快完成所有任务
    'batch_size': 1024,
    'epochs': 100,
}
```

---

## 📊 与其他问题的对比

### mTSP vs TSP

| 特性 | TSP | mTSP |
|------|-----|------|
| 代理数量 | 1 | 多个（≥2） |
| 路径结构 | 单条闭合路径 | 多条闭合路径 |
| 优化目标 | 单一路径长度 | minmax或sum |
| 复杂度 | NP-hard | NP-hard |
| 应用场景 | 单车辆/单人任务 | 多车辆/团队协作 |

### mTSP vs CVRP

| 特性 | mTSP | CVRP |
|------|------|------|
| 容量约束 | ❌ 无 | ✅ 有 |
| 代理数量 | 固定 | 可变（取决于需求） |
| 返回depot | 每个代理一次 | 可能多次 |
| 优先考虑 | 路径均衡或总长 | 容量满足 |
| 复杂度 | NP-hard | NP-hard |

---

## 🚀 使用流程

### 1. 导入和配置

```python
from modules.problems import get_problem_class

# 创建mTSP问题实例
config = {
    'problem_type': 'mtsp',
    'num_loc': 50,
    'num_agents': 5,
    'cost_type': 'minmax',
    'batch_size': 512,
}

problem_class = get_problem_class('mtsp')
problem = problem_class(config)
```

### 2. 创建环境

```python
# 创建RL4CO环境
env = problem.create_environment()

# 查看环境信息
print(f"问题类型: {problem.get_problem_name()}")
print(f"城市数量: {problem.num_loc}")
print(f"代理数量: {problem.num_agents}")
print(f"优化目标: {problem.cost_type}")
```

### 3. 训练模型

```python
from rl4co.models import AttentionModel
from rl4co.utils import RL4COTrainer

# 创建策略模型
policy = AttentionModel(env)

# 配置训练器
trainer = RL4COTrainer(
    max_epochs=100,
    accelerator='gpu',
    precision='16-mixed'
)

# 开始训练
trainer.fit(policy)
```

### 4. 可视化结果

```python
# 获取可视化函数
viz_funcs = problem.get_visualization_functions()

# 生成路线动画
viz_funcs['animation'](
    td=test_data,
    actions=solution,
    save_path='mtsp_animation.gif',
    title='mTSP路线生成过程'
)

# 生成对比图
viz_funcs['comparison'](
    td=test_data,
    actions=solution,
    save_path='mtsp_comparison.png',
    cost=final_cost,
    title='mTSP最终路线对比'
)
```

---

## 📈 性能建议

### 参数调优建议

#### 代理数量 (num_agents)

```python
# 推荐范围
if num_loc <= 20:
    num_agents = 2-3      # 小规模问题
elif num_loc <= 50:
    num_agents = 3-5      # 中等规模
elif num_loc <= 100:
    num_agents = 5-10     # 大规模
else:
    num_agents = 10-20    # 超大规模
```

**经验法则**：
- 代理数不宜过多（效率递减）
- 建议每个代理负责 5-20 个城市
- 根据实际场景灵活调整

#### 批次大小 (batch_size)

```python
# 根据问题规模调整
small_scale:  batch_size = 1024  # num_loc <= 20
medium_scale: batch_size = 512   # 20 < num_loc <= 100
large_scale:  batch_size = 256   # num_loc > 100
```

#### 训练轮数 (epochs)

```python
# minmax目标通常需要更多训练
if cost_type == 'minmax':
    epochs = 150-200
else:  # sum
    epochs = 100-150
```

### 优化目标选择

**选择 minmax 的场景**：
- ✅ 需要均衡工作负载
- ✅ 关注最大完成时间
- ✅ 团队协作任务
- ✅ 避免单点瓶颈

**选择 sum 的场景**：
- ✅ 关注总体成本
- ✅ 资源消耗敏感
- ✅ 代理能力一致
- ✅ 追求整体效率

---

## ⚠️ 注意事项

### 1. 参数约束

```python
# 必须满足的约束
assert num_agents >= 1, "至少需要1个代理"
assert num_agents <= num_loc, "代理数不能超过城市数"
assert cost_type in ['minmax', 'sum'], "cost_type必须是'minmax'或'sum'"
```

### 2. 动作序列格式

mTSP的动作序列使用 0（depot）来分隔不同代理的路径：

```python
# 示例：3个代理访问6个城市
actions = [1, 2, 0, 3, 4, 0, 5, 6, 0]
#          ^^^^^ agent1  ^^^^^ agent2  ^^^^^ agent3
```

每个代理的子路径：
- Agent 1: depot → 1 → 2 → depot
- Agent 2: depot → 3 → 4 → depot
- Agent 3: depot → 5 → 6 → depot

### 3. 可视化限制

- 支持的代理颜色数：8种
- 超过8个代理时，颜色会循环使用
- 建议代理数不超过10个以保证可视化清晰度

### 4. 计算资源

```python
# GPU显存估算（approximate）
memory_GB = (num_loc * num_agents * batch_size * 4) / (1024**3)

# 建议配置
if memory_GB > 8:
    # 减小batch_size或使用梯度累积
    batch_size = batch_size // 2
```

---

## 🔍 故障排除

### 问题1：训练不收敛

**可能原因**：
- 代理数量过多
- 学习率不合适
- 训练轮数不足

**解决方案**：
```python
# 1. 减少代理数量
config['num_agents'] = max(2, config['num_agents'] // 2)

# 2. 调整学习率
config['optimizer_kwargs'] = {'lr': 1e-4}  # 降低学习率

# 3. 增加训练轮数
config['epochs'] = 200
```

### 问题2：GPU内存不足

**解决方案**：
```python
# 1. 减小批次大小
config['batch_size'] = 256

# 2. 使用梯度累积
config['accumulate_grad_batches'] = 4

# 3. 使用混合精度训练
trainer = RL4COTrainer(precision='16-mixed')
```

### 问题3：可视化错误

**常见问题**：
- 动作序列格式不正确
- 缺少depot分隔符（0）

**检查方法**：
```python
# 验证动作序列
def validate_actions(actions, num_loc):
    # 检查是否包含depot
    assert 0 in actions, "动作序列必须包含depot(0)"
    
    # 检查城市范围
    non_depot = [a for a in actions if a != 0]
    assert all(1 <= a <= num_loc for a in non_depot), \
        f"城市编号必须在1-{num_loc}之间"
    
    # 检查是否有重复（除depot外）
    assert len(non_depot) == len(set(non_depot)), \
        "存在重复访问的城市"
```

---

## 📚 相关资源

### RL4CO官方文档
- [mTSP Environment](https://rl4.co/docs/content/api/envs/routing/#multiple-traveling-salesman-problem-mtsp)
- [RL4CO GitHub](https://github.com/ai4co/rl4co)

### 学术论文
- Bektas, T. (2006). "The multiple traveling salesman problem: an overview"
- Kara, I., & Bektas, T. (2006). "Integer linear programming formulations of multiple salesman problems"

### 项目内相关文档
- [问题类型README](README.md) - 所有问题类型概览
- [算法和策略指南](../ALGORITHM_AND_POLICY_GUIDE.md) - 算法选择建议
- [模块架构说明](../README.md) - 整体架构

---

## 🎓 最佳实践

1. **从小规模开始**：先用小数据（20城市，2-3代理）验证配置
2. **渐进式训练**：逐步增加问题规模和复杂度
3. **监控指标**：关注收敛曲线和代理负载均衡
4. **对比实验**：测试不同cost_type和num_agents的效果
5. **保存检查点**：定期保存训练进度，便于恢复和对比

---

**更新时间**：2026-02-04  
**版本**：v1.0  
**维护者**：RL4CO Display Team
