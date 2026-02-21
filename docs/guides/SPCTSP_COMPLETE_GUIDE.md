# SPCTSP 完整使用指南

**随机奖励收集旅行商问题 (Stochastic Prize Collecting TSP)**

---

## 目录

1. [问题概述](#问题概述)
2. [SPCTSP vs PCTSP 对比](#spctsp-vs-pctsp-对比)
3. [随机奖励机制详解](#随机奖励机制详解)
4. [参数说明](#参数说明)
5. [推荐配置](#推荐配置)
6. [可视化解读](#可视化解读)
7. [随机奖励对训练的影响](#随机奖励对训练的影响)
8. [策略与算法限制](#策略与算法限制)
9. [常见问题](#常见问题)

---

## 问题概述

SPCTSP（Stochastic Prize Collecting TSP）是 PCTSP 的随机版本，由 RL4CO 官方实现
（`rl4co.envs.routing.SPCTSPEnv`）。

**问题描述**：
- 给定一组节点，每个节点有**期望奖励**（`deterministic_prize`，决策前可见）和**惩罚**（`penalty`）
- 到达节点后才揭晓该节点的**真实奖励**（`real_prize`），服从随机分布
- 必须收集至少 `prize_required` 的**真实**总奖励才能返回 depot
- 未访问节点会产生惩罚
- 目标：最大化（节省的惩罚 − 路径长度）

**关键差异（相比 PCTSP）**：
> PCTSP 中 `real_prize = deterministic_prize`（完全信息）；
> SPCTSP 中 `real_prize ~ Uniform(0, 2 × deterministic_prize)`（部分信息，访问后才知）

---

## SPCTSP vs PCTSP 对比

| 特性 | PCTSP | SPCTSP |
|------|-------|--------|
| **奖励信息** | 完全已知（确定性） | 仅知期望值，真实值需访问后才知 |
| **`real_prize`** | = `deterministic_prize` | ~ Uniform(0, 2×deterministic_prize) |
| **`deterministic_prize`** | 等于真实奖励 | 期望奖励（规划依据） |
| **决策难度** | 较低（完全信息） | 较高（部分信息，需鲁棒策略） |
| **训练稳定性** | 较稳定 | 需要 PPO 等方差控制算法 |
| **现实对应** | 收益固定的路径规划 | 收益不确定的推销/服务场景 |
| **可视化** | 路线图 + 奖励标注 | 路线图 + 期望 vs 真实奖励对比图 |
| **环境类** | `PCTSPEnv` | `SPCTSPEnv`（继承自 PCTSPEnv） |
| **支持策略** | Attention Model | Attention Model（POMO/MatNet 不适用）|
| **推荐算法** | PPO / A2C | **PPO**（随机性需要方差控制）|

---

## 随机奖励机制详解

### 奖励生成原理

```python
# SPCTSPEnv 内部（rl4co 源码）
deterministic_prize = Uniform(0, 1)          # 期望奖励，规划前可见
real_prize = Uniform(0, 2 * deterministic_prize)  # 真实奖励，访问后才知

# 因此：E[real_prize] = deterministic_prize（期望相同，但有随机波动）
```

### 信息状态对比

| 时间点 | PCTSP | SPCTSP |
|--------|-------|--------|
| 规划前 | 所有奖励已知 | 只知 `deterministic_prize`（期望值） |
| 访问节点后 | 无变化 | 获得 `real_prize`（真实值，可能高/低于期望）|
| 约束检查 | 基于真实奖励（= 期望） | 基于访问后累计的 `real_prize`（有随机性）|

### 对策略的影响

**好运情况**（`real_prize > deterministic_prize`）：
- 早期节点奖励超出预期，可能提前满足 `prize_required` 约束
- 可以减少访问节点数，节省路径长度

**坏运情况**（`real_prize < deterministic_prize`）：
- 实际奖励低于预期，可能需要访问更多节点才能满足约束
- 策略必须具备鲁棒性，不能过于激进

---

## 参数说明

与 PCTSP 使用完全相同的参数集：

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `num_loc` | int | 20 | 10–100 | 客户节点数量（不含 depot） |
| `penalty_factor` | float | 3.0 | 1.0–5.0 | 惩罚缩放系数，值越大未访问代价越高 |
| `prize_required` | float | 1.0 | 0.5–2.0 | 必须收集的最低真实奖励总量 |

### 参数选择建议

**`num_loc`**：
- 10–20：适合快速验证，训练时间短
- 20–50：标准规模，效果与速度均衡
- 50+：复杂场景，需要更长训练时间

**`penalty_factor`**：
- 较小（1.0–2.0）：惩罚轻，策略倾向于跳过更多节点
- 默认（3.0）：平衡的探索-利用权衡
- 较大（4.0–5.0）：惩罚重，策略倾向于访问更多节点

**`prize_required`**：
- 随机性使满足此约束比 PCTSP 更不稳定
- 建议保持默认值 1.0，避免设置过高（> 1.5）导致约束过难满足

---

## 推荐配置

### 最佳质量配置

```yaml
problem: spctsp
num_loc: 20
penalty_factor: 3.0
prize_required: 1.0
model: attention         # 唯一支持的策略
algorithm: ppo           # 推荐！PPO方差控制更适合随机奖励
epochs: 100
batch_size: 512
learning_rate: 0.0001
```

### 快速验证配置

```yaml
problem: spctsp
num_loc: 15
penalty_factor: 3.0
prize_required: 1.0
model: attention
algorithm: a2c           # 快速收敛
epochs: 50
batch_size: 256
learning_rate: 0.0002
```

### 入门学习配置

```yaml
problem: spctsp
num_loc: 10
penalty_factor: 2.0
prize_required: 0.8      # 稍低的约束，更易学习
model: attention
algorithm: reinforce
epochs: 30
batch_size: 128
learning_rate: 0.0001
```

---

## 可视化解读

SPCTSP 训练完成后会生成两类可视化：

### 1. 路线对比图（PNG）

**左图：路线图**
- **节点大小** ∝ `deterministic_prize`（期望奖励，决策依据）
- **节点颜色** ∝ `real_prize`（真实奖励，访问后揭晓；越红越高）
- **节点注释**：`E: 期望值 / R: 真实值`
- **节点边框** ∝ `penalty`（惩罚值，越深越粗代价越高）
- **灰色节点**：未访问（承担惩罚）
- **彩色节点**：已访问（获得真实奖励）

**右图：随机奖励对比柱形图**
- 蓝色柱：`deterministic_prize`（期望奖励）
- 橙色柱：`real_prize`（真实奖励）
- 差值标注：绿色 = 好运（real > det），红色 = 坏运（real < det）
- 直观展示随机波动程度

### 2. 路线动画（GIF）

- 逐步展示访问过程
- 标题实时显示：**期望累计奖励 vs 真实累计奖励**
- 可以观察到随机奖励揭晓的动态过程

---

## 随机奖励对训练的影响

### 为什么 PPO 比 REINFORCE 更适合 SPCTSP？

**REINFORCE 的问题**：
- 奖励的随机性导致策略梯度方差极大
- 需要更多训练样本才能收敛
- 容易训练不稳定

**PPO 的优势**：
- 裁剪目标函数限制策略更新步长，防止在高方差信号下过拟合
- 更稳定的训练曲线
- 对随机奖励的噪声具有天然鲁棒性

### 训练曲线特征

SPCTSP 的训练曲线通常比 PCTSP 更**波动**：

```
PCTSP 曲线（平滑）：  ─────────/──────────
SPCTSP 曲线（波动）：  ────/\/\/─/─/\───
```

这是**正常现象**，因为每个 batch 中真实奖励的随机性引入了额外噪声。
使用 PPO + 较大 batch_size（512+）可以有效抑制波动。

---

## 策略与算法限制

### 为什么 POMO 不适用？

POMO（Policy Optimization with Multiple Optima）通过从多个起点构建路径来利用问题的**旋转对称性**。

SPCTSP 破坏对称性的原因：
1. 每个节点有不同的奖励和惩罚（非对称）
2. 随机奖励的不确定性使多起点策略失效
3. `prize_required` 约束是全局性的，不满足旋转对称性

**结论**：SPCTSP 只支持 **Attention Model**。

### 算法优先级

```
PPO > A2C > REINFORCE
（稳定性）  （速度）   （基础）
```

---

## 常见问题

### Q1: SPCTSP 和 PCTSP 训练结果一样吗？

不一样。SPCTSP 的奖励（训练目标）基于**真实奖励**计算，受随机性影响，通常比 PCTSP 的奖励波动更大、平均值略低。

### Q2: 为什么 SPCTSP 的奖励看起来比 PCTSP 低？

这是随机性的正常结果：
- 坏运情况下（真实奖励 < 期望），策略无法获得预期收益
- 统计上，长期平均奖励应与 PCTSP 相近

### Q3: `prize_required` 约束还是基于真实奖励吗？

是的。SPCTSP 中约束仍然基于**真实**奖励（`real_prize`）。
由于真实奖励的随机性，在某些情况下策略可能需要访问比 PCTSP 更多的节点。

### Q4: 能直接加载 PCTSP 的检查点用于 SPCTSP 吗？

不建议。两者使用不同的环境（`PCTSPEnv` vs `SPCTSPEnv`），虽然模型架构相同，但环境状态空间有差异（`deterministic_prize` 字段）。

### Q5: 可视化中的「期望 vs 真实对比图」如何解读？

- 蓝橙柱高度差越大 → 该 episode 随机性越强
- 所有节点的蓝橙柱基本相同 → 随机性较小（偶然接近确定性）
- 红色差值标注（real < det）→ 坏运节点
- 绿色差值标注（real > det）→ 好运节点

---

## 相关文件

| 文件 | 说明 |
|------|------|
| `modules/rl_training/spctsp_trainer.py` | SPCTSP 训练器（继承 PCTSPTrainer）|
| `modules/rl_training/visualizations/spctsp_viz.py` | SPCTSP 可视化函数 |
| `modules/rl_training/pctsp_trainer.py` | PCTSP 训练器（基类）|
| `modules/compatibility.py` | 兼容性配置（含 spctsp 规则）|
| `docs/guides/PCTSP_COMPLETE_GUIDE.md` | PCTSP 使用指南（对照参考）|
| `rl4co-main/rl4co/envs/routing/spctsp/env.py` | RL4CO 官方 SPCTSP 环境 |
