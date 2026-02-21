# PCTSP (Prize Collecting TSP) 完整指南

> 奖励收集旅行商问题 — 在惩罚与收益之间寻找最优路径

**最后更新**: 2026-02-17  
**版本**: v1.0

---

## 📖 目录

1. [快速开始](#快速开始)
2. [问题介绍](#问题介绍)
3. [PCTSP vs OP 对比](#pctsp-vs-op-对比)
4. [参数配置](#参数配置)
5. [训练建议](#训练建议)
6. [可视化解读](#可视化解读)
7. [常见问题](#常见问题)
8. [故障排查](#故障排查)

---

## 快速开始

### 3步开始训练

**第 1 步**：在训练页面选择 **PCTSP - 奖励收集TSP**

**第 2 步**：配置参数（推荐入门配置）
```
客户节点数: 20
惩罚系数:   3.0
最低奖励要求: 1.0
```

**第 3 步**：选择训练算法
- 策略: **Attention Model**（自动选择，PCTSP 唯一支持的策略）
- 算法: **PPO**（推荐）
- 轮次: 10-30

---

## 问题介绍

### 核心概念

PCTSP（Prize Collecting TSP，奖励收集旅行商问题）在经典 TSP 基础上增加了两个维度：

| 概念 | 说明 |
|------|------|
| **奖励（Prize）** | 访问某节点获得的收益，各节点不同 |
| **惩罚（Penalty）** | **未访问**某节点所承担的代价，各节点不同 |
| **约束** | 收集的总奖励必须 ≥ `prize_required`（默认1.0）才能返回 depot |
| **目标** | 最大化：节省的惩罚 − 路径长度 |

### 完整奖励公式

```
Reward = Σ(已跳过节点的 penalty) − (路径总长度 + Σ(所有节点的 penalty))
       = Σ(已访问节点的 saved_penalty) − 路径长度
```

### 实际应用场景

1. **选择性配送**：并非所有客户必须服务，但放弃会有违约金
2. **采矿路径规划**：选择价值高的矿点，跳过低价值点
3. **广告投放路径**：选择高转化率地点，跳过低价值地点
4. **医疗巡诊**：优先高风险患者，安排低风险患者远程诊断

---

## PCTSP vs OP 对比

两者都是"选择性访问"问题，但有本质区别：

| 特征 | PCTSP | OP（定向问题） |
|------|-------|---------------|
| **约束类型** | 奖励约束（必须收集≥1.0） | 路径长度约束 |
| **跳过代价** | 有惩罚（penalty） | 无惩罚（仅错失奖励） |
| **返回条件** | 收集足够奖励后才能返回 | 路径用尽时返回 |
| **优化目标** | 最大化节省惩罚 − 路径长度 | 最大化收集的奖励 |
| **节点特征** | 每节点有 prize 和 penalty 两个属性 | 每节点只有 prize |
| **问题难度** | 较难（双目标权衡） | 较简单 |
| **推荐策略** | Attention Model（仅此） | Attention Model |

---

## 参数配置

### 参数详解

| 参数 | 含义 | 范围 | 默认值 |
|------|------|------|--------|
| **num_loc** | 客户节点数（不含depot） | 10-100 | 20 |
| **penalty_factor** | 惩罚缩放系数 | 1.0-5.0 | 3.0 |
| **prize_required** | 最低总奖励要求 | 0.5-2.0 | 1.0 |

### penalty_factor 的影响

```
penalty_factor 越大：
  → 跳过节点代价越高
  → 模型倾向于访问更多节点
  → 问题更类似于 TSP（全访问）

penalty_factor 越小：
  → 跳过节点代价越低
  → 模型更积极地跳过节点
  → 路径更短但节省更多惩罚
```

### 配置模板

#### 入门配置（快速验证）
```yaml
num_loc: 20
penalty_factor: 3.0
prize_required: 1.0
algorithm: reinforce
epochs: 10
```
⏱️ 预计时间: ~5分钟

#### 推荐配置（标准训练）
```yaml
num_loc: 50
penalty_factor: 3.0
prize_required: 1.0
algorithm: ppo
epochs: 30
batch_size: 512
```
⏱️ 预计时间: ~20分钟

#### 高质量配置
```yaml
num_loc: 100
penalty_factor: 3.0
prize_required: 1.0
algorithm: ppo
epochs: 50
batch_size: 512
embed_dim: 256
```
⏱️ 预计时间: ~60分钟

---

## 训练建议

### 为什么只能用 Attention Model？

PCTSP 中每个节点有 `prize` 和 `penalty` 两个独立属性，且各节点扮演不同角色（depot、必须访问、可选访问）。

- **POMO**：依赖旋转对称性（从任意节点出发效果相同），而 PCTSP 的 depot 是固定的特殊节点，奖励惩罚破坏了对称性
- **MatNet**：专为非对称距离矩阵设计，不支持 prize/penalty 类型的节点属性

**Attention Model** 具备通用的 encoder-decoder 结构，通过 `PCTSPInitEmbedding` 处理 prize 和 penalty 特征，是 PCTSP 的唯一合适策略。

### 算法选择

| 算法 | 收敛速度 | 稳定性 | 推荐场景 |
|------|----------|--------|----------|
| **PPO** | 快 | 高 | 生产环境，20+ 节点 |
| **A2C** | 较快 | 中 | 快速验证，资源有限 |
| **REINFORCE** | 慢 | 低 | 学习/调试目的 |

> PCTSP 问题有奖励/惩罚的权衡，**推荐 PPO**，比 REINFORCE 收敛快且稳定。

---

## 可视化解读

### 对比图说明

训练完成后生成路线对比图，解读方式：

```
节点视觉含义：
  ● 大节点    = 高奖励（值得访问）
  ● 小节点    = 低奖励（价值低）
  ■ 深边框    = 高惩罚（跳过代价大）
  □ 浅边框    = 低惩罚（可以考虑跳过）

颜色含义：
  🟠🔴 橙红彩色（实心）= 已访问节点
  ⚪ 灰色（半透明）   = 已跳过节点
  🟩 绿色方形        = Depot（出发/返回点）
  ➡️ 蓝色箭头        = 行驶路径
```

### 统计信息解读

图表标题中显示：
- **奖励（Reward）**: 模型获得的总奖励，越高越好（通常为正数）
- **收集奖励**: 已访问节点的奖励总和（必须 ≥ prize_required = 1.0）
- **节省惩罚**: 已访问节点的惩罚总和（"节省"是指主动访问从而避免被动承受惩罚）
- **路径长度**: 实际行驶距离
- **访问比例**: 访问了多少节点（如 `8/20`）

### 奖励值的含义

**奖励为正** = 好的解（节省的惩罚 > 路径成本）  
**奖励为负** = 差的解（路径成本 > 节省的惩罚，通常在训练初期出现）

---

## 常见问题

### Q1: 为什么不能选择 POMO 策略？

POMO 依赖"问题对称性"——对于 TSP，从任意节点出发、以任意节点结束，本质上是同一问题的不同旋转。但 PCTSP 中：
- Depot 是固定的特殊节点（出发和返回点）
- 每个节点有不同的 prize 和 penalty
- 节点角色不对称

因此前端会自动锁定为 Attention Model，POMO/MatNet 选项被禁用。

---

### Q2: prize_required 怎么设置？

`prize_required` 控制"最低必须收集的奖励"：

```
prize_required = 1.0  ← 推荐，约需访问 1/4 到 1/2 的节点
prize_required = 0.5  ← 宽松，只需少量访问
prize_required = 2.0  ← 严格，需要访问大多数节点（类似TSP）
```

> 技术细节：rl4co 中每个节点的 `deterministic_prize` 均匀采样自 [0, 4/n]，期望和为 2.0，所以 prize_required=1.0 约需访问一半节点。

---

### Q3: penalty_factor 和 prize_required 如何配合？

```
高惩罚 + 低奖励要求：访问更多高惩罚节点，路径较长
低惩罚 + 高奖励要求：访问更多节点（靠奖励驱动），路径较长
低惩罚 + 低奖励要求：灵活跳过，路径最短，但奖励较少

推荐平衡：penalty_factor=3.0 + prize_required=1.0
```

---

### Q4: 训练奖励为负数正常吗？

训练**初期**奖励为负是正常的（策略随机，路径很长但节省惩罚不足）。随着训练推进：
- 路径逐渐优化 → 路径长度下降
- 策略学会跳过低价值节点 → 节省更多惩罚
- 最终奖励趋向正值

如果训练30轮后奖励仍持续为负，可以：
1. 切换到 PPO 算法（比 REINFORCE 更稳定）
2. 减小问题规模（num_loc = 10）
3. 增大 penalty_factor（让节省惩罚更有价值）

---

### Q5: 如何判断解的质量？

**参考基准**（Attention Model + PPO，30轮）：

| 规模 | 平均奖励 | 访问比例 |
|------|----------|----------|
| 20 节点 | ~0.5 - 1.5 | 40%-60% |
| 50 节点 | ~1.0 - 3.0 | 35%-55% |
| 100 节点 | ~2.0 - 6.0 | 30%-50% |

> 注：具体值依赖随机种子和训练配置，以上为参考范围。

---

## 故障排查

### 错误：No module named 'rl4co'

```bash
# 确保使用正确的 Python 环境
source venv/bin/activate
pip install rl4co --upgrade
```

---

### 错误：PCTSPEnv not found

```bash
# 检查 rl4co 版本
python -c "import rl4co; print(rl4co.__version__)"
# 需要 >= 0.4.0
```

---

### 警告：约束未满足（Total prize < prize_required）

这是训练初期的正常现象，模型还未学会满足约束。若持续出现：

1. 降低 `prize_required`（如 0.5）
2. 检查环境是否正确初始化
3. 增加训练轮次

---

### 可视化为空或报错

**原因**：模型推理时动作序列格式不符合预期

**检查**：
```python
# 确认 actions 形状
print(actions.shape)  # 应为 [batch_size, seq_len]
print(actions.min(), actions.max())  # 应在 [0, num_loc] 范围内
```

---

## 技术细节

### 环境初始化

```python
from rl4co.envs.routing import PCTSPEnv

env = PCTSPEnv(generator_params={
    'num_loc': 20,
    'penalty_factor': 3.0,
    'prize_required': 1.0,
})

td = env.reset(batch_size=[4])
# td 包含的键：
# locs         [4, 21, 2]   depot + 20 客户节点坐标
# real_prize   [4, 21]      各节点奖励（depot=0）
# penalty      [4, 21]      各节点惩罚（depot=0）
# prize_required [4]        最低奖励要求（1.0）
```

### 兼容的策略

```python
from rl4co.models import AttentionModelPolicy

policy = AttentionModelPolicy(
    env_name='pctsp',
    embed_dim=128,
    num_encoder_layers=3,
    num_heads=8,
)
```

---

## 更新日志

**v1.0 (2026-02-17)**:
- 初始版本，PCTSP 集成完成
- 支持 num_loc、penalty_factor、prize_required 参数
- 可视化：对比图 + 动画

---

## 相关文档

- [项目主文档](../../README.md)
- [OP 完整指南](./FFSP_COMPLETE_GUIDE.md)（类似的奖励收集问题）
- [添加新问题类型指南](../../modules/ADD_NEW_PROBLEM_GUIDE.md)
- [兼容性矩阵](../../modules/COMPATIBILITY_MATRIX.md)

---

**如有问题，请查看项目 Issues 或联系开发团队。**
