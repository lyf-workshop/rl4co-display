# 问题类型兼容性说明

## 📊 策略和算法兼容性矩阵

本文档说明每个问题类型支持哪些策略模型和强化学习算法。

---

## 🎯 问题类型总览

| 问题类型 | 中文名称 | 策略支持 | 算法支持 | 特殊说明 |
|---------|---------|---------|---------|---------|
| **TSP** | 旅行商问题 | ✅ Attention<br>✅ POMO | ✅ REINFORCE<br>✅ PPO<br>✅ A2C | 对称问题，全支持 |
| **ATSP** | 非对称TSP | ✅ Attention<br>❌ POMO | ✅ REINFORCE<br>✅ PPO<br>✅ A2C | POMO不支持非对称 |
| **mTSP** | 多旅行商问题 | ✅ Attention<br>✅ POMO | ✅ REINFORCE<br>✅ PPO<br>✅ A2C | 对称问题，全支持 |
| **CVRP** | 车辆路径问题 | ✅ Attention<br>✅ POMO | ✅ REINFORCE<br>✅ PPO<br>✅ A2C | 对称问题，全支持 |
| **SDVRP** | 分割配送VRP | ✅ Attention<br>⚠️ POMO | ✅ REINFORCE<br>✅ PPO<br>✅ A2C | POMO效果未验证 |
| **VRPTW** | 带时间窗VRP | ✅ Attention<br>⚠️ POMO | ✅ REINFORCE<br>✅ PPO<br>✅ A2C | POMO效果未验证 |

**图例**：
- ✅ 完全支持
- ⚠️ 技术上可行但效果未验证
- ❌ 不支持

---

## 📋 详细兼容性说明

### 1. TSP - 旅行商问题

**问题特点**：对称距离，单代理，无约束

#### 支持的策略
| 策略 | 支持 | 说明 |
|-----|------|-----|
| Attention Model | ✅ 完全支持 | 经典方法，速度快 |
| POMO | ✅ 完全支持 | 高质量解，利用对称性 |

#### 支持的算法
| 算法 | 支持 | 说明 |
|-----|------|-----|
| REINFORCE | ✅ 完全支持 | 简单易懂，适合入门 |
| PPO | ✅ 完全支持 | 稳定性好，推荐生产使用 |
| A2C | ✅ 完全支持 | 收敛快，性能介于两者之间 |

#### 推荐组合
- **最佳质量**：POMO + PPO
- **快速原型**：Attention + PPO
- **学习入门**：Attention + REINFORCE

---

### 2. ATSP - 非对称旅行商问题

**问题特点**：非对称距离（A→B ≠ B→A），单代理

#### 支持的策略
| 策略 | 支持 | 说明 |
|-----|------|-----|
| Attention Model | ✅ 完全支持 | 可处理非对称距离 |
| POMO | ❌ 不支持 | POMO依赖对称性假设 |

#### 支持的算法
| 算法 | 支持 | 说明 |
|-----|------|-----|
| REINFORCE | ✅ 完全支持 | 但收敛较慢 |
| PPO | ✅ 完全支持 | 推荐使用 |
| A2C | ✅ 完全支持 | 推荐使用 |

#### 推荐组合
- **最佳质量**：Attention + PPO
- **快速收敛**：Attention + A2C
- **生产使用**：Attention + PPO

⚠️ **重要提示**：ATSP 不建议使用 REINFORCE（收敛困难）

---

### 3. mTSP - 多旅行商问题 ⭐ 新增

**问题特点**：对称距离，多代理，共享起点

#### 支持的策略
| 策略 | 支持 | 说明 |
|-----|------|-----|
| Attention Model | ✅ 完全支持 | 通用性好，稳定可靠 |
| POMO | ✅ 完全支持 | 利用对称性，高质量解 |

#### 支持的算法
| 算法 | 支持 | 说明 |
|-----|------|-----|
| REINFORCE | ✅ 完全支持 | 适合入门和小规模问题 |
| PPO | ✅ 完全支持 | 推荐生产使用 |
| A2C | ✅ 完全支持 | 收敛快，性能好 |

#### 推荐组合
- **最佳质量**：POMO + PPO
- **快速原型**：Attention + PPO
- **学习入门**：Attention + REINFORCE

#### mTSP 特有参数
```python
config = {
    'problem': 'mtsp',
    'model': 'pomo',
    'algorithm': 'ppo',
    'num_loc': 50,          # 城市数量
    'num_agents': 5,        # 代理数量（2-10）
    'cost_type': 'minmax',  # 优化目标：'minmax' 或 'sum'
}
```

**优化目标说明**：
- **minmax**：最小化最长路径（均衡各代理负载）
- **sum**：最小化总路径长度（追求整体效率）

---

### 4. CVRP - 车辆路径问题

**问题特点**：对称距离，容量约束，多次返回仓库

#### 支持的策略
| 策略 | 支持 | 说明 |
|-----|------|-----|
| Attention Model | ✅ 完全支持 | 经典方法 |
| POMO | ✅ 完全支持 | 高质量解 |

#### 支持的算法
| 算法 | 支持 | 说明 |
|-----|------|-----|
| REINFORCE | ✅ 完全支持 | 适合小规模 |
| PPO | ✅ 完全支持 | 推荐 |
| A2C | ✅ 完全支持 | 性能好 |

#### 推荐组合
- **最佳质量**：POMO + PPO
- **快速原型**：Attention + PPO
- **学习入门**：Attention + REINFORCE

---

### 5. SDVRP - 分割配送车辆路径问题

**问题特点**：对称距离，容量约束，允许分割配送

#### 支持的策略
| 策略 | 支持 | 说明 |
|-----|------|-----|
| Attention Model | ✅ 完全支持 | 推荐使用 |
| POMO | ⚠️ 未验证 | 效果未经充分测试 |

#### 支持的算法
| 算法 | 支持 | 说明 |
|-----|------|-----|
| REINFORCE | ✅ 完全支持 | 适合小规模 |
| PPO | ✅ 完全支持 | 推荐 |
| A2C | ✅ 完全支持 | 推荐 |

#### 推荐组合
- **最佳质量**：Attention + PPO
- **快速收敛**：Attention + A2C
- **学习入门**：Attention + REINFORCE

⚠️ **注意**：POMO 在 SDVRP 上效果未经验证，建议使用 Attention Model

---

### 6. VRPTW - 带时间窗车辆路径问题

**问题特点**：对称距离，容量约束，时间窗约束

#### 支持的策略
| 策略 | 支持 | 说明 |
|-----|------|-----|
| Attention Model | ✅ 完全支持 | 推荐使用 |
| POMO | ⚠️ 未验证 | 效果未经测试 |

#### 支持的算法
| 算法 | 支持 | 说明 |
|-----|------|-----|
| REINFORCE | ✅ 支持 | 但不推荐（收敛困难） |
| PPO | ✅ 完全支持 | 强烈推荐 |
| A2C | ✅ 完全支持 | 推荐 |

#### 推荐组合
- **最佳质量**：Attention + PPO
- **快速收敛**：Attention + A2C
- **生产使用**：Attention + PPO

⚠️ **注意**：
1. VRPTW 复杂度高，不建议使用 REINFORCE
2. POMO 效果未验证，建议使用 Attention Model

---

## 🎯 快速选择指南

### 根据问题类型选择

**对称问题**（TSP、mTSP、CVRP）：
- ✅ Attention Model + POMO 都支持
- 推荐：POMO + PPO（最佳质量）

**非对称问题**（ATSP）：
- ✅ 只能用 Attention Model
- 推荐：Attention + PPO

**复杂约束问题**（SDVRP、VRPTW）：
- ✅ 优先使用 Attention Model
- 推荐：Attention + PPO

### 根据使用场景选择

**学习入门**：
- 策略：Attention Model
- 算法：REINFORCE
- 原因：简单易懂，适合理解原理

**快速原型**：
- 策略：Attention Model
- 算法：PPO
- 原因：速度快，效果好

**生产使用**：
- 策略：POMO（对称问题）或 Attention（复杂问题）
- 算法：PPO
- 原因：稳定性好，质量高

---

## 📖 相关文档

- [兼容性矩阵详细版](./COMPATIBILITY_MATRIX.md) - 包含警告信息和特殊约束
- [算法和策略指南](./ALGORITHM_AND_POLICY_GUIDE.md) - 算法和策略的详细说明
- [mTSP 使用指南](./problems/MTSP_GUIDE.md) - mTSP 专用文档

---

## ❓ 常见问题

### Q1: 为什么 POMO 不支持 ATSP？
A: POMO 利用问题的对称性（从任意节点出发解的质量相同）。ATSP 的距离矩阵非对称，不满足这个假设。

### Q2: mTSP 和 TSP 在兼容性上有什么区别？
A: 没有区别！mTSP 是对称问题，支持所有 TSP 支持的策略和算法。唯一的区别是 mTSP 需要额外配置 `num_agents` 和 `cost_type` 参数。

### Q3: 为什么某些问题不推荐使用 REINFORCE？
A: REINFORCE 是最基础的策略梯度算法，在复杂问题（ATSP、VRPTW）上收敛困难、方差大。PPO 和 A2C 通过各种技巧改进了这些问题。

### Q4: 如何验证我的配置组合是否有效？
A: 系统会自动验证！前端选择问题类型后，会自动过滤掉不兼容的策略和算法。如果某个组合不可用，说明存在兼容性问题。

---

**最后更新**：2026-02-04  
**文档状态**：包含 mTSP 支持
