# RL4CO 组件兼容性矩阵

## 🎯 **兼容性规则说明**

强化学习训练涉及三个关键组件：
1. **问题类型** (Problem): TSP, CVRP, SDVRP...
2. **策略模型** (Policy): Attention Model, POMO...
3. **RL算法** (Algorithm): REINFORCE, PPO, A2C...

并非所有组合都有效，本文档定义了兼容性规则。

---

## 📊 **1. 策略 → 问题兼容性**

### Attention Model (AM)
```yaml
适用问题: 
  - TSP ✅ (经典应用)
  - CVRP ✅ (经典应用)
  - SDVRP ✅ (支持)
  - PCTSP ✅ (支持)
  - OP ✅ (支持)

特点:
  - 通用性强，几乎适用所有路径规划问题
  - 单次解码，速度快
  - 质量中等

推荐场景:
  - 快速原型
  - 实时推理
  - 初学者入门
```

### POMO
```yaml
适用问题:
  - TSP ✅ (最佳选择)
  - CVRP ✅ (优秀效果)
  - SDVRP ⚠️ (理论支持，效果待验证)
  - PCTSP ❌ (不适用 - 不是对称问题)
  - OP ❌ (不适用 - 不是对称问题)

特点:
  - 利用问题对称性
  - 多起点同时构建
  - 质量高，但计算量大

限制:
  - 仅适用于对称问题（所有节点都可以作为起点）
  - 需要更多显存

推荐场景:
  - 质量优先
  - TSP/CVRP 大规模问题
  - 有GPU支持
```

### SymNCO (未实现)
```yaml
适用问题:
  - TSP ✅
  - CVRP ✅
  - 其他对称问题 ✅

特点:
  - 显式利用对称性
  - 数据增强技术
```

---

## 📊 **2. 算法 → 问题兼容性**

### REINFORCE
```yaml
适用问题:
  - TSP ✅
  - CVRP ✅
  - SDVRP ✅
  - PCTSP ✅
  - OP ✅
  - 所有问题 ✅

特点:
  - 最基础的策略梯度算法
  - 通用性强，适用所有问题
  - 训练不稳定，方差大

推荐场景:
  - 教学演示
  - 算法原型
  - 简单问题
```

### PPO
```yaml
适用问题:
  - TSP ✅
  - CVRP ✅
  - SDVRP ✅
  - PCTSP ✅
  - OP ✅
  - 所有问题 ✅

特点:
  - 工业界首选算法
  - 训练稳定，超参数不敏感
  - 收敛快，性能好

推荐场景:
  - 生产环境
  - 大规模问题
  - 需要稳定训练

限制:
  - 需要RL4CO版本支持
```

### A2C
```yaml
适用问题:
  - TSP ✅
  - CVRP ✅
  - SDVRP ✅
  - PCTSP ✅
  - OP ✅
  - 所有问题 ✅

特点:
  - Actor-Critic架构
  - 方差小于REINFORCE
  - 性能介于REINFORCE和PPO之间

推荐场景:
  - 中等规模问题
  - 需要快速收敛

限制:
  - 需要RL4CO版本支持
```

---

## 📊 **3. 策略 → 算法兼容性**

### Attention Model + 算法
```yaml
REINFORCE: ✅ 完美兼容
PPO:       ✅ 兼容 (需要Critic网络)
A2C:       ✅ 兼容 (需要Critic网络)
DQN:       ❌ 不兼容 (策略网络 vs 价值网络)
```

### POMO + 算法
```yaml
REINFORCE: ✅ 完美兼容 (原论文使用)
PPO:       ✅ 理论兼容 (需要适配多起点)
A2C:       ✅ 理论兼容 (需要适配多起点)
DQN:       ❌ 不兼容
```

---

## 🎯 **完整兼容性矩阵**

### TSP (旅行商问题)

| 策略 \ 算法 | REINFORCE | PPO | A2C |
|------------|-----------|-----|-----|
| **Attention Model** | ✅ 推荐 | ✅ 最佳 | ✅ 良好 |
| **POMO** | ✅ 高质量 | ✅ 最佳组合 | ✅ 良好 |

**推荐组合**:
1. 🥇 POMO + PPO (最佳质量+稳定性)
2. 🥈 Attention Model + PPO (平衡选择)
3. 🥉 POMO + REINFORCE (高质量，训练较慢)

---

### CVRP (车辆路径问题)

| 策略 \ 算法 | REINFORCE | PPO | A2C |
|------------|-----------|-----|-----|
| **Attention Model** | ✅ 推荐 | ✅ 最佳 | ✅ 良好 |
| **POMO** | ✅ 高质量 | ✅ 最佳组合 | ✅ 良好 |

**推荐组合**:
1. 🥇 POMO + PPO (最佳质量+稳定性)
2. 🥈 Attention Model + PPO (平衡选择)
3. 🥉 Attention Model + REINFORCE (快速原型)

---

### SDVRP (分割配送VRP)

| 策略 \ 算法 | REINFORCE | PPO | A2C |
|------------|-----------|-----|-----|
| **Attention Model** | ✅ 推荐 | ✅ 良好 | ✅ 良好 |
| **POMO** | ⚠️ 实验性 | ⚠️ 实验性 | ⚠️ 实验性 |

**推荐组合**:
1. 🥇 Attention Model + PPO (推荐)
2. 🥈 Attention Model + A2C (良好)
3. 🥉 Attention Model + REINFORCE (基准)

**注意**: POMO在SDVRP上的效果未经充分验证

---

## 🚫 **不兼容组合**

### 策略限制
```yaml
❌ POMO + PCTSP: 
   原因: PCTSP不是对称问题，不适合多起点

❌ POMO + OP:
   原因: OP不是对称问题，不适合多起点
```

### 算法限制
```yaml
❌ 任何策略 + DQN:
   原因: DQN是价值网络方法，与构造式策略不兼容
   
❌ 任何策略 + SAC:
   原因: SAC设计用于连续动作空间，不适合组合优化
```

---

## 📝 **前端约束规则**

### 规则1: 基于问题类型限制策略

```javascript
const policyConstraints = {
    'tsp': ['attention', 'pomo'],           // TSP支持所有策略
    'cvrp': ['attention', 'pomo'],          // CVRP支持所有策略
    'sdvrp': ['attention'],                 // SDVRP只推荐AM
    'pctsp': ['attention'],                 // PCTSP只能用AM
    'op': ['attention'],                    // OP只能用AM
};
```

### 规则2: 基于问题类型限制算法

```javascript
const algorithmConstraints = {
    'tsp': ['reinforce', 'ppo', 'a2c'],     // TSP支持所有算法
    'cvrp': ['reinforce', 'ppo', 'a2c'],    // CVRP支持所有算法
    'sdvrp': ['reinforce', 'ppo', 'a2c'],   // SDVRP支持所有算法
    'pctsp': ['reinforce', 'ppo', 'a2c'],   // PCTSP支持所有算法
    'op': ['reinforce', 'ppo', 'a2c'],      // OP支持所有算法
};
```

### 规则3: 基于策略限制算法

```javascript
const policyAlgorithmConstraints = {
    'attention': ['reinforce', 'ppo', 'a2c'],  // AM支持所有算法
    'pomo': ['reinforce', 'ppo', 'a2c'],       // POMO支持所有算法
};
```

### 规则4: 警告组合

```javascript
const warningCombinations = [
    {
        problem: 'sdvrp',
        policy: 'pomo',
        message: '⚠️ POMO在SDVRP上的效果未经充分验证，建议使用Attention Model'
    },
    // 可以添加更多警告
];
```

---

## 🎨 **推荐配置模板**

### 入门学习
```yaml
problem: tsp
policy: attention
algorithm: reinforce
epochs: 5
说明: 最简单组合，适合理解原理
```

### 快速原型
```yaml
problem: tsp / cvrp
policy: attention
algorithm: a2c
epochs: 10
说明: 收敛快，性能稳定
```

### 高质量解
```yaml
problem: tsp / cvrp
policy: pomo
algorithm: ppo
epochs: 20
num_starts: 50
说明: 最佳组合，质量最高
```

### 生产环境
```yaml
problem: 任意
policy: attention (速度) / pomo (质量)
algorithm: ppo
epochs: 根据需求
说明: 稳定可靠，工业界验证
```

---

## 🔧 **后端验证逻辑**

```python
def validate_combination(problem, policy, algorithm):
    """
    验证组合是否有效
    
    返回: (is_valid, message, level)
        is_valid: bool - 是否有效
        message: str - 提示信息
        level: str - 'error', 'warning', 'info'
    """
    # 检查策略是否支持该问题
    if not is_policy_compatible_with_problem(policy, problem):
        return False, f'{policy}策略不适用于{problem}问题', 'error'
    
    # 检查算法是否支持该问题
    if not is_algorithm_compatible_with_problem(algorithm, problem):
        return False, f'{algorithm}算法不适用于{problem}问题', 'error'
    
    # 检查策略和算法是否兼容
    if not is_policy_compatible_with_algorithm(policy, algorithm):
        return False, f'{policy}策略不兼容{algorithm}算法', 'error'
    
    # 检查是否有警告
    warning = get_combination_warning(problem, policy, algorithm)
    if warning:
        return True, warning, 'warning'
    
    return True, '配置有效', 'info'
```

---

## 📚 **参考资料**

### 论文引用
- Attention Model: Kool et al., ICLR 2019
- POMO: Kwon et al., NeurIPS 2020
- REINFORCE: Williams, 1992
- PPO: Schulman et al., 2017
- A2C: Mnih et al., 2016

### 性能数据
基于 TSP-50 的 Gap (与最优解的差距):
- AM + REINFORCE: ~1.4%
- AM + PPO: ~1.2%
- POMO + REINFORCE: ~0.9%
- POMO + PPO: ~0.8%

---

**创建日期**: 2024年12月16日  
**维护者**: RL4CO Display Team  
**版本**: 1.0.0






