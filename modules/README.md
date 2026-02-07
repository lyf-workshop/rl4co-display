# Modules 文件夹架构说明

## 📚 **文档导航**

- 📖 [**新增问题类型完整指南**](./ADD_NEW_PROBLEM_GUIDE.md) ⭐ **推荐首读**
- 📖 [问题兼容性说明](./PROBLEM_COMPATIBILITY.md)
- 📖 [兼容性矩阵](./COMPATIBILITY_MATRIX.md)
- 📖 [算法和策略指南](./ALGORITHM_AND_POLICY_GUIDE.md)
- 📖 [RL训练使用指南](./rl_training/USAGE_GUIDE.md)

---

## 📊 **整体架构（更新后）**

```
modules/
│
├── problems/              # 问题定义层（What - 解决什么问题）
│   ├── __init__.py       # 问题注册表
│   ├── base_problem.py   # 问题基类
│   ├── tsp.py            # TSP问题
│   ├── atsp.py           # ATSP问题
│   ├── mtsp.py           # mTSP问题（多旅行商）
│   ├── cvrp.py           # CVRP问题
│   ├── sdvrp.py          # SDVRP问题
│   └── vrptw.py          # VRPTW问题
│
├── policies/              # ✨ 策略模型层（What Network - 用什么网络）
│   ├── __init__.py       # 策略注册表
│   ├── base_policy.py    # 策略基类
│   ├── attention_model_policy.py  # Attention Model
│   └── pomo_policy.py    # POMO
│
├── algorithms/            # ✨ RL算法层（How - 怎么训练）
│   ├── __init__.py       # 算法注册表
│   ├── base_algorithm.py # 算法基类
│   ├── reinforce_algo.py # REINFORCE
│   ├── ppo_algo.py       # PPO
│   └── a2c_algo.py       # A2C
│
└── rl_training/           # 训练执行层（Orchestration - 编排调度）
    ├── __init__.py
    ├── base_trainer.py   # 基础训练器（组合 problems + policies + algorithms）
    ├── tsp_trainer.py    # TSP训练器
    ├── mtsp_trainer.py   # mTSP训练器
    ├── cvrp_trainer.py   # CVRP训练器
    ├── sdvrp_trainer.py  # SDVRP训练器
    ├── vrptw_trainer.py  # VRPTW训练器
    └── visualizations/   # 可视化函数
        ├── tsp_viz.py
        ├── mtsp_viz.py
        ├── cvrp_viz.py
        ├── sdvrp_viz.py
        └── vrptw_viz.py
```

---

## 🎯 **模块职责划分**

### 1️⃣ **problems/** - 问题定义层

**职责**: 定义组合优化问题

```python
# 示例：TSP问题
class TSProblem(BaseProblem):
    def create_environment(self):
        return TSPEnv(generator_params={'num_loc': 50})
    
    def get_problem_features(self):
        return ['无容量约束', '单条路径', 'NP-hard']
```

**包含**:
- 环境创建
- 参数定义和验证
- 问题元信息

**不包含**:
- 训练算法
- 策略网络
- 可视化函数

---

### 2️⃣ **policies/** - 策略模型层 ✨ 新增

**职责**: 定义策略网络结构

```python
# 示例：Attention Model策略
class AttentionModelPolicyWrapper(BasePolicy):
    def create_policy(self, env):
        return AttentionModelPolicy(
            env_name=env.name,
            embed_dim=128,
            num_encoder_layers=3,
            num_heads=8,
        )
```

**包含**:
- 网络结构定义
- 编码器/解码器配置
- 策略特定参数

**不包含**:
- 训练算法
- 损失函数
- 优化器配置

---

### 3️⃣ **algorithms/** - RL算法层 ✨ 新增

**职责**: 定义强化学习训练算法

```python
# 示例：REINFORCE算法
class REINFORCEAlgorithm(BaseAlgorithm):
    def create_model(self, env, policy):
        return REINFORCE(
            env,
            policy,
            baseline='rollout',
            batch_size=512,
        )
```

**包含**:
- 训练算法逻辑
- 损失函数定义
- 基线方法选择

**不包含**:
- 网络结构
- 问题定义
- 可视化逻辑

---

### 4️⃣ **rl_training/** - 训练执行层

**职责**: 组合以上三层，执行完整训练流程

```python
# 示例：基础训练器
class BaseTrainer:
    def train(self):
        # 1. 创建问题环境
        env = self.initialize_environment()
        
        # 2. 创建策略网络
        policy = self.create_policy(env)
        
        # 3. 创建RL模型
        model = self.create_model(env, policy)
        
        # 4. 训练
        trainer.fit(model)
        
        # 5. 可视化
        self.generate_visualizations(...)
```

**包含**:
- 训练流程编排
- 进度回调
- 可视化生成
- 文件管理

---

## 🔄 **数据流**

```
用户配置
   ↓
┌─────────────────────────────────────┐
│         BaseTrainer                 │
│  (训练编排层)                         │
└─────────────────────────────────────┘
         ↓           ↓           ↓
    ┌────────┐  ┌────────┐  ┌──────────┐
    │Problem │  │Policy  │  │Algorithm │
    │(问题)  │  │(策略)  │  │(算法)    │
    └────────┘  └────────┘  └──────────┘
         ↓           ↓           ↓
    ┌─────────────────────────────────┐
    │        RL4CO Library            │
    │  (实际的强化学习训练)             │
    └─────────────────────────────────┘
```

---

## 💡 **设计优势**

### ✅ **关注点分离**
每个模块只负责一件事，职责清晰

### ✅ **高度解耦**
- Problems 不依赖 Policies 和 Algorithms
- Policies 不依赖 Problems 和 Algorithms
- Algorithms 不依赖 Problems 和 Policies
- Trainer 组合使用三者

### ✅ **易于扩展**
- 添加新问题：创建新的 Problem 类
- 添加新策略：创建新的 Policy 类
- 添加新算法：创建新的 Algorithm 类
- 三者独立扩展，互不影响

### ✅ **注册表模式**
统一的调用接口，避免硬编码

```python
# 通过字符串动态获取类
ProblemClass = get_problem_class('tsp')
PolicyClass = get_policy_class('pomo')
AlgorithmClass = get_algorithm_class('ppo')
```

### ✅ **配置驱动**
用户只需修改配置，无需修改代码

```python
config = {
    'problem': 'tsp',      # 问题类型
    'model': 'pomo',       # 策略模型
    'algorithm': 'ppo',    # RL算法
    'num_loc': 50,
    'epochs': 10,
}
```

---

## 📈 **对比：优化前 vs 优化后**

### ❌ 优化前

```python
# base_trainer.py
class BaseTrainer:
    def create_policy(self, env):
        # ❌ 硬编码，只支持 Attention Model
        policy = AttentionModelPolicy(...)
        return policy
    
    def create_model(self, env, policy):
        # ❌ 硬编码，只支持 REINFORCE
        model = REINFORCE(...)
        return model
```

**问题**:
- 只支持一种策略（AM）
- 只支持一种算法（REINFORCE）
- 添加新模型需要修改训练器代码
- 用户无法从前端选择

---

### ✅ 优化后

```python
# base_trainer.py
class BaseTrainer:
    def create_policy(self, env):
        # ✅ 动态选择策略
        PolicyClass = get_policy_class(self.policy_name)
        policy_wrapper = PolicyClass(self.config)
        return policy_wrapper.create_policy(env)
    
    def create_model(self, env, policy):
        # ✅ 动态选择算法
        AlgorithmClass = get_algorithm_class(self.algorithm_name)
        algorithm_wrapper = AlgorithmClass(self.config)
        return algorithm_wrapper.create_model(env, policy)
```

**优势**:
- ✅ 支持多种策略（AM、POMO、SymNCO...）
- ✅ 支持多种算法（REINFORCE、PPO、A2C...）
- ✅ 添加新模型只需新建文件，注册即可
- ✅ 用户可以从前端自由选择组合

---

## 🚀 **使用示例**

### 示例1: TSP + Attention Model + REINFORCE

```python
config = {
    'problem': 'tsp',
    'model': 'attention',
    'algorithm': 'reinforce',
    'num_loc': 50,
    'epochs': 10,
}
```

### 示例2: CVRP + POMO + PPO

```python
config = {
    'problem': 'cvrp',
    'model': 'pomo',
    'algorithm': 'ppo',
    'num_loc': 100,
    'vehicle_capacity': 1.0,
    'num_starts': 50,
    'clip_ratio': 0.2,
    'epochs': 20,
}
```

### 示例3: mTSP + POMO + PPO（多旅行商问题）

```python
config = {
    'problem': 'mtsp',
    'model': 'pomo',
    'algorithm': 'ppo',
    'num_loc': 50,
    'num_agents': 5,        # 代理数量
    'cost_type': 'minmax',  # 优化目标：minmax 或 sum
    'num_starts': 50,
    'epochs': 10,
}
```

### 示例4: TSP + Attention Model + A2C

```python
config = {
    'problem': 'tsp',
    'model': 'attention',
    'algorithm': 'a2c',
    'num_loc': 50,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'epochs': 10,
}
```

---

## 📚 **相关文档**

- [算法和策略管理指南](./ALGORITHM_AND_POLICY_GUIDE.md) - 详细使用说明
- [问题兼容性说明](./PROBLEM_COMPATIBILITY.md) - 各问题支持的策略和算法
- [完整兼容性矩阵](./COMPATIBILITY_MATRIX.md) - 详细的兼容性规则和警告
- [问题模块文档](./problems/README.md) - 如何添加新问题
- [训练模块文档](./rl_training/README.md) - 训练器使用指南

---

## 🎯 **快速开始**

### 1. 查看可用的算法和策略

```python
from modules.algorithms import list_available_algorithms
from modules.policies import list_available_policies

print("可用算法:", list_available_algorithms())
print("可用策略:", list_available_policies())
```

### 2. 创建训练配置

```python
config = {
    'problem': 'tsp',
    'model': 'pomo',        # 策略模型
    'algorithm': 'ppo',     # RL算法
    'num_loc': 50,
    'epochs': 10,
}
```

### 3. 启动训练

```python
from modules.rl_training import real_rl4co_training

real_rl4co_training(
    config=config,
    session_id='xxx',
    user_id=1,
    queue=queue,
    training_status=status,
    get_background_db_func=get_db
)
```

---

## ✨ **未来扩展**

### 算法层
- [ ] DQN (Deep Q-Network)
- [ ] SAC (Soft Actor-Critic)
- [ ] TD3 (Twin Delayed DDPG)

### 策略层
- [ ] SymNCO (Symmetric NCO)
- [ ] MDAM (Multi-Decoder Attention Model)
- [ ] MatNet (Matrix Encoding Network)
- [ ] DeepACO (Deep Ant Colony Optimization)

### 问题层
- [x] TSP (Traveling Salesman Problem) ✅
- [x] ATSP (Asymmetric TSP) ✅
- [x] mTSP (Multiple TSP) ✅
- [x] CVRP (Capacitated VRP) ✅
- [x] SDVRP (Split Delivery VRP) ✅
- [x] VRPTW (VRP with Time Windows) ✅
- [ ] PCTSP (Prize Collecting TSP) - 计划中
- [ ] OP (Orienteering Problem) - 计划中
- [ ] MDVRP (Multi-Depot VRP) - 计划中

---

## 🔗 **兼容性约束**

### 策略和算法兼容性

不同问题类型支持不同的策略和算法组合。详细信息请查看：

📖 **[问题兼容性说明文档](./PROBLEM_COMPATIBILITY.md)**

### 快速参考

| 问题类型 | 支持的策略 | 支持的算法 |
|---------|-----------|-----------|
| **TSP** | Attention, POMO | REINFORCE, PPO, A2C |
| **ATSP** | Attention only | REINFORCE, PPO, A2C |
| **mTSP** ⭐ | Attention, POMO | REINFORCE, PPO, A2C |
| **CVRP** | Attention, POMO | REINFORCE, PPO, A2C |
| **SDVRP** | Attention (推荐) | REINFORCE, PPO, A2C |
| **VRPTW** | Attention (推荐) | PPO (推荐), A2C |

**重要提示**：
- ❌ POMO 不支持 ATSP（非对称问题）
- ⚠️ POMO 在 SDVRP/VRPTW 上效果未验证
- ⚠️ REINFORCE 在 ATSP/VRPTW 上收敛困难，建议使用 PPO 或 A2C

---

**创建日期**: 2024年12月16日  
**最后更新**: 2026年02月04日（添加 mTSP 支持）  
**维护者**: RL4CO Display Team  
**状态**: ✅ **活跃维护**






