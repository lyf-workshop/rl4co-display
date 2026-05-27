# MDAM + DeepACO 策略模型集成实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 rl4co-display 平台中新增 MDAM（多解码器注意力模型）和 DeepACO（深度蚁群优化）两种策略模型，覆盖后端策略封装、算法模型创建、兼容性注册以及前端UI更新。

**Architecture:** 每个新模型遵循现有的 BasePolicy → PolicyWrapper → POLICY_REGISTRY 模式；MDAM 和 DeepACO 各自有专属的 rl4co 模型类（继承自 REINFORCE），在 `base_trainer.py` 中添加与 SymNCO 同等的特殊分支。前端 `index.html` 扩展下拉选项、兼容性矩阵和特有参数面板。

**Tech Stack:** Python 3.8+, rl4co (`rl4co.models.zoo.mdam`, `rl4co.models.zoo.deepaco`), Flask, Jinja2, pytest

---

## 文件结构

| 操作 | 文件 | 说明 |
|---|---|---|
| 创建 | `modules/policies/mdam_policy.py` | MDAMPolicyWrapper |
| 创建 | `modules/policies/deepaco_policy.py` | DeepACOPolicyWrapper |
| 修改 | `modules/policies/__init__.py` | 注册 mdam/deepaco |
| 修改 | `modules/compatibility.py` | 兼容性规则 |
| 修改 | `modules/rl_training/base_trainer.py` | `_create_mdam_model`, `_create_deepaco_model`, `create_model` 分支 |
| 修改 | `templates/index.html` | 下拉选项、描述、DeepACO参数面板 |
| 创建 | `tests/test_new_policies.py` | 策略封装 + 兼容性单元测试 |

---

## Task 1: MDAM 策略封装 + 模型创建

**Files:**
- Create: `modules/policies/mdam_policy.py`
- Modify: `modules/rl_training/base_trainer.py`

- [ ] **Step 1: 写失败测试**

新建 `tests/test_new_policies.py`，写 MDAM 部分的测试（此时代码不存在，测试必定失败）：

```python
"""新策略模型（MDAM、DeepACO）的单元测试"""
import pytest


# =========================================================
# MDAM 策略封装测试
# =========================================================

class TestMDAMPolicyWrapper:

    def test_get_policy_name(self):
        from modules.policies.mdam_policy import MDAMPolicyWrapper
        w = MDAMPolicyWrapper({'embed_dim': 128, 'num_encoder_layers': 3, 'num_heads': 8})
        assert w.get_policy_name() == 'mdam'

    def test_validate_config_valid(self):
        from modules.policies.mdam_policy import MDAMPolicyWrapper
        w = MDAMPolicyWrapper({'embed_dim': 128, 'num_encoder_layers': 3, 'num_heads': 8})
        valid, msg = w.validate_config()
        assert valid, msg

    def test_validate_config_embed_not_divisible(self):
        """embed_dim=100 不能被 num_heads=8 整除，BasePolicy 应拒绝"""
        from modules.policies.mdam_policy import MDAMPolicyWrapper
        w = MDAMPolicyWrapper({'embed_dim': 100, 'num_encoder_layers': 3, 'num_heads': 8})
        valid, _ = w.validate_config()
        assert not valid

    def test_policy_params_returns_standard_keys(self):
        from modules.policies.mdam_policy import MDAMPolicyWrapper
        w = MDAMPolicyWrapper({'embed_dim': 64, 'num_encoder_layers': 2, 'num_heads': 4})
        params = w.get_policy_params()
        assert 'embed_dim' in params
        assert params['embed_dim'] == 64
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_new_policies.py::TestMDAMPolicyWrapper -v
```

期望输出：`ImportError: No module named 'modules.policies.mdam_policy'`

- [ ] **Step 3: 创建 `modules/policies/mdam_policy.py`**

```python
"""
MDAM (Multi-Decoder Attention Model) 策略网络封装

多解码器注意力模型：同时运行多个 Transformer 解码器，
推理时取最优路径，在与 AM 相同训练成本下提升解质量。
"""

from typing import Dict, Any
from .base_policy import BasePolicy


class MDAMPolicyWrapper(BasePolicy):
    """
    MDAM 策略封装类

    特点：
        - 多个解码器并行解码，取最优解
        - 兼容 AM 支持的所有路由问题（不含 ATSP/FFSP）
        - 架构参数与 AM 一致（embed_dim, num_encoder_layers, num_heads）
        - 使用内置 MDAM(REINFORCE 子类) 训练，不兼容外部 PPO/A2C
    """

    def _init_policy_params(self):
        """MDAM 使用标准 embed_dim/layers/heads，无额外参数"""
        pass

    def get_policy_name(self) -> str:
        return 'mdam'

    def create_policy(self, env):
        """
        创建 MDAMPolicy 策略网络

        参数:
            env: RL4CO 环境（提供 env.name）

        返回:
            MDAMPolicy 实例
        """
        try:
            from rl4co.models.zoo.mdam import MDAMPolicy
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或当前版本不包含MDAM。\n"
                "请安装最新版: pip install rl4co"
            )

        policy = MDAMPolicy(
            env_name=env.name,
            embed_dim=self.embed_dim,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
        )
        return policy
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_new_policies.py::TestMDAMPolicyWrapper -v
```

期望：4 个测试全部 PASS

- [ ] **Step 5: 在 `base_trainer.py` 中添加 `_create_mdam_model` 方法**

在 `_create_symnco_model` 方法之后（约第 586 行）、`create_model` 方法之前，插入：

```python
    def _create_mdam_model(self, env, policy):
        """为 MDAM 策略创建专用训练模型（内置多解码器 REINFORCE）"""
        try:
            from rl4co.models.zoo.mdam import MDAM
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或当前版本不包含MDAM。\n"
                "请安装最新版: pip install rl4co"
            )

        model = MDAM(
            env,
            policy,
            baseline=self.config.get('baseline', 'rollout'),
            batch_size=self.batch_size,
            train_data_size=10_000,
            val_data_size=1_000,
            optimizer_kwargs={'lr': self.learning_rate},
        )
        self.send_message('info', '✅ MDAM模型创建成功（多解码器注意力）')
        return model
```

- [ ] **Step 6: 在 `create_model` 方法中添加 mdam 分支**

在 `create_model` 方法开头，紧接 SymNCO 分支后（约第 591 行），添加：

```python
        # MDAM 使用内置的多解码器 REINFORCE 子类
        if self.policy_name == 'mdam':
            return self._create_mdam_model(env, policy)
```

完整的 `create_model` 开头应如下：

```python
    def create_model(self, env, policy):
        """创建RL模型（支持多种算法）"""
        # SymNCO 使用内置的自定义多损失训练算法，跳过算法注册表
        if self.policy_name == 'symnco':
            return self._create_symnco_model(env, policy)

        # MDAM 使用内置的多解码器 REINFORCE 子类
        if self.policy_name == 'mdam':
            return self._create_mdam_model(env, policy)

        # ========== 使用新的算法注册表（如果可用） ==========
        # ... 以下代码保持不变 ...
```

- [ ] **Step 7: 提交**

```bash
git add modules/policies/mdam_policy.py modules/rl_training/base_trainer.py tests/test_new_policies.py
git commit -m "feat: add MDAMPolicyWrapper and _create_mdam_model to base_trainer"
```

---

## Task 2: MDAM 注册（策略表 + 兼容性）

**Files:**
- Modify: `modules/policies/__init__.py`
- Modify: `modules/compatibility.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_new_policies.py` 末尾追加：

```python
# =========================================================
# MDAM 注册 + 兼容性测试
# =========================================================

class TestMDAMRegistration:

    def test_in_policy_registry(self):
        from modules.policies import POLICY_REGISTRY
        assert 'mdam' in POLICY_REGISTRY

    def test_in_policy_info(self):
        from modules.policies import POLICY_INFO
        assert 'mdam' in POLICY_INFO
        assert POLICY_INFO['mdam']['type'] == 'multi-decoder-constructive'

    def test_get_policy_class_returns_wrapper(self):
        from modules.policies import get_policy_class
        from modules.policies.mdam_policy import MDAMPolicyWrapper
        cls = get_policy_class('mdam')
        assert cls is MDAMPolicyWrapper


class TestMDAMCompatibility:

    def test_compat_tsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('mdam', 'tsp') is True

    def test_compat_cvrp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('mdam', 'cvrp') is True

    def test_compat_op(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('mdam', 'op') is True

    def test_compat_pctsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('mdam', 'pctsp') is True

    def test_not_compat_atsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('mdam', 'atsp') is False

    def test_not_compat_ffsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('mdam', 'ffsp') is False

    def test_algo_reinforce_ok(self):
        from modules.compatibility import is_policy_compatible_with_algorithm
        assert is_policy_compatible_with_algorithm('mdam', 'reinforce') is True

    def test_algo_ppo_not_ok(self):
        from modules.compatibility import is_policy_compatible_with_algorithm
        assert is_policy_compatible_with_algorithm('mdam', 'ppo') is False
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_new_policies.py::TestMDAMRegistration tests/test_new_policies.py::TestMDAMCompatibility -v
```

期望：`KeyError: 'mdam'` 或 `AssertionError`

- [ ] **Step 3: 在 `modules/policies/__init__.py` 注册 MDAM**

在导入区块末尾添加导入，在注册表和元信息中添加条目。

在现有 6 个 import 后添加：
```python
from .mdam_policy import MDAMPolicyWrapper
```

在 `POLICY_REGISTRY` 字典中添加：
```python
    'mdam': MDAMPolicyWrapper,  # MDAM - 多解码器注意力，兼容性与AM相同
```

在 `POLICY_INFO` 字典中添加（置于 `'symnco'` 条目后）：
```python
    'mdam': {
        'name': 'MDAM',
        'full_name': 'Multi-Decoder Attention Model',
        'cn_name': '多解码器注意力模型',
        'type': 'multi-decoder-constructive',
        'year': 2021,
        'status': 'active',
        'description': '多解码器并行生成多条路径，取最优解，与AM相同兼容性但解质量更高',
        'advantages': ['解质量高于AM', '兼容性广', '训练稳定', '推理多样化'],
        'disadvantages': ['推理略慢（多解码器）', '不支持PPO/A2C'],
        'suitable_for': ['TSP', 'CVRP', 'mTSP', 'OP', 'PDP', 'PCTSP', 'SPCTSP', 'VRPTW', 'SDVRP'],
        'params': {
            'embed_dim': {'default': 128, 'range': [64, 256]},
            'num_encoder_layers': {'default': 3, 'range': [1, 6]},
            'num_heads': {'default': 8, 'range': [4, 16]},
        }
    },
```

在 `__all__` 列表中添加：
```python
    'MDAMPolicyWrapper',
```

- [ ] **Step 4: 在 `modules/compatibility.py` 注册 MDAM**

在 `POLICY_PROBLEM_COMPATIBILITY` 字典中，`'symnco'` 条目之后添加：
```python
    # MDAM：多解码器AM，兼容性与AM相同（不含ATSP/FFSP）
    'mdam': _ROUTING_WITHOUT_ATSP,
```

在 `POLICY_ALGORITHM_COMPATIBILITY` 字典中添加：
```python
    # MDAM内置REINFORCE子类，PPO/A2C的外部训练循环与之不兼容
    'mdam': ['reinforce'],
```

在 `WARNING_COMBINATIONS` 列表末尾添加：
```python
    # MDAM 不兼容警告
    {
        'problem': 'atsp',
        'policy': 'mdam',
        'message': 'MDAM不支持ATSP：ATSPEnv只提供cost_matrix，无locs坐标，MDAM无法处理。请使用MatNet',
        'severity': 'error'
    },
    {
        'problem': 'ffsp',
        'policy': 'mdam',
        'message': 'MDAM不支持FFSP调度问题。请使用MatNet',
        'severity': 'error'
    },
    {
        'policy': 'mdam',
        'algorithm': 'ppo',
        'message': 'MDAM使用内置多解码器REINFORCE子类，不支持外部PPO算法。算法选项将被忽略',
        'severity': 'warning'
    },
    {
        'policy': 'mdam',
        'algorithm': 'a2c',
        'message': 'MDAM使用内置多解码器REINFORCE子类，不支持外部A2C算法。算法选项将被忽略',
        'severity': 'warning'
    },
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
pytest tests/test_new_policies.py::TestMDAMRegistration tests/test_new_policies.py::TestMDAMCompatibility -v
```

期望：10 个测试全部 PASS

- [ ] **Step 6: 提交**

```bash
git add modules/policies/__init__.py modules/compatibility.py tests/test_new_policies.py
git commit -m "feat: register MDAM in policy registry and compatibility matrix"
```

---

## Task 3: MDAM 前端（下拉选项 + 兼容性映射 + 描述）

**Files:**
- Modify: `templates/index.html`

- [ ] **Step 1: 在 `#model-select` 下拉中添加 MDAM 选项**

在 `index.html` 中找到：
```html
<option value="symnco">SymNCO（对称优化，最高质量）🆕</option>
```

在其后添加：
```html
<option value="mdam">MDAM（多解码器，高质量）🆕</option>
```

- [ ] **Step 2: 在 JS `POLICY_PROBLEM_COMPAT` 中添加 mdam**

找到 `updatePolicyConstraints()` 中的 `POLICY_PROBLEM_COMPAT` 对象，添加：
```javascript
'mdam':      ['tsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'op', 'pdp', 'pctsp', 'spctsp'],
```

- [ ] **Step 3: 在 JS `POLICY_ALGO_COMPAT` 中添加 mdam**

找到 `updateAlgorithmConstraints()` 中的 `POLICY_ALGO_COMPAT` 对象，添加：
```javascript
'mdam':      ['reinforce'],
```

- [ ] **Step 4: 在 `descriptions` 对象中添加 mdam 描述**

找到 `updatePolicyDescription()` 中的 `descriptions` 对象，添加：
```javascript
'mdam': '多解码器注意力模型：多个解码器并行解码后取最优路径，与AM兼容性相同但解质量更高，训练成本相近',
```

- [ ] **Step 5: 提交**

```bash
git add templates/index.html
git commit -m "feat: add MDAM option to frontend model selector"
```

---

## Task 4: DeepACO 策略封装 + 模型创建

**Files:**
- Create: `modules/policies/deepaco_policy.py`
- Modify: `modules/rl_training/base_trainer.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_new_policies.py` 末尾追加：

```python
# =========================================================
# DeepACO 策略封装测试
# =========================================================

class TestDeepACOPolicyWrapper:

    def test_get_policy_name(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({'embed_dim': 128, 'num_encoder_layers': 3,
                                  'num_heads': 8, 'n_ants': 30})
        assert w.get_policy_name() == 'deepaco'

    def test_default_n_ants(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({'embed_dim': 128, 'num_encoder_layers': 3, 'num_heads': 8})
        assert w.n_ants == 30

    def test_validate_config_valid(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({'embed_dim': 128, 'num_encoder_layers': 3,
                                  'num_heads': 8, 'n_ants': 30})
        valid, msg = w.validate_config()
        assert valid, msg

    def test_validate_n_ants_zero_fails(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({'embed_dim': 128, 'num_encoder_layers': 3,
                                  'num_heads': 8, 'n_ants': 0})
        valid, _ = w.validate_config()
        assert not valid

    def test_validate_n_ants_too_many_fails(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({'embed_dim': 128, 'num_encoder_layers': 3,
                                  'num_heads': 8, 'n_ants': 300})
        valid, _ = w.validate_config()
        assert not valid

    def test_policy_params_includes_n_ants(self):
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        w = DeepACOPolicyWrapper({'embed_dim': 128, 'num_encoder_layers': 3,
                                  'num_heads': 8, 'n_ants': 40})
        params = w.get_policy_params()
        assert params['n_ants'] == 40
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_new_policies.py::TestDeepACOPolicyWrapper -v
```

期望：`ImportError: No module named 'modules.policies.deepaco_policy'`

- [ ] **Step 3: 创建 `modules/policies/deepaco_policy.py`**

```python
"""
DeepACO (Deep Ant Colony Optimization) 策略网络封装

深度学习 + 蚁群算法混合模型：
- GNN 编码器学习信息素矩阵（启发式热图）
- AntSystem 利用该热图执行 ACO 搜索
- 非自回归范式，与 AM/POMO 等自回归模型形成对比
"""

from typing import Dict, Any
from .base_policy import BasePolicy


class DeepACOPolicyWrapper(BasePolicy):
    """
    DeepACO 策略封装类

    特点：
        - 非自回归：先学习信息素热图，再用 ACO 搜索
        - 蚂蚁数量 n_ants 控制探索宽度（训练/测试可分别设置）
        - 兼容 TSP、CVRP、mTSP、OP 等标准路由问题
        - 使用内置 DeepACO(REINFORCE 子类) 训练，不兼容外部 PPO/A2C
    """

    def _init_policy_params(self):
        """初始化 DeepACO 特定参数"""
        self.n_ants = int(self.config.get('n_ants', 30))
        self.n_iterations_train = int(self.config.get('n_iterations_train', 1))
        self.n_iterations_test = int(self.config.get('n_iterations_test', 5))

    def get_policy_name(self) -> str:
        return 'deepaco'

    def create_policy(self, env):
        """
        创建 DeepACOPolicy 策略网络

        参数:
            env: RL4CO 环境（提供 env.name）

        返回:
            DeepACOPolicy 实例
        """
        try:
            from rl4co.models.zoo.deepaco import DeepACOPolicy
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或当前版本不包含DeepACO。\n"
                "请安装最新版: pip install rl4co"
            )

        policy = DeepACOPolicy(
            env_name=env.name,
            n_ants={
                'train': self.n_ants,
                'val': self.n_ants,
                'test': self.n_ants,
            },
            n_iterations={
                'train': self.n_iterations_train,
                'val': self.n_iterations_test,
                'test': self.n_iterations_test,
            },
        )
        return policy

    def get_policy_params(self) -> Dict[str, Any]:
        """获取 DeepACO 策略参数"""
        params = super().get_policy_params()
        params['n_ants'] = self.n_ants
        params['n_iterations_train'] = self.n_iterations_train
        params['n_iterations_test'] = self.n_iterations_test
        return params

    def _validate_policy_params(self):
        """验证 DeepACO 参数"""
        if self.n_ants < 1:
            return False, "n_ants必须大于0"
        if self.n_ants > 200:
            return False, "n_ants不建议超过200（显存限制）"
        if self.n_iterations_train < 1:
            return False, "n_iterations_train必须大于0"
        return True, ""
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_new_policies.py::TestDeepACOPolicyWrapper -v
```

期望：6 个测试全部 PASS

- [ ] **Step 5: 在 `base_trainer.py` 中添加 `_create_deepaco_model` 方法**

在 `_create_mdam_model` 方法之后，插入：

```python
    def _create_deepaco_model(self, env, policy):
        """为 DeepACO 策略创建专用训练模型（深度蚁群 REINFORCE 子类）"""
        try:
            from rl4co.models.zoo.deepaco import DeepACO
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或当前版本不包含DeepACO。\n"
                "请安装最新版: pip install rl4co"
            )

        model = DeepACO(
            env,
            policy,
            baseline='no',  # DeepACO 内部实现共享基线，跳过外部 baseline
            batch_size=self.batch_size,
            train_data_size=10_000,
            val_data_size=1_000,
            optimizer_kwargs={'lr': self.learning_rate},
        )
        self.send_message('info', '✅ DeepACO模型创建成功（深度蚁群优化）')
        return model
```

- [ ] **Step 6: 在 `create_model` 中添加 deepaco 分支**

紧接 mdam 分支之后添加：

```python
        # DeepACO 使用内置的蚁群 REINFORCE 子类
        if self.policy_name == 'deepaco':
            return self._create_deepaco_model(env, policy)
```

完整的开头应为：
```python
    def create_model(self, env, policy):
        """创建RL模型（支持多种算法）"""
        if self.policy_name == 'symnco':
            return self._create_symnco_model(env, policy)

        if self.policy_name == 'mdam':
            return self._create_mdam_model(env, policy)

        if self.policy_name == 'deepaco':
            return self._create_deepaco_model(env, policy)

        # ========== 使用新的算法注册表（如果可用） ==========
        # ... 以下代码保持不变 ...
```

- [ ] **Step 7: 提交**

```bash
git add modules/policies/deepaco_policy.py modules/rl_training/base_trainer.py tests/test_new_policies.py
git commit -m "feat: add DeepACOPolicyWrapper and _create_deepaco_model to base_trainer"
```

---

## Task 5: DeepACO 注册（策略表 + 兼容性）

**Files:**
- Modify: `modules/policies/__init__.py`
- Modify: `modules/compatibility.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_new_policies.py` 末尾追加：

```python
# =========================================================
# DeepACO 注册 + 兼容性测试
# =========================================================

class TestDeepACORegistration:

    def test_in_policy_registry(self):
        from modules.policies import POLICY_REGISTRY
        assert 'deepaco' in POLICY_REGISTRY

    def test_in_policy_info(self):
        from modules.policies import POLICY_INFO
        assert 'deepaco' in POLICY_INFO
        assert POLICY_INFO['deepaco']['type'] == 'hybrid-aco'

    def test_get_policy_class_returns_wrapper(self):
        from modules.policies import get_policy_class
        from modules.policies.deepaco_policy import DeepACOPolicyWrapper
        cls = get_policy_class('deepaco')
        assert cls is DeepACOPolicyWrapper


class TestDeepACOCompatibility:

    def test_compat_tsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'tsp') is True

    def test_compat_cvrp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'cvrp') is True

    def test_compat_op(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'op') is True

    def test_not_compat_atsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'atsp') is False

    def test_not_compat_ffsp(self):
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'ffsp') is False

    def test_not_compat_pctsp(self):
        """DeepACO 对奖励收集类问题未经验证，兼容性矩阵中不包含"""
        from modules.compatibility import is_policy_compatible_with_problem
        assert is_policy_compatible_with_problem('deepaco', 'pctsp') is False

    def test_algo_reinforce_ok(self):
        from modules.compatibility import is_policy_compatible_with_algorithm
        assert is_policy_compatible_with_algorithm('deepaco', 'reinforce') is True

    def test_algo_ppo_not_ok(self):
        from modules.compatibility import is_policy_compatible_with_algorithm
        assert is_policy_compatible_with_algorithm('deepaco', 'ppo') is False
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_new_policies.py::TestDeepACORegistration tests/test_new_policies.py::TestDeepACOCompatibility -v
```

期望：`AssertionError` (key 不存在)

- [ ] **Step 3: 在 `modules/policies/__init__.py` 注册 DeepACO**

在 MDAM 导入之后添加：
```python
from .deepaco_policy import DeepACOPolicyWrapper
```

在 `POLICY_REGISTRY` 中添加：
```python
    'deepaco': DeepACOPolicyWrapper,  # DeepACO - 深度蚁群优化
```

在 `POLICY_INFO` 中添加（置于 `'mdam'` 条目后）：
```python
    'deepaco': {
        'name': 'DeepACO',
        'full_name': 'Deep Ant Colony Optimization',
        'cn_name': '深度蚁群优化',
        'type': 'hybrid-aco',
        'year': 2023,
        'status': 'active',
        'description': '深度学习+蚁群算法混合：GNN学习信息素热图，ACO执行搜索，展示不同于纯RL的混合范式',
        'advantages': ['混合范式，教育价值高', '热图可解释性强', '支持ACO后处理'],
        'disadvantages': ['推理较慢（ACO迭代）', '不支持PPO/A2C', '问题支持范围较窄'],
        'suitable_for': ['TSP', 'CVRP', 'mTSP', 'OP'],
        'params': {
            'n_ants': {'default': 30, 'range': [5, 200]},
            'n_iterations_train': {'default': 1, 'range': [1, 10]},
            'n_iterations_test': {'default': 5, 'range': [1, 20]},
        }
    },
```

在 `__all__` 中添加：
```python
    'DeepACOPolicyWrapper',
```

- [ ] **Step 4: 在 `modules/compatibility.py` 注册 DeepACO**

在 `POLICY_PROBLEM_COMPATIBILITY` 中，`'mdam'` 条目之后添加：
```python
    # DeepACO：深度蚁群优化，非自回归+ACO
    # 支持标准路由问题（不含 ATSP/FFSP/PDP/PCTSP/SPCTSP）
    # 原因：DeepACO的NARGNNEncoder基于坐标图；PDP的前驱约束和PCTSP/SPCTSP的
    # 奖励收集逻辑与ACO搜索的兼容性未在官方文档中验证
    'deepaco': ['tsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'op'],
```

在 `POLICY_ALGORITHM_COMPATIBILITY` 中添加：
```python
    # DeepACO内置REINFORCE子类，PPO/A2C的外部训练循环与之不兼容
    'deepaco': ['reinforce'],
```

在 `WARNING_COMBINATIONS` 列表末尾添加：
```python
    # DeepACO 不兼容警告
    {
        'problem': 'atsp',
        'policy': 'deepaco',
        'message': 'DeepACO不支持ATSP：ATSPEnv只提供cost_matrix，无locs坐标。请使用MatNet',
        'severity': 'error'
    },
    {
        'problem': 'ffsp',
        'policy': 'deepaco',
        'message': 'DeepACO不支持FFSP调度问题。请使用MatNet',
        'severity': 'error'
    },
    {
        'problem': 'pctsp',
        'policy': 'deepaco',
        'message': 'DeepACO对奖励收集类问题（PCTSP）的兼容性未经验证。请使用Attention Model',
        'severity': 'error'
    },
    {
        'problem': 'spctsp',
        'policy': 'deepaco',
        'message': 'DeepACO对随机奖励收集（SPCTSP）的兼容性未经验证。请使用Attention Model',
        'severity': 'error'
    },
    {
        'problem': 'pdp',
        'policy': 'deepaco',
        'message': 'DeepACO对取送货问题（PDP）的前驱约束兼容性未经验证。请使用HAM',
        'severity': 'error'
    },
    {
        'policy': 'deepaco',
        'algorithm': 'ppo',
        'message': 'DeepACO使用内置蚁群REINFORCE子类，不支持外部PPO算法。算法选项将被忽略',
        'severity': 'warning'
    },
    {
        'policy': 'deepaco',
        'algorithm': 'a2c',
        'message': 'DeepACO使用内置蚁群REINFORCE子类，不支持外部A2C算法。算法选项将被忽略',
        'severity': 'warning'
    },
```

- [ ] **Step 5: 运行测试，确认通过**

```bash
pytest tests/test_new_policies.py::TestDeepACORegistration tests/test_new_policies.py::TestDeepACOCompatibility -v
```

期望：11 个测试全部 PASS

- [ ] **Step 6: 运行全部新策略测试**

```bash
pytest tests/test_new_policies.py -v
```

期望：所有测试 PASS

- [ ] **Step 7: 提交**

```bash
git add modules/policies/__init__.py modules/compatibility.py tests/test_new_policies.py
git commit -m "feat: register DeepACO in policy registry and compatibility matrix"
```

---

## Task 6: DeepACO 前端（下拉选项 + n_ants 参数面板）

**Files:**
- Modify: `templates/index.html`

- [ ] **Step 1: 在 `#model-select` 中添加 DeepACO 选项**

在 MDAM 选项之后添加：
```html
<option value="deepaco">DeepACO（蚁群优化，混合方法）🆕</option>
```

- [ ] **Step 2: 添加 DeepACO 参数面板 HTML**

在现有 `<div id="symnco-params">` 面板结束标签 `</div>` 之后，添加：

```html
<!-- DeepACO特有参数（条件显示） -->
<div id="deepaco-params" class="form-group advanced-params" style="display: none;">
    <label for="n-ants">🐜 蚂蚁数量 (n_ants)</label>
    <input type="number" id="n-ants" name="n_ants" class="form-control"
           value="30" min="5" max="200" step="5">
    <small class="form-text text-muted">蚂蚁数量影响解质量与速度，推荐 20-50（越多越慢）</small>
</div>
```

- [ ] **Step 3: 在 JS `POLICY_PROBLEM_COMPAT` 中添加 deepaco**

```javascript
'deepaco':   ['tsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'op'],
```

- [ ] **Step 4: 在 JS `POLICY_ALGO_COMPAT` 中添加 deepaco**

```javascript
'deepaco':   ['reinforce'],
```

- [ ] **Step 5: 在 `descriptions` 中添加 deepaco 描述**

```javascript
'deepaco': '深度学习+蚁群算法混合：GNN学习信息素热图，AntSystem执行搜索，展示与纯RL截然不同的混合优化范式',
```

- [ ] **Step 6: 在 `showPolicySpecificParams()` 中显示/隐藏 deepaco-params**

在现有 symnco-params 的显示逻辑之后添加：
```javascript
const deepacoParams = document.getElementById('deepaco-params');
if (deepacoParams) {
    deepacoParams.style.display = this.currentPolicy === 'deepaco' ? 'block' : 'none';
}
```

- [ ] **Step 7: 在配置收集代码中读取 n_ants 并传入 config**

找到收集 POMO `num_starts` 参数的代码段（约在 `getConfig()` 函数中）：
```javascript
if (model === 'pomo') {
    const numStarts = document.getElementById('num-starts');
    if (numStarts) {
```

在该段之后添加：
```javascript
if (model === 'deepaco') {
    const nAnts = document.getElementById('n-ants');
    if (nAnts) config.n_ants = parseInt(nAnts.value) || 30;
}
```

- [ ] **Step 8: 提交**

```bash
git add templates/index.html
git commit -m "feat: add DeepACO option to frontend with n_ants parameter panel"
```

---

## Task 7: 全量测试验证

- [ ] **Step 1: 运行全部新策略测试**

```bash
pytest tests/test_new_policies.py -v
```

期望：所有测试 PASS

- [ ] **Step 2: 运行完整测试套件（排除已知 torch 环境错误）**

```bash
pytest tests/ -v --ignore=tests/test_gpu_feature.py --ignore=tests/test_atsp_integration.py
```

期望：所有非预期的测试 PASS（`test_parse_dataset.py` 全 18 项通过，其余测试无回归）

- [ ] **Step 3: 运行兼容性一致性检查**

```bash
python -c "from modules.compatibility import validate_system_consistency; validate_system_consistency()"
```

期望输出：`✅ 所有检查通过，配置一致！`

- [ ] **Step 4: 最终提交（如有遗漏的修改）**

```bash
git status
# 如果有未提交的修改，添加并提交
git add -p
git commit -m "fix: address any remaining issues found during final verification"
```

---

## 快速参考：关键 API

```python
# MDAM
from rl4co.models.zoo.mdam import MDAMPolicy, MDAM
policy = MDAMPolicy(env_name='tsp', embed_dim=128, num_encoder_layers=3, num_heads=8)
model  = MDAM(env, policy, baseline='rollout', batch_size=512, optimizer_kwargs={'lr': 1e-4})

# DeepACO
from rl4co.models.zoo.deepaco import DeepACOPolicy, DeepACO
policy = DeepACOPolicy(env_name='tsp',
                       n_ants={'train': 30, 'val': 30, 'test': 30},
                       n_iterations={'train': 1, 'val': 5, 'test': 5})
model  = DeepACO(env, policy, baseline='no', batch_size=512, optimizer_kwargs={'lr': 1e-4})
```

## 注意事项

1. **MDAM 推理输出形状**：`decode_type="greedy"` 时 `reward` 预期为 `[batch]`；若实际返回 `[batch, num_decoders]`，在 `_create_mdam_model` 中可用 `.max(dim=-1).values` 归一化。
2. **DeepACO `actions` 键**：部分 trainer 可视化代码使用 `out['actions']`；DeepACO 测试阶段应返回该键，但若为 `None` 则可视化将降级跳过。
3. **SymNCO 参考**：`_create_symnco_model` 是本计划两个新方法的直接参考实现，遇到问题先对照 SymNCO 的同等代码。
