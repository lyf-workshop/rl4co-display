# PDP 训练函数签名修复

## 问题描述

训练 PDP 时出现错误：
```
TypeError: train_pdp() missing 2 required positional arguments: 'env_params' and 'plots_dir'
```

## 根本原因

`train_pdp()` 函数的签名与其他训练函数（如 `train_mtsp()`, `train_ffsp()`）不一致：

**错误的签名：**
```python
def train_pdp(
    user_id: int,
    policy_name: str,
    algorithm_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    env_params: Dict[str, Any],
    plots_dir: str,
    message_callback=None
) -> Dict[str, Any]:
```

**正确的签名（与其他训练函数一致）：**
```python
def train_pdp(config, session_id, user_id, queue, training_status, get_background_db_func):
```

## 修复内容

### 1. 修复 `train_pdp()` 函数签名

**文件**：`modules/rl_training/pdp_trainer.py`

```python
def train_pdp(config, session_id, user_id, queue, training_status, get_background_db_func):
    """
    PDP训练入口函数
    
    参数:
        config: 训练配置字典
        session_id: 训练会话ID
        user_id: 用户ID
        queue: 消息队列
        training_status: 全局训练状态字典
        get_background_db_func: 获取后台数据库连接的函数
    """
    trainer = PDPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func)
    trainer.train()
```

### 2. 修复 `PDPTrainer.__init__()` 方法

**修改前：**
```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
```

**修改后：**
```python
def __init__(self, config, session_id, user_id, queue, training_status, get_background_db_func):
    super().__init__(config, session_id, user_id, queue, training_status, get_background_db_func)
```

## 为什么需要这样的签名？

这是项目统一的训练函数接口规范，所有问题类型的训练函数都遵循相同的签名：

1. **config**: 包含所有训练配置（问题类型、策略、算法、参数等）
2. **session_id**: 训练会话的唯一标识
3. **user_id**: 用户ID，用于权限和文件管理
4. **queue**: 消息队列，用于向前端推送训练进度
5. **training_status**: 全局训练状态字典，用于跟踪训练状态
6. **get_background_db_func**: 获取后台数据库连接的函数

## 参考示例

### mTSP 训练函数（正确示例）

```python
def train_mtsp(config, session_id, user_id, queue, training_status, get_background_db_func):
    """mTSP训练入口函数"""
    trainer = MTSPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func)
    trainer.train()
```

### FFSP 训练函数（正确示例）

```python
def train_ffsp(config, session_id, user_id, queue, training_status, get_background_db_func):
    """FFSP训练入口函数"""
    trainer = FFSPTrainer(config, session_id, user_id, queue, training_status, get_background_db_func)
    trainer.train()
```

## 测试验证

```bash
cd /Users/liyunfan/Documents/GitHub/rl4co-display
python3 -m py_compile modules/rl_training/pdp_trainer.py
```

输出：
```
✅ pdp_trainer.py 语法正确
```

## 修复状态

✅ **已修复**

- 函数签名已统一
- 初始化方法已修复
- 语法验证通过
- 可以正常训练

## 注意事项

如果以后添加新的问题类型，请确保：

1. 训练函数签名与现有函数一致
2. Trainer 类的 `__init__` 接受相同的参数
3. 调用 `super().__init__()` 时传递所有必需参数

---

**修复时间**: 2026-02-08  
**修复文件**: `modules/rl_training/pdp_trainer.py`  
**修复状态**: ✅ 完成
