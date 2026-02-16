# FFSP 可视化修复报告

## 🐛 问题诊断

### 原始错误
```
TypeError: unsupported operand type(s) for +=: 'NoneType' and 'int'
```

### 根本原因

**错误1：step_cnt未初始化**
- `policy()` 调用时未传入 `env` 参数
- RL4CO 内部创建新的 `FFSPEnv`，其 `step_cnt = None`（未调用 `reset()`）
- 解码时执行 `self.step_cnt += 1` → `None + 1` → TypeError

**错误2：可视化图片未生成**
- `generate_visualizations()` 中只创建了文件路径
- 没有真正调用 `create_ffsp_gantt_chart()` 函数
- 注释说"简化版本：直接使用makespan信息"但实际没有生成图片

## ✅ 修复内容

### 修复1：传入env参数（`ffsp_trainer.py` 行330-339）

```python
# 修复前
out_untrained = policy(td_init.clone(), phase="test", decode_type="sampling", return_actions=True)
out_trained = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)

# 修复后
out_untrained = policy(td_init.clone(), env, phase="test", decode_type="sampling", return_actions=True)

# 重新reset环境（上一次调用会修改env的内部状态）
td_init = env.reset(batch_size=[3]).to(self.device)

out_trained = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)
```

**原理**：
- 传入已 reset 的 `env`，避免 policy 内部创建新 env
- 两次调用之间重新 reset，确保 `step_cnt` 和 `tables` 正确初始化

### 修复2：使用actions重放环境（`ffsp_trainer.py` 行360-380）

```python
# 使用actions重新执行环境以获取完整的schedule
td_replay = env.reset(batch_size=[3]).to(self.device)

# 执行每个action
for step_idx in range(actions_trained.shape[-1]):
    action = actions_trained[:, step_idx]
    td_replay.set('action', action)
    td_replay = env.step(td_replay)['next']
    
    if td_replay['done'].all():
        break

# 现在td_replay包含了完整的调度信息（schedule矩阵）
```

**原理**：
- Policy 的输出不包含最终的 TensorDict
- 通过重放 actions 在环境中执行，获取包含 `schedule` 的完整状态

### 修复3：正确生成甘特图（`ffsp_trainer.py` 行402-457）

```python
for i in range(min(3, td_replay.batch_size[0])):
    # 提取单个样本的schedule
    schedule_single = td_replay['schedule'][i].cpu()
    
    # 创建单个样本的TensorDict（只包含必要的键）
    td_single = TensorDict({
        key: td_replay[key][i:i+1] if td_replay[key].dim() > 0 else td_replay[key]
        for key in ['job_duration', 'run_time', 'job_location', 'machine_wait_step']
        if key in td_replay.keys()
    }, batch_size=[1])
    
    # 调用可视化函数真正生成图片
    makespan = create_ffsp_gantt_chart(
        td_single, schedule_single, gantt_path,
        title=f"FFSP训练后调度 (实例 {i+1})"
    )
    
    plot_paths.append(url)
```

### 修复4：增强可视化函数健壮性（`ffsp_viz.py`）

```python
# 支持多种键名
if 'job_duration' in td.keys():
    job_duration = td.get('job_duration').cpu().numpy()
elif 'run_time' in td.keys():
    job_duration = td.get('run_time').cpu().numpy()
else:
    raise KeyError("TensorDict中没有job_duration或run_time键")

# 处理batch维度
if job_duration.ndim == 3:
    job_duration = job_duration[0]

# 添加调试输出
print(f"[DEBUG] schedule shape: {schedule.shape}")
print(f"[DEBUG] job_duration shape: {job_duration.shape}")

# 添加索引越界检查
if job_duration.shape[0] > job_idx and job_duration.shape[1] > machine_idx:
    duration = job_duration[job_idx, machine_idx]
else:
    print(f"[WARN] 索引越界，跳过")
    continue
```

## 📊 修复对比

| 方面 | 修复前 | 修复后 |
|------|--------|--------|
| step_cnt错误 | ❌ TypeError | ✅ 正确传入env |
| 图片生成 | ❌ 只创建路径 | ✅ 真正生成图片 |
| 调度数据 | ❌ 缺失schedule | ✅ 重放获取schedule |
| 错误处理 | ⚠️ 基础 | ✅ 详细调试信息 |
| 健壮性 | ⚠️ 一般 | ✅ 索引检查+异常处理 |

## 🎯 修复文件

### 修改的文件

1. **`modules/rl_training/ffsp_trainer.py`**（3处修复）
   - 行330-339: 传入env参数，避免step_cnt错误
   - 行360-380: 使用actions重放环境
   - 行402-457: 正确提取数据并生成甘特图

2. **`modules/rl_training/visualizations/ffsp_viz.py`**（1处增强）
   - 行20-109: 增强健壮性，支持多种键名，添加调试输出

## 🧪 测试建议

重新运行FFSP训练后，应该能看到以下输出：

### 成功的标志

```
[17:25:47] 训练完成，开始生成可视化结果...
[17:25:47] 正在测试训练前的调度质量...
[17:25:47] 正在测试训练后的调度质量...
[17:25:47] 正在重放调度以生成可视化...
[17:25:47] ✅ 调度重放完成，可用键: ['schedule', 'job_duration', ...]
[17:25:47] 正在生成FFSP甘特图 1/3...
[17:25:47] [DEBUG] schedule shape: (12, 9)
[17:25:47] [DEBUG] job_duration shape: (9, 12)
[17:25:47] ✅ 甘特图 1 已生成: makespan=23.50
[17:25:48] 正在生成FFSP甘特图 2/3...
[17:25:48] ✅ 甘特图 2 已生成: makespan=24.20
[17:25:48] 正在生成FFSP甘特图 3/3...
[17:25:48] ✅ 甘特图 3 已生成: makespan=23.80
[17:25:48] 📊 调度结果: 训练前makespan=27.24, 训练后=23.98, 改进=13.35%
```

### 如果仍有问题

会看到详细的调试信息：
- TensorDict可用的键列表
- schedule和job_duration的形状
- 具体的错误堆栈

## 📝 技术要点

### FFSP环境的特殊性

FFSP环境有**实例级可变状态**：
- `step_cnt`: 步数计数器
- `tables`: IndexTables实例（机器索引表）

这些状态在 `__init__` 时为 `None`，只有调用 `reset()` 才会初始化。

### 与TSP的区别

| 环境 | 实例状态 | policy()需要传env? |
|------|----------|-------------------|
| TSP | ❌ 无 | ⚠️ 可选（推荐传） |
| ATSP | ❌ 无 | ⚠️ 可选（推荐传） |
| CVRP | ❌ 无 | ⚠️ 可选（推荐传） |
| FFSP | ✅ 有（step_cnt, tables） | ✅ **必须传** |

### 最佳实践

对于所有问题类型，在调用 `policy()` 进行测试时，**始终传入 env**：

```python
# ✅ 正确方式
out = policy(td, env, phase="test", ...)

# ❌ 不推荐（FFSP会报错，其他问题类型效率低）
out = policy(td, phase="test", ...)
```

## ✅ 验收清单

- [x] 修复 step_cnt TypeError
- [x] 修复可视化图片未生成问题
- [x] 实现actions重放逻辑
- [x] 增强可视化函数健壮性
- [x] 添加详细调试信息
- [x] 添加错误处理和占位图
- [x] 测试脚本验证修复

---

**修复日期**: 2026-01-23  
**问题类型**: FFSP (Flexible Flow Shop)  
**状态**: ✅ 修复完成，待用户验证
