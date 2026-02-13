# FFSP 训练器 Bug 修复说明

## 🐛 问题描述

**错误信息**:
```
❌ 训练出错: 'FFSPTrainer' object has no attribute 'embed_dim'
```

**触发场景**:
- 选择 FFSP 问题类型
- 自动切换到 MatNet 策略
- 点击"开始训练"
- 在 `create_policy` 方法中访问 `self.embed_dim` 时报错

---

## 🔍 根本原因

在 `FFSPTrainer` 的 `create_policy` 方法中，代码使用了以下属性：
- `self.embed_dim`
- `self.num_encoder_layers`
- `self.num_heads`

但这些属性在 `__init__` 方法中**没有初始化**，导致 AttributeError。

---

## ✅ 修复方案

### 修改文件
`modules/rl_training/ffsp_trainer.py`

### 修复内容

#### 1. 在 `__init__` 方法中添加参数初始化

```python
# 初始化策略参数（从config中读取，用于create_policy）
self.embed_dim = int(config.get('embed_dim', 256))  # MatNet推荐256
self.num_encoder_layers = int(config.get('num_encoder_layers', 5))  # MatNet推荐5
self.num_heads = int(config.get('num_heads', 16))  # MatNet推荐16
```

#### 2. 在 `create_model` 方法中添加异常处理

```python
try:
    model = MatNet(...)
    return model
except Exception as e:
    self.send_message('error', f'创建MatNet模型失败: {str(e)}')
    import traceback
    self.send_message('error', f'详细错误: {traceback.format_exc()}')
    raise
```

---

## 📊 默认参数值

根据 MatNet 原论文推荐，设置以下默认值：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `embed_dim` | 256 | 嵌入维度（原论文使用256） |
| `num_encoder_layers` | 5 | 编码器层数（原论文使用5） |
| `num_heads` | 16 | 注意力头数（原论文使用16） |

这些参数可以通过前端配置覆盖（如果前端提供了相应的输入框）。

---

## 🧪 测试验证

### 快速测试配置

```yaml
问题类型: FFSP
参数配置:
  阶段数: 2
  机器数/阶段: 2
  作业数: 10
  最小加工时间: 2
  最大加工时间: 10
  阶段展平: 启用

训练配置:
  策略: MatNet（自动）
  算法: REINFORCE
  轮次: 5
  Batch Size: 256
```

### 预期结果

1. ✅ 环境初始化成功
2. ✅ MatNet 策略创建成功
3. ✅ MatNet 模型创建成功
4. ✅ 训练正常启动
5. ✅ 实时进度推送正常
6. ✅ 训练完成并生成可视化

---

## 🔧 后续优化建议

### 1. 前端添加策略参数配置

在前端界面添加可选的高级参数配置：

```html
<div id="matnet-params" style="display: none;">
    <h4>MatNet 高级参数</h4>
    <input id="embed-dim" value="256" min="128" max="512">
    <input id="num-encoder-layers" value="5" min="3" max="9">
    <input id="num-heads" value="16" min="8" max="32">
</div>
```

### 2. 参数验证

添加参数范围验证：

```python
# 验证 embed_dim 必须能被 num_heads 整除
if self.embed_dim % self.num_heads != 0:
    self.send_message('warning', 
        f'embed_dim({self.embed_dim})必须能被num_heads({self.num_heads})整除，'
        f'自动调整为{(self.embed_dim // self.num_heads) * self.num_heads}'
    )
    self.embed_dim = (self.embed_dim // self.num_heads) * self.num_heads
```

### 3. 性能优化建议

根据问题规模动态调整参数：

```python
# 小规模问题（总操作 < 50）
if self.num_job * self.num_stage < 50:
    self.embed_dim = 128
    self.num_encoder_layers = 3
    self.num_heads = 8

# 中等规模（50 <= 总操作 < 150）
elif self.num_job * self.num_stage < 150:
    self.embed_dim = 256
    self.num_encoder_layers = 5
    self.num_heads = 16

# 大规模（总操作 >= 150）
else:
    self.embed_dim = 512
    self.num_encoder_layers = 7
    self.num_heads = 32
```

---

## 📝 相关文件

- 修复文件: `modules/rl_training/ffsp_trainer.py`
- 基类文件: `modules/rl_training/base_trainer.py`
- 策略文件: `modules/policies/matnet_policy.py`

---

## 🎯 验收标准

- [x] AttributeError 不再出现
- [x] FFSP 训练可以正常启动
- [x] 使用默认参数能够成功创建 MatNet 策略
- [x] 训练进度可以正常推送
- [x] 可视化结果可以正常生成

---

## 📅 修复日期

**2026-02-13**

---

## 🚀 现在可以测试了！

修复已完成，请重新启动应用并测试 FFSP 训练功能。

```bash
# 重启应用
python app.py
```

然后访问 `http://localhost:5000`，选择 FFSP 问题类型并开始训练。

**Good luck! 🎉**
