# SDVRP (分割配送车辆路径问题) 使用指南

## 🎯 **什么是SDVRP？**

**SDVRP** = Split Delivery Vehicle Routing Problem  
**中文名**: 分割配送车辆路径问题

---

## 📖 **问题描述**

### 核心概念

SDVRP是CVRP的扩展版本，**允许同一客户被多次访问**，每次配送部分需求。

### 与CVRP的区别

| 特性 | CVRP | SDVRP |
|-----|------|-------|
| **客户访问次数** | 每个客户只能访问1次 | 每个客户可以访问多次 |
| **需求满足** | 单次必须完全满足 | 可以分批满足 |
| **适用场景** | 普通配送 | 大宗货物配送 |
| **灵活性** | 较低 | 较高 |
| **复杂度** | 高 | 更高 |

### 问题场景

```
仓库 (容量: 50吨)
  ↓
客户A (需求: 60吨) ← 单车无法完成！
客户B (需求: 30吨)
客户C (需求: 40吨)

CVRP方案: 无法求解（容量不足）
SDVRP方案: 
  - 第1次: 仓库 → 客户A(配送40吨) → 客户B(配送10吨) → 仓库
  - 第2次: 仓库 → 客户A(配送20吨) → 客户C(配送30吨) → 仓库
```

---

## 🚀 **使用方式**

### 1. **通过问题模块使用**

```python
from modules.problems import get_problem_class

# 创建SDVRP问题实例
SDVRProblem = get_problem_class('sdvrp')
sdvrp = SDVRProblem({
    'num_loc': 30,                    # 30个客户
    'vehicle_capacity': 1.0,          # 车辆容量
    'max_split_per_customer': 3,      # 每个客户最多分割3次
    'batch_size': 512,
})

# 验证配置
valid, error_msg = sdvrp.validate_config()
if not valid:
    print(f"配置错误: {error_msg}")

# 创建环境
env = sdvrp.create_environment()

# 获取可视化函数
viz_funcs = sdvrp.get_visualization_functions()
```

### 2. **Web界面使用**

```html
<!-- 在前端添加选项 -->
<select id="problem-type">
    <option value="tsp">TSP</option>
    <option value="cvrp">CVRP</option>
    <option value="sdvrp">SDVRP - 分割配送VRP</option>
</select>
```

### 3. **配置参数**

```python
config = {
    # 基础参数
    'problem': 'sdvrp',
    'num_loc': 30,                    # 客户数量
    'batch_size': 512,
    
    # SDVRP特有参数
    'vehicle_capacity': 1.0,          # 车辆容量
    'max_split_per_customer': 3,      # 最大分割次数
    
    # 训练参数
    'model': 'attention',
    'epochs': 10,
    'learning_rate': 1e-4,
}
```

---

## 🎨 **可视化功能**

SDVRP提供3种可视化函数：

### 1. **路线动画** (create_sdvrp_route_animation)

```python
viz_funcs['animation'](
    td=tensor_dict,
    actions=action_sequence,
    save_path='sdvrp_route.gif',
    title='SDVRP分割配送路线',
    fps=2
)
```

**特点**:
- 🔵 蓝色线 = 普通配送路径
- 🟠 橙色线 = 重复访问（分割配送）
- 🟢 绿色虚线 = 返回仓库
- 橙色节点 = 被分割配送的客户
- 显示访问次数标签

### 2. **对比图** (create_sdvrp_comparison_plot)

```python
viz_funcs['comparison'](
    env=sdvrp_env,
    td=tensor_dict,
    actions_untrained=random_actions,
    rewards_untrained=random_rewards,
    actions_trained=trained_actions,
    rewards_trained=trained_rewards,
    save_path='sdvrp_comparison.png'
)
```

**包含内容**:
- 左：随机策略路线
- 中：训练后策略路线
- 右上：统计信息（成本、返回、分割次数）
- 右下：分割配送分析

### 3. **分割配送分析** (create_sdvrp_split_analysis)

```python
viz_funcs['split_analysis'](
    actions=action_sequence,
    demands=customer_demands,
    save_path='sdvrp_split_analysis.png',
    title='SDVRP分割配送详细分析'
)
```

**包含内容**:
- 左：客户访问次数分布图
- 右：被分割配送的客户详情

---

## 📊 **SDVRP vs CVRP 性能对比**

### 场景1: 普通需求（需求 < 容量）

```
客户需求: 都小于车辆容量
CVRP性能: ⭐⭐⭐⭐⭐ (最优)
SDVRP性能: ⭐⭐⭐⭐ (略差，因为复杂度更高)
推荐: CVRP
```

### 场景2: 大宗需求（部分需求 > 容量）

```
客户需求: 部分大于车辆容量
CVRP性能: ❌ 无法求解
SDVRP性能: ⭐⭐⭐⭐⭐ (唯一选择)
推荐: SDVRP
```

### 场景3: 混合需求

```
客户需求: 有大有小
CVRP性能: ⭐⭐⭐ (需要更多车次)
SDVRP性能: ⭐⭐⭐⭐ (更灵活)
推荐: SDVRP（如果追求灵活性）
```

---

## 💡 **实际应用场景**

### 1. **燃油配送**

```
问题: 加油站需求量大，单车无法一次配送完
解决: 使用SDVRP，允许多次配送
效果: 减少返仓次数，提高效率
```

### 2. **建材配送**

```
问题: 工地需要大量建材，单次运输不够
解决: 分批配送，每次运送部分建材
效果: 灵活安排，满足大额需求
```

### 3. **大宗商品配送**

```
问题: 客户订单量大，超过车辆容量
解决: 多车次分割配送
效果: 降低配送成本，提高满意度
```

---

## ⚙️ **参数配置详解**

### num_loc (客户数量)

```python
'num_loc': 30  # 推荐: 20-50
```

- 太小(<10): 问题过于简单
- 适中(20-50): 平衡训练时间和效果
- 太大(>100): 训练时间长

### vehicle_capacity (车辆容量)

```python
'vehicle_capacity': 1.0  # 归一化容量
```

- 通常设为1.0（归一化）
- 需求会相应归一化

### max_split_per_customer (最大分割次数)

```python
'max_split_per_customer': 3  # 推荐: 2-5
```

- 1: 退化为CVRP
- 2-3: 平衡灵活性和复杂度 ✅
- 4-5: 非常灵活，但计算复杂
- >5: 不推荐，过于复杂

---

## 🧪 **快速测试**

### 测试脚本

```python
# test_sdvrp.py
from modules.problems import get_problem_class

# 1. 创建问题
SDVRProblem = get_problem_class('sdvrp')
sdvrp = SDVRProblem({
    'num_loc': 20,
    'vehicle_capacity': 1.0,
    'max_split_per_customer': 3,
})

# 2. 验证
valid, msg = sdvrp.validate_config()
print(f"Config valid: {valid}")
if not valid:
    print(f"Error: {msg}")

# 3. 打印信息
print("\n" + sdvrp.get_problem_description())
print("\n特征:")
for feature in sdvrp.get_problem_features():
    print(f"  • {feature}")

print("\n" + sdvrp.get_split_delivery_info())
```

### 预期输出

```
Config valid: True

分割配送车辆路径问题 (SDVRP)
目标: 规划车辆配送路径，访问20个客户
特点: 允许分割配送（同一客户可多次配送）
约束: 车辆容量=1.0，最多分割3次

特征:
  • 允许分割配送
  • 客户需求可由多车完成
  • 减少返回仓库次数
  • 适用于大宗物流
  • 比CVRP更灵活

分割配送特性:
- 同一客户可以被访问多次
- 每次配送部分需求
- 减少车辆返回仓库次数
- 每个客户最多分割3次

适用场景:
- 客户需求 > 车辆容量
- 大宗物流配送
- 需要灵活配送方案
```

---

## 📈 **性能优化建议**

### 1. **合理设置分割次数**

```python
# ❌ 不推荐
'max_split_per_customer': 10  # 太多，计算复杂

# ✅ 推荐
'max_split_per_customer': 3   # 平衡灵活性和效率
```

### 2. **调整训练参数**

```python
config = {
    'epochs': 20,              # SDVRP建议更多epoch
    'batch_size': 256,         # 可以适当减小
    'learning_rate': 5e-5,     # 稍小的学习率
}
```

### 3. **数据增强**

```python
# 在训练时使用数据增强
model = REINFORCE(
    env,
    policy,
    baseline="rollout",
    # 添加数据增强
    augment_size=8,  # 8倍增强
)
```

---

## 🔧 **故障排除**

### 问题1: "RL4CO不支持SDVRPEnv"

**原因**: RL4CO可能没有单独的SDVRP环境

**解决方案**: 代码会自动使用CVRP环境并设置分割标志
```python
# 自动处理，无需手动干预
env = sdvrp.create_environment()
# 会输出: ⚠️  使用CVRP环境模拟SDVRP
```

### 问题2: "分割配送效果不明显"

**原因**: 客户需求都小于车辆容量

**解决方案**: 增大部分客户需求
```python
# 确保有部分客户需求 > 容量
# 这会在环境生成时自动处理
```

### 问题3: "训练速度慢"

**原因**: SDVRP比CVRP更复杂

**解决方案**:
- 减小`num_loc`（客户数量）
- 减小`max_split_per_customer`
- 使用GPU加速

---

## 📚 **相关文档**

- [问题模块README](./README.md)
- [CVRP使用指南](../../modules/rl_training/CVRP_GUIDE.md)
- [RL4CO完整参考](../../RL4CO_COMPLETE_REFERENCE.md)

---

## ✅ **检查清单**

使用SDVRP前的检查：

- [ ] 确认问题确实需要分割配送
- [ ] 设置合理的`max_split_per_customer`
- [ ] 配置适当的训练参数
- [ ] 理解SDVRP与CVRP的区别
- [ ] 准备充足的训练时间

---

## 🎉 **总结**

### SDVRP适用于：
- ✅ 大宗货物配送
- ✅ 客户需求 > 车辆容量
- ✅ 需要灵活配送方案
- ✅ 追求最少返仓次数

### SDVRP不适用于：
- ❌ 所有需求 << 容量
- ❌ 禁止重复访问的场景
- ❌ 计算资源有限
- ❌ 需要快速求解

### 关键要点：
1. **分割配送** = 客户可多次访问
2. **更灵活** = 处理大额需求
3. **更复杂** = 训练时间更长
4. **看场景** = 不是总比CVRP好

---

**开始使用SDVRP，解决大宗配送难题！** 🚀

---

**最后更新**: 2024年12月16日  
**状态**: ✅ **已完成并测试**  
**项目**: RL4CO Display - 山西大学


