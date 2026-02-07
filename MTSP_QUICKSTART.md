# mTSP 快速开始指南

## 🚀 5分钟上手 mTSP

### 什么是 mTSP？

**mTSP (Multiple Traveling Salesman Problem)** = 多个旅行商协同完成所有城市的访问

简单来说：
- 🚗 有多个车辆/代理
- 📍 从同一起点（depot）出发
- 🎯 访问所有城市
- 🔄 最后都返回起点
- ⚖️ 目标：均衡负载或最小化总成本

---

## 📦 快速使用

### 方式 1: 通过 Web 界面（推荐）

1. **启动应用**
   ```bash
   bash scripts/start.sh
   ```

2. **配置训练**
   - 访问 http://localhost:5000
   - 选择问题类型: `mTSP`
   - 设置参数:
     - 城市数量: `50`
     - 代理数量: `5`
     - 优化目标: `minmax` 或 `sum`
   - 点击"开始训练"

3. **查看结果**
   - 实时查看训练进度
   - 下载可视化动画和对比图

### 方式 2: 通过 Python 代码

```python
from modules.problems import get_problem_class

# 1. 创建问题
config = {
    'problem_type': 'mtsp',
    'num_loc': 50,        # 50个城市
    'num_agents': 5,      # 5个代理
    'cost_type': 'minmax', # 最小化最大路径
    'batch_size': 512,
    'epochs': 100,
}

problem_class = get_problem_class('mtsp')
problem = problem_class(config)

# 2. 创建环境
env = problem.create_environment()

# 3. 训练模型（这里省略训练代码）
# ...

# 4. 生成可视化
viz_funcs = problem.get_visualization_functions()
viz_funcs['animation'](td, actions, 'mtsp_animation.gif')
viz_funcs['comparison'](td, actions, 'mtsp_result.png', cost)
```

---

## 🎯 两种优化目标

### Minmax - 均衡负载

**适合场景**：团队协作、多车配送

```python
config = {
    'cost_type': 'minmax',  # 最小化最长路径
}
```

**效果**：
- ✅ 各代理工作量相对均衡
- ✅ 整体完成时间最短
- ✅ 避免单个代理过载

**示例**：
```
Agent 1: 10 个城市，距离 8.5
Agent 2: 11 个城市，距离 9.2  ← 最长
Agent 3: 10 个城市，距离 8.8
→ Cost = 9.2 (最大值)
```

### Sum - 最小化总成本

**适合场景**：成本优先、资源有限

```python
config = {
    'cost_type': 'sum',  # 最小化总路径
}
```

**效果**：
- ✅ 总体成本最低
- ✅ 资源利用最优
- ⚠️ 可能存在负载不均

**示例**：
```
Agent 1: 15 个城市，距离 12.0
Agent 2: 8 个城市，距离 6.5
Agent 3: 7 个城市，距离 5.8
→ Cost = 24.3 (总和)
```

---

## ⚙️ 参数配置建议

### 代理数量推荐

```python
# 经验公式
num_agents = num_loc // 10  # 每10个城市1个代理

# 具体建议
if num_loc <= 20:
    num_agents = 2-3
elif num_loc <= 50:
    num_agents = 3-5
elif num_loc <= 100:
    num_agents = 5-10
else:
    num_agents = 10-20
```

### 训练参数推荐

```python
# 小规模 (20城市)
config = {
    'num_loc': 20,
    'num_agents': 3,
    'batch_size': 1024,
    'epochs': 100,
}

# 中等规模 (50城市)
config = {
    'num_loc': 50,
    'num_agents': 5,
    'batch_size': 512,
    'epochs': 150,
}

# 大规模 (100城市)
config = {
    'num_loc': 100,
    'num_agents': 10,
    'batch_size': 256,
    'epochs': 200,
}
```

---

## 📊 可视化示例

### 1. 路线动画 GIF

生成逐步构建路径的动画：

```python
viz_funcs['animation'](
    td=test_data,
    actions=solution,
    save_path='mtsp_animation.gif',
    title='mTSP路线生成过程',
    fps=2  # 每秒2帧
)
```

**效果**：
- 🎨 不同代理用不同颜色
- ➡️ 显示行进方向箭头
- 📍 高亮当前访问城市
- 📊 实时显示距离统计

### 2. 对比图 PNG

生成最终路径对比图：

```python
viz_funcs['comparison'](
    td=test_data,
    actions=solution,
    save_path='mtsp_comparison.png',
    cost=final_cost,
    title='mTSP最终路线对比'
)
```

**效果**：
- 🗺️ 全局路径展示
- 📏 各代理距离统计
- 📈 负载均衡分析
- 🎯 优化成本显示

---

## ⚠️ 常见问题

### Q1: 代理数量怎么选择？

**A**: 一般建议每个代理负责 5-20 个城市

```python
# 太少：单个代理负载过重
num_agents = 2  # for 100 cities ❌

# 合适：负载均衡
num_agents = 8  # for 100 cities ✅

# 太多：协调开销大
num_agents = 50  # for 100 cities ❌
```

### Q2: minmax 和 sum 怎么选？

| 场景 | 推荐 |
|------|------|
| 多车配送，关注完成时间 | minmax |
| 无人机巡检，关注电量消耗 | sum |
| 团队任务，要求负载均衡 | minmax |
| 成本敏感，总费用重要 | sum |

### Q3: 训练不收敛怎么办？

**解决方案**：
1. 减少代理数量
2. 降低学习率
3. 增加训练轮数
4. 尝试不同的策略模型

```python
# 原配置
config = {'num_agents': 10, 'epochs': 100}

# 优化后
config = {'num_agents': 5, 'epochs': 200}
```

### Q4: 可视化时颜色不够用？

**A**: 系统支持 8 种颜色，超过会循环

```python
# 建议：代理数不超过 10
if num_agents > 10:
    print("警告：可视化可能不清晰")
```

---

## 🎓 学习资源

### 项目文档
- 📖 [完整使用指南](modules/problems/MTSP_GUIDE.md)
- 🔧 [兼容性矩阵](modules/COMPATIBILITY_MATRIX.md)
- 📝 [集成报告](MTSP_INTEGRATION_COMPLETE.md)

### RL4CO 官方
- 🌐 [mTSP Environment](https://rl4.co/docs/content/api/envs/routing/#multiple-traveling-salesman-problem-mtsp)
- 💻 [GitHub](https://github.com/ai4co/rl4co)

### 学术论文
- Bektas, T. (2006). "The multiple traveling salesman problem: an overview"

---

## 🎯 最佳实践

1. **从小开始**: 先用 20 城市 + 2-3 代理测试
2. **渐进增加**: 逐步增加规模和复杂度
3. **对比实验**: 测试 minmax vs sum 的效果
4. **监控指标**: 关注训练曲线和负载均衡
5. **保存检查点**: 定期保存训练进度

---

## 🔗 下一步

- ✅ 已完成基本配置？→ 查看 [完整使用指南](modules/problems/MTSP_GUIDE.md)
- 🎨 需要自定义可视化？→ 查看 `mtsp_viz.py` 代码
- 🐛 遇到问题？→ 查看 [故障排除](modules/problems/MTSP_GUIDE.md#故障排除)
- 💡 想了解原理？→ 查看 [RL4CO 文档](https://rl4.co)

---

**开始使用 mTSP 吧！** 🚀

有问题？查看完整文档或提 Issue。
