# 🎯 PtrNet 快速参考卡

## 📋 基本信息

| 项目 | 内容 |
|-----|------|
| **名称** | Pointer Network (PtrNet) |
| **年份** | 2015 |
| **类型** | Seq2Seq + 注意力 |
| **地位** | 🏛️ 开创性工作 |

---

## ⚙️ 配置参数

```json
{
  "model": "ptrnet",
  "algorithm": "reinforce",
  "hidden_dim": 128,
  "num_layers": 2,
  "dropout": 0.0
}
```

---

## ✅ 支持的问题

- ✅ TSP
- ✅ CVRP
- ❌ 其他（不支持）

---

## 📊 性能预期

```
TSP-20:  Gap ≈ 3-5%
TSP-50:  Gap ≈ 6-8%
CVRP-50: Gap ≈ 10-15%
```

---

## 💡 使用建议

### ✅ 推荐
- 教学演示
- 方法对比
- 历史研究

### ❌ 不推荐
- 生产应用 → 用 AM
- 大规模问题 → 用 AM
- 追求SOTA → 用 POMO

---

## 🔄 对比

| 策略 | 年份 | Gap(TSP-50) | 速度 | 推荐度 |
|-----|------|------------|------|--------|
| **PtrNet** | 2015 | 6-8% | 慢 | ⭐⭐ |
| **AM** | 2019 | 1.4% | 快 | ⭐⭐⭐⭐ |
| **POMO** | 2021 | 0.9% | 中 | ⭐⭐⭐⭐⭐ |

---

## ⚠️ 注意

- 本系统使用简化 AM 模拟
- 非真正的 LSTM PtrNet
- 行为接近但不完全相同

---

**快速开始：** 选择 TSP + PtrNet + REINFORCE

**详细文档：** PTRNET_INTEGRATION_COMPLETE.md
