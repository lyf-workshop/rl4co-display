# 文档中心

## 📚 文档索引

欢迎来到 RL4CO Display 文档中心！这里包含了项目的所有技术文档。

---

## 🎯 快速导航

### 新手入门
- 📖 [快速开始指南](../README.md) - 5分钟快速上手
- 📖 [mTSP 快速开始](../MTSP_QUICKSTART.md) - mTSP 问题快速开始

### 开发文档
- 📖 [**API 接口协议**](./API_PROTOCOL.md) ⭐ - 前后端接口完整说明
- 📖 [**API 使用示例**](./API_EXAMPLES.md) ⭐ - 实际代码示例
- 📖 [**添加新问题类型指南**](./ADD_NEW_PROBLEM_TYPE_GUIDE.md) ⭐ - 完整的开发指南
- 📖 [架构说明](./ARCHITECTURE.md) - 系统架构设计

### 模块文档
- 📖 [Modules 架构说明](../modules/README.md) - 模块职责划分
- 📖 [问题兼容性说明](../modules/PROBLEM_COMPATIBILITY.md) - 策略和算法支持
- 📖 [完整兼容性矩阵](../modules/COMPATIBILITY_MATRIX.md) - 详细兼容性规则
- 📖 [算法和策略指南](../modules/ALGORITHM_AND_POLICY_GUIDE.md) - 算法策略使用

### 问题使用指南
- 📖 [mTSP 使用指南](../modules/problems/MTSP_GUIDE.md) - 多旅行商问题
- 📖 [PDP 使用指南](./PDP_USER_GUIDE.md) ⭐新增 - 取送货问题
- 📖 [OP 使用指南](./OP_USER_GUIDE.md) ⭐新增 - 定向问题（推荐）
- 📖 [更多问题指南](../modules/problems/) - 其他问题类型

---

## 📁 文档分类

### 接口文档（API）

| 文档名称 | 描述 | 页数 | 更新时间 |
|---------|------|------|---------|
| [API_PROTOCOL.md](./API_PROTOCOL.md) | 前后端接口协议完整说明 | 1532行 | 2026-02-04 |
| [API_EXAMPLES.md](./API_EXAMPLES.md) | 接口使用示例代码 | 600+行 | 2026-02-04 |

**包含内容**:
- ✅ 所有 REST API 端点
- ✅ SSE 实时消息格式
- ✅ 数据结构定义
- ✅ 错误处理规范
- ✅ 完整的前后端示例代码
- ✅ curl 和 Python requests 测试示例

---

### 开发指南（Development）

| 文档名称 | 描述 | 页数 | 更新时间 |
|---------|------|------|---------|
| [ADD_NEW_PROBLEM_TYPE_GUIDE.md](./ADD_NEW_PROBLEM_TYPE_GUIDE.md) | 添加新问题类型完整指南 | 1278行 | 2026-02-04 |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 系统架构设计文档 | - | - |

**包含内容**:
- ✅ 6 大步骤详细说明
- ✅ 11 个常见错误预防
- ✅ 完整的代码模板
- ✅ mTSP 实际示例
- ✅ 文件清单和检查表

---

### 模块文档（Modules）

| 文档名称 | 描述 | 路径 |
|---------|------|------|
| [Modules README](../modules/README.md) | 模块架构总览 | `/modules/` |
| [问题兼容性](../modules/PROBLEM_COMPATIBILITY.md) | 各问题支持的策略和算法 | `/modules/` |
| [兼容性矩阵](../modules/COMPATIBILITY_MATRIX.md) | 详细兼容性规则 | `/modules/` |
| [算法策略指南](../modules/ALGORITHM_AND_POLICY_GUIDE.md) | 算法和策略使用说明 | `/modules/` |

---

### 问题指南（Problems）

| 问题类型 | 文档路径 | 状态 |
|---------|---------|------|
| mTSP | [modules/problems/MTSP_GUIDE.md](../modules/problems/MTSP_GUIDE.md) | ✅ |
| PDP | [docs/PDP_USER_GUIDE.md](./PDP_USER_GUIDE.md) | ✅ ⭐新增 |
| OP | [docs/OP_USER_GUIDE.md](./OP_USER_GUIDE.md) | ✅ ⭐推荐 |
| TSP | - | 待补充 |
| CVRP | - | 待补充 |
| VRPTW | - | 待补充 |

---

## 🔍 按需查找

### 我想了解...

**如何使用 API？**
→ 阅读 [API_PROTOCOL.md](./API_PROTOCOL.md) 和 [API_EXAMPLES.md](./API_EXAMPLES.md)

**如何添加新问题类型？**
→ 阅读 [ADD_NEW_PROBLEM_TYPE_GUIDE.md](./ADD_NEW_PROBLEM_TYPE_GUIDE.md)

**为什么某些策略和算法不能组合？**
→ 阅读 [问题兼容性说明](../modules/PROBLEM_COMPATIBILITY.md)

**系统架构是怎样的？**
→ 阅读 [Modules 架构说明](../modules/README.md)

**如何使用 mTSP？**
→ 阅读 [mTSP 使用指南](../modules/problems/MTSP_GUIDE.md) 或 [快速开始](../MTSP_QUICKSTART.md)

**如何使用 PDP？**
→ 阅读 [PDP 使用指南](./PDP_USER_GUIDE.md)

**如何使用 OP？**
→ 阅读 [OP 使用指南](./OP_USER_GUIDE.md)（推荐：旅游规划、无人机巡检）


**遇到错误怎么办？**
→ 查看 [常见错误](./ADD_NEW_PROBLEM_TYPE_GUIDE.md#常见错误) 章节

---

## 📖 推荐阅读路径

### 路径 1: 前端开发者

1. [API_PROTOCOL.md](./API_PROTOCOL.md) - 了解所有接口
2. [API_EXAMPLES.md](./API_EXAMPLES.md) - 查看使用示例
3. [问题兼容性说明](../modules/PROBLEM_COMPATIBILITY.md) - 理解业务规则

### 路径 2: 后端开发者

1. [Modules 架构说明](../modules/README.md) - 理解模块结构
2. [ADD_NEW_PROBLEM_TYPE_GUIDE.md](./ADD_NEW_PROBLEM_TYPE_GUIDE.md) - 学习如何扩展
3. [API_PROTOCOL.md](./API_PROTOCOL.md) - 了解接口规范

### 路径 3: 用户/研究人员

1. [快速开始](../README.md) - 快速上手
2. [mTSP 快速开始](../MTSP_QUICKSTART.md) - mTSP 使用
3. [问题兼容性说明](../modules/PROBLEM_COMPATIBILITY.md) - 选择合适的配置

---

## 🆕 最新文档（2026-02-04）

| 文档 | 描述 | 重要性 |
|------|------|--------|
| [API_PROTOCOL.md](./API_PROTOCOL.md) | 前后端接口协议（1532行） | ⭐⭐⭐⭐⭐ |
| [API_EXAMPLES.md](./API_EXAMPLES.md) | 接口使用示例（600+行） | ⭐⭐⭐⭐⭐ |
| [ADD_NEW_PROBLEM_TYPE_GUIDE.md](./ADD_NEW_PROBLEM_TYPE_GUIDE.md) | 添加新问题指南（1278行） | ⭐⭐⭐⭐⭐ |

---

## 📊 文档统计

| 分类 | 数量 | 总行数 |
|------|------|--------|
| 接口文档 | 2 | 2100+ 行 |
| 开发指南 | 1 | 1278 行 |
| 模块文档 | 4 | 1600+ 行 |
| 问题指南 | 1 | 300+ 行 |
| **总计** | **8+** | **5300+ 行** |

---

## 🔗 外部资源

- [RL4CO 官方文档](https://github.com/ai4co/rl4co)
- [PyTorch Lightning 文档](https://lightning.ai/docs/pytorch/stable/)
- [Flask 官方文档](https://flask.palletsprojects.com/)
- [Server-Sent Events (SSE) 规范](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)

---

## 💬 贡献和反馈

如果您发现文档有误或需要补充，请：
1. 提交 Issue 说明问题
2. 提交 Pull Request 改进文档
3. 联系维护团队

---

**文档中心维护者**: RL4CO Display Team  
**最后更新**: 2026-02-04  
**状态**: ✅ 活跃维护
