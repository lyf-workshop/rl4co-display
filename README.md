# RL4CO Display - 强化学习优化可视化平台

> 山西大学 计算机科学与技术学院  
> 基于 RL4CO 的交互式强化学习训练与可视化平台

---

## 📝 项目简介

RL4CO Display 是一个集成了真实 RL4CO（Reinforcement Learning for Combinatorial Optimization）强化学习框架的 Web 可视化平台，支持通过浏览器进行交互式模型训练、实时监控和结果可视化。

### 核心特性

- ✅ **多用户系统** - 完整的用户认证和数据隔离
- ✅ **真实训练** - 集成 RL4CO 库，支持多种强化学习算法
- ✅ **实时监控** - 实时显示训练进度、Loss 和 Reward 曲线
- ✅ **结果可视化** - 自动生成路径对比图和动态 GIF
- ✅ **多模型支持** - Attention Model、POMO、SymNCO 等
- ✅ **多问题类型** - TSP、CVRP、OP、PCTSP 等组合优化问题

---

## 🚀 本地部署（快速开始）

### 环境要求

- Python 3.8+
- MySQL 8.0+
- 8GB+ 内存
- GPU（可选，用于加速训练）

### 第 1 步：克隆项目

```bash
git clone https://github.com/your-repo/rl4co-display.git
cd rl4co-display
```

### 第 2 步：安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖包
pip install -r requirements.txt
```

主要依赖：Flask 3.0.0、PyTorch、RL4CO、PyTorch Lightning、Matplotlib

### 第 3 步：配置数据库

**1. 创建数据库**

```bash
mysql -u root -p
```

```sql
CREATE DATABASE flaskdemo_user CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EXIT;
```

**2. 初始化数据表**

```bash
mysql -u root -p flaskdemo_user < config/database_init_with_auth.sql
```

**3. 修改配置文件**

编辑 `config/config.py`：

```python
MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'your_password'  # 修改为你的密码
MYSQL_DB = 'flaskdemo_user'
```

### 第 4 步：启动应用

```bash
# 方式1: 使用快速启动脚本（推荐，macOS/Linux）
bash scripts/start.sh

# 方式2: 直接运行
python app.py
```

访问 `http://localhost:5000` 即可使用！

### 第 5 步：注册并开始训练

1. 访问 `http://localhost:5000/register` 注册新用户
2. 登录后即可开始训练模型
3. 选择模型（如 POMO）、问题类型（如 TSP）
4. 配置训练参数，点击"开始训练"
5. 实时查看训练进度和结果

---

## 📂 项目结构

```
rl4co-display/
├── app.py                    # 主应用入口（重构后约370行）
├── app_auth.py               # 认证路由模块
├── app_pages.py              # 页面路由模块
├── app_stats.py              # 统计API模块
├── app_compat.py             # 兼容性API模块
├── app_training.py           # 训练API模块
├── app_files.py              # 文件管理API模块
├── auth_module.py            # 用户认证核心模块
├── logging_config.py         # 日志配置（新增）
├── config.py                 # 应用配置
├── requirements.txt          # Python 生产依赖
├── requirements-dev.txt      # 开发和测试依赖（新增）
├── pytest.ini                # Pytest配置（新增）
├── database_init_with_auth.sql  # 数据库初始化脚本
│
├── modules/                  # 业务逻辑模块
│   ├── algorithms/           # 强化学习算法
│   ├── policies/             # 策略模型
│   ├── problems/             # 问题定义（TSP/CVRP/VRPTW/SDVRP等）
│   ├── rl_training/          # 训练核心逻辑
│   │   ├── training_functions.py
│   │   └── visualizations/   # 可视化模块
│   └── compatibility.py      # 配置兼容性检查
│
├── templates/                # HTML 模板
│   ├── index.html           # 训练主页面
│   ├── login.html           # 登录页面
│   ├── register.html        # 注册页面
│   └── ...
│
├── static/                   # 静态资源
│   ├── css/                 # 样式文件
│   │   ├── navigation.css
│   │   ├── layout.css       # 全局布局（新增）
│   │   └── training.css     # 训练样式（新增）
│   ├── js/                  # JavaScript
│   └── model_plots/         # 训练结果（自动生成）
│
├── tests/                    # 测试目录（新增）
│   ├── __init__.py
│   ├── conftest.py           # Pytest配置
│   ├── test_parse_dataset.py  # 数据集解析测试
│   └── test_compatibility_api.py  # 兼容性API测试
│
├── logs/                     # 日志文件（运行时生成）
├── checkpoints/              # 模型检查点（自动生成）
├── datasets/                 # 用户上传的数据集
├── lightning_logs/           # 训练日志（自动生成）
│
└── docs/                     # 完整文档
    ├── README.md            # 文档索引
    ├── ARCHITECTURE.md      # 系统架构文档（新增）
    └── ...
```

### 架构改进（2026-01重构）

本项目在 2026-01-19 完成了大规模重构，主要改进包括：

1. **模块化路由**：将 `app.py`（1644行）拆分为 6 个 Blueprint 模块
2. **日志系统**：引入统一的日志配置和错误码体系
3. **样式分离**：抽离 CSS 到独立文件（`layout.css`, `training.css`）
4. **测试框架**：添加 Pytest 单元测试和集成测试
5. **文档完善**：新增架构文档和重构进度报告

详细信息请查看：
- [系统架构文档](docs/ARCHITECTURE.md)
- [重构进度报告](REFACTORING_PROGRESS.md)

---

## 🎯 功能特性

### 支持的模型

- Attention Model (AM)
- POMO
- SymNCO
- MatNet
- MDAM
- DeepACO
- 等 20+ 种模型

### 支持的问题

- **TSP** - 旅行商问题
- **ATSP** - 非对称旅行商问题 ⭐新增
- **CVRP** - 带容量约束的车辆路径问题
- **SDVRP** - 分割配送车辆路径问题
- **VRPTW** - 带时间窗的车辆路径问题
- **PCTSP** - 带奖励的旅行商问题（计划中）
- **OP** - 定向问题（计划中）
- 等多种组合优化问题

### 可视化功能

- 实时训练曲线（Loss & Reward）
- 路径对比图（训练前后）
- 动态路线生成 GIF
- 算法性能对比
- 模型知识库

---

## 🐛 常见问题

### Q1: 数据库连接失败？

**检查项**：
1. MySQL 是否运行：`mysql -u root -p`
2. `config.py` 中密码是否正确
3. 数据库 `flaskdemo_user` 是否已创建

### Q2: RL4CO 未安装？

```bash
pip install rl4co
```

如果安装失败，系统会使用模拟训练模式（用于演示）。

### Q3: 训练速度慢？

**优化建议**：
- 使用 GPU（提升 10-50 倍）
- 减小批次大小（Batch Size）
- 减少训练轮数（Epochs）
- 使用更轻量的模型

### Q4: 端口 5000 被占用？

修改 `app.py` 中的端口：
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # 改为 5001
```

---

## 📚 完整文档

详细功能文档位于 `docs/` 目录：

- [文档索引](docs/README.md) - 所有文档的导航
- [TSP路线生成指南](docs/TSP路线生成完整指南.md)
- [实时训练曲线指南](docs/实时训练曲线功能完整指南.md)
- [文件管理指南](docs/文件管理功能完整指南.md)
- [算法对比指南](docs/算法对比页面功能完整指南.md)
- [模型知识库指南](docs/模型知识库功能完整指南.md)

---

## 🔧 技术栈

- **后端**: Flask 3.0、MySQL、PyTorch、RL4CO、PyTorch Lightning
- **前端**: HTML5/CSS3、JavaScript、ECharts
- **可视化**: Matplotlib
- **实时推送**: Server-Sent Events (SSE)

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [RL4CO](https://github.com/ai4co/rl4co) - 强化学习组合优化库
- [PyTorch Lightning](https://lightning.ai/) - 训练框架
- [Flask](https://flask.palletsprojects.com/) - Web 框架

---

## 📞 联系方式

- **项目文档**: [docs/README.md](docs/README.md)
- **开发单位**: 山西大学 计算机科学与技术学院

---

**开始你的强化学习之旅！** 🚀
