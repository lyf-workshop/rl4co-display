# RL4CO Display

<div align="center">

**基于深度强化学习的组合优化问题可视化训练平台**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com)
[![RL4CO](https://img.shields.io/badge/RL4CO-0.4.0+-orange.svg)](https://github.com/ai4co/rl4co)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[功能特性](#功能特性) • [快速开始](#快速开始) • [使用指南](#使用指南) • [文档](#文档) • [开发团队](#开发团队)

</div>

---

## 项目简介

RL4CO Display 是一个基于 Web 的强化学习训练平台，专注于组合优化问题（如 TSP、VRP 等）的可视化训练与结果展示。平台集成了多种先进的深度强化学习算法，提供友好的交互界面和完整的训练流程管理。

**适用于：**
- 🎓 强化学习课程教学演示
- 🔬 组合优化问题研究
- 📊 算法性能对比分析
- 💼 工业界路径优化应用

---

## 功能特性

### 🚀 核心功能

#### 1. **多模型支持**
- Attention Model (AM)
- POMO (Policy Optimization with Multiple Optima)
- SymNCO (Symmetric Neural Combinatorial Optimization)
- MatNet (Matrix Network)
- MDAM (Multi-Decoder Attention Model)
- DeepACO (Deep Ant Colony Optimization)

#### 2. **问题类型**
- ✅ TSP (Traveling Salesman Problem) - 已支持
- 🔜 CVRP (Capacitated Vehicle Routing Problem)
- 🔜 SDVRP (Split Delivery VRP)
- 🔜 PCTSP (Prize Collecting TSP)
- 🔜 OP (Orienteering Problem)

#### 3. **训练管理**
- 实时训练进度跟踪
- 动态训练曲线可视化
- 训练历史记录查询
- 模型检查点管理
- 支持自定义数据集上传

#### 4. **可视化展示**
- TSP 路径动态生成 GIF
- 训练曲线实时更新
- 算法性能对比图表
- 路径优化前后对比

#### 5. **数据管理**
- 数据集上传与管理
- 训练文件分类存储
- 模型检查点备份
- 支持多种数据格式（TXT、JSON、TSPLIB）

#### 6. **用户系统**
- 用户注册与登录
- 个人训练记录
- 文件隔离管理
- 权限控制

---

## 技术栈

### 后端
- **框架**: Flask 3.0.0
- **数据库**: MySQL
- **深度学习**: PyTorch 2.0+
- **强化学习**: RL4CO 0.4.0+
- **训练框架**: PyTorch Lightning 2.0+

### 前端
- **基础**: HTML5 + CSS3 + JavaScript
- **可视化**: Matplotlib
- **样式**: 原生 CSS（渐变主题）

### 部署
- **本地开发**: Python venv
- **生产环境**: Gunicorn + Nginx
- **容器化**: Docker 支持

---

## 快速开始

### 系统要求

- **操作系统**: Windows / Linux / macOS
- **Python**: 3.8+
- **MySQL**: 5.7+ 或 8.0+
- **内存**: 建议 8GB+
- **GPU**: 可选（加速训练）

### 本地开发部署

#### 1. 解压并打开项目

```bash
cd rl4co-display
```

#### 2. 创建虚拟环境

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 配置数据库

编辑 `config.py` 文件，设置数据库连接信息：

```python
MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'your_password'  # 修改为你的密码
MYSQL_DB = 'rl4co_display'
```

#### 5. 初始化数据库

```bash
mysql -u root -p < database_init_with_auth.sql
```

#### 6. 启动应用

```bash
python app.py
```

访问 `http://localhost:5000` 即可使用！

---

## 使用指南

### 1️⃣ 注册与登录

首次使用需要注册账号：
- 访问 `/register` 页面
- 填写用户名、邮箱、密码
- 登录后即可开始训练

### 2️⃣ 开始训练

1. **选择模型**: 从下拉菜单选择 RL 模型（如 POMO）
2. **选择问题**: 选择要解决的问题类型（如 TSP）
3. **配置参数**:
   - 训练轮数（Epochs）
   - 批次大小（Batch Size）
   - 学习率（Learning Rate）
4. **数据集选择**:
   - 使用随机生成数据集
   - 或上传自定义数据集
5. **开始训练**: 点击"开始训练"按钮

### 3️⃣ 查看结果

训练完成后可以查看：
- ✅ 训练曲线（Loss & Reward）
- ✅ 路径可视化（静态图）
- ✅ 动态路线生成过程（GIF）
- ✅ 最优解与路径长度

### 4️⃣ 算法对比

访问 `/benchmark` 页面：
- 对比不同算法性能
- 查看收敛速度
- 分析算法优劣

### 5️⃣ 文件管理

访问 `/file_manager` 页面：
- 查看所有训练文件
- 按训练分组管理
- 下载检查点文件
- 删除过时数据

---

## 项目结构

```
rl4co-display/
├── app.py                      # Flask 主应用
├── auth_module.py              # 用户认证模块
├── config.py                   # 配置文件
├── model_database.py           # 模型数据库
├── requirements.txt            # Python 依赖
├── database_init_with_auth.sql # 数据库初始化脚本
│
├── modules/                    # 功能模块
│   └── rl_training/           # RL 训练模块
│       ├── trainer.py
│       └── visualizer.py
│
├── static/                     # 静态资源
│   ├── css/                   # 样式文件
│   ├── js/                    # JavaScript
│   ├── img/                   # 图片资源
│   └── model_plots/           # 生成的可视化图表
│
├── templates/                  # HTML 模板
│   ├── index.html             # 训练首页
│   ├── benchmark.html         # 算法对比
│   ├── model_info.html        # 模型知识库
│   ├── file_manager.html      # 文件管理
│   ├── profile.html           # 个人中心
│   └── login.html             # 登录页面
│
├── checkpoints/               # 模型检查点（按用户分组）
├── datasets/                  # 数据集文件（按用户分组）
├── lightning_logs/            # 训练日志
│
└── docs/                      # 文档
    ├── 快速部署参考.md
    ├── 部署指南.md
    ├── TSP路线生成完整指南.md
    ├── 数据集上传功能使用指南.md
    ├── 文件管理功能完整指南.md
    ├── 模型知识库功能完整指南.md
    ├── 算法对比页面功能完整指南.md
    ├── 性能优化指南.md
    └── 数据库清理完整指南.md
```

---

## 文档

### 📘 用户文档
- [快速部署参考](docs/快速部署参考.md) - 各种部署方式的完整命令
- [部署指南](docs/部署指南.md) - 详细的部署说明

### 📗 功能指南
- [TSP路线生成完整指南](docs/TSP路线生成完整指南.md)
- [数据集上传功能使用指南](docs/数据集上传功能使用指南.md)
- [文件管理功能完整指南](docs/文件管理功能完整指南.md)
- [模型知识库功能完整指南](docs/模型知识库功能完整指南.md)
- [算法对比页面功能完整指南](docs/算法对比页面功能完整指南.md)

### 📕 运维文档
- [性能优化指南](docs/性能优化指南.md)
- [数据库清理完整指南](docs/数据库清理完整指南.md)

---

## 部署方案

### 开发环境（本地）

适用于开发调试和功能测试。

```bash
python app.py
# 访问 http://localhost:5000
```

### 生产环境（服务器）

#### 方案 1: Gunicorn + Nginx

**优点**: 高性能、稳定、易维护

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动服务（4个工作进程）
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# 配置 Nginx 反向代理
# 参考 docs/部署指南.md
```

#### 方案 2: Docker 容器化

**优点**: 环境隔离、一键部署

```bash
# 构建镜像
docker build -t rl4co-display .

# 运行容器
docker run -d -p 5000:5000 \
  -e MYSQL_HOST=your_mysql_host \
  -e MYSQL_PASSWORD=your_password \
  --name rl4co-display \
  rl4co-display
```

详细部署步骤请参考 [快速部署参考](docs/快速部署参考.md)。

---

## 常见问题

### Q1: 训练时提示找不到 RL4CO 模块？

**A**: 确保已正确安装依赖：
```bash
pip install rl4co>=0.4.0
```

### Q2: 数据库连接失败？

**A**: 检查以下几点：
1. MySQL 服务是否启动
2. `config.py` 中的密码是否正确
3. 数据库 `rl4co_display` 是否已创建
4. 用户权限是否足够

### Q3: 训练很慢怎么办？

**A**: 优化建议：
1. 减小 batch_size
2. 减少训练 epochs
3. 使用 GPU 加速（需安装 CUDA 版本的 PyTorch）
4. 参考 [性能优化指南](docs/性能优化指南.md)

### Q4: 如何清理训练数据？

**A**: 参考 [数据库清理完整指南](docs/数据库清理完整指南.md)

### Q5: 支持哪些数据集格式？

**A**: 支持三种格式：
- **TXT**: 每行一个坐标对 `x y`
- **JSON**: `{"coordinates": [[x1,y1], [x2,y2], ...]}`
- **TSP**: TSPLIB 标准格式

---

## 更新日志

### v1.0.0 (2025-01-15)
- ✨ 首次发布
- ✅ 支持 6 种 RL 模型
- ✅ TSP 问题完整训练流程
- ✅ 用户系统与权限管理
- ✅ 文件管理与数据集上传
- ✅ 算法对比可视化
- ✅ 模型知识库

---

## 开发团队

**山西大学 计算机科学与技术学院**

- 项目负责人: [待补充]
- 开发团队: [待补充]
- 技术指导: [待补充]

---

## 致谢

本项目基于以下开源项目：

- [RL4CO](https://github.com/ai4co/rl4co) - 强化学习组合优化库
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [PyTorch Lightning](https://lightning.ai/) - 训练框架
- [Flask](https://flask.palletsprojects.com/) - Web 框架

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 联系方式

- **问题反馈**: [GitHub Issues](https://github.com/your-repo/rl4co-display/issues)
- **邮箱**: [your-email@example.com]
- **文档**: [docs/](docs/)

---

## Star History

如果这个项目对你有帮助，欢迎 Star ⭐️

---

<div align="center">

**🎓 山西大学 · RL4CO Display**

*让强化学习训练更简单、更直观*

[⬆ 回到顶部](#rl4co-display)

</div>
