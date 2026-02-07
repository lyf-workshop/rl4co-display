# 🚀 快速开始指南

> 项目已完成文件整理，现在结构清晰有序！

## 📂 新的目录结构

```
rl4co-display/
├── config/          ← 配置文件
├── scripts/         ← 启动脚本
├── docs/            ← 项目文档
├── tests/           ← 测试代码
├── data/            ← 数据目录
├── archive/         ← 归档文件
├── modules/         ← 核心模块
├── static/          ← 静态资源
└── templates/       ← HTML模板
```

## ⚡ 快速启动

### macOS/Linux

```bash
# 方式1: 使用启动脚本（推荐）
bash scripts/start.sh

# 方式2: 手动启动
source venv/bin/activate
python app.py
```

### Windows

```bash
# 激活虚拟环境
venv\Scripts\activate

# 启动应用
python app.py
```

## 📝 配置说明

### 数据库配置
编辑 `config/config.py`：

```python
class Config:
    MYSQL_HOST = 'localhost'
    MYSQL_USER = 'root'
    MYSQL_PASSWORD = '你的密码'  # ← 修改这里
    MYSQL_DB = 'flaskdemo_user'
```

### 数据库初始化

```bash
# 创建数据库
mysql -u root -p
CREATE DATABASE flaskdemo_user CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EXIT;

# 初始化表结构
mysql -u root -p flaskdemo_user < config/database_init_with_auth.sql
```

## 📚 文档导航

| 文档类型 | 位置 | 说明 |
|---------|------|------|
| **部署文档** | `docs/deployment/` | macOS/Linux/Windows部署指南 |
| **使用指南** | `docs/guides/` | 功能使用说明 |
| **开发文档** | `docs/development/` | 架构、重构记录 |
| **项目结构** | `PROJECT_STRUCTURE.md` | 详细目录说明 |

## 🧪 运行测试

```bash
# 运行所有测试
pytest

# 运行指定测试
pytest tests/unit/test_auth_功能测试.py

# 查看测试数据
ls tests/data/
```

## 📊 数据目录

```
data/
├── datasets/        ← 用户上传的数据集
├── checkpoints/     ← 训练模型检查点
└── logs/           ← 训练日志
```

## 🔍 常见问题

### Q: 找不到 config.py？
**A**: 配置文件已移动到 `config/config.py`

### Q: 数据库初始化脚本在哪？
**A**: `config/database_init_with_auth.sql`

### Q: 启动脚本在哪？
**A**: `scripts/start.sh`

### Q: 测试数据在哪？
**A**: `tests/data/`

## 🎯 主要改进

✅ 根目录文件减少 70%（50+ → 15）
✅ 文档分类清晰（3个子目录）
✅ 测试代码集中管理
✅ 数据目录统一规划
✅ 配置文件模块化

## 📖 延伸阅读

- [完整部署指南](docs/deployment/DEPLOYMENT_MACOS.md)
- [项目架构说明](docs/ARCHITECTURE.md)
- [整理完成报告](CLEANUP_COMPLETE.md)

---

**祝你使用愉快！** 🎉

有问题？查看 [README.md](README.md) 或文档目录 [docs/](docs/)
