# RL4CO Display - 项目结构说明

## 📁 目录结构

```
rl4co-display/
├── 📂 app/                          # 应用代码（预留，待迁移）
│   ├── blueprints/                 # Flask蓝图模块
│   ├── core/                       # 核心模块
│   └── utils/                      # 工具函数
│
├── 📂 config/                       # ✅ 配置文件
│   ├── __init__.py                 # 配置模块导出
│   ├── config.py                   # 应用配置
│   └── database_init_with_auth.sql # 数据库初始化脚本
│
├── 📂 scripts/                      # ✅ 工具脚本
│   ├── start.sh                    # 快速启动脚本
│   └── diagnose_sdvrp.py           # SDVRP诊断工具
│
├── 📂 docs/                         # ✅ 项目文档
│   ├── deployment/                 # 部署文档
│   │   ├── DEPLOYMENT.md           # 通用部署指南
│   │   ├── DEPLOYMENT_MACOS.md    # macOS部署指南
│   │   ├── INSTALL_FIX.md         # 安装问题修复
│   │   └── 部署完整指南.md        # 完整部署文档
│   ├── development/                # 开发文档
│   │   ├── CONFIG.md               # 配置文件说明
│   │   ├── REFACTORING_COMPLETE.md# 重构记录
│   │   ├── PROJECT_OPTIMIZATION_*  # 优化报告
│   │   └── *_COMPLETE.md          # 功能完成记录
│   ├── guides/                     # 使用指南
│   │   ├── TSP路线生成完整指南.md
│   │   ├── 性能优化指南.md
│   │   ├── 数据集上传功能使用指南.md
│   │   ├── 文件管理功能完整指南.md
│   │   ├── 模型知识库功能完整指南.md
│   │   └── 算法对比页面功能完整指南.md
│   ├── ARCHITECTURE.md             # 架构设计文档
│   └── README.md                   # 文档索引
│
├── 📂 tests/                        # ✅ 测试代码
│   ├── data/                       # 测试数据集
│   │   ├── 10cities*.txt           # 10城市测试数据
│   │   ├── test_dataset_*.txt     # 测试数据集
│   │   ├── test_dataset_*.json    # JSON测试数据
│   │   └── test_dataset_*.tsp     # TSPLIB格式数据
│   ├── unit/                       # 单元测试
│   │   ├── test_auth_功能测试.py
│   │   ├── test_cvrp_viz.py
│   │   └── test_import.py
│   ├── __init__.py
│   ├── conftest.py                 # pytest配置
│   ├── test_atsp_integration.py
│   ├── test_compatibility_api.py
│   └── test_parse_dataset.py
│
├── 📂 archive/                      # ✅ 归档文件
│   ├── ollama/                     # Ollama助手相关
│   │   ├── Ollama智能助手-文件清单.txt
│   │   ├── Ollama智能助手-部署完成.txt
│   │   └── Ollama智能助手使用指南.md
│   ├── templates_backup/          # 模板备份
│   ├── app_old_backup.py          # 旧版本备份
│   └── 项目整理完成.txt           # 历史标记文件
│
├── 📂 data/                         # ✅ 数据目录
│   ├── datasets/                   # 用户数据集
│   │   ├── user_4/                # 用户4的数据集
│   │   └── user_5/                # 用户5的数据集
│   ├── checkpoints/                # 模型检查点
│   └── logs/                       # 日志文件
│       └── lightning/              # Lightning训练日志
│           ├── version_0/
│           ├── version_1/
│           └── version_2/
│
├── 📂 modules/                      # 核心功能模块
│   ├── algorithms/                 # 强化学习算法
│   ├── policies/                   # 策略网络
│   ├── problems/                   # 问题定义
│   ├── envs/                       # 环境包装器
│   ├── rl_training/               # 训练功能
│   └── compatibility.py            # 兼容性检查
│
├── 📂 static/                       # 静态资源
│   ├── css/                        # 样式文件
│   ├── js/                         # JavaScript
│   └── model_plots/                # 训练结果可视化
│
├── 📂 templates/                    # HTML模板
│   ├── includes/                   # 公共组件
│   └── *.html                      # 页面模板
│
├── 📄 app.py                        # ✅ 主应用入口
├── 📄 app_*.py                      # Flask蓝图模块
├── 📄 auth_module.py                # 认证模块
├── 📄 logging_config.py             # 日志配置
├── 📄 model_database.py             # 模型知识库
│
├── 📄 requirements.txt              # Python依赖
├── 📄 requirements-dev.txt          # 开发依赖
├── 📄 pytest.ini                    # pytest配置
├── 📄 README.md                     # 项目说明
└── 📄 .gitignore                    # Git忽略配置
```

## 🔄 主要改进

### 1. **配置管理**
- ✅ 所有配置集中到 `config/` 目录
- ✅ 导入路径更新为 `from config.config import Config`
- ✅ 数据库初始化脚本统一管理

### 2. **文档组织**
- ✅ 部署文档 → `docs/deployment/`
- ✅ 开发文档 → `docs/development/`
- ✅ 使用指南 → `docs/guides/`
- ✅ 清晰的文档分类

### 3. **测试管理**
- ✅ 测试代码 → `tests/unit/`
- ✅ 测试数据 → `tests/data/`
- ✅ 测试配置统一管理

### 4. **数据隔离**
- ✅ 用户数据集 → `data/datasets/`
- ✅ 模型检查点 → `data/checkpoints/`
- ✅ 训练日志 → `data/logs/`

### 5. **脚本集中**
- ✅ 工具脚本 → `scripts/`
- ✅ 启动脚本 → `scripts/start.sh`
- ✅ 诊断工具统一存放

### 6. **归档管理**
- ✅ 旧版本文件 → `archive/`
- ✅ 备份文件统一归档
- ✅ 历史文档分类存放

## 📝 使用说明

### 启动应用
```bash
# 方式1: 使用启动脚本（推荐）
bash scripts/start.sh

# 方式2: 直接运行
python app.py
```

### 查看文档
```bash
# 部署文档
cat docs/deployment/DEPLOYMENT_MACOS.md

# 使用指南
ls docs/guides/

# 开发文档
ls docs/development/
```

### 运行测试
```bash
# 运行所有测试
pytest

# 运行指定测试
pytest tests/unit/test_auth_功能测试.py
```

### 数据库初始化
```bash
mysql -u root -p flaskdemo_user < config/database_init_with_auth.sql
```

## 🎯 后续优化建议

1. **代码模块化**（待迁移）
   - 将 `app_*.py` 移动到 `app/blueprints/`
   - 将 `auth_module.py` 移动到 `app/core/`
   - 将 `logging_config.py` 移动到 `app/core/`

2. **环境配置**
   - 使用 `.env` 文件管理敏感配置
   - 分离开发/生产环境配置

3. **日志管理**
   - 统一日志输出到 `data/logs/`
   - 增加日志轮转策略

4. **测试覆盖**
   - 补充单元测试
   - 添加集成测试

## 📊 整理前后对比

| 项目 | 整理前 | 整理后 |
|------|--------|--------|
| 根目录文件数 | 50+ | 15 |
| 文档分类 | 无 | 3类 |
| 测试组织 | 散乱 | 集中 |
| 配置管理 | 分散 | 统一 |
| 数据隔离 | 差 | 良好 |

## 🔗 相关文档

- [README.md](README.md) - 项目介绍
- [DEPLOYMENT_MACOS.md](docs/deployment/DEPLOYMENT_MACOS.md) - macOS部署
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - 架构设计
- [docs/README.md](docs/README.md) - 文档导航

---

**最后更新**: 2026-02-04
**整理人**: AI Assistant
