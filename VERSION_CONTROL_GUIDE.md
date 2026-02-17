# 版本管理指南

本文档说明 RL4CO Display 项目的版本管理策略和最佳实践。

## 配置文件管理

### 敏感信息处理

项目中的敏感信息（如数据库密码）已从版本控制中移除，改用配置模板：

1. **配置模板文件**（已提交到Git）：
   - `config/config.example.py` - Python配置模板
   - `.env.example` - 环境变量模板

2. **实际配置文件**（不提交到Git）：
   - `config/config.py` - 包含实际密码
   - `.env` - 环境变量配置

### 首次配置步骤

克隆项目后，需要创建实际配置文件：

```bash
# 1. 复制配置模板
cp config/config.example.py config/config.py

# 2. 编辑配置文件，修改密码
vim config/config.py  # 或使用其他编辑器

# 3. （可选）使用环境变量
cp .env.example .env
vim .env
```

### 环境变量支持

`config/config.py` 支持从环境变量读取配置：

```python
# 优先级：环境变量 > config.py 中的默认值
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '默认值')
```

推荐在生产环境使用环境变量，开发环境使用 `config.py`。

## 文件分类

### 应该提交的文件 ✅

- **源代码**: `*.py`, `*.html`, `*.css`, `*.js`
- **配置模板**: `config.example.py`, `.env.example`
- **文档**: `README.md`, `docs/`
- **测试**: `tests/`
- **依赖**: `requirements.txt`, `requirements-dev.txt`
- **数据库脚本**: `config/database_init_with_auth.sql`

### 不应提交的文件 ❌

#### 1. 敏感信息
- `config/config.py` - 包含数据库密码
- `.env` - 环境变量配置
- `*.secret` - 任何密钥文件

#### 2. IDE 配置
- `.idea/` - PyCharm
- `.vscode/` - VS Code
- `*.swp`, `*.swo` - Vim临时文件

#### 3. 训练生成文件
- `checkpoints/` - 模型检查点 (~173MB)
- `lightning_logs/` - 训练日志 (~626MB)
- `static/model_plots/user_*/` - 用户生成的图片 (~99MB)
- `logs/*.log` - 应用日志

#### 4. Python 缓存
- `__pycache__/`
- `*.pyc`
- `venv/`, `.venv/`

#### 5. 数据文件
- `datasets/` - 用户上传的数据集
- `*.pkl`, `*.pickle` - 序列化数据

#### 6. 临时文件
- `*.bak` - 备份文件
- `*.tmp` - 临时文件
- `.DS_Store` - macOS 系统文件

## .gitignore 结构

项目的 `.gitignore` 分为以下几个部分：

1. **Python 标准忽略规则** - 字节码、虚拟环境等
2. **IDE 配置** - `.idea/`, `.vscode/`
3. **敏感配置** - `config/config.py`, `.env`
4. **训练生成文件** - `checkpoints/`, `lightning_logs/`
5. **数据集** - `datasets/`, `*.pkl`
6. **临时文件** - `*.bak`, `.DS_Store`

## 常见问题

### Q: 如何知道哪些文件被忽略了？

```bash
# 查看当前状态
git status

# 检查特定文件是否被忽略
git check-ignore -v config/config.py
```

### Q: 误提交了敏感文件怎么办？

如果已提交但未推送：

```bash
# 从索引移除（保留本地文件）
git rm --cached config/config.py
git commit -m "Remove sensitive file"
```

如果已推送到远程，需要重写历史（谨慎操作）：

```bash
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config/config.py" \
  --prune-empty --tag-name-filter cat -- --all
```

### Q: 为什么训练生成的文件不提交？

这些文件：
1. 体积巨大（近1GB），会让仓库变得臃肿
2. 可以重新生成，不是源代码
3. 每个用户的训练结果不同，提交会造成冲突

如需分享模型，建议使用专门的模型托管服务（如 Hugging Face）。

### Q: 团队协作时如何同步配置？

1. **不要**直接共享 `config.py`（包含密码）
2. **应该**共享配置模板 `config.example.py`
3. 每个成员根据模板创建自己的 `config.py`
4. 使用文档或口头方式传递必要的配置信息

## 验证清单

提交前检查：

- [ ] `git status` 不显示敏感文件
- [ ] `config/config.py` 不在跟踪列表中
- [ ] `.idea/` 等 IDE 配置已被忽略
- [ ] 训练生成文件（checkpoints, logs）已被忽略
- [ ] 配置模板文件已提交

## 相关文档

- [README.md](README.md) - 项目总体说明
- [.gitignore](.gitignore) - 忽略规则配置
- [config/config.example.py](config/config.example.py) - 配置模板

---

**最后更新**: 2026-02-17  
**维护者**: RL4CO Display 开发团队
