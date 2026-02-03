# RL4CO Display - 系统架构文档

## 架构概览

RL4CO Display 是一个基于 Flask 的 Web 应用，采用 Blueprint 模块化架构，支持多用户的强化学习模型训练与可视化。

```
┌─────────────────────────────────────────────────────────┐
│                    前端 (Templates + JS)                 │
│  ┌──────────┬──────────┬──────────┬──────────┐          │
│  │ 训练界面  │ 算法对比  │ 文件管理  │ 个人中心│          │
│  └──────────┴──────────┴──────────┴──────────┘          │
└─────────────────┬───────────────────────────────────────┘
                  │ HTTP/REST API + SSE
┌─────────────────┴───────────────────────────────────────┐
│              后端 (Flask + Blueprint)                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │                   app.py (入口)                   │   │
│  │  - 应用配置  - 数据库管理  - 全局变量             │   │
│  └──────┬──────────────────────────────────┬─────────┘   │
│         │                                  │             │
│  ┌──────┴───────┐  ┌─────────────┐  ┌─────┴──────┐      │
│  │ app_auth.py  │  │ app_pages.py│  │app_stats.py│      │
│  │ (认证路由)    │  │ (页面路由)  │  │ (统计API)  │      │
│  └──────────────┘  └─────────────┘  └────────────┘      │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────┐      │
│  │app_compat.py │  │app_training │  │app_files.py│      │
│  │ (兼容性API)  │  │ .py (训练)  │  │ (文件管理) │      │
│  └──────────────┘  └─────────────┘  └────────────┘      │
└────────────────┬──────────────────────┬──────────────────┘
                 │                      │
        ┌────────┴──────┐      ┌────────┴──────┐
        │   MySQL DB    │      │  File Storage │
        │ - 用户数据     │      │ - 模型文件    │
        │ - 训练会话     │      │ - 可视化图片  │
        │ - 文件记录     │      │ - 检查点文件  │
        └───────────────┘      └───────────────┘
```

## 目录结构

```
rl4co-display/
├── app.py                    # 主应用入口（约370行）
├── app_auth.py               # 认证相关路由（174行）
├── app_pages.py              # 页面路由（109行）
├── app_stats.py              # 统计API（173行）
├── app_compat.py             # 兼容性API（132行）
├── app_training.py           # 训练API（166行）
├── app_files.py              # 文件管理API（598行）
├── auth_module.py            # 认证核心模块
├── config.py                 # 配置文件
├── logging_config.py         # 日志配置（新增）
├── model_database.py         # 模型知识库
│
├── modules/                  # 业务逻辑模块
│   ├── algorithms/           # 强化学习算法
│   ├── policies/             # 策略模型
│   ├── problems/             # 问题定义（TSP/CVRP等）
│   ├── rl_training/          # 训练核心逻辑
│   │   ├── training_functions.py
│   │   ├── tsp_trainer.py
│   │   ├── cvrp_trainer.py
│   │   └── visualizations/   # 可视化模块
│   └── compatibility.py      # 配置兼容性检查
│
├── templates/                # Jinja2模板
│   ├── index.html            # 训练主页
│   ├── login.html            # 登录页
│   ├── benchmark.html        # 算法对比
│   └── ...
│
├── static/                   # 静态资源
│   ├── css/
│   │   ├── navigation.css    # 导航栏样式
│   │   ├── layout.css        # 全局布局（新增）
│   │   └── training.css      # 训练页面样式（新增）
│   ├── js/
│   │   └── navigation.js
│   └── model_plots/          # 训练结果图片
│
├── tests/                    # 测试（新增）
│   ├── __init__.py
│   ├── conftest.py           # Pytest配置
│   ├── test_parse_dataset.py # 数据集解析测试
│   └── test_compatibility_api.py  # 兼容性测试
│
├── logs/                     # 日志文件（运行时生成）
├── checkpoints/              # 模型检查点
├── datasets/                 # 用户上传的数据集
└── docs/                     # 文档
```

## 核心组件

### 1. 应用入口 (app.py)

**职责**:
- Flask 应用创建与配置
- 数据库连接管理（请求上下文 + 后台任务）
- 全局变量管理（训练状态、队列、缓存）
- Blueprint 注册
- 日志系统初始化

**关键函数**:
- `get_db()`: 获取请求级数据库连接
- `get_background_db()`: 为后台任务创建独立连接
- `cached_api()`: API 响应缓存装饰器
- `simulate_training()`: 模拟训练函数（fallback）

### 2. Blueprint 模块

#### app_auth.py - 认证路由
- **功能**: 用户注册、登录、登出、会话管理
- **路由**:
  - `POST /api/register` - 用户注册
  - `POST /api/login` - 用户登录
  - `POST /api/logout` - 用户登出
  - `GET /api/current_user` - 获取当前用户信息
  - `GET /login`, `/register`, `/logout` - 页面路由

#### app_pages.py - 页面路由
- **功能**: 主要页面渲染
- **路由**:
  - `GET /` - 训练首页
  - `GET /benchmark` - 算法对比页
  - `GET /file_manager` - 文件管理页
  - `GET /profile` - 个人资料页
  - `GET /model_info` - 模型知识库

#### app_training.py - 训练API
- **功能**: 训练任务启动、进度监控
- **路由**:
  - `POST /api/start_training` - 启动训练
  - `GET /api/training_progress/<session_id>` - SSE 进度流
  - `GET /api/training_status/<session_id>` - 查询训练状态

**关键机制**:
- 使用 `threading.Thread` 在后台执行训练
- 使用 `Queue` 进行进度消息传递
- 使用 SSE (Server-Sent Events) 实时推送进度

#### app_files.py - 文件管理API
- **功能**: 数据集管理、训练文件管理
- **路由**:
  - `POST /api/upload_dataset` - 上传数据集
  - `GET /api/list_datasets` - 列出数据集
  - `DELETE /api/delete_dataset` - 删除数据集
  - `GET /api/list_files` - 列出训练文件
  - `DELETE /api/delete_file` - 删除文件
  - `GET /api/download_checkpoint/<filename>` - 下载检查点

**数据集解析**:
- 支持格式: JSON, TXT, TSP (TSPLIB)
- `parse_dataset()` 函数负责解析逻辑

### 3. 日志系统 (logging_config.py)

**功能**:
- 统一的日志配置
- 错误码常量定义
- 标准化错误/成功响应生成

**错误码分类**:
- `1xxx`: 认证相关错误
- `2xxx`: 参数相关错误
- `3xxx`: 资源相关错误
- `4xxx`: 操作相关错误
- `5xxx`: 系统相关错误

### 4. 数据库设计

#### 主要表结构

**users** - 用户表
- `id`: 主键
- `username`: 用户名（唯一）
- `password_hash`: 密码哈希
- `email`: 邮箱
- `create_time`, `last_login`: 时间戳

**training_sessions** - 训练会话表
- `id`: 主键
- `user_id`: 外键关联users
- `session_id`: UUID（唯一）
- `model_type`, `problem_type`: 模型和问题类型
- `config`: JSON配置
- `status`: 状态（running/completed/failed）
- `start_time`, `end_time`: 时间戳
- `final_reward`: 最终奖励值

**training_files** - 训练文件表
- `id`: 主键
- `user_id`, `session_id`: 外键
- `file_name`, `file_path`: 文件信息
- `file_type`: 文件类型（plot/animation/curve/checkpoint）
- `file_size`: 文件大小
- `create_time`: 创建时间

## 数据流

### 训练流程

```
用户 → [前端表单] → POST /api/start_training
                        ↓
                 [配置验证] (validate_combination)
                        ↓
                 [创建session记录] (DB)
                        ↓
                 [启动后台线程] (real_rl4co_training)
                        ↓
                 [返回session_id]
                        
用户 → [前端订阅] → EventSource /api/training_progress/<session_id>
                        ↓
                 [SSE流式推送进度]
                   (通过Queue传递消息)
                        ↓
                 [前端更新UI]
                        
后台线程 → [训练完成] → [保存文件到磁盘]
                        ↓
                 [记录文件到DB]
                        ↓
                 [推送complete消息]
```

### 文件访问流程

```
用户 → GET /api/list_files
         ↓
  [查询training_files表] (按user_id + session分组)
         ↓
  [返回文件列表]
         
用户 → 点击图片
         ↓
  GET /static/model_plots/user_<id>/<filename>
         ↓
  [Flask静态文件服务]
```

## 安全机制

1. **用户认证**:
   - Session-based 认证
   - `@login_required` 装饰器保护路由

2. **数据隔离**:
   - 所有查询强制带 `user_id` 过滤
   - 文件存储按 `user_<id>` 目录隔离

3. **路径安全**:
   - 文件操作使用 `os.path.basename()` 防止路径遍历
   - 检查 `abs_file_path.startswith(user_dir)` 确保访问权限

4. **输入验证**:
   - 文件上传检查扩展名
   - 数据集解析异常处理
   - 配置组合兼容性验证

## 性能优化

1. **API缓存**: 统计数据缓存5分钟（`SimpleCache`）
2. **数据库连接池**: 使用请求上下文复用连接
3. **异步训练**: 训练任务在独立线程执行，不阻塞HTTP请求
4. **静态文件**: 训练结果图片由Flask静态文件服务直接提供

## 测试策略

- **单元测试**: `tests/test_parse_dataset.py`, `tests/test_compatibility_api.py`
- **测试框架**: Pytest
- **运行测试**: `pytest tests/`
- **覆盖率**: `pytest --cov=. tests/`

## 部署注意事项

1. **环境变量**:
   - `SECRET_KEY`: Flask 密钥（生产环境必须设置）
   - 数据库配置在 `config.py`

2. **日志**:
   - 日志文件位于 `logs/` 目录
   - 按日期轮转（`rl4co_display_YYYYMMDD.log`）

3. **文件存储**:
   - `static/model_plots/user_<id>/`: 用户训练图片
   - `checkpoints/user_<id>/`: 用户模型检查点
   - `datasets/user_<id>/`: 用户上传的数据集

4. **数据库初始化**:
   ```bash
   mysql -u root -p < database_init_with_auth.sql
   ```

---

*最后更新: 2026-01-19*


