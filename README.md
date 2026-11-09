# RL4CO Display

## 项目简介

RL4CO Display 是一个基于 Flask、PyTorch 和 RL4CO 的组合优化强化学习训练与可视化平台。用户可以在浏览器中选择问题类型、策略模型和训练算法，配置训练参数，并实时查看训练日志、Loss/Reward 曲线、路径对比图及动态 GIF。

项目支持 TSP、ATSP、mTSP、CVRP、SDVRP、VRPTW、PDP、OP、PCTSP、SPCTSP、FFSP 和芯片布线等问题，并提供用户管理、自定义数据集、GPU 分配、训练检查点、算法对比和大模型助手功能。

## 技术栈

- 后端：Python 3.11、Flask 3.0、Gunicorn
- 强化学习：PyTorch、Lightning、RL4CO
- 数据库：MySQL 8.0+
- 可视化：Matplotlib、Pillow
- 前端：HTML、CSS、JavaScript、SSE
- GPU：NVIDIA CUDA，可选

## 主要目录

```text
app.py                 Flask 应用入口
app_*.py               认证、训练、文件、GPU 和页面接口
config/                配置模板及数据库初始化脚本
modules/               问题、策略、算法、Trainer 和可视化实现
templates/             HTML 页面
static/                CSS、JavaScript、图片和运行时结果目录
tests/                 自动化测试及示例数据集
rl4co-main/            项目配套的 RL4CO 源码
scripts/               部署、诊断及打包脚本
```

## 部署流程

### 1. 环境要求

- Python 3.11
- MySQL 8.0 或更高版本
- 建议使用 Conda 管理 Python 环境
- GPU 训练需要 NVIDIA 驱动及与其兼容的 CUDA 版 PyTorch

### 2. 创建 Python 环境

进入项目目录后执行：

```bash
conda create -n rl4co-display python=3.11 -y
conda activate rl4co-display
```

安装项目依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -e ./rl4co-main
```

如需指定 CUDA 版本，建议先按照 PyTorch 官方命令安装对应的 PyTorch，再执行其余依赖安装，避免覆盖 CUDA 构建。

Linux 生产部署还需要 Gunicorn：

```bash
python -m pip install gunicorn
```

### 3. 初始化 MySQL

登录 MySQL：

```bash
mysql -u root -p
```

创建数据库：

```sql
CREATE DATABASE flaskdemo_user
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
EXIT;
```

导入表结构：

```bash
mysql -u root -p flaskdemo_user < config/database_init_with_auth.sql
```

### 4. 配置项目

Linux：

```bash
cp config/config.example.py config/config.py
```

Windows PowerShell：

```powershell
Copy-Item config/config.example.py config/config.py
```

编辑 `config/config.py`，填写实际数据库信息：

```python
MYSQL_HOST = "127.0.0.1"
MYSQL_USER = "数据库用户名"
MYSQL_PASSWORD = "数据库密码"
MYSQL_DB = "flaskdemo_user"
```

云端大模型功能可在同一配置文件的 `LLM_PROVIDERS` 中按需配置。不要提交包含真实密码或 API Key 的 `config/config.py`。

生产环境应设置独立的会话密钥：

```bash
export SECRET_KEY="请替换为随机长字符串"
```

### 5. 本地开发启动

```bash
python app.py
```

默认访问：`http://127.0.0.1:5000`

### 6. Linux 服务器启动

如果只使用指定 GPU，可先设置 `CUDA_VISIBLE_DEVICES`。单卡可填写一个编号或 UUID，多卡使用逗号分隔：

```bash
export CUDA_VISIBLE_DEVICES="7"
```

创建日志目录并通过 Gunicorn 启动，以下示例使用 `15080` 端口：

```bash
mkdir -p logs
MPLBACKEND=Agg PYTHONUNBUFFERED=1 nohup python -m gunicorn \
  --bind 127.0.0.1:15080 \
  --worker-class gthread \
  --workers 1 \
  --threads 16 \
  --timeout 3600 \
  --pid logs/gunicorn-15080.pid \
  --access-logfile logs/gunicorn-access.log \
  --error-logfile logs/gunicorn-error.log \
  --capture-output app:app >/dev/null 2>&1 &
```

检查服务：

```bash
curl -I http://127.0.0.1:15080/
tail -n 50 logs/gunicorn-error.log
```

返回 `HTTP 302` 通常表示服务正常并跳转到登录页面。

### 7. 从本地访问服务器

如果服务只监听服务器的 `127.0.0.1:15080`，在本地 PowerShell 建立 SSH 隧道：

```powershell
ssh -p <SSH端口> -N -L 15080:127.0.0.1:15080 <用户名>@<服务器地址>
```

保持该窗口运行，然后访问：`http://127.0.0.1:15080`

### 8. 停止和重启

停止服务：

```bash
kill -TERM "$(cat logs/gunicorn-15080.pid)"
```

确认端口释放后，重新执行第 6 步的 Gunicorn 启动命令即可完成重启。

## 测试

安装开发依赖并运行测试：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

当前交付版本全量测试结果为 `180 passed`。
