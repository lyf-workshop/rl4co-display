# 依赖安装修复说明

## 问题
`flask-mysqldb` 需要编译 `mysqlclient`，在 macOS 上需要安装额外的系统依赖（pkg-config、MySQL 客户端库等）。

## 解决方案
已将 `flask-mysqldb` 替换为 `mysql-connector-python`，这是纯 Python 实现，无需编译。

## 重新安装步骤

```bash
# 1. 确保虚拟环境已激活
source venv/bin/activate

# 2. 重新安装依赖（会自动使用更新后的 requirements.txt）
pip install -r requirements.txt
```

## 如果仍有问题

如果安装 PyTorch 或其他大型包时速度慢，可以使用国内镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 验证安装

安装完成后，验证关键包：

```bash
python -c "import flask; print(f'Flask: {flask.__version__}')"
python -c "import mysql.connector; print('mysql-connector-python: OK')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```
