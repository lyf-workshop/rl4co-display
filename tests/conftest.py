"""
Pytest配置和共享fixtures
"""
import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_UNIT = os.path.join(os.path.dirname(__file__), 'unit')

# 排除脚本风格文件：这些文件含顶层 sys.exit()，被 pytest 收集时会导致 INTERNALERROR
# 正确用法是直接 python 运行，不通过 pytest
collect_ignore = [
    os.path.join(_UNIT, 'test_cvrp_viz.py'),
    os.path.join(_UNIT, 'test_import.py'),
    os.path.join(_UNIT, 'test_auth_功能测试.py'),
]


@pytest.fixture
def app():
    """创建Flask测试应用（mock数据库，无需真实MySQL）"""
    mock_db = MagicMock()
    mock_db.is_connected.return_value = True

    with patch('app.get_db', return_value=mock_db), \
         patch('app.get_background_db', return_value=mock_db), \
         patch('auth_module.init_db_accessors'):
        from app import app as flask_app
        flask_app.config['TESTING'] = True
        flask_app.config['WTF_CSRF_ENABLED'] = False
        flask_app.config['SECRET_KEY'] = 'test-secret-key'
        yield flask_app


@pytest.fixture
def client(app):
    """创建Flask测试客户端"""
    return app.test_client()


@pytest.fixture
def sample_coordinates():
    """提供示例坐标数据"""
    return [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0]
    ]


