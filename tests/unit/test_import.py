#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 app.py 的代码结构完整性（pytest 单元测试版）
不依赖数据库，不启动 Flask，仅做静态代码检查。
"""
import ast
import os

# app.py 的绝对路径（相对于本文件的两级父目录）
_APP_PY = os.path.join(os.path.dirname(__file__), '..', '..', 'app.py')


def _read_app_py() -> str:
    with open(_APP_PY, 'r', encoding='utf-8') as f:
        return f.read()


def test_app_py_syntax():
    """app.py 必须没有语法错误"""
    code = _read_app_py()
    ast.parse(code)  # SyntaxError 会自动被 pytest 捕获为 FAILED


def test_simple_cache_defined():
    """SimpleCache 缓存类已定义"""
    code = _read_app_py()
    assert 'class SimpleCache:' in code


def test_cached_api_defined():
    """cached_api 装饰器函数已定义"""
    code = _read_app_py()
    assert 'def cached_api' in code


def test_cleanup_reaper_defined():
    """_start_cleanup_reaper 清理线程函数已定义（内存泄漏修复）"""
    code = _read_app_py()
    assert 'def _start_cleanup_reaper' in code, \
        "_start_cleanup_reaper 未找到，内存泄漏修复可能已被误删"


def test_reaper_is_called():
    """_start_cleanup_reaper 在字典初始化后被调用"""
    code = _read_app_py()
    assert '_start_cleanup_reaper(training_status' in code, \
        "Reaper 未被启动，training_status 无法自动清理"


def test_cache_defined_before_reaper():
    """SimpleCache 必须在 _start_cleanup_reaper 之前定义"""
    code = _read_app_py()
    cache_pos = code.find('class SimpleCache:')
    reaper_pos = code.find('def _start_cleanup_reaper')
    assert cache_pos != -1, "SimpleCache 未找到"
    assert reaper_pos != -1, "_start_cleanup_reaper 未找到"
    assert cache_pos < reaper_pos, \
        f"SimpleCache（位置 {cache_pos}）应在 _start_cleanup_reaper（位置 {reaper_pos}）之前定义"


def test_all_blueprints_registered():
    """所有 Blueprint 均已注册到 Flask 应用"""
    code = _read_app_py()
    blueprints = ['auth_bp', 'pages_bp', 'stats_bp', 'compat_bp',
                  'training_bp', 'files_bp', 'gpu_bp']
    for bp in blueprints:
        assert bp in code, f"Blueprint '{bp}' 未在 app.py 中注册"


def test_threading_imported():
    """threading 模块已导入（Reaper 线程依赖）"""
    code = _read_app_py()
    assert 'import threading' in code


def test_login_required_correct_endpoint():
    """login_required 使用正确的 Blueprint 端点名 auth.login_page"""
    auth_module_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'auth_module.py')
    with open(auth_module_path, 'r', encoding='utf-8') as f:
        code = f.read()
    assert "url_for('auth.login_page')" in code, \
        "login_required 仍使用错误的端点名 'login'，应为 'auth.login_page'"
    assert "url_for('login')" not in code, \
        "发现旧的错误端点名 url_for('login')，会导致 BuildError"
