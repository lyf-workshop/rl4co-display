"""
页面路由模块
包含主页、算法对比、文件管理、个人资料、模型知识库等页面
"""
from flask import Blueprint, render_template, redirect, url_for, session
from auth_module import (
    login_required,
    get_user_manager,
    get_current_user_id,
    get_current_username
)
from model_database import MODEL_DATABASE

pages_bp = Blueprint('pages', __name__)


@pages_bp.route('/')
def index():
    """主页 - 强制登录检查"""
    # 清理可能的无效session
    user_id = get_current_user_id()
    
    # 强制检查用户是否真实存在
    if not user_id:
        # 清除可能损坏的session
        session.clear()
        return redirect(url_for('auth.login_page'))
    
    # 验证用户是否在数据库中存在
    user_manager = get_user_manager()
    if user_manager:
        user_info = user_manager.get_user(user_id)
        if not user_info:
            # 用户不存在，清除session并重定向
            session.clear()
            return redirect(url_for('auth.login_page'))
    
    username = get_current_username()
    return render_template('index.html', 
                         is_logged_in=True, 
                         username=username,
                         active_page='home')


@pages_bp.route('/benchmark')
@login_required
def benchmark():
    """算法性能对比页面 - 需要登录"""
    return render_template('benchmark.html', active_page='benchmark')


@pages_bp.route('/file_manager')
@login_required
def file_manager():
    """文件管理页面 - 需要登录"""
    return render_template('file_manager.html', active_page='file_manager')


@pages_bp.route('/profile')
@login_required
def profile():
    """我的账户页面 - 需要登录"""
    user_id = get_current_user_id()
    username = get_current_username()
    
    # 获取用户详细信息
    user_data = {
        'username': username,
        'email': None,
        'create_time': None,
        'last_login': None
    }
    
    user_manager = get_user_manager()
    if user_manager:
        user_info = user_manager.get_user(user_id)
        if user_info:
            user_data['email'] = user_info.get('email')
            user_data['create_time'] = user_info.get('create_time').strftime('%Y-%m-%d %H:%M') if user_info.get('create_time') else None
            user_data['last_login'] = user_info.get('last_login').strftime('%Y-%m-%d %H:%M') if user_info.get('last_login') else None
    
    return render_template('profile.html', **user_data, active_page='profile')


@pages_bp.route('/model_info')
@login_required
def model_info_list():
    """模型知识库列表页面"""
    # 获取所有分类
    categories = set()
    for model in MODEL_DATABASE.values():
        if 'category' in model:
            categories.add(model['category'])
    
    return render_template('model_list.html', 
                         models=MODEL_DATABASE, 
                         categories=sorted(categories),
                         active_page='model_info')


@pages_bp.route('/model_info/<model_id>')
@login_required
def model_info_detail(model_id):
    """模型详情页面"""
    if model_id not in MODEL_DATABASE:
        return "模型不存在", 404
    
    model_data = MODEL_DATABASE[model_id]
    return render_template('model_info.html', model_data=model_data, active_page='model_info')


