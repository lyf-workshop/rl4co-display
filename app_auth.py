"""
认证相关路由模块
包含用户登录、注册、登出等功能
"""
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, session
from auth_module import (
    get_user_manager,
    set_user_session,
    clear_user_session,
    get_current_user_id
)

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login')
def login_page():
    """登录页面 - 如果已登录则跳转到主页"""
    user_id = get_current_user_id()
    if user_id:
        # 已登录，直接跳转到主页
        return redirect(url_for('pages.index'))
    return render_template('login.html')


@auth_bp.route('/register')
def register_page():
    """注册页面"""
    return render_template('register.html')


@auth_bp.route('/res')
def Index_res():
    """旧版注册页面 - 重定向到新页面"""
    return redirect(url_for('auth.register_page'))


@auth_bp.route('/api/register', methods=['POST'])
def api_register():
    """用户注册API"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        
        if not username or not password:
            return jsonify({
                'success': False, 
                'message': '用户名和密码不能为空'
            }), 400
        
        user_manager = get_user_manager()
        if user_manager is None:
            return jsonify({
                'success': False,
                'message': '认证模块未初始化'
            }), 500
        
        success, message, user_id = user_manager.create_user(username, password, email)
        
        if success:
            return jsonify({
                'success': True, 
                'message': message,
                'user_id': user_id
            })
        else:
            return jsonify({
                'success': False, 
                'message': message
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'注册失败：{str(e)}'
        }), 500


@auth_bp.route('/api/login', methods=['POST'])
def api_login():
    """用户登录API"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                'success': False, 
                'message': '用户名和密码不能为空'
            }), 400
        
        user_manager = get_user_manager()
        if user_manager is None:
            return jsonify({
                'success': False,
                'message': '认证模块未初始化'
            }), 500
        
        success, message, user_data = user_manager.verify_user(username, password)
        
        if success:
            set_user_session(user_data)
            return jsonify({
                'success': True, 
                'message': message,
                'user': user_data
            })
        else:
            return jsonify({
                'success': False, 
                'message': message
            }), 401
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'登录失败：{str(e)}'
        }), 500


@auth_bp.route('/api/logout', methods=['POST'])
def api_logout():
    """用户登出API"""
    clear_user_session()
    return jsonify({
        'success': True, 
        'message': '已退出登录'
    })


@auth_bp.route('/logout')
def logout_page():
    """直接访问的登出页面 - 清除session并跳转到登录页"""
    session.clear()
    clear_user_session()
    return redirect(url_for('auth.login_page'))


@auth_bp.route('/api/current_user', methods=['GET'])
def api_current_user():
    """获取当前登录用户信息"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({
            'success': False, 
            'message': '未登录'
        }), 401
    
    user_manager = get_user_manager()
    if user_manager is None:
        return jsonify({
            'success': False,
            'message': '认证模块未初始化'
        }), 500
    
    user_data = user_manager.get_user(user_id)
    return jsonify({
        'success': True, 
        'user': user_data
    })


