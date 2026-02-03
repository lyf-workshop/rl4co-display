"""
统计相关路由模块
包含用户统计数据和活动记录等功能
"""
from flask import Blueprint, jsonify
from functools import wraps
import logging
from auth_module import (
    login_required,
    get_db,
    get_session_manager,
    get_file_manager,
    get_current_user_id
)

stats_bp = Blueprint('stats', __name__)
logger = logging.getLogger('rl4co_display')

# 从 app.py 导入缓存装饰器的引用需要在注册时处理
# 这里先定义一个占位符，实际会在 app.py 中替换
def cached_api(key_prefix=''):
    """API缓存装饰器占位符 - 将在app.py中注入真实实现"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            return f(*args, **kwargs)
        return decorated_function
    return decorator


@stats_bp.route('/api/user_stats', methods=['GET'])
@login_required
def get_user_stats():
    """获取用户统计数据 - 实时数据"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'success': False, 'message': '未登录'}), 401
        
        stats = {
            'total_sessions': 0,
            'completed_sessions': 0,
            'running_sessions': 0,
            'failed_sessions': 0,
            'total_files': 0,
            'total_size_mb': 0
        }
        
        # 从数据库获取统计数据
        db = get_db()
        session_manager = get_session_manager()
        file_manager = get_file_manager()
        
        if db and session_manager and file_manager:
            try:
                # 获取训练会话统计
                cursor = db.cursor(dictionary=True)
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                        SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                    FROM training_sessions
                    WHERE user_id = %s
                """, (user_id,))
                session_stats = cursor.fetchone()
                
                if session_stats:
                    stats['total_sessions'] = session_stats['total'] or 0
                    stats['completed_sessions'] = session_stats['completed'] or 0
                    stats['running_sessions'] = session_stats['running'] or 0
                    stats['failed_sessions'] = session_stats['failed'] or 0
                
                # 获取文件统计
                storage_stats = file_manager.get_user_storage_stats(user_id)
                if storage_stats:
                    stats['total_files'] = storage_stats['total_files'] or 0
                    stats['total_size_mb'] = storage_stats['total_mb'] or 0
                    
            except Exception as e:
                logger.error(f"获取统计数据失败: {str(e)}", exc_info=True)
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取统计失败: {str(e)}'
        }), 500


@stats_bp.route('/api/user_activity', methods=['GET'])
@login_required
def get_user_activity():
    """获取用户最近活动记录 - 实时数据"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'success': False, 'message': '未登录'}), 401
        
        activities = []
        
        # 从数据库获取最近的训练会话
        db = get_db()
        session_manager = get_session_manager()
        
        if db and session_manager:
            try:
                cursor = db.cursor(dictionary=True)
                cursor.execute("""
                    SELECT 
                        session_id,
                        model_type,
                        problem_type,
                        status,
                        start_time,
                        end_time,
                        final_reward
                    FROM training_sessions
                    WHERE user_id = %s
                    ORDER BY start_time DESC
                    LIMIT 10
                """, (user_id,))
                
                sessions = cursor.fetchall()
                
                for session in sessions:
                    time_str = session['start_time'].strftime('%Y-%m-%d %H:%M')
                    
                    if session['status'] == 'completed':
                        text = f"完成了 {session['problem_type'].upper()} 问题的训练 ({session['model_type']})"
                        if session['final_reward']:
                            text += f" - 奖励: {session['final_reward']:.2f}"
                    elif session['status'] == 'running':
                        text = f"开始训练 {session['model_type']} 模型 ({session['problem_type'].upper()})"
                    elif session['status'] == 'failed':
                        text = f"训练失败: {session['model_type']} ({session['problem_type'].upper()})"
                    else:
                        text = f"训练 {session['model_type']} - {session['status']}"
                    
                    activities.append({
                        'time': time_str,
                        'text': text,
                        'status': session['status']
                    })
                    
            except Exception as e:
                logger.error(f"获取活动记录失败: {str(e)}", exc_info=True)
        
        # 如果没有活动记录，返回提示
        if not activities:
            activities = [{
                'time': '无记录',
                'text': '还没有训练记录，开始你的第一次训练吧！',
                'status': 'info'
            }]
        
        return jsonify({
            'success': True,
            'activities': activities
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取活动记录失败: {str(e)}'
        }), 500

