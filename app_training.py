"""
训练相关路由模块
包含训练启动、进度监控、状态查询等功能
"""
from flask import Blueprint, request, jsonify, Response
from queue import Queue
import json
import threading
import time
import uuid
import logging
from auth_module import (
    login_required,
    get_session_manager,
    get_current_user_id
)
from modules.compatibility import validate_combination
from app_gpu import _get_allocations

training_bp = Blueprint('training', __name__)
logger = logging.getLogger('rl4co_display')

# 这些全局变量需要从app.py注入
training_status = {}
training_queues = {}
RL4CO_AVAILABLE = False
real_rl4co_training = None
simulate_training = None
get_background_db = None


def init_training_globals(status_dict, queues_dict, rl4co_available, real_training_func, simulate_training_func, bg_db_func):
    """从app.py注入全局变量和函数"""
    global training_status, training_queues, RL4CO_AVAILABLE, real_rl4co_training, simulate_training, get_background_db
    training_status = status_dict
    training_queues = queues_dict
    RL4CO_AVAILABLE = rl4co_available
    real_rl4co_training = real_training_func
    simulate_training = simulate_training_func
    get_background_db = bg_db_func


def _allocate_gpu(gpu_id, session_id, user_id):
    """训练启动时在数据库记录 GPU 占用（内部调用）"""
    if gpu_id is None or get_background_db is None:
        return
    try:
        db = get_background_db()
        if db is None:
            return
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO gpu_allocations (gpu_id, session_id, user_id, status)
            VALUES (%s, %s, %s, 'allocated')
        """, (gpu_id, session_id, user_id))
        cursor.close()
        db.close()
        logger.info(f"GPU {gpu_id} 占用记录已写入 (session={session_id})")
    except Exception as e:
        logger.error(f"写入 GPU 占用记录失败: {e}")


def _release_gpu(session_id):
    """训练结束/失败时释放 GPU 占用（内部调用，使用后台数据库连接）"""
    if get_background_db is None:
        return
    try:
        from datetime import datetime
        db = get_background_db()
        if db is None:
            return
        cursor = db.cursor()
        cursor.execute("""
            UPDATE gpu_allocations
            SET status = 'released', released_at = %s
            WHERE session_id = %s AND status = 'allocated'
        """, (datetime.now(), session_id))
        cursor.close()
        db.close()
        logger.info(f"GPU 占用已释放 (session={session_id})")
    except Exception as e:
        logger.error(f"释放 GPU 占用失败: {e}")


@training_bp.route('/api/start_training', methods=['POST'])
@login_required
def start_training():
    """接收训练配置并启动训练 - 需要登录"""
    try:
        # ========== 获取当前用户ID ==========
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        config = request.json

        # ========== 解析并写入 gpu_id ==========
        gpu_id = config.get('gpu_id')
        if gpu_id is not None:
            try:
                gpu_id = int(gpu_id)
                config['gpu_id'] = gpu_id
            except (ValueError, TypeError):
                gpu_id = None
                config.pop('gpu_id', None)

        # ========== 验证配置组合 ==========
        problem = config.get('problem', 'tsp')
        policy = config.get('model', 'attention')
        algorithm = config.get('algorithm', 'reinforce')
        
        is_valid, message, level = validate_combination(problem, policy, algorithm)
        if not is_valid:
            return jsonify({
                'success': False,
                'message': f'配置验证失败: {message}'
            }), 400
        
        # 如果有警告，记录日志
        if level == 'warning':
            logger.warning(f'训练配置警告: {message}')
        
        # 生成唯一的会话 ID
        session_id = str(uuid.uuid4())
        
        # ========== 记录训练会话到数据库 ==========
        session_manager = get_session_manager()
        if session_manager:
            try:
                session_manager.create_session(
                    user_id=user_id,
                    session_id=session_id,
                    model_type=config.get('model', 'attention'),
                    problem_type=config.get('problem', 'tsp'),
                    config=json.dumps(config)
                )
            except Exception as e:
                logger.error(f"记录训练会话失败: {str(e)}", exc_info=True)
        
        # 创建消息队列
        training_queues[session_id] = Queue()

        # ========== 写入 GPU 占用记录 ==========
        if gpu_id is not None:
            _allocate_gpu(gpu_id, session_id, user_id)

        # 包装训练函数：训练结束后自动释放 GPU，并记录完成时间供 Reaper 清理
        def _run_with_gpu_release(train_func, *args):
            try:
                train_func(*args)
            finally:
                _release_gpu(session_id)
                if session_id in training_status:
                    training_status[session_id]['_completed_at'] = time.time()

        # 根据 RL4CO 是否可用选择训练函数
        if RL4CO_AVAILABLE:
            # 真实训练模式 - 传入必要的全局对象和函数
            training_thread = threading.Thread(
                target=_run_with_gpu_release,
                args=(real_rl4co_training, config, session_id, user_id,
                      training_queues[session_id], training_status, get_background_db),
                daemon=True
            )
            mode = "真实训练模式"
        else:
            # 模拟训练模式
            training_thread = threading.Thread(
                target=_run_with_gpu_release,
                args=(simulate_training, config, session_id, user_id),
                daemon=True
            )
            mode = "模拟训练模式"

        training_thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'gpu_id': gpu_id,
            'message': f'训练已启动 ({mode})' + (f'，使用 GPU {gpu_id}' if gpu_id is not None else '，使用 CPU')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'启动训练失败: {str(e)}'
        }), 500


@training_bp.route('/api/training_progress/<session_id>')
def training_progress(session_id):
    """使用 Server-Sent Events (SSE) 推送训练进度"""
    def generate():
        if session_id not in training_queues:
            yield f"data: {json.dumps({'type': 'error', 'message': '无效的会话 ID'})}\n\n"
            return

        queue = training_queues[session_id]
        try:
            while True:
                try:
                    # 从队列中获取消息（阻塞等待）
                    message = queue.get(timeout=1)
                    yield f"data: {message}\n\n"

                    # 如果收到完成或错误消息，则结束流
                    data = json.loads(message)
                    if data['type'] in ['complete', 'error']:
                        break

                except Exception:
                    # 超时或队列为空，发送心跳
                    if session_id in training_status:
                        if training_status[session_id]['status'] == 'completed':
                            break
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            # 无论正常结束、客户端断连还是异常，都立即释放队列对象
            training_queues.pop(session_id, None)
            logger.debug(f"已清理会话 {session_id} 的消息队列")

    return Response(generate(), mimetype='text/event-stream')


@training_bp.route('/api/training_status/<session_id>')
def get_training_status(session_id):
    """获取当前训练状态"""
    if session_id in training_status:
        return jsonify({
            'success': True,
            'status': training_status[session_id]
        })
    else:
        return jsonify({
            'success': False,
            'message': '未找到训练会话'
        }), 404

