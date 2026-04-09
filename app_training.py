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
training_events = {}  # session_id -> threading.Event（set=运行，clear=暂停）
training_lock = threading.Lock()
RL4CO_AVAILABLE = False
real_rl4co_training = None
simulate_training = None
get_background_db = None


def init_training_globals(status_dict, queues_dict, rl4co_available, real_training_func, simulate_training_func, bg_db_func, lock=None, events_dict=None):
    """从app.py注入全局变量和函数"""
    global training_status, training_queues, training_events, training_lock, RL4CO_AVAILABLE, real_rl4co_training, simulate_training, get_background_db
    training_status = status_dict
    training_queues = queues_dict
    if events_dict is not None:
        training_events = events_dict
    if lock is not None:
        training_lock = lock
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
        
        # 创建消息队列和暂停事件（加锁保护并发写）
        pause_event = threading.Event()
        pause_event.set()  # 初始状态：运行中（set=运行，clear=暂停）
        with training_lock:
            training_queues[session_id] = Queue()
            training_events[session_id] = pause_event

        # ========== 写入 GPU 占用记录 ==========
        if gpu_id is not None:
            _allocate_gpu(gpu_id, session_id, user_id)

        # 包装训练函数：训练结束后自动释放 GPU，并记录完成时间供 Reaper 清理
        def _run_with_gpu_release(train_func, *args):
            try:
                train_func(*args)
            finally:
                _release_gpu(session_id)
                with training_lock:
                    if session_id in training_status:
                        training_status[session_id]['_completed_at'] = time.time()
                    # 确保暂停事件被 set，防止训练线程在异常时永久阻塞
                    evt = training_events.get(session_id)
                    if evt is not None:
                        evt.set()

        # 根据 RL4CO 是否可用选择训练函数
        if RL4CO_AVAILABLE:
            # 真实训练模式 - 传入必要的全局对象和函数
            training_thread = threading.Thread(
                target=_run_with_gpu_release,
                args=(real_rl4co_training, config, session_id, user_id,
                      training_queues[session_id], training_status, get_background_db,
                      pause_event),
                daemon=True
            )
            mode = "真实训练模式"
        else:
            # 模拟训练模式
            training_thread = threading.Thread(
                target=_run_with_gpu_release,
                args=(simulate_training, config, session_id, user_id, pause_event),
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
        logger.error(f"启动训练失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': '启动训练失败，请检查配置后重试'}), 500


@training_bp.route('/api/training_progress/<session_id>')
def training_progress(session_id):
    """使用 Server-Sent Events (SSE) 推送训练进度"""
    def generate():
        with training_lock:
            if session_id not in training_queues:
                yield f"data: {json.dumps({'type': 'error', 'message': '无效的会话 ID'})}\n\n"
                return
            queue = training_queues[session_id]

        try:
            while True:
                try:
                    # 从队列中获取消息（阻塞等待）；Queue 本身线程安全，无需加锁
                    message = queue.get(timeout=1)
                    yield f"data: {message}\n\n"

                    # 如果收到完成、错误或中止消息，则结束流
                    data = json.loads(message)
                    if data['type'] in ['complete', 'error', 'stopped']:
                        break

                except Exception:
                    # 超时或队列为空，发送心跳
                    with training_lock:
                        status = training_status.get(session_id, {})
                    if status.get('status') == 'completed':
                        break
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            # 仅在训练真正结束时清理队列；
            # 若训练仍在运行（客户端只是断连/切换页面），保留队列供重连使用，
            # Reaper 线程会在会话 TTL 到期后统一清理。
            with training_lock:
                terminal_status = training_status.get(session_id, {}).get('status', '')
                if terminal_status in ('completed', 'error', 'stopped'):
                    training_queues.pop(session_id, None)
                    logger.debug(f"已清理会话 {session_id} 的消息队列（训练已结束）")
                else:
                    logger.debug(f"SSE 连接关闭 (session={session_id})，训练仍在运行，保留队列供重连")

    return Response(generate(), mimetype='text/event-stream')


@training_bp.route('/api/training_status/<session_id>')
def get_training_status(session_id):
    """获取当前训练状态"""
    with training_lock:
        status = training_status.get(session_id)
    if status is not None:
        # 过滤掉非 JSON 可序列化的内部字段（以 _ 开头）
        public_status = {k: v for k, v in status.items() if not k.startswith('_')}
        return jsonify({'success': True, 'status': public_status})
    return jsonify({'success': False, 'message': '未找到训练会话'}), 404


@training_bp.route('/api/pause_training/<session_id>', methods=['POST'])
@login_required
def pause_training(session_id):
    """暂停指定训练会话（在当前 Epoch 结束后生效）"""
    with training_lock:
        event = training_events.get(session_id)
        status = training_status.get(session_id, {})

    if event is None:
        return jsonify({'success': False, 'message': '未找到训练会话'}), 404

    current_status = status.get('status', '')
    if current_status != 'running':
        return jsonify({'success': False, 'message': f'无法暂停：当前状态为 {current_status}'}), 400

    event.clear()  # 清除事件 → 训练将在下一个 epoch 边界暂停
    with training_lock:
        if session_id in training_status:
            training_status[session_id]['status'] = 'pausing'

    logger.info(f"训练暂停请求已发送 (session={session_id})")
    return jsonify({'success': True, 'message': '暂停请求已发送，将在当前 Epoch 结束后暂停'})


@training_bp.route('/api/resume_training/<session_id>', methods=['POST'])
@login_required
def resume_training(session_id):
    """恢复指定训练会话"""
    with training_lock:
        event = training_events.get(session_id)
        status = training_status.get(session_id, {})

    if event is None:
        return jsonify({'success': False, 'message': '未找到训练会话'}), 404

    current_status = status.get('status', '')
    if current_status not in ('paused', 'pausing'):
        return jsonify({'success': False, 'message': f'无法恢复：当前状态为 {current_status}'}), 400

    event.set()  # 重新设置事件 → 训练继续
    logger.info(f"训练恢复请求已发送 (session={session_id})")
    return jsonify({'success': True, 'message': '训练已恢复'})


@training_bp.route('/api/stop_training/<session_id>', methods=['POST'])
@login_required
def stop_training(session_id):
    """中止指定训练会话（在当前 Epoch 结束后生效）"""
    with training_lock:
        status_info = training_status.get(session_id)
        event = training_events.get(session_id)

    if status_info is None:
        return jsonify({'success': False, 'message': '未找到训练会话'}), 404

    current_status = status_info.get('status', '')
    if current_status not in ('running', 'pausing', 'paused'):
        return jsonify({'success': False, 'message': f'无法中止：当前状态为 {current_status}'}), 400

    with training_lock:
        training_status[session_id]['stop_requested'] = True
        training_status[session_id]['status'] = 'stopping'
        # 若当前处于暂停阻塞，先解除阻塞让训练线程有机会检查 stop_requested
        if event is not None and not event.is_set():
            event.set()

    logger.info(f"训练中止请求已发送 (session={session_id})")
    return jsonify({'success': True, 'message': '中止请求已发送，将在当前 Epoch 结束后停止'})

