"""
GPU 资源管理模块
提供 GPU 状态查询、占用申请和释放接口
- 有真实 GPU：调用 pynvml 获取实时硬件数据
- 无真实 GPU：返回 Mock 数据，用于开发和演示
"""
from flask import Blueprint, jsonify, request
from datetime import datetime
import logging

from auth_module import login_required, get_current_user_id

gpu_bp = Blueprint('gpu', __name__)
logger = logging.getLogger('rl4co_display')

# ============================================
# pynvml 可用性检测
# ============================================
try:
    import pynvml
    pynvml.nvmlInit()
    _gpu_count = pynvml.nvmlDeviceGetCount()
    PYNVML_AVAILABLE = _gpu_count > 0
    logger.info(f"✓ pynvml 初始化成功，检测到 {_gpu_count} 块 GPU")
except Exception as e:
    PYNVML_AVAILABLE = False
    logger.warning(f"pynvml 不可用，将使用 Mock GPU 数据: {e}")

# ============================================
# Mock GPU 数据（无真实 GPU 时使用）
# ============================================
MOCK_GPUS = [
    {
        "id": 0,
        "name": "NVIDIA A100 80GB PCIe",
        "memory_total_mb": 81920,
        "memory_used_mb": 12340,
        "utilization_pct": 23,
    },
    {
        "id": 1,
        "name": "NVIDIA A100 80GB PCIe",
        "memory_total_mb": 81920,
        "memory_used_mb": 65530,
        "utilization_pct": 87,
    },
    {
        "id": 2,
        "name": "NVIDIA RTX 3090",
        "memory_total_mb": 24576,
        "memory_used_mb": 0,
        "utilization_pct": 0,
    },
    {
        "id": 3,
        "name": "NVIDIA RTX 3090",
        "memory_total_mb": 24576,
        "memory_used_mb": 8192,
        "utilization_pct": 45,
    },
]

# ============================================
# 数据库访问函数（由 app.py 注入）
# ============================================
get_db = None


def init_gpu_globals(get_db_func):
    """从 app.py 注入数据库连接函数"""
    global get_db
    get_db = get_db_func


# ============================================
# 内部辅助函数
# ============================================

def _query_real_gpus():
    """通过 pynvml 查询真实 GPU 硬件信息"""
    gpus = []
    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpus.append({
            "id": i,
            "name": name,
            "memory_total_mb": mem_info.total // (1024 * 1024),
            "memory_used_mb": mem_info.used // (1024 * 1024),
            "utilization_pct": util.gpu,
        })
    return gpus


def _get_allocations():
    """从数据库查询当前所有 allocated 状态的 GPU 占用记录"""
    if get_db is None:
        return {}
    db = get_db()
    if db is None:
        return {}
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT ga.gpu_id, ga.session_id, ga.user_id, ga.allocated_at, u.username
            FROM gpu_allocations ga
            LEFT JOIN users u ON ga.user_id = u.id
            WHERE ga.status = 'allocated'
            ORDER BY ga.allocated_at ASC
        """)
        rows = cursor.fetchall()
        cursor.close()
        # 同一块 GPU 可能有多条记录（软约束模式），返回列表
        result = {}
        for row in rows:
            gid = row['gpu_id']
            if gid not in result:
                result[gid] = []
            result[gid].append({
                'session_id': row['session_id'],
                'user_id': row['user_id'],
                'username': row['username'],
                'allocated_at': row['allocated_at'].isoformat() if row['allocated_at'] else None,
            })
        return result
    except Exception as e:
        logger.error(f"查询 GPU 占用记录失败: {e}")
        return {}


def _build_gpu_list():
    """组合硬件信息与数据库占用信息，返回完整的 GPU 列表"""
    if PYNVML_AVAILABLE:
        raw_gpus = _query_real_gpus()
        is_mock = False
    else:
        raw_gpus = list(MOCK_GPUS)
        is_mock = True

    allocations = _get_allocations()

    result = []
    for gpu in raw_gpus:
        gid = gpu['id']
        mem_total = gpu['memory_total_mb']
        mem_used = gpu['memory_used_mb']
        mem_pct = round(mem_used / mem_total * 100, 1) if mem_total > 0 else 0
        sessions = allocations.get(gid, [])

        # 状态判断：utilization > 90% 或有训练任务占用 → busy
        if sessions:
            status = 'occupied'
        elif gpu['utilization_pct'] >= 90:
            status = 'busy'
        else:
            status = 'idle'

        result.append({
            "id": gid,
            "name": gpu['name'],
            "memory_total_mb": mem_total,
            "memory_used_mb": mem_used,
            "memory_pct": mem_pct,
            "utilization_pct": gpu['utilization_pct'],
            "status": status,
            "sessions": sessions,
        })

    return result, is_mock


# ============================================
# API 路由
# ============================================

@gpu_bp.route('/api/gpu_status')
def gpu_status():
    """
    查询所有 GPU 的当前状态
    返回：硬件信息 + 数据库占用状态
    Mock 模式时字段 is_mock=True
    """
    try:
        gpus, is_mock = _build_gpu_list()
        return jsonify({
            'success': True,
            'gpus': gpus,
            'is_mock': is_mock,
            'timestamp': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"查询 GPU 状态失败: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'查询失败: {str(e)}'}), 500


@gpu_bp.route('/api/gpu_allocate', methods=['POST'])
@login_required
def gpu_allocate():
    """
    申请占用指定 GPU
    请求体：{"gpu_id": 0, "session_id": "xxx"}
    """
    try:
        user_id = get_current_user_id()
        data = request.json or {}
        gpu_id = data.get('gpu_id')
        session_id = data.get('session_id')

        if gpu_id is None or session_id is None:
            return jsonify({'success': False, 'message': '缺少 gpu_id 或 session_id'}), 400

        if get_db is None:
            return jsonify({'success': True, 'message': '数据库不可用，跳过占用记录（Mock 模式）'})

        db = get_db()
        if db is None:
            return jsonify({'success': True, 'message': '数据库连接失败，跳过占用记录'})

        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO gpu_allocations (gpu_id, session_id, user_id, status)
            VALUES (%s, %s, %s, 'allocated')
        """, (gpu_id, session_id, user_id))
        cursor.close()

        logger.info(f"GPU {gpu_id} 已被用户 {user_id} (session: {session_id}) 占用")
        return jsonify({'success': True, 'message': f'GPU {gpu_id} 占用成功'})

    except Exception as e:
        logger.error(f"GPU 占用失败: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'占用失败: {str(e)}'}), 500


@gpu_bp.route('/api/gpu_release', methods=['POST'])
@login_required
def gpu_release():
    """
    释放指定训练会话占用的 GPU
    请求体：{"session_id": "xxx"}
    """
    try:
        data = request.json or {}
        session_id = data.get('session_id')

        if not session_id:
            return jsonify({'success': False, 'message': '缺少 session_id'}), 400

        if get_db is None:
            return jsonify({'success': True, 'message': 'Mock 模式，跳过释放'})

        db = get_db()
        if db is None:
            return jsonify({'success': True, 'message': '数据库连接失败，跳过释放'})

        cursor = db.cursor()
        cursor.execute("""
            UPDATE gpu_allocations
            SET status = 'released', released_at = %s
            WHERE session_id = %s AND status = 'allocated'
        """, (datetime.now(), session_id))
        affected = cursor.rowcount
        cursor.close()

        logger.info(f"Session {session_id} 的 GPU 占用已释放（影响行数: {affected}）")
        return jsonify({'success': True, 'message': 'GPU 释放成功'})

    except Exception as e:
        logger.error(f"GPU 释放失败: {e}", exc_info=True)
        return jsonify({'success': False, 'message': f'释放失败: {str(e)}'}), 500
