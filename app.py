"""
RL4CO Display - 主应用入口
重构后的Flask应用，路由分离到各个Blueprint模块
"""
import json
import threading
from flask import Flask, g
from config.config import Config
import os
import time
from datetime import timedelta
from functools import wraps
import mysql.connector as mysql_connector
import logging 

# ========== 导入日志配置 ==========
from logging_config import setup_logging

# ========== 导入认证模块（需要的函数）==========
from auth_module import (
    UserManager, 
    TrainingSessionManager,
    FileManager,
    get_current_user_id
)

# ========== 配置matplotlib（必须在导入pyplot之前）==========
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================
# Flask应用创建与配置
# ============================================

app = Flask(__name__)
app.config.from_object(Config)
_secret_key = os.environ.get('SECRET_KEY')
if not _secret_key:
    import warnings
    _secret_key = 'rl4co-display-secret-key-dev-only-change-in-production'
warnings.warn(
        "SECRET_KEY 未通过环境变量设置，正在使用默认开发密钥。"
        "生产环境必须设置 SECRET_KEY 环境变量，否则 session 可被伪造！",
        stacklevel=1
    )
app.config['SECRET_KEY'] = _secret_key
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# ============================================
# 配置日志系统（必须在其他日志调用之前）
# ============================================
logger = setup_logging('rl4co_display', logging.INFO)

# ========== 导入 RL4CO 相关模块 ==========
# tensordict 是 rl4co 的依赖包，会随 rl4co 一起安装
try:
    from rl4co.envs import TSPEnv, CVRPEnv
    from rl4co.models import AttentionModelPolicy, REINFORCE
    from rl4co.utils.trainer import RL4COTrainer
    from lightning.pytorch.callbacks import Callback
    from tensordict import TensorDict  # rl4co 依赖
    RL4CO_AVAILABLE = True
    logger.info("✓ RL4CO 库加载成功")
except ImportError as e:
    RL4CO_AVAILABLE = False
    TensorDict = None
    logger.warning(f"RL4CO 库未安装，将使用模拟训练模式: {e}")
logger.info("=" * 60)
logger.info("RL4CO Display 应用启动")
logger.info("=" * 60)

# ============================================
# 数据库连接管理（使用 Flask 请求上下文）
# ============================================

def get_db():
    """
    获取当前请求的数据库连接（线程安全）
    
    返回:
        数据库连接对象，连接失败时返回 None
    """
    if 'db' not in g:
        try:
            g.db = mysql_connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                autocommit=True
            )
            logger.debug("数据库连接创建成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}", exc_info=True)
            g.db = None
    return g.db


@app.teardown_appcontext
def close_db(error=None):
    """
    在请求结束时关闭数据库连接
    
    参数:
        error: Flask传递的错误对象（如果有）
    """
    if error:
        logger.error(f"请求处理出错: {error}")
    
    db = g.pop('db', None)
    if db is not None:
        try:
            db.close()
        except Exception as e:
            logger.warning(f"关闭数据库连接时出错: {e}")


def get_user_manager():
    """
    获取当前请求的 UserManager 实例
    
    返回:
        UserManager 实例，数据库连接失败时返回 None
    """
    db = get_db()
    if db is None:
        return None
    if 'user_manager' not in g:
        g.user_manager = UserManager(db)
    return g.user_manager


def get_session_manager():
    """
    获取当前请求的 TrainingSessionManager 实例
    
    返回:
        TrainingSessionManager 实例，数据库连接失败时返回 None
    """
    db = get_db()
    if db is None:
        return None
    if 'session_manager' not in g:
        g.session_manager = TrainingSessionManager(db)
    return g.session_manager


def get_file_manager():
    """
    获取当前请求的 FileManager 实例
    
    返回:
        FileManager 实例，数据库连接失败时返回 None
    """
    db = get_db()
    if db is None:
        return None
    if 'file_manager' not in g:
        g.file_manager = FileManager(db)
    return g.file_manager


def get_background_db():
    """
    为后台任务创建独立的数据库连接（不使用 Flask g 对象）
    
    注意:
        此函数用于后台线程，不依赖 Flask 请求上下文
    
    返回:
        数据库连接对象，连接失败时返回 None
    """
    try:
        db = mysql_connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DB,
            autocommit=True
        )
        logger.debug("后台数据库连接创建成功")
        return db
    except Exception as e:
        logger.error(f"后台数据库连接失败: {str(e)}", exc_info=True)
        return None


logger.info("✓ 用户认证模块（请求上下文模式）配置完成")

# 将数据库访问函数注入 auth_module（通过正式 API，而非运行时属性赋值）
import auth_module
auth_module.init_db_accessors(get_db, get_user_manager, get_session_manager, get_file_manager)

# ============================================
# 全局变量：训练状态和进度
# ============================================

training_status = {}
training_queues = {}
training_events = {}  # session_id -> threading.Event（set=运行，clear=暂停）
# 保护上述三个字典的并发读写
training_lock = threading.Lock()

# 创建输出目录
PLOTS_DIR = "static/model_plots"
CHECKPOINTS_DIR = "checkpoints"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ============================================
# 性能优化：API响应缓存
# ============================================

class SimpleCache:
    """简单的API响应缓存"""
    def __init__(self, timeout=300):
        self.cache = {}
        self.timeout = timeout
    
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.timeout:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, time.time())
    
    def clear(self):
        self.cache.clear()


api_cache = SimpleCache(timeout=300)  # 5分钟缓存


def _start_cleanup_reaper(status_dict, queues_dict, events_dict, lock, ttl_seconds=1800):
    """
    启动后台守护线程，定期清理过期的训练状态和队列。

    每 10 分钟扫描一次 status_dict，将状态为 completed/error 且完成时间
    超过 ttl_seconds（默认 30 分钟）的条目从三个字典中同时删除。
    队列通常已由 SSE 生成器的 finally 块提前清理，此处为兜底。
    """
    def _reaper():
        while True:
            time.sleep(600)  # 每 10 分钟检查一次
            now = time.time()
            with lock:
                expired = [
                    sid for sid, info in list(status_dict.items())
                    if info.get('status') in ('completed', 'error', 'stopped')
                    and (now - info.get('_completed_at', now)) > ttl_seconds
                ]
                for sid in expired:
                    status_dict.pop(sid, None)
                    queues_dict.pop(sid, None)
                    # 确保事件被 set，防止训练线程（如果还在阻塞中）永久挂起
                    evt = events_dict.pop(sid, None)
                    if evt is not None:
                        evt.set()
            if expired:
                logger.info(f"[Reaper] 已清理 {len(expired)} 个过期会话: {expired}")

    t = threading.Thread(target=_reaper, daemon=True, name='session-reaper')
    t.start()
    logger.info("✓ 训练会话清理线程（Reaper）已启动")


def cached_api(key_prefix=''):
    """API缓存装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 生成缓存键
            user_id = get_current_user_id()
            cache_key = f"{key_prefix}:{user_id}"
            
            # 尝试从缓存获取
            cached_result = api_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 调用原函数
            result = f(*args, **kwargs)
            
            # 缓存结果（仅缓存成功的响应）
            if isinstance(result, tuple):
                data, status_code = result if len(result) == 2 else (result, 200)
                if status_code == 200:
                    api_cache.set(cache_key, result)
            else:
                api_cache.set(cache_key, result)
            
            return result
        return decorated_function
    return decorator


# 启动后台清理线程，防止 training_status / training_queues / training_events 无限增长
_start_cleanup_reaper(training_status, training_queues, training_events, training_lock, ttl_seconds=1800)


def _cleanup_stale_gpu_allocations():
    """
    应用启动时，将数据库中所有残留的 allocated GPU 记录标记为 released。

    原因：进程重启后所有训练线程均已消亡，其遗留的 allocated 记录是孤立数据，
    会导致前端始终显示 GPU 被占用。此清理在进程生命周期内只执行一次。
    """
    try:
        db = get_background_db()
        if db is None:
            logger.warning("[启动清理] 无法连接数据库，跳过 GPU 占用记录清理")
            return
        cursor = db.cursor()
        cursor.execute("""
            UPDATE gpu_allocations
            SET status = 'released', released_at = NOW()
            WHERE status = 'allocated'
        """)
        affected = cursor.rowcount
        cursor.close()
        db.close()
        if affected > 0:
            logger.info(f"[启动清理] 已释放 {affected} 条残留 GPU 占用记录（上次进程未正常退出）")
        else:
            logger.info("[启动清理] 无残留 GPU 占用记录")
    except Exception as e:
        logger.warning(f"[启动清理] 清理 GPU 占用记录失败: {e}")


_cleanup_stale_gpu_allocations()

# ============================================
# 模拟训练函数（当 RL4CO 不可用时）
# ============================================

def simulate_training(config, session_id, user_id, pause_event=None):
    """模拟强化学习训练过程（当 RL4CO 不可用时）"""
    with training_lock:
        queue = training_queues[session_id]

    try:
        with training_lock:
            training_status[session_id] = {
                'status': 'running',
                'progress': 0,
                'epoch': 0,
                'loss': 0,
                'reward': 0,
                'best_reward': 0
            }

        epochs = int(config.get('epochs', 10))
        model = config.get('model', 'attention')
        problem = config.get('problem', 'tsp')

        queue.put(json.dumps({
            'type': 'info',
            'message': f'[模拟模式] 开始训练 {model.upper()} 模型，问题类型: {problem.upper()}'
        }))

        stopped = False
        for epoch in range(1, epochs + 1):
            time.sleep(0.5)

            # 中止检测（每个 epoch 开始前）
            with training_lock:
                if training_status.get(session_id, {}).get('stop_requested'):
                    training_status[session_id]['status'] = 'stopped'
                    stopped = True
                    break

            import random
            progress = (epoch / epochs) * 100
            loss = 10 * (1 - epoch / epochs) + random.uniform(0, 0.5)
            reward = -20 + (15 * epoch / epochs) + random.uniform(-1, 1)
            with training_lock:
                best_reward = max(training_status[session_id].get('best_reward', float('-inf')), reward)
                training_status[session_id].update({
                    'progress': progress,
                    'epoch': epoch,
                    'loss': round(loss, 4),
                    'reward': round(reward, 4),
                    'best_reward': round(best_reward, 4)
                })

            queue.put(json.dumps({
                'type': 'progress',
                'epoch': epoch,
                'total_epochs': epochs,
                'progress': round(progress, 2),
                'loss': round(loss, 4),
                'reward': round(reward, 4),
                'best_reward': round(best_reward, 4)
            }))

            if epoch % 2 == 0 or epoch == epochs:
                queue.put(json.dumps({
                    'type': 'info',
                    'message': f'Epoch {epoch}/{epochs} - Loss: {loss:.4f}, Reward: {reward:.4f}'
                }))

            # 暂停检测（每个 epoch 结束后）
            if pause_event is not None and not pause_event.is_set():
                with training_lock:
                    if session_id in training_status:
                        training_status[session_id]['status'] = 'paused'
                queue.put(json.dumps({
                    'type': 'paused',
                    'message': f'训练已暂停（已完成 Epoch {epoch}/{epochs}）',
                    'epoch': epoch,
                    'progress': round(progress, 2)
                }))
                pause_event.wait()  # 阻塞，直到 resume 或 stop
                # 被唤醒后检查是否为中止请求
                with training_lock:
                    if training_status.get(session_id, {}).get('stop_requested'):
                        training_status[session_id]['status'] = 'stopped'
                        stopped = True
                        break
                    if session_id in training_status:
                        training_status[session_id]['status'] = 'running'
                if stopped:
                    break
                queue.put(json.dumps({
                    'type': 'resumed',
                    'message': f'训练已恢复，继续 Epoch {epoch + 1}'
                }))

        if stopped:
            queue.put(json.dumps({
                'type': 'stopped',
                'message': f'[模拟模式] 训练已中止',
                'epoch': training_status.get(session_id, {}).get('epoch', 0),
                'progress': training_status.get(session_id, {}).get('progress', 0)
            }))
            return

        with training_lock:
            training_status[session_id]['status'] = 'completed'
            final = {
                'final_loss': training_status[session_id]['loss'],
                'final_reward': training_status[session_id]['reward'],
                'best_reward': training_status[session_id]['best_reward']
            }
        queue.put(json.dumps({
            'type': 'complete',
            'message': '[模拟模式] 训练完成！',
            'results': {
                'model': model,
                'problem': problem,
                'strategy': 'REINFORCE',
                'total_epochs': epochs,
                **final
            }
        }))

    except Exception as e:
        with training_lock:
            training_status[session_id]['status'] = 'error'
        queue.put(json.dumps({
            'type': 'error',
            'message': f'训练出错: {str(e)}'
        }))


# ============================================
# 导入训练模块（需要在注册Blueprint之前）
# ============================================
try:
    from modules.rl_training import real_rl4co_training
except ImportError:
    logger.warning("无法导入 real_rl4co_training，将仅使用模拟训练")
    real_rl4co_training = None

# ============================================
# 注册 Blueprint 模块
# ============================================

# 导入所有Blueprint模块
from app_auth import auth_bp
from app_pages import pages_bp
from app_stats import stats_bp
from app_compat import compat_bp
from app_training import training_bp, init_training_globals
from app_files import files_bp
from app_gpu import gpu_bp, init_gpu_globals

# 为训练模块注入全局变量（含线程锁和暂停事件字典）
init_training_globals(
    training_status,
    training_queues,
    RL4CO_AVAILABLE,
    real_rl4co_training,
    simulate_training,
    get_background_db,
    lock=training_lock,
    events_dict=training_events
)

# 为统计模块注入cached_api装饰器
import app_stats
app_stats.cached_api = cached_api

# 为 GPU 模块注入数据库访问函数
init_gpu_globals(get_db)

# 注册Blueprint - 保持URL路径不变
app.register_blueprint(auth_bp)    # 认证路由：/login, /register, /api/login等
app.register_blueprint(pages_bp)   # 页面路由：/, /benchmark, /profile等
app.register_blueprint(stats_bp)   # 统计API：/api/user_stats, /api/user_activity
app.register_blueprint(compat_bp)  # 兼容性API：/api/compatibility/*
app.register_blueprint(training_bp) # 训练API：/api/start_training, /api/training_progress等
app.register_blueprint(files_bp)   # 文件管理API：/api/upload_dataset, /api/list_files等
app.register_blueprint(gpu_bp)     # GPU管理API：/api/gpu_status, /api/gpu_allocate等

logger.info("✓ 所有Blueprint模块已注册")

# 导出logger供其他模块使用
app.logger = logger

# ============================================
# 应用启动入口
# ============================================

if __name__ == '__main__':
    logger.info("启动Flask开发服务器...")
    app.run(debug=True)

