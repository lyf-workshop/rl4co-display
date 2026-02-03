"""
RL4CO Display - 主应用入口
重构后的Flask应用，路由分离到各个Blueprint模块
"""
from flask import Flask, g
from flask_mysqldb import MySQL
from config import Config
import json
import time
import os
from datetime import timedelta
from queue import Queue
import mysql.connector as mysql_connector
from functools import wraps
import logging

# ========== 导入日志配置 ==========
from logging_config import setup_logging, ErrorCode, error_response, success_response

# ========== 导入模型数据库 ==========
from model_database import MODEL_DATABASE

# ========== 导入认证模块 ==========
from auth_module import (
    login_required, 
    UserManager, 
    TrainingSessionManager,
    FileManager,
    get_current_user_id,
    get_current_username,
    set_user_session,
    clear_user_session,
    get_user_plot_dir,
    get_user_checkpoint_dir,
    safe_join_path
)

# ========== 导入训练模块 ==========
from modules.rl_training import real_rl4co_training, ProgressCallback, create_route_animation

# ========== 配置matplotlib ==========
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 导入 RL4CO 相关模块 ==========
try:
    from rl4co.envs import TSPEnv, CVRPEnv
    from rl4co.models import AttentionModelPolicy, REINFORCE
    from rl4co.utils.trainer import RL4COTrainer
    from lightning.pytorch.callbacks import Callback
    from tensordict import TensorDict
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    TensorDict = None
    logger.warning("RL4CO 库未安装，将使用模拟训练模式")

# ============================================
# Flask应用创建与配置
# ============================================

app = Flask(__name__)
app.config.from_object(Config)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'rl4co-display-secret-key-2024-change-in-production'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

mysql = MySQL(app)

# ============================================
# 配置日志系统
# ============================================
logger = setup_logging('rl4co_display', logging.INFO)
logger.info("=" * 60)
logger.info("RL4CO Display 应用启动")
logger.info("=" * 60)

# ============================================
# 数据库连接管理（使用 Flask 请求上下文）
# ============================================

def get_db():
    """获取当前请求的数据库连接（线程安全）"""
    if 'db' not in g:
        try:
            g.db = mysql_connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                autocommit=True
            )
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}", exc_info=True)
            g.db = None
    return g.db


@app.teardown_appcontext
def close_db(error):
    """在请求结束时关闭数据库连接"""
    db = g.pop('db', None)
    if db is not None:
        try:
            db.close()
        except:
            pass


def get_user_manager():
    """获取当前请求的 UserManager 实例"""
    db = get_db()
    if db is None:
        return None
    if 'user_manager' not in g:
        g.user_manager = UserManager(db)
    return g.user_manager


def get_session_manager():
    """获取当前请求的 TrainingSessionManager 实例"""
    db = get_db()
    if db is None:
        return None
    if 'session_manager' not in g:
        g.session_manager = TrainingSessionManager(db)
    return g.session_manager


def get_file_manager():
    """获取当前请求的 FileManager 实例"""
    db = get_db()
    if db is None:
        return None
    if 'file_manager' not in g:
        g.file_manager = FileManager(db)
    return g.file_manager


def get_background_db():
    """为后台任务创建独立的数据库连接（不使用 Flask g 对象）"""
    try:
        db = mysql_connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DB,
            autocommit=True
        )
        return db
    except Exception as e:
        logger.error(f"后台数据库连接失败: {str(e)}", exc_info=True)
        return None


logger.info("✓ 用户认证模块（请求上下文模式）配置完成")

# 将数据库访问函数添加到auth_module以便Blueprint使用
import auth_module
auth_module.get_db = get_db
auth_module.get_user_manager = get_user_manager
auth_module.get_session_manager = get_session_manager
auth_module.get_file_manager = get_file_manager

# ============================================
# 全局变量：训练状态和进度
# ============================================

training_status = {}
training_queues = {}

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


# ============================================
# 模拟训练函数（当 RL4CO 不可用时）
# ============================================

def simulate_training(config, session_id, user_id):
    """模拟强化学习训练过程（当 RL4CO 不可用时）"""
    queue = training_queues[session_id]
    
    try:
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
        
        for epoch in range(1, epochs + 1):
            time.sleep(0.5)
            
            import random
            progress = (epoch / epochs) * 100
            loss = 10 * (1 - epoch / epochs) + random.uniform(0, 0.5)
            reward = -20 + (15 * epoch / epochs) + random.uniform(-1, 1)
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
        
        training_status[session_id]['status'] = 'completed'
        queue.put(json.dumps({
            'type': 'complete',
            'message': '[模拟模式] 训练完成！',
            'results': {
                'model': model,
                'problem': problem,
                'strategy': 'REINFORCE',
                'total_epochs': epochs,
                'final_loss': training_status[session_id]['loss'],
                'final_reward': training_status[session_id]['reward'],
                'best_reward': training_status[session_id]['best_reward']
            }
        }))
        
    except Exception as e:
        training_status[session_id]['status'] = 'error'
        queue.put(json.dumps({
            'type': 'error',
            'message': f'训练出错: {str(e)}'
        }))


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

# 为训练模块注入全局变量
init_training_globals(
    training_status, 
    training_queues, 
    RL4CO_AVAILABLE, 
    real_rl4co_training, 
    simulate_training, 
    get_background_db
)

# 为统计模块注入cached_api装饰器
import app_stats
app_stats.cached_api = cached_api

# 注册Blueprint - 保持URL路径不变
app.register_blueprint(auth_bp)  # 认证路由：/login, /register, /api/login等
app.register_blueprint(pages_bp)  # 页面路由：/, /benchmark, /profile等
app.register_blueprint(stats_bp)  # 统计API：/api/user_stats, /api/user_activity
app.register_blueprint(compat_bp)  # 兼容性API：/api/compatibility/*
app.register_blueprint(training_bp)  # 训练API：/api/start_training, /api/training_progress等
app.register_blueprint(files_bp)  # 文件管理API：/api/upload_dataset, /api/list_files等

logger.info("✓ 所有Blueprint模块已注册")

# 导出logger供其他模块使用
app.logger = logger

# ============================================
# 应用启动入口
# ============================================

if __name__ == '__main__':
    logger.info("启动Flask开发服务器...")
    app.run(debug=True)

