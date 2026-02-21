from flask import Flask, render_template, request, render_template_string, redirect, url_for, jsonify, Response, session, g
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config  # 导入配置文件
import json
import time
import threading
from queue import Queue
import torch
import os
import sys
from io import StringIO
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from datetime import timedelta, datetime
from functools import wraps
import uuid as uuid_module
import mysql.connector as mysql_connector
# ========== 导入模型数据库 ==========
from model_database import MODEL_DATABASE

# ========== 导入认证模块 ==========
from auth_module import (
    login_required, 
    UserManager, 
    TrainingSessionManager,
    FileManager,
    get_user_plot_dir,
    get_user_checkpoint_dir,
    set_user_session,
    clear_user_session,
    get_current_user_id,
    get_current_username,
    safe_join_path
)

# ========== 导入训练模块 ==========
from modules.rl_training import real_rl4co_training, ProgressCallback, create_route_animation

# ========== 导入兼容性验证模块 ==========
from modules.compatibility import (
    validate_combination,
    get_available_policies,
    get_available_algorithms,
    get_recommended_combination,
    get_compatibility_info,
    get_frontend_constraints
)

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 导入 RL4CO 相关模块
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
    print("警告: RL4CO 库未安装，将使用模拟训练模式")

app = Flask(__name__)
# 加载配置
app.config.from_object(Config)

# ========== 添加 SECRET_KEY 配置（用户认证必需） ==========
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'rl4co-display-secret-key-2024-change-in-production'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

mysql = MySQL(app)

# ========== 数据库连接管理（使用 Flask 请求上下文） ==========
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
            print(f"✗ 数据库连接失败: {str(e)}")
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
        print(f"✗ 后台数据库连接失败: {str(e)}")
        return None

print("✓ 用户认证模块（请求上下文模式）配置完成")

# 用于存储训练状态和进度的全局字典
training_status = {}
training_queues = {}

# 创建输出目录
PLOTS_DIR = "static/model_plots"
CHECKPOINTS_DIR = "checkpoints"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ========== 性能优化：API响应缓存 ==========
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
            
            # 缓存结果（仅缓存成功地响应）
            if isinstance(result, tuple):
                data, status_code = result if len(result) == 2 else (result, 200)
                if status_code == 200:
                    api_cache.set(cache_key, result)
            else:
                api_cache.set(cache_key, result)
            
            return result
        return decorated_function
    return decorator


# ========== create_route_animation 已迁移到 modules/rl_training/training_functions.py ==========


# ============================================
# 用户认证路由（新增）
# ============================================

@app.route('/')
def index():
    """主页 - 强制登录检查"""
    # 清理可能的无效session
    user_id = get_current_user_id()
    
    # 强制检查用户是否真实存在
    if not user_id:
        # 清除可能损坏的session
        session.clear()
        return redirect(url_for('login_page'))
    
    # 验证用户是否在数据库中存在
    user_manager = get_user_manager()
    if user_manager:
        user_info = user_manager.get_user(user_id)
        if not user_info:
            # 用户不存在，清除session并重定向
            session.clear()
            return redirect(url_for('login_page'))
    
    username = get_current_username()
    return render_template('index.html', 
                         is_logged_in=True, 
                         username=username,
                         active_page='home')

@app.route('/login')
def login_page():
    """登录页面 - 如果已登录则跳转到主页"""
    user_id = get_current_user_id()
    if user_id:
        # 已登录，直接跳转到主页
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register')
def register_page():
    """注册页面"""
    return render_template('register.html')

@app.route('/api/register', methods=['POST'])
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

@app.route('/api/login', methods=['POST'])
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

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """用户登出API"""
    clear_user_session()
    return jsonify({
        'success': True, 
        'message': '已退出登录'
    })

@app.route('/logout')
def logout_page():
    """直接访问的登出页面 - 清除session并跳转到登录页"""
    session.clear()
    clear_user_session()
    return redirect(url_for('login_page'))

@app.route('/api/current_user', methods=['GET'])
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

# ============================================
# 原有路由（保持兼容）
# ============================================

@app.route('/res')
def Index_res():
    """旧版注册页面 - 重定向到新页面"""
    return redirect(url_for('register_page'))

@app.route('/benchmark')
@login_required
def benchmark():
    """算法性能对比页面 - 需要登录"""
    return render_template('benchmark.html', active_page='benchmark')

@app.route('/file_manager')
@login_required
def file_manager():
    """文件管理页面 - 需要登录"""
    return render_template('file_manager.html', active_page='file_manager')

@app.route('/profile')
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

@app.route('/api/user_stats', methods=['GET'])
@login_required
@cached_api(key_prefix='user_stats')
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
                print(f"获取统计数据失败: {str(e)}")
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取统计失败: {str(e)}'
        }), 500

@app.route('/api/user_activity', methods=['GET'])
@login_required
@cached_api(key_prefix='user_activity')
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
                print(f"获取活动记录失败: {str(e)}")
        
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

# 模型知识库数据

@app.route('/model_info')
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

@app.route('/model_info/<model_id>')
@login_required
def model_info_detail(model_id):
    """模型详情页面"""
    if model_id not in MODEL_DATABASE:
        return "模型不存在", 404
    
    model_data = MODEL_DATABASE[model_id]
    return render_template('model_info.html', model_data=model_data, active_page='model_info')

# ============================================
# 旧的 register 和 login 路由已被新的 API 替代，已删除 ==========
#   新路由在文件开头：/api/register, /api/login, /api/logout

# ========== ProgressCallback 和 real_rl4co_training 已迁移到 modules/rl_training/training_functions.py ==========

# 占位标记：原 ProgressCallback 类在此处，现已迁移

# 真实的 RL4CO 训练函数

# 模拟训练函数（备用）
def simulate_training(config, session_id, user_id):  # ========== 添加user_id参数 ==========
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


@app.route('/api/upload_dataset', methods=['POST'])
@login_required
def upload_dataset():
    """处理TSP数据集上传 - 需要登录"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        # 检查是否有文件上传
        if 'dataset' not in request.files:
            return jsonify({
                'success': False,
                'message': '没有找到上传的文件'
            }), 400
        
        file = request.files['dataset']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': '未选择文件'
            }), 400
        
        # 获取文件扩展名
        filename = file.filename
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        if file_ext not in ['txt', 'json', 'tsp']:
            return jsonify({
                'success': False,
                'message': f'不支持的文件格式: {file_ext}，仅支持 .txt, .json, .tsp'
            }), 400
        
        # 读取文件内容
        file_content = file.read().decode('utf-8')
        
        # 解析数据集
        coordinates = parse_dataset(file_content, file_ext)
        
        if coordinates is None or len(coordinates) == 0:
            return jsonify({
                'success': False,
                'message': '解析数据集失败，请检查文件格式'
            }), 400
        
        # 生成数据集ID
        import uuid
        dataset_id = str(uuid.uuid4())
        
        # 创建用户数据集目录
        user_dataset_dir = os.path.join('datasets', f'user_{user_id}')
        os.makedirs(user_dataset_dir, exist_ok=True)
        
        # 保存数据集为JSON格式
        dataset_path = os.path.join(user_dataset_dir, f'{dataset_id}.json')
        with open(dataset_path, 'w') as f:
            json.dump({
                'dataset_id': dataset_id,
                'user_id': user_id,
                'filename': filename,
                'coordinates': coordinates,
                'num_cities': len(coordinates),
                'upload_time': datetime.now().isoformat()
            }, f)
        
        # 计算坐标范围
        coords_array = np.array(coordinates)
        coord_min = coords_array.min(axis=0)
        coord_max = coords_array.max(axis=0)
        coord_range = f"X: [{coord_min[0]:.2f}, {coord_max[0]:.2f}], Y: [{coord_min[1]:.2f}, {coord_max[1]:.2f}]"
        
        return jsonify({
            'success': True,
            'message': f'数据集上传成功！包含 {len(coordinates)} 个城市',
            'dataset_id': dataset_id,
            'dataset_info': {
                'filename': filename,
                'num_cities': len(coordinates),
                'coord_range': coord_range
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'上传失败: {str(e)}'
        }), 500


def parse_dataset(content, file_ext):
    """解析不同格式的TSP数据集文件"""
    try:
        if file_ext == 'json':
            # JSON格式: {"coordinates": [[x1,y1], [x2,y2], ...]}
            data = json.loads(content)
            if 'coordinates' in data:
                return data['coordinates']
            else:
                return None
                
        elif file_ext == 'txt':
            # TXT格式: 每行一个坐标对 "x y"
            coordinates = []
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):  # 跳过空行和注释
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            x = float(parts[0])
                            y = float(parts[1])
                            coordinates.append([x, y])
                        except ValueError:
                            continue
            return coordinates if len(coordinates) > 0 else None
            
        elif file_ext == 'tsp':
            # TSPLIB格式
            coordinates = []
            in_coord_section = False
            lines = content.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('NODE_COORD_SECTION'):
                    in_coord_section = True
                    continue
                    
                if line in ['EOF', 'DISPLAY_DATA_SECTION', 'EDGE_WEIGHT_SECTION']:
                    break
                    
                if in_coord_section and line:
                    parts = line.split()
                    if len(parts) >= 3:  # TSPLIB格式: index x y
                        try:
                            x = float(parts[1])
                            y = float(parts[2])
                            coordinates.append([x, y])
                        except (ValueError, IndexError):
                            continue
            
            return coordinates if len(coordinates) > 0 else None
            
    except Exception as e:
        print(f"解析数据集错误: {str(e)}")
        return None


@app.route('/api/list_datasets', methods=['GET'])
@login_required
def list_datasets():
    """获取当前用户的所有数据集 - 需要登录"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        # 获取用户数据集目录
        user_dataset_dir = os.path.join('datasets', f'user_{user_id}')
        
        if not os.path.exists(user_dataset_dir):
            return jsonify({
                'success': True,
                'datasets': []
            })
        
        # 读取所有数据集文件
        datasets = []
        for filename in os.listdir(user_dataset_dir):
            if filename.endswith('.json'):
                dataset_path = os.path.join(user_dataset_dir, filename)
                try:
                    with open(dataset_path, 'r') as f:
                        dataset_data = json.load(f)
                        datasets.append({
                            'dataset_id': dataset_data.get('dataset_id'),
                            'filename': dataset_data.get('filename'),
                            'num_cities': dataset_data.get('num_cities'),
                            'upload_time': dataset_data.get('upload_time')
                        })
                except Exception as e:
                    print(f"读取数据集失败: {filename}, {str(e)}")
        
        # 按上传时间倒序排序
        datasets.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'datasets': datasets
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取数据集列表失败: {str(e)}'
        }), 500


@app.route('/api/delete_dataset', methods=['POST'])
@login_required
def delete_dataset():
    """删除指定的数据集 - 需要登录"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        data = request.json
        dataset_id = data.get('dataset_id')
        
        if not dataset_id:
            return jsonify({
                'success': False,
                'message': '缺少数据集ID'
            }), 400
        
        # 构建数据集文件路径
        dataset_path = os.path.join('datasets', f'user_{user_id}', f'{dataset_id}.json')
        
        if not os.path.exists(dataset_path):
            return jsonify({
                'success': False,
                'message': '数据集不存在'
            }), 404
        
        # 删除文件
        os.remove(dataset_path)
        
        return jsonify({
            'success': True,
            'message': '数据集已删除'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'删除数据集失败: {str(e)}'
        }), 500


# ============================================
# 兼容性检查API（新增）
# ============================================

@app.route('/api/compatibility/info', methods=['GET'])
@login_required
def get_compatibility_matrix():
    """获取完整的兼容性信息"""
    try:
        info = get_compatibility_info()
        return jsonify({
            'success': True,
            'data': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取兼容性信息失败: {str(e)}'
        }), 500


@app.route('/api/compatibility/validate', methods=['POST'])
@login_required
def validate_config_combination():
    """验证配置组合是否有效"""
    try:
        data = request.json
        problem = data.get('problem', 'tsp')
        policy = data.get('model', 'attention')
        algorithm = data.get('algorithm', 'reinforce')
        
        is_valid, message, level = validate_combination(problem, policy, algorithm)
        
        return jsonify({
            'success': True,
            'valid': is_valid,
            'message': message,
            'level': level
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'验证失败: {str(e)}'
        }), 500


@app.route('/api/compatibility/constraints/<problem_type>', methods=['GET'])
@login_required
def get_problem_constraints(problem_type):
    """获取特定问题的约束信息"""
    try:
        constraints = get_frontend_constraints(problem_type)
        return jsonify({
            'success': True,
            'data': constraints
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取约束失败: {str(e)}'
        }), 500


@app.route('/api/compatibility/available_policies', methods=['GET'])
@login_required
def get_available_policies_api():
    """获取问题类型支持的策略"""
    try:
        problem = request.args.get('problem', 'tsp')
        policies = get_available_policies(problem)
        return jsonify({
            'success': True,
            'policies': policies
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取策略列表失败: {str(e)}'
        }), 500


@app.route('/api/compatibility/available_algorithms', methods=['GET'])
@login_required
def get_available_algorithms_api():
    """获取问题类型（和策略）支持的算法"""
    try:
        problem = request.args.get('problem', 'tsp')
        policy = request.args.get('policy', None)
        algorithms = get_available_algorithms(problem, policy)
        return jsonify({
            'success': True,
            'algorithms': algorithms
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取算法列表失败: {str(e)}'
        }), 500


@app.route('/api/compatibility/recommend', methods=['GET'])
@login_required
def get_recommended_config():
    """获取推荐的配置组合"""
    try:
        problem = request.args.get('problem', 'tsp')
        preference = request.args.get('preference', 'best')  # best, fast, simple
        
        recommended = get_recommended_combination(problem, preference)
        return jsonify({
            'success': True,
            'recommended': recommended
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取推荐配置失败: {str(e)}'
        }), 500


# ============================================
# 训练相关API
# ============================================

@app.route('/api/start_training', methods=['POST'])
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
            print(f'⚠️  训练配置警告: {message}')
        
        # 生成唯一的会话 ID
        import uuid
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
                print(f"记录训练会话失败: {str(e)}")
        
        # 创建消息队列
        training_queues[session_id] = Queue()
        
        # 根据 RL4CO 是否可用选择训练函数
        if RL4CO_AVAILABLE:
            # 真实训练模式 - 传入必要的全局对象和函数
            training_thread = threading.Thread(
                target=real_rl4co_training,
                args=(config, session_id, user_id, training_queues[session_id], training_status, get_background_db),
                daemon=True
            )
            mode = "真实训练模式"
        else:
            # 模拟训练模式
            training_thread = threading.Thread(
                target=simulate_training,
                args=(config, session_id, user_id),
                daemon=True
            )
            mode = "模拟训练模式"
        
        training_thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': f'训练已启动 ({mode})'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'启动训练失败: {str(e)}'
        }), 500


@app.route('/api/training_progress/<session_id>')
def training_progress(session_id):
    """使用 Server-Sent Events (SSE) 推送训练进度"""
    def generate():
        if session_id not in training_queues:
            yield f"data: {json.dumps({'type': 'error', 'message': '无效的会话 ID'})}\n\n"
            return
        
        queue = training_queues[session_id]
        
        while True:
            try:
                # 从队列中获取消息（阻塞等待）
                message = queue.get(timeout=1)
                yield f"data: {message}\n\n"
                
                # 如果收到完成或错误消息，则结束流
                data = json.loads(message)
                if data['type'] in ['complete', 'error']:
                    break
                    
            except:
                # 超时或队列为空，发送心跳
                if session_id in training_status:
                    if training_status[session_id]['status'] == 'completed':
                        break
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/training_status/<session_id>')
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


@app.route('/api/list_files', methods=['GET'])
@login_required
def list_training_files():
    """列出当前用户的训练产生的文件 - 按训练会话分组 - 需要登录"""
    try:
        # ========== 获取当前用户ID，只显示该用户的文件 ==========
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        # ========== 从数据库获取用户文件列表（按会话分组） ==========
        file_manager = get_file_manager()
        session_mgr = get_session_manager()
        
        if file_manager and session_mgr:
            try:
                # 获取用户的所有训练会话
                user_sessions = session_mgr.get_user_sessions(user_id, limit=100)
                storage_stats = file_manager.get_user_storage_stats(user_id)
                
                # 按会话分组的数据结构
                sessions_data = []
                
                for session in user_sessions:
                    session_id = session['session_id']
                    
                    # 获取该会话的所有文件
                    cursor = file_manager.db.cursor(dictionary=True)
                    cursor.execute("""
                        SELECT * FROM training_files 
                        WHERE session_id = %s AND user_id = %s
                        ORDER BY create_time DESC
                    """, (session_id, user_id))
                    session_files = cursor.fetchall()
                    
                    # 分类文件
                    files_by_type = {
                        'comparison': None,
                        'animation': None,
                        'curve': None,
                        'checkpoint': None,
                        'other': []
                    }
                    
                    total_session_size = 0
                    
                    for file_record in session_files:
                        file_info = {
                            'id': file_record['id'],
                            'name': file_record['file_name'],
                            'type': file_record['file_type'],
                            'size': file_record['file_size'],
                            'size_mb': round(file_record['file_size'] / (1024 * 1024), 2),
                            'path': f"/static/model_plots/user_{user_id}/{file_record['file_name']}" if file_record['file_type'] != 'checkpoint' else None,
                            'create_time': file_record['create_time'].strftime('%Y-%m-%d %H:%M')
                        }
                        
                        total_session_size += file_record['file_size']
                        
                        # 按类型归类（每种类型只保留第一个）
                        file_type = file_record['file_type']
                        
                        # 将 'plot' 类型映射到 'comparison' (都是对比图)
                        if file_type == 'plot':
                            file_type = 'comparison'
                        
                        if file_type in ['comparison', 'animation', 'curve', 'checkpoint']:
                            if files_by_type.get(file_type) is None:
                                files_by_type[file_type] = file_info
                        else:
                            files_by_type['other'].append(file_info)
                    
                    # 只添加有文件的会话
                    if session_files:
                        session_data = {
                            'session_id': session_id,
                            'model_type': session['model_type'],
                            'problem_type': session['problem_type'],
                            'status': session['status'],
                            'start_time': session['start_time'].strftime('%Y-%m-%d %H:%M'),
                            'file_count': len(session_files),
                            'total_size_mb': round(total_session_size / (1024 * 1024), 2),
                            'files': files_by_type
                        }
                        sessions_data.append(session_data)
                
                return jsonify({
                    'success': True,
                    'sessions': sessions_data,
                    'total_sessions': len(sessions_data),
                    'total_size_mb': storage_stats['total_mb'] if storage_stats else 0,
                    'total_files': storage_stats['total_files'] if storage_stats else 0
                })
                
            except Exception as e:
                print(f"从数据库获取文件失败: {str(e)}")
                import traceback
                traceback.print_exc()
                # 降级到文件系统扫描
                pass
        
        # ========== 降级方案：返回空会话列表 ==========
        return jsonify({
            'success': True,
            'sessions': [],
            'total_sessions': 0,
            'total_size_mb': 0,
            'total_files': 0
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取文件列表失败: {str(e)}'
        }), 500


@app.route('/api/delete_file', methods=['POST'])
@login_required
def delete_training_file():
    """删除指定的训练文件 - 需要登录且只能删除自己的文件"""
    try:
        # ========== 获取当前用户ID ==========
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        data = request.json
        file_id = data.get('file_id')  # ========== 使用file_id而非filename ==========
        filename = data.get('filename')  # 兼容旧版
        
        # ========== 使用数据库方式删除（推荐） ==========
        file_manager = get_file_manager()
        if file_id and file_manager:
            success, message = file_manager.delete_file(file_id, user_id)
            if success:
                return jsonify({
                    'success': True,
                    'message': message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': message
                }), 403
        
        # ========== 降级方案：直接删除文件（兼容） ==========
        if not filename:
            return jsonify({
                'success': False,
                'message': '未提供文件名或文件ID'
            }), 400
        
        file_type = data.get('file_type', 'plot')
        USER_PLOTS_DIR = get_user_plot_dir(user_id)
        USER_CHECKPOINTS_DIR = get_user_checkpoint_dir(user_id)
        
        # 确定文件路径（用户专属目录）
        if file_type == 'checkpoint':
            file_path = os.path.join(USER_CHECKPOINTS_DIR, filename)
        else:
            file_path = os.path.join(USER_PLOTS_DIR, filename)
        
        # 安全检查：确保文件在用户目录内
        abs_file_path = os.path.abspath(file_path)
        abs_user_plots = os.path.abspath(USER_PLOTS_DIR)
        abs_user_checkpoints = os.path.abspath(USER_CHECKPOINTS_DIR)
        
        if not (abs_file_path.startswith(abs_user_plots) or abs_file_path.startswith(abs_user_checkpoints)):
            return jsonify({
                'success': False,
                'message': '无效的文件路径或无权访问'
            }), 403
        
        # 删除文件
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({
                'success': True,
                'message': f'文件 {filename} 已删除'
            })
        else:
            return jsonify({
                'success': False,
                'message': '文件不存在'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'删除文件失败: {str(e)}'
        }), 500


@app.route('/api/download_checkpoint/<filename>')
@login_required
def download_checkpoint(filename):
    """下载检查点文件 - 需要登录且只能下载自己的文件"""
    try:
        # ========== 获取当前用户ID ==========
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        USER_CHECKPOINTS_DIR = get_user_checkpoint_dir(user_id)
        
        # 安全检查：防止路径遍历攻击
        safe_filename = os.path.basename(filename)
        file_path = os.path.join(USER_CHECKPOINTS_DIR, safe_filename)
        
        # 验证文件路径在用户目录内
        abs_file_path = os.path.abspath(file_path)
        abs_user_dir = os.path.abspath(USER_CHECKPOINTS_DIR)
        
        if not abs_file_path.startswith(abs_user_dir):
            return jsonify({
                'success': False,
                'message': '无效的文件路径'
            }), 403
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': '文件不存在'
            }), 404
        
        # 发送文件
        from flask import send_file
        return send_file(
            file_path,
            as_attachment=True,
            download_name=safe_filename,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'下载失败: {str(e)}'
        }), 500


@app.route('/api/delete_session', methods=['POST'])
@login_required
def delete_session():
    """删除整个训练会话及其所有文件 - 需要登录"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'message': '缺少session_id参数'
            }), 400
        
        # 验证会话所有权
        session_mgr = get_session_manager()
        if session_mgr and not session_mgr.verify_session_owner(session_id, user_id):
            return jsonify({
                'success': False,
                'message': '无权删除该训练会话'
            }), 403
        
        # 删除所有相关文件
        file_manager = get_file_manager()
        if file_manager:
            # 获取该会话的所有文件
            cursor = file_manager.db.cursor(dictionary=True)
            cursor.execute("""
                SELECT * FROM training_files 
                WHERE session_id = %s AND user_id = %s
            """, (session_id, user_id))
            session_files = cursor.fetchall()
            
            # 删除文件记录和实际文件
            deleted_count = 0
            for file_record in session_files:
                success, msg = file_manager.delete_file(file_record['id'], user_id)
                if success:
                    deleted_count += 1
            
            return jsonify({
                'success': True,
                'message': f'已删除训练会话及其 {deleted_count} 个文件'
            })
        else:
            return jsonify({
                'success': False,
                'message': '文件管理器未初始化'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'删除失败: {str(e)}'
        }), 500


@app.route('/api/delete_by_session', methods=['POST'])
@login_required
def delete_by_session():
    """根据session_id删除相关的所有文件 - 需要登录且只能删除自己的文件"""
    try:
        # ========== 获取当前用户ID ==========
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'message': '未提供session_id'
            }), 400
        
        # 取前8位作为文件名前缀
        session_prefix = session_id[:8]
        deleted_files = []
        
        USER_PLOTS_DIR = get_user_plot_dir(user_id)
        
        # 删除用户的可视化文件
        if os.path.exists(USER_PLOTS_DIR):
            for filename in os.listdir(USER_PLOTS_DIR):
                if session_prefix in filename:
                    file_path = os.path.join(USER_PLOTS_DIR, filename)
                    try:
                        os.remove(file_path)
                        deleted_files.append(filename)
                    except Exception as e:
                        print(f"删除文件 {filename} 失败: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': f'已删除 {len(deleted_files)} 个文件',
            'deleted_files': deleted_files
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'批量删除失败: {str(e)}'
        }), 500


@app.route('/api/clear_all_files', methods=['POST'])
@login_required
def clear_all_files():
    """清空当前用户的所有训练文件（谨慎使用）"""
    try:
        # ========== 获取当前用户ID ==========
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        data = request.json
        confirm = data.get('confirm', False)
        
        if not confirm:
            return jsonify({
                'success': False,
                'message': '需要确认操作'
            }), 400
        
        deleted_count = 0
        
        USER_PLOTS_DIR = get_user_plot_dir(user_id)
        USER_CHECKPOINTS_DIR = get_user_checkpoint_dir(user_id)
        
        # 清空用户的可视化文件
        if os.path.exists(USER_PLOTS_DIR):
            for filename in os.listdir(USER_PLOTS_DIR):
                file_path = os.path.join(USER_PLOTS_DIR, filename)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"删除文件 {filename} 失败: {str(e)}")
        
        # 清空用户的检查点文件
        if os.path.exists(USER_CHECKPOINTS_DIR):
            for filename in os.listdir(USER_CHECKPOINTS_DIR):
                if filename.endswith('.ckpt'):
                    file_path = os.path.join(USER_CHECKPOINTS_DIR, filename)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"删除文件 {filename} 失败: {str(e)}")
        
        # ========== 同时清空数据库记录 ==========
        file_manager = get_file_manager()
        if file_manager:
            try:
                db = get_db()
                if db:
                    cursor = db.cursor()
                    cursor.execute("DELETE FROM training_files WHERE user_id = %s", (user_id,))
                    # 注意：使用 get_db() 返回的连接已经设置了 autocommit=True，所以不需要手动 commit
                    deleted_db_count = cursor.rowcount
                    print(f"已清空用户 {user_id} 的数据库记录，共删除 {deleted_db_count} 条")
            except Exception as e:
                print(f"清空数据库记录失败: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': f'已清空所有文件，共删除 {deleted_count} 个文件'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'清空文件失败: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
