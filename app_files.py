"""
文件管理相关路由模块
包含数据集上传/管理、训练文件管理、检查点下载等功能
"""
from flask import Blueprint, request, jsonify, send_file
import os
import json
import uuid
import numpy as np
from datetime import datetime
import logging
from auth_module import (
    login_required,
    get_db,
    get_session_manager,
    get_file_manager,
    get_current_user_id,
    get_user_plot_dir,
    get_user_checkpoint_dir
)

files_bp = Blueprint('files', __name__)
logger = logging.getLogger('rl4co_display')


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
        logger.error(f"解析数据集错误: {str(e)}", exc_info=True)
        return None


@files_bp.route('/api/upload_dataset', methods=['POST'])
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
        logger.error(f"上传数据集失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': '上传失败，请稍后重试'}), 500


@files_bp.route('/api/list_datasets', methods=['GET'])
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
                    logger.error(f"读取数据集失败: {filename}, {str(e)}")
        
        # 按上传时间倒序排序
        datasets.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'datasets': datasets
        })
        
    except Exception as e:
        logger.error(f"获取数据集列表失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': '获取数据集列表失败，请稍后重试'}), 500


@files_bp.route('/api/delete_dataset', methods=['POST'])
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
        logger.error(f"删除数据集失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': '删除数据集失败，请稍后重试'}), 500


@files_bp.route('/api/list_files', methods=['GET'])
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
                logger.error(f"从数据库获取文件失败: {str(e)}", exc_info=True)
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
        logger.error(f"获取文件列表失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': '获取文件列表失败，请稍后重试'}), 500


@files_bp.route('/api/delete_file', methods=['POST'])
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
        logger.error(f"删除文件失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': '删除文件失败，请稍后重试'}), 500


@files_bp.route('/api/download_checkpoint/<filename>')
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
        return send_file(
            file_path,
            as_attachment=True,
            download_name=safe_filename,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"下载检查点失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': '下载失败，请稍后重试'}), 500


@files_bp.route('/api/delete_session', methods=['POST'])
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
        logger.error(f"删除会话文件失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': '删除失败，请稍后重试'}), 500


@files_bp.route('/api/delete_by_session', methods=['POST'])
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
                        logger.warning(f"删除文件 {filename} 失败: {e}")
        
        return jsonify({
            'success': True,
            'message': f'已删除 {len(deleted_files)} 个文件',
            'deleted_files': deleted_files
        })
        
    except Exception as e:
        logger.error(f"批量删除文件失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': '批量删除失败，请稍后重试'}), 500


@files_bp.route('/api/clear_all_files', methods=['POST'])
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
                        logger.warning(f"删除文件 {filename} 失败: {e}")
        
        # 清空用户的检查点文件
        if os.path.exists(USER_CHECKPOINTS_DIR):
            for filename in os.listdir(USER_CHECKPOINTS_DIR):
                if filename.endswith('.ckpt'):
                    file_path = os.path.join(USER_CHECKPOINTS_DIR, filename)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"删除文件 {filename} 失败: {e}")
        
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
                    logger.info(f"已清空用户 {user_id} 的数据库记录，共删除 {deleted_db_count} 条")
            except Exception as e:
                logger.error(f"清空数据库记录失败: {str(e)}", exc_info=True)
        
        return jsonify({
            'success': True,
            'message': f'已清空所有文件，共删除 {deleted_count} 个文件'
        })
        
    except Exception as e:
        logger.error(f"清空文件失败: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'message': '清空文件失败，请稍后重试'}), 500

