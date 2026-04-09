"""
日志配置模块
提供统一的日志配置和错误码常量
"""
import logging
import os
from datetime import datetime

# ============================================
# 错误码常量定义
# ============================================

class ErrorCode:
    """API错误码常量"""
    # 认证相关 (1xxx)
    AUTH_REQUIRED = '1001'  # 未登录
    AUTH_FAILED = '1002'  # 认证失败
    AUTH_MODULE_ERROR = '1003'  # 认证模块未初始化
    PERMISSION_DENIED = '1004'  # 权限不足
    
    # 参数相关 (2xxx)
    PARAM_MISSING = '2001'  # 缺少必需参数
    PARAM_INVALID = '2002'  # 参数格式错误
    CONFIG_INVALID = '2003'  # 配置验证失败
    
    # 资源相关 (3xxx)
    RESOURCE_NOT_FOUND = '3001'  # 资源不存在
    FILE_NOT_FOUND = '3002'  # 文件不存在
    DATASET_NOT_FOUND = '3003'  # 数据集不存在
    SESSION_NOT_FOUND = '3004'  # 训练会话不存在
    
    # 操作相关 (4xxx)
    OPERATION_FAILED = '4001'  # 操作失败
    UPLOAD_FAILED = '4002'  # 上传失败
    DELETE_FAILED = '4003'  # 删除失败
    PARSE_FAILED = '4004'  # 解析失败
    TRAINING_START_FAILED = '4005'  # 训练启动失败
    
    # 系统相关 (5xxx)
    DATABASE_ERROR = '5001'  # 数据库错误
    INTERNAL_ERROR = '5002'  # 内部错误


# ============================================
# 日志配置
# ============================================

def setup_logging(app_name='rl4co_display', log_level=logging.INFO):
    """
    配置应用日志系统
    
    参数:
        app_name: 应用名称，用于日志文件命名
        log_level: 日志级别（默认INFO）
    
    返回:
        配置好的logger实例
    """
    # 创建logs目录
    logs_dir = 'logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    # 生成日志文件名（按日期）
    log_filename = os.path.join(
        logs_dir, 
        f"{app_name}_{datetime.now().strftime('%Y%m%d')}.log"
    )
    
    # 创建logger
    logger = logging.getLogger(app_name)
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


# ============================================
# 便捷的错误响应生成函数
# ============================================

def error_response(message, code=None, status_code=400):
    """
    生成标准的错误响应
    
    参数:
        message: 错误消息
        code: 错误码（可选）
        status_code: HTTP状态码（默认400）
    
    返回:
        (dict, int) 元组，包含错误信息和HTTP状态码
    """
    response = {
        'success': False,
        'message': message
    }
    if code:
        response['code'] = code
    return response, status_code


def success_response(data=None, message='操作成功'):
    """
    生成标准的成功响应
    
    参数:
        data: 返回的数据（可选）
        message: 成功消息
    
    返回:
        dict 包含成功信息
    """
    response = {
        'success': True,
        'message': message
    }
    if data is not None:
        if isinstance(data, dict):
            response.update(data)
        else:
            response['data'] = data
    return response


# ============================================
# 日志装饰器
# ============================================

def log_api_call(logger):
    """
    API调用日志装饰器
    
    用法:
        @log_api_call(logger)
        def my_api_function():
            pass
    """
    def decorator(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"API调用: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"API成功: {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"API失败: {func.__name__} - {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator



