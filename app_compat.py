"""
兼容性检查相关路由模块
包含配置验证、推荐配置、可用算法查询等功能
"""
from flask import Blueprint, request, jsonify
from auth_module import login_required
from modules.compatibility import (
    validate_combination,
    get_available_policies,
    get_available_algorithms,
    get_recommended_combination,
    get_compatibility_info,
    get_frontend_constraints
)

compat_bp = Blueprint('compat', __name__)


@compat_bp.route('/api/compatibility/info', methods=['GET'])
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


@compat_bp.route('/api/compatibility/validate', methods=['POST'])
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


@compat_bp.route('/api/compatibility/constraints/<problem_type>', methods=['GET'])
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


@compat_bp.route('/api/compatibility/available_policies', methods=['GET'])
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


@compat_bp.route('/api/compatibility/available_algorithms', methods=['GET'])
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


@compat_bp.route('/api/compatibility/recommend', methods=['GET'])
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


