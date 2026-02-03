"""
VRPTW (Vehicle Routing Problem with Time Windows) - 带时间窗的车辆路径问题
"""

from typing import Dict, Any, Callable
from .base_problem import BaseProblem


class VRPTWProblem(BaseProblem):
    """
    带时间窗的车辆路径问题类
    
    目标：在满足容量约束和时间窗约束的前提下，规划最短配送路径
    特点：
        - 有容量约束
        - 有时间窗约束（每个客户必须在指定时间段内访问）
        - 可多次返回仓库
        - 实用性强（快递配送、外卖配送）
    """
    
    def _init_problem_params(self):
        """初始化VRPTW特定参数"""
        self.vehicle_capacity = float(self.config.get('vehicle_capacity', 1.0))
        self.num_vehicles = int(self.config.get('num_vehicles', 1))
        
        # 时间窗参数
        self.time_window_width = float(self.config.get('time_window_width', 100.0))  # 时间窗宽度
        self.service_time = float(self.config.get('service_time', 10.0))  # 服务时间
        self.max_time = float(self.config.get('max_time', 480.0))  # 最大配送时间（如8小时=480分钟）
        self.speed = float(self.config.get('speed', 1.0))  # 车辆速度
        
        # 时间窗约束严格程度
        self.hard_time_windows = self.config.get('hard_time_windows', True)  # True=硬时间窗，False=软时间窗
    
    def get_problem_type(self) -> str:
        return 'vrptw'
    
    def get_problem_name(self) -> str:
        return 'Vehicle Routing Problem with Time Windows'
    
    def create_environment(self):
        """
        创建VRPTW环境
        
        返回:
            VRPTWEnv或CVRPEnvWithTimeWindows: VRPTW环境实例
        """
        try:
            # 尝试导入原生VRPTW环境
            try:
                from rl4co.envs import VRPTWEnv
                env = VRPTWEnv(generator_params=self.get_env_params())
                print("✅ 使用RL4CO原生VRPTWEnv")
            except (ImportError, AttributeError):
                # 使用自定义VRPTW环境包装器（带正确的reward计算）
                print("⚠️ Warning: VRPTWEnv not found, using custom wrapper with time window rewards")
                from modules.envs.vrptw_env_wrapper import CVRPEnvWithTimeWindows
                
                generator_params = {
                    'num_loc': self.num_loc,
                    'vehicle_capacity': self.vehicle_capacity,
                }
                
                time_window_params = {
                    'time_window_width': self.time_window_width,
                    'service_time': self.service_time,
                    'max_time': self.max_time,
                    'hard_time_windows': self.hard_time_windows,
                }
                
                env = CVRPEnvWithTimeWindows(generator_params, time_window_params)
            
            return env
            
        except ImportError:
            raise ImportError(
                "RL4CO库未安装，无法创建VRPTW环境。\n"
                "请安装: pip install rl4co"
            )
    
    def get_visualization_functions(self) -> Dict[str, Callable]:
        """
        获取VRPTW可视化函数
        
        返回:
            dict: {
                'animation': 创建配送路线动画（带时间轴）,
                'comparison': 创建对比图,
                'time_schedule': 创建时间调度图
            }
        """
        from modules.rl_training.visualizations.vrptw_viz import (
            create_vrptw_route_animation,
            create_vrptw_comparison_plot,
            create_vrptw_time_schedule
        )
        
        return {
            'animation': create_vrptw_route_animation,
            'comparison': create_vrptw_comparison_plot,
            'time_schedule': create_vrptw_time_schedule,
        }
    
    def get_problem_description(self) -> str:
        tw_type = "硬时间窗" if self.hard_time_windows else "软时间窗"
        return (
            f"带时间窗的车辆路径问题 (VRPTW)\n"
            f"目标: 规划车辆配送路径，访问{self.num_loc}个客户\n"
            f"约束: \n"
            f"  - 车辆容量={self.vehicle_capacity}\n"
            f"  - 时间窗宽度={self.time_window_width}\n"
            f"  - 服务时间={self.service_time}\n"
            f"  - 最大配送时间={self.max_time}\n"
            f"  - 约束类型={tw_type}"
        )
    
    def get_problem_features(self) -> list:
        return [
            '容量约束',
            '时间窗约束',
            '必须在指定时间段访问',
            '适用于快递/外卖配送',
            '比CVRP更复杂',
            '实时性要求高',
        ]
    
    def get_env_params(self) -> Dict[str, Any]:
        """VRPTW环境参数"""
        return {
            'num_loc': self.num_loc,
            'vehicle_capacity': self.vehicle_capacity,
            # 时间窗参数
            'time_window_width': self.time_window_width,
            'service_time': self.service_time,
            'max_time': self.max_time,
            'speed': self.speed,
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取VRPTW默认参数"""
        params = super().get_default_params()
        params.update({
            'vehicle_capacity': self.vehicle_capacity,
            'num_vehicles': self.num_vehicles,
            'time_window_width': self.time_window_width,
            'service_time': self.service_time,
            'max_time': self.max_time,
            'speed': self.speed,
            'hard_time_windows': self.hard_time_windows,
        })
        return params
    
    def _validate_problem_params(self):
        """VRPTW参数验证"""
        # 容量验证
        if self.vehicle_capacity <= 0:
            return False, "vehicle_capacity必须大于0"
        
        if self.num_vehicles < 1:
            return False, "num_vehicles必须大于等于1"
        
        # 时间窗验证
        if self.time_window_width <= 0:
            return False, "time_window_width必须大于0"
        
        if self.service_time < 0:
            return False, "service_time必须大于等于0"
        
        if self.max_time <= 0:
            return False, "max_time必须大于0"
        
        if self.speed <= 0:
            return False, "speed必须大于0"
        
        # 逻辑验证
        if self.time_window_width > self.max_time:
            return False, "time_window_width不应超过max_time"
        
        if self.num_loc > 100:
            return False, "VRPTW客户数量不建议超过100（时间窗约束使问题更复杂）"
        
        return True, ""
    
    def get_time_window_info(self) -> str:
        """获取时间窗说明"""
        tw_type = "硬时间窗" if self.hard_time_windows else "软时间窗"
        return (
            f"时间窗约束特性 ({tw_type}):\n"
            f"- 时间窗宽度: {self.time_window_width}\n"
            f"- 服务时间: {self.service_time}\n"
            f"- 最大配送时间: {self.max_time}\n"
            f"- 车辆速度: {self.speed}\n"
            "\n"
            "硬时间窗 vs 软时间窗:\n"
            "- 硬时间窗: 必须严格遵守，超出时间窗则不可行\n"
            "- 软时间窗: 可以违反，但会产生惩罚成本\n"
            "\n"
            "适用场景:\n"
            "- 快递配送（客户指定送达时间）\n"
            "- 外卖配送（30分钟必达）\n"
            "- 医疗配送（药品送达时效）\n"
            "- 预约上门服务"
        )



