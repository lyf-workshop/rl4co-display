"""
组合优化问题基类
定义所有问题类型的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Tuple


class BaseProblem(ABC):
    """
    组合优化问题基类
    
    所有具体问题类型都应该继承此类并实现抽象方法
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化问题
        
        参数:
            config: 配置字典，包含问题特定的参数
        """
        self.config = config
        self.problem_type = self.get_problem_type()
        
        # 通用参数
        self.num_loc = int(config.get('num_loc', 50))
        self.batch_size = int(config.get('batch_size', 512))
        
        # 问题特定参数（子类可以添加）
        self._init_problem_params()
    
    @abstractmethod
    def get_problem_type(self) -> str:
        """
        返回问题类型标识符
        
        返回:
            str: 问题类型 (如 'tsp', 'cvrp')
        """
        pass
    
    @abstractmethod
    def get_problem_name(self) -> str:
        """
        返回问题完整名称
        
        返回:
            str: 问题全名 (如 'Traveling Salesman Problem')
        """
        pass
    
    @abstractmethod
    def create_environment(self):
        """
        创建RL4CO环境实例
        
        返回:
            RL4CO环境对象
        """
        pass
    
    @abstractmethod
    def get_visualization_functions(self) -> Dict[str, Callable]:
        """
        获取可视化函数集合
        
        返回:
            dict: {
                'animation': 创建动画的函数,
                'comparison': 创建对比图的函数,
                'other': 其他可视化函数
            }
        """
        pass
    
    def _init_problem_params(self):
        """
        初始化问题特定参数
        子类可以重写此方法来添加特定参数
        """
        pass
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        获取问题的默认参数
        
        返回:
            dict: 默认参数字典
        """
        return {
            'num_loc': self.num_loc,
            'batch_size': self.batch_size,
        }
    
    def validate_config(self) -> Tuple[bool, str]:
        """
        验证配置是否有效
        
        返回:
            (bool, str): (是否有效, 错误信息)
        """
        # 基本验证
        if self.num_loc < 2:
            return False, "num_loc必须大于等于2"
        
        if self.batch_size < 1:
            return False, "batch_size必须大于0"
        
        # 子类可以添加更多验证
        return self._validate_problem_params()
    
    def _validate_problem_params(self) -> Tuple[bool, str]:
        """
        验证问题特定参数
        子类可以重写此方法
        """
        return True, ""
    
    def get_env_params(self) -> Dict[str, Any]:
        """
        获取环境初始化参数
        
        返回:
            dict: 传递给RL4CO环境的参数
        """
        return {
            'num_loc': self.num_loc,
        }
    
    def get_problem_description(self) -> str:
        """
        获取问题描述
        
        返回:
            str: 问题的文字描述
        """
        return f"{self.get_problem_name()} - {self.get_problem_type().upper()}"
    
    def get_problem_features(self) -> List[str]:
        """
        获取问题特征列表
        
        返回:
            list: 问题特征
        """
        return []
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"type={self.problem_type}, "
            f"num_loc={self.num_loc})"
        )


