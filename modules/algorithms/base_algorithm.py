"""
强化学习算法基类
定义所有RL算法的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAlgorithm(ABC):
    """
    强化学习算法基类
    
    所有具体算法都应该继承此类并实现抽象方法
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化算法
        
        参数:
            config: 配置字典，包含算法特定的参数
        """
        self.config = config
        self.algorithm_name = self.get_algorithm_name()
        
        # 通用参数
        self.batch_size = int(config.get('batch_size', 512))
        self.learning_rate = float(config.get('learning_rate', 1e-4))
        self.train_data_size = int(config.get('train_data_size', 10_000))
        self.val_data_size = int(config.get('val_data_size', 1_000))
        
        # 算法特定参数（子类可以添加）
        self._init_algorithm_params()
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """
        返回算法名称标识符
        
        返回:
            str: 算法名称 (如 'reinforce', 'ppo')
        """
        pass
    
    @abstractmethod
    def create_model(self, env, policy):
        """
        创建RL模型实例
        
        参数:
            env: RL4CO环境对象
            policy: 策略网络对象
        
        返回:
            RL4CO模型对象
        """
        pass
    
    def _init_algorithm_params(self):
        """
        初始化算法特定参数
        子类可以重写此方法来添加特定参数
        """
        pass
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        获取模型初始化参数
        
        返回:
            dict: 传递给RL4CO模型的参数
        """
        return {
            'batch_size': self.batch_size,
            'train_data_size': self.train_data_size,
            'val_data_size': self.val_data_size,
            'optimizer_kwargs': {'lr': self.learning_rate},
        }
    
    def validate_config(self) -> tuple[bool, str]:
        """
        验证配置是否有效
        
        返回:
            (bool, str): (是否有效, 错误信息)
        """
        # 基本验证
        if self.batch_size < 1:
            return False, "batch_size必须大于0"
        
        if self.learning_rate <= 0:
            return False, "learning_rate必须大于0"
        
        # 子类可以添加更多验证
        return self._validate_algorithm_params()
    
    def _validate_algorithm_params(self) -> tuple[bool, str]:
        """
        验证算法特定参数
        子类可以重写此方法
        """
        return True, ""
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        获取算法信息（用于前端显示）
        
        返回:
            dict: 算法信息
        """
        return {
            'name': self.algorithm_name,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
        }
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"algorithm={self.algorithm_name}, "
            f"batch_size={self.batch_size}, "
            f"lr={self.learning_rate})"
        )






