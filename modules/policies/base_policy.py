"""
策略网络基类
定义所有策略模型的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePolicy(ABC):
    """
    策略网络基类
    
    封装RL4CO的策略模型，提供统一接口
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化策略
        
        参数:
            config: 配置字典，包含策略特定的参数
        """
        self.config = config
        self.policy_name = self.get_policy_name()
        
        # 通用参数
        self.embed_dim = int(config.get('embed_dim', 128))
        self.num_encoder_layers = int(config.get('num_encoder_layers', 3))
        self.num_heads = int(config.get('num_heads', 8))
        
        # 策略特定参数（子类可以添加）
        self._init_policy_params()
    
    @abstractmethod
    def get_policy_name(self) -> str:
        """
        返回策略名称标识符
        
        返回:
            str: 策略名称 (如 'attention', 'pomo')
        """
        pass
    
    @abstractmethod
    def create_policy(self, env):
        """
        创建策略网络实例
        
        参数:
            env: RL4CO环境对象
        
        返回:
            RL4CO策略对象
        """
        pass
    
    def _init_policy_params(self):
        """
        初始化策略特定参数
        子类可以重写此方法来添加特定参数
        """
        pass
    
    def get_policy_params(self) -> Dict[str, Any]:
        """
        获取策略初始化参数
        
        返回:
            dict: 传递给RL4CO策略的参数
        """
        return {
            'embed_dim': self.embed_dim,
            'num_encoder_layers': self.num_encoder_layers,
            'num_heads': self.num_heads,
        }
    
    def validate_config(self) -> tuple[bool, str]:
        """
        验证配置是否有效
        
        返回:
            (bool, str): (是否有效, 错误信息)
        """
        # 基本验证
        if self.embed_dim < 1:
            return False, "embed_dim必须大于0"
        
        if self.num_encoder_layers < 1:
            return False, "num_encoder_layers必须大于0"
        
        if self.num_heads < 1:
            return False, "num_heads必须大于0"
        
        if self.embed_dim % self.num_heads != 0:
            return False, "embed_dim必须能被num_heads整除"
        
        # 子类可以添加更多验证
        return self._validate_policy_params()
    
    def _validate_policy_params(self) -> tuple[bool, str]:
        """
        验证策略特定参数
        子类可以重写此方法
        """
        return True, ""
    
    def get_policy_info(self) -> Dict[str, Any]:
        """
        获取策略信息（用于前端显示）
        
        返回:
            dict: 策略信息
        """
        return {
            'name': self.policy_name,
            'embed_dim': self.embed_dim,
            'num_encoder_layers': self.num_encoder_layers,
            'num_heads': self.num_heads,
        }
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"policy={self.policy_name}, "
            f"embed_dim={self.embed_dim}, "
            f"layers={self.num_encoder_layers})"
        )






