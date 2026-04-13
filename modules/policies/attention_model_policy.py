"""
Attention Model 策略网络封装
经典的基于Transformer的构造式模型
"""

from typing import Dict, Any
from .base_policy import BasePolicy


class AttentionModelPolicyWrapper(BasePolicy):
    """
    Attention Model策略封装类
    
    特点：
        - 基于Transformer架构
        - 编码器-解码器结构
        - 适合各种路径规划问题
        - 推理速度快
    """
    
    def get_policy_name(self) -> str:
        return 'attention'
    
    def create_policy(self, env):
        """
        创建AttentionModel策略
        
        参数:
            env: RL4CO环境
        
        返回:
            AttentionModelPolicy实例
        """
        try:
            from rl4co.models import AttentionModelPolicy
        except ImportError:
            raise ImportError(
                "RL4CO库未安装，无法创建AttentionModelPolicy。\n"
                "请安装: pip install rl4co"
            )
        
        policy = AttentionModelPolicy(
            env_name=env.name,
            embed_dim=self.embed_dim,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
        )
        
        return policy
    
    def get_policy_params(self) -> Dict[str, Any]:
        """获取AM策略参数"""
        return super().get_policy_params()
    
    def _validate_policy_params(self):
        """验证AM参数"""
        # AM没有特殊的参数验证
        return True, ""






