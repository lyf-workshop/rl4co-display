"""
A2C (Advantage Actor-Critic) 算法实现
Actor-Critic方法，比REINFORCE更稳定
"""

from typing import Dict, Any
from .base_algorithm import BaseAlgorithm


class A2CAlgorithm(BaseAlgorithm):
    """
    A2C算法类
    
    特点：
        - 使用Critic网络估计价值函数
        - 方差比REINFORCE小
        - 收敛速度快
        - 性能稳定
    """
    
    def _init_algorithm_params(self):
        """初始化A2C特定参数"""
        self.value_loss_coef = float(self.config.get('value_loss_coef', 0.5))
        self.entropy_coef = float(self.config.get('entropy_coef', 0.01))
        self.use_gae = self.config.get('use_gae', True)
        self.gae_lambda = float(self.config.get('gae_lambda', 0.95))
    
    def get_algorithm_name(self) -> str:
        return 'a2c'
    
    def create_model(self, env, policy):
        """
        创建A2C模型
        
        参数:
            env: RL4CO环境
            policy: 策略网络
        
        返回:
            A2C模型实例
        """
        try:
            from rl4co.models import A2C
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或不支持A2C，无法创建A2C模型。\n"
                "请确保RL4CO版本支持A2C算法"
            )
        
        model = A2C(
            env,
            policy,
            value_loss_coef=self.value_loss_coef,
            entropy_coef=self.entropy_coef,
            batch_size=self.batch_size,
            train_data_size=self.train_data_size,
            val_data_size=self.val_data_size,
            optimizer_kwargs={"lr": self.learning_rate},
        )
        
        return model
    
    def get_model_params(self) -> Dict[str, Any]:
        """获取A2C模型参数"""
        params = super().get_model_params()
        params.update({
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef,
            'use_gae': self.use_gae,
            'gae_lambda': self.gae_lambda,
        })
        return params
    
    def _validate_algorithm_params(self):
        """验证A2C参数"""
        if self.value_loss_coef < 0:
            return False, "value_loss_coef必须大于等于0"
        
        if self.entropy_coef < 0:
            return False, "entropy_coef必须大于等于0"
        
        if self.gae_lambda < 0 or self.gae_lambda > 1:
            return False, "gae_lambda必须在[0, 1]范围内"
        
        return True, ""
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        info = super().get_algorithm_info()
        info.update({
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef,
            'use_gae': self.use_gae,
        })
        return info






