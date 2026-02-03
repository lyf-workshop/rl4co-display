"""
PPO (Proximal Policy Optimization) 算法实现
工业界首选的强化学习算法
"""

from typing import Dict, Any
from .base_algorithm import BaseAlgorithm


class PPOAlgorithm(BaseAlgorithm):
    """
    PPO算法类
    
    特点：
        - 训练稳定性强
        - 对超参数不敏感
        - 工业界广泛应用
        - 使用重要性采样和裁剪目标
    """
    
    def _init_algorithm_params(self):
        """初始化PPO特定参数"""
        self.clip_ratio = float(self.config.get('clip_ratio', 0.2))
        self.value_loss_coef = float(self.config.get('value_loss_coef', 0.5))
        self.entropy_coef = float(self.config.get('entropy_coef', 0.01))
        self.epochs_per_update = int(self.config.get('epochs_per_update', 4))
        self.gae_lambda = float(self.config.get('gae_lambda', 0.95))
    
    def get_algorithm_name(self) -> str:
        return 'ppo'
    
    def create_model(self, env, policy):
        """
        创建PPO模型
        
        参数:
            env: RL4CO环境
            policy: 策略网络
        
        返回:
            PPO模型实例
        """
        try:
            from rl4co.models import PPO
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或不支持PPO，无法创建PPO模型。\n"
                "请确保RL4CO版本支持PPO算法"
            )
        
        model = PPO(
            env,
            policy,
            clip_ratio=self.clip_ratio,
            value_loss_coef=self.value_loss_coef,
            entropy_coef=self.entropy_coef,
            epochs_per_update=self.epochs_per_update,
            batch_size=self.batch_size,
            train_data_size=self.train_data_size,
            val_data_size=self.val_data_size,
            optimizer_kwargs={"lr": self.learning_rate},
        )
        
        return model
    
    def get_model_params(self) -> Dict[str, Any]:
        """获取PPO模型参数"""
        params = super().get_model_params()
        params.update({
            'clip_ratio': self.clip_ratio,
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef,
            'epochs_per_update': self.epochs_per_update,
            'gae_lambda': self.gae_lambda,
        })
        return params
    
    def _validate_algorithm_params(self):
        """验证PPO参数"""
        if self.clip_ratio <= 0 or self.clip_ratio > 1:
            return False, "clip_ratio必须在(0, 1]范围内"
        
        if self.value_loss_coef < 0:
            return False, "value_loss_coef必须大于等于0"
        
        if self.entropy_coef < 0:
            return False, "entropy_coef必须大于等于0"
        
        if self.epochs_per_update < 1:
            return False, "epochs_per_update必须大于0"
        
        return True, ""
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        info = super().get_algorithm_info()
        info.update({
            'clip_ratio': self.clip_ratio,
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef,
            'epochs_per_update': self.epochs_per_update,
        })
        return info






