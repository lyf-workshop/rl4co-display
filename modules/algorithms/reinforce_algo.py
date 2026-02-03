"""
REINFORCE算法实现
最基础的策略梯度算法
"""

from typing import Dict, Any
from .base_algorithm import BaseAlgorithm


class REINFORCEAlgorithm(BaseAlgorithm):
    """
    REINFORCE算法类
    
    特点：
        - 最简单的策略梯度方法
        - 使用蒙特卡洛采样估计梯度
        - 方差较大，可通过baseline降低
    """
    
    def _init_algorithm_params(self):
        """初始化REINFORCE特定参数"""
        self.baseline = self.config.get('baseline', 'rollout')
        self.entropy_coef = float(self.config.get('entropy_coef', 0.0))
    
    def get_algorithm_name(self) -> str:
        return 'reinforce'
    
    def create_model(self, env, policy):
        """
        创建REINFORCE模型
        
        参数:
            env: RL4CO环境
            policy: 策略网络
        
        返回:
            REINFORCE模型实例
        """
        try:
            from rl4co.models import REINFORCE
        except ImportError:
            raise ImportError(
                "RL4CO库未安装，无法创建REINFORCE模型。\n"
                "请安装: pip install rl4co"
            )
        
        model = REINFORCE(
            env,
            policy,
            baseline=self.baseline,
            batch_size=self.batch_size,
            train_data_size=self.train_data_size,
            val_data_size=self.val_data_size,
            optimizer_kwargs={"lr": self.learning_rate},
        )
        
        return model
    
    def get_model_params(self) -> Dict[str, Any]:
        """获取REINFORCE模型参数"""
        params = super().get_model_params()
        params.update({
            'baseline': self.baseline,
            'entropy_coef': self.entropy_coef,
        })
        return params
    
    def _validate_algorithm_params(self):
        """验证REINFORCE参数"""
        valid_baselines = ['rollout', 'exponential', 'critic', 'no']
        if self.baseline not in valid_baselines:
            return False, f"baseline必须是 {valid_baselines} 之一"
        
        if self.entropy_coef < 0 or self.entropy_coef > 1:
            return False, "entropy_coef必须在[0, 1]范围内"
        
        return True, ""
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        info = super().get_algorithm_info()
        info.update({
            'baseline': self.baseline,
            'entropy_coef': self.entropy_coef,
        })
        return info






