"""
POMO (Policy Optimization with Multiple Optima) 策略网络封装
多起点优化，适合对称问题
"""

from typing import Dict, Any
from .base_policy import BasePolicy


class POMOPolicyWrapper(BasePolicy):
    """
    POMO策略封装类
    
    特点：
        - 同时从多个起点开始构建路径
        - 利用问题对称性
        - 质量高于单起点AM
        - 适合TSP、CVRP等对称问题
    """
    
    def _init_policy_params(self):
        """初始化POMO特定参数"""
        self.num_starts = int(self.config.get('num_starts', 50))
        # POMO通常需要更深的网络
        if self.num_encoder_layers < 6:
            self.num_encoder_layers = 6
    
    def get_policy_name(self) -> str:
        return 'pomo'
    
    def create_policy(self, env):
        """
        创建POMO策略
        
        参数:
            env: RL4CO环境
        
        返回:
            POMO策略实例
        """
        try:
            # 尝试导入POMO专用策略
            try:
                from rl4co.models.nn import POMOPolicy
                policy = POMOPolicy(
                    env_name=env.name,
                    embed_dim=self.embed_dim,
                    num_encoder_layers=self.num_encoder_layers,
                    num_heads=self.num_heads,
                    num_starts=self.num_starts,
                )
            except (ImportError, AttributeError):
                # 降级使用AttentionModelPolicy + POMO解码
                from rl4co.models.nn import AttentionModelPolicy
                policy = AttentionModelPolicy(
                    env_name=env.name,
                    embed_dim=self.embed_dim,
                    num_encoder_layers=self.num_encoder_layers,
                    num_heads=self.num_heads,
                )
                # 标记为POMO模式
                policy.use_pomo = True
                policy.num_starts = self.num_starts
        except ImportError:
            raise ImportError(
                "RL4CO库未安装，无法创建POMO策略。\n"
                "请安装: pip install rl4co"
            )
        
        return policy
    
    def get_policy_params(self) -> Dict[str, Any]:
        """获取POMO策略参数"""
        params = super().get_policy_params()
        params['num_starts'] = self.num_starts
        return params
    
    def _validate_policy_params(self):
        """验证POMO参数"""
        if self.num_starts < 1:
            return False, "num_starts必须大于0"
        
        if self.num_starts > 100:
            return False, "num_starts不建议超过100（显存限制）"
        
        return True, ""
    
    def get_policy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        info = super().get_policy_info()
        info['num_starts'] = self.num_starts
        return info






