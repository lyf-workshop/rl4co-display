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
        创建POMO策略网络

        说明：
            rl4co 0.6.0 中 **不存在** 独立的 POMOPolicy 类。POMO 是一个训练*模型*
            （见 rl4co.models.zoo.pomo.POMO，继承自 REINFORCE），其能力来自
            “共享基线 + 多起点解码 + 数据增强”，而 policy 本身就是 AttentionModelPolicy。
            因此这里按 POMO 论文/官方默认构建 AM 策略，真正的 POMO 训练逻辑由
            base_trainer._create_pomo_model 用 POMO 模型包裹本策略实现。

        参数:
            env: RL4CO环境

        返回:
            AttentionModelPolicy 实例（POMO 论文版配置）
        """
        try:
            from rl4co.models import AttentionModelPolicy
        except ImportError:
            raise ImportError(
                "RL4CO库未安装，无法创建POMO策略。\n"
                "请安装: pip install rl4co"
            )

        # POMO 官方默认策略配置（rl4co 0.6.0 POMO.__init__）：
        #   num_encoder_layers=6, normalization="instance", use_graph_context=False
        # use_graph_context=False：论文不使用图上下文，避免对训练图规模过拟合。
        policy = AttentionModelPolicy(
            env_name=env.name,
            embed_dim=self.embed_dim,
            num_encoder_layers=self.num_encoder_layers,  # _init_policy_params 已确保 >= 6
            num_heads=self.num_heads,
            normalization="instance",
            use_graph_context=False,
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






