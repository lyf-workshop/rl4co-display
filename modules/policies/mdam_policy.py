"""
MDAM (Multi-Decoder Attention Model) 策略网络封装

多解码器注意力模型：同时运行多个 Transformer 解码器，
推理时取最优路径，在与 AM 相同训练成本下提升解质量。
"""

from typing import Dict, Any
from .base_policy import BasePolicy


class MDAMPolicyWrapper(BasePolicy):
    """
    MDAM 策略封装类

    特点：
        - 多个解码器并行解码，取最优解
        - 兼容 AM 支持的所有路由问题（不含 ATSP/FFSP）
        - 架构参数与 AM 一致（embed_dim, num_encoder_layers, num_heads）
        - 使用内置 MDAM(REINFORCE 子类) 训练，不兼容外部 PPO/A2C
    """

    def _init_policy_params(self):
        """MDAM 使用标准 embed_dim/layers/heads，无额外参数"""
        pass

    def get_policy_name(self) -> str:
        return 'mdam'

    def create_policy(self, env):
        """
        创建 MDAMPolicy 策略网络

        参数:
            env: RL4CO 环境（提供 env.name）

        返回:
            MDAMPolicy 实例
        """
        try:
            from rl4co.models.zoo.mdam import MDAMPolicy
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或当前版本不包含MDAM。\n"
                "请安装最新版: pip install rl4co"
            )

        policy = MDAMPolicy(
            env_name=env.name,
            embed_dim=self.embed_dim,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
        )
        return policy
