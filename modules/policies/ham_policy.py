"""
Heterogeneous Attention Model (HAM) 策略网络封装
专为 PDP（取送货问题）设计的异构注意力模型
论文: https://ieeexplore.ieee.org/document/9352489
"""

from typing import Dict, Any
from .base_policy import BasePolicy


class HAMPolicyWrapper(BasePolicy):
    """
    HAM 策略封装类

    特点：
        - 基于异构多头注意力（Heterogeneous MHA），区分 depot / pickup / delivery 节点
        - 继承 AttentionModelPolicy 的解码器，替换为异构编码器
        - 专为 PDP（Pickup and Delivery Problem）设计
    """

    def get_policy_name(self) -> str:
        return 'ham'

    def _init_policy_params(self):
        """读取 HAM 特有参数"""
        self.normalization = self.config.get('normalization', 'batch')
        self.feedforward_hidden = int(self.config.get('feedforward_hidden', 512))

    def create_policy(self, env):
        """
        创建 HeterogeneousAttentionModelPolicy 实例

        参数:
            env: RL4CO 环境（预期 env.name == 'pdp'）

        返回:
            HeterogeneousAttentionModelPolicy 实例
        """
        try:
            from rl4co.models.zoo.ham.policy import HeterogeneousAttentionModelPolicy
        except ImportError:
            raise ImportError(
                "RL4CO 库未安装或版本过低，无法导入 HeterogeneousAttentionModelPolicy。\n"
                "请安装: pip install rl4co"
            )

        policy = HeterogeneousAttentionModelPolicy(
            env_name=env.name,
            embed_dim=self.embed_dim,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            normalization=self.normalization,
            feedforward_hidden=self.feedforward_hidden,
        )

        return policy

    def get_policy_params(self) -> Dict[str, Any]:
        """获取 HAM 策略参数"""
        params = super().get_policy_params()
        params.update({
            'normalization': self.normalization,
            'feedforward_hidden': self.feedforward_hidden,
        })
        return params

    def _validate_policy_params(self):
        """验证 HAM 参数"""
        if self.normalization not in ('batch', 'instance', 'layer'):
            return False, f"normalization 必须为 batch/instance/layer，当前: {self.normalization}"

        if self.feedforward_hidden < 1:
            return False, "feedforward_hidden 必须大于 0"

        return True, ""
