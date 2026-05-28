"""
SymNCO (Symmetric Neural Combinatorial Optimization) 策略网络封装
基于 AM + 投影头，利用二面体8对称增强和多损失训练
"""

from typing import Dict, Any
from .base_policy import BasePolicy


class SymNCOPolicyWrapper(BasePolicy):
    """
    SymNCO策略封装类

    特点：
        - 继承自 AttentionModelPolicy，额外增加投影头（Projection Head）
        - 使用二面体8对称增强（dihedral_8_augmentation）生成 num_augment 个对称副本
        - 可选地从多个起点构建路径（num_starts > 0）
        - 训练时同时优化三种损失：
            * L_ps（问题对称性损失）：不同对称增强视角的 REINFORCE 基准
            * L_ss（解对称性损失）：不同起点的 REINFORCE 基准
            * L_inv（不变性损失）：投影嵌入的余弦相似度
        - 适合 TSP、mTSP、CVRP 等具有二维坐标对称性的路由问题

    参考文献：
        Kim et al. (2022) "Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization"
        https://arxiv.org/abs/2205.13209
    """

    def _init_policy_params(self):
        """初始化 SymNCO 特定参数"""
        self.num_augment = int(self.config.get('num_augment', 4))
        self.num_starts = int(self.config.get('num_starts', 0))
        self.alpha = float(self.config.get('symnco_alpha', 0.2))
        self.beta = float(self.config.get('symnco_beta', 1.0))
        self.augment_fn = self.config.get('augment_fn', 'dihedral_8_augmentation')
        # SymNCO 通常需要至少 3 层 encoder 以获得足够的表达能力
        if self.num_encoder_layers < 3:
            self.num_encoder_layers = 3

    def get_policy_name(self) -> str:
        return 'symnco'

    def create_policy(self, env):
        """
        创建 SymNCOPolicy 策略网络

        参数:
            env: RL4CO环境

        返回:
            SymNCOPolicy 实例（AttentionModelPolicy + 投影头）
        """
        try:
            from rl4co.models.zoo.symnco import SymNCOPolicy
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或当前版本不包含SymNCO。\n"
                "请安装最新版: pip install rl4co"
            )

        policy = SymNCOPolicy(
            env_name=env.name,
            embed_dim=self.embed_dim,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
        )

        return policy

    def get_policy_params(self) -> Dict[str, Any]:
        """获取 SymNCO 策略参数"""
        params = super().get_policy_params()
        params.update({
            'num_augment': self.num_augment,
            'num_starts': self.num_starts,
            'alpha': self.alpha,
            'beta': self.beta,
            'augment_fn': self.augment_fn,
        })
        return params

    def _validate_policy_params(self):
        """验证 SymNCO 参数"""
        if self.num_augment < 1:
            return False, "num_augment 必须大于0（推荐4或8）"

        if self.num_augment > 10:
            return False, "num_augment 不建议超过10（显存限制，dihedral_8_augmentation最多8个变换）"

        if self.alpha < 0:
            return False, "alpha（不变性损失权重）必须为非负数（推荐0.2）"

        if self.beta < 0:
            return False, "beta（解对称性损失权重）必须为非负数（推荐1.0）"

        return True, ""

    def get_policy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        info = super().get_policy_info()
        info.update({
            'num_augment': self.num_augment,
            'num_starts': self.num_starts,
            'alpha': self.alpha,
            'beta': self.beta,
            'augment_fn': self.augment_fn,
        })
        return info
