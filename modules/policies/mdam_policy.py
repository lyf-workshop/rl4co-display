"""MDAM (Multi-Decoder Attention Model) 策略网络封装。"""

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
        self.num_paths = int(self.config.get("num_paths", 5))
        self.eg_step_gap = int(self.config.get("eg_step_gap", 200))

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
            from .mdam_runtime import (
                RL4CODisplayMDAMDecoder,
                RL4CODisplayMDAMPolicy,
            )
        except ImportError as exc:
            raise ImportError(
                "RL4CO库未安装或当前版本不包含MDAM。\n"
                "请安装最新版: pip install rl4co"
            ) from exc

        decoder = RL4CODisplayMDAMDecoder(
            env_name=env.name,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_paths=self.num_paths,
            eg_step_gap=self.eg_step_gap,
        )
        policy = RL4CODisplayMDAMPolicy(
            env_name=env.name,
            embed_dim=self.embed_dim,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            decoder=decoder,
        )
        return policy

    def get_policy_params(self):
        params = super().get_policy_params()
        params.update(
            {
                "num_paths": self.num_paths,
                "eg_step_gap": self.eg_step_gap,
            }
        )
        return params

    def _validate_policy_params(self):
        if self.num_paths < 1:
            return False, "num_paths必须大于0"
        if self.eg_step_gap < 1:
            return False, "eg_step_gap必须大于0"
        return True, ""
