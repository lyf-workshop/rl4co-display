"""
Q-Learning 算法兼容入口。

当前阶段复用项目中稳定的 REINFORCE 训练模型，保留独立算法标识，
后续实现完整 Q-Learning 时只需替换 create_model()。
"""

from .reinforce_algo import REINFORCEAlgorithm


class QLearningAlgorithm(REINFORCEAlgorithm):
    """提供可训练链路的 Q-Learning 算法适配器。"""

    training_backend = "reinforce"

    def get_algorithm_name(self) -> str:
        return "qlearning"
