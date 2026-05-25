"""
DeepACO (Deep Ant Colony Optimization) 策略网络封装

深度学习 + 蚁群算法混合模型：
- GNN 编码器学习信息素矩阵（启发式热图）
- AntSystem 利用该热图执行 ACO 搜索
- 非自回归范式，与 AM/POMO 等自回归模型形成对比
"""

from typing import Dict, Any
from .base_policy import BasePolicy


class DeepACOPolicyWrapper(BasePolicy):
    """
    DeepACO 策略封装类

    特点：
        - 非自回归：先学习信息素热图，再用 ACO 搜索
        - 蚂蚁数量 n_ants 控制探索宽度（训练/测试可分别设置）
        - 兼容 TSP、CVRP、mTSP、OP 等标准路由问题
        - 使用内置 DeepACO(REINFORCE 子类) 训练，不兼容外部 PPO/A2C
    """

    def _init_policy_params(self):
        """初始化 DeepACO 特定参数"""
        self.n_ants = int(self.config.get('n_ants', 30))
        self.n_iterations_train = int(self.config.get('n_iterations_train', 1))
        self.n_iterations_test = int(self.config.get('n_iterations_test', 5))

    def get_policy_name(self) -> str:
        return 'deepaco'

    def create_policy(self, env):
        """
        创建 DeepACOPolicy 策略网络

        参数:
            env: RL4CO 环境（提供 env.name）

        返回:
            DeepACOPolicy 实例
        """
        try:
            from rl4co.models.zoo.deepaco import DeepACOPolicy
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或当前版本不包含DeepACO。\n"
                "请安装最新版: pip install rl4co"
            )

        policy = DeepACOPolicy(
            env_name=env.name,
            n_ants={
                'train': self.n_ants,
                'val': self.n_ants,
                'test': self.n_ants,
            },
            n_iterations={
                'train': self.n_iterations_train,
                'val': self.n_iterations_test,
                'test': self.n_iterations_test,
            },
        )
        return policy

    def get_policy_params(self) -> Dict[str, Any]:
        """获取 DeepACO 策略参数"""
        params = super().get_policy_params()
        params['n_ants'] = self.n_ants
        params['n_iterations_train'] = self.n_iterations_train
        params['n_iterations_test'] = self.n_iterations_test
        return params

    def _validate_policy_params(self):
        """验证 DeepACO 参数"""
        if self.n_ants < 1:
            return False, "n_ants必须大于0"
        if self.n_ants > 200:
            return False, "n_ants不建议超过200（显存限制）"
        if self.n_iterations_train < 1:
            return False, "n_iterations_train必须大于0"
        return True, ""
