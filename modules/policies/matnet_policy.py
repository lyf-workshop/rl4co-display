"""
MatNet 策略网络封装
专为 ATSP 和 FFSP (Flexible Flow Shop Problem) 设计的矩阵注意力网络

参考文献:
    Kwon et al., 2021: "Matrix Encoding Networks for Neural Combinatorial Optimization"
    https://arxiv.org/abs/2106.11113
"""

from typing import Dict, Any
from .base_policy import BasePolicy


class MatNetPolicyWrapper(BasePolicy):
    """
    MatNet策略封装类
    
    特点：
        - 矩阵注意力机制（行列编码）
        - 专为非对称问题设计（ATSP、FFSP）
        - 支持多阶段调度问题
        - 继承自POMO的多起点优化
    
    适用问题：
        - ATSP（非对称旅行商问题）
        - FFSP（柔性流水车间调度问题）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化MatNet策略
        
        参数:
            config: 配置字典，包含策略参数
        """
        super().__init__(config)
        
        # MatNet特有参数
        self.use_graph_context = config.get('use_graph_context', False)
        self.normalization = config.get('normalization', 'instance')
        
        # FFSP特定参数
        self.flatten_stages = config.get('flatten_stages', True)
        self.num_stage = config.get('num_stage', 3)  # FFSP的阶段数
    
    def get_policy_name(self) -> str:
        return 'matnet'
    
    def create_policy(self, env):
        """
        创建MatNet策略
        
        参数:
            env: RL4CO环境
        
        返回:
            MatNetPolicy实例
        """
        try:
            from rl4co.models.zoo.matnet import MatNetPolicy
            from rl4co.models.zoo.matnet.policy import MultiStageFFSPPolicy
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或版本过旧，无法创建MatNetPolicy。\n"
                "请安装/更新: pip install rl4co --upgrade"
            )
        
        # 根据环境类型选择策略
        if env.name == 'ffsp':
            # FFSP使用专门的多阶段策略或单阶段策略
            if self.flatten_stages:
                # flatten_stages=True: 使用标准MatNetPolicy
                policy = MatNetPolicy(
                    env_name=env.name,
                    embed_dim=self.embed_dim,
                    num_encoder_layers=self.num_encoder_layers,
                    num_heads=self.num_heads,
                    normalization=self.normalization,
                    use_graph_context=self.use_graph_context,
                )
            else:
                # flatten_stages=False: 使用多阶段策略（每个阶段独立的encoder/decoder）
                policy = MultiStageFFSPPolicy(
                    stage_cnt=self.num_stage,
                    embed_dim=self.embed_dim,
                    num_encoder_layers=self.num_encoder_layers,
                    num_heads=self.num_heads,
                    normalization=self.normalization,
                    use_graph_context=self.use_graph_context,
                )
        else:
            # 其他问题（如ATSP）使用标准MatNetPolicy
            policy = MatNetPolicy(
                env_name=env.name,
                embed_dim=self.embed_dim,
                num_encoder_layers=self.num_encoder_layers,
                num_heads=self.num_heads,
                normalization=self.normalization,
                use_graph_context=self.use_graph_context,
            )
        
        return policy
    
    def get_policy_params(self) -> Dict[str, Any]:
        """获取MatNet策略参数"""
        params = super().get_policy_params()
        params.update({
            'use_graph_context': self.use_graph_context,
            'normalization': self.normalization,
        })
        
        # 如果是FFSP，添加阶段相关参数
        if hasattr(self, 'num_stage'):
            params.update({
                'flatten_stages': self.flatten_stages,
                'num_stage': self.num_stage,
            })
        
        return params
    
    def _validate_policy_params(self):
        """验证MatNet参数"""
        # MatNet推荐的embed_dim范围
        if self.embed_dim < 128:
            return False, "MatNet的embed_dim建议至少为128（原论文使用256）"
        
        # MatNet推荐的层数
        if self.num_encoder_layers < 3:
            return False, "MatNet的num_encoder_layers建议至少为3（原论文使用5）"
        
        # MatNet推荐的注意力头数
        if self.num_heads < 8:
            return False, "MatNet的num_heads建议至少为8（原论文使用16）"
        
        # FFSP阶段数验证
        if hasattr(self, 'num_stage') and self.num_stage < 2:
            return False, "FFSP的num_stage必须至少为2"
        
        return True, ""
    
    def get_policy_info(self) -> Dict[str, Any]:
        """获取策略信息（用于前端显示）"""
        info = super().get_policy_info()
        info.update({
            'use_graph_context': self.use_graph_context,
            'normalization': self.normalization,
        })
        
        if hasattr(self, 'num_stage'):
            info.update({
                'flatten_stages': self.flatten_stages,
                'num_stage': self.num_stage,
            })
        
        return info
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"policy={self.policy_name}, "
            f"embed_dim={self.embed_dim}, "
            f"layers={self.num_encoder_layers}, "
            f"heads={self.num_heads})"
        )
