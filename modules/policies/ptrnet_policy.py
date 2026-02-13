"""
Pointer Network (PtrNet) 策略网络封装
开创性的序列到序列模型，首次将深度学习应用于组合优化
"""

from typing import Dict, Any
from .base_policy import BasePolicy


class PtrNetPolicyWrapper(BasePolicy):
    """
    Pointer Network 策略封装类
    
    特点：
        - 基于 LSTM 的序列到序列模型
        - 使用注意力机制"指向"输入
        - 首个深度学习CO方法（2015）
        - 适合理解和教学
    
    历史意义：
        - 开创性工作，为后续方法奠定基础
        - 证明了深度学习在CO上的可行性
        - Attention Model 的前身
    
    适用场景：
        - 学习和研究深度学习CO
        - 理解基础架构
        - 小规模问题快速原型
    """
    
    def _init_policy_params(self):
        """初始化 PtrNet 特定参数"""
        # PtrNet 使用 LSTM，参数与 Transformer 不同
        self.hidden_dim = int(self.config.get('hidden_dim', 128))
        self.num_layers = int(self.config.get('num_layers', 2))
        self.dropout = float(self.config.get('dropout', 0.0))
        
        # 重写父类的 Transformer 参数（PtrNet 不使用这些）
        self.embed_dim = self.hidden_dim  # 兼容性
        self.num_encoder_layers = self.num_layers
        self.num_heads = 1  # PtrNet 不使用多头注意力
    
    def get_policy_name(self) -> str:
        return 'ptrnet'
    
    def create_policy(self, env):
        """
        创建 Pointer Network 策略
        
        注意：RL4CO 可能没有独立的 PtrNetPolicy，
        我们使用 AttentionModelPolicy 的简化配置来模拟
        
        参数:
            env: RL4CO 环境
        
        返回:
            策略实例
        """
        try:
            # 尝试导入 PtrNet 专用策略（如果存在）
            try:
                from rl4co.models.nn import PointerNetworkPolicy
                
                policy = PointerNetworkPolicy(
                    env_name=env.name,
                    hidden_dim=self.hidden_dim,
                    num_layers=self.num_layers,
                    dropout=self.dropout,
                )
                
                return policy
                
            except (ImportError, AttributeError):
                # RL4CO 没有独立的 PtrNet，使用 AM 的简化版本模拟
                from rl4co.models.nn import AttentionModelPolicy
                
                # 使用最小配置的 AM 来近似 PtrNet 的效果
                policy = AttentionModelPolicy(
                    env_name=env.name,
                    embed_dim=self.hidden_dim,
                    num_encoder_layers=self.num_layers,  # 较少的层数
                    num_heads=1,  # 单头注意力（类似 PtrNet）
                    normalization='instance',  # PtrNet 风格
                )
                
                # 标记为 PtrNet 模式（用于日志）
                policy._is_ptrnet_mode = True
                
                return policy
                
        except ImportError:
            raise ImportError(
                "RL4CO 库未安装，无法创建 Pointer Network 策略。\n"
                "请安装: pip install rl4co"
            )
    
    def get_policy_params(self) -> Dict[str, Any]:
        """获取 PtrNet 策略参数"""
        return {
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'policy_type': 'ptrnet',
        }
    
    def _validate_policy_params(self):
        """验证 PtrNet 参数"""
        if self.hidden_dim < 32 or self.hidden_dim > 512:
            return False, "hidden_dim 应在 32-512 之间"
        
        if self.num_layers < 1 or self.num_layers > 4:
            return False, "num_layers 应在 1-4 之间（PtrNet 通常较浅）"
        
        if self.dropout < 0 or self.dropout > 0.5:
            return False, "dropout 应在 0-0.5 之间"
        
        return True, ""
    
    def get_policy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        return {
            'name': 'PtrNet',
            'full_name': 'Pointer Network',
            'type': 'seq2seq',
            'year': 2015,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'description': '开创性的序列到序列模型，深度学习CO的先驱',
            'note': '使用 AttentionModel 的简化版本模拟（RL4CO 无独立 PtrNet）'
        }


__all__ = ['PtrNetPolicyWrapper']
