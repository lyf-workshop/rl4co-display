"""
FFSP (Flexible Flow Shop Problem) - 柔性流水车间调度问题
"""

from typing import Dict, Any, Callable
from .base_problem import BaseProblem


class FFSProblem(BaseProblem):
    """
    柔性流水车间调度问题类
    
    目标：最小化完工时间（makespan）
    特点：
        - 多阶段生产流程
        - 每个阶段有多台并行机器
        - 作业必须按阶段顺序加工
        - 经典的调度优化问题
    
    问题结构：
        - num_stage: 阶段数（如3个阶段：切割、加工、组装）
        - num_machine: 每个阶段的机器数
        - num_job: 作业数量
        - run_time: 每个作业在每台机器上的加工时间矩阵
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化FFSP问题
        
        参数:
            config: 配置字典
                - num_stage: 阶段数 (默认: 3)
                - num_machine: 每阶段机器数 (默认: 4)
                - num_job: 作业数 (默认: 20)
                - min_time: 最小加工时间 (默认: 2)
                - max_time: 最大加工时间 (默认: 10)
                - flatten_stages: 是否展平阶段 (默认: True)
        """
        # FFSP使用num_job而不是num_loc
        if 'num_job' in config and 'num_loc' not in config:
            config['num_loc'] = config['num_job']
        
        super().__init__(config)
    
    def _init_problem_params(self):
        """初始化FFSP特定参数"""
        # 阶段数
        self.num_stage = int(self.config.get('num_stage', 3))
        
        # 每个阶段的机器数（当前版本要求各阶段机器数相同）
        self.num_machine = int(self.config.get('num_machine', 4))
        
        # 作业数
        self.num_job = int(self.config.get('num_job', 20))
        self.num_loc = self.num_job  # 兼容性
        
        # 总机器数
        self.num_machine_total = self.num_machine * self.num_stage
        
        # 加工时间范围
        self.min_time = int(self.config.get('min_time', 2))
        self.max_time = int(self.config.get('max_time', 10))
        
        # 是否展平阶段（影响embedding策略）
        # True: 每个阶段的机器独立编码
        # False: 不同阶段的同位置机器共享embedding
        self.flatten_stages = self.config.get('flatten_stages', True)
    
    def get_problem_type(self) -> str:
        return 'ffsp'
    
    def get_problem_name(self) -> str:
        return 'Flexible Flow Shop Problem'
    
    def create_environment(self):
        """
        创建FFSP环境
        
        返回:
            FFSPEnv: RL4CO的FFSP环境实例
        """
        try:
            from rl4co.envs.scheduling import FFSPEnv
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或版本过旧，无法创建FFSP环境。\n"
                "请安装/更新: pip install rl4co --upgrade"
            )
        
        env = FFSPEnv(generator_params=self.get_env_params())
        return env
    
    def get_visualization_functions(self) -> Dict[str, Callable]:
        """
        获取FFSP可视化函数
        
        返回:
            dict: {
                'gantt_chart': 创建甘特图,
                'schedule_comparison': 创建调度对比图
            }
        """
        from modules.rl_training.visualizations.ffsp_viz import (
            create_ffsp_gantt_chart,
            create_ffsp_schedule_comparison
        )
        
        return {
            'gantt_chart': create_ffsp_gantt_chart,
            'schedule_comparison': create_ffsp_schedule_comparison,
        }
    
    def get_problem_description(self) -> str:
        return (
            f"柔性流水车间调度问题 (FFSP)\n"
            f"目标: 最小化{self.num_job}个作业的完工时间 (makespan)\n"
            f"结构: {self.num_stage}个阶段，每个阶段{self.num_machine}台并行机器\n"
            f"约束: 作业按阶段顺序加工，同一时刻机器只能处理一个作业"
        )
    
    def get_problem_features(self) -> list:
        return [
            '多阶段生产调度',
            '并行机器配置',
            'NP-hard调度问题',
            '制造系统优化',
            '最小化完工时间',
        ]
    
    def get_env_params(self) -> Dict[str, Any]:
        """FFSP环境参数"""
        return {
            'num_stage': self.num_stage,
            'num_machine': self.num_machine,
            'num_job': self.num_job,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'flatten_stages': self.flatten_stages,
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        """获取FFSP的默认参数"""
        return {
            'num_stage': self.num_stage,
            'num_machine': self.num_machine,
            'num_job': self.num_job,
            'num_machine_total': self.num_machine_total,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'flatten_stages': self.flatten_stages,
            'batch_size': self.batch_size,
        }
    
    def _validate_problem_params(self):
        """FFSP参数验证"""
        if self.num_stage < 2:
            return False, "FFSP的阶段数(num_stage)必须至少为2"
        
        if self.num_machine < 1:
            return False, "每个阶段的机器数(num_machine)必须至少为1"
        
        if self.num_job < 2:
            return False, "作业数(num_job)必须至少为2"
        
        if self.num_job > 200:
            return False, "FFSP作业数不建议超过200（计算复杂度过高）"
        
        if self.min_time < 1:
            return False, "最小加工时间(min_time)必须大于0"
        
        if self.max_time <= self.min_time:
            return False, "最大加工时间(max_time)必须大于最小加工时间(min_time)"
        
        if self.num_machine_total > 50:
            return False, f"总机器数({self.num_machine_total})过大，建议减少阶段数或机器数"
        
        return True, ""
    
    def get_complexity_info(self) -> Dict[str, Any]:
        """
        获取问题复杂度信息
        
        返回:
            dict: 复杂度相关信息
        """
        # 状态空间大小估计
        state_space_size = (self.num_job ** self.num_machine_total)
        
        # 决策步数
        decision_steps = self.num_job * self.num_stage
        
        return {
            'state_space_size': state_space_size,
            'decision_steps': decision_steps,
            'num_machines': self.num_machine_total,
            'complexity_class': 'NP-hard',
            'problem_scale': self._get_scale_description(),
        }
    
    def _get_scale_description(self) -> str:
        """获取问题规模描述"""
        total_ops = self.num_job * self.num_stage
        
        if total_ops < 50:
            return '小规模（适合快速实验）'
        elif total_ops < 100:
            return '中等规模（标准测试）'
        elif total_ops < 200:
            return '大规模（计算密集）'
        else:
            return '超大规模（需要高性能计算）'
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"type={self.problem_type}, "
            f"stages={self.num_stage}, "
            f"machines={self.num_machine}, "
            f"jobs={self.num_job})"
        )
