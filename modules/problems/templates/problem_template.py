"""
[问题名称] - [Problem Full Name]

复制此模板创建新问题类型：
1. 复制此文件到 modules/problems/[problem_name].py
2. 修改类名和方法实现
3. 在 modules/problems/__init__.py 中注册
4. 创建对应的可视化函数
"""

from typing import Dict, Any, Callable, Tuple
from .base_problem import BaseProblem


class TemplateProblem(BaseProblem):
    """
    [问题中文名] - [Problem Full Name]
    
    目标：[描述优化目标]
    特点：
        - [特点1]
        - [特点2]
        - [特点3]
    
    应用场景：
        - [应用1]
        - [应用2]
    """
    
    def _init_problem_params(self):
        """
        初始化问题特定参数
        
        示例：
            self.some_param = float(self.config.get('some_param', 1.0))
            self.another_param = int(self.config.get('another_param', 10))
        """
        # TODO: 添加问题特定参数
        pass
    
    def get_problem_type(self) -> str:
        """
        返回问题类型标识符
        
        返回:
            str: 小写问题缩写 (如 'tsp', 'cvrp', 'pctsp')
        """
        # TODO: 修改为实际的问题类型
        return 'template'
    
    def get_problem_name(self) -> str:
        """
        返回问题完整英文名称
        
        返回:
            str: 问题全名 (如 'Traveling Salesman Problem')
        """
        # TODO: 修改为实际的问题名称
        return 'Template Problem'
    
    def create_environment(self):
        """
        创建RL4CO环境实例
        
        返回:
            RL4CO环境对象
        
        示例:
            from rl4co.envs import SomeEnv
            env = SomeEnv(generator_params=self.get_env_params())
            return env
        """
        try:
            # TODO: 导入对应的RL4CO环境
            # from rl4co.envs import YourEnv
            # env = YourEnv(generator_params=self.get_env_params())
            # return env
            
            raise NotImplementedError(
                "请在子类中实现 create_environment() 方法"
            )
        except ImportError:
            raise ImportError(
                "RL4CO库未安装或不包含此环境。\n"
                "请安装: pip install rl4co\n"
                "或检查环境名称是否正确"
            )
    
    def get_visualization_functions(self) -> Dict[str, Callable]:
        """
        获取可视化函数集合
        
        返回:
            dict: {
                'animation': 创建动画的函数,
                'comparison': 创建对比图的函数,
                'other': 其他可视化函数 (可选)
            }
        
        示例:
            from modules.rl_training.visualizations.your_viz import (
                create_your_route_animation,
                create_your_comparison_plot
            )
            return {
                'animation': create_your_route_animation,
                'comparison': create_your_comparison_plot,
            }
        """
        # TODO: 导入并返回可视化函数
        raise NotImplementedError(
            "请在子类中实现 get_visualization_functions() 方法\n"
            "并创建对应的可视化函数文件"
        )
    
    def get_problem_description(self) -> str:
        """
        获取问题的详细文字描述
        
        返回:
            str: 多行描述文本
        """
        # TODO: 修改为实际的问题描述
        return (
            f"[问题中文名] ([问题缩写])\n"
            f"目标: [优化目标描述]\n"
            f"约束: [约束条件描述]\n"
            f"规模: {self.num_loc}个节点"
        )
    
    def get_problem_features(self) -> list:
        """
        获取问题特征列表
        
        返回:
            list: 问题特征字符串列表
        """
        # TODO: 修改为实际的问题特征
        return [
            '[特征1]',
            '[特征2]',
            '[特征3]',
            '[应用领域]',
        ]
    
    def get_env_params(self) -> Dict[str, Any]:
        """
        获取传递给RL4CO环境的参数
        
        返回:
            dict: 环境初始化参数
        
        示例:
            return {
                'num_loc': self.num_loc,
                'your_param': self.your_param,
                'another_param': self.another_param,
            }
        """
        # TODO: 添加问题特定的环境参数
        return {
            'num_loc': self.num_loc,
            # 添加其他参数...
        }
    
    def get_default_params(self) -> Dict[str, Any]:
        """
        获取问题的默认参数（包含基类参数）
        
        返回:
            dict: 完整的默认参数字典
        """
        params = super().get_default_params()
        
        # TODO: 添加问题特定的默认参数
        # params.update({
        #     'your_param': self.your_param,
        #     'another_param': self.another_param,
        # })
        
        return params
    
    def _validate_problem_params(self) -> Tuple[bool, str]:
        """
        验证问题特定参数
        
        返回:
            (bool, str): (是否有效, 错误信息)
        
        示例:
            if self.your_param <= 0:
                return False, "your_param必须大于0"
            
            if self.num_loc > 500:
                return False, "此问题节点数量不建议超过500"
            
            return True, ""
        """
        # TODO: 添加参数验证逻辑
        return True, ""


# ============================================
# 使用示例
# ============================================

if __name__ == '__main__':
    """
    测试新问题类型
    """
    
    # 1. 创建配置
    config = {
        'num_loc': 50,
        'batch_size': 512,
        # 添加问题特定参数...
    }
    
    # 2. 创建问题实例
    problem = TemplateProblem(config)
    
    # 3. 验证配置
    valid, error_msg = problem.validate_config()
    if not valid:
        print(f"配置无效: {error_msg}")
        exit(1)
    
    # 4. 打印问题信息
    print("=" * 60)
    print(problem.get_problem_description())
    print("=" * 60)
    print(f"\n特征:")
    for feature in problem.get_problem_features():
        print(f"  • {feature}")
    
    # 5. 创建环境
    try:
        env = problem.create_environment()
        print(f"\n✅ 环境创建成功: {env.name}")
    except Exception as e:
        print(f"\n❌ 环境创建失败: {e}")
    
    # 6. 获取可视化函数
    try:
        viz_funcs = problem.get_visualization_functions()
        print(f"\n✅ 可视化函数: {list(viz_funcs.keys())}")
    except Exception as e:
        print(f"\n❌ 可视化函数未实现: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


