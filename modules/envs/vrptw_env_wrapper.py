"""
VRPTW自定义Reward包装器
用于在CVRPEnv基础上添加时间窗约束的reward计算
"""

import torch
from rl4co.envs import CVRPEnv


class CVRPEnvWithTimeWindows(CVRPEnv):
    """
    扩展CVRPEnv以支持时间窗约束的reward计算
    
    在基础CVRP的距离reward基础上，添加时间窗违反的惩罚：
    - 早到需要等待：小惩罚（软时间窗）
    - 迟到违反时间窗：大惩罚（硬时间窗）
    - 超过最大配送时间：中等惩罚
    """
    
    def __init__(self, generator_params, time_window_params):
        """
        参数:
            generator_params: CVRP环境生成器参数
                - num_loc: 客户数量
                - vehicle_capacity: 车辆容量
            time_window_params: 时间窗参数字典
                - time_window_width: 时间窗宽度
                - service_time: 服务时间
                - max_time: 最大配送时间
                - hard_time_windows: 是否硬时间窗
        """
        # 确保generator_params包含所有必需的参数
        complete_params = {
            'num_loc': generator_params.get('num_loc', 20),
            'min_loc': 0.0,  # 位置坐标最小值
            'max_loc': 1.0,  # 位置坐标最大值
            'vehicle_capacity': generator_params.get('vehicle_capacity', 1.0),
            'min_demand': 1,  # 最小需求
            'max_demand': 10,  # 最大需求
        }
        
        super().__init__(generator_params=complete_params)
        self.tw_params = time_window_params
        print(f"✅ 使用自定义VRPTW环境（带时间窗reward计算）")
        print(f"   - 客户数量: {complete_params['num_loc']}")
        print(f"   - 车辆容量: {complete_params['vehicle_capacity']}")
        print(f"   - 时间窗宽度: {time_window_params['time_window_width']}")
        print(f"   - 服务时间: {time_window_params['service_time']}")
        print(f"   - 约束类型: {'硬时间窗' if time_window_params['hard_time_windows'] else '软时间窗'}")
    
    def _get_reward(self, td, actions):
        """
        计算带时间窗惩罚的reward
        
        reward = -(total_distance + time_window_penalty)
        
        参数:
            td: TensorDict
            actions: 动作序列
        
        返回:
            reward: 负的总成本（距离+时间惩罚）
        """
        # 1. 基础CVRP reward（距离）
        base_reward = super()._get_reward(td, actions)
        
        # 2. 计算时间窗惩罚
        tw_penalty = self._calculate_time_window_penalty(td, actions)
        
        # 3. 组合reward（base_reward已经是负数）
        total_reward = base_reward - tw_penalty
        
        return total_reward
    
    def _calculate_time_window_penalty(self, td, actions):
        """
        计算时间窗违反惩罚
        
        参数:
            td: TensorDict，包含locs和可选的time_windows
            actions: 访问顺序 [batch_size, seq_len]
        
        返回:
            penalty: 惩罚值tensor [batch_size]，>=0
        """
        batch_size = actions.shape[0]
        device = actions.device
        
        # 提取位置
        locs = td['locs']  # [batch, num_nodes, 2]
        
        # 提取或生成时间窗
        time_windows = td.get('time_windows', None)
        if time_windows is None:
            # 生成默认时间窗
            num_nodes = locs.shape[1]
            time_windows = torch.zeros(batch_size, num_nodes, 2, device=device)
            # 为每个客户随机生成时间窗起始时间
            starts = torch.rand(batch_size, num_nodes, device=device) * 100
            time_windows[:, :, 0] = starts
            time_windows[:, :, 1] = starts + self.tw_params['time_window_width']
            # 仓库的时间窗始终是0到max_time
            time_windows[:, 0, 0] = 0
            time_windows[:, 0, 1] = self.tw_params['max_time']
        
        penalties = torch.zeros(batch_size, device=device)
        
        # 参数
        service_time = self.tw_params['service_time']
        max_time = self.tw_params['max_time']
        hard_tw = self.tw_params['hard_time_windows']
        speed = 1.0  # 行驶速度
        
        # 对每个batch计算惩罚
        for b in range(batch_size):
            current_time = 0.0
            current_pos = locs[b, 0]  # 从仓库开始
            
            for idx in range(actions.shape[1]):
                action = actions[b, idx].item()
                
                if action == 0:  # 返回仓库
                    # 计算返回仓库的旅行时间
                    next_pos = locs[b, 0]
                    travel_time = torch.norm(next_pos - current_pos).item() / speed
                    current_time += travel_time
                    
                    # 检查是否超过最大时间
                    if current_time > max_time:
                        overtime = current_time - max_time
                        penalties[b] += overtime * 3.0
                    
                    # 重置时间（开始新路径）
                    current_time = 0.0
                    current_pos = locs[b, 0]
                    continue
                
                # 访问客户
                next_pos = locs[b, action]
                travel_time = torch.norm(next_pos - current_pos).item() / speed
                current_time += travel_time
                
                # 获取时间窗
                tw_start = time_windows[b, action, 0].item()
                tw_end = time_windows[b, action, 1].item()
                
                # 检查时间窗
                if current_time < tw_start:
                    # 早到：需要等待
                    wait_time = tw_start - current_time
                    current_time = tw_start
                    # 软时间窗给予小惩罚
                    if not hard_tw:
                        penalties[b] += wait_time * 0.1
                
                elif current_time > tw_end:
                    # 迟到：违反时间窗
                    late_time = current_time - tw_end
                    if hard_tw:
                        # 硬时间窗：大惩罚
                        penalties[b] += late_time * 10.0
                    else:
                        # 软时间窗：中等惩罚
                        penalties[b] += late_time * 1.0
                
                # 执行服务
                current_time += service_time
                current_pos = next_pos
        
        return penalties

