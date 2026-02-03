"""
[PROBLEM_NAME]问题专用可视化函数模板
请将 [PROBLEM_NAME] 替换为具体问题名称

使用步骤：
1. 复制此模板文件
2. 重命名为 {problem}_viz.py，如 op_viz.py
3. 替换所有 [PROBLEM_NAME] 和 [problem] 标记
4. 根据问题特点实现可视化逻辑
5. 在 visualizations/__init__.py 中导出新函数
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def create_[problem]_route_animation(td, actions, save_path, title="[PROBLEM_NAME]路线生成过程", fps=2):
    """
    创建[PROBLEM_NAME]路线逐步生成的动态GIF
    
    参数:
        td: TensorDict，包含问题相关信息
        actions: numpy数组，动作序列
        save_path: GIF保存路径
        title: 图表标题
        fps: 帧率（每秒帧数）
    """
    # TODO: 提取问题特有的数据
    # 例如：
    # locs = td['locs'].cpu().numpy()  # 节点坐标
    # prizes = td.get('prizes', None)  # 奖励值（如果有）
    # demands = td.get('demand', None)  # 需求（如果有）
    
    if hasattr(td, 'get'):
        locs = td.get('locs', td['locs']).cpu().numpy()
    else:
        locs = td['locs'].cpu().numpy()
    
    num_nodes = len(locs)
    frames = []
    
    # TODO: 实现度量计算函数
    def calculate_metrics(locs, actions, step):
        """计算到第step步为止的累计指标"""
        if step < 1:
            return 0.0  # 返回相关指标
        
        # TODO: 根据问题类型计算相关指标
        # 例如：总距离、收集的奖励、满足的需求等
        total_metric = 0.0
        
        for i in range(step):
            # 计算逻辑...
            pass
        
        return total_metric
    
    # 为每一步生成一帧图像
    for step in range(len(actions) + 1):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # TODO: 绘制节点
        # 根据问题特点自定义节点样式
        ax.scatter(locs[:, 0], locs[:, 1], c='lightblue', s=200, 
                  zorder=3, alpha=0.6, edgecolors='black', linewidths=2)
        
        # TODO: 标注节点信息
        for i, (x, y) in enumerate(locs):
            # 可以显示：编号、奖励、需求等
            ax.text(x, y, str(i), fontsize=10, ha='center', va='center',
                   fontweight='bold', color='darkblue')
        
        # TODO: 绘制已经构建的路径
        if step > 0:
            for i in range(step):
                if i + 1 < len(actions):
                    start_idx = actions[i]
                    end_idx = actions[i + 1]
                    
                    start_pos = locs[start_idx]
                    end_pos = locs[end_idx]
                    
                    # 绘制路径线
                    ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                           'b-', linewidth=3, alpha=0.7, zorder=1)
                    
                    # 添加箭头
                    mid_x = (start_pos[0] + end_pos[0]) / 2
                    mid_y = (start_pos[1] + end_pos[1]) / 2
                    dx = end_pos[0] - start_pos[0]
                    dy = end_pos[1] - start_pos[1]
                    ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                              xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                              arrowprops=dict(arrowstyle='->', color='blue', 
                                            lw=2, alpha=0.7))
        
        # TODO: 高亮当前节点
        if step > 0 and step <= len(actions):
            current_node = actions[step - 1]
            current_pos = locs[current_node]
            ax.scatter(current_pos[0], current_pos[1], 
                      c='red', s=400, zorder=5, marker='*', 
                      edgecolors='darkred', linewidths=2,
                      label=f'当前: 节点 {current_node}')
        
        # TODO: 计算并显示当前指标
        current_metric = calculate_metrics(locs, actions, step)
        
        # 设置标题和信息
        if step == 0:
            info_text = "开始构建路线..."
        elif step < len(actions):
            info_text = f"第 {step} 步 | 指标: {current_metric:.3f}"
        else:
            info_text = f"完成！总指标: {current_metric:.3f}"
        
        ax.set_title(f"{title}\n{info_text}", fontsize=14, fontweight='bold', pad=20)
        
        # 设置坐标轴
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        
        # 添加图例
        if step > 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # 添加进度条
        progress = step / len(actions) if len(actions) > 0 else 0
        ax.text(0.5, -0.12, f"进度: {int(progress * 100)}%", 
               ha='center', va='top', transform=ax.transAxes,
               fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 保存当前帧为图像
        fig.tight_layout()
        
        # 将图形转换为PIL Image
        fig.canvas.draw()
        try:
            buf = fig.canvas.buffer_rgba()
            image = np.frombuffer(buf, dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            image = image[:, :, :3]
        except AttributeError:
            try:
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                image = buf[:, :, 1:]
        
        frames.append(Image.fromarray(image))
        plt.close(fig)
    
    # 在最后一帧停留更长时间
    for _ in range(3):
        frames.append(frames[-1])
    
    # 保存为GIF
    duration = int(1000 / fps)
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False
    )


def create_[problem]_comparison_plot(env, td, actions_untrained, rewards_untrained, 
                                    actions_trained, rewards_trained, save_path, index=1):
    """
    创建[PROBLEM_NAME]路线对比图（训练前vs训练后）
    
    参数:
        env: [PROBLEM_NAME]环境
        td: TensorDict
        actions_untrained: 未训练模型的动作序列
        rewards_untrained: 未训练模型的奖励
        actions_trained: 训练后模型的动作序列
        rewards_trained: 训练后模型的奖励
        save_path: 保存路径
        index: 图片索引（用于多实例）
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # 渲染未训练和训练后的路线
    env.render(td, actions_untrained, ax=axs[0])
    env.render(td, actions_trained, ax=axs[1])
    
    # 设置标题
    axs[0].set_title(f"Random | Reward = {rewards_untrained.item():.3f}")
    axs[1].set_title(f"Trained | Reward = {rewards_trained.item():.3f}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



