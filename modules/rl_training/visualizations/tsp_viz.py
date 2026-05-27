"""
TSP问题专用可视化函数
提供TSP路线动画、对比图等可视化
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


def create_tsp_route_animation(td, actions, save_path, title="TSP路线生成过程", fps=2):
    """
    创建TSP路线逐步生成的动态GIF
    
    参数:
        td: TensorDict，包含城市坐标等信息
        actions: numpy数组，访问城市的顺序
        save_path: GIF保存路径
        title: 图表标题
        fps: 帧率（每秒帧数）
    """
    # 提取城市坐标
    if hasattr(td, 'get'):
        locs = td.get('locs', td['locs']).cpu().numpy()
    else:
        locs = td['locs'].cpu().numpy()
    
    num_cities = len(locs)
    frames = []
    
    # 计算每一步的累计距离
    def calculate_partial_distance(locs, actions, step):
        """计算到第step步为止的累计距离"""
        if step < 1:
            return 0.0
        total_dist = 0.0
        for i in range(step):
            city_a = locs[actions[i]]
            # 如果是最后一步，返回起点；否则继续下一个城市
            if i + 1 < len(actions):
                city_b = locs[actions[i + 1]]
            else:
                city_b = locs[actions[0]]  # 返回起点
            dist = np.sqrt(np.sum((city_a - city_b) ** 2))
            total_dist += dist
        return total_dist
    
    # 为每一步生成一帧图像
    for step in range(num_cities + 1):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制所有城市点
        ax.scatter(locs[:, 0], locs[:, 1], c='lightblue', s=200, 
                  zorder=3, alpha=0.6, edgecolors='black', linewidths=2)
        
        # 标注城市编号
        for i, (x, y) in enumerate(locs):
            ax.text(x, y, str(i), fontsize=10, ha='center', va='center',
                   fontweight='bold', color='darkblue')
        
        # 绘制已经构建的路径
        if step > 0:
            for i in range(step):
                start = locs[actions[i]]
                if i + 1 < len(actions):
                    end = locs[actions[i + 1]]
                else:
                    end = locs[actions[0]]  # 最后返回起点
                
                # 绘制路径线
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       'b-', linewidth=3, alpha=0.7, zorder=1)
                
                # 添加箭头表示方向
                mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
                dx, dy = end[0] - start[0], end[1] - start[1]
                ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                          xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                          arrowprops=dict(arrowstyle='->', color='blue', 
                                        lw=2, alpha=0.7))
        
        # 高亮当前访问的城市
        if step > 0 and step <= num_cities:
            current_city = actions[step - 1]
            ax.scatter(locs[current_city, 0], locs[current_city, 1], 
                      c='red', s=400, zorder=5, marker='*', 
                      edgecolors='darkred', linewidths=2,
                      label=f'当前: 城市 {current_city}')
        
        # 高亮起点
        start_city = actions[0]
        ax.scatter(locs[start_city, 0], locs[start_city, 1], 
                  c='green', s=300, zorder=4, marker='s',
                  edgecolors='darkgreen', linewidths=2,
                  label=f'起点: 城市 {start_city}')
        
        # 计算当前累计成本
        current_cost = calculate_partial_distance(locs, actions, step)
        
        # 设置标题和信息
        if step == 0:
            info_text = "开始构建路线..."
        elif step < num_cities:
            info_text = f"第 {step} 步 | 已访问 {step} 个城市 | 累计成本: {current_cost:.3f}"
        else:
            # 最后一步，返回起点
            final_dist = np.sqrt(np.sum((locs[actions[-1]] - locs[actions[0]]) ** 2))
            total_cost = current_cost + final_dist
            info_text = f"完成！总共 {num_cities} 个城市 | 总成本: {total_cost:.3f}"
        
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
        progress = step / num_cities
        ax.text(0.5, -0.12, f"进度: {int(progress * 100)}%", 
               ha='center', va='top', transform=ax.transAxes,
               fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 保存当前帧为图像
        fig.tight_layout()
        
        # 将图形转换为PIL Image（兼容新旧版本matplotlib）
        fig.canvas.draw()
        try:
            # 新版本 matplotlib (>= 3.8)
            buf = fig.canvas.buffer_rgba()
            image = np.frombuffer(buf, dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # 转换 RGBA 到 RGB
            image = image[:, :, :3]
        except AttributeError:
            # 旧版本 matplotlib
            try:
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # 更老的版本，使用 tostring_argb
                buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                # ARGB 转 RGB
                image = buf[:, :, 1:]
        
        frames.append(Image.fromarray(image))
        
        plt.close(fig)
    
    # 在最后一帧停留更长时间
    for _ in range(3):
        frames.append(frames[-1])
    
    # 保存为GIF
    duration = int(1000 / fps)  # 每帧持续时间（毫秒）
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False
    )


def create_tsp_comparison_plot(env, td, actions_untrained, rewards_untrained,
                               actions_trained, rewards_trained, save_path, index=1,
                               left_label="Random"):
    """
    创建TSP路线对比图（基线 vs 训练后）

    参数:
        env: TSP环境
        td: TensorDict
        actions_untrained: 基线模型的动作序列
        rewards_untrained: 基线模型的奖励
        actions_trained: 训练后模型的动作序列
        rewards_trained: 训练后模型的奖励
        save_path: 保存路径
        index: 图片索引（用于多实例）
        left_label: 左图标题前缀，默认"Random"；
                    DeepACO 等非自回归策略传 "Random (ACO Baseline)"
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # 渲染基线和训练后的路线
    env.render(td, actions_untrained, ax=axs[0])
    env.render(td, actions_trained, ax=axs[1])

    # 设置标题
    axs[0].set_title(f"{left_label} | Cost = {-rewards_untrained.item():.3f}")
    axs[1].set_title(f"Trained (ACO) | Cost = {-rewards_trained.item():.3f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



