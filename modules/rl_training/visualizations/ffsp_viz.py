"""
FFSP问题专用可视化函数
提供FFSP甘特图、调度对比图等可视化
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def create_ffsp_gantt_chart(td, schedule, save_path, title="FFSP调度甘特图", num_machine_per_stage=None):
    """
    创建FFSP调度的甘特图
    
    参数:
        td: TensorDict，包含作业信息
        schedule: 调度矩阵 [num_machine, num_job+1]，记录每个作业在每台机器上的开始时间
        save_path: 图片保存路径
        title: 图表标题
        num_machine_per_stage: 每个阶段的机器数量，用于计算阶段编号
    """
    try:
        # 提取数据
        if isinstance(schedule, torch.Tensor):
            schedule = schedule.cpu().numpy()
        
        if hasattr(td, 'get'):
            job_duration = td.get('job_duration', td['job_duration']).cpu().numpy()
        else:
            job_duration = td['job_duration'].cpu().numpy()
        
        # schedule: [num_machine_total, num_job+1]
        # job_duration: [num_job+1, num_machine_total]
        num_machines = schedule.shape[0]
        num_jobs = schedule.shape[1] - 1  # 排除dummy job
        
        # 如果未提供每阶段机器数，尝试猜测（假设只有一个阶段或均匀分布）
        # 这里默认如果不提供，就不显示阶段信息
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, max(8, num_machines * 0.8)))
        
        # 使用更美观的配色方案 (Set3 / Pastel)
        # 扩展颜色列表以支持更多作业
        base_colors = plt.cm.Set3(np.linspace(0, 1, 12))
        if num_jobs > 12:
            extra_colors = plt.cm.Set2(np.linspace(0, 1, 8))
            colors = np.vstack((base_colors, extra_colors))
            # 如果还不够，循环使用
            if num_jobs > 20:
                repeats = (num_jobs // 20) + 1
                colors = np.tile(colors, (repeats, 1))
        else:
            colors = base_colors
            
        colors = colors[:num_jobs]
        
        # 计算makespan
        makespan = 0
        
        # 绘制每台机器的调度
        for machine_idx in range(num_machines):
            # 计算当前机器所属的阶段
            stage_idx = -1
            machine_in_stage_idx = machine_idx
            if num_machine_per_stage:
                stage_idx = machine_idx // num_machine_per_stage
                machine_in_stage_idx = machine_idx % num_machine_per_stage
            
            for job_idx in range(num_jobs):
                start_time = schedule[machine_idx, job_idx]
                
                # 跳过未调度的作业（开始时间为负数）
                if start_time < 0:
                    continue
                
                # 获取作业时长
                duration = job_duration[job_idx, machine_idx]
                end_time = start_time + duration
                
                # 更新makespan
                makespan = max(makespan, end_time)
                
                # 绘制矩形块表示作业
                # 使用圆角矩形或普通矩形，这里用普通矩形但加深边框
                rect = Rectangle(
                    (start_time, machine_idx - 0.4),
                    duration,
                    0.8,
                    facecolor=colors[job_idx],
                    edgecolor='#333333', # 深灰色边框
                    linewidth=1.0,
                    alpha=0.9
                )
                ax.add_patch(rect)
                
                # 构建标注文本
                label_text = f'J{job_idx}'
                if stage_idx >= 0:
                    label_text += f'\nS{stage_idx}'
                
                # 在矩形内标注作业编号
                # 根据矩形大小动态调整字体
                font_size = 9
                if duration < makespan * 0.05: # 如果块太小
                    font_size = 7
                    label_text = f'J{job_idx}' # 简化文本
                
                text_color = 'black'
                # 简单的亮度计算来决定文字颜色（黑/白）
                rgb = colors[job_idx][:3]
                brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
                if brightness < 0.5:
                    text_color = 'white'
                
                ax.text(
                    start_time + duration / 2,
                    machine_idx,
                    label_text,
                    ha='center',
                    va='center',
                    fontsize=font_size,
                    fontweight='bold',
                    color=text_color
                )
        
        # 设置坐标轴
        ax.set_ylim(-0.5, num_machines - 0.5)
        ax.set_xlim(0, makespan * 1.05)
        
        # Y轴：机器编号 (包含阶段信息)
        yticks = range(num_machines)
        yticklabels = []
        for i in range(num_machines):
            if num_machine_per_stage:
                s_idx = i // num_machine_per_stage
                m_idx = i % num_machine_per_stage
                yticklabels.append(f'Stage {s_idx}\nMachine {m_idx}')
            else:
                yticklabels.append(f'Machine {i}')
        
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=10)
        ax.invert_yaxis()  # 倒转Y轴，使机器0在顶部
        
        # X轴：时间
        ax.set_xlabel('时间 (Time)', fontsize=12, fontweight='bold')
        ax.set_ylabel('机器 (Machine)', fontsize=12, fontweight='bold')
        
        # 设置标题
        ax.set_title(f"{title}\n最大完工时间 (Makespan): {makespan:.1f}", 
                    fontsize=16, fontweight='bold', pad=20)
        
        # 添加网格
        ax.grid(True, axis='x', alpha=0.4, linestyle='--', color='gray')
        
        # 添加阶段分隔线
        if num_machine_per_stage:
            num_stages = num_machines // num_machine_per_stage
            for s in range(1, num_stages):
                y_pos = s * num_machine_per_stage - 0.5
                ax.axhline(y=y_pos, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        
        # 添加图例（作业列表）
        # 优化布局：放在图表下方
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor='black', 
                     label=f'Job {i}', alpha=0.9)
            for i in range(min(num_jobs, 15))  # 最多显示15个作业
        ]
        
        if num_jobs > 15:
            legend_elements.append(
                Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black',
                         label=f'... (+{num_jobs-15})', alpha=0.5)
            )
        
        # 将图例放在下方
        ax.legend(handles=legend_elements, loc='upper center', 
                 bbox_to_anchor=(0.5, -0.1), fontsize=10, ncol=min(8, num_jobs), frameon=False)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight') # 提高DPI
        plt.close()
        
        return makespan

        
    except Exception as e:
        print(f"创建FFSP甘特图时出错: {str(e)}")
        # 创建空白图片
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'甘特图生成失败\n{str(e)}', 
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return 0


def create_ffsp_schedule_comparison(td_before, td_after, schedule_before, schedule_after,
                                    save_path, title="FFSP训练前后对比", num_machine_per_stage=None):
    """
    创建FFSP训练前后的调度对比图
    
    参数:
        td_before: 训练前的TensorDict
        td_after: 训练后的TensorDict
        schedule_before: 训练前的调度
        schedule_after: 训练后的调度
        save_path: 图片保存路径
        title: 图表标题
        num_machine_per_stage: 每个阶段的机器数量
    """
    try:
        # 提取数据
        if isinstance(schedule_before, torch.Tensor):
            schedule_before = schedule_before.cpu().numpy()
        if isinstance(schedule_after, torch.Tensor):
            schedule_after = schedule_after.cpu().numpy()
        
        if hasattr(td_before, 'get'):
            job_duration_before = td_before.get('job_duration', td_before['job_duration']).cpu().numpy()
        else:
            job_duration_before = td_before['job_duration'].cpu().numpy()
        
        if hasattr(td_after, 'get'):
            job_duration_after = td_after.get('job_duration', td_after['job_duration']).cpu().numpy()
        else:
            job_duration_after = td_after['job_duration'].cpu().numpy()
        
        # 计算makespan
        def compute_makespan(schedule, job_duration):
            num_machines = schedule.shape[0]
            num_jobs = schedule.shape[1] - 1
            makespan = 0
            
            for machine_idx in range(num_machines):
                for job_idx in range(num_jobs):
                    start_time = schedule[machine_idx, job_idx]
                    if start_time >= 0:
                        duration = job_duration[job_idx, machine_idx]
                        end_time = start_time + duration
                        makespan = max(makespan, end_time)
            
            return makespan
        
        makespan_before = compute_makespan(schedule_before, job_duration_before)
        makespan_after = compute_makespan(schedule_after, job_duration_after)
        
        # 计算改进百分比
        improvement = ((makespan_before - makespan_after) / makespan_before) * 100
        
        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        num_machines = schedule_before.shape[0]
        num_jobs = schedule_before.shape[1] - 1
        
        # 颜色方案
        base_colors = plt.cm.Set3(np.linspace(0, 1, 12))
        if num_jobs > 12:
            extra_colors = plt.cm.Set2(np.linspace(0, 1, 8))
            colors = np.vstack((base_colors, extra_colors))
            if num_jobs > 20:
                repeats = (num_jobs // 20) + 1
                colors = np.tile(colors, (repeats, 1))
        else:
            colors = base_colors
        colors = colors[:num_jobs]
        
        # 辅助函数：绘制单个甘特图
        def plot_gantt_on_ax(ax, schedule, job_duration, chart_title, title_color):
            for machine_idx in range(num_machines):
                stage_idx = -1
                if num_machine_per_stage:
                    stage_idx = machine_idx // num_machine_per_stage
                
                for job_idx in range(num_jobs):
                    start_time = schedule[machine_idx, job_idx]
                    if start_time < 0:
                        continue
                    duration = job_duration[job_idx, machine_idx]
                    
                    rect = Rectangle(
                        (start_time, machine_idx - 0.4),
                        duration, 0.8,
                        facecolor=colors[job_idx],
                        edgecolor='#333333',
                        linewidth=1.0,
                        alpha=0.9
                    )
                    ax.add_patch(rect)
                    
                    label_text = f'J{job_idx}'
                    if stage_idx >= 0:
                        label_text += f'\nS{stage_idx}'
                    
                    font_size = 8
                    # 简单的亮度计算
                    rgb = colors[job_idx][:3]
                    brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
                    text_color = 'white' if brightness < 0.5 else 'black'
                    
                    if duration > 1.0: # 只有足够宽才显示文字
                        ax.text(
                            start_time + duration / 2, machine_idx,
                            label_text,
                            ha='center', va='center',
                            fontsize=font_size, fontweight='bold',
                            color=text_color
                        )
            
            # 设置坐标轴
            current_makespan = compute_makespan(schedule, job_duration)
            ax.set_ylim(-0.5, num_machines - 0.5)
            ax.set_xlim(0, max(makespan_before, makespan_after) * 1.05)
            
            yticks = range(num_machines)
            yticklabels = []
            for i in range(num_machines):
                if num_machine_per_stage:
                    s_idx = i // num_machine_per_stage
                    m_idx = i % num_machine_per_stage
                    yticklabels.append(f'S{s_idx}-M{m_idx}')
                else:
                    yticklabels.append(f'M{i}')
            
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('时间 (Time)', fontsize=11, fontweight='bold')
            ax.set_ylabel('机器 (Machine)', fontsize=11, fontweight='bold')
            ax.set_title(f'{chart_title}\nMakespan: {current_makespan:.1f}', 
                        fontsize=14, fontweight='bold', color=title_color)
            ax.grid(True, axis='x', alpha=0.4, linestyle='--', color='gray')
            
            # 添加阶段分隔线
            if num_machine_per_stage:
                num_stages = num_machines // num_machine_per_stage
                for s in range(1, num_stages):
                    y_pos = s * num_machine_per_stage - 0.5
                    ax.axhline(y=y_pos, color='black', linestyle='-', linewidth=1.0, alpha=0.4)

        # 绘制训练前的甘特图
        plot_gantt_on_ax(axes[0], schedule_before, job_duration_before, '训练前 (Before)', 'red')
        
        # 绘制训练后的甘特图
        plot_gantt_on_ax(axes[1], schedule_after, job_duration_after, '训练后 (After)', 'green')
        
        # 总标题
        fig.suptitle(
            f"{title}\n改进 (Improvement): {improvement:.2f}% | "
            f"Makespan: {makespan_before:.1f} → {makespan_after:.1f}",
            fontsize=16, fontweight='bold', y=0.98
        )
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return {
            'makespan_before': makespan_before,
            'makespan_after': makespan_after,
            'improvement': improvement
        }
        
    except Exception as e:
        print(f"创建FFSP对比图时出错: {str(e)}")
        # 创建空白图片
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'对比图生成失败\n{str(e)}', 
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return {
            'makespan_before': 0,
            'makespan_after': 0,
            'improvement': 0
        }


def create_ffsp_statistics_plot(makespans, save_path, title="FFSP训练收敛曲线"):
    """
    创建FFSP训练过程中的makespan收敛曲线
    
    参数:
        makespans: makespan历史记录列表
        save_path: 图片保存路径
        title: 图表标题
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = list(range(1, len(makespans) + 1))
        
        # 绘制makespan曲线
        ax.plot(epochs, makespans, 'b-', linewidth=2, marker='o', 
               markersize=5, label='Makespan')
        
        # 绘制最佳makespan水平线
        best_makespan = min(makespans)
        ax.axhline(y=best_makespan, color='r', linestyle='--', 
                  linewidth=2, label=f'最佳: {best_makespan:.2f}')
        
        # 设置标签和标题
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=12, fontweight='bold')
        ax.set_ylabel('完工时间 (Makespan)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 添加网格和图例
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=11)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"创建FFSP统计图时出错: {str(e)}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'统计图生成失败\n{str(e)}', 
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
