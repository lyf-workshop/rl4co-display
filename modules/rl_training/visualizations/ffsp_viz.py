"""
FFSP问题专用可视化函数
提供FFSP甘特图、调度对比图等可视化
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch

logger = logging.getLogger('rl4co_display')

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def create_ffsp_gantt_chart(td, schedule, save_path, title="FFSP调度甘特图"):
    """
    创建FFSP调度的甘特图
    
    参数:
        td: TensorDict，包含作业信息
        schedule: 调度矩阵 [num_machine, num_job+1]，记录每个作业在每台机器上的开始时间
        save_path: 图片保存路径
        title: 图表标题
    """
    try:
        # 提取数据
        if isinstance(schedule, torch.Tensor):
            schedule = schedule.cpu().numpy()
        
        # 获取job_duration（可能是run_time）
        if hasattr(td, 'get'):
            # 尝试获取job_duration或run_time
            if 'job_duration' in td.keys():
                job_duration = td.get('job_duration').cpu().numpy()
            elif 'run_time' in td.keys():
                job_duration = td.get('run_time').cpu().numpy()
            else:
                raise KeyError(f"TensorDict中没有job_duration或run_time键，可用键: {list(td.keys())}")
        else:
            if 'job_duration' in td:
                job_duration = td['job_duration'].cpu().numpy()
            elif 'run_time' in td:
                job_duration = td['run_time'].cpu().numpy()
            else:
                raise KeyError(f"TensorDict中没有job_duration或run_time键")
        
        # 处理batch维度（如果有的话）
        if job_duration.ndim == 3:
            job_duration = job_duration[0]  # 取第一个batch
        
        logger.debug("schedule shape: %s, job_duration shape: %s, min/max: %.2f/%.2f",
                     schedule.shape, job_duration.shape, schedule.min(), schedule.max())
        
        # schedule: [num_machine_total, num_job+1]
        # job_duration: [num_job+1, num_machine_total] 或 [num_job, num_machine_total]
        num_machines = schedule.shape[0]
        num_jobs = schedule.shape[1] - 1  # 排除dummy job
        
        logger.debug("num_machines: %d, num_jobs: %d, job_duration expected shape: (%d, %d) or (%d, %d)",
                     num_machines, num_jobs, num_jobs, num_machines, num_jobs + 1, num_machines)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(14, max(8, num_machines * 0.6)))
        
        # 为每个作业分配不同的颜色
        colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
        
        # 计算makespan
        makespan = 0
        
        # 绘制每台机器的调度
        for machine_idx in range(num_machines):
            for job_idx in range(num_jobs):
                start_time = schedule[machine_idx, job_idx]
                
                # 跳过未调度的作业（开始时间为负数或0）
                if start_time < 0:
                    continue
                
                # 获取作业时长 - 注意索引顺序可能需要调整
                try:
                    # 尝试标准索引：job_duration[job_idx, machine_idx]
                    if job_duration.shape[0] > job_idx and job_duration.shape[1] > machine_idx:
                        duration = job_duration[job_idx, machine_idx]
                    else:
                        # 索引越界，跳过
                        logger.warning("索引越界: job_idx=%d, machine_idx=%d, shape=%s", job_idx, machine_idx, job_duration.shape)
                        continue
                except Exception as e:
                    logger.error(f"获取duration失败: {e}")
                    continue
                
                # 确保duration是标量
                if hasattr(duration, 'item'):
                    duration = duration.item()
                
                end_time = start_time + duration
                
                # 更新makespan
                if hasattr(end_time, 'item'):
                    makespan = max(makespan, end_time.item())
                else:
                    makespan = max(makespan, end_time)
                
                # 绘制矩形块表示作业
                rect = Rectangle(
                    (start_time, machine_idx - 0.4),
                    duration,
                    0.8,
                    facecolor=colors[job_idx],
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.8
                )
                ax.add_patch(rect)
                
                # 在矩形内标注作业编号
                ax.text(
                    start_time + duration / 2,
                    machine_idx,
                    f'J{job_idx}',
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    color='white' if duration > 2 else 'black'
                )
        
        # 设置坐标轴
        ax.set_ylim(-0.5, num_machines - 0.5)
        ax.set_xlim(0, makespan * 1.05)
        
        # Y轴：机器编号
        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f'机器 {i}' for i in range(num_machines)])
        ax.invert_yaxis()  # 倒转Y轴，使机器0在顶部
        
        # X轴：时间
        ax.set_xlabel('时间', fontsize=12, fontweight='bold')
        ax.set_ylabel('机器', fontsize=12, fontweight='bold')
        
        # 设置标题
        ax.set_title(f"{title}\n完工时间(Makespan): {makespan:.1f}", 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 添加网格
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # 添加图例（作业列表）
        legend_elements = [
            Rectangle((0, 0), 1, 1, facecolor=colors[i], edgecolor='black', 
                     label=f'作业 {i}', alpha=0.8)
            for i in range(min(num_jobs, 10))  # 最多显示10个作业
        ]
        
        if num_jobs > 10:
            legend_elements.append(
                Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black',
                         label=f'... (共{num_jobs}个作业)', alpha=0.5)
            )
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.15, 1), fontsize=9)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return makespan
        
    except Exception as e:
        logger.error(f"创建FFSP甘特图时出错: {str(e)}")
        # 创建空白图片
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'甘特图生成失败\n{str(e)}', 
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return 0


def create_ffsp_schedule_comparison(td_before, td_after, schedule_before, schedule_after,
                                    save_path, title="FFSP训练前后对比"):
    """
    创建FFSP训练前后的调度对比图
    
    参数:
        td_before: 训练前的TensorDict
        td_after: 训练后的TensorDict
        schedule_before: 训练前的调度
        schedule_after: 训练后的调度
        save_path: 图片保存路径
        title: 图表标题
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
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        num_machines = schedule_before.shape[0]
        num_jobs = schedule_before.shape[1] - 1
        colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
        
        # 绘制训练前的甘特图
        ax = axes[0]
        for machine_idx in range(num_machines):
            for job_idx in range(num_jobs):
                start_time = schedule_before[machine_idx, job_idx]
                if start_time < 0:
                    continue
                duration = job_duration_before[job_idx, machine_idx]
                
                rect = Rectangle(
                    (start_time, machine_idx - 0.4),
                    duration, 0.8,
                    facecolor=colors[job_idx],
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.8
                )
                ax.add_patch(rect)
                ax.text(
                    start_time + duration / 2, machine_idx,
                    f'J{job_idx}',
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white' if duration > 2 else 'black'
                )
        
        ax.set_ylim(-0.5, num_machines - 0.5)
        ax.set_xlim(0, max(makespan_before, makespan_after) * 1.05)
        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
        ax.invert_yaxis()
        ax.set_xlabel('时间', fontsize=11, fontweight='bold')
        ax.set_ylabel('机器', fontsize=11, fontweight='bold')
        ax.set_title(f'训练前\nMakespan: {makespan_before:.1f}', 
                    fontsize=13, fontweight='bold', color='red')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # 绘制训练后的甘特图
        ax = axes[1]
        for machine_idx in range(num_machines):
            for job_idx in range(num_jobs):
                start_time = schedule_after[machine_idx, job_idx]
                if start_time < 0:
                    continue
                duration = job_duration_after[job_idx, machine_idx]
                
                rect = Rectangle(
                    (start_time, machine_idx - 0.4),
                    duration, 0.8,
                    facecolor=colors[job_idx],
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.8
                )
                ax.add_patch(rect)
                ax.text(
                    start_time + duration / 2, machine_idx,
                    f'J{job_idx}',
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='white' if duration > 2 else 'black'
                )
        
        ax.set_ylim(-0.5, num_machines - 0.5)
        ax.set_xlim(0, max(makespan_before, makespan_after) * 1.05)
        ax.set_yticks(range(num_machines))
        ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
        ax.invert_yaxis()
        ax.set_xlabel('时间', fontsize=11, fontweight='bold')
        ax.set_ylabel('机器', fontsize=11, fontweight='bold')
        ax.set_title(f'训练后\nMakespan: {makespan_after:.1f}', 
                    fontsize=13, fontweight='bold', color='green')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # 总标题
        fig.suptitle(
            f"{title}\n改进: {improvement:.2f}% | "
            f"Makespan: {makespan_before:.1f} → {makespan_after:.1f}",
            fontsize=15, fontweight='bold', y=0.98
        )
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'makespan_before': makespan_before,
            'makespan_after': makespan_after,
            'improvement': improvement
        }
        
    except Exception as e:
        logger.error(f"创建FFSP对比图时出错: {str(e)}")
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
        logger.error(f"创建FFSP统计图时出错: {str(e)}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'统计图生成失败\n{str(e)}', 
               ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
