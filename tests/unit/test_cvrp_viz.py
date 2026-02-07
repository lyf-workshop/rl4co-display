# -*- coding: utf-8 -*-
"""测试CVRP可视化模块"""

import sys
import os

# 添加项目根目录
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("测试CVRP可视化模块")
print("=" * 60)

# 测试1: 导入可视化函数
print("\n[测试1] 导入CVRP可视化函数...")
try:
    from modules.rl_training.visualizations.cvrp_viz import (
        create_cvrp_route_animation,
        create_cvrp_comparison_plot
    )
    print("[OK] CVRP可视化函数导入成功")
except Exception as e:
    print(f"[FAIL] 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试2: 导入CVRP训练器
print("\n[测试2] 导入CVRP训练器...")
try:
    from modules.rl_training import CVRPTrainer, train_cvrp
    print("[OK] CVRP训练器导入成功")
except Exception as e:
    print(f"[FAIL] 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 检查训练器的generate_visualizations方法
print("\n[测试3] 检查generate_visualizations方法...")
if hasattr(CVRPTrainer, 'generate_visualizations'):
    print("[OK] generate_visualizations方法存在")
    
    # 检查方法签名
    import inspect
    sig = inspect.signature(CVRPTrainer.generate_visualizations)
    print(f"[INFO] 方法签名: {sig}")
else:
    print("[FAIL] generate_visualizations方法不存在")
    sys.exit(1)

# 测试4: 检查保存文件记录的方式
print("\n[测试4] 检查文件保存逻辑...")
try:
    # 读取cvrp_trainer.py源码
    trainer_file = "modules/rl_training/cvrp_trainer.py"
    with open(trainer_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否使用正确的保存方式
    if 'self.bg_file_manager.save_file_record' in content:
        print("[OK] 使用正确的文件保存方式: self.bg_file_manager.save_file_record")
    else:
        print("[WARN] 可能使用了错误的文件保存方式")
    
    # 检查是否有错误的调用
    if 'self.save_file_record(' in content:
        print("[WARN] 发现可能的错误调用: self.save_file_record()")
        print("[建议] 应该使用: self.bg_file_manager.save_file_record()")
    
except Exception as e:
    print(f"[FAIL] 检查失败: {e}")

# 测试5: 检查索引问题
print("\n[测试5] 检查demands数组索引...")
try:
    # 读取cvrp_viz.py源码
    viz_file = "modules/rl_training/visualizations/cvrp_viz.py"
    with open(viz_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 查找enumerate和demands的使用
    issues_found = []
    for i, line in enumerate(lines, start=1):
        # 检查是否有危险的enumerate(start=1)和demands索引组合
        if 'enumerate' in line and 'start=1' in line:
            # 检查后续几行是否有demands索引
            context_start = max(0, i-1)
            context_end = min(len(lines), i+10)
            context = ''.join(lines[context_start:context_end])
            if 'demands[i]' in context or 'demands[idx]' in context:
                issues_found.append((i, line.strip()))
    
    if issues_found:
        print(f"[WARN] 发现可能的索引问题:")
        for line_num, line_content in issues_found:
            print(f"  第{line_num}行: {line_content}")
        print("[建议] 检查enumerate的索引是否正确")
    else:
        print("[OK] 未发现明显的索引问题")
        
        # 检查是否使用了修复后的方式
        viz_content = ''.join(lines)
        if 'enumerate(customers)' in viz_content and 'idx + 1' in viz_content:
            print("[OK] 使用了修复后的索引方式 (enumerate without start=1)")
        
except Exception as e:
    print(f"[FAIL] 检查失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)


