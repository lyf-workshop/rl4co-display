"""
RL4CO训练模块测试工具
用于验证模块结构和导入是否正确
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


def test_imports():
    """测试所有模块导入"""
    print("=" * 60)
    print("Testing Module Imports...")
    print("=" * 60)
    
    tests = []
    
    # 测试基础模块
    try:
        from modules.rl_training import BaseTrainer, ProgressCallback
        print("[PASS] BaseTrainer and ProgressCallback imported")
        tests.append(("Base Module", True, None))
    except Exception as e:
        print(f"[FAIL] Base module import failed: {str(e)}")
        tests.append(("Base Module", False, str(e)))
    
    # 测试TSP训练器
    try:
        from modules.rl_training import TSPTrainer, train_tsp
        print("[PASS] TSPTrainer imported")
        tests.append(("TSP Trainer", True, None))
    except Exception as e:
        print(f"[FAIL] TSP trainer import failed: {str(e)}")
        tests.append(("TSP Trainer", False, str(e)))
    
    # 测试CVRP训练器
    try:
        from modules.rl_training import CVRPTrainer, train_cvrp
        print("[PASS] CVRPTrainer imported")
        tests.append(("CVRP Trainer", True, None))
    except Exception as e:
        print(f"[FAIL] CVRP trainer import failed: {str(e)}")
        tests.append(("CVRP Trainer", False, str(e)))
    
    # 测试统一入口
    try:
        from modules.rl_training import real_rl4co_training
        print("[PASS] real_rl4co_training entry point imported")
        tests.append(("Unified Entry", True, None))
    except Exception as e:
        print(f"[FAIL] Unified entry import failed: {str(e)}")
        tests.append(("Unified Entry", False, str(e)))
    
    # 测试TSP可视化
    try:
        from modules.rl_training.visualizations.tsp_viz import (
            create_tsp_route_animation,
            create_tsp_comparison_plot
        )
        print("[PASS] TSP visualization functions imported")
        tests.append(("TSP Visualization", True, None))
    except Exception as e:
        print(f"[FAIL] TSP visualization import failed: {str(e)}")
        tests.append(("TSP Visualization", False, str(e)))
    
    # 测试CVRP可视化
    try:
        from modules.rl_training.visualizations.cvrp_viz import (
            create_cvrp_route_animation,
            create_cvrp_comparison_plot
        )
        print("[PASS] CVRP visualization functions imported")
        tests.append(("CVRP Visualization", True, None))
    except Exception as e:
        print(f"[FAIL] CVRP visualization import failed: {str(e)}")
        tests.append(("CVRP Visualization", False, str(e)))
    
    # 测试通用可视化
    try:
        from modules.rl_training.visualizations.common import create_training_curve_plot
        print("[PASS] Common visualization functions imported")
        tests.append(("Common Visualization", True, None))
    except Exception as e:
        print(f"[FAIL] Common visualization import failed: {str(e)}")
        tests.append(("Common Visualization", False, str(e)))
    
    # 测试向后兼容性
    try:
        from modules.rl_training import create_route_animation
        print("[PASS] Backward compatibility (create_route_animation)")
        tests.append(("Backward Compat", True, None))
    except Exception as e:
        print(f"[FAIL] Backward compatibility import failed: {str(e)}")
        tests.append(("Backward Compat", False, str(e)))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in tests if success)
    total = len(tests)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Module structure is correct.")
        return True
    else:
        print("\n[WARNING] Failed tests:")
        for name, success, error in tests:
            if not success:
                print(f"  - {name}: {error}")
        return False


def test_module_structure():
    """测试模块文件结构"""
    print("\n" + "=" * 60)
    print("Checking File Structure...")
    print("=" * 60)
    
    base_dir = os.path.dirname(__file__)
    
    required_files = [
        '__init__.py',
        'base_trainer.py',
        'tsp_trainer.py',
        'cvrp_trainer.py',
        'training_functions.py',  # 保留用于向后兼容
        'README.md',
        'visualizations/__init__.py',
        'visualizations/common.py',
        'visualizations/tsp_viz.py',
        'visualizations/cvrp_viz.py',
        'visualizations/README.md',
        'templates/problem_trainer_template.py',
        'templates/visualization_template.py',
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        exists = os.path.exists(full_path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n[SUCCESS] All required files exist!")
        return True
    else:
        print("\n[WARNING] Some files are missing.")
        return False


def show_usage_examples():
    """显示使用示例"""
    print("\n" + "=" * 60)
    print("Usage Examples")
    print("=" * 60)
    
    examples = """
1. 使用统一入口训练（推荐）：
   
   from modules.rl_training import real_rl4co_training
   
   real_rl4co_training(
       config={'problem': 'tsp', 'epochs': 10, 'batch_size': 512},
       session_id='session-123',
       user_id=1,
       queue=message_queue,
       training_status=status_dict,
       get_background_db_func=get_db_connection
   )

2. 直接使用TSP训练器：
   
   from modules.rl_training import train_tsp
   
   train_tsp(config, session_id, user_id, queue, training_status, get_db)

3. 直接使用CVRP训练器：
   
   from modules.rl_training import train_cvrp
   
   train_cvrp(config, session_id, user_id, queue, training_status, get_db)

4. 创建TSP路线动画：
   
   from modules.rl_training.visualizations.tsp_viz import create_tsp_route_animation
   
   create_tsp_route_animation(
       td=tensor_dict,
       actions=[0, 3, 1, 4, 2],
       save_path='tsp_route.gif',
       title='TSP路线生成过程',
       fps=2
   )

5. 创建CVRP路线动画：
   
   from modules.rl_training.visualizations.cvrp_viz import create_cvrp_route_animation
   
   create_cvrp_route_animation(
       td=tensor_dict,
       actions=[0, 3, 1, 0, 4, 2, 0],
       save_path='cvrp_route.gif',
       title='CVRP配送路线',
       fps=2
   )
"""
    print(examples)


def show_directory_tree():
    """显示目录树结构"""
    print("\n" + "=" * 60)
    print("Module Directory Structure")
    print("=" * 60)
    
    tree = """
modules/rl_training/
│
├── __init__.py                      # 统一导出接口
│
├── base_trainer.py                  # 通用训练基类
│   ├── BaseTrainer                  # 训练器基类
│   └── ProgressCallback             # 训练进度回调
│
├── tsp_trainer.py                   # TSP专用训练器
│   ├── TSPTrainer                   # TSP训练器类
│   └── train_tsp()                  # TSP训练入口函数
│
├── cvrp_trainer.py                  # CVRP专用训练器
│   ├── CVRPTrainer                  # CVRP训练器类
│   └── train_cvrp()                 # CVRP训练入口函数
│
├── training_functions.py            # [保留] 旧版本（向后兼容）
│
├── README.md                        # 模块说明文档
│
├── test_module.py                   # 本测试脚本
│
├── templates/                       # 模板文件夹
│   ├── problem_trainer_template.py # 新问题训练器模板
│   └── visualization_template.py   # 新可视化函数模板
│
└── visualizations/                  # 可视化模块
    ├── __init__.py                  # 可视化导出接口
    ├── README.md                    # 可视化说明文档
    ├── common.py                    # 通用可视化函数
    │   └── create_training_curve_plot()
    ├── tsp_viz.py                   # TSP可视化
    │   ├── create_tsp_route_animation()
    │   └── create_tsp_comparison_plot()
    └── cvrp_viz.py                  # CVRP可视化
        ├── create_cvrp_route_animation()
        └── create_cvrp_comparison_plot()
"""
    print(tree)


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("RL4CO Training Module Test Tool")
    print("=" * 60)
    
    # 显示目录结构
    show_directory_tree()
    
    # 测试文件结构
    structure_ok = test_module_structure()
    
    # 测试导入
    imports_ok = test_imports()
    
    # 显示使用示例
    show_usage_examples()
    
    # 最终结果
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    
    if structure_ok and imports_ok:
        print("[SUCCESS] Module is correctly configured and ready to use!")
        return 0
    else:
        print("[ERROR] Issues found, please fix and retry.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

