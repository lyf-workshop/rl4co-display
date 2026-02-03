"""
问题模块测试脚本
验证所有问题类型是否正确实现
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


def test_module_structure():
    """测试模块文件结构"""
    print("=" * 60)
    print("Testing Module Structure...")
    print("=" * 60)
    
    base_dir = os.path.dirname(__file__)
    
    required_files = [
        '__init__.py',
        'base_problem.py',
        'tsp.py',
        'cvrp.py',
        'README.md',
        'APP_INTEGRATION_GUIDE.md',
        'templates/problem_template.py',
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


def test_imports():
    """测试模块导入"""
    print("\n" + "=" * 60)
    print("Testing Module Imports...")
    print("=" * 60)
    
    tests = []
    
    # 测试基类导入
    try:
        from modules.problems import BaseProblem
        print("[PASS] BaseProblem imported")
        tests.append(("BaseProblem", True, None))
    except Exception as e:
        print(f"[FAIL] BaseProblem import failed: {str(e)}")
        tests.append(("BaseProblem", False, str(e)))
    
    # 测试TSP问题
    try:
        from modules.problems import TSProblem
        print("[PASS] TSProblem imported")
        tests.append(("TSProblem", True, None))
    except Exception as e:
        print(f"[FAIL] TSProblem import failed: {str(e)}")
        tests.append(("TSProblem", False, str(e)))
    
    # 测试CVRP问题
    try:
        from modules.problems import CVRProblem
        print("[PASS] CVRProblem imported")
        tests.append(("CVRProblem", True, None))
    except Exception as e:
        print(f"[FAIL] CVRProblem import failed: {str(e)}")
        tests.append(("CVRProblem", False, str(e)))
    
    # 测试注册表
    try:
        from modules.problems import PROBLEM_REGISTRY, PROBLEM_INFO
        print("[PASS] Problem registry imported")
        tests.append(("Registry", True, None))
    except Exception as e:
        print(f"[FAIL] Registry import failed: {str(e)}")
        tests.append(("Registry", False, str(e)))
    
    # 测试工具函数
    try:
        from modules.problems import (
            get_problem_class,
            get_problem_info,
            list_available_problems,
            list_problems_by_category
        )
        print("[PASS] Utility functions imported")
        tests.append(("Utilities", True, None))
    except Exception as e:
        print(f"[FAIL] Utilities import failed: {str(e)}")
        tests.append(("Utilities", False, str(e)))
    
    passed = sum(1 for _, success, _ in tests if success)
    total = len(tests)
    
    print(f"\nPassed: {passed}/{total}")
    return passed == total


def test_problem_registry():
    """测试问题注册表"""
    print("\n" + "=" * 60)
    print("Testing Problem Registry...")
    print("=" * 60)
    
    try:
        from modules.problems import PROBLEM_REGISTRY, PROBLEM_INFO
        
        print(f"\nRegistered problems: {len(PROBLEM_REGISTRY)}")
        for problem_type, problem_class in PROBLEM_REGISTRY.items():
            info = PROBLEM_INFO.get(problem_type, {})
            status = info.get('status', 'unknown')
            cn_name = info.get('cn_name', 'N/A')
            print(f"  - {problem_type}: {problem_class.__name__} ({cn_name}) [{status}]")
        
        print(f"\nProblem info entries: {len(PROBLEM_INFO)}")
        
        return True
    except Exception as e:
        print(f"\n[FAIL] Registry test failed: {str(e)}")
        return False


def test_problem_classes():
    """测试问题类功能"""
    print("\n" + "=" * 60)
    print("Testing Problem Classes...")
    print("=" * 60)
    
    try:
        from modules.problems import get_problem_class
        
        # 测试TSP
        print("\n[Test 1] TSP Problem")
        print("-" * 40)
        TSProblem = get_problem_class('tsp')
        tsp = TSProblem({'num_loc': 50, 'batch_size': 512})
        
        print(f"Problem type: {tsp.get_problem_type()}")
        print(f"Problem name: {tsp.get_problem_name()}")
        print(f"Num locations: {tsp.num_loc}")
        
        valid, error_msg = tsp.validate_config()
        print(f"Config valid: {valid}")
        if not valid:
            print(f"Error: {error_msg}")
        
        print(f"Features: {tsp.get_problem_features()}")
        
        # 测试CVRP
        print("\n[Test 2] CVRP Problem")
        print("-" * 40)
        CVRProblem = get_problem_class('cvrp')
        cvrp = CVRProblem({
            'num_loc': 50,
            'batch_size': 512,
            'vehicle_capacity': 1.0
        })
        
        print(f"Problem type: {cvrp.get_problem_type()}")
        print(f"Problem name: {cvrp.get_problem_name()}")
        print(f"Num locations: {cvrp.num_loc}")
        print(f"Vehicle capacity: {cvrp.vehicle_capacity}")
        
        valid, error_msg = cvrp.validate_config()
        print(f"Config valid: {valid}")
        if not valid:
            print(f"Error: {error_msg}")
        
        print(f"Features: {cvrp.get_problem_features()}")
        
        # 测试无效问题类型
        print("\n[Test 3] Invalid Problem Type")
        print("-" * 40)
        try:
            invalid = get_problem_class('invalid_problem')
            print("[FAIL] Should raise ValueError")
            return False
        except ValueError as e:
            print(f"[PASS] Correctly raised ValueError: {str(e)[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Problem class test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_utility_functions():
    """测试工具函数"""
    print("\n" + "=" * 60)
    print("Testing Utility Functions...")
    print("=" * 60)
    
    try:
        from modules.problems import (
            list_available_problems,
            list_problems_by_category,
            get_problem_info
        )
        
        # 测试列出所有问题
        print("\n[Test 1] List Available Problems")
        print("-" * 40)
        problems = list_available_problems()
        print(f"Available problems: {len(problems)}")
        for problem_type, status, cn_name in problems:
            print(f"  - {problem_type}: {cn_name} [{status}]")
        
        # 测试按类别列出
        print("\n[Test 2] List Problems by Category")
        print("-" * 40)
        categories = list_problems_by_category()
        print(f"Categories: {len(categories)}")
        for category, problem_list in categories.items():
            print(f"\n{category}:")
            for problem in problem_list:
                print(f"  - {problem['type']}: {problem['cn_name']} [{problem['status']}]")
        
        # 测试获取问题信息
        print("\n[Test 3] Get Problem Info")
        print("-" * 40)
        info = get_problem_info('tsp')
        print(f"TSP Info:")
        print(f"  Name: {info['name']}")
        print(f"  Full name: {info['full_name']}")
        print(f"  CN name: {info['cn_name']}")
        print(f"  Category: {info['category']}")
        print(f"  Difficulty: {info['difficulty']}")
        print(f"  Status: {info['status']}")
        print(f"  Description: {info['description']}")
        print(f"  Params: {info['params']}")
        print(f"  Features: {info['features']}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Utility function test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_creation():
    """测试环境创建（需要RL4CO）"""
    print("\n" + "=" * 60)
    print("Testing Environment Creation...")
    print("=" * 60)
    
    try:
        import torch
        from modules.problems import get_problem_class
        
        # 测试TSP环境
        print("\n[Test 1] TSP Environment")
        print("-" * 40)
        try:
            TSProblem = get_problem_class('tsp')
            tsp = TSProblem({'num_loc': 20})
            env = tsp.create_environment()
            print(f"[PASS] TSP environment created: {env.name}")
        except ImportError as e:
            print(f"[SKIP] RL4CO not installed: {str(e)}")
            return None
        except Exception as e:
            print(f"[FAIL] TSP environment creation failed: {str(e)}")
            return False
        
        # 测试CVRP环境
        print("\n[Test 2] CVRP Environment")
        print("-" * 40)
        try:
            CVRProblem = get_problem_class('cvrp')
            cvrp = CVRProblem({'num_loc': 20, 'vehicle_capacity': 1.0})
            env = cvrp.create_environment()
            print(f"[PASS] CVRP environment created: {env.name}")
        except Exception as e:
            print(f"[FAIL] CVRP environment creation failed: {str(e)}")
            return False
        
        return True
        
    except ImportError:
        print("\n[SKIP] torch not installed, skipping environment tests")
        return None
    except Exception as e:
        print(f"\n[FAIL] Environment test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_functions():
    """测试可视化函数（需要可视化模块）"""
    print("\n" + "=" * 60)
    print("Testing Visualization Functions...")
    print("=" * 60)
    
    try:
        from modules.problems import get_problem_class
        
        # 测试TSP可视化
        print("\n[Test 1] TSP Visualization Functions")
        print("-" * 40)
        TSProblem = get_problem_class('tsp')
        tsp = TSProblem({'num_loc': 20})
        viz_funcs = tsp.get_visualization_functions()
        print(f"[PASS] TSP viz functions: {list(viz_funcs.keys())}")
        
        # 测试CVRP可视化
        print("\n[Test 2] CVRP Visualization Functions")
        print("-" * 40)
        CVRProblem = get_problem_class('cvrp')
        cvrp = CVRProblem({'num_loc': 20, 'vehicle_capacity': 1.0})
        viz_funcs = cvrp.get_visualization_functions()
        print(f"[PASS] CVRP viz functions: {list(viz_funcs.keys())}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Visualization test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Problem Module Test Suite")
    print("=" * 60)
    
    results = []
    
    # 测试模块结构
    result = test_module_structure()
    results.append(("Module Structure", result))
    
    # 测试导入
    result = test_imports()
    results.append(("Imports", result))
    
    # 测试注册表
    result = test_problem_registry()
    results.append(("Registry", result))
    
    # 测试问题类
    result = test_problem_classes()
    results.append(("Problem Classes", result))
    
    # 测试工具函数
    result = test_utility_functions()
    results.append(("Utility Functions", result))
    
    # 测试环境创建
    result = test_environment_creation()
    results.append(("Environment Creation", result))
    
    # 测试可视化函数
    result = test_visualization_functions()
    results.append(("Visualization Functions", result))
    
    # 总结
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)
    total = len(results)
    
    print(f"\nResults:")
    for test_name, result in results:
        if result is True:
            status = "[PASS]"
        elif result is False:
            status = "[FAIL]"
        else:
            status = "[SKIP]"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if failed == 0:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[ERROR] {failed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)


