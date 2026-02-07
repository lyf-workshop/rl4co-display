#!/usr/bin/env python3
"""
全局警告检查脚本
验证所有 Python 文件是否可以正确编译和导入
"""

import sys
import os
import py_compile
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 颜色定义
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

def print_header(text):
    """打印标题"""
    print(f"\n{GREEN}{'=' * 60}{NC}")
    print(f"{GREEN}{text}{NC}")
    print(f"{GREEN}{'=' * 60}{NC}\n")

def check_compile(file_path):
    """检查 Python 文件是否可以编译"""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except Exception as e:
        return False, str(e)

def check_import(module_name):
    """检查模块是否可以导入"""
    try:
        __import__(module_name)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    print_header("RL4CO Display - 全局警告检查")
    
    # ========== 1. 检查主要 Python 文件编译 ==========
    print(f"{YELLOW}1. 检查 Python 文件编译...{NC}\n")
    
    files_to_check = [
        'app.py',
        'app_auth.py',
        'app_pages.py',
        'app_training.py',
        'app_files.py',
        'app_stats.py',
        'app_compat.py',
        'auth_module.py',
        'logging_config.py',
        'model_database.py',
    ]
    
    compile_errors = []
    for filename in files_to_check:
        filepath = project_root / filename
        if filepath.exists():
            success, error = check_compile(str(filepath))
            if success:
                print(f"  ✅ {filename}")
            else:
                print(f"  ❌ {filename}: {error}")
                compile_errors.append((filename, error))
        else:
            print(f"  ⚠️  {filename}: 文件不存在")
    
    # ========== 2. 检查标准库导入 ==========
    print(f"\n{YELLOW}2. 检查标准库导入...{NC}\n")
    
    std_modules = ['json', 'os', 'time', 'logging', 'threading', 'uuid']
    import_errors = []
    
    for module in std_modules:
        success, error = check_import(module)
        if success:
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module}: {error}")
            import_errors.append((module, error))
    
    # ========== 3. 检查第三方库导入 ==========
    print(f"\n{YELLOW}3. 检查第三方库导入...{NC}\n")
    
    third_party = {
        'flask': 'Flask==3.0.0',
        'mysql.connector': 'mysql-connector-python',
        'numpy': 'numpy>=1.24.0',
        'matplotlib': 'matplotlib>=3.7.0',
    }
    
    missing_packages = []
    for module, package in third_party.items():
        success, error = check_import(module)
        if success:
            print(f"  ✅ {module} ({package})")
        else:
            print(f"  ❌ {module} ({package}): 未安装")
            missing_packages.append(package)
    
    # ========== 4. 检查 RL4CO (可选) ==========
    print(f"\n{YELLOW}4. 检查 RL4CO 库（可选）...{NC}\n")
    
    rl4co_modules = ['rl4co', 'tensordict', 'lightning']
    rl4co_available = True
    
    for module in rl4co_modules:
        success, error = check_import(module)
        if success:
            print(f"  ✅ {module}")
        else:
            print(f"  ⚠️  {module}: 未安装（将使用模拟训练）")
            rl4co_available = False
    
    # ========== 5. 检查项目模块导入 ==========
    print(f"\n{YELLOW}5. 检查项目模块导入...{NC}\n")
    
    # 切换到项目根目录
    os.chdir(project_root)
    
    project_modules = [
        'config.config',
        'logging_config',
        'auth_module',
        'model_database',
    ]
    
    module_errors = []
    for module in project_modules:
        success, error = check_import(module)
        if success:
            print(f"  ✅ {module}")
        else:
            print(f"  ❌ {module}: {error}")
            module_errors.append((module, error))
    
    # ========== 总结 ==========
    print_header("检查总结")
    
    total_issues = len(compile_errors) + len(import_errors) + len(missing_packages) + len(module_errors)
    
    if total_issues == 0:
        print(f"{GREEN}✅ 所有检查通过！代码没有真实错误。{NC}\n")
        
        if not rl4co_available:
            print(f"{YELLOW}⚠️  RL4CO 未安装，将使用模拟训练模式{NC}")
            print(f"{YELLOW}   如需真实训练，请运行: pip install rl4co{NC}\n")
        
        print(f"{GREEN}🎉 应用可以正常运行！{NC}\n")
        return 0
    else:
        print(f"{RED}❌ 发现 {total_issues} 个问题需要修复：{NC}\n")
        
        if compile_errors:
            print(f"{RED}编译错误：{NC}")
            for filename, error in compile_errors:
                print(f"  - {filename}: {error}")
        
        if import_errors:
            print(f"{RED}标准库导入错误：{NC}")
            for module, error in import_errors:
                print(f"  - {module}: {error}")
        
        if missing_packages:
            print(f"{YELLOW}缺少的包（运行以下命令安装）：{NC}")
            print(f"  pip install {' '.join(missing_packages)}")
        
        if module_errors:
            print(f"{RED}项目模块导入错误：{NC}")
            for module, error in module_errors:
                print(f"  - {module}: {error}")
        
        print()
        return 1
    
    # ========== IDE 警告说明 ==========
    print(f"\n{YELLOW}ℹ️  关于 IDE 警告：{NC}")
    print(f"如果你在 IDE 中看到警告（如 'json 未解析'），请查看：")
    print(f"  📖 IDE_WARNINGS_GUIDE.md - 详细说明和解决方案\n")

if __name__ == '__main__':
    sys.exit(main())
