"""
SDVRP训练诊断脚本
帮助定位为什么没有生成可视化文件
"""

import sys
import os

# 添加项目根目录
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)


def check_training_route():
    """检查训练路由是否正确"""
    print("=" * 60)
    print("Step 1: 检查训练路由")
    print("=" * 60)
    
    try:
        from modules.rl_training import real_rl4co_training
        
        # 模拟配置
        config = {'problem': 'sdvrp', 'num_loc': 20}
        
        print(f"\n测试配置: {config}")
        print(f"问题类型: {config.get('problem')}")
        
        # 检查路由逻辑
        problem_type = config.get('problem', 'tsp').lower()
        print(f"处理后的问题类型: {problem_type}")
        
        if problem_type == 'sdvrp':
            print("[OK] SDVRP路由正确")
            return True
        else:
            print("[FAIL] SDVRP路由错误")
            return False
            
    except Exception as e:
        print(f"[FAIL] 路由检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_sdvrp_trainer():
    """检查SDVRP训练器是否正确实现"""
    print("\n" + "=" * 60)
    print("Step 2: 检查SDVRP训练器")
    print("=" * 60)
    
    try:
        from modules.rl_training import SDVRPTrainer, train_sdvrp
        
        print("[OK] SDVRPTrainer导入成功")
        print("[OK] train_sdvrp函数导入成功")
        
        # 检查训练器是否有generate_visualizations方法
        if hasattr(SDVRPTrainer, 'generate_visualizations'):
            print("[OK] generate_visualizations方法存在")
        else:
            print("[FAIL] generate_visualizations方法不存在")
            return False
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 训练器检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_visualization_functions():
    """检查SDVRP可视化函数"""
    print("\n" + "=" * 60)
    print("Step 3: 检查可视化函数")
    print("=" * 60)
    
    try:
        from modules.rl_training.visualizations.sdvrp_viz import (
            create_sdvrp_route_animation,
            create_sdvrp_comparison_plot,
            create_sdvrp_split_analysis
        )
        
        print("[OK] create_sdvrp_route_animation 导入成功")
        print("[OK] create_sdvrp_comparison_plot 导入成功")
        print("[OK] create_sdvrp_split_analysis 导入成功")
        
        return True
        
    except ImportError as e:
        print(f"[FAIL] 可视化函数导入失败: {e}")
        print("\n可能原因: torch或matplotlib未安装")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] 可视化函数检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_user_directories(user_id=1):
    """检查用户输出目录"""
    print("\n" + "=" * 60)
    print("Step 4: 检查用户输出目录")
    print("=" * 60)
    
    try:
        # 检查plots目录
        plots_dir = f"static/model_plots/user_{user_id}"
        checkpoints_dir = f"checkpoints/user_{user_id}"
        
        print(f"\n检查目录:")
        print(f"  Plots: {plots_dir}")
        print(f"  Checkpoints: {checkpoints_dir}")
        
        plots_exists = os.path.exists(plots_dir)
        checkpoints_exists = os.path.exists(checkpoints_dir)
        
        print(f"\n状态:")
        print(f"  Plots目录: {'[OK] 存在' if plots_exists else '[WARN] 不存在'}")
        print(f"  Checkpoints目录: {'[OK] 存在' if checkpoints_exists else '[WARN] 不存在'}")
        
        # 列出plots目录中的文件
        if plots_exists:
            files = os.listdir(plots_dir)
            print(f"\n  Plots目录文件数: {len(files)}")
            
            # 查找SDVRP相关文件
            sdvrp_files = [f for f in files if 'sdvrp' in f.lower() or 'split' in f.lower()]
            print(f"  SDVRP相关文件: {len(sdvrp_files)}")
            
            if sdvrp_files:
                print("\n  SDVRP文件列表:")
                for f in sdvrp_files[:10]:  # 只显示前10个
                    file_path = os.path.join(plots_dir, f)
                    size = os.path.getsize(file_path)
                    print(f"    - {f} ({size/1024:.1f} KB)")
            else:
                print("\n  [WARN] 未找到SDVRP相关文件")
                
                # 显示最近的文件
                if files:
                    print("\n  最近的文件:")
                    files_sorted = sorted(files, key=lambda x: os.path.getmtime(os.path.join(plots_dir, x)), reverse=True)
                    for f in files_sorted[:5]:
                        print(f"    - {f}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 目录检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_sessions(user_id=1):
    """检查训练会话记录"""
    print("\n" + "=" * 60)
    print("Step 5: 检查训练会话记录")
    print("=" * 60)
    
    try:
        from config import Config
        import mysql.connector as mysql_connector
        
        # 连接数据库
        db = mysql_connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DB,
        )
        
        cursor = db.cursor(dictionary=True)
        
        # 查询最近的训练会话
        cursor.execute("""
            SELECT session_id, model_type, problem_type, status, start_time
            FROM training_sessions
            WHERE user_id = %s
            ORDER BY start_time DESC
            LIMIT 5
        """, (user_id,))
        
        sessions = cursor.fetchall()
        
        print(f"\n最近的训练会话 (用户{user_id}):")
        if sessions:
            for session in sessions:
                print(f"\n  会话ID: {session['session_id'][:16]}...")
                print(f"    问题类型: {session['problem_type']}")
                print(f"    模型: {session['model_type']}")
                print(f"    状态: {session['status']}")
                print(f"    时间: {session['start_time']}")
                
                # 检查该会话的文件
                cursor.execute("""
                    SELECT file_name, file_type, file_size
                    FROM training_files
                    WHERE session_id = %s AND user_id = %s
                """, (session['session_id'], user_id))
                
                files = cursor.fetchall()
                if files:
                    print(f"    生成文件: {len(files)}个")
                    for f in files:
                        print(f"      - {f['file_type']}: {f['file_name']} ({f['file_size']/1024:.1f} KB)")
                else:
                    print(f"    生成文件: 0个 [WARN] 没有文件！")
        else:
            print("  [WARN] 没有训练记录")
        
        db.close()
        return True
        
    except Exception as e:
        print(f"[FAIL] 数据库检查失败: {e}")
        print("提示: 确保MySQL正在运行且配置正确")
        return False


def provide_solutions():
    """提供解决方案"""
    print("\n" + "=" * 60)
    print("可能的问题和解决方案")
    print("=" * 60)
    
    solutions = """
    问题1: 训练使用了旧版代码
    ────────────────────────────────
    原因: app.py可能还在使用 training_functions.py
    解决: 
      1. 重启Flask应用 (python app.py)
      2. 确保导入了新的模块
      3. 清除Python缓存: rm -rf **/__pycache__
    
    问题2: SDVRP路由未生效
    ────────────────────────────────
    原因: 问题类型识别错误
    解决:
      1. 检查前端发送的problem参数
      2. 确认是 'sdvrp' 而不是 'SDVRP'
      3. 查看浏览器控制台
    
    问题3: 可视化生成时异常
    ────────────────────────────────
    原因: 生成过程中出错但被捕获
    解决:
      1. 查看控制台错误日志
      2. 检查 generate_visualizations 返回值
      3. 确认torch/matplotlib已安装
    
    问题4: 文件保存路径错误
    ────────────────────────────────
    原因: 用户目录不存在或权限问题
    解决:
      1. 检查 static/model_plots/user_X 目录
      2. 确认有写入权限
      3. 手动创建目录
    
    问题5: 数据库记录失败
    ────────────────────────────────
    原因: 文件记录未保存到数据库
    解决:
      1. 检查MySQL连接
      2. 查看training_files表
      3. 确认FileManager正常工作
    """
    
    print(solutions)


def main():
    """主诊断流程"""
    print("\n" + "=" * 60)
    print("SDVRP训练诊断工具")
    print("=" * 60)
    
    results = []
    
    # 步骤1: 检查路由
    result = check_training_route()
    results.append(("训练路由", result))
    
    # 步骤2: 检查训练器
    result = check_sdvrp_trainer()
    results.append(("SDVRP训练器", result))
    
    # 步骤3: 检查可视化函数
    result = check_visualization_functions()
    results.append(("可视化函数", result))
    
    # 步骤4: 检查目录
    user_id = input("\n请输入你的用户ID (默认1): ").strip() or "1"
    result = check_user_directories(int(user_id))
    results.append(("用户目录", result))
    
    # 步骤5: 检查数据库
    result = check_training_sessions(int(user_id))
    results.append(("训练记录", result))
    
    # 总结
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n检查项: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    
    print("\n详细结果:")
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {name}")
    
    # 提供解决方案
    if passed < total:
        provide_solutions()
    else:
        print("\n[SUCCESS] 所有检查通过！")
        print("\n如果仍然没有可视化文件，请检查:")
        print("  1. 训练是否真的完成（查看状态）")
        print("  2. 是否选择了正确的问题类型（sdvrp）")
        print("  3. 查看控制台日志中的错误信息")
        print("  4. 重启Flask应用并重新训练")
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[INFO] 诊断已中断")
        sys.exit(0)


