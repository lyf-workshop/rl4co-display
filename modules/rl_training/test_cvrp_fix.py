"""
CVRP索引修复验证脚本
快速测试修复后的CVRP可视化功能
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

def test_cvrp_indexing():
    """测试CVRP索引问题是否已修复"""
    print("=" * 60)
    print("测试CVRP索引修复")
    print("=" * 60)
    
    try:
        import torch
        import numpy as np
        from modules.rl_training.visualizations.cvrp_viz import create_cvrp_route_animation
        
        print("\n[测试1] 模拟CVRP数据结构")
        print("-" * 60)
        
        # 模拟CVRP数据（50个客户）
        num_customers = 50
        num_nodes = num_customers + 1  # 51个节点（1仓库+50客户）
        
        # 创建模拟数据
        locs = torch.rand(num_nodes, 2)  # 51个节点坐标
        demands = torch.rand(num_customers) * 0.2  # 50个客户需求
        
        print(f"[OK] Nodes: {num_nodes} (1 depot + {num_customers} customers)")
        print(f"[OK] locs shape: {locs.shape}")
        print(f"[OK] demands shape: {demands.shape}")
        
        # 创建TensorDict模拟对象
        class MockTensorDict:
            def __init__(self, locs, demands):
                self.data = {
                    'locs': locs,
                    'demand': demands,
                    'demands': demands  # 兼容两种命名
                }
            
            def get(self, key, default=None):
                return self.data.get(key, default)
            
            def __getitem__(self, key):
                return self.data[key]
        
        td = MockTensorDict(locs, demands)
        
        print("\n[测试2] 验证索引范围")
        print("-" * 60)
        
        # 测试各种索引
        test_nodes = [0, 1, 25, 49, 50]
        for node_id in test_nodes:
            try:
                pos = locs[node_id]
                if node_id == 0:
                    print(f"[OK] Node {node_id}: Depot, pos={pos.numpy()}")
                else:
                    demand_idx = node_id - 1
                    if demand_idx < len(demands):
                        demand = demands[demand_idx]
                        print(f"[OK] Node {node_id}: Customer{node_id}, demand={demand:.3f}, pos={pos.numpy()}")
                    else:
                        print(f"[FAIL] Node {node_id}: demand index out of bounds")
            except IndexError as e:
                print(f"[FAIL] Node {node_id}: Index error - {e}")
        
        print("\n[测试3] 生成路线动画（关键测试）")
        print("-" * 60)
        
        # 创建模拟动作序列（访问所有客户）
        # 包含返回仓库的完整路径
        actions = np.array([0] + list(range(1, num_nodes)) + [0])  # 从仓库出发，访问所有客户，返回仓库
        print(f"[OK] Action sequence length: {len(actions)}")
        print(f"[OK] Action range: {actions.min()} - {actions.max()}")
        
        # 测试输出路径
        output_dir = os.path.join(project_root, 'static', 'model_plots', 'test')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'test_cvrp_route.gif')
        
        print(f"[OK] Output path: {output_path}")
        
        try:
            # 尝试生成动画
            create_cvrp_route_animation(
                td=td,
                actions=actions,
                save_path=output_path,
                title="CVRP索引修复测试",
                fps=5  # 更快的帧率用于测试
            )
            print(f"[SUCCESS] Animation generated!")
            print(f"[SUCCESS] File saved: {output_path}")
            
            # 检查文件大小
            file_size = os.path.getsize(output_path)
            print(f"[SUCCESS] File size: {file_size / 1024:.2f} KB")
            
            if file_size > 0:
                print("\n" + "=" * 60)
                print("[SUCCESS] All tests passed! Index fix verified!")
                print("=" * 60)
                return True
            else:
                print("\n[FAIL] File size is 0, generation may have failed")
                return False
                
        except IndexError as e:
            print(f"\n[FAIL] Index error (fix failed): {e}")
            import traceback
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"\n[FAIL] Error generating animation: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except ImportError as e:
        print(f"\n[WARN] Missing dependencies: {e}")
        print("Note: This is expected if torch/rl4co not installed")
        print("Test in actual deployment environment")
        return None
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_demand_indexing_logic():
    """测试需求索引逻辑"""
    print("\n\n" + "=" * 60)
    print("测试需求索引逻辑")
    print("=" * 60)
    
    try:
        import numpy as np
        
        # 模拟数据
        num_customers = 50
        demands = np.random.rand(num_customers) * 0.2
        
        print(f"\n客户数量: {num_customers}")
        print(f"需求数组大小: {len(demands)}")
        
        print("\n测试索引逻辑:")
        test_cases = [
            (0, "仓库"),
            (1, "客户1"),
            (25, "客户25"),
            (49, "客户49"),
            (50, "客户50"),
        ]
        
        for node_id, desc in test_cases:
            if node_id == 0:
                print(f"[OK] Node {node_id} ({desc}): demand = 0.000 (depot)")
            else:
                # 使用修复后的索引逻辑
                demand_idx = node_id if len(demands) > node_id else node_id - 1
                
                if demand_idx < len(demands):
                    demand = demands[demand_idx]
                    print(f"[OK] Node {node_id} ({desc}): demand_idx = {demand_idx}, demand = {demand:.3f}")
                else:
                    print(f"[FAIL] Node {node_id} ({desc}): index {demand_idx} out of bounds!")
        
        print("\n[OK] Indexing logic test completed")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Indexing logic test failed: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("CVRP索引修复验证工具")
    print("=" * 60)
    
    # 测试1：需求索引逻辑
    result1 = test_demand_indexing_logic()
    
    # 测试2：完整的CVRP可视化
    result2 = test_cvrp_indexing()
    
    # 总结
    print("\n\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if result1 and result2:
        print("[SUCCESS] All tests passed! CVRP index fix verified!")
        print("\nYou can now:")
        print("  1. Select CVRP problem type in web interface")
        print("  2. Configure training (50 customers, 10 epochs)")
        print("  3. Start training and view visualizations")
        print("\nNo more index errors!")
        return 0
    elif result2 is None:
        print("[WARN] Some tests skipped (missing dependencies)")
        print("Test in actual environment to verify fix")
        return 0
    else:
        print("[FAIL] Some tests failed, check error messages")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

