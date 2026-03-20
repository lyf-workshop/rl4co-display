"""
全组合冒烟测试脚本
测试所有「问题类型 × 策略模型 × 训练算法」的有效组合，
使用极小配置（1 epoch、极少节点）快速验证每个组合是否能正常初始化并完成训练。

运行方式：
    cd F:/Github/rl4co-display
    python test_all_combinations.py

可选参数（直接修改下方 CONFIG 区域）：
    EPOCHS          - 每次测试的训练轮数（默认 1）
    BATCH_SIZE      - 批次大小（默认 16）
    TRAIN_DATA_SIZE - 训练样本数（默认 200）
    VAL_DATA_SIZE   - 验证样本数（默认 50）
    TIMEOUT_SEC     - 单个组合超时时间（秒，默认 120）
    FILTER_POLICY   - 只测试某个策略（如 'symnco'，None=全部）
    FILTER_PROBLEM  - 只测试某个问题（如 'tsp'，None=全部）
    SKIP_SLOW       - 跳过 FFSP 等较慢的组合
"""

import sys
import os
import queue
import json
import time
import signal
import traceback
import threading
from datetime import datetime, timedelta

# ============================================================
# 测试参数配置区域（按需修改）
# ============================================================
EPOCHS          = 1        # 每个组合的训练轮数
BATCH_SIZE      = 16       # 批次大小（极小以加速）
TRAIN_DATA_SIZE = 200      # 训练数据量（轻量问题）
VAL_DATA_SIZE   = 50       # 验证数据量
TIMEOUT_SEC     = 360      # 默认单个组合超时（秒）
FILTER_POLICY   = None     # None = 全部；或指定如 'symnco'
FILTER_PROBLEM  = None     # None = 全部；或指定如 'tsp'
SKIP_SLOW       = False    # True = 跳过 FFSP（耗时较长）
USE_GPU         = False    # True = 尝试使用 GPU；False = 强制 CPU

# 按问题复杂度设置不同超时（秒）
# PPO/A2C 比 REINFORCE 慢 3-5x；CVRP/SDVRP 比 TSP 慢
TIMEOUT_PER_PROBLEM = {
    'tsp':    300,
    'atsp':   300,
    'mtsp':   600,
    'cvrp':   600,
    'sdvrp':  600,
    'vrptw':  600,
    'pdp':    300,
    'op':     300,
    'pctsp':  300,
    'spctsp': 300,
    'ffsp':   300,
}
# ============================================================

# 确保从项目根目录运行
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

# ── 颜色输出 ────────────────────────────────────────────────
try:
    import colorama
    colorama.init(autoreset=True)
    GREEN  = colorama.Fore.GREEN
    RED    = colorama.Fore.RED
    YELLOW = colorama.Fore.YELLOW
    CYAN   = colorama.Fore.CYAN
    BOLD   = colorama.Style.BRIGHT
    RESET  = colorama.Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ''


# ────────────────────────────────────────────────────────────
# 问题特定的默认配置（满足各 trainer 的必要参数）
# ────────────────────────────────────────────────────────────
PROBLEM_EXTRA_CONFIGS = {
    'tsp':    {'num_loc': 10},
    'atsp':   {'num_loc': 10},
    'mtsp':   {'num_loc': 10, 'num_agents': 2, 'cost_type': 'minmax'},
    'cvrp':   {'num_loc': 10, 'vehicle_capacity': 1.0},
    'sdvrp':  {'num_loc': 10, 'vehicle_capacity': 1.0},
    'vrptw':  {
        'num_loc': 10,
        'vehicle_capacity': 1.0,
        'time_window_width': 100,
        'service_time': 10,
        'max_processing_time': 500,
        'hard_time_windows': False,
    },
    'pdp':    {'num_loc': 10},   # PDP num_loc 代表取货点数（实际节点 = num_loc×2+1）
    'op':     {'num_loc': 10, 'max_length': 2.0, 'prize_type': 'dist'},
    'pctsp':  {'num_loc': 10, 'penalty_factor': 3.0},
    'spctsp': {'num_loc': 10, 'penalty_factor': 3.0},
    'ffsp':   {'num_stage': 2, 'num_machine': 2, 'num_job': 5,
               'min_time': 1, 'max_time': 5, 'flatten_stages': True},
}


# ────────────────────────────────────────────────────────────
# 从兼容性矩阵生成所有待测组合
# ────────────────────────────────────────────────────────────
def build_test_cases():
    from modules.compatibility import (
        POLICY_PROBLEM_COMPATIBILITY,
        POLICY_ALGORITHM_COMPATIBILITY,
    )

    # 排除别名，只保留正式名称
    ALIAS_KEYS = {'am', 'ptr'}
    cases = []

    for policy, problems in POLICY_PROBLEM_COMPATIBILITY.items():
        if policy in ALIAS_KEYS:
            continue
        if FILTER_POLICY and policy != FILTER_POLICY:
            continue

        algos = POLICY_ALGORITHM_COMPATIBILITY.get(policy, ['reinforce'])

        for problem in problems:
            if FILTER_PROBLEM and problem != FILTER_PROBLEM:
                continue
            if SKIP_SLOW and problem == 'ffsp':
                continue

            for algo in algos:
                cases.append({
                    'problem': problem,
                    'policy':  policy,
                    'algorithm': algo,
                })

    return cases


# ────────────────────────────────────────────────────────────
# 测试产物清理（每次测试结束后删除 checkpoint 和生成的图片）
# ────────────────────────────────────────────────────────────
TEST_USER_ID = 0
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints', f'user_{TEST_USER_ID}')
PLOTS_DIR      = os.path.join(ROOT_DIR, 'static', 'model_plots', f'user_{TEST_USER_ID}')


def cleanup_test_artifacts(problem: str, policy: str, session_id: str):
    """
    删除本次测试产生的所有临时文件：
      - checkpoints/user_0/{problem}-{policy}.ckpt
      - lightning_logs 里本次产生的 checkpoint（trainer 自动保存）
      - static/model_plots/user_0/ 下含 session_id 前缀的所有图片/GIF
      - static/model_plots/user_0/ 下含 session_id 前缀的训练曲线图
    """
    removed = []

    # 1. 项目级 checkpoint（base_trainer 保存的那个）
    ckpt = os.path.join(CHECKPOINT_DIR, f'{problem}-{policy}.ckpt')
    if os.path.exists(ckpt):
        try:
            os.remove(ckpt)
            removed.append(ckpt)
        except OSError:
            pass

    # 2. 可视化图片 / GIF / 训练曲线（文件名包含 session_id 前8位）
    sid_prefix = session_id[:8]
    if os.path.isdir(PLOTS_DIR):
        for fname in os.listdir(PLOTS_DIR):
            if sid_prefix in fname:
                fpath = os.path.join(PLOTS_DIR, fname)
                try:
                    os.remove(fpath)
                    removed.append(fpath)
                except OSError:
                    pass

    return removed


# ────────────────────────────────────────────────────────────
# 单个组合的训练函数（在独立线程中运行）
# ────────────────────────────────────────────────────────────
def _run_training(config, session_id, result_box):
    """在独立线程中执行训练，将结果写入 result_box[0]"""
    try:
        from modules.rl_training import real_rl4co_training

        msg_queue       = queue.Queue()
        training_status = {}

        real_rl4co_training(
            config                 = config,
            session_id             = session_id,
            user_id                = TEST_USER_ID,
            queue                  = msg_queue,
            training_status        = training_status,
            get_background_db_func = lambda: None,   # 不使用数据库
        )

        final_status = training_status.get(session_id, {}).get('status', 'unknown')

        # 收集所有消息（用于错误分析）
        messages = []
        while not msg_queue.empty():
            try:
                raw = msg_queue.get_nowait()
                msg = json.loads(raw)
                messages.append(msg)
            except Exception:
                pass

        if final_status == 'completed':
            result_box[0] = ('PASS', None, messages)
        else:
            errors = [m.get('message', '') for m in messages if m.get('type') == 'error']
            err_str = errors[-1] if errors else f'status={final_status}'
            result_box[0] = ('FAIL', err_str, messages)

    except Exception as e:
        result_box[0] = ('FAIL', f'{type(e).__name__}: {e}', [])


def run_single_test(case):
    """
    运行单个 (problem, policy, algorithm) 组合的测试。
    测试结束后自动清理产生的 checkpoint 和图片文件。
    超时则返回 ('TIMEOUT', ...) 元组。
    """
    problem   = case['problem']
    policy    = case['policy']
    algorithm = case['algorithm']

    # 先删除可能遗留的同名 checkpoint，避免跨组合误加载
    stale_ckpt = os.path.join(CHECKPOINT_DIR, f'{problem}-{policy}.ckpt')
    if os.path.exists(stale_ckpt):
        try:
            os.remove(stale_ckpt)
        except OSError:
            pass

    config = {
        'epochs':          EPOCHS,
        'batch_size':      BATCH_SIZE,
        'train_data_size': TRAIN_DATA_SIZE,
        'val_data_size':   VAL_DATA_SIZE,
        'learning_rate':   1e-4,
        'problem':         problem,
        'model':           policy,
        'algorithm':       algorithm,
        'gpu_id':          0 if USE_GPU else None,
        **PROBLEM_EXTRA_CONFIGS.get(problem, {}),
    }

    session_id = f"test_{problem}_{policy}_{algorithm}_{int(time.time())}"
    result_box = [None]

    timeout = TIMEOUT_PER_PROBLEM.get(problem, TIMEOUT_SEC)

    t = threading.Thread(target=_run_training, args=(config, session_id, result_box), daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        # 线程仍在运行（超时），清理后返回
        cleanup_test_artifacts(problem, policy, session_id)
        return ('TIMEOUT', f'超过 {timeout}s 未完成', [])

    # 无论成功或失败，都清理本次产生的文件
    cleanup_test_artifacts(problem, policy, session_id)

    return result_box[0] if result_box[0] else ('FAIL', '线程无返回值', [])


# ────────────────────────────────────────────────────────────
# 主测试流程
# ────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  RL4CO Display — 全组合冒烟测试{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"  启动时间  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  训练轮数  : {EPOCHS} epoch / 组合")
    print(f"  批次大小  : {BATCH_SIZE}")
    print(f"  训练数据量: {TRAIN_DATA_SIZE}")
    print(f"  超时时间  : {TIMEOUT_SEC}s / 组合")
    print(f"  使用设备  : {'GPU 0' if USE_GPU else 'CPU'}")
    if FILTER_POLICY:
        print(f"  策略过滤  : 只测试 {FILTER_POLICY}")
    if FILTER_PROBLEM:
        print(f"  问题过滤  : 只测试 {FILTER_PROBLEM}")
    if SKIP_SLOW:
        print(f"  跳过 FFSP : 是")

    # 构建测试用例
    try:
        cases = build_test_cases()
    except ImportError as e:
        print(f"\n{RED}无法导入项目模块: {e}{RESET}")
        print("请确保在项目根目录（包含 modules/ 的目录）中运行此脚本。")
        sys.exit(1)

    if not cases:
        print(f"\n{YELLOW}没有符合条件的测试用例（检查 FILTER_* 设置）{RESET}")
        sys.exit(0)

    print(f"\n  共 {BOLD}{len(cases)}{RESET} 个测试用例")
    print(f"{CYAN}{'─'*70}{RESET}\n")

    # 按问题类型分组展示
    problem_order = ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw',
                     'pdp', 'op', 'pctsp', 'spctsp', 'ffsp']
    cases_sorted = sorted(cases, key=lambda c: (
        problem_order.index(c['problem']) if c['problem'] in problem_order else 99,
        c['policy'],
        c['algorithm'],
    ))

    # ── 运行测试 ──────────────────────────────────────────────
    results = []
    pass_count = fail_count = timeout_count = 0

    for idx, case in enumerate(cases_sorted, 1):
        problem   = case['problem']
        policy    = case['policy']
        algorithm = case['algorithm']

        label = f"{problem.upper():<8} + {policy:<10} + {algorithm:<10}"
        print(f"  [{idx:02d}/{len(cases_sorted)}] {label}", end='', flush=True)

        t0     = time.time()
        status, err_msg, messages = run_single_test(case)
        elapsed = time.time() - t0

        # 根据结果打印状态
        if status == 'PASS':
            print(f"  {GREEN}✓ PASS{RESET}  ({elapsed:.1f}s)")
            pass_count += 1
        elif status == 'TIMEOUT':
            print(f"  {YELLOW}⏱ TIMEOUT{RESET}  ({elapsed:.1f}s)")
            timeout_count += 1
        else:
            short_err = (err_msg or '未知错误')[:80]
            print(f"  {RED}✗ FAIL{RESET}  ({elapsed:.1f}s)  — {short_err}")
            fail_count += 1

        results.append({
            'idx':       idx,
            'problem':   problem,
            'policy':    policy,
            'algorithm': algorithm,
            'status':    status,
            'error':     err_msg,
            'elapsed':   elapsed,
        })

    # ── 最终报告 ──────────────────────────────────────────────
    total     = len(results)
    end_time  = datetime.now()

    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}  测试完成报告{RESET}")
    print(f"{CYAN}{'='*70}{RESET}")
    print(f"  总计: {total}  {GREEN}通过: {pass_count}{RESET}  "
          f"{RED}失败: {fail_count}{RESET}  {YELLOW}超时: {timeout_count}{RESET}")
    print(f"  结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 失败/超时明细 ─────────────────────────────────────────
    failed = [r for r in results if r['status'] != 'PASS']
    if failed:
        print(f"\n{BOLD}{'─'*70}{RESET}")
        print(f"{BOLD}{RED}  失败 / 超时 明细{RESET}")
        print(f"{'─'*70}")
        for r in failed:
            tag = f"{YELLOW}TIMEOUT{RESET}" if r['status'] == 'TIMEOUT' else f"{RED}FAIL{RESET}"
            print(f"\n  [{r['idx']:02d}] {r['problem'].upper()} + {r['policy']} + {r['algorithm']}  →  {tag}")
            if r['error']:
                # 打印错误，限制行数
                lines = r['error'].splitlines()
                for line in lines[:10]:
                    print(f"       {line}")
                if len(lines) > 10:
                    print(f"       ... (共 {len(lines)} 行，已截断)")

    # ── 按问题汇总 ────────────────────────────────────────────
    print(f"\n{BOLD}{'─'*70}{RESET}")
    print(f"{BOLD}  按问题类型汇总{RESET}")
    print(f"{'─'*70}")
    from itertools import groupby
    results_by_problem = {}
    for r in results:
        results_by_problem.setdefault(r['problem'], []).append(r)

    for prob in problem_order:
        if prob not in results_by_problem:
            continue
        rows   = results_by_problem[prob]
        passed = sum(1 for r in rows if r['status'] == 'PASS')
        total_ = len(rows)
        bar    = GREEN if passed == total_ else (YELLOW if passed > 0 else RED)
        print(f"  {prob.upper():<8}  {bar}{passed}/{total_}{RESET}")

    # ── 按策略汇总 ────────────────────────────────────────────
    print(f"\n{BOLD}{'─'*70}{RESET}")
    print(f"{BOLD}  按策略模型汇总{RESET}")
    print(f"{'─'*70}")
    results_by_policy = {}
    for r in results:
        results_by_policy.setdefault(r['policy'], []).append(r)

    for pol in ['attention', 'pomo', 'symnco', 'ptrnet', 'matnet', 'ham']:
        if pol not in results_by_policy:
            continue
        rows   = results_by_policy[pol]
        passed = sum(1 for r in rows if r['status'] == 'PASS')
        total_ = len(rows)
        bar    = GREEN if passed == total_ else (YELLOW if passed > 0 else RED)
        print(f"  {pol:<12}  {bar}{passed}/{total_}{RESET}")

    # ── 保存 JSON 报告 ────────────────────────────────────────
    report_path = os.path.join(ROOT_DIR, 'test_results.json')
    report_data = {
        'run_at':       end_time.isoformat(),
        'config': {
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
            'train_data_size': TRAIN_DATA_SIZE, 'val_data_size': VAL_DATA_SIZE,
            'timeout_sec': TIMEOUT_SEC, 'use_gpu': USE_GPU,
        },
        'summary': {'total': total, 'pass': pass_count,
                    'fail': fail_count, 'timeout': timeout_count},
        'results': results,
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\n  详细报告已保存至: {report_path}")
    print(f"{CYAN}{'='*70}{RESET}\n")

    # 有失败/超时时以非零退出码退出
    sys.exit(0 if fail_count + timeout_count == 0 else 1)


if __name__ == '__main__':
    # 处理 Ctrl+C
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}用户中断测试{RESET}")
        sys.exit(130)
