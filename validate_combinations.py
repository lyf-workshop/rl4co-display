#!/usr/bin/env python3
"""
验证脚本：测试所有有效的（问题类型, 策略模型, RL算法）组合是否可以成功完成训练。

运行方式:
    python validate_combinations.py                          # 测试所有组合
    python validate_combinations.py --problem tsp            # 只测试 TSP 相关
    python validate_combinations.py --policy attention       # 只测试注意力模型
    python validate_combinations.py --algorithm reinforce    # 只测试 REINFORCE
    python validate_combinations.py --dry-run               # 只列出组合，不实际运行
    python validate_combinations.py --output-dir ./results  # 指定结果保存目录

输出:
    - 控制台：实时进度 + 最终摘要表格
    - validation_results_<timestamp>.json
    - validation_results_<timestamp>.csv
"""

import sys
import os
import argparse
import json
import csv
import time
import traceback
import datetime
import tempfile
import io

# ── Windows GBK 终端兼容：强制 stdout/stderr 使用 UTF-8 ───────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() in ('gbk', 'cp936', 'gb2312', 'gb18030'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── 路径设置：将项目根目录加入 sys.path ────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── 检查核心依赖 ───────────────────────────────────────────────────────────────
try:
    import torch
except ImportError:
    print("❌ PyTorch 未安装，请先安装: pip install torch")
    sys.exit(1)

try:
    from rl4co.utils.trainer import RL4COTrainer
    RL4CO_AVAILABLE = True
except ImportError:
    print("❌ RL4CO 未安装，请先安装: pip install rl4co")
    sys.exit(1)

# ── 检查可选依赖 ───────────────────────────────────────────────────────────────
try:
    import torch_geometric  # noqa: F401
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

# ── 导入项目模块（不依赖 auth_module / MySQL） ─────────────────────────────────
from modules.compatibility import (
    POLICY_PROBLEM_COMPATIBILITY,
    POLICY_ALGORITHM_COMPATIBILITY,
    validate_combination,
)
from modules.policies import get_policy_class
from modules.algorithms import get_algorithm_class


# ═══════════════════════════════════════════════════════════════════════════════
# FFSP 环境适配器（内联定义，避免导入 ffsp_trainer.py → auth_module）
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_index_tables():
    """为 RL4CO 的 IndexTables 类打补丁，添加缺失的 augment_machine_tables 方法。"""
    try:
        from rl4co.envs.scheduling.ffsp.env import IndexTables
        if hasattr(IndexTables, 'augment_machine_tables'):
            return
        def augment_machine_tables(self, num_starts):
            if hasattr(self, 'bs') and self.bs is not None:
                self.bs = self.bs * num_starts
            else:
                self.set_bs(num_starts)
        IndexTables.augment_machine_tables = augment_machine_tables
    except Exception:
        pass


def _patch_ffsp_decoder():
    """兼容补丁：RL4CO 0.6.0 中 MatNetFFSPDecoder 接受 out_bias_pointer_attn，
    而 MatNetPolicy 传入的是 out_bias，导致 TypeError。"""
    try:
        from rl4co.models.zoo.matnet.decoder import MatNetFFSPDecoder
        if getattr(MatNetFFSPDecoder, '_patched_out_bias', False):
            return
        _orig = MatNetFFSPDecoder.__init__
        def _patched(self, *args, **kwargs):
            if 'out_bias' in kwargs and 'out_bias_pointer_attn' not in kwargs:
                kwargs['out_bias_pointer_attn'] = kwargs.pop('out_bias')
            _orig(self, *args, **kwargs)
        MatNetFFSPDecoder.__init__ = _patched
        MatNetFFSPDecoder._patched_out_bias = True
    except Exception:
        pass


class _FFSPEnvAdapter:
    """
    轻量级 FFSP 环境适配器：在 reset() 返回的 TensorDict 中自动补充
    cost_matrix = run_time，以满足 MatNet 的初始化嵌入要求。
    """

    def __init__(self, base_env):
        object.__setattr__(self, '_base', base_env)
        for attr in ('name', 'num_stage', 'num_machine', 'num_job',
                     'num_machine_total', 'flatten_stages', 'generator',
                     'observation_spec', 'action_spec', 'reward_spec', 'done_spec',
                     'dataset_cls', 'data_dir', 'train_file', 'val_file', 'test_file'):
            object.__setattr__(self, attr, getattr(base_env, attr, None))

    @property
    def _env(self):
        return object.__getattribute__(self, '_base')

    @property
    def tables(self):
        return getattr(self._env, 'tables', None)

    @property
    def step_cnt(self):
        return getattr(self._env, 'step_cnt', None)

    def reset(self, *args, **kwargs):
        td = self._env.reset(*args, **kwargs)
        if 'run_time' in td.keys() and 'cost_matrix' not in td.keys():
            td['cost_matrix'] = td['run_time']
        return td

    def step(self, *args, **kwargs):
        return self._env.step(*args, **kwargs)

    def get_reward(self, *args, **kwargs):
        return self._env.get_reward(*args, **kwargs)

    def dataset(self, batch_size=[], phase='train', filename=None):
        return self._env.dataset(batch_size, phase, filename)

    def get_num_starts(self, td):
        return self._env.get_num_starts(td)

    def select_start_nodes(self, td, num_starts):
        return self._env.select_start_nodes(td, num_starts)

    def pre_step(self, td):
        if hasattr(self._env, 'pre_step'):
            return self._env.pre_step(td)
        return td

    def load_data(self, *args, **kwargs):
        return self._env.load_data(*args, **kwargs)

    def __repr__(self):
        return f"_FFSPEnvAdapter({self._env!r})"


# ═══════════════════════════════════════════════════════════════════════════════
# 最小训练配置（小模型 + 极少数据，速度优先）
# ═══════════════════════════════════════════════════════════════════════════════

NUM_LOC = 10          # 所有问题统一使用 10 个节点
BATCH_SIZE = 32       # 小批量
TRAIN_DATA = 320      # 极小训练集（10 个批次）
VAL_DATA = 64         # 极小验证集

BASE_CONFIG = {
    # 模型超参（小）
    'embed_dim': 64,
    'num_encoder_layers': 2,
    'num_heads': 4,
    # 训练规模
    'batch_size': BATCH_SIZE,
    'train_data_size': TRAIN_DATA,
    'val_data_size': VAL_DATA,
    'learning_rate': 1e-4,
    # REINFORCE baseline
    'baseline': 'rollout',
    # SymNCO 参数
    'num_augment': 2,
    'num_starts': 0,
    'symnco_alpha': 0.2,
    'symnco_beta': 1.0,
    # DeepACO 参数
    'n_ants': 5,
    'n_iterations_train': 1,
    'n_iterations_test': 2,
    # POMO 参数
    'num_starts': 0,
    # MDAM 参数
    'num_paths': 2,
    # MatNet
    'use_graph_context': False,
    'flatten_stages': True,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 环境工厂
# ═══════════════════════════════════════════════════════════════════════════════

def create_env(problem: str):
    """根据问题类型创建 RL4CO 环境实例。"""
    p = problem.lower()

    if p == 'tsp':
        from rl4co.envs import TSPEnv
        return TSPEnv(generator_params={'num_loc': NUM_LOC})

    elif p == 'atsp':
        from rl4co.envs.routing import ATSPEnv
        return ATSPEnv(generator_params={'num_loc': NUM_LOC})

    elif p == 'mtsp':
        from rl4co.envs import MTSPEnv
        return MTSPEnv(generator_params={
            'num_loc': NUM_LOC,
            'min_num_agents': 2,
            'max_num_agents': 2,
        })

    elif p == 'cvrp':
        from rl4co.envs import CVRPEnv
        return CVRPEnv(generator_params={'num_loc': NUM_LOC})

    elif p == 'sdvrp':
        try:
            from rl4co.envs import SDVRPEnv
            return SDVRPEnv(generator_params={'num_loc': NUM_LOC})
        except (ImportError, AttributeError):
            from rl4co.envs import CVRPEnv
            return CVRPEnv(generator_params={'num_loc': NUM_LOC})

    elif p == 'vrptw':
        # RL4CO 0.6.0 中 VRPTW 对应 CVRPTWEnv（别名 VRPTWEnv 已移除）
        try:
            from rl4co.envs.routing import CVRPTWEnv
            return CVRPTWEnv(generator_params={'num_loc': NUM_LOC})
        except (ImportError, AttributeError):
            try:
                from rl4co.envs import VRPTWEnv
                return VRPTWEnv(generator_params={'num_loc': NUM_LOC})
            except (ImportError, AttributeError):
                from modules.envs.vrptw_env_wrapper import CVRPEnvWithTimeWindows
                return CVRPEnvWithTimeWindows(
                    generator_params={'num_loc': NUM_LOC, 'vehicle_capacity': 1.0}
                )

    elif p == 'op':
        from rl4co.envs.routing import OPEnv
        return OPEnv(generator_params={'num_loc': NUM_LOC, 'prize_type': 'dist'})

    elif p == 'pdp':
        from rl4co.envs.routing import PDPEnv
        return PDPEnv(generator_params={'num_loc': NUM_LOC})

    elif p == 'pctsp':
        from rl4co.envs.routing import PCTSPEnv
        return PCTSPEnv(generator_params={'num_loc': NUM_LOC})

    elif p == 'spctsp':
        from rl4co.envs.routing import SPCTSPEnv
        return SPCTSPEnv(generator_params={'num_loc': NUM_LOC})

    elif p == 'ffsp':
        _patch_index_tables()
        _patch_ffsp_decoder()
        from rl4co.envs.scheduling import FFSPEnv
        base = FFSPEnv(
            generator_params={
                'num_stage': 2,
                'num_machine': 2,
                'num_job': 4,
            },
            flatten_stages=True,
        )
        return _FFSPEnvAdapter(base)

    else:
        raise ValueError(f"未知问题类型: {problem}")


# ═══════════════════════════════════════════════════════════════════════════════
# 策略 / 模型工厂
# ═══════════════════════════════════════════════════════════════════════════════

def make_policy(policy_name: str, env, config: dict):
    """创建策略网络。"""
    full_cfg = {**BASE_CONFIG, **config, 'model': policy_name}
    wrapper = get_policy_class(policy_name)(full_cfg)
    return wrapper.create_policy(env)


def make_model(policy_name: str, algorithm_name: str, env, policy, config: dict):
    """创建 RL 训练模型。"""
    full_cfg = {
        **BASE_CONFIG, **config,
        'model': policy_name,
        'algorithm': algorithm_name,
    }
    bs  = full_cfg['batch_size']
    tds = full_cfg['train_data_size']
    vds = full_cfg['val_data_size']
    lr  = full_cfg.get('learning_rate', 1e-4)

    # FFSP 必须用 MatNet 模型（POMO 子类，内置 REINFORCE，不兼容外部 A2C/PPO）
    env_name = getattr(env, 'name', '')
    if env_name == 'ffsp' or (hasattr(env, '_base') and getattr(env._base, 'name', '') == 'ffsp'):
        from rl4co.models.zoo.matnet import MatNet
        return MatNet(
            env, policy,
            num_starts=2,   # MatNet 要求 num_starts > 1 才能训练
            batch_size=bs,
            train_data_size=tds,
            val_data_size=vds,
            optimizer_kwargs={'lr': lr},
        )

    if policy_name == 'symnco':
        from rl4co.models.zoo.symnco import SymNCO
        return SymNCO(
            env, policy,
            baseline='symnco',
            num_augment=full_cfg.get('num_augment', 2),
            num_starts=full_cfg.get('num_starts', 0),
            alpha=full_cfg.get('symnco_alpha', 0.2),
            beta=full_cfg.get('symnco_beta', 1.0),
            batch_size=bs,
            train_data_size=tds,
            val_data_size=vds,
            optimizer_kwargs={'lr': lr},
        )

    elif policy_name == 'mdam':
        from rl4co.models.zoo.mdam import MDAM
        return MDAM(
            env, policy,
            baseline='rollout',
            batch_size=bs,
            train_data_size=tds,
            val_data_size=vds,
            optimizer_kwargs={'lr': lr},
        )

    elif policy_name == 'deepaco':
        from rl4co.models.zoo.deepaco import DeepACO
        return DeepACO(
            env, policy,
            baseline='no',
            train_with_local_search=False,
            batch_size=bs,
            train_data_size=tds,
            val_data_size=vds,
            optimizer_kwargs={'lr': lr},
        )

    else:
        # 通用算法路径（REINFORCE / PPO / A2C）
        wrapper = get_algorithm_class(algorithm_name)(full_cfg)
        return wrapper.create_model(env, policy)


# ═══════════════════════════════════════════════════════════════════════════════
# 枚举有效组合
# ═══════════════════════════════════════════════════════════════════════════════

# 别名去重：只保留标准名称，跳过同义词
_CANONICAL = {
    'attention': 'attention',
    'am':        None,       # = attention
    'ptrnet':    'ptrnet',
    'ptr':       None,       # = ptrnet
    'pomo':      'pomo',
    'matnet':    'matnet',
    'ham':       'ham',
    'symnco':    'symnco',
    'mdam':      'mdam',
    'deepaco':   'deepaco',
}


def enumerate_combinations(filter_problem=None, filter_policy=None, filter_algorithm=None):
    """返回所有有效且去重的 (problem, policy, algorithm) 三元组列表。"""
    combos = []
    seen   = set()

    for policy_raw, problems in POLICY_PROBLEM_COMPATIBILITY.items():
        canon = _CANONICAL.get(policy_raw)
        if canon is None:
            continue  # 跳过别名

        if filter_policy and canon != filter_policy.lower():
            continue

        algos = POLICY_ALGORITHM_COMPATIBILITY.get(policy_raw, [])

        for problem in problems:
            if filter_problem and problem != filter_problem.lower():
                continue

            for algorithm in algos:
                if filter_algorithm and algorithm != filter_algorithm.lower():
                    continue

                key = (problem, canon, algorithm)
                if key in seen:
                    continue
                seen.add(key)

                # 最终确认三方都兼容
                is_valid, _, _ = validate_combination(problem, canon, algorithm)
                if not is_valid:
                    continue

                combos.append({'problem': problem, 'policy': canon, 'algorithm': algorithm})

    # 排序：按问题 → 策略 → 算法，方便阅读
    combos.sort(key=lambda c: (c['problem'], c['policy'], c['algorithm']))
    return combos


# ═══════════════════════════════════════════════════════════════════════════════
# 执行单个组合测试
# ═══════════════════════════════════════════════════════════════════════════════

def run_one(problem: str, policy: str, algorithm: str, tmpdir: str) -> dict:
    """
    对单个组合运行 1 个 epoch 的最小训练。

    返回 dict 含字段：
        status    : 'PASS' | 'FAIL' | 'SKIP'
        error     : 错误堆栈（仅 FAIL 时填写）
        duration_s: 耗时（秒）
    """
    t0 = time.time()
    result = {
        'problem': problem,
        'policy':  policy,
        'algorithm': algorithm,
        'status':  'FAIL',
        'error':   '',
        'duration_s': 0.0,
    }

    # ── DeepACO 需要 torch_geometric ─────────────────────────────────────────
    if policy == 'deepaco' and not TORCH_GEOMETRIC_AVAILABLE:
        result['status'] = 'SKIP'
        result['error']  = 'torch_geometric 未安装（DeepACO 依赖）'
        result['duration_s'] = round(time.time() - t0, 2)
        return result

    try:
        config = dict(BASE_CONFIG)
        config['num_loc'] = NUM_LOC

        # 1. 环境
        env = create_env(problem)

        # 2. 策略
        policy_obj = make_policy(policy, env, config)

        # 3. 模型
        model = make_model(policy, algorithm, env, policy_obj, config)

        # 4. Lightning Trainer（1 epoch，无日志，无检查点）
        trainer = RL4COTrainer(
            max_epochs=1,
            accelerator='cpu',
            devices='auto',
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            default_root_dir=tmpdir,  # 防止在当前目录生成 lightning_logs/
        )

        # 5. 训练
        trainer.fit(model)

        result['status'] = 'PASS'

    except Exception as e:
        result['status'] = 'FAIL'
        result['error']  = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

    result['duration_s'] = round(time.time() - t0, 2)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 输出工具
# ═══════════════════════════════════════════════════════════════════════════════

_SYM   = {'PASS': '✅', 'FAIL': '❌', 'SKIP': '⏭️ '}
_COLOR = {'PASS': '\033[92m', 'FAIL': '\033[91m', 'SKIP': '\033[93m', 'RESET': '\033[0m'}


def _c(text, status):
    """给文本着色（仅支持 ANSI 终端）。"""
    return f"{_COLOR[status]}{text}{_COLOR['RESET']}"


def print_line(idx, total, r):
    sym  = _SYM[r['status']]
    comb = f"{r['problem'].upper():<8} {r['policy']:<10} {r['algorithm']:<10}"
    dur  = f"{r['duration_s']:6.1f}s"
    err  = ''
    if r['error']:
        first_line = r['error'].splitlines()[0]
        err = f"  ← {first_line[:70]}"
    print(f"[{idx:3d}/{total}] {sym} {_c(comb, r['status'])} {dur}{err}")


def print_summary(results):
    counts = {'PASS': 0, 'FAIL': 0, 'SKIP': 0}
    total_dur = 0.0
    for r in results:
        counts[r['status']] += 1
        total_dur += r.get('duration_s', 0)

    total = len(results)
    print()
    print("═" * 72)
    print(f"  验证结果摘要  （共 {total} 个组合，总耗时 {total_dur:.1f}s）")
    print("═" * 72)
    print(f"  ✅ PASS  : {counts['PASS']:3d}  ({counts['PASS']/total*100:5.1f}%)")
    print(f"  ❌ FAIL  : {counts['FAIL']:3d}  ({counts['FAIL']/total*100:5.1f}%)")
    print(f"  ⏭️  SKIP  : {counts['SKIP']:3d}  ({counts['SKIP']/total*100:5.1f}%)")
    print("═" * 72)

    if counts['FAIL'] > 0:
        print(f"\n{'─'*72}")
        print("❌ 失败的组合")
        print(f"{'─'*72}")
        for r in results:
            if r['status'] == 'FAIL':
                short = r['error'].splitlines()[0][:100] if r['error'] else '(无错误信息)'
                label = f"{r['problem'].upper()} + {r['policy'].upper()} + {r['algorithm'].upper()}"
                print(f"  {label}")
                print(f"    └─ {short}")

    if counts['SKIP'] > 0:
        print(f"\n{'─'*72}")
        print("⏭️  跳过的组合")
        print(f"{'─'*72}")
        for r in results:
            if r['status'] == 'SKIP':
                label = f"{r['problem'].upper()} + {r['policy'].upper()} + {r['algorithm'].upper()}"
                print(f"  {label}")
                print(f"    └─ {r['error']}")


def save_results(results, output_dir='.'):
    """将结果保存为 JSON 和 CSV 文件，返回两个文件路径。"""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, f'validation_results_{ts}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f'validation_results_{ts}.csv')
    fieldnames = ['problem', 'policy', 'algorithm', 'status', 'duration_s', 'error']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            if row['error']:
                # 错误信息只保留第一行，避免 CSV 换行问题
                row['error'] = row['error'].splitlines()[0][:200]
            writer.writerow(row)

    return json_path, csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='验证所有（问题, 策略, 算法）组合能否成功训练',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--problem',    metavar='PROB',  help='只测试指定问题，如 tsp / cvrp / ffsp')
    parser.add_argument('--policy',     metavar='POL',   help='只测试指定策略，如 attention / pomo / matnet')
    parser.add_argument('--algorithm',  metavar='ALGO',  help='只测试指定算法，如 reinforce / ppo / a2c')
    parser.add_argument('--dry-run',    action='store_true', help='只列出组合，不实际运行训练')
    parser.add_argument('--output-dir', default='.',     help='结果文件保存目录（默认当前目录）')
    args = parser.parse_args()

    print("═" * 72)
    print("  RL4CO 训练组合验证脚本")
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  设备   : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"  问题规模: {NUM_LOC} 个节点，{BATCH_SIZE} batch，{TRAIN_DATA} 训练样本，1 epoch")
    print(f"  torch_geometric 可用: {TORCH_GEOMETRIC_AVAILABLE}（DeepACO 需要）")
    print("═" * 72)

    # 枚举组合
    combos = enumerate_combinations(
        filter_problem=args.problem,
        filter_policy=args.policy,
        filter_algorithm=args.algorithm,
    )
    total = len(combos)

    if total == 0:
        print("\n⚠️  未找到符合筛选条件的有效组合，请检查 --problem / --policy / --algorithm 参数。")
        return

    print(f"\n共找到 {total} 个有效组合\n")

    # ── Dry Run：只列出组合 ──────────────────────────────────────────────────
    if args.dry_run:
        print("─── Dry Run 模式：仅列出组合，不实际训练 ───")
        for i, c in enumerate(combos, 1):
            print(f"  [{i:3d}] {c['problem'].upper():<8} {c['policy']:<10} {c['algorithm']:<10}")
        print(f"\n共 {total} 个组合。")
        return

    # ── 正式运行 ─────────────────────────────────────────────────────────────
    results = []
    with tempfile.TemporaryDirectory(prefix='rl4co_val_') as tmpdir:
        for idx, c in enumerate(combos, 1):
            p, pol, alg = c['problem'], c['policy'], c['algorithm']
            print(f"[{idx:3d}/{total}] 🔄 {p.upper():<8} {pol:<10} {alg:<10} ...", end='', flush=True)

            r = run_one(p, pol, alg, tmpdir)
            results.append(r)

            sym = _SYM[r['status']]
            dur = f"{r['duration_s']:6.1f}s"
            err = ''
            if r['error']:
                first = r['error'].splitlines()[0][:60]
                err = f"  ← {first}"
            label = f"{p.upper():<8} {pol:<10} {alg:<10}"
            # 用回车覆盖"..."那一行
            print(f"\r[{idx:3d}/{total}] {sym} {_c(label, r['status'])} {dur}{err}")

    print_summary(results)

    json_path, csv_path = save_results(results, args.output_dir)
    print(f"\n📄 结果已保存至:")
    print(f"   JSON → {json_path}")
    print(f"   CSV  → {csv_path}")


if __name__ == '__main__':
    main()
