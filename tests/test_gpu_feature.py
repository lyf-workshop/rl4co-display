"""
GPU 选择功能测试
测试目标：在无真实 GPU 的环境下验证完整流程：
  1. /api/gpu_status 返回 Mock 数据且格式正确
  2. gpu_id 参数链路：前端 → API → config → BaseTrainer
  3. gpu_allocations 表的占用/释放逻辑（使用内存数据库模拟）
  4. BaseTrainer 在不同 gpu_id 输入下的设备降级行为
"""
import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ============================================================
# 1. /api/gpu_status Mock 数据格式测试
# ============================================================

class TestGpuStatusEndpoint(unittest.TestCase):
    """测试 /api/gpu_status 接口返回格式"""

    def setUp(self):
        """创建 Flask 测试客户端，强制使用 Mock 模式"""
        # 在导入 app 之前，patch 掉 pynvml 让其不可用
        self._patcher = patch.dict('sys.modules', {'pynvml': None})
        self._patcher.start()

        # 重新导入 app_gpu 以使 patch 生效
        import importlib
        import auth_module
        import app_gpu
        # 将 login_required 替换为透传装饰器，使测试无需真实 session
        auth_module.login_required = lambda f: f
        importlib.reload(app_gpu)
        app_gpu.PYNVML_AVAILABLE = False
        app_gpu.TORCH_CUDA_AVAILABLE = False  # 同时禁用 torch.cuda 兜底，强制 Mock

        from flask import Flask
        self.flask_app = Flask(__name__)
        self.flask_app.config['TESTING'] = True
        self.flask_app.register_blueprint(app_gpu.gpu_bp)
        self.client = self.flask_app.test_client()

    def tearDown(self):
        self._patcher.stop()

    def test_status_returns_200(self):
        """接口正常返回 200"""
        resp = self.client.get('/api/gpu_status')
        self.assertEqual(resp.status_code, 200)

    def test_status_json_structure(self):
        """返回 JSON 包含必要字段"""
        resp = self.client.get('/api/gpu_status')
        data = json.loads(resp.data)

        self.assertIn('success', data)
        self.assertTrue(data['success'])
        self.assertIn('gpus', data)
        self.assertIn('is_mock', data)
        self.assertIn('timestamp', data)

    def test_mock_mode_flag(self):
        """无真实 GPU 时 is_mock 应为 True"""
        resp = self.client.get('/api/gpu_status')
        data = json.loads(resp.data)
        self.assertTrue(data['is_mock'])

    def test_gpu_list_not_empty(self):
        """Mock 模式下 GPU 列表不为空"""
        resp = self.client.get('/api/gpu_status')
        data = json.loads(resp.data)
        self.assertGreater(len(data['gpus']), 0)

    def test_each_gpu_has_required_fields(self):
        """每块 GPU 的必要字段都存在"""
        resp = self.client.get('/api/gpu_status')
        data = json.loads(resp.data)
        required = {'id', 'name', 'memory_total_mb', 'memory_used_mb',
                    'memory_pct', 'utilization_pct', 'status', 'sessions'}
        for gpu in data['gpus']:
            for field in required:
                self.assertIn(field, gpu, f'缺少字段: {field}')

    def test_gpu_status_values(self):
        """status 字段只能是合法值"""
        resp = self.client.get('/api/gpu_status')
        data = json.loads(resp.data)
        valid_statuses = {'idle', 'busy', 'occupied'}
        for gpu in data['gpus']:
            self.assertIn(gpu['status'], valid_statuses,
                          f"非法 status: {gpu['status']}")

    def test_memory_pct_in_range(self):
        """显存占用百分比在 0~100 之间"""
        resp = self.client.get('/api/gpu_status')
        data = json.loads(resp.data)
        for gpu in data['gpus']:
            self.assertGreaterEqual(gpu['memory_pct'], 0)
            self.assertLessEqual(gpu['memory_pct'], 100)


# ============================================================
# 2. gpu_id 参数链路测试（不依赖数据库）
# 说明：app_training 从 auth_module 导入 get_session_manager，
#      该函数由 app.py 运行时注入，测试时需提前 mock。
# ============================================================

def _make_mock_auth_module():
    """
    补全 auth_module 在运行时由 app.py 注入的函数，
    并把 login_required 替换为透传装饰器。
    必须在 reload(app_training) 之前调用，
    这样 reload 时 `from auth_module import login_required` 会拿到透传版本。
    """
    import auth_module as real_auth
    # 透传装饰器：直接返回原函数，不做 session 检查
    real_auth.login_required = lambda f: f
    # 补全运行时注入的函数
    real_auth.get_session_manager = MagicMock(return_value=None)
    real_auth.get_file_manager = MagicMock(return_value=None)
    real_auth.get_db = MagicMock(return_value=None)
    real_auth.get_current_user_id = MagicMock(return_value=1)
    return real_auth


def _make_training_app(fake_training_fn):
    """
    创建一个最小 Flask 测试 app，注入 mock 训练函数。
    返回 (flask_app, client)
    """
    # 先补全 auth_module（含 login_required 透传），再 reload app_training
    # reload 时 `from auth_module import login_required` 会绑定透传版本
    _make_mock_auth_module()

    import importlib
    import app_training as at
    importlib.reload(at)

    status_dict = {}
    queues_dict = {}
    at.init_training_globals(
        status_dict, queues_dict,
        False,
        None,
        fake_training_fn,
        None,
    )

    from flask import Flask
    flask_app = Flask(__name__)
    flask_app.config['TESTING'] = True
    flask_app.config['SECRET_KEY'] = 'test'

    with patch('app_training._allocate_gpu', return_value=None), \
         patch('app_training._release_gpu', return_value=None), \
         patch('modules.compatibility.validate_combination',
               return_value=(True, 'ok', 'success')):
        flask_app.register_blueprint(at.training_bp)

    return flask_app, flask_app.test_client()


class TestGpuIdPropagation(unittest.TestCase):
    """测试 gpu_id 从 API 请求到响应的传递"""

    def _post_training(self, client, payload):
        """封装 POST 请求，统一 patch _allocate_gpu/_release_gpu"""
        with patch('app_training._allocate_gpu', return_value=None), \
             patch('app_training._release_gpu', return_value=None), \
             patch('modules.compatibility.validate_combination',
                   return_value=(True, 'ok', 'success')):
            return client.post(
                '/api/start_training',
                data=json.dumps(payload),
                content_type='application/json',
            )

    def test_gpu_id_passed_in_config(self):
        """POST /api/start_training 携带 gpu_id=2，响应 gpu_id 应为 2"""
        _, client = _make_training_app(lambda *a, **kw: None)
        payload = {
            'problem': 'tsp', 'model': 'attention',
            'algorithm': 'reinforce', 'gpu_id': 2,
            'epochs': 1, 'batch_size': 64,
            'learning_rate': 0.0001, 'num_loc': 10,
        }
        resp = self._post_training(client, payload)
        data = json.loads(resp.data)
        self.assertTrue(data.get('success'), f"启动失败: {data}")
        self.assertEqual(data.get('gpu_id'), 2, "响应中 gpu_id 应为 2")

    def test_no_gpu_id_uses_cpu(self):
        """不传 gpu_id 时响应 gpu_id 应为 null"""
        _, client = _make_training_app(lambda *a, **kw: None)
        payload = {
            'problem': 'tsp', 'model': 'attention',
            'algorithm': 'reinforce', 'epochs': 1,
            'batch_size': 64, 'learning_rate': 0.0001, 'num_loc': 10,
        }
        resp = self._post_training(client, payload)
        data = json.loads(resp.data)
        self.assertTrue(data.get('success'), f"启动失败: {data}")
        self.assertIsNone(data.get('gpu_id'), "未传 gpu_id 时应为 null")

    def test_invalid_gpu_id_handled(self):
        """传非法 gpu_id='abc' 时应优雅处理，不返回 500"""
        _, client = _make_training_app(lambda *a, **kw: None)
        payload = {
            'problem': 'tsp', 'model': 'attention',
            'algorithm': 'reinforce', 'epochs': 1,
            'batch_size': 64, 'learning_rate': 0.0001,
            'num_loc': 10, 'gpu_id': 'abc',
        }
        resp = self._post_training(client, payload)
        self.assertNotEqual(resp.status_code, 500,
                            "非法 gpu_id 不应导致服务器 500 错误")


# ============================================================
# 3. gpu_allocations 数据库逻辑测试（内存 SQLite 模拟）
# ============================================================

class TestGpuAllocationDb(unittest.TestCase):
    """用 SQLite 内存数据库模拟 GPU 占用/释放逻辑"""

    def setUp(self):
        """创建内存 SQLite，建 gpu_allocations 表"""
        import sqlite3
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT
            )
        """)
        self.conn.execute("INSERT INTO users VALUES (1, 'testuser')")
        self.conn.execute("""
            CREATE TABLE gpu_allocations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gpu_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                allocated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                released_at TIMESTAMP,
                status TEXT DEFAULT 'allocated'
            )
        """)
        self.conn.commit()

    def tearDown(self):
        self.conn.close()

    def _allocate(self, gpu_id, session_id, user_id=1):
        self.conn.execute("""
            INSERT INTO gpu_allocations (gpu_id, session_id, user_id, status)
            VALUES (?, ?, ?, 'allocated')
        """, (gpu_id, session_id, user_id))
        self.conn.commit()

    def _release(self, session_id):
        from datetime import datetime
        self.conn.execute("""
            UPDATE gpu_allocations
            SET status = 'released', released_at = ?
            WHERE session_id = ? AND status = 'allocated'
        """, (datetime.now().isoformat(), session_id))
        self.conn.commit()

    def _get_active(self, gpu_id):
        cur = self.conn.execute("""
            SELECT * FROM gpu_allocations
            WHERE gpu_id = ? AND status = 'allocated'
        """, (gpu_id,))
        return cur.fetchall()

    def test_allocate_creates_record(self):
        """占用 GPU 后数据库中应有对应记录"""
        self._allocate(0, 'sess-001')
        rows = self._get_active(0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['session_id'], 'sess-001')

    def test_release_marks_released(self):
        """释放 GPU 后记录 status 应变为 released"""
        self._allocate(0, 'sess-002')
        self._release('sess-002')
        rows = self._get_active(0)
        self.assertEqual(len(rows), 0, "释放后不应有 allocated 记录")

        cur = self.conn.execute(
            "SELECT status FROM gpu_allocations WHERE session_id = ?",
            ('sess-002',)
        )
        row = cur.fetchone()
        self.assertEqual(row['status'], 'released')

    def test_multiple_sessions_same_gpu(self):
        """同一 GPU 可有多个 session 占用（软约束）"""
        self._allocate(1, 'sess-A')
        self._allocate(1, 'sess-B')
        rows = self._get_active(1)
        self.assertEqual(len(rows), 2)

    def test_release_only_affects_target_session(self):
        """释放 sess-A 不应影响 sess-B"""
        self._allocate(1, 'sess-A')
        self._allocate(1, 'sess-B')
        self._release('sess-A')
        rows = self._get_active(1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['session_id'], 'sess-B')

    def test_double_release_is_idempotent(self):
        """重复释放同一 session 不报错，影响行数为 0"""
        self._allocate(2, 'sess-C')
        self._release('sess-C')
        # 再次释放
        from datetime import datetime
        self.conn.execute("""
            UPDATE gpu_allocations
            SET status = 'released', released_at = ?
            WHERE session_id = ? AND status = 'allocated'
        """, (datetime.now().isoformat(), 'sess-C'))
        self.conn.commit()
        rows = self._get_active(2)
        self.assertEqual(len(rows), 0)

    def test_different_gpus_independent(self):
        """不同 GPU 的占用互不干扰"""
        self._allocate(0, 'sess-X')
        self._allocate(1, 'sess-Y')
        self._release('sess-X')

        self.assertEqual(len(self._get_active(0)), 0)
        self.assertEqual(len(self._get_active(1)), 1)


# ============================================================
# 4. BaseTrainer GPU 设备选择逻辑测试
# ============================================================

import importlib
_torch_available = importlib.util.find_spec('torch') is not None

@unittest.skipUnless(_torch_available, 'torch 未安装，跳过 BaseTrainer 设备选择测试')
class TestBaseTrainerDeviceSelection(unittest.TestCase):
    """测试 BaseTrainer 在不同 gpu_id 配置下的设备选择"""

    def _make_trainer(self, config_overrides=None):
        """构造一个最简 BaseTrainer（不真正训练）"""
        from modules.rl_training.base_trainer import BaseTrainer

        base_config = {
            'problem': 'tsp',
            'model': 'attention',
            'algorithm': 'reinforce',
            'epochs': 1,
            'batch_size': 64,
            'learning_rate': 1e-4,
        }
        if config_overrides:
            base_config.update(config_overrides)

        mock_queue = MagicMock()
        mock_queue.put = MagicMock()

        # get_background_db_func 返回 None（无数据库）
        trainer = BaseTrainer(
            config=base_config,
            session_id='test-session',
            user_id=1,
            queue=mock_queue,
            training_status={},
            get_background_db_func=lambda: None
        )
        return trainer

    def test_no_gpu_id_uses_cpu_when_no_cuda(self):
        """无 CUDA 环境下，不传 gpu_id 应使用 CPU"""
        with patch('torch.cuda.is_available', return_value=False):
            trainer = self._make_trainer()
        self.assertEqual(trainer.accelerator, 'cpu')
        self.assertEqual(trainer.devices, 'auto')

    def test_gpu_id_none_uses_cpu(self):
        """不传 gpu_id 时（用户选择 CPU），即使 CUDA 可用也应使用 CPU"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=4):
            trainer = self._make_trainer()
        self.assertEqual(trainer.accelerator, 'cpu')
        self.assertEqual(trainer.devices, 'auto')

    def test_valid_gpu_id_sets_device_list(self):
        """有 CUDA 且 gpu_id 有效时，devices 应为 [gpu_id]"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=4):
            trainer = self._make_trainer({'gpu_id': 2})
        self.assertEqual(trainer.accelerator, 'gpu')
        self.assertEqual(trainer.devices, [2])

    def test_out_of_range_gpu_id_falls_back_to_0(self):
        """gpu_id 超出 GPU 数量时应回退到 cuda:0"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2):
            trainer = self._make_trainer({'gpu_id': 99})
        self.assertEqual(trainer.accelerator, 'gpu')
        self.assertEqual(trainer.devices, [0])

    def test_gpu_id_zero_is_valid(self):
        """gpu_id=0 是合法的最小值"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=4):
            trainer = self._make_trainer({'gpu_id': 0})
        self.assertEqual(trainer.devices, [0])

    def test_invalid_string_gpu_id_falls_back_to_cpu(self):
        """传字符串类型 gpu_id 应不崩溃，ValueError 回退到 CPU"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=4):
            trainer = self._make_trainer({'gpu_id': 'bad_value'})
        self.assertEqual(trainer.accelerator, 'cpu')
        self.assertEqual(trainer.devices, 'auto')

    def test_no_cuda_with_gpu_id_uses_cpu(self):
        """无 CUDA 时即使传了 gpu_id 也应使用 CPU"""
        with patch('torch.cuda.is_available', return_value=False):
            trainer = self._make_trainer({'gpu_id': 1})
        self.assertEqual(trainer.accelerator, 'cpu')


if __name__ == '__main__':
    unittest.main(verbosity=2)
