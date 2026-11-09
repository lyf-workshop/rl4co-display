"""Regression checks for additive UI adapters and training session isolation."""
import json
import threading
from datetime import datetime
from decimal import Decimal
from queue import Queue
from unittest.mock import MagicMock

import pytest
from flask import Flask, session

import app_files
import app_training


@pytest.fixture
def ui_client(monkeypatch, tmp_path):
    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY='isolated-ui-contract-test')
    app.register_blueprint(app_files.files_bp)
    app.register_blueprint(app_training.training_bp)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app_training, 'training_status', {})
    monkeypatch.setattr(app_training, 'training_contexts', {})
    monkeypatch.setattr(app_training, 'training_queues', {})
    monkeypatch.setattr(app_training, 'training_events', {})
    monkeypatch.setattr(app_training, 'training_lock', threading.Lock())
    monkeypatch.setattr(app_training, 'get_session_manager', lambda: None)
    # Existing GPU tests reload the module with a user-ID stub; isolate this fixture.
    monkeypatch.setattr(app_training, 'get_current_user_id', lambda: session.get('user_id'))
    client = app.test_client()
    with client.session_transaction() as session:
        session['user_id'] = 7
        session['username'] = 'contract-test'
    return client


def live_session(owner=7, status='running'):
    app_training.training_status['mine'] = {
        'status': status, 'epoch': 2, 'loss': 0.5, 'reward': -3.5,
        '_last_heartbeat': 1, 'best_reward': -3.2,
    }
    app_training.training_contexts['mine'] = {
        'user_id': owner, 'config': {'problem': 'tsp', 'model': 'pomo', 'epochs': 10},
        'mode': 'real', 'started_at': 1,
    }


@pytest.mark.parametrize('endpoint,method', [
    ('training_status', 'get'), ('training_progress', 'get'),
    ('pause_training', 'post'), ('resume_training', 'post'),
    ('stop_training', 'post'), ('training_heartbeat', 'post'),
])
def test_other_users_cannot_read_or_control_live_session(ui_client, endpoint, method):
    live_session(owner=8)
    response = getattr(ui_client, method)(f'/api/{endpoint}/mine')
    assert response.status_code == 403
    assert app_training.training_status['mine']['status'] == 'running'


def test_snapshot_preserves_old_keys_and_hides_internal_fields(ui_client):
    live_session()
    data = ui_client.get('/api/training_status/mine').get_json()
    assert data['success'] is True
    assert data['status']['loss'] == 0.5
    assert data['status']['total_epochs'] == 10
    assert '_last_heartbeat' not in data['status']
    assert 'user_id' not in data['context']


def test_completed_snapshot_recovers_result_without_consuming_sse(ui_client):
    live_session(status='completed')
    queue = Queue()
    event = json.dumps({'type': 'complete', 'results': {'checkpoint_path': 'saved.ckpt'}})
    queue.put(event)
    app_training.training_queues['mine'] = queue
    data = ui_client.get('/api/training_status/mine').get_json()
    assert data['context']['results']['checkpoint_path'] == 'saved.ckpt'
    assert queue.get_nowait() == event


def test_session_list_merges_db_records_and_scopes_live_state(ui_client, monkeypatch):
    live_session()
    app_training.training_status['other'] = {'status': 'running'}
    app_training.training_contexts['other'] = {
        'user_id': 8, 'config': {}, 'mode': 'real', 'started_at': 1,
    }
    manager = MagicMock()
    manager.get_user_sessions.return_value = [{
        'session_id': 'old', 'config': '{"num_loc":20}',
        'start_time': datetime(2026, 1, 1), 'best_reward': Decimal('-3.75'),
    }]
    monkeypatch.setattr(app_training, 'get_session_manager', lambda: manager)
    data = ui_client.get('/api/training_sessions').get_json()
    assert {row['session_id'] for row in data['sessions']} == {'mine', 'old'}
    old = next(row for row in data['sessions'] if row['session_id'] == 'old')
    assert old['config']['num_loc'] == 20
    assert old['best_reward'] == -3.75
    manager.get_user_sessions.assert_called_once_with(7, limit=100)


def test_dataset_download_uses_only_logged_in_users_directory(ui_client, tmp_path):
    own = tmp_path / 'datasets' / 'user_7'
    other = tmp_path / 'datasets' / 'user_8'
    own.mkdir(parents=True)
    other.mkdir(parents=True)
    (own / 'owned.json').write_text('{"coordinates":[[0,1]]}', encoding='utf-8')
    (other / 'private.json').write_text('{"coordinates":[[1,0]]}', encoding='utf-8')
    response = ui_client.get('/api/download_dataset/owned')
    assert response.status_code == 200
    assert response.get_json()['coordinates'] == [[0, 1]]
    assert 'attachment' in response.headers['Content-Disposition']
    response.close()
    assert ui_client.get('/api/download_dataset/private').status_code == 404
    assert ui_client.get('/api/download_dataset/C:%5Cprivate').status_code == 400
    assert ui_client.get('/api/download_dataset/..%5Cprivate').status_code == 400


def test_all_files_does_not_drop_multiple_artifacts_of_one_type(ui_client, monkeypatch):
    files = MagicMock()
    sessions = MagicMock()
    sessions.get_user_sessions.return_value = [{
        'session_id': 'mine', 'model_type': 'pomo', 'problem_type': 'tsp',
        'status': 'completed', 'start_time': datetime(2026, 1, 1),
    }]
    cursor = files.db.cursor.return_value
    cursor.fetchall.return_value = [{
        'id': i, 'file_name': f'comparison_{i}.png', 'file_type': 'plot',
        'file_size': 10, 'create_time': datetime(2026, 1, 1),
    } for i in (1, 2, 3)]
    files.get_user_storage_stats.return_value = {'total_mb': 0.1, 'total_files': 3}
    monkeypatch.setattr(app_files, 'get_file_manager', lambda: files)
    monkeypatch.setattr(app_files, 'get_session_manager', lambda: sessions)
    data = ui_client.get('/api/list_files').get_json()['sessions'][0]
    assert len(data['all_files']) == data['file_count'] == 3
    assert data['files']['comparison']['id'] == 1  # original API shape remains
