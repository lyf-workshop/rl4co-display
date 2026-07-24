"""
云端 LLM 服务商代理模块
支持任意兼容 OpenAI API 格式的服务商（DeepSeek、通义千问、Moonshot 等）

设计原则：
  - API Key 存储在 config/config.py（已加入 .gitignore），永远不暴露给前端
  - 流式响应：将 OpenAI SSE 格式规范化为与 Ollama 相同的 NDJSON 格式，
    前端 handleStreamResponse 无需修改
  - 前端只传 provider_id + model，后端负责注入 Authorization 头
"""
import json
import logging
import os
import urllib.request
import urllib.error

from flask import Blueprint, Response, request, jsonify, stream_with_context
from auth_module import login_required

llm_bp = Blueprint('llm', __name__)
logger = logging.getLogger('rl4co_display')

MAX_REQUEST_SIZE = 512_000   # 512 KB，防止超大 context 滥用
_READ_TIMEOUT    = 120       # 秒

# ──────────────────────────────────────────────────────────────────────────────
# 启动时加载服务商配置（静态，不随请求变化）
# ──────────────────────────────────────────────────────────────────────────────

def _load_provider_source():
    provider_file = os.environ.get('LLM_PROVIDERS_FILE', '').strip()
    if provider_file:
        try:
            with open(provider_file, 'r', encoding='utf-8') as file:
                payload = json.load(file)
            if isinstance(payload, dict):
                return payload.get('providers', [])
            if isinstance(payload, list):
                return payload
            logger.error('[LLM proxy] Provider file must contain a list or providers object')
            return []
        except (OSError, ValueError) as error:
            logger.error('[LLM proxy] Failed to load provider file %s: %s', provider_file, error)
            return []

    try:
        from config.config import Config
        return getattr(Config, 'LLM_PROVIDERS', [])
    except (ImportError, AttributeError):
        return []


def _load_providers():
    providers = {}
    for raw_provider in _load_provider_source():
        if not isinstance(raw_provider, dict):
            continue

        provider = dict(raw_provider)
        provider_id = str(provider.get('id') or '').strip()
        base_url = str(provider.get('base_url') or '').strip()
        api_key = str(provider.get('api_key') or '').strip()
        models = [
            str(model).strip()
            for model in provider.get('models', [])
            if str(model).strip()
        ]
        if not provider_id or not base_url or not api_key or not models:
            continue

        provider.update({
            'id': provider_id,
            'name': str(provider.get('name') or provider_id),
            'base_url': base_url,
            'api_key': api_key,
            'models': models,
            'default_model': provider.get('default_model') or models[0],
        })
        providers[provider_id] = provider

    return providers

_PROVIDERS = _load_providers()


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/llm/providers  —— 返回可用服务商列表（不含 API Key）
# ──────────────────────────────────────────────────────────────────────────────

@llm_bp.route('/api/llm/providers')
@login_required
def llm_providers():
    """前端用于构建来源下拉框。只返回 id / name / models，不含密钥。"""
    result = [
        {
            'id': p['id'],
            'name': p['name'],
            'models': p.get('models', []),
            'default_model': p.get('default_model') or (p.get('models') or [''])[0],
        }
        for p in _PROVIDERS.values()
    ]
    return jsonify({'providers': result})


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/llm/chat  —— 代理请求到上游，流式输出规范化为 Ollama NDJSON
# ──────────────────────────────────────────────────────────────────────────────

@llm_bp.route('/api/llm/chat', methods=['POST'])
@login_required
def llm_chat():
    raw_body = request.get_data()
    if len(raw_body) > MAX_REQUEST_SIZE:
        return jsonify({'error': '请求体过大（超过 512 KB）'}), 413

    try:
        payload = json.loads(raw_body)
    except Exception:
        return jsonify({'error': '请求体不是有效的 JSON'}), 400

    provider_id = payload.get('provider_id', '')
    if provider_id not in _PROVIDERS:
        return jsonify({'error': f'未知或未配置的服务商: {provider_id}'}), 400

    provider = _PROVIDERS[provider_id]
    model    = payload.get('model') or provider.get('default_model', '')
    messages = payload.get('messages', [])
    stream   = payload.get('stream', True)

    upstream_body = json.dumps({
        'model':    model,
        'messages': messages,
        'stream':   stream,
    }).encode('utf-8')

    upstream_req = urllib.request.Request(
        f"{provider['base_url'].rstrip('/')}/chat/completions",
        data=upstream_body,
        headers={
            'Content-Type':  'application/json',
            'Authorization': f"Bearer {provider['api_key']}",
        },
        method='POST',
    )

    # ── 流式 ──────────────────────────────────────────────────────────────────
    if stream:
        def _generate():
            try:
                with urllib.request.urlopen(upstream_req, timeout=_READ_TIMEOUT) as resp:
                    for raw_line in resp:
                        line = raw_line.decode('utf-8').strip()
                        if not line or not line.startswith('data:'):
                            continue
                        data_str = line[5:].strip()
                        if data_str == '[DONE]':
                            yield json.dumps({'done': True}).encode('utf-8') + b'\n'
                            return
                        try:
                            chunk   = json.loads(data_str)
                            choice  = (chunk.get('choices') or [{}])[0]
                            content = choice.get('delta', {}).get('content', '')
                            done    = choice.get('finish_reason') is not None
                            yield json.dumps({
                                'message': {'role': 'assistant', 'content': content},
                                'done': done,
                            }).encode('utf-8') + b'\n'
                        except Exception:
                            continue

            except urllib.error.HTTPError as e:
                body = e.read().decode('utf-8', errors='replace')
                logger.warning('[LLM代理] HTTP %s: %s', e.code, body[:200])
                yield json.dumps({'error': f'服务商返回错误 {e.code}', 'done': True}).encode('utf-8') + b'\n'
            except urllib.error.URLError as e:
                reason = getattr(e, 'reason', str(e))
                logger.warning('[LLM代理] 连接失败: %s', reason)
                yield json.dumps({'error': str(reason), 'done': True}).encode('utf-8') + b'\n'
            except Exception as e:
                logger.warning('[LLM代理] 未知异常: %s', e)
                yield json.dumps({'error': str(e), 'done': True}).encode('utf-8') + b'\n'

        return Response(
            stream_with_context(_generate()),
            status=200,
            mimetype='application/x-ndjson',
            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
        )

    # ── 非流式 ────────────────────────────────────────────────────────────────
    try:
        with urllib.request.urlopen(upstream_req, timeout=_READ_TIMEOUT) as resp:
            data    = json.loads(resp.read())
        content = (data.get('choices') or [{}])[0].get('message', {}).get('content', '')
        return jsonify({'message': {'role': 'assistant', 'content': content}, 'done': True})
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        return jsonify({'error': f'服务商返回错误 {e.code}: {body[:200]}'}), 502
    except urllib.error.URLError as e:
        return jsonify({'error': str(getattr(e, 'reason', e))}), 503
