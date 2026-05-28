"""
Ollama 代理模块
将前端的 Ollama API 请求代理到本地 Ollama 服务（127.0.0.1:11434）

为什么需要代理：
  1. CSP connect-src 'self'  —— 浏览器 Content-Security-Policy 只允许同域请求，
                                 直接访问 127.0.0.1:11434 会被拦截
  2. CORS                    —— 浏览器跨域限制，从 localhost:5000 到 127.0.0.1:11434 被拒绝

通过此代理，前端统一调用 /api/ollama/...，服务端转发到 Ollama，完全绕过上述限制。
"""
import json
import logging
import urllib.request
import urllib.error

from flask import Blueprint, Response, request, jsonify, stream_with_context

ollama_bp = Blueprint('ollama', __name__)
logger = logging.getLogger('rl4co_display')

OLLAMA_BASE    = 'http://127.0.0.1:11434/api'
CONNECT_TIMEOUT = 3    # 连接超时（秒）：本地服务，3 秒足够
READ_TIMEOUT    = 120  # 读取超时（秒）：流式生成可能较慢


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/ollama/tags  →  GET http://127.0.0.1:11434/api/tags
# ──────────────────────────────────────────────────────────────────────────────

@ollama_bp.route('/api/ollama/tags')
def ollama_tags():
    """
    前端用于检查 Ollama 是否在线，并获取已安装模型列表。
    正常返回 Ollama 的 JSON；Ollama 未运行时返回 503。
    """
    try:
        req = urllib.request.Request(
            f'{OLLAMA_BASE}/tags',
            headers={'Accept': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT) as resp:
            data = resp.read()
        return Response(data, status=200, mimetype='application/json')

    except urllib.error.URLError as e:
        reason = getattr(e, 'reason', str(e))
        logger.warning(f'[Ollama代理] /tags 连接失败: {reason}')
        return jsonify({'error': 'Ollama服务不可达', 'detail': str(reason)}), 503

    except Exception as e:
        logger.warning(f'[Ollama代理] /tags 异常: {e}')
        return jsonify({'error': str(e)}), 503


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/ollama/chat  →  POST http://127.0.0.1:11434/api/chat
# ──────────────────────────────────────────────────────────────────────────────

@ollama_bp.route('/api/ollama/chat', methods=['POST'])
def ollama_chat():
    """
    代理对话请求，支持 stream: true（流式，前端实时显示）和 stream: false（一次性返回）。
    """
    raw_body = request.get_data()

    try:
        payload = json.loads(raw_body)
    except Exception:
        return jsonify({'error': '请求体不是有效的 JSON'}), 400

    is_stream = payload.get('stream', True)

    upstream_req = urllib.request.Request(
        f'{OLLAMA_BASE}/chat',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )

    # ── 流式响应 ────────────────────────────────────────────────────────────
    if is_stream:
        def _generate():
            try:
                with urllib.request.urlopen(upstream_req, timeout=READ_TIMEOUT) as resp:
                    while True:
                        chunk = resp.read(1024)
                        if not chunk:
                            break
                        yield chunk
            except urllib.error.URLError as e:
                reason = getattr(e, 'reason', str(e))
                logger.warning(f'[Ollama代理] /chat 流式读取失败: {reason}')
                yield json.dumps({'error': str(reason), 'done': True}).encode('utf-8')
            except Exception as e:
                logger.warning(f'[Ollama代理] /chat 流式异常: {e}')
                yield json.dumps({'error': str(e), 'done': True}).encode('utf-8')

        return Response(
            stream_with_context(_generate()),
            status=200,
            mimetype='application/x-ndjson',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',   # 禁止 Nginx 缓冲，确保 chunk 实时到达
            },
        )

    # ── 非流式响应 ───────────────────────────────────────────────────────────
    try:
        with urllib.request.urlopen(upstream_req, timeout=READ_TIMEOUT) as resp:
            data = resp.read()
        return Response(data, status=200, mimetype='application/json')
    except urllib.error.URLError as e:
        reason = getattr(e, 'reason', str(e))
        return jsonify({'error': str(reason)}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 503
