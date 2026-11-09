(() => {
  'use strict';
  const escape = (value) =>
    String(value ?? '').replace(
      /[&<>"']/g,
      (c) =>
        ({
          '&': '&amp;',
          '<': '&lt;',
          '>': '&gt;',
          '"': '&quot;',
          "'": '&#39;',
        })[c],
    );
  async function request(url, options = {}) {
    let response;
    try {
      response = await fetch(url, {
        credentials: 'same-origin',
        signal: AbortSignal.timeout(20000),
        ...options,
      });
    } catch (error) {
      throw new Error(
        error.name === 'TimeoutError'
          ? '请求超时，请重试。'
          : '网络连接失败，请检查连接后重试。',
      );
    }
    if (
      !(response.headers.get('content-type') || '').includes('application/json')
    )
      throw new Error(
        response.redirected
          ? '登录状态已过期，请重新登录。'
          : `服务器响应异常（HTTP ${response.status}）`,
      );
    const data = await response.json();
    if (!response.ok || data.success === false || data.error)
      throw new Error(
        data.message || data.error || `请求失败（HTTP ${response.status}）`,
      );
    return data;
  }
  const post = (url, payload) =>
    request(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload ?? {}),
    });
  function toast(message, type = 'info') {
    const node = document.createElement('div');
    node.className = `toast toast-${type}`;
    node.setAttribute('role', type === 'error' ? 'alert' : 'status');
    const text = document.createElement('span');
    text.textContent = message;
    const close = document.createElement('button');
    close.type = 'button';
    close.textContent = '×';
    close.setAttribute('aria-label', '关闭通知');
    close.onclick = () => node.remove();
    node.append(text, close);
    document.getElementById('toast-container').append(node);
    setTimeout(() => node.remove(), type === 'error' ? 10000 : 5500);
  }
  const busy = (button, value) => {
    button.disabled = value;
    button.setAttribute('aria-busy', String(value));
  };
  const focusable = (root) =>
    [
      ...root.querySelectorAll(
        'button, a[href], input, select, textarea, summary, [tabindex="0"]',
      ),
    ].filter((el) => !el.disabled && el.getClientRects().length);
  let layers = [];
  function openLayer(root, close) {
    const layer = { root, close, trigger: document.activeElement };
    layers.push(layer);
    root.hidden = false;
    document.body.style.overflow = 'hidden';
    const dialog = root.querySelector('[role="dialog"]') || root;
    (focusable(dialog)[0] || dialog).focus();
    return () => {
      root.hidden = true;
      layers = layers.filter((item) => item !== layer);
      if (!layers.length) document.body.style.overflow = '';
      if (layer.trigger?.isConnected) layer.trigger.focus();
    };
  }
  document.addEventListener('keydown', (event) => {
    const layer = layers.at(-1);
    if (!layer) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      layer.close();
    }
    if (event.key === 'Tab') {
      const list = focusable(layer.root);
      if (!list.length) {
        event.preventDefault();
        return;
      }
      const first = list[0],
        last = list.at(-1);
      if (
        event.shiftKey &&
        (document.activeElement === first ||
          !layer.root.contains(document.activeElement))
      ) {
        event.preventDefault();
        last.focus();
      } else if (
        !event.shiftKey &&
        (document.activeElement === last ||
          !layer.root.contains(document.activeElement))
      ) {
        event.preventDefault();
        first.focus();
      }
    }
  });
  function confirmAction({
    title,
    description,
    label = '确认删除',
    action,
    danger = true,
  }) {
    return new Promise((resolve) => {
      const root = document.getElementById('modal-root'),
        submit = document.getElementById('modal-submit'),
        cancel = document.getElementById('modal-cancel'),
        error = document.getElementById('modal-error');
      document.getElementById('modal-title').textContent = title;
      document.getElementById('modal-description').textContent = description;
      error.hidden = true;
      submit.textContent = label;
      submit.className = `btn ${danger ? 'btn-danger' : 'btn-primary'}`;
      busy(submit, false);
      cancel.disabled = false;
      let pending = false;
      const close = () => {
        if (pending) return;
        cleanup();
        resolve(false);
      };
      const cleanupLayer = openLayer(root, close);
      function cleanup() {
        cleanupLayer();
        root.onclick = null;
        cancel.onclick = null;
        submit.onclick = null;
      }
      cancel.onclick = close;
      root.onclick = (event) => {
        if (event.target === root) close();
      };
      submit.onclick = async () => {
        if (pending) return;
        pending = true;
        busy(submit, true);
        cancel.disabled = true;
        error.hidden = true;
        try {
          await action();
          pending = false;
          cleanup();
          resolve(true);
        } catch (cause) {
          error.textContent = cause.message;
          error.hidden = false;
          pending = false;
          busy(submit, false);
          cancel.disabled = false;
        }
      };
    });
  }
  function state(root, kind, message, retry) {
    root.replaceChildren();
    const box = document.createElement('div');
    box.className = `${kind}-state`;
    box.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    const text = document.createElement('p');
    text.textContent = message;
    box.append(text);
    if (retry) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'btn';
      button.textContent = '重试';
      button.onclick = retry;
      box.append(button);
    }
    root.append(box);
  }
  window.Studio = {
    escape,
    request,
    post,
    toast,
    busy,
    confirmAction,
    openLayer,
    state,
    number: (value) =>
      value == null || value === '' || !Number.isFinite(Number(value))
        ? '—'
        : Number(value).toFixed(4),
  };
})();
