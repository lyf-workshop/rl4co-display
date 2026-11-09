(() => {
  'use strict';
  const button = document.querySelector('.menu-toggle'),
    sidebar = document.getElementById('app-sidebar'),
    overlay = document.querySelector('.sidebar-overlay');
  let closeLayer;
  const close = () => {
    sidebar?.classList.remove('is-open');
    button?.setAttribute('aria-expanded', 'false');
    closeLayer?.();
    if (sidebar) sidebar.hidden = false;
    closeLayer = null;
    if (overlay) overlay.hidden = true;
  };
  button?.addEventListener('click', () => {
    if (closeLayer) return close();
    sidebar.classList.add('is-open');
    button.setAttribute('aria-expanded', 'true');
    overlay.hidden = false;
    closeLayer = Studio.openLayer(sidebar, close);
    overlay.onclick = close;
  });
  window.addEventListener('resize', () => {
    if (innerWidth >= 1200 && closeLayer) close();
  });
  async function gpu() {
    const name = document.getElementById('shell-gpu-name'),
      status = document.getElementById('shell-gpu-status');
    if (!name) return;
    try {
      const data = await Studio.request('/api/gpu_status');
      window.dispatchEvent(new CustomEvent('studio:gpu', { detail: data }));
      if (data.is_mock || data.source === 'mock') {
        name.textContent = 'CPU · 无真实 GPU';
        status.textContent = '后端 GPU 数据为演示模式';
        return;
      }
      const device =
        data.gpus.find((item) => item.status === 'idle') || data.gpus[0];
      if (!device) {
        name.textContent = 'CPU';
        status.textContent = '未检测到 GPU';
        return;
      }
      name.textContent = `GPU ${device.id} · ${device.name}`;
      status.textContent = `${device.status === 'idle' ? '可用' : '占用 / 高负载'} · ${((device.memory_total_mb - device.memory_used_mb) / 1024).toFixed(1)} GB 空闲`;
    } catch (error) {
      name.textContent = '资源状态读取失败';
      status.textContent = error.message;
    }
  }
  if (sidebar) {
    gpu();
    let timer = setInterval(gpu, 15000);
    window.addEventListener('pagehide', () => clearInterval(timer));
    window.addEventListener('pageshow', (event) => {
      if (event.persisted) {
        clearInterval(timer);
        gpu();
        timer = setInterval(gpu, 15000);
      }
    });
  }
})();
