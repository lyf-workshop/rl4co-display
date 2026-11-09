// Server state is authoritative. Reconnecting only reads status/SSE; it never starts a task.
let currentSessionId = null;
let eventSource = null;
let heartbeatTimer = null;
let statusTimer = null;
let clockTimer = null;
let monitorContext = {};
let trainingRows = [];
let terminalHandled = false;
let startingTraining = false;
let lastStatus = {};
let rewardChart, lossChart;
const SESSION_STORAGE_KEY =
  'rl4co_training_session:' + document.body.dataset.userId;
window.trainingActive = false;
function stored(key, value) {
  try {
    if (value === undefined)
      return JSON.parse(localStorage.getItem(key) || 'null');
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch {
    return null;
  }
}
function saveTrainingSession(id) {
  stored(SESSION_STORAGE_KEY, { session_id: id });
}
function clearTrainingSession() {
  stored(SESSION_STORAGE_KEY, null);
}
function formatTrainingTime(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  const minutes = Math.floor(seconds / 60);
  return minutes >= 60
    ? Math.floor(minutes / 60) + 'h ' + (minutes % 60) + 'm'
    : minutes + 'm ' + Math.floor(seconds % 60) + 's';
}
function setWorkspaceView(view) {
  window.scrollTo(0, 0);
  document.getElementById('configuration-view').hidden = view !== 'config';
  document.getElementById('training-section').hidden = view !== 'monitor';
  document.getElementById('results-panel').hidden = view !== 'results';
  for (const key of ['config', 'monitor', 'results'])
    document
      .getElementById('view-' + key)
      .setAttribute('aria-selected', String(key === view));
  requestAnimationFrame(() => StudioCharts.resize());
}
function connection(message, type = 'info', retry = false) {
  const node = document.getElementById('connection-banner');
  node.className = 'banner banner-' + type;
  document.getElementById('connection-text').textContent = message;
  document.getElementById('reconnect-training').hidden = !retry;
}
function setTrainingState(state) {
  const active = [
    'starting',
    'running',
    'pausing',
    'paused',
    'stopping',
  ].includes(state);
  window.trainingActive = active;
  document.getElementById('training-form').inert = active;
  document.getElementById('start-button').disabled =
    active || !window.configurationValid;
  const label =
    {
      starting: '正在启动',
      running: '训练中',
      pausing: '正在暂停',
      paused: '已暂停',
      stopping: '正在中止',
      completed: '已完成',
      stopped: '已中止',
      error: '训练失败',
    }[state] || '新实验';
  const badge = document.getElementById('workspace-status');
  badge.textContent = label;
  badge.className =
    'status ' +
    (state === 'completed'
      ? 'status-success'
      : ['paused', 'pausing', 'stopping', 'stopped'].includes(state)
        ? 'status-warning'
        : state === 'error'
          ? 'status-error'
          : active
            ? 'status-running'
            : '');
  document.getElementById('training-controls').hidden = !active;
  const pause = document.getElementById('pause-btn'),
    resume = document.getElementById('resume-btn'),
    stop = document.getElementById('stop-btn');
  pause.style.display = ['paused', 'pausing'].includes(state) ? 'none' : '';
  pause.disabled = state !== 'running';
  resume.style.display = ['paused', 'pausing'].includes(state) ? '' : 'none';
  resume.disabled = !['paused', 'pausing'].includes(state);
  stop.disabled = state === 'stopping' || state === 'starting';
  document
    .getElementById('progress-bar')
    .classList.toggle('paused', ['pausing', 'paused'].includes(state));
  document.getElementById('new-experiment').hidden = active;
}
function updateResumeHint() {
  document.getElementById('resume-hint').textContent = document.getElementById(
    'resume-checkbox',
  ).checked
    ? '从匹配的问题与模型的已保存 Checkpoint 继续训练。'
    : '默认开始全新训练。';
}
async function startTraining() {
  if (startingTraining || window.trainingActive) return;
  if (!window.configurationValid) {
    Studio.toast('请先通过兼容性检查。', 'warning');
    return;
  }
  const form = document.getElementById('training-form');
  const inputs = [...form.querySelectorAll('input, select')].filter(
    (el) =>
      !el.disabled &&
      (el.getClientRects().length || el.closest('#advanced-parameters')) &&
      !['file', 'checkbox', 'radio'].includes(el.type),
  );
  inputs.forEach((input) => input.removeAttribute('aria-invalid'));
  form.querySelectorAll('[data-field-error]').forEach((node) => node.remove());
  const invalid = inputs.find(
    (input) => !input.checkValidity() || input.value === '',
  );
  if (invalid) {
    invalid.setAttribute('aria-invalid', 'true');
    const error = document.createElement('p');
    error.className = 'field-error';
    error.dataset.fieldError = invalid.id;
    error.id = invalid.id + '-error';
    error.textContent =
      invalid.min && invalid.max
        ? '请输入 ' +
          invalid.min +
          '–' +
          invalid.max +
          ' 范围内的有效数值，并遵守步长 ' +
          (invalid.step || '1') +
          '。'
        : '请填写此字段。';
    invalid.insertAdjacentElement('afterend', error);
    invalid.setAttribute('aria-describedby', error.id);
    if (invalid.closest('details')) invalid.closest('details').open = true;
    invalid.focus();
    Studio.toast('请检查参数范围，并填写有效数值。', 'error');
    return;
  }
  const config = buildTrainingConfig();
  if (config.dataset_mode === 'upload' && !config.dataset_id) {
    Studio.toast('请先上传或选择数据集。', 'warning');
    return;
  }
  startingTraining = true;
  const button = document.getElementById('start-button');
  Studio.busy(button, true);
  try {
    const result = await Studio.post('/api/start_training', config);
    currentSessionId = result.session_id;
    const sessionUrl = new URL(location.href);
    sessionUrl.searchParams.set('session', currentSessionId);
    history.replaceState(null, '', sessionUrl);
    terminalHandled = false;
    trainingRows = [];
    lastStatus = {};
    monitorContext = {
      config,
      started_at: Date.now() / 1000,
      mode: result.message.includes('模拟') ? 'simulation' : 'real',
    };
    saveTrainingSession(currentSessionId);
    stored(SESSION_STORAGE_KEY + ':rows:' + currentSessionId, []);
    document.getElementById('training-log').replaceChildren();
    document.getElementById('view-monitor').disabled = false;
    document.getElementById('view-results').disabled = true;
    document.getElementById('monitor-session').textContent =
      'Session · ' + currentSessionId;
    setTrainingState('starting');
    setWorkspaceView('monitor');
    Studio.toast(result.message, 'success');
    addLogEntry(result.message);
    listenToTrainingProgress(currentSessionId);
    startStatusPolling();
    await pollTrainingStatus();
  } catch (error) {
    Studio.toast(error.message, 'error');
    const node = document.getElementById('status-message');
    node.className = 'status-message show error';
    node.textContent = error.message;
  } finally {
    startingTraining = false;
    Studio.busy(button, false);
    button.disabled = window.trainingActive || !window.configurationValid;
  }
}
function closeTrainingConnection() {
  eventSource?.close();
  eventSource = null;
  clearInterval(heartbeatTimer);
  clearInterval(statusTimer);
  clearInterval(clockTimer);
  heartbeatTimer = statusTimer = clockTimer = null;
}
function listenToTrainingProgress(id) {
  if (
    eventSource &&
    currentSessionId === id &&
    eventSource.readyState !== EventSource.CLOSED
  )
    return;
  eventSource?.close();
  currentSessionId = id;
  const source = new EventSource(
    '/api/training_progress/' + encodeURIComponent(id),
  );
  eventSource = source;
  connection('正在连接训练事件…');
  source.onopen = () => {
    if (eventSource !== source) return;
    connection(
      '实时连接已建立' +
        (monitorContext.mode === 'simulation' ? ' · 模拟训练模式' : ''),
      'success',
    );
  };
  source.onerror = () => {
    if (eventSource !== source || terminalHandled) return;
    connection(
      '连接中断，正在自动重连。后台训练状态继续读取。',
      'warning',
      true,
    );
    pollTrainingStatus();
  };
  source.onmessage = (event) => {
    if (eventSource !== source || terminalHandled) return;
    try {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'progress':
          updateProgress(data);
          break;
        case 'plot':
          updateTrainingCurve(data.plot_url, data.message);
          break;
        case 'complete':
          handleTrainingComplete(data);
          break;
        case 'stopped':
          finishTraining('stopped', data);
          break;
        case 'error':
          handleTrainingError(data);
          break;
        case 'paused':
          setTrainingState('paused');
          addLogEntry(data.message);
          break;
        case 'resumed':
          setTrainingState('running');
          addLogEntry(data.message);
          break;
        case 'info':
        case 'warning':
          addLogEntry(data.message);
          break;
      }
    } catch (error) {
      addLogEntry('无法解析训练事件：' + error.message);
    }
  };
  clearInterval(heartbeatTimer);
  heartbeatTimer = setInterval(
    () =>
      Studio.post('/api/training_heartbeat/' + encodeURIComponent(id)).catch(
        () => {},
      ),
    30000,
  );
  Studio.post('/api/training_heartbeat/' + encodeURIComponent(id)).catch(
    () => {},
  );
}
function startStatusPolling() {
  clearInterval(statusTimer);
  statusTimer = setInterval(pollTrainingStatus, 4000);
  clearInterval(clockTimer);
  clockTimer = setInterval(updateClock, 1000);
}
let pollingStatus = false;
async function pollTrainingStatus() {
  if (!currentSessionId || terminalHandled || pollingStatus) return;
  pollingStatus = true;
  const sessionId = currentSessionId;
  try {
    const data = await Studio.request(
      '/api/training_status/' + encodeURIComponent(sessionId),
    );
    if (currentSessionId !== sessionId || terminalHandled) return;
    monitorContext = { ...monitorContext, ...data.context };
    lastStatus = data.status;
    updateProgress({
      ...data.status,
      total_epochs: data.status.total_epochs || monitorContext.config?.epochs,
    });
    const state = data.status.status || 'starting';
    if (state === 'completed')
      handleTrainingComplete({
        message: '训练已完成',
        results: data.context?.results || {
          final_loss: data.status.loss,
          final_reward: data.status.reward,
          best_reward: data.status.best_reward,
        },
      });
    else if (['stopped', 'error'].includes(state))
      finishTraining(state, {
        message:
          state === 'error' ? '后端报告训练失败，请查看日志。' : '训练已中止',
      });
    else setTrainingState(state);
  } catch (error) {
    if (currentSessionId === sessionId && !terminalHandled)
      connection(
        '状态读取失败：' + error.message + ' · 将自动重试',
        'warning',
        true,
      );
  } finally {
    pollingStatus = false;
  }
}
function updateProgress(data) {
  if (data.epoch != null)
    document.getElementById('metric-epoch').textContent =
      data.epoch +
      ' / ' +
      (data.total_epochs ?? monitorContext.config?.epochs ?? '—');
  if (data.loss != null)
    document.getElementById('metric-loss').textContent = Studio.number(
      data.loss,
    );
  if (data.reward != null)
    document.getElementById('metric-current-reward').textContent =
      Studio.number(data.reward);
  if (data.best_reward != null)
    document.getElementById('metric-reward').textContent = Studio.number(
      data.best_reward,
    );
  if (data.progress != null) {
    const value = Math.min(100, Math.max(0, Number(data.progress))),
      bar = document.getElementById('progress-bar');
    bar.style.width = value + '%';
    bar.textContent = Math.round(value) + '%';
    document
      .querySelector('[role="progressbar"]')
      .setAttribute('aria-valuenow', value);
  }
  if (data.epoch > 0 && (data.reward != null || data.loss != null)) {
    const existing = trainingRows.findIndex((row) => row.epoch === data.epoch),
      row = { epoch: data.epoch, reward: data.reward, loss: data.loss };
    if (existing >= 0) trainingRows[existing] = row;
    else trainingRows.push(row);
    trainingRows.sort((a, b) => a.epoch - b.epoch);
    stored(SESSION_STORAGE_KEY + ':rows:' + currentSessionId, trainingRows);
    updateCharts();
  }
  updateClock();
  window.dispatchEvent(
    new CustomEvent('studio:training', {
      detail: { session_id: currentSessionId, ...data, ...monitorContext },
    }),
  );
}
function updateCharts() {
  rewardChart ||= StudioCharts.create('reward-chart');
  lossChart ||= StudioCharts.create('loss-chart');
  StudioCharts.line(
    rewardChart,
    trainingRows,
    'reward',
    'Reward',
    '--color-brand',
  );
  StudioCharts.line(lossChart, trainingRows, 'loss', 'Loss', '--color-success');
  const latest = trainingRows.at(-1);
  if (latest) {
    document.getElementById('reward-chart-summary').textContent =
      'Epoch ' + latest.epoch + ' · Reward ' + Studio.number(latest.reward);
    document.getElementById('loss-chart-summary').textContent =
      'Epoch ' + latest.epoch + ' · Loss ' + Studio.number(latest.loss);
  }
}
function updateClock() {
  const elapsed = monitorContext.started_at
    ? (monitorContext.finished_at || Date.now() / 1000) -
      monitorContext.started_at
    : null;
  document.getElementById('metric-elapsed').textContent =
    elapsed == null ? '—' : formatTrainingTime(elapsed);
  const total = monitorContext.config?.epochs,
    epoch = lastStatus.epoch;
  document.getElementById('metric-eta').textContent =
    window.trainingActive &&
    lastStatus.status === 'running' &&
    epoch >= 2 &&
    elapsed > 0 &&
    total > epoch
      ? '预计剩余 ' + formatTrainingTime((elapsed / epoch) * (total - epoch))
      : 'ETA —';
}
function updateTrainingCurve(path, message) {
  if (!safeArtifactPath(path)) return;
  document.getElementById('training-curves-container').style.display = 'block';
  document.getElementById('training-curve-img').src = path + '?t=' + Date.now();
  document.getElementById('curve-update-info').textContent = message || '';
}
function addLogEntry(message) {
  if (!message) return;
  const node = document.getElementById('training-log'),
    row = document.createElement('div');
  row.className = 'log-entry';
  row.textContent = '[' + new Date().toLocaleTimeString() + '] ' + message;
  node.append(row);
  while (node.children.length > 500) node.firstElementChild.remove();
  node.scrollTop = node.scrollHeight;
}
async function controlTraining(action) {
  if (!currentSessionId) return;
  const response = await Studio.post(
    '/api/' + action + '_training/' + encodeURIComponent(currentSessionId),
  );
  if (action === 'pause' || action === 'stop') {
    lastStatus.status = action === 'pause' ? 'pausing' : 'stopping';
    setTrainingState(lastStatus.status);
  }
  addLogEntry(response.message);
  Studio.toast(response.message, 'success');
  await pollTrainingStatus();
}
async function pauseTraining() {
  const button = document.getElementById('pause-btn');
  Studio.busy(button, true);
  try {
    await controlTraining('pause');
  } catch (error) {
    Studio.toast(error.message, 'error');
  } finally {
    Studio.busy(button, false);
    setTrainingState(lastStatus.status || 'running');
  }
}
async function resumeTraining() {
  const button = document.getElementById('resume-btn');
  Studio.busy(button, true);
  try {
    await controlTraining('resume');
  } catch (error) {
    Studio.toast(error.message, 'error');
  } finally {
    Studio.busy(button, false);
  }
}
async function stopTraining() {
  if (!currentSessionId) return;
  await Studio.confirmAction({
    title: '中止当前训练？',
    description:
      '中止将在当前 Epoch 边界生效。已有日志与后端实际保存的 Checkpoint 会保留；是否生成其他产物取决于训练器。',
    label: '确认中止',
    action: () => controlTraining('stop'),
  });
}
function finishTraining(state, data) {
  if (terminalHandled) return;
  terminalHandled = true;
  closeTrainingConnection();
  setTrainingState(state);
  clearTrainingSession();
  addLogEntry(data.message);
  connection(
    state === 'completed' ? '训练已结束，实时连接已关闭。' : data.message,
    state === 'error' ? 'error' : 'warning',
  );
  const node = document.getElementById('status-message');
  node.className =
    'status-message show ' +
    (state === 'completed' ? 'success' : state === 'error' ? 'error' : '');
  node.textContent = data.message;
  Studio.toast(
    data.message,
    state === 'completed' ? 'success' : state === 'error' ? 'error' : 'warning',
  );
}
function handleTrainingComplete(data) {
  if (terminalHandled) return;
  finishTraining('completed', data);
  renderTrainingResults(data.results, monitorContext);
  document.getElementById('compare-result').href =
    '/benchmark?source=experiments&session=' +
    encodeURIComponent(currentSessionId);
  document.getElementById('view-results').disabled = false;
  setWorkspaceView('results');
  const id = currentSessionId;
  // The thread records its end time immediately after publishing completion.
  Studio.request('/api/training_status/' + encodeURIComponent(id))
    .then((snapshot) => {
      if (currentSessionId !== id) return;
      monitorContext = { ...monitorContext, ...snapshot.context };
      renderTrainingResults(
        snapshot.context?.results || data.results,
        monitorContext,
      );
      loadResultArtifacts(id, snapshot.context?.results || data.results || {});
    })
    .catch(() => loadResultArtifacts(id, data.results || {}));
}
function handleTrainingError(data) {
  finishTraining('error', data);
}
async function loadResultArtifacts(id, results) {
  if (currentSessionId !== id) return;
  document.getElementById('saved-artifacts')?.remove();
  const section = document.createElement('section');
  section.id = 'saved-artifacts';
  section.className = 'card';
  document.getElementById('results-content').append(section);
  Studio.state(section, 'loading', '读取已保存产物…');
  try {
    const data = await Studio.request('/api/list_files'),
      session = data.sessions.find((row) => row.session_id === id);
    if (currentSessionId !== id) return;
    if (!session) {
      Studio.state(section, 'empty', '此 Session 没有已登记的产物。');
      return;
    }
    const files =
      session.all_files || Object.values(session.files).flat().filter(Boolean);
    section.innerHTML = '<h2>已保存产物</h2>';
    files.forEach((file) => {
      const row = document.createElement('p'),
        link = document.createElement('a');
      link.textContent = file.name;
      link.href =
        file.type === 'checkpoint'
          ? '/api/download_checkpoint/' + encodeURIComponent(file.name)
          : file.path;
      link.download = '';
      row.append(link);
      section.append(row);
    });
  } catch (error) {
    if (currentSessionId === id)
      Studio.state(section, 'error', error.message, () =>
        loadResultArtifacts(id, results),
      );
  }
}
async function tryReconnectSession() {
  let id =
    new URLSearchParams(location.search).get('session') ||
    stored(SESSION_STORAGE_KEY)?.session_id;
  try {
    const sessionData = await Studio.request('/api/training_sessions'),
      sessions = sessionData.sessions;
    let mode = document.getElementById('training-mode');
    if (!mode) {
      mode = document.createElement('p');
      mode.id = 'training-mode';
      document.getElementById('start-button').parentElement.append(mode);
    }
    mode.className =
      sessionData.training_mode === 'simulation'
        ? 'banner banner-warning'
        : 'description';
    mode.textContent =
      sessionData.training_mode === 'simulation'
        ? '当前服务器未接入 RL4CO，使用项目原有模拟训练模式。'
        : 'RL4CO 已接入 · 真实训练模式';
    if (id && !sessions.some((row) => row.session_id === id && row.live)) {
      clearTrainingSession();
      id = null;
    }
    if (!id)
      id = sessions.find(
        (row) =>
          row.live &&
          ['starting', 'running', 'pausing', 'paused', 'stopping'].includes(
            row.status,
          ),
      )?.session_id;
    if (!id) return;
    const data = await Studio.request(
      '/api/training_status/' + encodeURIComponent(id),
    );
    currentSessionId = id;
    terminalHandled = false;
    monitorContext = data.context || {};
    await compatibilityManager.ready;
    restoreTrainingForm(monitorContext.config || {});
    trainingRows = stored(SESSION_STORAGE_KEY + ':rows:' + id) || [];
    lastStatus = data.status;
    document.getElementById('view-monitor').disabled = false;
    document.getElementById('monitor-session').textContent = 'Session · ' + id;
    setWorkspaceView('monitor');
    updateProgress(data.status);
    if (data.status.status === 'completed') {
      handleTrainingComplete({
        message: '训练已在后台完成',
        results: data.context?.results || {
          final_loss: data.status.loss,
          final_reward: data.status.reward,
          best_reward: data.status.best_reward,
        },
      });
      return;
    }
    if (['stopped', 'error'].includes(data.status.status)) {
      finishTraining(data.status.status, {
        message: '后台训练状态：' + data.status.status,
      });
      return;
    }
    saveTrainingSession(id);
    setTrainingState(data.status.status || 'starting');
    addLogEntry('已恢复后台 Session 监控');
    listenToTrainingProgress(id);
    startStatusPolling();
  } catch (error) {
    const node = document.getElementById('status-message');
    node.className = 'status-message show error';
    node.textContent = '恢复监控失败：' + error.message + '。';
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'btn btn-small';
    retry.textContent = '重试恢复';
    retry.onclick = () => {
      node.className = 'status-message';
      tryReconnectSession();
    };
    node.append(retry);
  }
}
document.getElementById('training-form').addEventListener('input', (event) => {
  const input = event.target;
  if (!input.id) return;
  document.getElementById(input.id + '-error')?.remove();
  input.removeAttribute('aria-invalid');
  input.removeAttribute('aria-describedby');
});
document.getElementById('training-form').addEventListener('submit', (event) => {
  event.preventDefault();
  startTraining();
});
for (const view of ['config', 'monitor', 'results'])
  document.getElementById('view-' + view).onclick = () =>
    setWorkspaceView(view);
document.getElementById('reconnect-training').onclick = () => {
  if (currentSessionId) {
    eventSource?.close();
    eventSource = null;
    listenToTrainingProgress(currentSessionId);
    pollTrainingStatus();
  } else tryReconnectSession();
};
document.getElementById('new-experiment').onclick = () => {
  const url = new URL(location.href);
  url.searchParams.delete('session');
  history.replaceState(null, '', url);
  closeTrainingConnection();
  currentSessionId = null;
  monitorContext = {};
  trainingRows = [];
  setTrainingState('idle');
  setWorkspaceView('config');
  document.getElementById('status-message').className = 'status-message';
};
window.addEventListener('pagehide', closeTrainingConnection);
window.addEventListener('pageshow', (event) => {
  if (event.persisted && window.trainingActive) tryReconnectSession();
});
