// ============================================================
// 全局变量
// ============================================================
let eventSource = null;
let currentSessionId = null;
let heartbeatTimer = null;

// 训练会话持久化工具（localStorage + TTL，2小时过期）
const SESSION_STORAGE_KEY = 'rl4co_training_session';
const SESSION_TTL_MS = 2 * 60 * 60 * 1000; // 2小时

function saveTrainingSession(sessionId) {
    const data = {
        session_id: sessionId,
        saved_at: Date.now()
    };
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(data));
}

function loadTrainingSession() {
    const raw = localStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return null;
    try {
        const data = JSON.parse(raw);
        const age = Date.now() - (data.saved_at || 0);
        if (age > SESSION_TTL_MS) {
            localStorage.removeItem(SESSION_STORAGE_KEY);
            return null;
        }
        return data.session_id;
    } catch {
        localStorage.removeItem(SESSION_STORAGE_KEY);
        return null;
    }
}

function clearTrainingSession() {
    localStorage.removeItem(SESSION_STORAGE_KEY);
}

// 心跳机制：每30秒向服务器报告客户端仍在监控
function startHeartbeat(sessionId) {
    stopHeartbeat();
    heartbeatTimer = setInterval(async () => {
        try {
            await fetch(`/api/training_heartbeat/${sessionId}`, { method: 'POST' });
        } catch (e) {
            console.warn('心跳发送失败:', e);
        }
    }, 30000); // 30秒
}

function stopHeartbeat() {
    if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
}

// ============================================================
// 浏览器通知工具（Web Notification API）
// ============================================================
function requestNotificationPermission() {
    if (!('Notification' in window)) {
        console.log('当前浏览器不支持桌面通知');
        return;
    }
    if (Notification.permission === 'default') {
        Notification.requestPermission().then(permission => {
            if (permission === 'granted') {
                console.log('✓ 已获得通知权限');
            }
        });
    }
}

function sendNotification(title, body, icon = '🎉') {
    // 始终显示页面内横幅通知（兜底）
    showInPageBanner(title, body);

    if (!('Notification' in window)) {
        console.log('[通知] 浏览器不支持 Notification API');
        return;
    }
    console.log('[通知] 当前权限:', Notification.permission, '| 标题:', title);
    if (Notification.permission !== 'granted') {
        // 尝试再次请求权限（用户可能之前未处理弹窗）
        Notification.requestPermission().then(p => {
            if (p === 'granted') {
                _fireNotification(title, body);
            }
        });
        return;
    }
    _fireNotification(title, body);
}

function _fireNotification(title, body) {
    try {
        const notification = new Notification(title, {
            body: body,
            icon: '/static/favicon.ico',
            tag: 'rl4co-training',
            requireInteraction: false,
            silent: false
        });
        notification.onclick = function() {
            window.focus();
            notification.close();
        };
        setTimeout(() => notification.close(), 5000);
    } catch (e) {
        console.warn('[通知] 发送桌面通知失败:', e);
    }
}

function showInPageBanner(title, body) {
    let banner = document.getElementById('inpage-notify-banner');
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'inpage-notify-banner';
        banner.style.cssText = [
            'position:fixed', 'top:20px', 'right:20px', 'z-index:99999',
            'max-width:360px', 'padding:14px 18px', 'border-radius:10px',
            'background:rgba(30,30,40,0.95)', 'color:#fff',
            'box-shadow:0 4px 20px rgba(0,0,0,0.4)',
            'font-size:14px', 'line-height:1.5',
            'transition:opacity .4s', 'opacity:0', 'pointer-events:none'
        ].join(';');
        document.body.appendChild(banner);
    }
    banner.innerHTML = `<strong style="font-size:15px">${title}</strong><br><span style="opacity:.85">${body}</span>`;
    banner.style.opacity = '1';
    banner.style.pointerEvents = 'auto';
    clearTimeout(banner._hideTimer);
    banner._hideTimer = setTimeout(() => {
        banner.style.opacity = '0';
        banner.style.pointerEvents = 'none';
    }, 6000);
}

// 页面加载时尝试重连上次未完成的训练会话
async function tryReconnectSession() {
    const savedId = loadTrainingSession();
    if (!savedId) return;

    try {
        const resp = await fetch(`/api/training_status/${savedId}`);
        if (!resp.ok) {
            clearTrainingSession();
            return;
        }
        const data = await resp.json();
        if (!data.success) {
            clearTrainingSession();
            return;
        }

        const status = data.status.status;
        const statusMsg        = document.getElementById('status-message');
        const progressContainer = document.getElementById('progress-container');
        const progressBar      = document.getElementById('progress-bar');
        const startButton      = document.getElementById('start-button');

        // ── 训练已在后台完成 ──────────────────────────────────────
        if (status === 'completed') {
            clearTrainingSession();
            progressContainer.className = 'progress-container show';
            progressBar.style.width = '100%';
            progressBar.textContent = '100%';
            const pct2 = data.status.progress || 100;
            const ep2  = data.status.epoch || 0;
            statusMsg.className = 'status-message show success';
            statusMsg.innerHTML = `✅ 训练已在后台完成！（Epoch ${ep2}，进度 ${Math.round(pct2)}%）`;
            addLogEntry(`✅ 训练已在后台完成（Epoch ${ep2}）`);

            // 发送浏览器通知
            sendNotification('🎉 训练已在后台完成！', `Epoch ${ep2} | 最优 Reward: ${(data.status.best_reward || 0).toFixed(4)}`);
            return;
        }

        // ── 训练已在后台被中止 ────────────────────────────────────
        if (status === 'stopped') {
            clearTrainingSession();
            const ep3 = data.status.epoch || 0;
            progressContainer.className = 'progress-container show';
            statusMsg.className = 'status-message show';
            statusMsg.style.cssText += 'background:#fff3cd;color:#856404;border-color:#ffc107;';
            statusMsg.innerHTML = `■ 训练已在后台被中止（已完成 Epoch ${ep3}）`;

            // 发送浏览器通知
            sendNotification('■ 训练已中止', `已完成 Epoch ${ep3}`);
            return;
        }

        // ── 训练出错 ──────────────────────────────────────────────
        if (status === 'error') {
            clearTrainingSession();
            return;
        }

        // ── 仍在进行中：running / pausing / paused / stopping ─────
        if (!['running', 'pausing', 'paused', 'stopping'].includes(status)) {
            clearTrainingSession();
            return;
        }

        currentSessionId = savedId;
        const trainingControls = document.getElementById('training-controls');
        const pauseBtn = document.getElementById('pause-btn');
        const resumeBtn = document.getElementById('resume-btn');

        progressContainer.className = 'progress-container show';
        trainingControls.style.display = 'flex';
        startButton.disabled = true;
        startButton.style.opacity = '0.6';
        startButton.textContent = '训练进行中...';

        // 恢复进度条数值
        const pct = data.status.progress || 0;
        progressBar.style.width = pct + '%';
        progressBar.textContent = Math.round(pct) + '%';
        const metricEpoch = document.getElementById('metric-epoch');
        const metricLoss = document.getElementById('metric-loss');
        const metricReward = document.getElementById('metric-reward');
        if (metricEpoch) metricEpoch.textContent = `${data.status.epoch || 0}/...`;
        if (metricLoss)  metricLoss.textContent  = (data.status.loss  || 0).toFixed(4);
        if (metricReward) metricReward.textContent = (data.status.reward || 0).toFixed(4);

        if (status === 'paused') {
            progressBar.classList.add('paused');
            pauseBtn.style.display = 'none';
            resumeBtn.style.display = '';
            statusMsg.className = 'status-message show';
            statusMsg.innerHTML = `⏸ 训练已在后台暂停，可点击"继续"恢复`;
            addLogEntry(`⏸ 已重连暂停中的训练会话 ${savedId.substring(0, 8)}...`);
            // 暂停状态下也启动心跳（表明用户仍在关注）
            startHeartbeat(savedId);
        } else if (status === 'stopping') {
            // 中止中：禁用所有控制按钮，等待 stopped 事件
            pauseBtn.style.display = 'none';
            resumeBtn.style.display = 'none';
            const stopBtn = document.getElementById('stop-btn');
            stopBtn.disabled = true;
            stopBtn.textContent = '■ 中止中...';
            statusMsg.className = 'status-message show';
            statusMsg.innerHTML = `■ 训练正在后台中止，请稍候...`;
            listenToTrainingProgress(savedId); // 内部会启动心跳
        } else {
            // running / pausing：后台训练中，自动重连 SSE
            pauseBtn.style.display = '';
            resumeBtn.style.display = 'none';
            statusMsg.className = 'status-message show success';
            statusMsg.innerHTML = `▶ 训练正在后台运行，已自动重连（会话 ${savedId.substring(0, 8)}...）`;
            addLogEntry(`▶ 训练一直在后台运行，已自动重连`);
            listenToTrainingProgress(savedId); // 内部会启动心跳
        }
    } catch (e) {
        // 网络异常，静默忽略
        clearTrainingSession();
    }
}

// 开始训练功能
function updateResumeHint() {
    const checked = document.getElementById('resume-checkbox').checked;
    const hint = document.getElementById('resume-hint');
    if (checked) {
        hint.textContent = '已勾选：从上次保存的检查点继续训练（相同问题+模型）';
    } else {
        hint.textContent = '未勾选：每次全新训练（默认，推荐）';
    }
}

async function startTraining() {
    // 获取所有配置参数
    const problem = document.getElementById('problem-select').value;
    const model = document.getElementById('model-select').value;
    const algorithm = document.getElementById('algorithm-select').value;
    const numLoc = document.getElementById('num-loc').value;
    const epochs = document.getElementById('epochs').value;
    const batchSize = document.getElementById('batch-size').value;
    const learningRate = document.getElementById('learning-rate').value;

    const statusMsg = document.getElementById('status-message');
    const startButton = document.getElementById('start-button');
    const progressContainer = document.getElementById('progress-container');
    const resultsPanel = document.getElementById('results-panel');

    // 禁用开始按钮
    startButton.disabled = true;
    startButton.style.opacity = '0.6';
    startButton.textContent = '⏳ 训练中...';

    // 隐藏之前的结果
    resultsPanel.className = 'results-panel';

    // 准备训练配置
    const config = {
        problem: problem,
        model: model,
        algorithm: algorithm,
        num_loc: parseInt(numLoc),
        epochs: parseInt(epochs),
        batch_size: parseInt(batchSize),
        learning_rate: parseFloat(learningRate),
        dataset_mode: document.getElementById('dataset-mode').value,
        dataset_id: window.uploadedDatasetId || null,
        gpu_id: (typeof window.selectedGpuId === 'number') ? window.selectedGpuId : null,
        resume_checkpoint: document.getElementById('resume-checkbox').checked
    };

    // 添加CVRP特有参数
    if (problem === 'cvrp' || problem === 'sdvrp' || problem === 'vrptw') {
        const vehicleCapacity = document.getElementById('vehicle-capacity');
        if (vehicleCapacity) {
            config.vehicle_capacity = parseFloat(vehicleCapacity.value);
        }
    }

    // 添加VRPTW特有参数
    if (problem === 'vrptw') {
        const timeWindowWidth = document.getElementById('time-window-width');
        const serviceTime = document.getElementById('service-time');
        const maxTime = document.getElementById('max-time');
        const hardTimeWindows = document.getElementById('hard-time-windows');

        if (timeWindowWidth) config.time_window_width = parseFloat(timeWindowWidth.value);
        if (serviceTime) config.service_time = parseFloat(serviceTime.value);
        if (maxTime) config.max_processing_time = parseFloat(maxTime.value);
        if (hardTimeWindows) config.hard_time_windows = hardTimeWindows.value === 'true';
    }

    // 添加mTSP特有参数
    if (problem === 'mtsp') {
        const numAgents = document.getElementById('num-agents');
        const costType = document.getElementById('cost-type');

        if (numAgents) config.num_agents = parseInt(numAgents.value);
        if (costType) config.cost_type = costType.value;
    }


    // 添加PDP特有参数
    if (problem === 'pdp') {
        const pdpNumLoc = document.getElementById('pdp-num-loc');
        const forceStartDepot = document.getElementById('force-start-depot');

        if (pdpNumLoc) config.num_loc = parseInt(pdpNumLoc.value);
        if (forceStartDepot) config.force_start_at_depot = forceStartDepot.value === 'true';
    }

    // 添加OP特有参数
    if (problem === 'op') {
        const opNumLoc = document.getElementById('op-num-loc');
        const maxLength = document.getElementById('max-length');
        const prizeType = document.getElementById('prize-type');

        if (opNumLoc) config.num_loc = parseInt(opNumLoc.value);
        if (maxLength) config.max_length = parseFloat(maxLength.value);
        if (prizeType) config.prize_type = prizeType.value;
    }

    // 添加PCTSP特有参数
    if (problem === 'pctsp') {
        const pctspNumLoc = document.getElementById('pctsp-num-loc');
        const penaltyFactor = document.getElementById('penalty-factor');
        const prizeRequired = document.getElementById('prize-required');

        if (pctspNumLoc) config.num_loc = parseInt(pctspNumLoc.value);
        if (penaltyFactor) config.penalty_factor = parseFloat(penaltyFactor.value);
        if (prizeRequired) config.prize_required = parseFloat(prizeRequired.value);

        // PCTSP 强制使用 Attention Model
        config.model = 'attention';
    }

    // 添加SPCTSP特有参数
    if (problem === 'spctsp') {
        const spctspNumLoc = document.getElementById('spctsp-num-loc');
        const spctspPenaltyFactor = document.getElementById('spctsp-penalty-factor');
        const spctspPrizeRequired = document.getElementById('spctsp-prize-required');

        if (spctspNumLoc) config.num_loc = parseInt(spctspNumLoc.value);
        if (spctspPenaltyFactor) config.penalty_factor = parseFloat(spctspPenaltyFactor.value);
        if (spctspPrizeRequired) config.prize_required = parseFloat(spctspPrizeRequired.value);

        // SPCTSP 强制使用 Attention Model
        config.model = 'attention';
    }

    // 添加SPCTSP特有参数
    if (problem === 'spctsp') {
        const spctspNumLoc = document.getElementById('spctsp-num-loc');
        const spctspPenaltyFactor = document.getElementById('spctsp-penalty-factor');
        const spctspPrizeRequired = document.getElementById('spctsp-prize-required');

        if (spctspNumLoc) config.num_loc = parseInt(spctspNumLoc.value);
        if (spctspPenaltyFactor) config.penalty_factor = parseFloat(spctspPenaltyFactor.value);
        if (spctspPrizeRequired) config.prize_required = parseFloat(spctspPrizeRequired.value);

        // SPCTSP 强制使用 Attention Model
        config.model = 'attention';
    }

    // 添加FFSP特有参数
    if (problem === 'ffsp') {
        const numStage = document.getElementById('num-stage');
        const numMachine = document.getElementById('num-machine');
        const numJob = document.getElementById('num-job');
        const minTime = document.getElementById('min-time');
        const maxTimeFfsp = document.getElementById('max-time-ffsp');
        const flattenStages = document.getElementById('flatten-stages');

        if (numStage) config.num_stage = parseInt(numStage.value);
        if (numMachine) config.num_machine = parseInt(numMachine.value);
        if (numJob) config.num_job = parseInt(numJob.value);
        if (minTime) config.min_time = parseInt(minTime.value);
        if (maxTimeFfsp) config.max_time = parseInt(maxTimeFfsp.value);
        if (flattenStages) config.flatten_stages = flattenStages.value === 'true';

        // FFSP 强制使用 MatNet
        config.model = 'matnet';
    }

    // 添加POMO特有参数
    if (model === 'pomo') {
        const numStarts = document.getElementById('num-starts');
        if (numStarts) {
            config.num_starts = parseInt(numStarts.value);
        }
    }

    // 添加HAM特有参数
    if (model === 'ham') {
        const hamNorm = document.getElementById('ham-normalization');
        const hamFF = document.getElementById('ham-feedforward-hidden');
        if (hamNorm) config.normalization = hamNorm.value;
        if (hamFF) config.feedforward_hidden = parseInt(hamFF.value);
    }

    // 添加DeepACO特有参数
    if (model === 'deepaco') {
        const nAnts = document.getElementById('n-ants');
        const nIterTrain = document.getElementById('n-iterations-train');
        const nIterTest = document.getElementById('n-iterations-test');
        if (nAnts) config.n_ants = parseInt(nAnts.value);
        if (nIterTrain) config.n_iterations_train = parseInt(nIterTrain.value);
        if (nIterTest) config.n_iterations_test = parseInt(nIterTest.value);
    }

    try {
        // 发送训练请求到后端
        const response = await fetch('/api/start_training', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        const result = await response.json();

        if (result.success) {
            // 显示成功消息
            statusMsg.className = 'status-message show success';
            statusMsg.innerHTML = `✅ 训练已启动！会话 ID: ${result.session_id.substring(0, 8)}...`;

            // 显示进度容器
            progressContainer.className = 'progress-container show';

            // 显示训练控制按钮（暂停/继续）
            const trainingControls = document.getElementById('training-controls');
            const pauseBtn = document.getElementById('pause-btn');
            const resumeBtn = document.getElementById('resume-btn');
            trainingControls.style.display = 'flex';
            pauseBtn.style.display = '';
            resumeBtn.style.display = 'none';
            const stopBtn = document.getElementById('stop-btn');
            stopBtn.disabled = false;
            stopBtn.textContent = '■ 中止';

            // 持久化 session_id（localStorage + TTL），关闭浏览器后仍可重连
            saveTrainingSession(result.session_id);

            // 开始监听训练进度
            listenToTrainingProgress(result.session_id);

        } else {
            throw new Error(result.message || '启动训练失败');
        }

    } catch (error) {
        statusMsg.className = 'status-message show error';
        statusMsg.innerHTML = `❌ 错误: ${error.message}`;

        // 重新启用按钮
        startButton.disabled = false;
        startButton.style.opacity = '1';
        startButton.textContent = '🚀 开始训练';
    }
}

// 监听训练进度（使用 Server-Sent Events）
function listenToTrainingProgress(sessionId) {
    currentSessionId = sessionId;

    // 关闭之前的连接和心跳
    if (eventSource) {
        eventSource.close();
    }
    stopHeartbeat();

    // 创建新的 SSE 连接
    eventSource = new EventSource(`/api/training_progress/${sessionId}`);

    // 启动心跳定时器（每30秒向服务器报告客户端在线）
    startHeartbeat(sessionId);

    // 清空日志
    const trainingLog = document.getElementById('training-log');
    trainingLog.innerHTML = '';

    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);

        switch(data.type) {
            case 'progress':
                updateProgress(data);
                break;
            case 'info':
                addLogEntry(data.message);
                break;
            case 'plot':
                updateTrainingCurve(data.plot_url, data.message);
                addLogEntry('📊 ' + data.message);
                break;
            case 'complete':
                handleTrainingComplete(data);
                break;
            case 'error':
                handleTrainingError(data);
                break;
            case 'paused':
                handleTrainingPaused(data);
                break;
            case 'resumed':
                handleTrainingResumed(data);
                break;
            case 'stopped':
                handleTrainingStopped(data);
                break;
            case 'warning':
                addLogEntry('⚠️ ' + data.message);
                break;
            case 'heartbeat':
                // 心跳消息，不做处理
                break;
        }
    };

    eventSource.onerror = function(error) {
        console.error('SSE Error:', error);
        eventSource.close();
    };
}

// 更新进度显示
function updateProgress(data) {
    const progressBar = document.getElementById('progress-bar');
    const metricEpoch = document.getElementById('metric-epoch');
    const metricLoss = document.getElementById('metric-loss');
    const metricReward = document.getElementById('metric-reward');

    // 更新进度条
    progressBar.style.width = data.progress + '%';
    progressBar.textContent = Math.round(data.progress) + '%';

    // 更新指标
    metricEpoch.textContent = `${data.epoch}/${data.total_epochs}`;
    metricLoss.textContent = data.loss.toFixed(4);
    metricReward.textContent = data.reward.toFixed(4);
}

// 更新训练曲线图
function updateTrainingCurve(plotUrl, message) {
    const container = document.getElementById('training-curves-container');
    const img = document.getElementById('training-curve-img');
    const info = document.getElementById('curve-update-info');

    // 显示曲线容器
    container.style.display = 'block';

    // 添加时间戳防止浏览器缓存
    const timestamp = new Date().getTime();
    img.src = plotUrl + '?t=' + timestamp;

    // 更新提示信息
    info.textContent = message;

    // 添加淡入效果
    img.style.opacity = '0.5';
    img.onload = function() {
        img.style.transition = 'opacity 0.3s';
        img.style.opacity = '1';
    };
}

// 暂停训练
async function pauseTraining() {
    if (!currentSessionId) return;
    const pauseBtn = document.getElementById('pause-btn');
    pauseBtn.disabled = true;
    pauseBtn.textContent = '⏸ 暂停中...';
    try {
        const resp = await fetch(`/api/pause_training/${currentSessionId}`, { method: 'POST' });
        const data = await resp.json();
        if (!data.success) {
            addLogEntry('⚠️ 暂停失败: ' + data.message);
            pauseBtn.disabled = false;
            pauseBtn.textContent = '⏸ 暂停';
        }
        // 成功时等待后端发来 paused SSE 事件再切换 UI
    } catch (e) {
        addLogEntry('⚠️ 暂停请求失败: ' + e.message);
        pauseBtn.disabled = false;
        pauseBtn.textContent = '⏸ 暂停';
    }
}

// 继续训练
async function resumeTraining() {
    if (!currentSessionId) return;
    const resumeBtn = document.getElementById('resume-btn');
    resumeBtn.disabled = true;
    resumeBtn.textContent = '▶ 恢复中...';
    try {
        const resp = await fetch(`/api/resume_training/${currentSessionId}`, { method: 'POST' });
        const data = await resp.json();
        if (data.success) {
            // 若 SSE 连接不存在（页面曾跳转离开），立即重建，
            // 否则沿用现有连接——resumed 事件到达后 handleTrainingResumed 负责切换 UI
            if (!eventSource || eventSource.readyState === EventSource.CLOSED) {
                listenToTrainingProgress(currentSessionId);
            }
        } else {
            addLogEntry('⚠️ 恢复失败: ' + data.message);
            resumeBtn.disabled = false;
            resumeBtn.textContent = '▶ 继续';
        }
    } catch (e) {
        addLogEntry('⚠️ 恢复请求失败: ' + e.message);
        resumeBtn.disabled = false;
        resumeBtn.textContent = '▶ 继续';
    }
}

// 处理训练暂停事件
function handleTrainingPaused(data) {
    const progressBar = document.getElementById('progress-bar');
    const pauseBtn = document.getElementById('pause-btn');
    const resumeBtn = document.getElementById('resume-btn');

    // 进度条变为橙色暂停状态
    progressBar.classList.add('paused');

    // 切换按钮：隐藏暂停，显示继续
    pauseBtn.style.display = 'none';
    pauseBtn.disabled = false;
    pauseBtn.textContent = '⏸ 暂停';
    resumeBtn.style.display = '';
    resumeBtn.disabled = false;
    resumeBtn.textContent = '▶ 继续';

    addLogEntry('⏸ ' + data.message);
}

// 处理训练恢复事件
function handleTrainingResumed(data) {
    const progressBar = document.getElementById('progress-bar');
    const pauseBtn = document.getElementById('pause-btn');
    const resumeBtn = document.getElementById('resume-btn');

    // 进度条恢复正常颜色
    progressBar.classList.remove('paused');

    // 切换按钮：显示暂停，隐藏继续
    resumeBtn.style.display = 'none';
    resumeBtn.disabled = false;
    resumeBtn.textContent = '▶ 继续';
    pauseBtn.style.display = '';
    pauseBtn.disabled = false;
    pauseBtn.textContent = '⏸ 暂停';

    addLogEntry('▶ ' + data.message);
}

// 中止训练
async function stopTraining() {
    if (!currentSessionId) return;
    if (!confirm('确定要中止当前训练吗？\n中止后将在本轮 Epoch 结束时停止，已完成的进度不会丢失。')) return;
    const stopBtn = document.getElementById('stop-btn');
    stopBtn.disabled = true;
    stopBtn.textContent = '■ 中止中...';
    try {
        const resp = await fetch(`/api/stop_training/${currentSessionId}`, { method: 'POST' });
        const data = await resp.json();
        if (!data.success) {
            addLogEntry('⚠️ 中止失败: ' + data.message);
            stopBtn.disabled = false;
            stopBtn.textContent = '■ 中止';
        }
        // 成功时等待后端发来 stopped SSE 事件再更新 UI
    } catch (e) {
        addLogEntry('⚠️ 中止请求失败: ' + e.message);
        stopBtn.disabled = false;
        stopBtn.textContent = '■ 中止';
    }
}

// 处理训练中止事件
function handleTrainingStopped(data) {
    const statusMsg   = document.getElementById('status-message');
    const startButton = document.getElementById('start-button');
    const progressBar = document.getElementById('progress-bar');

    if (eventSource) eventSource.close();
    stopHeartbeat();

    document.getElementById('training-controls').style.display = 'none';
    progressBar.classList.remove('paused');
    currentSessionId = null;
    clearTrainingSession();

    addLogEntry('■ ' + data.message);

    startButton.disabled = false;
    startButton.style.opacity = '1';
    startButton.textContent = '🚀 开始训练';

    statusMsg.className = 'status-message show';
    statusMsg.style.background = '#fff3cd';
    statusMsg.style.color = '#856404';
    statusMsg.style.borderColor = '#ffc107';
    statusMsg.innerHTML = `■ 训练已中止（已完成 Epoch ${data.epoch || 0}，进度 ${Math.round(data.progress || 0)}%）`;

    // 发送浏览器通知
    sendNotification('■ 训练已中止', `已完成 Epoch ${data.epoch || 0}，进度 ${Math.round(data.progress || 0)}%`);
}

// 添加日志条目
function addLogEntry(message) {
    const trainingLog = document.getElementById('training-log');
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';

    const timestamp = new Date().toLocaleTimeString();
    logEntry.textContent = `[${timestamp}] ${message}`;

    trainingLog.appendChild(logEntry);

    // 自动滚动到底部
    trainingLog.scrollTop = trainingLog.scrollHeight;
}

// 处理训练完成
function handleTrainingComplete(data) {
    // ========== 调试信息 ==========
    console.log('='.repeat(80));
    console.log('训练完成！接收到 complete 消息');
    console.log('='.repeat(80));
    console.log('完整 data 对象:', JSON.stringify(data, null, 2));
    console.log('data.results 存在?', 'results' in data);
    if (data.results) {
        console.log('results 对象:', JSON.stringify(data.results, null, 2));
        console.log('plot_paths 存在?', 'plot_paths' in data.results);
        console.log('animation_paths 存在?', 'animation_paths' in data.results);
        console.log('plot_paths 长度:', data.results.plot_paths ? data.results.plot_paths.length : 0);
        console.log('animation_paths 长度:', data.results.animation_paths ? data.results.animation_paths.length : 0);
    } else {
        console.error('❌ data.results 不存在！');
    }
    console.log('='.repeat(80));
    // ========== 调试信息结束 ==========

    const statusMsg = document.getElementById('status-message');
    const startButton = document.getElementById('start-button');
    const resultsPanel = document.getElementById('results-panel');
    const resultsContent = document.getElementById('results-content');

    // 关闭 SSE 连接
    if (eventSource) {
        eventSource.close();
    }
    stopHeartbeat();

    // 隐藏训练控制按钮，清理暂停状态
    document.getElementById('training-controls').style.display = 'none';
    document.getElementById('progress-bar').classList.remove('paused');
    currentSessionId = null;
    clearTrainingSession();

    // 显示完成消息
    addLogEntry('✅ ' + data.message);

    // 重新启用按钮
    startButton.disabled = false;
    startButton.style.opacity = '1';
    startButton.textContent = '🚀 开始训练';

    // 显示训练结果
    const results = data.results;
    // 安全格式化数值：将 null/undefined/非数字 统一转为 0 再格式化
    const fmt = (v) => (typeof v === 'number' && isFinite(v) ? v : parseFloat(v) || 0).toFixed(4);
    let resultsHTML = `
        <div class="result-item">
            <div class="result-label">模型</div>
            <div class="result-value">${(results.model || '').toUpperCase()}</div>
        </div>
        <div class="result-item">
            <div class="result-label">问题类型</div>
            <div class="result-value">${(results.problem || '').toUpperCase()}</div>
        </div>
        <div class="result-item">
            <div class="result-label">训练策略</div>
            <div class="result-value">${(results.strategy || '').toUpperCase()}</div>
        </div>
        <div class="result-item">
            <div class="result-label">训练轮数</div>
            <div class="result-value">${results.total_epochs ?? '-'}</div>
        </div>
        <div class="result-item">
            <div class="result-label">最终 Loss</div>
            <div class="result-value">${fmt(results.final_loss)}</div>
        </div>
        <div class="result-item">
            <div class="result-label">最终 Reward</div>
            <div class="result-value">${fmt(results.final_reward)}</div>
        </div>
        <div class="result-item" style="grid-column: 1 / -1;">
            <div class="result-label">最佳 Reward</div>
            <div class="result-value" style="font-size: 2rem;">🏆 ${fmt(results.best_reward)}</div>
        </div>
    `;

    // 如果有训练曲线，显示训练曲线
    if (results.training_curve) {
        resultsHTML += `
            <div style="grid-column: 1 / -1; margin-top: 1rem;">
                <div class="result-label" style="margin-bottom: 1rem;">📈 训练曲线</div>
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px;">
                    <img src="${results.training_curve}?t=${new Date().getTime()}"
                         style="width: 100%; border-radius: 8px; background: white;"
                         alt="训练曲线">
                    <p style="margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.9; text-align: center;">
                        📊 Loss 和 Reward 随训练轮次的变化趋势
                    </p>
                </div>
            </div>
        `;
    }

    // 如果有可视化图片，添加图片展示
    console.log('检查可视化图片...');
    console.log('plot_paths:', results.plot_paths);
    console.log('animation_paths:', results.animation_paths);

    if (results.plot_paths && results.plot_paths.length > 0) {
        console.log('✅ 将显示', results.plot_paths.length, '个可视化图片');
        resultsHTML += '<div style="grid-column: 1 / -1; margin-top: 1rem;">';
        resultsHTML += '<div class="result-label" style="margin-bottom: 1rem;">📊 路径可视化对比</div>';
        resultsHTML += '<div style="display: grid; gap: 1.5rem;">';
        results.plot_paths.forEach((path, index) => {
            console.log(`添加图片 ${index + 1}:`, path);
            resultsHTML += `
                <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px;">
                    <h4 style="margin-bottom: 1rem; font-size: 1.1rem;">问题 ${index + 1} - 静态对比图</h4>
                    <img src="${path}" style="width: 100%; border-radius: 8px; margin-bottom: 1rem;" alt="路径对比图 ${index + 1}">

                    ${results.animation_paths && results.animation_paths[index] ? `
                        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 2px solid rgba(255,255,255,0.2);">
                            <h4 style="margin-bottom: 1rem; font-size: 1.1rem;">🎬 动态路线生成过程</h4>
                            <img src="${results.animation_paths[index]}" style="width: 100%; border-radius: 8px; background: white;" alt="动态路线图 ${index + 1}">
                            <p style="margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.9; text-align: center;">
                                ⏱️ 动画展示路线逐步构建过程和成本变化
                            </p>
                        </div>
                    ` : ''}
                </div>
            `;
        });
        resultsHTML += '</div></div>';
    } else {
        console.log('❌ 没有可视化图片可显示');
        console.log('  - plot_paths 是否存在:', 'plot_paths' in results);
        console.log('  - plot_paths 值:', results.plot_paths);
        console.log('  - plot_paths 长度:', results.plot_paths ? results.plot_paths.length : 'N/A');
    }

    // 如果有 checkpoint 路径，显示
    if (results.checkpoint_path) {
        resultsHTML += `
            <div class="result-item" style="grid-column: 1 / -1;">
                <div class="result-label">检查点已保存</div>
                <div class="result-value" style="font-size: 0.9rem; word-break: break-all;">${results.checkpoint_path}</div>
            </div>
        `;
    }

    resultsContent.innerHTML = resultsHTML;
    resultsPanel.className = 'results-panel show';

    // 更新状态消息
    statusMsg.className = 'status-message show success';
    statusMsg.innerHTML = '✅ 训练完成！请查看下方结果详情。';

    // 发送浏览器通知（复用上方已声明的 results 变量）
    const notifyBody = `${(results.problem || '').toUpperCase()}-${(results.model || '').toUpperCase()} | 最优 Reward: ${(results.best_reward || 0).toFixed(4)}`;
    sendNotification('🎉 训练完成！', notifyBody);
}

// 处理训练错误
function handleTrainingError(data) {
    const statusMsg = document.getElementById('status-message');
    const startButton = document.getElementById('start-button');

    // 关闭 SSE 连接
    if (eventSource) {
        eventSource.close();
    }
    stopHeartbeat();

    // 隐藏训练控制按钮，清理暂停状态
    document.getElementById('training-controls').style.display = 'none';
    document.getElementById('progress-bar').classList.remove('paused');
    currentSessionId = null;
    clearTrainingSession();

    // 显示错误消息
    addLogEntry('❌ ' + data.message);

    // 重新启用按钮
    startButton.disabled = false;
    startButton.style.opacity = '1';
    startButton.textContent = '🚀 开始训练';

    statusMsg.className = 'status-message show error';
    statusMsg.innerHTML = '❌ 训练出错: ' + data.message;

    // 发送浏览器通知
    sendNotification('❌ 训练出错', data.message);
}
