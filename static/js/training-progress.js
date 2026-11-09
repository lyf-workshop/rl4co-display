// The original request fields remain here so additions to problem/policy controls stay reviewable.
function buildTrainingConfig() {
    const problem = document.getElementById('problem-select').value;
    const model = document.getElementById('model-select').value;
    const algorithm = document.getElementById('algorithm-select').value;
    const numLoc = Number(document.getElementById('num-loc').value);
    const epochs = document.getElementById('epochs').value;
    const batchSize = document.getElementById('batch-size').value;
    const learningRate = document.getElementById('learning-rate').value;
    const config = {
        problem: problem,
        model: model,
        algorithm: algorithm,
        num_loc: numLoc,
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


    }

    // 添加SPCTSP特有参数
    if (problem === 'spctsp') {
        const spctspNumLoc = document.getElementById('spctsp-num-loc');
        const spctspPenaltyFactor = document.getElementById('spctsp-penalty-factor');
        const spctspPrizeRequired = document.getElementById('spctsp-prize-required');

        if (spctspNumLoc) config.num_loc = parseInt(spctspNumLoc.value);
        if (spctspPenaltyFactor) config.penalty_factor = parseFloat(spctspPenaltyFactor.value);
        if (spctspPrizeRequired) config.prize_required = parseFloat(spctspPrizeRequired.value);


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

    // 芯片全局布线首版：Attention 网络排序 + REINFORCE + A* 路由
    if (problem === 'chip_routing') {
        config.grid_width = parseInt(document.getElementById('chip-grid-width').value);
        config.grid_height = parseInt(document.getElementById('chip-grid-height').value);
        config.num_nets = parseInt(document.getElementById('chip-num-nets').value);
        config.num_loc = config.num_nets;
        config.obstacle_ratio = parseFloat(
            document.getElementById('chip-obstacle-ratio').value
        );
        config.edge_capacity = parseInt(
            document.getElementById('chip-edge-capacity').value
        );
        config.model = 'attention';
        config.algorithm = 'reinforce';
        config.dataset_mode = 'random';
        config.dataset_id = null;
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

    // 添加SymNCO特有参数
    if (model === 'symnco') {
        const numAugment = document.getElementById('num-augment');
        const symncoNumStarts = document.getElementById('symnco-num-starts');
        const symncoAlpha = document.getElementById('symnco-alpha');
        const symncoBeta = document.getElementById('symnco-beta');
        if (numAugment) config.num_augment = parseInt(numAugment.value);
        if (symncoNumStarts) config.num_starts = parseInt(symncoNumStarts.value);
        if (symncoAlpha) config.symnco_alpha = parseFloat(symncoAlpha.value);
        if (symncoBeta) config.symnco_beta = parseFloat(symncoBeta.value);
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

    config.train_data_size = Number(document.getElementById('train-data-size').value);
    config.val_data_size = Number(document.getElementById('val-data-size').value);
    if (problem === 'chip_routing') config.seed = Number(document.getElementById('random-seed').value);
    return config;
}

function getTrainingAlgorithmDisplayName(value) { return String(value || '').toLowerCase() === 'a2c' ? 'A3C' : String(value || '—').toUpperCase(); }
function restoreTrainingForm(config) {
    const manager = compatibilityManager;
    for (const [key,id] of [['problem','problem-select'],['model','model-select'],['algorithm','algorithm-select']]) {
        const select = document.getElementById(id);
        if (config[key] != null && [...select.options].some(option => option.value === config[key])) select.value = config[key];
    }
    manager.currentProblem = document.getElementById('problem-select').value;
    manager.currentPolicy = document.getElementById('model-select').value;
    manager.currentAlgorithm = document.getElementById('algorithm-select').value;
    updateDatasetSection(manager.currentProblem); manager.showProblemSpecificParams(); manager.showPolicySpecificParams(); manager.updatePolicyDescription(); manager.updateAlgorithmDescription(); manager.updateAvailableOptions();
    for (const input of document.getElementById('training-form').querySelectorAll('input[name],select[name]')) {
        if (input.name === 'gpu_select' || ['problem','model','algorithm'].includes(input.name)) continue;
        const value = config[input.name === 'force_start_depot' ? 'force_start_at_depot' : input.name];
        if (value == null) continue;
        if (input.type === 'checkbox') input.checked = Boolean(value); else input.value = String(value);
    }
    window.uploadedDatasetId = config.dataset_id || null;
    toggleDatasetUpload(); updateResumeHint(); window.selectedGpuId = config.gpu_id ?? null;
    if (config.dataset_id) {document.getElementById('dataset-info').style.display = 'block'; document.getElementById('dataset-filename').textContent = '数据集 '+config.dataset_id; document.getElementById('dataset-size').textContent = config.num_loc+' 个节点'; document.getElementById('dataset-range').textContent = '当前 Session 的原数据集';}
    manager.validateConfiguration();
}
function renderTrainingResults(results = {}, context = {}) {
    const target = document.getElementById('results-content'); target.replaceChildren();
    const metrics = document.createElement('div'); metrics.className = 'metric-grid';
    const add = (label, value) => { const card = document.createElement('div'); card.className = 'metric-card'; const title = document.createElement('div'); title.className = 'metric-label'; title.textContent = label; const content = document.createElement('div'); content.className = 'metric-value'; content.textContent = value; card.append(title, content); metrics.append(card); };
    add('Best Reward', Studio.number(results.best_reward));
    add('Final Loss', Studio.number(results.final_loss));
    add('Final Reward', Studio.number(results.final_reward));
    if (context.started_at && context.finished_at) add('训练耗时', formatTrainingTime(context.finished_at-context.started_at));
    if (results.best_epoch != null) add('Best Epoch', results.best_epoch);
    target.append(metrics);
    const metadata = document.createElement('p'); metadata.className = 'muted data'; metadata.textContent = [results.problem || context.config?.problem, results.model || context.config?.model, getTrainingAlgorithmDisplayName(results.strategy || context.config?.algorithm)].filter(Boolean).join(' · '); target.append(metadata);
    if (results.routing_metrics) {
        const section = document.createElement('section'); section.className = 'card'; const h = document.createElement('h2'); h.textContent = '芯片布线质量'; section.append(h);
        for (const [key,label] of [['baseline_objective','训练前目标值'],['trained_objective','训练后目标值'],['objective_improvement','目标值改善'],['wirelength','平均线长'],['overflow','拥塞溢出'],['routed_ratio','连通率']]) {
            if (results.routing_metrics[key] == null) continue;
            const p = document.createElement('p'); p.textContent = label+'：'+Studio.number(results.routing_metrics[key]); section.append(p);
        } target.append(section);
    }
    const artifacts = document.createElement('div'); artifacts.className = 'artifact-grid'; target.append(artifacts);
    const images = [
        ...(results.training_curve ? [{path:results.training_curve,label:'训练曲线'}] : []),
        ...(results.plot_paths || []).map((path,i) => ({path,label:'路径结果 / 训练前后对比 '+(i+1)})),
        ...(results.animation_paths || []).map((path,i) => ({path,label:'动态路线 '+(i+1)})),
        ...(results.analysis_paths || []).map((path,i) => ({path,label:'分析结果 '+(i+1)}))
    ];
    for (const {path,label} of images) {
        if (!safeArtifactPath(path)) continue;
        const card = document.createElement('section'); card.className = 'card'; const h = document.createElement('h3'); h.textContent = label;
        const img = document.createElement('img'); img.src = path; img.alt = label; img.loading = 'lazy';
        img.onerror = () => { img.replaceWith(Object.assign(document.createElement('p'), {textContent:'产物暂不可读，可从文件与数据重试。',className:'banner banner-warning'})); };
        const actions = document.createElement('div'); actions.className = 'actions';
        for (const [text,download] of [['查看',false],['下载',true]]) { const link = document.createElement('a'); link.className = 'btn btn-small'; link.href = path; link.textContent = text; if (download) link.download = ''; else {link.target = '_blank';link.rel = 'noopener';} actions.append(link); }
        card.append(h,img,actions); artifacts.append(card);
    }
    if (results.checkpoint_path) {
        const name = String(results.checkpoint_path).split(/[\\/]/).at(-1); const link = document.createElement('a'); link.className = 'btn'; link.href = '/api/download_checkpoint/'+encodeURIComponent(name); link.textContent = '下载 Checkpoint · '+name; target.append(link);
    }
    if (!images.length && !results.checkpoint_path) { const empty = document.createElement('div'); Studio.state(empty, 'empty', context.mode === 'simulation' ? '模拟训练未生成路径图或 Checkpoint。' : '当前响应未包含可下载产物，可从文件与数据查看已保存文件。'); target.append(empty); }
}
function safeArtifactPath(path) { try { const url = new URL(path, location.origin); return url.origin === location.origin && url.pathname.startsWith('/static/model_plots/'); } catch { return false; } }
