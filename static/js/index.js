// GPU and dataset adapters preserve the original IDs and API payload keys.
window.selectedGpuId = null;
let _gpuPollTimer = null;
function _renderGpuCards(gpus, source) {
    const container = document.getElementById('gpu-cards-container');
    ['mock','torch','pynvml'].forEach(name => { document.getElementById('gpu-'+name+'-banner').style.display = source === name ? 'block' : 'none'; });
    if (source === 'mock') { window.selectedGpuId = null; container.replaceChildren(); return; }
    if (!gpus?.length) { Studio.state(container, 'empty', '未检测到 GPU，可使用 CPU。'); return; }
    if (!gpus.some(g => g.id === window.selectedGpuId && (g.status !== 'occupied' || (g.sessions || []).some(session => typeof currentSessionId !== 'undefined' && session.session_id === currentSessionId)))) window.selectedGpuId = null;
    container.innerHTML = gpus.map(g => {
        const occupied = g.status === 'occupied';
        return '<label class="gpu-option"><input type="radio" name="gpu_select" value="'+g.id+'" '+(window.selectedGpuId === g.id ? 'checked' : '')+' '+(occupied ? 'disabled' : '')+'><span><strong>GPU '+g.id+' · '+Studio.escape(g.name)+'</strong><small>'+ (occupied ? '占用中' : g.status === 'busy' ? '高负载' : '可用')+' · '+((g.memory_total_mb-g.memory_used_mb)/1024).toFixed(1)+' / '+(g.memory_total_mb/1024).toFixed(1)+' GB 空闲</small><small>利用率 '+(g.utilization_pct < 0 ? '—' : g.utilization_pct+'%')+'</small>'+((g.sessions||[]).map(s => '<small>'+Studio.escape(s.username)+' · '+Studio.escape(s.session_id)+'</small>').join(''))+'</span></label>';
    }).join('');
    container.querySelectorAll('input').forEach(input => input.onchange = () => onGpuSelect(Number(input.value)));
    document.getElementById('gpu-select-cpu').checked = window.selectedGpuId === null;
}
async function loadGpuStatus() {
    try {
        const data = await Studio.request('/api/gpu_status');
        _renderGpuCards(data.gpus, data.source || (data.is_mock ? 'mock' : 'pynvml'));
        document.getElementById('gpu-last-update').textContent = new Date(data.timestamp).toLocaleTimeString();
    } catch (error) { Studio.state(document.getElementById('gpu-cards-container'), 'error', error.message, loadGpuStatus); }
}
function onGpuSelect(id) { window.selectedGpuId = id; loadGpuStatus(); }
function startGpuPolling() { stopGpuPolling(); _gpuPollTimer = setInterval(loadGpuStatus, 10000); }
function stopGpuPolling() { clearInterval(_gpuPollTimer); _gpuPollTimer = null; }
window.addEventListener('studio:gpu', e => _renderGpuCards(e.detail.gpus, e.detail.source));
window.addEventListener('pagehide', stopGpuPolling);
window.addEventListener('pageshow', event => { if (event.persisted) {loadGpuStatus(); startGpuPolling();} });

const DATASET_DESCRIPTIONS = {
    tsp: '支持 JSON（coordinates）、TXT（每行 x y）、TSPLIB。',
    cvrp: 'JSON：coordinates；可选 depot、demands（0–1）。',
    sdvrp: 'JSON：coordinates；可选 depot、demands（0–1）。',
    mtsp: 'JSON：coordinates；可选 depot。',
    vrptw: 'JSON：coordinates；可选 depot、demands、time_windows、service_times。',
    pdp: 'JSON：coordinates，节点数为偶数，前半取货、后半送货；可选 depot。',
    op: 'JSON：coordinates；可选 depot、prizes。',
    pctsp: 'JSON：coordinates；可选 depot、prizes、penalties。',
    spctsp: 'JSON：coordinates；可选 depot、prizes、penalties。',
    atsp: 'ATSP 当前使用随机代价矩阵，不支持上传数据集。',
    ffsp: 'FFSP 当前使用随机作业工时矩阵，不支持上传数据集。',
    chip_routing: '芯片布线当前使用随机网格，不支持上传数据集。'
};
function updateDatasetSection(problem) {
    document.getElementById('random-seed').disabled = problem !== 'chip_routing';
    window.uploadedDatasetId = null;
    document.getElementById('dataset-info').style.display = 'none';
    document.getElementById('upload-status').className = 'status-message';
    const unsupported = ['atsp','ffsp','chip_routing'].includes(problem);
    const mode = document.getElementById('dataset-mode'); mode.disabled = unsupported; mode.value = 'random';
    document.getElementById('dataset-description').textContent = DATASET_DESCRIPTIONS[problem] || '';
    document.getElementById('dataset-file').value = ''; toggleDatasetUpload();
}
function toggleDatasetUpload() { document.getElementById('dataset-upload-section').style.display = document.getElementById('dataset-mode').value === 'upload' ? 'block' : 'none'; }
function selectDataset(datasetId, filename, numLoc) {
    window.uploadedDatasetId = datasetId; document.getElementById('dataset-mode').value = 'upload'; toggleDatasetUpload();
    document.getElementById('dataset-info').style.display = 'block';
    document.getElementById('dataset-filename').textContent = filename;
    document.getElementById('dataset-size').textContent = numLoc + ' 个节点';
    document.getElementById('dataset-range').textContent = '已选择';
    const problem = document.getElementById('problem-select').value;
    const sizeId = {pdp:'pdp-num-loc',op:'op-num-loc',pctsp:'pctsp-num-loc',spctsp:'spctsp-num-loc'}[problem] || 'num-loc';
    document.getElementById(sizeId).value = numLoc;
    closeDatasetManager(); Studio.toast('已选择数据集：' + filename, 'success');
}
async function uploadDataset() {
    const file = document.getElementById('dataset-file').files[0], button = document.getElementById('dataset-upload-btn'), status = document.getElementById('upload-status');
    if (!file) { status.className = 'status-message show error'; status.textContent = '请选择数据集文件。'; return; }
    const problem = document.getElementById('problem-select').value;
    const data = new FormData(); data.append('dataset', file); data.append('problem_type', problem);
    Studio.busy(button, true); status.className = 'status-message show'; status.textContent = '正在上传并解析…';
    try {
        const result = await Studio.request('/api/upload_dataset', {method:'POST', body:data});
        if (document.getElementById('problem-select').value !== problem) { status.textContent = '上传完成，问题类型已切换，请从已有数据集重新选择。'; return; }
        selectDataset(result.dataset_id, result.dataset_info.filename, result.dataset_info.num_loc);
        status.className = 'status-message show success'; status.textContent = result.message;
    } catch (error) { status.className = 'status-message show error'; status.textContent = error.message; }
    finally { Studio.busy(button, false); }
}
let datasetLayerClose;
async function showDatasetManager() {
    const root = document.getElementById('dataset-manager-modal'), list = document.getElementById('datasets-list');
    if (!datasetLayerClose) datasetLayerClose = Studio.openLayer(root, closeDatasetManager);
    root.onclick = e => { if (e.target === root) closeDatasetManager(); };
    Studio.state(list, 'loading', '正在读取数据集…');
    try {
        const data = await Studio.request('/api/list_datasets'); if (!data.datasets.length) return Studio.state(list, 'empty', '还没有上传数据集。');
        const problem = document.getElementById('problem-select').value;
        list.innerHTML = data.datasets.map(d => '<div class="dataset-choice"><strong>'+Studio.escape(d.filename)+'</strong><p class="description">'+Studio.escape(d.problem_type.toUpperCase())+' · '+d.num_loc+' 个节点</p><div class="actions"><button type="button" onclick="selectDataset(this.dataset.id, this.dataset.name, Number(this.dataset.size))" class="btn btn-small" data-id="'+Studio.escape(d.dataset_id)+'" data-name="'+Studio.escape(d.filename)+'" data-size="'+d.num_loc+'" '+(d.problem_type !== problem ? 'disabled title="与当前问题类型不匹配"' : '')+'>使用</button><button type="button" onclick="deleteDataset(this.dataset.id)" class="btn btn-small btn-danger" data-id="'+Studio.escape(d.dataset_id)+'">删除</button><a class="btn btn-small" href="/api/download_dataset/'+encodeURIComponent(d.dataset_id)+'">下载</a></div></div>').join('');
    } catch (error) { Studio.state(list, 'error', error.message, showDatasetManager); }
}
function closeDatasetManager() { datasetLayerClose?.(); datasetLayerClose = null; }
async function deleteDataset(datasetId) {
    const success = await Studio.confirmAction({title:'删除数据集', description:'将删除此数据集，操作无法撤销。已生成的训练产物不受影响。', action:() => Studio.post('/api/delete_dataset', {dataset_id:datasetId})});
    if (success) {
        if (window.uploadedDatasetId === datasetId) updateDatasetSection(document.getElementById('problem-select').value);
        Studio.toast('数据集已删除', 'success'); showDatasetManager();
    }
}
class CompatibilityManager {
    constructor() {
        this.compatibilityData = null;
        this.currentProblem = document.getElementById('problem-select').value;
        this.currentPolicy = document.getElementById('model-select').value;
        this.currentAlgorithm = document.getElementById('algorithm-select').value;

        // ── 约束表（与后端 modules/compatibility.py 保持一致，唯一前端副本）──
        // 策略 → 支持的问题（对应 POLICY_PROBLEM_COMPATIBILITY）
        this.POLICY_PROBLEM_COMPAT = {
            'attention': ['tsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'op', 'pdp', 'pctsp', 'spctsp', 'chip_routing'],
            'pomo':      ['tsp', 'cvrp'],
            'symnco':    ['tsp', 'mtsp', 'cvrp'],
            'ham':       ['pdp'],
            'matnet':    ['atsp', 'ffsp'],
            'ptrnet':    ['tsp'],
            'mdam':      ['tsp', 'cvrp', 'op', 'pdp', 'pctsp', 'spctsp', 'vrptw'],
            'deepaco':   ['tsp', 'mtsp', 'cvrp', 'sdvrp', 'op'],
        };
        // 策略 → 支持的算法（对应 POLICY_ALGORITHM_COMPATIBILITY）
        // 只含单个算法的策略会被“锁定”（禁用算法下拉框）
        this.POLICY_ALGO_COMPAT = {
            'attention': ['reinforce', 'ppo', 'a2c', 'dqn', 'qlearning'],
            'pomo':      ['reinforce'],
            'symnco':    ['reinforce'],
            'ham':       ['reinforce', 'ppo', 'a2c'],
            'matnet':    ['reinforce'],
            'ptrnet':    ['reinforce'],
            'mdam':      ['reinforce'],
            'deepaco':   ['reinforce'],
        };
        // 算法 → 支持的问题（对应 ALGORITHM_PROBLEM_COMPATIBILITY，仅列出受限者）
        this.ALGO_PROBLEM_COMPAT = {
            'ppo': ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'op', 'pdp', 'pctsp', 'spctsp', 'ffsp'],
            'a2c': ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'op', 'pdp', 'pctsp', 'spctsp', 'ffsp'],
            'dqn': ['tsp'],
            'qlearning': ['tsp'],
        };
        // 自带训练逻辑的策略（用于锁定说明文案）
        this.SELF_CONTAINED_POLICIES = ['pomo', 'symnco', 'mdam', 'deepaco', 'matnet'];

        this.ready = this.init();
    }

    async init() {
        await this.loadCompatibilityData();
        this.bindEvents();
        this.updateAvailableOptions();
        // 初始化时显示当前问题的特定参数
        this.showProblemSpecificParams();
        this.showPolicySpecificParams();
        this.updatePolicyDescription();
        this.updateAlgorithmDescription();
        await this.validateConfiguration();
    }

    async loadCompatibilityData() {
        try {
            const response = await fetch('/api/compatibility/info');
            const result = await response.json();
            if (result.success) {
                this.compatibilityData = result.data;
                this.POLICY_PROBLEM_COMPAT = result.data.policy_problem;
                this.POLICY_ALGO_COMPAT = result.data.policy_algorithm;
                this.ALGO_PROBLEM_COMPAT = result.data.algorithm_problem;
                console.log(' 兼容性数据加载成功');
            }
        } catch (error) {
            console.error(' 加载兼容性数据失败:', error);
        }
    }

    bindEvents() {
        for (const id of ['problem-select', 'model-select', 'algorithm-select']) {
            document.getElementById(id).addEventListener('change', () => {
                window.configurationValid = false;
                document.getElementById('start-button').disabled = true;
            });
        }
        document.getElementById('problem-select').addEventListener('change', async (e) => {
            const selectedProblem = e.target.value;
            this.currentProblem = selectedProblem;
            this.showProblemSpecificParams();
            this.updateAvailableOptions();
            updateDatasetSection(selectedProblem);
            const recommendationApplied = await this.applyRecommendedConfig(
                'best',
                { automatic: true }
            );
            if (
                !recommendationApplied
                && this.currentProblem === selectedProblem
            ) {
                this.validateConfiguration();
            }
        });

        document.getElementById('model-select').addEventListener('change', (e) => {
            this.currentPolicy = e.target.value;
            this.updatePolicyDescription();
            this.showPolicySpecificParams();
            this.updateAlgorithmConstraints();
            this.validateConfiguration();
        });

        document.getElementById('algorithm-select').addEventListener('change', (e) => {
            this.currentAlgorithm = e.target.value;
            this.updateAlgorithmDescription();
            this.validateConfiguration();
        });

        document.getElementById('use-recommended-btn').addEventListener('click', () => {
            this.applyRecommendedConfig('best');
        });

        document.getElementById('use-fast-btn').addEventListener('click', () => {
            this.applyRecommendedConfig('fast');
        });
    }

    async updateAvailableOptions() {
        this.updatePolicyConstraints();
        this.updateAlgorithmConstraints();
    }

    async validateConfiguration() {
        const sequence = this.validationSequence = (this.validationSequence || 0) + 1;
        const button = document.getElementById('start-button');
        window.configurationValid = false; button.disabled = true;
        const alert = document.getElementById('config-validation-alert');
        alert.className = 'alert'; alert.textContent = '正在检查兼容性…'; alert.style.display = 'block';
        try {
            const result = await Studio.post('/api/compatibility/validate', {problem:this.currentProblem, model:this.currentPolicy, algorithm:this.currentAlgorithm});
            if (sequence !== this.validationSequence) return;
            this.showValidationMessage(result.valid, result.message, result.level);
        } catch (error) {
            if (sequence !== this.validationSequence) return;
            alert.className = 'alert error'; alert.textContent = '兼容性检查失败：'+error.message;
            const retry = document.createElement('button'); retry.type = 'button'; retry.className = 'btn btn-small'; retry.textContent = '重试检查'; retry.onclick = () => this.validateConfiguration(); alert.append(retry);
        }
    }

    showValidationMessage(valid, message, level) {
        const alert = document.getElementById('config-validation-alert');
        alert.className = 'alert '+(valid ? (level === 'warning' ? 'warning' : 'success') : 'error');
        alert.textContent = (valid ? '兼容性检查通过 · ' : '配置不兼容 · ')+String(message).replace(/[\u2600-\u27BF\uFE0F]/g, '');
        if (!valid) {
            const alternatives = Object.entries(this.POLICY_PROBLEM_COMPAT).filter(([,problems]) => problems.includes(this.currentProblem)).map(([policy]) => policy);
            alert.textContent += ' 可用模型：'+alternatives.join('、');
        }
        alert.style.display = 'block';
        window.configurationValid = valid;
        document.getElementById('start-button').disabled = !valid || Boolean(window.trainingActive);
    }

    async applyRecommendedConfig(preference = 'best', options = {}) {
        const requestedProblem = this.currentProblem;
        const automatic = options.automatic === true;
        try {
            const response = await fetch(
                `/api/compatibility/recommend?problem=${requestedProblem}&preference=${preference}`
            );
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const result = await response.json();

            if (result.success && requestedProblem === this.currentProblem) {
                const { policy, algorithm } = result.recommended;
                const modelSelect = document.getElementById('model-select');
                const algorithmSelect = document.getElementById('algorithm-select');
                modelSelect.value = policy;
                algorithmSelect.value = algorithm;

                this.currentPolicy = policy;
                this.currentAlgorithm = algorithm;

                this.updatePolicyDescription();
                this.updateAlgorithmDescription();
                this.showPolicySpecificParams();
                // 重新计算约束并应用算法锁定（推荐配置可能切到自带训练的策略）
                this.updateAlgorithmConstraints();
                await this.validateConfiguration();

                Studio.toast('已应用推荐配置', 'success');
                return true;
            }
            return false;
        } catch (error) {
            console.error('应用推荐配置失败:', error);
            return false;
        }
    }

    updatePolicyDescription() {
        const descriptions = {
            'attention': '经典的基于Transformer的构造式模型，速度快，适合入门',
            'pomo': '同时从多个起点构建路径，利用问题对称性，质量高但计算量大',
            'symnco': '利用二面体8对称增强+多损失训练（L_ps+L_ss+L_inv），TSP/mTSP/CVRP最高质量',
            'ham': '异构注意力模型，区分depot/pickup/delivery节点，PDP取送货问题首选',
            'matnet': '矩阵注意力网络，专为非对称距离矩阵和调度问题设计，ATSP/FFSP首选',
            'ptrnet': '开创性的Seq2Seq模型（2015），首个深度学习CO方法，适合学习和研究',
            'mdam': '多解码器注意力模型：多个解码器并行解码后取最优路径，与AM兼容性相同但解质量更高',
            'deepaco': '深度蚁群优化：GNN学习信息素热图，AntSystem执行搜索，展示混合AI+元启发式范式'
        };
        const desc = document.getElementById('model-description');
        if (desc) desc.textContent = descriptions[this.currentPolicy] || '';
    }

    updateAlgorithmDescription() {
        const descriptions = {
            'reinforce': '最基础的策略梯度算法，简单易懂，适合学习',
            'ppo': '工业界首选算法，训练稳定，超参数不敏感，推荐用于生产',
            'a2c': 'Actor-Critic方法，方差小，收敛快，性能介于REINFORCE和PPO之间',
            'dqn': '使用深度神经网络学习动作价值函数，适用于离散动作空间',
            'qlearning': '经典的无模型价值学习方法，通过动作价值更新学习策略'
        };
        const desc = document.getElementById('algorithm-description');
        if (desc) desc.textContent = descriptions[this.currentAlgorithm] || '';
    }

    showPolicySpecificParams() {
        const pomoParams = document.getElementById('pomo-params');
        if (pomoParams) {
            pomoParams.style.display = this.currentPolicy === 'pomo' ? 'block' : 'none';
        }
        const hamParams = document.getElementById('ham-params');
        if (hamParams) {
            hamParams.style.display = this.currentPolicy === 'ham' ? 'block' : 'none';
        }
        const symncoParams = document.getElementById('symnco-params');
        if (symncoParams) {
            symncoParams.style.display = this.currentPolicy === 'symnco' ? 'block' : 'none';
        }
        const deepacoParams = document.getElementById('deepaco-params');
        if (deepacoParams) {
            deepacoParams.style.display = this.currentPolicy === 'deepaco' ? 'block' : 'none';
        }
        // 算法下拉框的锁定/解锁由 updateAlgorithmConstraints → applyAlgorithmLock 统一处理
    }

    showProblemSpecificParams() {
        // 显示/隐藏CVRP特有参数
        const cvrpParams = document.getElementById('cvrp-params');
        const vrptwParams = document.getElementById('vrptw-params');
        const mtspParams = document.getElementById('mtsp-params');
        const pdpParams = document.getElementById('pdp-params');
        const opParams = document.getElementById('op-params');
        const chipRoutingParams = document.getElementById('chip-routing-params');
        const numLocGroup = document.getElementById('num-loc-group');
        const isChipRouting = this.currentProblem === 'chip_routing';

        if (cvrpParams) {
            const isCVRP = this.currentProblem === 'cvrp' || this.currentProblem === 'sdvrp' || this.currentProblem === 'vrptw';
            cvrpParams.style.display = isCVRP ? 'block' : 'none';
        }

        // 显示/隐藏VRPTW特有参数
        if (vrptwParams) {
            const isVRPTW = this.currentProblem === 'vrptw';
            vrptwParams.style.display = isVRPTW ? 'block' : 'none';
        }

        // 显示/隐藏mTSP特有参数
        if (mtspParams) {
            const isMTSP = this.currentProblem === 'mtsp';
            mtspParams.style.display = isMTSP ? 'block' : 'none';
        }

        // 显示/隐藏PDP特有参数
        if (pdpParams) {
            const isPDP = this.currentProblem === 'pdp';
            pdpParams.style.display = isPDP ? 'block' : 'none';
        }

        // 显示/隐藏OP特有参数
        if (opParams) {
            const isOP = this.currentProblem === 'op';
            opParams.style.display = isOP ? 'block' : 'none';
        }

        // 显示/隐藏PCTSP特有参数
        const pctspParams = document.getElementById('pctsp-params');
        if (pctspParams) {
            pctspParams.style.display = this.currentProblem === 'pctsp' ? 'block' : 'none';
        }

        // 显示/隐藏SPCTSP特有参数
        const spctspParams = document.getElementById('spctsp-params');
        if (spctspParams) {
            spctspParams.style.display = this.currentProblem === 'spctsp' ? 'block' : 'none';
        }

        // 显示/隐藏FFSP特有参数
        const ffspParams = document.getElementById('ffsp-params');
        if (ffspParams) {
            ffspParams.style.display = this.currentProblem === 'ffsp' ? 'block' : 'none';
        }

        if (chipRoutingParams) {
            chipRoutingParams.style.display = isChipRouting ? 'block' : 'none';
        }

        if (numLocGroup) {
            numLocGroup.style.display = isChipRouting ? 'none' : 'block';
        }

        // ── 统一的策略模型与算法约束 ──
        this.updatePolicyConstraints();
        this.updateAlgorithmConstraints();
    }

    updatePolicyConstraints() {
        const POLICY_PROBLEM_COMPAT = this.POLICY_PROBLEM_COMPAT;

        const PROBLEM_NAMES = {
            'tsp': 'TSP', 'atsp': 'ATSP', 'mtsp': 'mTSP',
            'cvrp': 'CVRP', 'sdvrp': 'SDVRP', 'vrptw': 'VRPTW',
            'op': 'OP', 'pdp': 'PDP', 'pctsp': 'PCTSP',
            'spctsp': 'SPCTSP', 'ffsp': 'FFSP', 'chip_routing': 'Chip Routing',
        };

        const modelSelect = document.getElementById('model-select');
        if (!modelSelect) return;

        const problem = this.currentProblem;
        let hasValidSelection = false;
        let firstValidValue = null;

        Array.from(modelSelect.options).forEach(opt => {
            const compatProblems = POLICY_PROBLEM_COMPAT[opt.value] || [];
            const isCompat = compatProblems.includes(problem);

            opt.disabled = !isCompat;
            opt.title = isCompat ? '' : `此策略不支持 ${PROBLEM_NAMES[problem] || problem.toUpperCase()} 问题`;

            if (isCompat && !firstValidValue) {
                firstValidValue = opt.value;
            }
            if (isCompat && opt.value === modelSelect.value) {
                hasValidSelection = true;
            }
        });

        if (!hasValidSelection && firstValidValue) {
            modelSelect.value = firstValidValue;
            this.currentPolicy = firstValidValue;
            this.updatePolicyDescription();
            this.showPolicySpecificParams();
        }
    }

    updateAlgorithmConstraints() {
        const POLICY_ALGO_COMPAT = this.POLICY_ALGO_COMPAT;
        const ALGO_PROBLEM_COMPAT = this.ALGO_PROBLEM_COMPAT;

        const POLICY_NAMES = {
            'attention': 'Attention Model', 'pomo': 'POMO',
            'symnco': 'SymNCO', 'ham': 'HAM',
            'matnet': 'MatNet', 'ptrnet': 'Pointer Network',
            'mdam': 'MDAM', 'deepaco': 'DeepACO',
        };

        const algoSelect = document.getElementById('algorithm-select');
        if (!algoSelect) return;

        const policy = this.currentPolicy;
        const compatAlgos = POLICY_ALGO_COMPAT[policy] || ['reinforce', 'ppo', 'a2c'];
        const allowedAlgos = compatAlgos.filter(algorithm => {
            const compatibleProblems = ALGO_PROBLEM_COMPAT[algorithm];
            return !compatibleProblems
                || compatibleProblems.includes(this.currentProblem);
        });
        let hasValidSelection = false;
        let firstValidValue = null;

        Array.from(algoSelect.options).forEach(opt => {
            const compatibleProblems = ALGO_PROBLEM_COMPAT[opt.value];
            const supportsProblem = !compatibleProblems
                || compatibleProblems.includes(this.currentProblem);
            const isCompat = compatAlgos.includes(opt.value) && supportsProblem;

            opt.disabled = !isCompat;
            opt.title = isCompat
                ? ''
                : `${POLICY_NAMES[policy] || policy} 或当前问题不支持此算法`;

            if (isCompat && !firstValidValue) {
                firstValidValue = opt.value;
            }
            if (isCompat && opt.value === algoSelect.value) {
                hasValidSelection = true;
            }
        });

        if (!hasValidSelection && firstValidValue) {
            algoSelect.value = firstValidValue;
            this.currentAlgorithm = firstValidValue;
            this.updateAlgorithmDescription();
        }

        // ── 锁定：策略仅支持单一算法时，禁用算法下拉框并说明 ──
        this.applyAlgorithmLock(policy, allowedAlgos);
    }

    /**
     * 当策略只支持一个算法（reinforce）时，锁定算法下拉框并显示说明；
     * 否则解锁。用户因此无法选出后端会拒绝的冲突组合（如 POMO + PPO）。
     */
    applyAlgorithmLock(policy, compatAlgos) {
        const algoSelect = document.getElementById('algorithm-select');
        const note = document.getElementById('algorithm-lock-note');
        const noteText = document.getElementById('algorithm-lock-text');
        if (!algoSelect) return;

        const locked = Array.isArray(compatAlgos) && compatAlgos.length === 1;

        if (locked) {
            const onlyAlgo = compatAlgos[0];
            algoSelect.value = onlyAlgo;
            this.currentAlgorithm = onlyAlgo;
            algoSelect.disabled = true;
            algoSelect.classList.add('locked');
            if (noteText) {
                noteText.textContent = this.SELF_CONTAINED_POLICIES.includes(policy)
                    ? '该模型自带训练逻辑，算法已锁定为 REINFORCE'
                    : `${this.getPolicyDisplayName(policy)} 仅支持 REINFORCE，算法已锁定`;
            }
            if (note) note.style.display = 'block';
            this.updateAlgorithmDescription();
        } else {
            algoSelect.disabled = false;
            algoSelect.classList.remove('locked');
            if (note) note.style.display = 'none';
        }
    }

    getPolicyDisplayName(policy) {
        const names = {
            'attention': 'Attention Model (经典)',
            'pomo': 'POMO (高质量)',
            'symnco': 'SymNCO (对称优化)',
            'ham': 'HAM (PDP专用)',
            'ptrnet': 'Pointer Network (开创性)',
            'ptr': 'Pointer Network (开创性)',
            'matnet': 'MatNet (调度专用)'
        };
        return names[policy] || policy.toUpperCase();
    }

    getAlgorithmDisplayName(algorithm) {
        const names = {
            'reinforce': 'REINFORCE (入门)',
            'ppo': 'PPO (生产推荐)',
            'a2c': 'A3C (平衡)',
            'dqn': 'DQN',
            'qlearning': 'Q-Learning'
        };
        return names[algorithm] || algorithm.toUpperCase();
    }
}

// 初始化兼容性管理器
let compatibilityManager;
document.addEventListener('DOMContentLoaded', () => {
    const query = new URLSearchParams(location.search);
    for (const [key, id] of [['problem','problem-select'], ['model','model-select']]) {
        const select = document.getElementById(id);
        if ([...select.options].some(option => option.value === query.get(key))) select.value = query.get(key);
    }
    compatibilityManager = new CompatibilityManager();
    loadGpuStatus(); startGpuPolling();
    tryReconnectSession();

    // 初始化数据集区域（根据默认问题类型）
    updateDatasetSection(document.getElementById('problem-select').value);
    compatibilityManager.ready.then(async () => {
        const datasetId = query.get('dataset');
        if (!datasetId) return;
        try {
            const data = await Studio.request('/api/list_datasets');
            const dataset = data.datasets.find(item => item.dataset_id === datasetId);
            if (!dataset || dataset.problem_type !== document.getElementById('problem-select').value) throw new Error('数据集不存在或与当前问题类型不匹配。');
            selectDataset(dataset.dataset_id, dataset.filename, dataset.num_loc);
        } catch (error) { Studio.toast(error.message, 'error'); }
    });

    // FFSP 参数实时更新逻辑
    const updateFFSPScale = () => {
        const numStage = document.getElementById('num-stage');
        const numMachine = document.getElementById('num-machine');
        const numJob = document.getElementById('num-job');
        const scaleInfo = document.getElementById('ffsp-scale-info');

        if (numStage && numMachine && numJob && scaleInfo) {
            const stages = parseInt(numStage.value) || 3;
            const machines = parseInt(numMachine.value) || 4;
            const jobs = parseInt(numJob.value) || 20;
            const totalMachines = stages * machines;
            const totalOps = jobs * stages;

            scaleInfo.textContent = `${stages}阶段 × ${machines}机器 × ${jobs}作业 = 总机器${totalMachines}台，总操作${totalOps}个`;
        }
    };

    // 监听 FFSP 参数变化
    ['num-stage', 'num-machine', 'num-job'].forEach(id => {
        const elem = document.getElementById(id);
        if (elem) {
            elem.addEventListener('input', updateFFSPScale);
            elem.addEventListener('change', updateFFSPScale);
        }
    });

    // 初始更新一次
    updateFFSPScale();
});

// 退出登录功能
async function logout() {
    try { await Studio.post('/api/logout'); location.href = '/login'; }
    catch (error) { Studio.toast(error.message, 'error'); }
}
