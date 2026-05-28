// 平滑滚动效果
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ============================================================
// GPU 选择模块
// ============================================================
window.selectedGpuId = null;   // null 表示使用 CPU
let _gpuPollTimer = null;

/** 渲染单块 GPU 卡片 */
function _buildGpuCard(gpu) {
    const memPct = gpu.memory_pct;
    const isOccupied = gpu.status === 'occupied';
    const isBusy    = gpu.status === 'busy';
    const isIdle    = gpu.status === 'idle';
    const isSelected = window.selectedGpuId === gpu.id;

    const statusColor = isOccupied ? '#dc3545' : isBusy ? '#fd7e14' : '#28a745';
    const statusText  = isOccupied ? '占用中' : isBusy ? '高负载' : '空闲';
    const memBarColor = memPct > 85 ? '#dc3545' : memPct > 60 ? '#fd7e14' : '#28a745';

    const sessions = gpu.sessions || [];
    const sessionTip = sessions.length
        ? sessions.map(s => `@${s.username}`).join(', ')
        : '';

    const borderStyle = isSelected
        ? 'border:2px solid #667eea;box-shadow:0 0 0 3px rgba(102,126,234,0.25);'
        : 'border:2px solid #eee;';
    const disabledAttr = isOccupied ? 'disabled' : '';

    return `
    <label style="display:block;cursor:${isOccupied?'not-allowed':'pointer'};
                  border-radius:12px;padding:0.9rem;background:white;
                  transition:box-shadow 0.2s,border-color 0.2s;
                  position:relative;${borderStyle}"
           title="${sessionTip}">
      <input type="radio" name="gpu_select" value="${gpu.id}"
             id="gpu-select-${gpu.id}" style="display:none;"
             ${isSelected ? 'checked' : ''} ${disabledAttr}
             onchange="onGpuSelect(${gpu.id})">

      <!-- Mock 角标 -->
      <span id="gpu-mock-badge-${gpu.id}" style="display:none;
          position:absolute;top:6px;right:6px;font-size:0.68rem;
          background:#ffc107;color:#856404;padding:1px 6px;border-radius:8px;">
        演示
      </span>

      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
        <span style="font-weight:600;font-size:0.88rem;color:#333;">
          GPU ${gpu.id}
        </span>
        <span style="font-size:0.78rem;padding:2px 8px;border-radius:10px;
                     background:${statusColor}22;color:${statusColor};font-weight:600;">
          ${statusText}
        </span>
      </div>

      <div style="font-size:0.8rem;color:#555;margin-bottom:0.5rem;
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
           title="${gpu.name}">
        ${gpu.name}
      </div>

      <!-- 显存进度条 -->
      <div style="font-size:0.75rem;color:#888;margin-bottom:3px;">
        显存 ${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB (${memPct}%)
      </div>
      <div style="height:6px;background:#eee;border-radius:3px;overflow:hidden;">
        <div style="height:100%;width:${memPct}%;
                    background:${memBarColor};border-radius:3px;
                    transition:width 0.4s;"></div>
      </div>

      <!-- 利用率 -->
      <div style="font-size:0.75rem;color:#888;margin-top:0.4rem;">
        GPU 利用率 ${gpu.utilization_pct === -1 ? 'N/A' : gpu.utilization_pct + '%'}
      </div>

      ${sessionTip ? `<div style="font-size:0.72rem;color:#dc3545;margin-top:0.3rem;">
        占用：${sessionTip}</div>` : ''}

    </label>`;
}

/** 渲染所有 GPU 卡片 */
function _renderGpuCards(gpus, source) {
    const container = document.getElementById('gpu-cards-container');
    if (!container) return;

    // 控制三种横幅的显示
    document.getElementById('gpu-mock-banner').style.display   = source === 'mock'   ? 'block' : 'none';
    document.getElementById('gpu-torch-banner').style.display  = source === 'torch'  ? 'block' : 'none';
    document.getElementById('gpu-pynvml-banner').style.display = source === 'pynvml' ? 'block' : 'none';

    if (!gpus || gpus.length === 0) {
        container.innerHTML = '<div style="color:#aaa;font-size:0.9rem;padding:1rem;">未检测到 GPU，将使用 CPU 训练</div>';
        return;
    }

    container.innerHTML = gpus.map(g => _buildGpuCard(g)).join('');

    // Mock 模式加演示角标
    if (source === 'mock') {
        gpus.forEach(g => {
            const badge = document.getElementById(`gpu-mock-badge-${g.id}`);
            if (badge) badge.style.display = 'inline';
        });
    }
}

/** 加载 GPU 状态（调接口） */
async function loadGpuStatus() {
    const container = document.getElementById('gpu-cards-container');
    try {
        const resp = await fetch('/api/gpu_status');
        if (!resp.ok) {
            if (resp.status === 401) {
                container.innerHTML = '<div style="color:#e74c3c;padding:1rem;">⚠️ 请先登录以查看 GPU 状态</div>';
            } else {
                container.innerHTML = `<div style="color:#e74c3c;padding:1rem;">⚠️ 加载失败 (HTTP ${resp.status})</div>`;
            }
            return;
        }
        const data = await resp.json();
        if (data.success) {
            // 兼容旧版（无 source 字段时根据 is_mock 推断）
            const source = data.source || (data.is_mock ? 'mock' : 'pynvml');
            _renderGpuCards(data.gpus, source);
            const ts = new Date(data.timestamp);
            const el = document.getElementById('gpu-last-update');
            if (el) el.textContent = `上次更新：${ts.toLocaleTimeString()}`;
        } else {
            container.innerHTML = `<div style="color:#e74c3c;padding:1rem;">⚠️ ${data.message || '查询失败'}</div>`;
        }
    } catch(e) {
        console.error('GPU 状态加载失败:', e);
        if (container) {
            container.innerHTML = `<div style="color:#e74c3c;padding:1rem;">⚠️ 网络错误: ${e.message}</div>`;
        }
    }
}

/** 用户选择 GPU */
function onGpuSelect(gpuId) {
    window.selectedGpuId = gpuId;
    // 更新 CPU 选项和 GPU 卡片的选中边框
    const cpuLabel = document.getElementById('cpu-option-label');
    if (cpuLabel) {
        cpuLabel.style.borderColor = gpuId === null ? '#667eea' : '#ddd';
    }
    // 重新渲染边框（简单刷新一次状态即可）
    loadGpuStatus();
}

/** 启动/停止 GPU 状态轮询（每 5 秒） */
function startGpuPolling() {
    stopGpuPolling();
    _gpuPollTimer = setInterval(loadGpuStatus, 5000);
}
function stopGpuPolling() {
    if (_gpuPollTimer) { clearInterval(_gpuPollTimer); _gpuPollTimer = null; }
}

// 页面加载完成后立即拉取 GPU 状态并启动轮询
document.addEventListener('DOMContentLoaded', function() {
    loadGpuStatus();
    startGpuPolling();
    requestNotificationPermission();
    tryReconnectSession();
});

// 页面滚动效果
window.addEventListener('scroll', function() {
    const header = document.querySelector('.header');
    if (window.scrollY > 50) {
        header.style.boxShadow = '0 2px 30px rgba(0,0,0,0.15)';
    } else {
        header.style.boxShadow = '0 2px 20px rgba(0,0,0,0.1)';
    }
});

// 表单验证
document.getElementById('epochs').addEventListener('input', function() {
    if (this.value < 1) this.value = 1;
    if (this.value > 10000) this.value = 10000;
});

document.getElementById('batch-size').addEventListener('input', function() {
    if (this.value < 1) this.value = 1;
    if (this.value > 2048) this.value = 2048;
});

// 各问题类型的数据集说明文字
const DATASET_DESCRIPTIONS = {
    tsp:    '💡 支持格式：JSON（含 coordinates 字段）、TXT（每行 x y）、TSP（TSPLIB 格式）。',
    cvrp:   '💡 上传 JSON 文件。必填：coordinates（客户坐标列表）。可选：depot（仓库坐标）、demands（需求量，值域 0-1）。',
    sdvrp:  '💡 上传 JSON 文件。必填：coordinates（客户坐标列表）。可选：depot（仓库坐标）、demands（需求量，值域 0-1）。',
    mtsp:   '💡 上传 JSON 文件。必填：coordinates（城市坐标列表）。可选：depot（仓库坐标）。',
    op:     '💡 上传 JSON 文件。必填：coordinates（地点坐标列表）。可选：depot（仓库坐标）、prizes（各地点奖励值，≥0）。',
    pdp:    '💡 上传 JSON 文件。必填：coordinates（节点坐标，数量必须是偶数，前半为取货点，后半为送货点）。可选：depot（仓库坐标）。',
    pctsp:  '💡 上传 JSON 文件。必填：coordinates（客户坐标列表）。可选：depot、prizes（奖励）、penalties（惩罚）。',
    spctsp: '💡 上传 JSON 文件。必填：coordinates（客户坐标列表）。可选：depot、prizes（期望奖励）、penalties（惩罚）。',
    vrptw:  '💡 上传 JSON 文件。必填：coordinates（客户坐标列表）。可选：depot、demands（需求量，值域 0-1）、time_windows（每项 [start, end]）、service_times（服务时长）。',
    atsp:   '⛔ ATSP 使用代价矩阵而非坐标，暂不支持上传自定义数据集。',
    ffsp:   '⛔ FFSP 使用作业工时矩阵而非坐标，暂不支持上传自定义数据集。',
};

function updateDatasetSection(problemType) {
    const unsupported = ['atsp', 'ffsp'];
    const disabled = unsupported.includes(problemType);

    const desc = DATASET_DESCRIPTIONS[problemType] || DATASET_DESCRIPTIONS['tsp'];
    const descEl = document.getElementById('dataset-description');
    if (descEl) {
        descEl.textContent = desc;
        descEl.style.color = disabled ? '#dc3545' : '#666';
    }

    const modeEl = document.getElementById('dataset-mode');
    const fileEl = document.getElementById('dataset-file');
    const uploadBtn = document.getElementById('dataset-upload-btn');
    const managerBtn = document.getElementById('dataset-manager-btn');

    if (modeEl) modeEl.disabled = disabled;
    if (fileEl) fileEl.disabled = disabled;
    if (uploadBtn) uploadBtn.disabled = disabled;
    if (managerBtn) managerBtn.disabled = disabled;

    if (disabled) {
        if (modeEl) modeEl.value = 'random';
        const uploadSection = document.getElementById('dataset-upload-section');
        if (uploadSection) uploadSection.style.display = 'none';
    }
}

// 切换数据集上传区域显示
function toggleDatasetUpload() {
    const mode = document.getElementById('dataset-mode').value;
    const uploadSection = document.getElementById('dataset-upload-section');

    if (mode === 'upload') {
        uploadSection.style.display = 'block';
    } else {
        uploadSection.style.display = 'none';
    }
}

// 上传数据集
async function uploadDataset() {
    const fileInput = document.getElementById('dataset-file');
    const uploadStatus = document.getElementById('upload-status');
    const datasetInfo = document.getElementById('dataset-info');

    if (!fileInput.files || fileInput.files.length === 0) {
        uploadStatus.className = 'status-message show error';
        uploadStatus.textContent = '❌ 请先选择一个数据集文件';
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('dataset', file);
    formData.append('problem_type', document.getElementById('problem-select').value);

    // 显示上传中状态
    uploadStatus.className = 'status-message show';
    uploadStatus.style.background = '#fff3cd';
    uploadStatus.style.color = '#856404';
    uploadStatus.textContent = '⏳ 正在上传和解析数据集...';

    try {
        const response = await fetch('/api/upload_dataset', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            uploadStatus.className = 'status-message show success';
            uploadStatus.textContent = '✅ ' + result.message;

            // 显示数据集信息
            datasetInfo.style.display = 'block';
            document.getElementById('dataset-filename').textContent = result.dataset_info.filename;
            document.getElementById('dataset-size').textContent = (result.dataset_info.num_loc || result.dataset_info.num_cities) + ' 个节点';
            document.getElementById('dataset-range').textContent = result.dataset_info.coord_range;

            // 保存数据集ID到全局变量
            window.uploadedDatasetId = result.dataset_id;
        } else {
            uploadStatus.className = 'status-message show error';
            uploadStatus.textContent = '❌ ' + result.message;
            datasetInfo.style.display = 'none';
        }
    } catch (error) {
        uploadStatus.className = 'status-message show error';
        uploadStatus.textContent = '❌ 上传失败: ' + error.message;
        datasetInfo.style.display = 'none';
    }
}

// 显示数据集管理器
async function showDatasetManager() {
    const modal = document.getElementById('dataset-manager-modal');
    const listDiv = document.getElementById('datasets-list');

    modal.style.display = 'block';
    listDiv.innerHTML = '<p style="text-align: center; color: #999;">正在加载...</p>';

    try {
        const response = await fetch('/api/list_datasets');
        const result = await response.json();

        if (result.success) {
            if (result.datasets.length === 0) {
                listDiv.innerHTML = '<p style="text-align: center; color: #999;">还没有上传任何数据集</p>';
            } else {
                const currentProblem = document.getElementById('problem-select').value;
                const sorted = [...result.datasets].sort((a, b) => {
                    const aMatch = (a.problem_type || 'tsp') === currentProblem ? 0 : 1;
                    const bMatch = (b.problem_type || 'tsp') === currentProblem ? 0 : 1;
                    return aMatch - bMatch;
                });
                let html = '<div style="display: grid; gap: 1rem;">';
                sorted.forEach(dataset => {
                    const uploadTime = new Date(dataset.upload_time).toLocaleString('zh-CN');
                    const numLoc = dataset.num_loc || dataset.num_cities || '?';
                    const ptLabel = (dataset.problem_type || 'tsp').toUpperCase();
                    const matched = (dataset.problem_type || 'tsp') === currentProblem;
                    html += `
                        <div style="border: 2px solid ${matched ? '#667eea' : '#e0e0e0'}; border-radius: 8px; padding: 1rem; background: #f8f9fa; opacity: ${matched ? 1 : 0.5};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div style="flex: 1;">
                                    <div style="margin-bottom: 0.4rem;">
                                        <span style="display:inline-block; padding:2px 8px; border-radius:12px; background:#667eea; color:#fff; font-size:0.75rem; margin-right:0.5rem;">${ptLabel}</span>
                                        <strong style="color: #333;">📁 ${dataset.filename}</strong>
                                    </div>
                                    <p style="margin: 0.3rem 0; color: #666; font-size: 0.9rem;">
                                        <strong>节点数量:</strong> ${numLoc}
                                    </p>
                                    <p style="margin: 0.3rem 0; color: #999; font-size: 0.85rem;">上传时间: ${uploadTime}</p>
                                    ${!matched ? '<p style="margin: 0.2rem 0; color: #dc3545; font-size: 0.8rem;">⚠️ 与当前问题类型不匹配</p>' : ''}
                                </div>
                                <div>
                                    <button onclick="selectDataset('${dataset.dataset_id}', '${dataset.filename}', ${numLoc})"
                                            ${!matched ? 'disabled title="请切换到对应问题类型后再选择"' : ''}
                                            style="padding: 0.5rem 1rem; background: ${matched ? '#667eea' : '#aaa'}; color: white; border: none; border-radius: 5px; cursor: ${matched ? 'pointer' : 'not-allowed'}; margin-right: 0.5rem;">
                                        选择
                                    </button>
                                    <button onclick="deleteDataset('${dataset.dataset_id}')"
                                            style="padding: 0.5rem 1rem; background: #dc3545; color: white; border: none; border-radius: 5px; cursor: pointer;">
                                        删除
                                    </button>
                                </div>
                            </div>
                        </div>
                    `;
                });
                html += '</div>';
                listDiv.innerHTML = html;
            }
        } else {
            listDiv.innerHTML = `<p style="text-align: center; color: #dc3545;">加载失败: ${result.message}</p>`;
        }
    } catch (error) {
        listDiv.innerHTML = `<p style="text-align: center; color: #dc3545;">加载失败: ${error.message}</p>`;
    }
}

// 关闭数据集管理器
function closeDatasetManager() {
    document.getElementById('dataset-manager-modal').style.display = 'none';
}

// 选择数据集
function selectDataset(datasetId, filename, numCities) {
    window.uploadedDatasetId = datasetId;

    // 切换到上传模式
    document.getElementById('dataset-mode').value = 'upload';
    toggleDatasetUpload();

    // 显示数据集信息
    const datasetInfo = document.getElementById('dataset-info');
    datasetInfo.style.display = 'block';
    document.getElementById('dataset-filename').textContent = filename;
    document.getElementById('dataset-size').textContent = numCities + ' 个城市';
    document.getElementById('dataset-range').textContent = '已选择';

    // 关闭弹窗
    closeDatasetManager();

    // 显示提示
    const uploadStatus = document.getElementById('upload-status');
    uploadStatus.className = 'status-message show success';
    uploadStatus.textContent = '✅ 已选择数据集: ' + filename;
}

// 删除数据集
async function deleteDataset(datasetId) {
    if (!confirm('确定要删除这个数据集吗？此操作无法撤销。')) {
        return;
    }

    try {
        const response = await fetch('/api/delete_dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ dataset_id: datasetId })
        });

        const result = await response.json();

        if (result.success) {
            alert('✅ ' + result.message);
            // 刷新数据集列表
            showDatasetManager();

            // 如果删除的是当前选中的数据集，清除选择
            if (window.uploadedDatasetId === datasetId) {
                window.uploadedDatasetId = null;
                document.getElementById('dataset-info').style.display = 'none';
                document.getElementById('upload-status').className = 'status-message';
            }
        } else {
            alert('❌ ' + result.message);
        }
    } catch (error) {
        alert('❌ 删除失败: ' + error.message);
    }
}

// 查看模型详情

// ============================================
// 兼容性管理器
// ============================================
class CompatibilityManager {
    constructor() {
        this.compatibilityData = null;
        this.currentProblem = 'tsp';
        this.currentPolicy = 'attention';
        this.currentAlgorithm = 'reinforce';

        this.init();
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
    }

    async loadCompatibilityData() {
        try {
            const response = await fetch('/api/compatibility/info');
            const result = await response.json();
            if (result.success) {
                this.compatibilityData = result.data;
                console.log('✅ 兼容性数据加载成功');
            }
        } catch (error) {
            console.error('❌ 加载兼容性数据失败:', error);
        }
    }

    bindEvents() {
        document.getElementById('problem-select').addEventListener('change', (e) => {
            this.currentProblem = e.target.value;
            this.showProblemSpecificParams();
            this.updateAvailableOptions();
            this.validateConfiguration();
            updateDatasetSection(e.target.value);
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
        try {
            const response = await fetch('/api/compatibility/validate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    problem: this.currentProblem,
                    model: this.currentPolicy,
                    algorithm: this.currentAlgorithm
                })
            });

            const result = await response.json();
            if (result.success) {
                this.showValidationMessage(result.valid, result.message, result.level);
            }
        } catch (error) {
            console.error('验证配置失败:', error);
        }
    }

    showValidationMessage(valid, message, level) {
        const alert = document.getElementById('config-validation-alert');
        alert.className = 'alert';

        if (level === 'success') {
            alert.style.backgroundColor = '#d4edda';
            alert.style.color = '#155724';
        } else if (level === 'warning') {
            alert.style.backgroundColor = '#fff3cd';
            alert.style.color = '#856404';
        } else {
            alert.style.backgroundColor = '#f8d7da';
            alert.style.color = '#721c24';
        }

        alert.innerHTML = message;
        alert.style.display = 'block';
        alert.style.padding = '10px';
        alert.style.borderRadius = '4px';

        if (level === 'success') {
            setTimeout(() => { alert.style.display = 'none'; }, 3000);
        }
    }

    async applyRecommendedConfig(preference = 'best') {
        try {
            const response = await fetch(`/api/compatibility/recommend?problem=${this.currentProblem}&preference=${preference}`);
            const result = await response.json();

            if (result.success) {
                const { policy, algorithm } = result.recommended;
                document.getElementById('model-select').value = policy;
                document.getElementById('algorithm-select').value = algorithm;

                this.currentPolicy = policy;
                this.currentAlgorithm = algorithm;

                this.updatePolicyDescription();
                this.updateAlgorithmDescription();
                this.showPolicySpecificParams();
                this.validateConfiguration();

                // 显示成功提示
                this.showValidationMessage(
                    true,
                    `✅ 已应用${preference === 'best' ? '最佳' : '快速'}推荐配置`,
                    'success'
                );
            }
        } catch (error) {
            console.error('应用推荐配置失败:', error);
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
            'a2c': 'Actor-Critic方法，方差小，收敛快，性能介于REINFORCE和PPO之间'
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
        // SymNCO/MDAM/DeepACO 使用内置训练算法，锁定算法下拉框
        const algoSelect = document.getElementById('algorithm-select');
        if (algoSelect) {
            const reinforce_only = ['symnco', 'mdam', 'deepaco'];
            algoSelect.disabled = reinforce_only.includes(this.currentPolicy);
            if (reinforce_only.includes(this.currentPolicy)) {
                algoSelect.value = 'reinforce';
                this.currentAlgorithm = 'reinforce';
            }
        }
    }

    showProblemSpecificParams() {
        // 显示/隐藏CVRP特有参数
        const cvrpParams = document.getElementById('cvrp-params');
        const vrptwParams = document.getElementById('vrptw-params');
        const mtspParams = document.getElementById('mtsp-params');
        const pdpParams = document.getElementById('pdp-params');
        const opParams = document.getElementById('op-params');

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

        // ── 统一的策略模型与算法约束 ──
        this.updatePolicyConstraints();
        this.updateAlgorithmConstraints();
    }

    updatePolicyConstraints() {
        const POLICY_PROBLEM_COMPAT = {
            'attention': ['tsp', 'atsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'op', 'pdp', 'pctsp', 'spctsp'],
            'pomo':      ['tsp', 'mtsp', 'cvrp'],
            'symnco':    ['tsp', 'mtsp', 'cvrp'],
            'ham':       ['pdp'],
            'matnet':    ['atsp', 'ffsp'],
            'ptrnet':    ['tsp', 'cvrp'],
            'mdam':      ['tsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'op', 'pdp', 'pctsp', 'spctsp'],
            'deepaco':   ['tsp', 'mtsp', 'cvrp', 'sdvrp', 'vrptw', 'op'],
        };

        const PROBLEM_NAMES = {
            'tsp': 'TSP', 'atsp': 'ATSP', 'mtsp': 'mTSP',
            'cvrp': 'CVRP', 'sdvrp': 'SDVRP', 'vrptw': 'VRPTW',
            'op': 'OP', 'pdp': 'PDP', 'pctsp': 'PCTSP',
            'spctsp': 'SPCTSP', 'ffsp': 'FFSP',
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
        const POLICY_ALGO_COMPAT = {
            'attention': ['reinforce', 'ppo', 'a2c'],
            'pomo':      ['reinforce', 'ppo', 'a2c'],
            'symnco':    ['reinforce'],
            'ham':       ['reinforce', 'ppo', 'a2c'],
            'matnet':    ['reinforce', 'ppo', 'a2c'],
            'ptrnet':    ['reinforce'],
            'mdam':      ['reinforce'],
            'deepaco':   ['reinforce'],
        };

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
        let hasValidSelection = false;
        let firstValidValue = null;

        Array.from(algoSelect.options).forEach(opt => {
            const isCompat = compatAlgos.includes(opt.value);

            opt.disabled = !isCompat;
            opt.title = isCompat ? '' : `${POLICY_NAMES[policy] || policy} 不支持此算法`;

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
            'a2c': 'A2C (平衡)'
        };
        return names[algorithm] || algorithm.toUpperCase();
    }
}

// 初始化兼容性管理器
let compatibilityManager;
document.addEventListener('DOMContentLoaded', () => {
    compatibilityManager = new CompatibilityManager();

    // 初始化数据集区域（根据默认问题类型）
    updateDatasetSection(document.getElementById('problem-select').value);

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
    if (!confirm('确定要退出登录吗？')) {
        return;
    }

    try {
        const response = await fetch('/api/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const result = await response.json();

        if (result.success) {
            alert('已退出登录');
            window.location.href = '/login';
        } else {
            alert('退出失败：' + result.message);
        }
    } catch (error) {
        console.error('退出登录错误:', error);
        alert('网络错误，请重试');
    }
}
