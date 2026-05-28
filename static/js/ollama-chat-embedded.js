/**
 * Ollama智能助手 - 嵌入式版本
 * RL4CO Display - 山西大学
 * 对话界面直接显示在首页中
 */

class OllamaChatEmbedded {
    constructor() {
        this.config = {
            apiUrl: '/api/ollama',   // 走 Flask 代理，绕过 CORS 和 CSP 限制
            defaultModel: 'llama2',
            storageKey: 'ollama_chat_history_embedded',
            maxHistoryLength: 50
        };

        this.state = {
            isConnected: false,
            currentModel: this.config.defaultModel,
            isTyping: false,
            messages: [],
            availableModels: []
        };

        this.elements = {};
        this.systemPrompt = this._buildSystemPrompt();
        this.init();
    }

    /**
     * 构建 system prompt，注入 RL4CO 领域知识
     */
    _buildSystemPrompt() {
        return `你是 RL4CO Display 平台的 AI 智能助手，专门帮助用户分析组合优化问题并配置强化学习训练参数。

## 你的核心能力
当用户描述一个实际问题或优化需求时，你需要：
1. 判断这属于哪种组合优化问题类型
2. 推荐最优的模型和算法组合
3. 配置合适的训练参数
4. 输出一个 JSON 配置块让系统自动填写表单

## 平台支持的问题类型

### 路由问题 (Routing)
- **TSP（旅行商问题）**：一个推销员要访问所有城市并回到起点，求最短路径。适用于：快递路线规划、电路板钻孔顺序、物流配送等。
- **ATSP（非对称旅行商问题）**：与TSP类似，但城市间的往返距离不同（如单行道、上下坡）。适用于：城市交通规划、单向道路网络。
- **mTSP（多旅行商问题）**：多个推销员协作访问所有城市。适用于：多配送员同时送货、多无人机协同巡检、多人任务分配。
- **CVRP（带容量约束的车辆路径问题）**：多辆有容量限制的车辆为客户送货。适用于：快递包裹配送、超市供货、垃圾回收。
- **SDVRP（分割配送VRP）**：允许一个客户的需求被多辆车分别配送。适用于：大宗货物配送、建材运输。
- **VRPTW（带时间窗VRP）**：配送有时间限制，必须在指定时间段到达。适用于：生鲜配送、外卖送餐、预约上门服务。
- **PDP（取送货问题）**：车辆需要先取货再送货，取送货必须配对。适用于：网约车接送乘客、快递揽收配送。
- **OP（定向问题）**：在路径长度限制内，选择访问哪些城市以最大化收益。适用于：旅游景点规划、销售拜访优先级。
- **PCTSP（奖励收集TSP）**：收集足够奖励的同时最小化路径和惩罚。适用于：广告投放路径优化。
- **SPCTSP（随机奖励收集TSP）**：PCTSP的随机版本，奖励具有不确定性。

### 调度问题 (Scheduling)
- **FFSP（柔性流水车间调度）**：多个工件需要经过多个加工阶段，每个阶段有多台机器可选。适用于：工厂生产排程、流水线优化。

## 支持的模型（策略网络）
| 模型 | 适用问题 | 特点 |
|------|---------|------|
| attention (AM) | 所有路由问题 | 通用性强，基于注意力机制 |
| pomo | TSP, mTSP, CVRP | 利用对称性，效果好，但仅适用对称问题 |
| symnco | TSP, mTSP, CVRP | 二面体8对称增强+多损失训练，质量最高，比POMO更好 |
| ptrnet | TSP, CVRP | 经典方法，适合学习研究 |
| matnet | ATSP, FFSP | 矩阵注意力，专为非对称/调度问题设计 |
| ham | PDP | 异构注意力，专为取送货问题设计 |

注意：symnco使用内置的自定义训练算法，选择symnco时algorithm字段填"reinforce"（实际使用SymNCO自定义多损失）。

## 支持的算法
| 算法 | 特点 |
|------|------|
| reinforce | 经典策略梯度，简单稳定 |
| ppo | 近端策略优化，复杂问题推荐 |
| a2c | 优势Actor-Critic，快速收敛 |

## 推荐组合（最佳配置）
- TSP → symnco + reinforce（最高质量），或 pomo + ppo（快速）
- ATSP → attention + ppo（POMO/SymNCO不支持非对称）
- mTSP → symnco + reinforce（最高质量），或 pomo + ppo（快速）
- CVRP → symnco + reinforce（最高质量），或 pomo + ppo（快速）
- SDVRP → attention + ppo
- VRPTW → attention + ppo
- PDP → ham + ppo
- OP → attention + ppo
- PCTSP/SPCTSP → attention + ppo
- FFSP → matnet + ppo（仅MatNet支持FFSP）

## 参数范围
- num_loc（问题规模）：5-200，小规模推荐20-30（快速），中等50，大规模100+
- epochs（训练轮数）：1-1000，快速验证3-5，常规10-20，充分训练50-100
- batch_size：32-2048，推荐512
- learning_rate：0.00001-0.01，推荐0.0001
- CVRP/SDVRP/VRPTW额外参数：vehicle_capacity（车辆容量，默认1.0）
- VRPTW额外参数：time_window_width（时间窗宽度，10-500），service_time（服务时间，0-60），max_processing_time（最大配送时间，60-1440），hard_time_windows（true/false）
- mTSP额外参数：num_agents（代理数量，2-10，推荐5），cost_type（minmax或sum）
- PDP额外参数：num_loc（偶数，4-40），force_start_at_depot（true/false）
- OP额外参数：num_loc（10-100），max_length（最大路径长度，1.0-5.0），prize_type（dist/unif/const）
- PCTSP/SPCTSP额外参数：num_loc（10-100），penalty_factor（惩罚因子，1-10），prize_required（需收集奖励比例，0.1-1.0）
- FFSP额外参数：num_stage（加工阶段数，2-6），num_machine（每阶段机器数，2-8），num_job（工件数，10-50），min_time（最小加工时间，1-5），max_time（最大加工时间，5-20），flatten_stages（true/false）
- SymNCO额外参数：num_augment（对称增强数量，1-10，推荐8），num_starts（多起点数量，0-50，默认0），symnco_alpha（不变性损失权重，默认0.2），symnco_beta（解对称损失权重，默认1.0）

## 输出格式要求（严格遵守）
当你分析出用户的问题类型并确定了配置后，**必须**在回答末尾输出一个 JSON 配置块。

格式规则：
1. 使用三个反引号（\`\`\`）开头和结尾，语言标签为 json
2. JSON 内所有 key 和字符串 value 必须用英文双引号包裹
3. 数字不加引号，小数必须写完整（如 0.0001，不能写 .0001）

示例（严格按此格式）：
\`\`\`json
{
  "problem": "tsp",
  "model": "symnco",
  "algorithm": "reinforce",
  "num_loc": 50,
  "epochs": 15,
  "batch_size": 512,
  "learning_rate": 0.0001
}
\`\`\`

## 注意事项
- 先用自然语言解释你的分析过程和推荐理由
- JSON 配置块放在回答的最后
- JSON 代码块必须格式正确，不能有缺引号、缺冒号等语法错误
- 如果用户的描述不够清晰，先询问关键信息再给出配置
- 如果用户只是闲聊或问知识性问题，正常回答即可，不需要输出 JSON
- 回复请使用中文`;
    }

    /**
     * 初始化
     */
    async init() {
        this.cacheElements();
        this.bindEvents();
        this.loadHistory();
        await this.checkConnection();
        await this.loadModels();
    }

    /**
     * 缓存DOM元素引用
     */
    cacheElements() {
        this.elements.messages = document.getElementById('ai-chat-messages');
        this.elements.input = document.getElementById('ai-chat-input');
        this.elements.sendBtn = document.getElementById('ai-send-btn');
        this.elements.modelSelect = document.getElementById('ai-model-select');
        this.elements.modelStatus = document.getElementById('ai-model-status');
        this.elements.clearBtn = document.getElementById('ai-clear-btn');
        this.elements.minimizeBtn = document.getElementById('ai-minimize-btn');
    }

    /**
     * 绑定事件
     */
    bindEvents() {
        // 清空对话
        if (this.elements.clearBtn) {
            this.elements.clearBtn.addEventListener('click', () => this.clearHistory());
        }

        // 最小化按钮（可选）
        if (this.elements.minimizeBtn) {
            this.elements.minimizeBtn.addEventListener('click', () => this.toggleMinimize());
        }

        // 发送按钮
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());

        // 输入框事件
        this.elements.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // 自动调整输入框高度
        this.elements.input.addEventListener('input', (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
        });

        // 模型选择
        this.elements.modelSelect.addEventListener('change', (e) => {
            this.state.currentModel = e.target.value;
            this.showNotification(`已切换到模型: ${e.target.value}`, 'success');
        });
    }

    /**
     * 最小化/展开面板（可选功能）
     */
    toggleMinimize() {
        const panel = document.querySelector('.ai-chat-panel-embedded');
        panel.classList.toggle('minimized');
    }

    /**
     * 检查Ollama连接状态
     */
    async checkConnection() {
        try {
            const response = await fetch(`${this.config.apiUrl}/tags`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                this.state.isConnected = true;
                this.updateConnectionStatus(true);
                return true;
            }
        } catch (error) {
            console.error('Ollama连接失败:', error);
            this.state.isConnected = false;
            this.updateConnectionStatus(false);
            return false;
        }
    }

    /**
     * 更新连接状态显示
     */
    updateConnectionStatus(connected) {
        const statusText = this.elements.modelStatus.querySelector('.status-text');
        
        if (connected) {
            this.elements.modelStatus.classList.add('connected');
            statusText.textContent = '已连接到Ollama';
        } else {
            this.elements.modelStatus.classList.remove('connected');
            statusText.textContent = 'Ollama未运行';
        }
    }

    /**
     * 加载可用模型列表
     */
    async loadModels() {
        if (!this.state.isConnected) {
            this.elements.modelSelect.innerHTML = '<option value="">请先启动Ollama</option>';
            return;
        }

        try {
            const response = await fetch(`${this.config.apiUrl}/tags`);
            const data = await response.json();

            if (data.models && data.models.length > 0) {
                this.state.availableModels = data.models;
                this.renderModelOptions();
            } else {
                this.elements.modelSelect.innerHTML = '<option value="">未找到模型</option>';
                this.showNotification('未找到可用模型，请先下载模型', 'warning');
            }
        } catch (error) {
            console.error('加载模型列表失败:', error);
            this.elements.modelSelect.innerHTML = '<option value="">加载失败</option>';
        }
    }

    /**
     * 渲染模型选项
     */
    renderModelOptions() {
        const options = this.state.availableModels.map(model => {
            const name = model.name || model;
            const selected = name === this.state.currentModel ? 'selected' : '';
            return `<option value="${name}" ${selected}>${name}</option>`;
        }).join('');

        this.elements.modelSelect.innerHTML = options;

        if (!this.state.availableModels.find(m => (m.name || m) === this.state.currentModel)) {
            this.state.currentModel = this.state.availableModels[0].name || this.state.availableModels[0];
            this.elements.modelSelect.value = this.state.currentModel;
        }
    }

    /**
     * 发送消息
     */
    async sendMessage() {
        const text = this.elements.input.value.trim();
        
        if (!text) return;
        if (!this.state.isConnected) {
            this.showNotification('请先启动Ollama服务', 'error');
            return;
        }
        if (this.state.isTyping) {
            this.showNotification('AI正在回复中，请稍候', 'warning');
            return;
        }

        // 添加用户消息
        this.addMessage('user', text);
        this.elements.input.value = '';
        this.elements.input.style.height = 'auto';

        // 显示加载动画
        this.showTypingIndicator();
        this.state.isTyping = true;
        this.elements.sendBtn.disabled = true;

        try {
            // 构建带 system prompt 的消息列表
            const chatMessages = [
                { role: 'system', content: this.systemPrompt }
            ];
            // 只取最近的对话历史（避免 token 过长）
            const recentMessages = this.state.messages.slice(-20);
            recentMessages.forEach(msg => {
                chatMessages.push({ role: msg.role, content: msg.content });
            });

            const response = await fetch(`${this.config.apiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: this.state.currentModel,
                    messages: chatMessages,
                    stream: true
                })
            });

            if (!response.ok) {
                throw new Error(`API错误: ${response.status}`);
            }

            await this.handleStreamResponse(response);

        } catch (error) {
            console.error('发送消息失败:', error);
            this.hideTypingIndicator();
            this.addMessage('assistant', `抱歉，发生了错误：${error.message}`);
        } finally {
            this.state.isTyping = false;
            this.elements.sendBtn.disabled = false;
        }
    }

    /**
     * 处理流式响应
     */
    async handleStreamResponse(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let assistantMessage = '';
        let messageElement = null;

        this.hideTypingIndicator();

        try {
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n').filter(line => line.trim());

                for (const line of lines) {
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.message && data.message.content) {
                            assistantMessage += data.message.content;
                            
                            if (!messageElement) {
                                messageElement = this.addMessage('assistant', assistantMessage, false);
                            } else {
                                this.updateMessageContent(messageElement, assistantMessage);
                            }
                        }

                        if (data.done) {
                            break;
                        }
                    } catch (e) {
                        console.error('解析响应失败:', e);
                    }
                }
            }

            if (assistantMessage) {
                this.state.messages.push({
                    role: 'assistant',
                    content: assistantMessage,
                    timestamp: Date.now()
                });
                this.saveHistory();

                // 先做一次最终的内容渲染（确保配置 JSON 被隐藏）
                if (messageElement) {
                    this.updateMessageContent(messageElement, assistantMessage);
                }

                // 检测是否包含训练配置块，渲染应用按钮
                const configData = this._extractConfig(assistantMessage);
                if (configData && messageElement) {
                    this._renderApplyButton(messageElement, configData);
                }
            }

        } catch (error) {
            console.error('处理流式响应失败:', error);
            throw error;
        }
    }

    /**
     * 从 LLM 回复中提取训练配置 JSON 块
     * 支持多种格式：```rl4co_config、```json、```、甚至裸 JSON
     */
    _extractConfig(text) {
        // 先从主响应（剥离思考块后）提取，优先级最高
        const mainText = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
        const fromMain = this._extractConfigFromText(mainText);
        if (fromMain) return fromMain;

        // 主响应无配置时，从 <think> 块中提取
        // 针对 deepseek-r1 等推理模型把配置写在思考过程中的情况
        const thinkMatch = text.match(/<think>([\s\S]*?)<\/think>/i);
        if (thinkMatch) {
            return this._extractConfigFromText(thinkMatch[1]);
        }

        return null;
    }

    _extractConfigFromText(text) {
        // 支持 2~3 个反引号、任意语言标签（json / rl4co_config / rl4_config 等）
        const codeBlockRegex = /`{2,3}(\w*)\s*\n([\s\S]*?)`{2,3}/g;
        let match;
        while ((match = codeBlockRegex.exec(text)) !== null) {
            const config = this._parseJsonLoose(match[2].trim());
            if (config && config.problem && (config.model || config.algorithm)) {
                return config;
            }
        }

        // 兜底：尝试匹配裸 JSON 对象（没有代码块包裹）
        const jsonRegex = /\{[^{}]*(?:"problem"|problem)[^{}]*\}/g;
        let jsonMatch;
        while ((jsonMatch = jsonRegex.exec(text)) !== null) {
            const config = this._parseJsonLoose(jsonMatch[0]);
            if (config && config.problem && (config.model || config.algorithm)) {
                return config;
            }
        }

        return null;
    }

    /**
     * 宽容 JSON 解析：先尝试标准解析，失败时做常见 LLM 格式修复再试
     */
    _parseJsonLoose(raw) {
        // 1. 标准解析
        try { return JSON.parse(raw); } catch (_) {}

        // 2. 修复常见 LLM 格式问题后再试
        let s = raw;

        // 修复 key 缺少开头引号：  num_loc": 50  →  "num_loc": 50
        s = s.replace(/([{,]\s*)([a-zA-Z_]\w*)(\s*"?\s*:)/g, '$1"$2"$3');

        // 修复 value 缺少开头引号：  "key":word"  →  "key":"word"
        s = s.replace(/:\s*([a-zA-Z_]\w*)"(\s*[,}])/g, ': "$1"$2');

        // 修复缺少冒号：  "key" "value"  →  "key": "value"
        s = s.replace(/"(\s+)"/g, '": "');

        // 修复小数缺少前导零：  .0001  →  0.0001
        s = s.replace(/:\s*\.(\d)/g, ': 0.$1');

        // 修复末尾缺少闭合引号：  "key": "val\n  →  "key": "val"
        s = s.replace(/"([^"\n]+)(\n\s*["}])/g, '"$1"$2');

        // 修复重复冒号：  ": :  →  :
        s = s.replace(/:\s*:/g, ':');

        try { return JSON.parse(s); } catch (_) {}

        // 3. 最后兜底：正则逐字段提取，容忍结构性残缺
        return this._extractKeyValues(raw);
    }

    /**
     * 从残缺 JSON 文本里逐字段抽取 key-value
     */
    _extractKeyValues(text) {
        const VALID_KEYS = new Set([
            'problem', 'model', 'algorithm', 'num_loc', 'epochs',
            'batch_size', 'learning_rate', 'vehicle_capacity', 'num_agents',
            'cost_type', 'time_window_width', 'service_time', 'max_processing_time',
            'hard_time_windows', 'max_length', 'prize_type', 'penalty_factor',
            'prize_required', 'num_stage', 'num_machine', 'num_job',
        ]);
        const config = {};
        // 匹配 "key": "value"  或  "key": number
        const re = /"?(\w+)"?\s*[":]\s*"?([^",\n}]+)"?/g;
        let m;
        while ((m = re.exec(text)) !== null) {
            const key = m[1].trim();
            if (!VALID_KEYS.has(key)) continue;
            const rawVal = m[2].trim().replace(/^["']|["']$/g, '');
            config[key] = isNaN(rawVal) ? rawVal : Number(rawVal);
        }
        return Object.keys(config).length > 0 ? config : null;
    }

    /**
     * 在消息气泡下方渲染"应用配置"按钮
     */
    _renderApplyButton(messageElement, configData) {
        const bubble = messageElement.querySelector('.ai-message-bubble');
        if (!bubble) return;

        const configSummary = this._buildConfigSummary(configData);

        const container = document.createElement('div');
        container.className = 'ai-config-apply-container';
        container.innerHTML = `
            <div class="ai-config-preview">
                <div class="ai-config-preview-title">🎯 识别到训练配置</div>
                <div class="ai-config-preview-items">${configSummary}</div>
            </div>
            <button class="ai-config-apply-btn" title="将此配置应用到训练表单">
                ⚡ 应用配置到表单
            </button>
        `;

        const applyBtn = container.querySelector('.ai-config-apply-btn');
        applyBtn.addEventListener('click', () => {
            this.applyConfiguration(configData);
            applyBtn.textContent = '✅ 配置已应用';
            applyBtn.classList.add('applied');
            applyBtn.disabled = true;
        });

        bubble.appendChild(container);
        this.scrollToBottom();
    }

    /**
     * 构建配置摘要 HTML
     */
    _buildConfigSummary(config) {
        const labels = {
            problem: '问题类型', model: '模型', algorithm: '算法',
            num_loc: '问题规模', epochs: '训练轮数', batch_size: '批次大小',
            learning_rate: '学习率', vehicle_capacity: '车辆容量',
            num_agents: '代理数量', cost_type: '优化目标',
            time_window_width: '时间窗宽度', service_time: '服务时间',
            max_processing_time: '最大配送时间', hard_time_windows: '硬时间窗',
            num_stage: '加工阶段', num_machine: '机器数', num_job: '工件数',
            min_time: '最小加工时间', max_time: '最大加工时间',
            max_length: '最大路径长度', prize_type: '奖励类型',
            penalty_factor: '惩罚因子', prize_required: '需收集奖励',
            flatten_stages: '展平阶段', force_start_at_depot: '从depot出发',
            num_augment: '对称增强数量', num_starts: '多起点数量',
            symnco_alpha: '不变性损失权重α', symnco_beta: '解对称损失权重β'
        };

        const problemNames = {
            tsp: 'TSP', atsp: 'ATSP', mtsp: 'mTSP', cvrp: 'CVRP',
            sdvrp: 'SDVRP', vrptw: 'VRPTW', pdp: 'PDP', op: 'OP',
            pctsp: 'PCTSP', spctsp: 'SPCTSP', ffsp: 'FFSP'
        };

        const modelNames = {
            attention: 'Attention Model', pomo: 'POMO', symnco: 'SymNCO',
            ptrnet: 'PtrNet', matnet: 'MatNet', ham: 'HAM'
        };

        const algoNames = {
            reinforce: 'REINFORCE', ppo: 'PPO', a2c: 'A2C'
        };

        let html = '';
        for (const [key, value] of Object.entries(config)) {
            const label = labels[key] || key;
            let displayValue = value;
            if (key === 'problem') displayValue = problemNames[value] || value;
            else if (key === 'model') displayValue = modelNames[value] || value;
            else if (key === 'algorithm') displayValue = algoNames[value] || value;

            html += `<span class="ai-config-tag"><b>${label}:</b> ${displayValue}</span>`;
        }
        return html;
    }

    /**
     * 规范化 LLM 输出的配置值，统一为表单 option 要求的格式
     */
    _normalizeConfig(raw) {
        const config = { ...raw };

        // 问题类型：统一小写
        if (config.problem) {
            config.problem = String(config.problem).toLowerCase().trim();
        }

        // 模型名：统一小写 + 别名映射
        if (config.model) {
            const modelMap = {
                'am': 'attention', 'attention_model': 'attention', 'attentionmodel': 'attention',
                'attention model': 'attention', 'pointer_network': 'ptrnet', 'pointernetwork': 'ptrnet',
                'pointer network': 'ptrnet', 'ptr': 'ptrnet',
                'sym_nco': 'symnco', 'sym-nco': 'symnco', 'symnco policy': 'symnco',
                'symmetric nco': 'symnco', 'symmetric_nco': 'symnco',
            };
            let m = String(config.model).toLowerCase().trim();
            config.model = modelMap[m] || m;
        }

        // 算法名：统一小写
        if (config.algorithm) {
            config.algorithm = String(config.algorithm).toLowerCase().trim();
        }

        // 奖励类型
        if (config.prize_type) {
            config.prize_type = String(config.prize_type).toLowerCase().trim();
        }

        // 优化目标
        if (config.cost_type) {
            config.cost_type = String(config.cost_type).toLowerCase().trim();
        }

        return config;
    }

    /**
     * 将 AI 推荐的配置应用到训练表单
     */
    applyConfiguration(rawConfig) {
        const config = this._normalizeConfig(rawConfig);

        const setSelect = (id, val) => {
            const el = document.getElementById(id);
            if (!el || val === undefined || val === null) return;
            // 确认 option 存在再赋值；disabled option 临时启用确保赋值成功
            const option = Array.from(el.options).find(o => o.value === String(val));
            if (option) {
                const wasDisabled = option.disabled;
                option.disabled = false;
                el.value = option.value;
                option.disabled = wasDisabled;
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
        };

        const setInput = (id, val) => {
            const el = document.getElementById(id);
            if (el && val !== undefined && val !== null) {
                el.value = val;
                el.dispatchEvent(new Event('input', { bubbles: true }));
            }
        };

        // 第一步：设置问题类型（会触发面板切换、可用模型/算法更新）
        if (config.problem) {
            setSelect('problem-select', config.problem);
        }

        // 延迟设置其他参数，等问题类型变更引发的异步操作完成
        setTimeout(() => {
            if (config.model) setSelect('model-select', config.model);
            if (config.algorithm) setSelect('algorithm-select', config.algorithm);
            if (config.num_loc) setInput('num-loc', config.num_loc);
            if (config.epochs) setInput('epochs', config.epochs);
            if (config.batch_size) setInput('batch-size', config.batch_size);
            if (config.learning_rate) setInput('learning-rate', config.learning_rate);

            // CVRP/SDVRP/VRPTW 参数
            if (config.vehicle_capacity) setInput('vehicle-capacity', config.vehicle_capacity);

            // VRPTW 参数
            if (config.time_window_width) setInput('time-window-width', config.time_window_width);
            if (config.service_time) setInput('service-time', config.service_time);
            if (config.max_processing_time) setInput('max-time', config.max_processing_time);
            if (config.hard_time_windows !== undefined) setSelect('hard-time-windows', String(config.hard_time_windows));

            // mTSP 参数
            if (config.num_agents) setInput('num-agents', config.num_agents);
            if (config.cost_type) setSelect('cost-type', config.cost_type);

            // PDP 参数
            if (config.problem === 'pdp' && config.num_loc) setInput('pdp-num-loc', config.num_loc);
            if (config.force_start_at_depot !== undefined) setSelect('force-start-depot', String(config.force_start_at_depot));

            // OP 参数
            if (config.problem === 'op' && config.num_loc) setInput('op-num-loc', config.num_loc);
            if (config.max_length) setInput('max-length', config.max_length);
            if (config.prize_type) setSelect('prize-type', config.prize_type);

            // PCTSP/SPCTSP 参数
            if (config.problem === 'pctsp' && config.num_loc) setInput('pctsp-num-loc', config.num_loc);
            if (config.problem === 'spctsp' && config.num_loc) setInput('spctsp-num-loc', config.num_loc);
            if (config.penalty_factor) setInput('penalty-factor', config.penalty_factor);
            if (config.prize_required) setInput('prize-required', config.prize_required);

            // FFSP 参数
            if (config.num_stage) setInput('num-stage', config.num_stage);
            if (config.num_machine) setInput('num-machine', config.num_machine);
            if (config.num_job) setInput('num-job', config.num_job);
            if (config.min_time) setInput('min-time', config.min_time);
            if (config.max_time) setInput('max-time-ffsp', config.max_time);
            if (config.flatten_stages !== undefined) setSelect('flatten-stages', String(config.flatten_stages));

            // POMO 参数
            if (config.num_starts) setInput('num-starts', config.num_starts);

            // SymNCO 参数
            if (config.num_augment) setInput('num-augment', config.num_augment);
            if (config.num_starts && config.model === 'symnco') setInput('symnco-num-starts', config.num_starts);
            if (config.symnco_alpha !== undefined) setInput('symnco-alpha', config.symnco_alpha);
            if (config.symnco_beta !== undefined) setInput('symnco-beta', config.symnco_beta);

            // 滚动到配置区域
            const configSection = document.getElementById('config');
            if (configSection) {
                configSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }

            this.showNotification('✅ AI 推荐配置已应用到表单！', 'success');
        }, 500);
    }

    /**
     * 添加消息到界面
     */
    addMessage(role, content, saveToHistory = true) {
        const emptyState = this.elements.messages.querySelector('.ai-chat-empty');
        if (emptyState) {
            emptyState.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `ai-message ${role}`;
        
        const avatar = role === 'user' ? '👤' : '🤖';
        
        messageDiv.innerHTML = `
            <div class="ai-message-avatar">${avatar}</div>
            <div class="ai-message-content">
                <div class="ai-message-bubble">${this.formatMessage(content)}</div>
                <div class="ai-message-actions">
                    <button class="ai-message-action-btn copy-btn" title="复制">📋 复制</button>
                    ${role === 'assistant' ? '<button class="ai-message-action-btn regen-btn" title="重新生成">🔄 重新生成</button>' : ''}
                </div>
            </div>
        `;

        this.elements.messages.appendChild(messageDiv);

        const copyBtn = messageDiv.querySelector('.copy-btn');
        copyBtn.addEventListener('click', () => this.copyMessage(content));

        if (role === 'assistant') {
            const regenBtn = messageDiv.querySelector('.regen-btn');
            regenBtn.addEventListener('click', () => this.regenerateResponse());
        }

        if (saveToHistory) {
            this.state.messages.push({
                role: role,
                content: content,
                timestamp: Date.now()
            });
            this.saveHistory();
        }

        this.scrollToBottom();

        return messageDiv;
    }

    /**
     * 更新消息内容
     */
    updateMessageContent(messageElement, content) {
        const bubble = messageElement.querySelector('.ai-message-bubble');
        if (bubble) {
            bubble.innerHTML = this.formatMessage(content);
            this.scrollToBottom();
        }
    }

    /**
     * 格式化消息内容
     */
    formatMessage(content) {
        // 剥离 <think>...</think> 推理块（deepseek-r1 等推理模型的内部思考过程，不展示给用户）
        let processed = content.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();

        // 移除包含训练配置的代码块（配置会通过专用按钮呈现），支持 2~3 个反引号
        let cleaned = processed.replace(/`{2,3}(\w*)\s*\n([\s\S]*?)`{2,3}/g, (match, lang, code) => {
            const parsed = this._parseJsonLoose(code.trim());
            if (parsed && parsed.problem && (parsed.model || parsed.algorithm)) {
                return '';
            }
            return match;
        }).trim();

        let formatted = cleaned
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            const language = lang || 'plaintext';
            return `<pre><code class="language-${language}">${code.trim()}</code></pre>`;
        });

        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
        formatted = formatted.replace(/\n/g, '<br>');
        formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');

        return formatted;
    }

    /**
     * 显示打字指示器
     */
    showTypingIndicator() {
        const emptyState = this.elements.messages.querySelector('.ai-chat-empty');
        if (emptyState) {
            emptyState.remove();
        }

        const indicator = document.createElement('div');
        indicator.className = 'ai-typing-indicator';
        indicator.id = 'ai-typing-indicator';
        indicator.innerHTML = `
            <div class="ai-message-avatar">🤖</div>
            <div class="ai-typing-dots">
                <span class="ai-typing-dot"></span>
                <span class="ai-typing-dot"></span>
                <span class="ai-typing-dot"></span>
            </div>
        `;

        this.elements.messages.appendChild(indicator);
        this.scrollToBottom();
    }

    /**
     * 隐藏打字指示器
     */
    hideTypingIndicator() {
        const indicator = document.getElementById('ai-typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    /**
     * 复制消息
     */
    async copyMessage(content) {
        try {
            await navigator.clipboard.writeText(content);
            this.showNotification('已复制到剪贴板', 'success');
        } catch (error) {
            console.error('复制失败:', error);
            this.showNotification('复制失败', 'error');
        }
    }

    /**
     * 重新生成回复
     */
    async regenerateResponse() {
        if (this.state.messages.length > 0 && 
            this.state.messages[this.state.messages.length - 1].role === 'assistant') {
            this.state.messages.pop();
            
            const messages = this.elements.messages.querySelectorAll('.ai-message.assistant');
            if (messages.length > 0) {
                messages[messages.length - 1].remove();
            }
        }

        if (this.state.messages.length > 0) {
            const lastUserMessage = [...this.state.messages]
                .reverse()
                .find(msg => msg.role === 'user');
            
            if (lastUserMessage) {
                this.elements.input.value = lastUserMessage.content;
                await this.sendMessage();
            }
        }
    }

    /**
     * 清空对话历史
     */
    clearHistory() {
        if (!confirm('确定要清空所有对话记录吗？')) {
            return;
        }

        this.state.messages = [];
        this.saveHistory();

        this.elements.messages.innerHTML = `
            <div class="ai-chat-empty">
                <div class="ai-chat-empty-icon">💬</div>
                <div class="ai-chat-empty-text">有什么可以帮助您的？</div>
                <div class="ai-chat-empty-hint">试试问我关于RL4CO的问题</div>
            </div>
        `;

        this.showNotification('对话记录已清空', 'success');
    }

    /**
     * 保存历史记录
     */
    saveHistory() {
        try {
            const recentMessages = this.state.messages.slice(-this.config.maxHistoryLength);
            localStorage.setItem(this.config.storageKey, JSON.stringify(recentMessages));
        } catch (error) {
            console.error('保存历史记录失败:', error);
        }
    }

    /**
     * 加载历史记录
     */
    loadHistory() {
        try {
            const saved = localStorage.getItem(this.config.storageKey);
            if (saved) {
                this.state.messages = JSON.parse(saved);
                this.renderHistory();
            }
        } catch (error) {
            console.error('加载历史记录失败:', error);
        }
    }

    /**
     * 渲染历史记录
     */
    renderHistory() {
        if (this.state.messages.length === 0) return;

        const emptyState = this.elements.messages.querySelector('.ai-chat-empty');
        if (emptyState) {
            emptyState.remove();
        }

        this.state.messages.forEach(msg => {
            const el = this.addMessage(msg.role, msg.content, false);
            // 为历史消息中的配置块也渲染应用按钮
            if (msg.role === 'assistant') {
                const configData = this._extractConfig(msg.content);
                if (configData && el) {
                    this._renderApplyButton(el, configData);
                }
            }
        });
    }

    /**
     * 滚动到底部
     */
    scrollToBottom() {
        requestAnimationFrame(() => {
            this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
        });
    }

    /**
     * 显示通知
     */
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 2rem;
            padding: 1rem 1.5rem;
            background: ${type === 'error' ? '#f8d7da' : type === 'warning' ? '#fff3cd' : '#d4edda'};
            color: ${type === 'error' ? '#721c24' : type === 'warning' ? '#856404' : '#155724'};
            border: 1px solid ${type === 'error' ? '#f5c6cb' : type === 'warning' ? '#ffeaa7' : '#c3e6cb'};
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            z-index: 10000;
            animation: slideInRight 0.3s ease;
        `;
        notification.textContent = message;

        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    window.ollamaChatEmbedded = new OllamaChatEmbedded();
});

// 添加动画CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(100px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes slideOutRight {
        from { opacity: 1; transform: translateX(0); }
        to { opacity: 0; transform: translateX(100px); }
    }
`;
document.head.appendChild(style);

