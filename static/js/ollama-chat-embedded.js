/**
 * Ollama智能助手 - 嵌入式版本
 * RL4CO Display - 山西大学
 * 对话界面直接显示在首页中
 */

class OllamaChatEmbedded {
    constructor() {
        this.config = {
            apiUrl: 'http://localhost:11434/api',
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
        this.init();
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
            const messages = this.state.messages.map(msg => ({
                role: msg.role,
                content: msg.content
            }));

            const response = await fetch(`${this.config.apiUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: this.state.currentModel,
                    messages: messages,
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
            }

        } catch (error) {
            console.error('处理流式响应失败:', error);
            throw error;
        }
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
        let formatted = content
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
            this.addMessage(msg.role, msg.content, false);
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

