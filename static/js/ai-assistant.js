(() => {
  'use strict';
  const layer = document.getElementById('assistant-layer');
  let closeLayer;
  function context() {
    const config =
      currentSessionId && monitorContext.config
        ? monitorContext.config
        : buildTrainingConfig();
    return {
      config,
      session: currentSessionId,
      metrics: lastStatus,
      mode: monitorContext.mode,
      compatibility:
        typeof compatibilityManager === 'undefined'
          ? {}
          : {
              policy_problem: compatibilityManager?.POLICY_PROBLEM_COMPAT,
              policy_algorithm: compatibilityManager?.POLICY_ALGO_COMPAT,
              algorithm_problem: compatibilityManager?.ALGO_PROBLEM_COMPAT,
            },
    };
  }
  window.getAssistantContext = context;
  function updateContext() {
    const data = context();
    document.getElementById('assistant-context').textContent =
      data.config.problem.toUpperCase() +
      ' · ' +
      data.config.model.toUpperCase() +
      ' · ' +
      getTrainingAlgorithmDisplayName(data.config.algorithm) +
      (data.session
        ? '\nSession ' +
          data.session +
          ' · Epoch ' +
          (data.metrics.epoch ?? '—') +
          ' · Reward ' +
          Studio.number(data.metrics.reward) +
          ' · Loss ' +
          Studio.number(data.metrics.loss)
        : ' · 尚未运行');
  }
  const close = () => {
    closeLayer?.();
    closeLayer = null;
  };
  document.querySelectorAll('[data-open-assistant]').forEach(
    (button) =>
      (button.onclick = () => {
        if (!closeLayer) closeLayer = Studio.openLayer(layer, close);
        updateContext();
      }),
  );
  document.getElementById('assistant-close').onclick = close;
  layer.onclick = (event) => {
    if (event.target === layer) close();
  };
  window.addEventListener('studio:training', updateContext);
  document
    .getElementById('training-form')
    .addEventListener('change', updateContext);
  document.querySelectorAll('[data-prompt]').forEach(
    (button) =>
      (button.onclick = () => {
        document.getElementById('ai-chat-input').value = button.dataset.prompt;
        window.ollamaChatEmbedded?.sendMessage();
      }),
  );
  document.getElementById('assistant-retry').onclick = async () => {
    const chat = window.ollamaChatEmbedded;
    if (!chat) return;
    const button = document.getElementById('assistant-retry');
    Studio.busy(button, true);
    try {
      await chat.loadProviders();
      await chat.handleSourceChange(chat.state.source);
    } finally {
      Studio.busy(button, false);
    }
  };
  document.getElementById('assistant-cancel').onclick = () =>
    window.ollamaChatEmbedded?.abortController?.abort();
  document.getElementById('assistant-message-retry').onclick = () => {
    document.getElementById('assistant-message-retry').hidden = true;
    window.ollamaChatEmbedded?.regenerateResponse();
  };
  const prototype = OllamaChatEmbedded.prototype;
  prototype.showTypingIndicator = function () {
    this.hideTypingIndicator();
    const node = document.createElement('div');
    node.id = 'ai-typing-indicator';
    node.className = 'ai-typing-indicator';
    node.setAttribute('role', 'status');
    node.textContent = '助手正在生成，请稍候…';
    this.elements.messages.append(node);
    this.scrollToBottom();
  };
  prototype.showNotification = (message, type) =>
    Studio.toast(String(message).replace(/[\u2600-\u27BF\uFE0F]/g, ''), type);
  prototype._buildSystemPrompt = function () {
    return '你是 RL4CO Studio 的研究助手，使用中文解释组合优化、Loss 与训练参数。所有推荐必须遵守随请求提供的真实兼容性矩阵。未知指标请明确说明，不要编造 GPU、训练产物或性能。用户请求配置建议时可输出一个带英文双引号的 JSON 代码块，包含 problem、model、algorithm 以及当前表单支持的参数。系统会显示应用配置按钮，由用户应用；不要声称已启动训练。';
  };
  prototype.applyConfiguration = async function (rawConfig) {
    if (window.trainingActive)
      return Studio.toast('训练进行中，结束后再应用新配置。', 'warning');
    const config = this._normalizeConfig(rawConfig),
      manager = compatibilityManager;
    const problem = document.getElementById('problem-select');
    if (
      config.problem &&
      [...problem.options].some((option) => option.value === config.problem)
    ) {
      problem.value = config.problem;
      manager.currentProblem = problem.value;
      updateDatasetSection(problem.value);
      manager.showProblemSpecificParams();
      manager.updatePolicyConstraints();
    }
    for (const [key, id] of [
      ['model', 'model-select'],
      ['algorithm', 'algorithm-select'],
    ]) {
      if (
        config[key] !== undefined &&
        [...document.getElementById(id).options].some(
          (option) => option.value === config[key],
        )
      )
        document.getElementById(id).value = config[key];
    }
    manager.currentPolicy = document.getElementById('model-select').value;
    manager.currentAlgorithm =
      document.getElementById('algorithm-select').value;
    manager.showPolicySpecificParams();
    manager.updatePolicyDescription();
    manager.updateAlgorithmDescription();
    manager.updateAlgorithmConstraints();
    const sizeId =
      {
        pdp: 'pdp-num-loc',
        op: 'op-num-loc',
        pctsp: 'pctsp-num-loc',
        spctsp: 'spctsp-num-loc',
      }[problem.value] || 'num-loc';
    if (config.num_loc != null)
      document.getElementById(sizeId).value = config.num_loc;
    for (const input of document
      .getElementById('training-form')
      .querySelectorAll('input[name],select[name]')) {
      if (
        ['problem', 'model', 'algorithm', 'num_loc', 'gpu_select'].includes(
          input.name,
        )
      )
        continue;
      if (config[input.name] === undefined) continue;
      const conditional = input.closest('.conditional-params');
      if (conditional && conditional.style.display === 'none') continue;
      if (input.type === 'checkbox')
        input.checked = Boolean(config[input.name]);
      else input.value = String(config[input.name]);
    }
    if (problem.value === 'spctsp') {
      for (const [key, id] of [
        ['penalty_factor', 'spctsp-penalty-factor'],
        ['prize_required', 'spctsp-prize-required'],
      ])
        if (config[key] != null)
          document.getElementById(id).value = config[key];
    }
    if (config.force_start_at_depot != null)
      document.getElementById('force-start-depot').value = String(
        config.force_start_at_depot,
      );
    await manager.validateConfiguration();
    setWorkspaceView('config');
    close();
    Studio.toast('已应用建议配置，请检查字段与兼容性。', 'success');
  };
  prototype.handleStreamResponse = async function (response) {
    const reader = response.body.getReader(),
      decoder = new TextDecoder();
    let buffer = '',
      text = '',
      element,
      completed = false;
    const process = (line) => {
      if (!line.trim()) return;
      let data;
      try {
        data = JSON.parse(line);
      } catch {
        throw new Error('流式响应格式无效，请重试。');
      }
      if (data.error) throw new Error(data.error);
      if (data.message?.content) {
        this.hideTypingIndicator();
        text += data.message.content;
        if (!element) element = this.addMessage('assistant', text, false);
        else this.updateMessageContent(element, text);
      }
      if (data.done) completed = true;
    };
    try {
      while (!completed) {
        const chunk = await reader.read();
        buffer += chunk.done
          ? decoder.decode()
          : decoder.decode(chunk.value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        lines.forEach(process);
        if (chunk.done) {
          if (buffer.trim()) process(buffer);
          break;
        }
      }
      if (!completed)
        throw new Error('输出连接中断，已保留收到的内容，可重试。');
    } finally {
      this.hideTypingIndicator();
      await reader.cancel().catch(() => {});
      if (text) {
        this.state.messages.push({
          role: 'assistant',
          content: text,
          timestamp: Date.now(),
        });
        this.saveHistory();
        if (element) {
          this.updateMessageContent(element, text);
          const config = this._extractConfig(text);
          if (config && completed) this._renderApplyButton(element, config);
        }
      }
    }
  };
  prototype.clearHistory = async function () {
    if (this.state.isTyping)
      return Studio.toast('请等待当前回复结束。', 'warning');
    const result = await Studio.confirmAction({
      title: '清空助手对话？',
      description:
        '将清空当前账户在此浏览器中的助手历史。训练 Session 不受影响。',
      label: '清空对话',
      action: () => {
        this.state.messages = [];
        this.saveHistory();
        this.elements.messages.innerHTML =
          '<div class="empty-state"><h3>从当前实验开始</h3><p>向助手提问以获取分析。</p></div>';
      },
    });
    if (result) Studio.toast('对话已清空', 'success');
  };
  window.addEventListener('pagehide', () =>
    window.ollamaChatEmbedded?.abortController?.abort(),
  );
})();
