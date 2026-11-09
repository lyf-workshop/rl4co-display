(() => {
  'use strict';
  let datasets = [],
    sessions = [],
    files = [],
    loading = false;
  const roots = {
    datasets: document.getElementById('datasets-table'),
    results: document.getElementById('results-table'),
    checkpoints: document.getElementById('checkpoints-table'),
  };
  const bytes = (value) =>
    value == null
      ? '—'
      : value >= 1048576
        ? (value / 1048576).toFixed(1) + ' MB'
        : (value / 1024).toFixed(1) + ' KB';
  function switchTab(key) {
    document.querySelectorAll('[data-tab]').forEach((button) => {
      const active = button.dataset.tab === key;
      button.setAttribute('aria-selected', String(active));
      button.tabIndex = active ? 0 : -1;
      document.getElementById('panel-' + button.dataset.tab).hidden = !active;
    });
  }
  const tabs = [...document.querySelectorAll('[data-tab]')];
  tabs.forEach((button, index) => {
    button.onclick = () => switchTab(button.dataset.tab);
    button.onkeydown = (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key))
        return;
      event.preventDefault();
      const next =
        event.key === 'Home'
          ? 0
          : event.key === 'End'
            ? tabs.length - 1
            : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) %
              tabs.length;
      tabs[next].click();
      tabs[next].focus();
    };
  });
  async function load() {
    if (loading) return;
    loading = true;
    document.getElementById('files-clear').disabled = true;
    Object.values(roots).forEach((root) =>
      Studio.state(root, 'loading', '正在读取文件…'),
    );
    const values = await Promise.allSettled([
      Studio.request('/api/list_datasets'),
      Studio.request('/api/list_files'),
      Studio.request('/api/training_sessions'),
    ]);
    if (values[0].status === 'fulfilled') {
      datasets = values[0].value.datasets;
      renderDatasets();
      document.getElementById('dataset-count').textContent = datasets.length;
    } else
      Studio.state(roots.datasets, 'error', values[0].reason.message, load);
    if (values[1].status === 'fulfilled') {
      const data = values[1].value;
      sessions = data.sessions;
      if (values[2].status === 'fulfilled') {
        const states = new Map(
          values[2].value.sessions.map((row) => [row.session_id, row]),
        );
        sessions.forEach((session) => {
          const state = states.get(session.session_id);
          if (state) session.status = state.live ? state.status : 'inactive';
        });
      }
      files = sessions.flatMap((session) =>
        (
          session.all_files ||
          Object.values(session.files).flat().filter(Boolean)
        ).map((file) => ({ ...file, session })),
      );
      document.getElementById('result-count').textContent = files.filter(
        (file) => file.type !== 'checkpoint',
      ).length;
      document.getElementById('checkpoint-count').textContent = files.filter(
        (file) => file.type === 'checkpoint',
      ).length;
      document.getElementById('storage-count').textContent =
        values[0].status === 'fulfilled'
          ? bytes(
              Number(data.total_size_mb || 0) * 1048576 +
                datasets.reduce((sum, item) => sum + (item.size || 0), 0),
            )
          : '—';
      renderResults();
      renderCheckpoints();
      document.getElementById('files-clear').disabled =
        !files.length ||
        sessions.some((session) =>
          ['starting', 'running', 'paused', 'pausing', 'stopping'].includes(
            session.status,
          ),
        );
    } else {
      Studio.state(roots.results, 'error', values[1].reason.message, load);
      Studio.state(roots.checkpoints, 'error', values[1].reason.message, load);
    }
    loading = false;
  }
  const table = (head, body) =>
    '<table class="data-table"><thead><tr>' +
    head.map((label) => '<th scope="col">' + label + '</th>').join('') +
    '</tr></thead><tbody>' +
    body +
    '</tbody></table>';
  function renderDatasets() {
    if (!datasets.length)
      return Studio.state(
        roots.datasets,
        'empty',
        '还没有数据集，上传一个文件开始实验。',
      );
    roots.datasets.innerHTML = table(
      ['数据集', '问题类型', '规模', '大小', '上传时间', '操作'],
      datasets
        .map(
          (item, index) =>
            '<tr><td class="file-name">' +
            Studio.escape(item.filename) +
            '</td><td><span class="badge">' +
            Studio.escape(item.problem_type.toUpperCase()) +
            '</span></td><td class="data">' +
            item.num_loc +
            '</td><td class="data">' +
            bytes(item.size) +
            '</td><td class="data">' +
            Studio.escape(
              item.upload_time
                ? new Date(item.upload_time).toLocaleString()
                : '—',
            ) +
            '</td><td><div class="actions"><a class="btn btn-small" href="/?problem=' +
            encodeURIComponent(item.problem_type) +
            '&dataset=' +
            encodeURIComponent(item.dataset_id) +
            '">使用</a><a class="btn btn-small" href="/api/download_dataset/' +
            encodeURIComponent(item.dataset_id) +
            '">下载</a><button type="button" class="btn btn-small btn-danger" data-delete-dataset="' +
            index +
            '">删除</button></div></td></tr>',
        )
        .join(''),
    );
    roots.datasets
      .querySelectorAll('[data-delete-dataset]')
      .forEach(
        (button) =>
          (button.onclick = () =>
            deleteDataset(datasets[Number(button.dataset.deleteDataset)])),
      );
  }
  function fileRow(file, index) {
    const path =
      file.type === 'checkpoint'
        ? '/api/download_checkpoint/' + encodeURIComponent(file.name)
        : file.path;
    return (
      '<tr><td class="file-name">' +
      Studio.escape(file.name) +
      '</td><td><span class="badge">' +
      Studio.escape(file.type) +
      '</span></td><td class="data">' +
      bytes(file.size) +
      '</td><td><div class="actions">' +
      (path
        ? '<a class="btn btn-small" href="' +
          Studio.escape(path) +
          '" download>下载</a>' +
          (file.type !== 'checkpoint'
            ? '<a class="btn btn-small" href="' +
              Studio.escape(path) +
              '" target="_blank" rel="noopener">查看</a>'
            : '')
        : '') +
      '<button type="button" class="btn btn-small btn-danger" data-delete-file="' +
      index +
      '">删除</button></div></td></tr>'
    );
  }
  function renderResults() {
    if (!sessions.length)
      return Studio.state(
        roots.results,
        'empty',
        '还没有训练产物，完成实验后会显示在这里。',
      );
    roots.results.innerHTML = sessions
      .map((session, index) => {
        const rows = files.filter(
          (file) =>
            file.session.session_id === session.session_id &&
            file.type !== 'checkpoint',
        );
        return (
          '<section class="session-files"><div class="session-files-header"><div><h3>' +
          Studio.escape((session.model_type || '—').toUpperCase()) +
          ' · ' +
          Studio.escape((session.problem_type || '—').toUpperCase()) +
          '</h3><p>Session ' +
          Studio.escape(session.session_id) +
          ' · ' +
          Studio.escape(session.start_time) +
          ' · ' +
          session.file_count +
          ' 个文件 · ' +
          session.total_size_mb +
          ' MB</p></div><button type="button" class="btn btn-small btn-danger" data-delete-session="' +
          index +
          '" ' +
          (['running', 'paused', 'pausing', 'stopping'].includes(session.status)
            ? 'disabled title="请先结束训练"'
            : '') +
          '>删除 Session 产物</button></div><div class="table-scroll">' +
          (rows.length
            ? table(
                ['文件名', '类型', '大小', '操作'],
                rows.map((file) => fileRow(file, files.indexOf(file))).join(''),
              )
            : '<div class="empty-state">此 Session 只有 Checkpoint，请切换标签查看。</div>') +
          '</div></section>'
        );
      })
      .join('');
    roots.results
      .querySelectorAll('[data-delete-session]')
      .forEach(
        (button) =>
          (button.onclick = () =>
            deleteSession(sessions[Number(button.dataset.deleteSession)])),
      );
    bindFileDeletes(roots.results);
  }
  function renderCheckpoints() {
    const records = files.filter((file) => file.type === 'checkpoint');
    if (!records.length)
      return Studio.state(
        roots.checkpoints,
        'empty',
        '还没有已保存的 Checkpoint。',
      );
    roots.checkpoints.innerHTML = table(
      ['Checkpoint', 'Session / 模型', '大小', '操作'],
      records
        .map(
          (file) =>
            '<tr><td class="file-name">' +
            Studio.escape(file.name) +
            '</td><td class="file-name data">' +
            Studio.escape(file.session.session_id) +
            '<p class="description">' +
            Studio.escape(file.session.model_type) +
            '</p></td><td class="data">' +
            bytes(file.size) +
            '</td><td><div class="actions"><a class="btn btn-small" href="/api/download_checkpoint/' +
            encodeURIComponent(file.name) +
            '">下载</a><button type="button" class="btn btn-small btn-danger" data-delete-file="' +
            files.indexOf(file) +
            '">删除</button></div></td></tr>',
        )
        .join(''),
    );
    bindFileDeletes(roots.checkpoints);
  }
  function bindFileDeletes(root) {
    root
      .querySelectorAll('[data-delete-file]')
      .forEach(
        (button) =>
          (button.onclick = () =>
            deleteFile(files[Number(button.dataset.deleteFile)])),
      );
  }
  async function destroy({ title, description, url, payload }) {
    const success = await Studio.confirmAction({
      title,
      description,
      action: async () => {
        const data = await Studio.post(url, payload);
        Studio.toast(data.message, 'success');
      },
    });
    if (success) load();
  }
  function deleteDataset(item) {
    return destroy({
      title: '删除数据集？',
      description:
        '将永久删除「' +
        item.filename +
        '」（' +
        item.num_loc +
        ' 个节点，' +
        bytes(item.size) +
        '）。已有训练结果不受影响。',
      url: '/api/delete_dataset',
      payload: { dataset_id: item.dataset_id },
    });
  }
  function deleteFile(file) {
    return destroy({
      title:
        file.type === 'checkpoint' ? '删除 Checkpoint？' : '删除训练文件？',
      description:
        '将删除「' +
        file.name +
        '」（' +
        bytes(file.size) +
        '）及其文件记录。此操作无法撤销。',
      url: '/api/delete_file',
      payload: { file_id: file.id, filename: file.name, file_type: file.type },
    });
  }
  function deleteSession(session) {
    return destroy({
      title: '删除 Session 产物？',
      description:
        'Session ' +
        session.session_id +
        ' 关联的 ' +
        session.file_count +
        ' 个文件（' +
        session.total_size_mb +
        ' MB）将被删除，包括图像、动画和 Checkpoint。后端保留训练会话记录，此操作无法撤销。',
      url: '/api/delete_session',
      payload: { session_id: session.session_id },
    });
  }
  document.getElementById('files-clear').onclick = () =>
    destroy({
      title: '清空全部训练产物？',
      description:
        '将删除当前账户的全部训练文件与 Checkpoint。已列出 ' +
        files.length +
        ' 个文件。数据集及训练会话记录保留。此操作无法撤销。',
      url: '/api/clear_all_files',
      payload: { confirm: true },
    });
  document.getElementById('files-refresh').onclick = load;
  document.getElementById('files-upload-form').onsubmit = async (event) => {
    event.preventDefault();
    const button = document.getElementById('files-upload'),
      status = document.getElementById('files-upload-status');
    if (button.disabled) return;
    const input = document.getElementById('upload-file');
    if (!input.files[0]) return;
    if (input.files[0].size > 5 * 1048576) {
      status.hidden = false;
      status.className = 'banner banner-error';
      status.textContent = '文件超过 5 MB 上限。';
      return;
    }
    Studio.busy(button, true);
    status.hidden = false;
    status.className = 'banner';
    status.textContent = '正在上传与解析…';
    try {
      const data = new FormData(event.target);
      const response = await Studio.request('/api/upload_dataset', {
        method: 'POST',
        body: data,
      });
      status.className = 'banner banner-success';
      status.textContent = response.message;
      input.value = '';
      Studio.toast(response.message, 'success');
      load();
    } catch (error) {
      status.className = 'banner banner-error';
      status.textContent = error.message;
    } finally {
      Studio.busy(button, false);
    }
  };
  load();
})();
