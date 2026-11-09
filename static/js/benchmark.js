(() => {
  'use strict';
  const source = document.getElementById('benchmark-source'),
    problem = document.getElementById('problem-select'),
    scale = document.getElementById('node-select'),
    metric = document.getElementById('metric-select'),
    category = document.getElementById('category-select'),
    picker = document.getElementById('experiment-picker'),
    feedback = document.getElementById('benchmark-feedback'),
    table = document.getElementById('benchmark-table');
  let sessions = [],
    reference = {},
    chosen = new Set(),
    rows = [],
    chart = StudioCharts.create('mainChart'),
    loading = false;
  const query = new URLSearchParams(location.search);
  source.value =
    query.get('source') === 'reference' ? 'reference' : 'experiments';
  const group = (session) =>
    JSON.stringify([
      session.config?.num_loc ?? null,
      session.config?.dataset_id ?? null,
      session.mode || 'unknown',
    ]);
  const duration = (session) => {
    if (session.finished_at && session.started_at)
      return session.finished_at - session.started_at;
    if (session.end_time && session.start_time)
      return (
        (Date.parse(session.end_time) - Date.parse(session.start_time)) / 1000
      );
    return null;
  };
  function options(select, values, preferred) {
    const previous = preferred ?? select.value;
    select.replaceChildren();
    values.forEach(([value, label]) => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = label;
      select.append(option);
    });
    if (values.some(([value]) => value === previous)) select.value = previous;
  }
  async function load() {
    if (loading) return;
    loading = true;
    Studio.state(feedback, 'loading', '正在加载对比数据…');
    try {
      if (source.value === 'experiments') {
        const data = await Studio.request('/api/training_sessions');
        sessions = data.sessions.filter((row) => row.status === 'completed');
      } else {
        const response = await fetch('/static/data/benchmark-reference.json');
        if (!response.ok) throw new Error('论文基准数据读取失败。');
        reference = await response.json();
      }
      feedback.replaceChildren();
      setup();
    } catch (error) {
      Studio.state(feedback, 'error', error.message, load);
    } finally {
      loading = false;
    }
  }
  function setup() {
    const personal = source.value === 'experiments';
    document.getElementById('benchmark-category-field').hidden = personal;
    picker.hidden = !personal;
    options(
      metric,
      personal
        ? [
            ['best_reward', 'Best Reward'],
            ['final_reward', 'Final Reward'],
            ['final_loss', 'Final Loss'],
            ['duration', '训练耗时 (s)'],
          ]
        : [
            ['gap', 'Gap (%)'],
            ['cost', 'Cost'],
            ['time', 'Time (s)'],
          ],
    );
    if (personal) {
      options(
        problem,
        [...new Set(sessions.map((row) => row.problem_type))].map((key) => [
          key,
          key.toUpperCase(),
        ]),
      );
      const selected = sessions.find(
        (row) => row.session_id === query.get('session'),
      );
      if (selected) problem.value = selected.problem_type;
    } else
      options(
        problem,
        [
          ...new Set(
            Object.values(reference).flatMap((value) => Object.keys(value)),
          ),
        ].map((key) => [key, key]),
      );
    updateScale();
  }
  function updateScale() {
    if (source.value === 'experiments') {
      const matches = sessions.filter(
        (row) => row.problem_type === problem.value,
      );
      const groups = [...new Set(matches.map(group))];
      const selected = matches.find(
        (row) => row.session_id === query.get('session'),
      );
      options(
        scale,
        groups.map((key) => {
          const [size, dataset, mode] = JSON.parse(key);
          return [
            key,
            (size == null ? '规模未知' : size + ' 节点') +
              ' · ' +
              (dataset ? '数据集 ' + dataset.slice(0, 8) : '随机数据') +
              (mode === 'simulation'
                ? ' · 模拟模式'
                : mode === 'unknown'
                  ? ' · 模式未记录'
                  : ''),
          ];
        }),
        selected ? group(selected) : undefined,
      );
      updatePicker();
    } else {
      options(
        scale,
        Object.keys(reference)
          .filter((key) => reference[key][problem.value])
          .map((key) => [key, key + ' 节点']),
      );
      render();
    }
  }
  function updatePicker() {
    const matches = sessions.filter(
      (row) => row.problem_type === problem.value && group(row) === scale.value,
    );
    chosen = new Set(matches.map((row) => row.session_id));
    picker.innerHTML = matches
      .map(
        (row) =>
          '<label><input type="checkbox" value="' +
          Studio.escape(row.session_id) +
          '" checked><span>' +
          Studio.escape((row.model_type || '—').toUpperCase()) +
          ' · ' +
          Studio.escape(row.session_id.slice(0, 8)) +
          '</span></label>',
      )
      .join('');
    picker.querySelectorAll('input').forEach(
      (input) =>
        (input.onchange = () => {
          if (input.checked) chosen.add(input.value);
          else chosen.delete(input.value);
          render();
        }),
    );
    render();
  }
  function seconds(text) {
    if (String(text).startsWith('<')) return null; // Do not turn an upper bound into an exact duration.
    let total = 0,
      found = false;
    for (const match of String(text).matchAll(/(\d+(?:\.\d+)?)\s*([hms])/g)) {
      total += Number(match[1]) * { h: 3600, m: 60, s: 1 }[match[2]];
      found = true;
    }
    return found ? total : null;
  }
  function render() {
    const personal = source.value === 'experiments';
    if (personal)
      rows = sessions
        .filter(
          (row) =>
            chosen.has(row.session_id) &&
            row.problem_type === problem.value &&
            group(row) === scale.value,
        )
        .map((row) => ({
          label:
            (row.model_type || '—').toUpperCase() +
            ' · ' +
            row.session_id.slice(0, 8),
          ...row,
          duration: duration(row),
        }));
    else {
      const data = reference[scale.value]?.[problem.value] || {};
      rows = (
        category.value === 'all'
          ? Object.entries(data).flatMap(([key, values]) =>
              values.map((row) => ({ ...row, category: key })),
            )
          : (data[category.value] || []).map((row) => ({
              ...row,
              category: category.value,
            }))
      ).map((row) => ({
        ...row,
        label: row.method + ' · ' + row.category,
        time_raw: row.time,
        time: seconds(row.time),
      }));
    }
    const note = document.getElementById('benchmark-note');
    note.textContent = personal
      ? '来自账户内已完成的训练 Session。随机数据与独立训练结果并非统一测试集评测，请结合数据来源解释差异。' +
        (rows.some((row) => row.mode === 'simulation')
          ? ' 当前选择为模拟训练，指标仅供流程验证。'
          : '') +
        (rows.some((row) => row.mode === 'unknown')
          ? ' 部分历史记录未保存训练模式，无法确认是否为模拟训练。'
          : '')
      : '保留原项目标注为 KDD 2025 的论文基准表，与账户实验分开比较。<1s 等上界保留原始文字，不转成精确数值。';
    document.getElementById('benchmark-source-label').textContent = personal
      ? '账户 Session 数据'
      : '原项目论文基准';
    const comparable = rows.filter(
      (row) =>
        row[metric.value] != null && Number.isFinite(Number(row[metric.value])),
    );
    const maximize =
      (personal && metric.value.includes('reward')) ||
      (!personal && metric.value === 'cost' && problem.value === 'OP');
    const sorted = [...comparable].sort((a, b) =>
        maximize
          ? b[metric.value] - a[metric.value]
          : a[metric.value] - b[metric.value],
      ),
      best = sorted[0];
    document.getElementById('compare-count').textContent = rows.length;
    document.getElementById('best-label').textContent = maximize
      ? '当前最高指标'
      : '当前最低指标';
    document.getElementById('best-method').textContent = best
      ? best.label.split(' · ')[0]
      : '—';
    document.getElementById('best-value').textContent = Studio.number(
      best?.[metric.value],
    );
    document.getElementById('average-value').textContent = comparable.length
      ? Studio.number(
          comparable.reduce((sum, row) => sum + Number(row[metric.value]), 0) /
            comparable.length,
        )
      : '—';
    if (!rows.length) {
      Studio.state(
        feedback,
        'empty',
        personal
          ? '没有可对比的已完成实验。完成训练后，从结果页加入对比。'
          : '没有匹配的基准数据。',
      );
      table.replaceChildren();
      document.getElementById('benchmark-chart-card').hidden = true;
      return;
    }
    feedback.replaceChildren();
    document.getElementById('benchmark-chart-card').hidden = false;
    const option = StudioCharts.base();
    option.xAxis = {
      ...option.xAxis,
      boundaryGap: true,
      data: rows.map((row) => row.label),
      axisLabel: { rotate: 30, fontSize: 10 },
    };
    option.grid.bottom = 64;
    option.series = [
      {
        name: metric.selectedOptions[0]?.textContent,
        type: 'bar',
        barMaxWidth: 44,
        data: rows.map((row) => row[metric.value] ?? null),
      },
    ];
    chart?.setOption(option, true);
    chart?.resize();
    document.getElementById('benchmark-chart-summary').textContent =
      rows.length +
      ' 项数据，' +
      comparable.length +
      ' 项包含当前指标。完整数值见下表。';
    const heads = personal
      ? [
          '实验 / 模型',
          '规模',
          'Best Reward',
          'Final Reward',
          'Final Loss',
          '耗时 (s)',
          '模式',
        ]
      : ['方法', '解码方式', 'Gap (%)', 'Cost', 'Time（原值）'];
    table.innerHTML =
      '<table class="data-table"><thead><tr>' +
      heads.map((text) => '<th scope="col">' + text + '</th>').join('') +
      '</tr></thead><tbody>' +
      rows
        .map(
          (row) =>
            '<tr>' +
            (personal
              ? [
                  row.label,
                  row.config?.num_loc ?? '—',
                  Studio.number(row.best_reward),
                  Studio.number(row.final_reward),
                  Studio.number(row.final_loss),
                  Studio.number(row.duration),
                  row.mode === 'simulation'
                    ? '模拟训练'
                    : row.mode === 'real'
                      ? '真实训练'
                      : '模式未记录',
                ]
              : [
                  row.method,
                  row.category,
                  Studio.number(row.gap),
                  Studio.number(row.cost),
                  row.time_raw,
                ]
            )
              .map((value) => '<td>' + Studio.escape(value) + '</td>')
              .join('') +
            '</tr>',
        )
        .join('') +
      '</tbody></table>';
  }
  source.onchange = load;
  problem.onchange = updateScale;
  scale.onchange = () =>
    source.value === 'experiments' ? updatePicker() : render();
  metric.onchange = render;
  category.onchange = render;
  document.getElementById('benchmark-refresh').onclick = load;
  document.getElementById('benchmark-export').onclick = () => {
    if (!rows.length) return Studio.toast('请先选择可导出的数据。', 'warning');
    const quote = (value) =>
      '"' + String(value ?? '').replace(/"/g, '""') + '"';
    const csv = [
      ['实验 / 方法', metric.value],
      ...rows.map((row) => [row.label, row[metric.value]]),
    ]
      .map((row) => row.map(quote).join(','))
      .join('\r\n');
    const url = URL.createObjectURL(
      new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' }),
    );
    const a = document.createElement('a');
    a.href = url;
    a.download = 'rl4co-comparison.csv';
    a.click();
    URL.revokeObjectURL(url);
  };
  load();
})();
