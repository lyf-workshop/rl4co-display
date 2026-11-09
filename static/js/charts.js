(() => {
  'use strict';
  const css = getComputedStyle(document.documentElement),
    token = (name) => css.getPropertyValue(name).trim();
  const charts = new Set();
  function create(id) {
    const node = document.getElementById(id);
    if (!node || !window.echarts) return null;
    const chart = echarts.init(node, null, { renderer: 'canvas' });
    charts.add(chart);
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(node);
    window.addEventListener('pagehide', (event) => {
      if (event.persisted) return;
      observer.disconnect();
      chart.dispose();
      charts.delete(chart);
    });
    return chart;
  }
  const base = () => ({
    animation: !matchMedia('(prefers-reduced-motion: reduce)').matches,
    textStyle: {
      fontFamily: token('--font-data'),
      color: token('--color-muted'),
    },
    color: [
      token('--color-brand'),
      token('--color-success'),
      token('--color-warning'),
    ],
    grid: { left: 18, right: 24, top: 28, bottom: 20, containLabel: true },
    tooltip: {
      trigger: 'axis',
      confine: true,
      backgroundColor: token('--color-surface'),
      borderColor: token('--color-border'),
      textStyle: { color: token('--color-ink'), fontSize: 12 },
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      axisLine: { lineStyle: { color: token('--color-border') } },
      axisTick: { show: false },
      axisLabel: { color: token('--color-muted') },
    },
    yAxis: {
      type: 'value',
      scale: true,
      axisLabel: { color: token('--color-muted') },
      splitLine: {
        lineStyle: { color: token('--color-border'), type: 'dashed' },
      },
    },
  });
  function line(chart, rows, key, name, color) {
    if (!chart) return;
    const option = base();
    option.xAxis.data = rows.map((item) => item.epoch);
    option.series = [
      {
        name,
        type: 'line',
        data: rows.map((item) => item[key] ?? null),
        showSymbol: rows.length === 1,
        connectNulls: false,
        lineStyle: { width: 2, color: token(color) },
        itemStyle: { color: token(color) },
      },
    ];
    chart.setOption(option, true);
  }
  window.StudioCharts = {
    create,
    base,
    line,
    resize: () => charts.forEach((chart) => chart.resize()),
  };
})();
